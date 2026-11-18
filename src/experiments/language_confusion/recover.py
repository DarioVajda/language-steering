"""ReCoVeR (Sterz et al., 2025) as a baseline on the cross-lingual LCB, against our single-layer steering.

ReCoVeR's training-free ("naive") steering, as on cross-lingual MGSM (../mgsm_crosslingual/recover.py): every block j
adds alpha * u_T(j) to its output, u_T(j) = (mu_T - mu_en)(j) / |(mu_T - mu_en)(j)|, the mean hidden states of the
997 FLORES-200 dev sentences, at every position but the first token of the prompt and, for the arm recover, at every
generated token; recover_prompt steers the prompt only. With restore_norm, each block output is rescaled to its
unsteered norm. The hooks are Model.generate(recover=...) of ../mgsm_steering/generate.py, checked against the
authors' code (../mgsm_crosslingual/recover_check.py: identical logits; their Llama LCB result reproduced).

The protocol is the one of our steering on LCB (generate.py, analyse.py): the LCB prompt as is (with its language
instruction) is the only user message, greedy, at most 100 new tokens; the dev split (102 prompts) for selection, the
test split (197) once. Per language:
  1. dev: the instruction, and both ReCoVeR arms over alpha x restore_norm (the grid of the MGSM baseline);
  2. per arm, the setting chosen by the rule of analyse.select: the highest LPR among settings whose degenerate rate
     is at most the instruction's + 0.05; ties go to the lower degenerate rate, the smaller alpha, then restore off;
  3. test: the instruction, both ReCoVeR arms at their chosen settings, and our instruction + steering arms
     (all-token and prompt-only) at the settings chosen in our runs (summary.json, summary_prompt.json), rerun here so
     that every comparison is paired within one job and one generator.
Then the shared setting (analyse.select_universal: mean over the languages, within the degenerate limit in every
language) is run on test the same way, against our shared settings. Our arms use our vector, mu_T(l) - mu_en(l) at
one layer, computed as in generate.py; the rerun is compared with the original test files (agreement).

Outputs in results/<LCB_MODEL>/generations/recover/: vectors.pt, <T>_dev.jsonl, <T>_test.jsonl,
<T>_test_universal.jsonl; the analysis in results/<LCB_MODEL>/summary_recover.json. A stage whose output exists is
skipped, so a run resumes. With --reverse the languages are done in reverse order: a helper job started next to a
running one does the languages from the other end, and each process skips the files the other has written. With
--test-helper a process waits for every dev file, then takes the test files from the end of the main order, claiming
each with an exclusive <file>.claim so that several helpers (one per GPU) never do the same file; no analysis.

Usage: LCB_MODEL=<model> python src/experiments/language_confusion/recover.py            # every stage, then analysis
       LCB_MODEL=<model> python src/experiments/language_confusion/recover.py --analyse  # the analysis only
       LCB_MODEL=<model> python src/experiments/language_confusion/recover.py --smoke    # quick check, ja, 24 prompts
       LCB_MODEL=<model> python src/experiments/language_confusion/recover.py --reverse  # a helper, languages from ar
       LCB_MODEL=<model> python src/experiments/language_confusion/recover.py --test-helper  # a test-stage helper
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import analyse
import data
import generate as lcb  # LANGS, FLORES_CODE, flores, write

HERE = Path(__file__).resolve().parent
REC = analyse.GEN / "recover"
ALPHAS = (0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16)  # as in ../mgsm_crosslingual/recover.py
ARMS = ("recover", "recover_prompt")
OURS = {"instruction_steer": "summary.json", "instruction_steer_prompt": "summary_prompt.json"}
MAX_BATCH = 64


def mgsm_generator():
    """../mgsm_steering/generate.py under another module name (this directory has its own generate.py)."""
    spec = importlib.util.spec_from_file_location("mgsm_generate", HERE.parent / "mgsm_steering" / "generate.py")
    mg = importlib.util.module_from_spec(spec)
    sys.modules["mgsm_generate"] = mg
    spec.loader.exec_module(mg)
    mg.MAX_NEW, mg.MAX_BATCH = lcb.MAX_NEW, MAX_BATCH
    return mg


def ours_settings():
    """{setting: {lang: {arm: (layer, alpha)}}} of our instruction + steering arms, from our LCB summaries."""
    out = {"test": {}, "test_universal": {}}
    for arm, f in OURS.items():
        s = json.loads((analyse.RESULTS / f).read_text())
        for L in lcb.LANGS:
            c = s[L]["chosen"]["instruction_steer"]
            u = s[L]["test_universal"]["instruction_steer"]
            out["test"].setdefault(L, {})[arm] = (c["layer"], c["alpha"])
            out["test_universal"].setdefault(L, {})[arm] = (u["layer"], u["alpha"])
    return out


class Runner:
    def __init__(self):
        import torch
        self.torch, self.mg = torch, mgsm_generator()
        self.M = self.mg.Model(analyse.MODELS[analyse.MODEL]["hf"])
        self._vec = None

    def vectors(self):
        """{"recover": {lang: [n_layers + 1, d]}, "ours": {lang: ...}}: the mean hidden states of FLORES-200 dev in
        each language. recover as in the MGSM baseline (Model.mean_states, positions from the padding); ours as in
        generate.py (no position ids), so that our rerun arms use the vector of our runs."""
        if self._vec is None:
            path = REC / "vectors.pt"
            if not path.exists():
                REC.mkdir(parents=True, exist_ok=True)
                v = {"recover": {}, "ours": {}}
                for L in ("en",) + lcb.LANGS:
                    texts = lcb.flores(L)
                    v["recover"][L] = self.M.mean_states(texts).cpu()
                    v["ours"][L] = self.ours_mean_states(texts).cpu()
                tmp = path.with_name(".vectors.pt.tmp")
                self.torch.save(v, tmp)
                tmp.rename(path)
            self._vec = self.torch.load(path)
        return self._vec

    def ours_mean_states(self, texts, bs=50):
        """generate.Model.mean_states (our LCB runs), on this model object."""
        torch, M = self.torch, self.M
        acc = torch.zeros(len(M.blocks) + 1, M.model.config.get_text_config().hidden_size, device="cuda")
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                b = M.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
                hs = M.model(**b, output_hidden_states=True).hidden_states
                m = b.attention_mask.clone()
                m[torch.arange(m.shape[0]), m.argmax(1)] = 0
                m = m.unsqueeze(-1).float()
                for l, h in enumerate(hs):
                    acc[l] += ((h.float() * m).sum(1) / m.sum(1)).sum(0)
        return acc / len(texts)

    def run(self, items, cond, alpha=None, restore_norm=None, layer=None):
        """Rows of one condition on items (one language): instruction, a ReCoVeR arm (alpha, restore_norm) or one of
        our arms (layer, alpha)."""
        L, v = items[0]["lang"], self.vectors()
        texts = [r["prompt"] for r in items]
        if cond in ARMS:
            vec = v["recover"][L] - v["recover"]["en"]
            res = self.M.generate(texts, vecs=[vec] * len(texts), prompt_only=cond == "recover_prompt",
                                  recover=dict(alpha=alpha, restore_norm=restore_norm))
        elif cond in OURS:
            vec = v["ours"][L][layer] - v["ours"]["en"][layer]
            res = self.M.generate(texts, layer=layer, vecs=[alpha * vec] * len(texts),
                                  prompt_only=cond == "instruction_steer_prompt")
        else:
            res = self.M.generate(texts)
        rows = [dict(r, cond=cond, layer=layer, alpha=alpha, restore_norm=restore_norm, response=t, n_tokens=k,
                     capped=c) for r, (t, k, c) in zip(items, res)]
        print(f"{L} {items[0]['split']} {cond} alpha={alpha} restore={restore_norm} layer={layer} | "
              f"LPR {analyse.score(rows)['lpr']:.2f} | {res[0][0][:70]!r}", flush=True)
        return rows


def load(path):
    return [json.loads(l) for l in open(path)]


def key(r):
    return r["cond"], r["alpha"], r["restore_norm"], r["layer"]


def dev_scores(rows):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return {k: analyse.score(v) for k, v in g.items()}


def select(scores_by_lang):
    """{arm: (alpha, restore_norm)} shared by the languages {lang: dev scores} (one language: per-language), with
    the rule of analyse.select_universal."""
    S = list(scores_by_lang.values())
    base = ("instruction", None, None, None)
    mean = lambda k, m: sum(s[k][m] for s in S) / len(S)
    within = lambda k: all(s[k]["degenerate"] <= s[base]["degenerate"] + analyse.MAX_EXTRA_DEGENERATE for s in S)
    out = {}
    for arm in ARMS:
        keys = [k for k in S[0] if k[0] == arm]
        ok = [k for k in keys if within(k)] or [min(keys, key=lambda k: mean(k, "degenerate"))]
        best = min(ok, key=lambda k: (-round(mean(k, "lpr"), 9), round(mean(k, "degenerate"), 9), k[1], k[2]))
        out[arm] = (best[1], best[2])
    return out


def run_test(R, L, setting, chosen, ours, f):
    """The test file f: the instruction, the ReCoVeR arms at their chosen settings, our arms at our settings."""
    test = [dict(r, split="test") for r in data.load("test", [L])]
    rows = R.run(test, "instruction")
    for arm, (a, rn) in chosen.items():
        rows += R.run(test, arm, a, rn)
    for arm, (layer, a) in ours.items():
        rows += R.run(test, arm, a, layer=layer)
    lcb.write(rows, f)


def stages(R, reverse=False):
    ours = ours_settings()
    langs = lcb.LANGS[::-1] if reverse else lcb.LANGS
    for L in langs:
        f = REC / f"{L}_dev.jsonl"
        if not f.exists():
            dev = [dict(r, split="dev") for r in data.load("dev", [L])]
            rows = R.run(dev, "instruction")
            for arm in ARMS:
                for a in ALPHAS:
                    for rn in (False, True):
                        rows += R.run(dev, arm, a, rn)
            lcb.write(rows, f)
    dev = {L: dev_scores(load(REC / f"{L}_dev.jsonl")) for L in lcb.LANGS}
    shared = select(dev)
    print("shared ReCoVeR setting:", shared, flush=True)
    for setting in ("test", "test_universal"):
        for L in langs:
            f = REC / f"{L}_{setting}.jsonl"
            if f.exists():
                continue
            run_test(R, L, setting, select({L: dev[L]}) if setting == "test" else shared, ours[setting][L], f)


def test_helper(R):
    """Test files only, from the end of the order of stages(), each claimed exclusively; waits for the dev files."""
    devs = [REC / f"{L}_dev.jsonl" for L in lcb.LANGS]
    while not all(f.exists() for f in devs):
        time.sleep(30)
    ours = ours_settings()
    dev = {L: dev_scores(load(REC / f"{L}_dev.jsonl")) for L in lcb.LANGS}
    shared = select(dev)
    todo = [(setting, L) for setting in ("test", "test_universal") for L in lcb.LANGS][::-1]
    for setting, L in todo:
        f = REC / f"{L}_{setting}.jsonl"
        if f.exists():
            continue
        try:
            os.close(os.open(f.with_suffix(".claim"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        if f.exists():
            continue
        run_test(R, L, setting, select({L: dev[L]}) if setting == "test" else shared, ours[setting][L], f)
        f.with_suffix(".claim").unlink()


def compare(a, b):
    """Paired over prompts (same ids): McNemar on line-level passes of b against a; mean LPR difference b - a."""
    assert [r["id"] for r in a] == [r["id"] for r in b]
    n01, n10, p = analyse.mcnemar([analyse.passes(r) for r in a], [analyse.passes(r) for r in b])
    return dict(better=n01, worse=n10, p=p, diff=analyse.score(b)["lpr"] - analyse.score(a)["lpr"])


PAIRS = (("recover", "instruction"), ("recover_prompt", "instruction"),
         ("instruction_steer", "instruction"), ("instruction_steer_prompt", "instruction"),
         ("recover", "instruction_steer"), ("recover", "instruction_steer_prompt"),
         ("recover_prompt", "instruction_steer"), ("recover_prompt", "instruction_steer_prompt"))
SHOWN = ("instruction", "recover", "recover_prompt", "instruction_steer", "instruction_steer_prompt")


def run_analysis():
    dev = {L: dev_scores(load(REC / f"{L}_dev.jsonl")) for L in lcb.LANGS}
    shared = select(dev)
    out = {"alphas": ALPHAS, "shared": {a: dict(alpha=x, restore_norm=r, at_grid_edge=x == ALPHAS[-1])
                                        for a, (x, r) in shared.items()}}
    original = {"test": {arm: f"{{L}}_test{'_prompt' if arm.endswith('_prompt') else ''}.jsonl" for arm in OURS},
                "test_universal": {arm: f"{{L}}_test_universal{'_prompt' if arm.endswith('_prompt') else ''}.jsonl"
                                   for arm in OURS}}
    for setting in ("test", "test_universal"):
        res = out[setting] = {}
        for L in lcb.LANGS:
            rows = load(REC / f"{L}_{setting}.jsonl")
            by = {c: [r for r in rows if r["cond"] == c] for c in SHOWN}
            d = res[L] = {"scores": {c: analyse.score(v) for c, v in by.items()},
                          "settings": {c: {k: v[0][k] for k in ("layer", "alpha", "restore_norm")} for c, v in by.items()
                                       if c != "instruction"},
                          "compare": {f"{x}-{y}": compare(by[y], by[x]) for x, y in PAIRS}}
            if setting == "test":
                d["dev_grid"] = [dict(cond=k[0], alpha=k[1], restore_norm=k[2],
                                      **{m: v[m] for m in ("lpr", "english", "degenerate", "thinking")})
                                 for k, v in dev[L].items()]
                d["chosen"] = {a: dict(alpha=x, restore_norm=r, at_grid_edge=x == ALPHAS[-1])
                               for a, (x, r) in select({L: dev[L]}).items()}
            agree = {}  # our arms rerun here against our original test runs (other generator, maybe other GPU)
            for arm in OURS:
                orig = {r["id"]: r for r in load(analyse.GEN / original[setting][arm].format(L=L))
                        if r["cond"] == "instruction_steer"}
                o = [orig[r["id"]] for r in by[arm]]
                assert (o[0]["layer"], o[0]["alpha"]) == (by[arm][0]["layer"], by[arm][0]["alpha"])
                agree[arm] = dict(identical=sum(a["response"].strip() == b["response"].strip()
                                                for a, b in zip(o, by[arm])) / len(o),
                                  lpr_original=analyse.score(o)["lpr"], lpr_rerun=d["scores"][arm]["lpr"])
            d["rerun_agreement"] = agree
        for pair in (f"{x}-{y}" for x, y in PAIRS):
            cs = [res[L]["compare"][pair] for L in lcb.LANGS]
            for c, q in zip(cs, analyse.holm([c["p"] for c in cs])):
                c["p_holm"] = q
        res["mean_lpr"] = {c: sum(res[L]["scores"][c]["lpr"] for L in lcb.LANGS) / len(lcb.LANGS) for c in SHOWN}
    (analyse.RESULTS / "summary_recover.json").write_text(json.dumps(out, indent=1) + "\n")
    report(out)


def report(out):
    print(f"\n{analyse.MODEL}: ReCoVeR on LCB (test, 197 prompts per language); shared ReCoVeR setting {out['shared']}")
    for setting in ("test", "test_universal"):
        res = out[setting]
        print(f"\n== {'per-language settings' if setting == 'test' else 'shared settings'}: LPR (degenerate)")
        print(f"{'':26}" + "".join(f"{L:>12}" for L in lcb.LANGS) + f"{'mean':>8}")
        for c in SHOWN:
            print(f"{c:26}" + "".join(f"{res[L]['scores'][c]['lpr']:>6.2f} ({res[L]['scores'][c]['degenerate']:.2f})"
                                      for L in lcb.LANGS) + f"{res['mean_lpr'][c]:>8.3f}")
        for c in ("recover", "recover_prompt"):
            print(f"{'  setting ' + c:26}" + "".join(f"{'a' + str(res[L]['settings'][c]['alpha']) + ('r' if res[L]['settings'][c]['restore_norm'] else ''):>12}"
                                                  for L in lcb.LANGS))
        print("a - b: LPR difference, prompts better/worse, languages significant better/worse (p < .05; Holm)")
        for pair in (f"{x}-{y}" for x, y in PAIRS):
            cs = {L: res[L]["compare"][pair] for L in lcb.LANGS}
            sig = lambda k: (sum(c[k] < .05 and c["better"] > c["worse"] for c in cs.values()),
                             sum(c[k] < .05 and c["better"] < c["worse"] for c in cs.values()))
            print(f"  {pair:44}" + " ".join(f"{L} {c['diff']:+.2f} {c['better']}/{c['worse']}" for L, c in cs.items())
                  + f"   p: {sig('p')}, Holm: {sig('p_holm')}")
        print("  rerun of our arms vs our original runs (identical responses, LPR original -> rerun):")
        for arm in OURS:
            print(f"    {arm:26}" + " ".join(
                f"{L} {res[L]['rerun_agreement'][arm]['identical']:.2f} {res[L]['rerun_agreement'][arm]['lpr_original']:.2f}->"
                f"{res[L]['rerun_agreement'][arm]['lpr_rerun']:.2f}" for L in lcb.LANGS))


def smoke(R):
    """A few configurations on 24 dev prompts of ja: our rerun arm and the instruction against our original dev
    runs (agreement of the two generators), and ReCoVeR at alpha 0.2 and 1; no file is written."""
    L, ours = "ja", ours_settings()["test"]["ja"]["instruction_steer"]
    dev = [dict(r, split="dev") for r in data.load("dev", [L])][::4][:24]
    ids = {r["id"] for r in dev}
    orig = {(r["cond"], r["layer"], r["alpha"], r["id"]): r["response"] for r in load(analyse.GEN / f"{L}_dev.jsonl")
            if r["id"] in ids}
    for cond, layer, a in (("instruction", None, None), ("instruction_steer", *ours)):
        rows = R.run(dev, cond, a, layer=layer)
        same = sum(orig[(cond, layer, a, r["id"])].strip() == r["response"] for r in rows)
        print(f"smoke {cond} layer={layer} alpha={a}: identical to the original run {same}/{len(rows)}", flush=True)
    for arm, a, rn in (("recover", 0.2, True), ("recover", 1, False), ("recover_prompt", 1, False)):
        R.run(dev, arm, a, rn)
    v = R.vectors()
    cos = R.torch.nn.functional.cosine_similarity(v["recover"][L] - v["recover"]["en"], v["ours"][L] - v["ours"]["en"], dim=-1)
    print(f"smoke: cosine of the two vector computations over layers: min {cos[1:].min():.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--analyse", action="store_true")
    g.add_argument("--smoke", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="the languages in reverse order (a helper job)")
    g.add_argument("--test-helper", action="store_true", help="test files only, claimed exclusively (a helper job)")
    a = ap.parse_args()
    if a.analyse:
        return run_analysis()
    R = Runner()
    if a.smoke:
        global REC
        REC = Path(analyse.GEN / "recover_smoke")
        return smoke(R)
    if a.test_helper:
        return test_helper(R)
    stages(R, a.reverse)
    run_analysis()


if __name__ == "__main__":
    main()
