"""ReCoVeR (Sterz et al., 2025) as a baseline on the cross-lingual LCB, against our single-layer steering.

ReCoVeR's training-free ("naive") steering as published, as on cross-lingual MGSM (../mgsm_crosslingual/recover.py):
every block j adds alpha * u_T(j) to its output, u_T(j) = (mu_T - mu_en)(j) / |(mu_T - mu_en)(j)|, the mean hidden
states of the 997 FLORES-200 dev sentences, at every position but the first token of the prompt and at every generated
token, and rescales each block output to its unsteered norm (--restore_norm, as in all their published runs). The
hooks are Model.generate(recover=...) of ../mgsm_steering/generate.py, checked against the authors' code
(../mgsm_crosslingual/recover_check.py: identical logits; their Llama LCB result reproduced).

The protocol is the one of our steering on LCB (generate.py, analyse.py): the LCB prompt as is (with its language
instruction) is the only user message, greedy, at most 100 new tokens; the dev split (102 prompts) for selection, the
test split (197) once. The only tuned setting is alpha (their LCB runs use 1), over BASE_ALPHAS, extended by EXTEND
doublings of the largest alpha while the best one (per language or shared) is the largest:
  1. dev: the instruction and ReCoVeR at every alpha of the grid, per language;
  2. alpha by the rule of analyse.select: the highest LPR among the alphas whose degenerate rate is at most the
     instruction's + 0.05; ties go to the lower degenerate rate, then the smaller alpha. Per language, and shared
     (analyse.select_universal: mean over the languages, within the degenerate limit in every language);
  3. test: ReCoVeR at the chosen alpha, per-language and shared. It is compared with the instruction and our
     instruction + steering arms (all-token and prompt-only) from our original test runs (generations/<T>_test*.jsonl).

Outputs in results/<LCB_MODEL>/generations/recover/: vectors.pt, <T>_dev.jsonl, <T>_test.jsonl (per-language alpha),
<T>_test_universal.jsonl (shared alpha); the analysis in results/<LCB_MODEL>/summary_recover.json. A stage whose
output exists is skipped, so a run resumes; rows of other settings in a dev file are ignored.

Usage: LCB_MODEL=<model> python src/experiments/language_confusion/recover.py            # every stage, then analysis
       LCB_MODEL=<model> python src/experiments/language_confusion/recover.py --analyse  # the analysis only
       LCB_MODEL=<model> python src/experiments/language_confusion/recover.py --smoke    # quick check, ja, 24 prompts
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import analyse
import data
import generate as lcb  # LANGS, FLORES_CODE, flores, write

HERE = Path(__file__).resolve().parent
REC = analyse.GEN / "recover"
BASE_ALPHAS = (0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16)  # as in ../mgsm_crosslingual/recover.py
EXTEND, MAX_ALPHA = 4, 2 ** 20
ARM = "recover"
INSTR = ("instruction", None, None, None)
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

    def run(self, items, cond, alpha=None, layer=None):
        """Rows of one condition on items (one language): instruction, ReCoVeR (alpha) or one of our arms (layer,
        alpha; only in the smoke test)."""
        L, v = items[0]["lang"], self.vectors()
        texts = [r["prompt"] for r in items]
        restore_norm = True if cond == ARM else None
        if cond == ARM:
            vec = v["recover"][L] - v["recover"]["en"]
            res = self.M.generate(texts, vecs=[vec] * len(texts), recover=dict(alpha=alpha, restore_norm=True))
        elif cond in OURS:
            vec = v["ours"][L][layer] - v["ours"]["en"][layer]
            res = self.M.generate(texts, layer=layer, vecs=[alpha * vec] * len(texts),
                                  prompt_only=cond == "instruction_steer_prompt")
        else:
            res = self.M.generate(texts)
        rows = [dict(r, cond=cond, layer=layer, alpha=alpha, restore_norm=restore_norm, response=t, n_tokens=k,
                     capped=c) for r, (t, k, c) in zip(items, res)]
        print(f"{L} {items[0]['split']} {cond} alpha={alpha} layer={layer} | "
              f"LPR {analyse.score(rows)['lpr']:.2f} | {res[0][0][:70]!r}", flush=True)
        return rows


def load(path):
    return [json.loads(l) for l in open(path)]


def key(r):
    return r["cond"], r["alpha"], r["restore_norm"], r["layer"]


def dev_scores(rows):
    """{key: scores} of the instruction and ReCoVeR (restore on) in a dev file; other rows are ignored."""
    g = {}
    for r in rows:
        if key(r) == INSTR or (r["cond"] == ARM and r["restore_norm"] is True):
            g.setdefault(key(r), []).append(r)
    return {k: analyse.score(v) for k, v in g.items()}


def rkey(alpha):
    return ARM, alpha, True, None


def select(scores_by_lang, grid):
    """The alpha of grid shared by the languages {lang: dev scores} (one language: per-language), with the rule of
    analyse.select_universal."""
    S = list(scores_by_lang.values())
    mean = lambda k, m: sum(s[k][m] for s in S) / len(S)
    within = lambda k: all(s[k]["degenerate"] <= s[INSTR]["degenerate"] + analyse.MAX_EXTRA_DEGENERATE for s in S)
    keys = [rkey(a) for a in grid]
    ok = [k for k in keys if within(k)] or [min(keys, key=lambda k: mean(k, "degenerate"))]
    return min(ok, key=lambda k: (-round(mean(k, "lpr"), 9), round(mean(k, "degenerate"), 9), k[1]))[1]


def load_dev():
    return {L: dev_scores(load(REC / f"{L}_dev.jsonl")) for L in lcb.LANGS if (REC / f"{L}_dev.jsonl").exists()}


def alphas(dev):
    """The alpha grid so far: BASE_ALPHAS, extended while every language has every alpha of it on dev and the best
    alpha of a language, or the shared one, is the largest."""
    g = list(BASE_ALPHAS)
    while (set(dev) == set(lcb.LANGS) and all(rkey(a) in dev[L] for L in lcb.LANGS for a in g) and g[-1] < MAX_ALPHA
           and g[-1] in [select({L: dev[L]}, g) for L in lcb.LANGS] + [select(dev, g)]):
        g += [g[-1] * 2 ** k for k in range(1, EXTEND + 1)]
    return tuple(g)


def stages(R):
    while True:  # dev, until every language has every alpha of the grid
        dev = load_dev()
        grid = alphas(dev)
        todo = [L for L in lcb.LANGS if L not in dev or any(rkey(a) not in dev[L] for a in grid)]
        if not todo:
            break
        for L in todo:
            f = REC / f"{L}_dev.jsonl"
            items = [dict(r, split="dev") for r in data.load("dev", [L])]
            rows = load(f) if f.exists() else []
            have = dev_scores(rows)
            if INSTR not in have:
                rows += R.run(items, "instruction")
            for a in grid:
                if rkey(a) not in have:
                    rows += R.run(items, ARM, a)
            lcb.write(rows, f)
    shared = select(dev, grid)
    print(f"alpha grid {grid}; shared alpha {shared}", flush=True)
    for setting in ("test", "test_universal"):
        for L in lcb.LANGS:
            f = REC / f"{L}_{setting}.jsonl"
            if not f.exists():
                a = select({L: dev[L]}, grid) if setting == "test" else shared
                lcb.write(R.run([dict(r, split="test") for r in data.load("test", [L])], ARM, a), f)


def compare(a, b):
    """Paired over prompts (same ids): McNemar on line-level passes of b against a; mean LPR difference b - a."""
    assert [r["id"] for r in a] == [r["id"] for r in b]
    n01, n10, p = analyse.mcnemar([analyse.passes(r) for r in a], [analyse.passes(r) for r in b])
    return dict(better=n01, worse=n10, p=p, diff=analyse.score(b)["lpr"] - analyse.score(a)["lpr"])


PAIRS = ((ARM, "instruction"), ("instruction_steer", "instruction"), ("instruction_steer_prompt", "instruction"),
         (ARM, "instruction_steer"), (ARM, "instruction_steer_prompt"))
SHOWN = ("instruction", ARM, "instruction_steer", "instruction_steer_prompt")


def test_rows(L, setting, alpha):
    """{cond: rows}: ReCoVeR (this run, at alpha), the instruction and our two arms (our original test runs)."""
    rec = load(REC / f"{L}_{setting}.jsonl")
    assert all((r["cond"], r["alpha"], r["restore_norm"]) == (ARM, alpha, True) for r in rec), (L, setting)
    orig = load(analyse.GEN / f"{L}_{setting}.jsonl")
    prompt = load(analyse.GEN / f"{L}_{setting}_prompt.jsonl")
    return {ARM: rec, "instruction": [r for r in orig if r["cond"] == "instruction"],
            "instruction_steer": [r for r in orig if r["cond"] == "instruction_steer"],
            "instruction_steer_prompt": [r for r in prompt if r["cond"] == "instruction_steer"]}


def run_analysis():
    dev = load_dev()
    grid = alphas(dev)
    shared = select(dev, grid)
    out = {"alphas": grid, "shared": dict(alpha=shared, at_grid_edge=shared == grid[-1])}
    for setting in ("test", "test_universal"):
        res = out[setting] = {}
        for L in lcb.LANGS:
            a = select({L: dev[L]}, grid) if setting == "test" else shared
            by = test_rows(L, setting, a)
            d = res[L] = {"scores": {c: analyse.score(v) for c, v in by.items()},
                          "settings": {c: {k: v[0].get(k) for k in ("layer", "alpha", "restore_norm")}
                                       for c, v in by.items() if c != "instruction"},
                          "compare": {f"{x}-{y}": compare(by[y], by[x]) for x, y in PAIRS}}
            if setting == "test":
                d["dev_grid"] = [dict(alpha=x, **{m: dev[L][rkey(x)][m] for m in ("lpr", "english", "degenerate",
                                                                                   "thinking")}) for x in grid]
                d["dev_instruction"] = {m: dev[L][INSTR][m] for m in ("lpr", "english", "degenerate", "thinking")}
                d["chosen"] = dict(alpha=a, at_grid_edge=a == grid[-1])
        for pair in (f"{x}-{y}" for x, y in PAIRS):
            cs = [res[L]["compare"][pair] for L in lcb.LANGS]
            for c, q in zip(cs, analyse.holm([c["p"] for c in cs])):
                c["p_holm"] = q
        res["mean_lpr"] = {c: sum(res[L]["scores"][c]["lpr"] for L in lcb.LANGS) / len(lcb.LANGS) for c in SHOWN}
    (analyse.RESULTS / "summary_recover.json").write_text(json.dumps(out, indent=1) + "\n")
    report(out)


def report(out):
    print(f"\n{analyse.MODEL}: ReCoVeR on LCB (test, 197 prompts per language); alpha grid {out['alphas']}, "
          f"shared alpha {out['shared']}")
    for setting in ("test", "test_universal"):
        res = out[setting]
        print(f"\n== {'per-language alpha' if setting == 'test' else 'shared alpha'}: LPR (degenerate)")
        print(f"{'':26}" + "".join(f"{L:>12}" for L in lcb.LANGS) + f"{'mean':>8}")
        for c in SHOWN:
            print(f"{c:26}" + "".join(f"{res[L]['scores'][c]['lpr']:>6.2f} ({res[L]['scores'][c]['degenerate']:.2f})"
                                      for L in lcb.LANGS) + f"{res['mean_lpr'][c]:>8.3f}")
        print(f"{'  ' + ARM + ' alpha':26}" + "".join(f"{res[L]['settings'][ARM]['alpha']:>12}" for L in lcb.LANGS))
        print("a - b: LPR difference, prompts better/worse, languages significant better/worse (p < .05; Holm)")
        for pair in (f"{x}-{y}" for x, y in PAIRS):
            cs = {L: res[L]["compare"][pair] for L in lcb.LANGS}
            sig = lambda k: (sum(c[k] < .05 and c["better"] > c["worse"] for c in cs.values()),
                             sum(c[k] < .05 and c["better"] < c["worse"] for c in cs.values()))
            print(f"  {pair:44}" + " ".join(f"{L} {c['diff']:+.2f} {c['better']}/{c['worse']}" for L, c in cs.items())
                  + f"   p: {sig('p')}, Holm: {sig('p_holm')}")


def smoke(R):
    """A few configurations on 24 dev prompts of ja: our arm and the instruction against our original dev runs
    (agreement of the two generators), and ReCoVeR at alpha 0.2 and 1; no file is written."""
    L, ours = "ja", ours_settings()["test"]["ja"]["instruction_steer"]
    dev = [dict(r, split="dev") for r in data.load("dev", [L])][::4][:24]
    ids = {r["id"] for r in dev}
    orig = {(r["cond"], r["layer"], r["alpha"], r["id"]): r["response"] for r in load(analyse.GEN / f"{L}_dev.jsonl")
            if r["id"] in ids}
    for cond, layer, a in (("instruction", None, None), ("instruction_steer", *ours)):
        rows = R.run(dev, cond, a, layer=layer)
        same = sum(orig[(cond, layer, a, r["id"])].strip() == r["response"] for r in rows)
        print(f"smoke {cond} layer={layer} alpha={a}: identical to the original run {same}/{len(rows)}", flush=True)
    for a in (0.2, 1):
        R.run(dev, ARM, a)
    v = R.vectors()
    cos = R.torch.nn.functional.cosine_similarity(v["recover"][L] - v["recover"]["en"], v["ours"][L] - v["ours"]["en"], dim=-1)
    print(f"smoke: cosine of the two vector computations over layers: min {cos[1:].min():.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--analyse", action="store_true")
    g.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        return run_analysis()
    R = Runner()
    if a.smoke:
        global REC
        REC = Path(analyse.GEN / "recover_smoke")
        return smoke(R)
    stages(R)
    run_analysis()


if __name__ == "__main__":
    main()
