"""The ablation of our two design choices against ReCoVeR on LCB, as on MGSM (../mgsm_crosslingual/ablation.py):
our method (instruction + steering, one layer, prompt-only: instruction_steer of summary_prompt.json) against
  tokens:  instruction_steer      one layer, all-token (summary.json);
  layers:  every_steer_prompt     every block but the last, prompt-only (recover.py, LCB_ARM=every_steer_prompt);
  baseline: recover               ReCoVeR as published (recover.py);
each at its setting chosen on dev, per language (test) and shared over the languages (test_universal); 197 test
prompts per language. Per comparison: the mean LPR difference ours - other over the six languages, with a 95%
bootstrap CI (prompts resampled within each language and source, the same prompts for every arm; LPR as in
analyse.score, the mean over sources of the line-level pass rate), and the languages significantly better/worse
for ours (McNemar on line-level passes, p < .05, and after Holm over the six languages).
Our two arms come from different test files (all-token and prompt-only jobs), so their comparison is paired over
prompts but not over runs; the instruction is taken from the prompt-only file. Login-node work (no model).

Usage: LCB_MODEL=<model> python ablation.py   -> results/<model>/summary_ablation.json
"""
import json

import numpy as np

import analyse
import data
import generate as lcb

OURS = "ours"
OTHERS = {"instruction_steer": "tokens: all-token", "every_steer_prompt": "layers: every layer",
          "recover": "baseline: ReCoVeR"}
SHOWN = ("instruction", OURS) + tuple(OTHERS)
N_BOOT, SEED = 2000, 0


def load(path):
    return [json.loads(x) for x in open(path)]


def rows(L, setting):
    G = analyse.GEN
    orig, prompt = load(G / f"{L}_{setting}.jsonl"), load(G / f"{L}_{setting}_prompt.jsonl")
    out = {"instruction": [r for r in prompt if r["cond"] == "instruction"],
           OURS: [r for r in prompt if r["cond"] == "instruction_steer"],
           "instruction_steer": [r for r in orig if r["cond"] == "instruction_steer"],
           "every_steer_prompt": load(G / "every_steer_prompt" / f"{L}_{setting}.jsonl"),
           "recover": load(G / "recover" / f"{L}_{setting}.jsonl")}
    ids = [r["id"] for r in out["instruction"]]
    assert all([r["id"] for r in v] == ids for v in out.values()), (L, setting)
    return out


def lpr(p, src, idx):
    """LPR of the prompts idx [..., n] (indices into p, src): mean over sources of the pass rate (None skipped)."""
    vals = []
    for s in data.SOURCES:
        m = (src[idx] == s) & ~np.isnan(p[idx])
        vals.append(np.where(m, p[idx], 0).sum(-1) / np.maximum(1, m.sum(-1)))
    return np.mean(vals, 0)


def main():
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "seed": SEED, "ours": "instruction_steer (prompt-only)"}
    for setting in ("test", "test_universal"):
        res = out[setting] = {"per_lang": {}, "mean_lpr": {}, "comparisons": {}}
        boot_lpr = {a: [] for a in SHOWN}
        cmp = {b: [] for b in OTHERS}
        for L in lcb.LANGS:
            by = rows(L, setting)
            src = np.array([r["source"] for r in by["instruction"]])
            ps = {a: np.array([np.nan if (x := analyse.passes(r)) is None else float(x) for r in v])
                  for a, v in by.items()}
            n = len(src)
            # stratified bootstrap: resample within each source, the same draw for every arm
            strata = [np.flatnonzero(src == s) for s in data.SOURCES]
            idx = np.concatenate([np.arange(n)[None]] + [np.concatenate(
                [rng.choice(g, len(g)) for g in strata])[None] for _ in range(N_BOOT)])
            res["per_lang"][L] = {a: float(lpr(ps[a], src, idx[:1])[0]) for a in SHOWN}
            for a in SHOWN:
                boot_lpr[a].append(lpr(ps[a], src, idx))
            for b in OTHERS:
                g, l, p = analyse.mcnemar([analyse.passes(r) for r in by[b]], [analyse.passes(r) for r in by[OURS]])
                cmp[b].append(dict(lang=L, better=g, worse=l, p=p))
            for a in SHOWN:  # the score of analyse.score, as a check of lpr()
                assert abs(analyse.score(by[a])["lpr"] - res["per_lang"][L][a]) < 1e-9, (L, a)
        bl = {a: np.mean(v, 0) for a, v in boot_lpr.items()}  # [1 + N_BOOT], mean over languages
        res["mean_lpr"] = {a: [float(bl[a][0]), *map(float, np.percentile(bl[a][1:], [2.5, 97.5]))] for a in SHOWN}
        for b, what in OTHERS.items():
            d = bl[OURS] - bl[b]
            qs = analyse.holm([c["p"] for c in cmp[b]])
            sig = lambda pv, s: sum(x < .05 and (c["better"] - c["worse"]) * s > 0 for c, x in zip(cmp[b], pv))
            pv = [c["p"] for c in cmp[b]]
            res["comparisons"][f"ours-{b}"] = dict(
                what=what, diff_ci=[float(d[0]), *map(float, np.percentile(d[1:], [2.5, 97.5]))],
                langs_better_worse=[sig(pv, 1), sig(pv, -1)], langs_better_worse_holm=[sig(qs, 1), sig(qs, -1)],
                per_lang=[dict(c, p_holm=q) for c, q in zip(cmp[b], qs)])
    f = lambda c, s="": f"{c[0]:{s}.3f} [{c[1]:{s}.3f}, {c[2]:{s}.3f}]"
    print(f"\n{analyse.MODEL}: LCB, 197 test prompts in each of {', '.join(lcb.LANGS)}")
    for setting in ("test", "test_universal"):
        res = out[setting]
        print(f"\n== {'per-language' if setting == 'test' else 'shared'} settings: mean LPR [95% CI]")
        for a in SHOWN:
            print(f"  {a:22}{f(res['mean_lpr'][a]):>24}   " + " ".join(f"{L} {res['per_lang'][L][a]:.2f}"
                                                                     for L in lcb.LANGS))
        print("  ours - other: LPR difference [95% CI]; languages better/worse for ours, p < .05 (Holm)")
        for k, d in res["comparisons"].items():
            bw, bh = d["langs_better_worse"], d["langs_better_worse_holm"]
            print(f"    {d['what']:22}{k:28}{f(d['diff_ci'], '+'):>26}   {bw[0]}/{bw[1]} ({bh[0]}/{bh[1]})")
    (analyse.RESULTS / "summary_ablation.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
