"""Answer-level agreement: does an arm get the same test items right and wrong as the native question, or as the
source question?

For an arm X and a reference R (native: the question in T; source: the question in S), on the 200 test items of a
pair: error consistency, i.e. Cohen's kappa on per-item correctness (the parsed number equals the gold answer;
the language is ignored), kappa = (p_obs - p_exp) / (1 - p_exp), with p_exp the agreement expected from the two
accuracies alone. Arms: source and steer_prompt (no instruction), instruction and instruction_steer_prompt (with
it), the steering arms at the setting chosen on dev. Means over all 110 pairs; 95% CIs from a bootstrap over
items (2000 resamples, the same items for every pair and arm in a resample, so differences get CIs too).

Usage: python src/experiments/mgsm_crosslingual/agreement.py   # -> results/agreement.json (after run.sh)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from analyse import correct, load  # noqa: E402
from generate_pairs import GEN, KEY, PAIRS  # noqa: E402

ARMS = ("source", "steer_prompt", "instruction", "instruction_steer_prompt")
REFS = ("native", "source")
# differences of interest: what steering changes, within each ladder
DIFFS = (("steer_prompt", "source"), ("instruction_steer_prompt", "instruction"))
N_BOOT, SEED = 2000, 0


def kappa(a, b):
    """Cohen's kappa over the last axis of two boolean arrays."""
    pa, pb = a.mean(-1), b.mean(-1)
    obs = (a == b).mean(-1)
    exp = pa * pb + (1 - pa) * (1 - pb)
    assert (exp < 1).all(), "kappa undefined: an arm is right on every item or wrong on every item"
    return (obs - exp) / (1 - exp)


def main():
    T = GEN / "test"
    ok = {}  # (arm, pair key) -> per-item correctness, items in the same order for every arm
    ids = None
    for s, t in PAIRS:
        k = KEY[(s, t)]
        rows = {"source": load(T / f"source_{s}.jsonl"), "native": load(T / f"native_{t}.jsonl")}
        rows.update({a: load(T / f"{a}_{k}.jsonl") for a in ARMS if a != "source"})
        for a, rs in rows.items():
            assert ids is None or [r["id"] for r in rs] == ids, (a, k)
            ids = [r["id"] for r in rs]
            ok[(a, k)] = np.array([correct(r) for r in rs])
    keys = [KEY[p] for p in PAIRS]
    n = len(ids)
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([np.arange(n)[None], rng.integers(0, n, (N_BOOT, n))])  # row 0: the data itself

    # kappa[(arm, ref)]: (1 + N_BOOT, pairs); row 0 is the estimate
    kap = {(a, r): np.stack([kappa(ok[(a, k)][idx], ok[(r, k)][idx]) for k in keys], 1)
           for a in ARMS + ("native",) for r in REFS if a != r}
    one = np.ones((1 + N_BOOT, len(keys)))
    diff = lambda a, b, r: (kap.get((a, r), one) - kap.get((b, r), one))  # kappa of a reference with itself is 1

    def summary(x):  # x: (1 + N_BOOT, pairs) -> mean over pairs with its bootstrap CI
        m = x.mean(1)
        lo, hi = np.percentile(m[1:], [2.5, 97.5])
        return dict(mean=float(m[0]), ci=[float(lo), float(hi)])

    out = {"n_items": n, "n_pairs": len(keys), "n_boot": N_BOOT, "seed": SEED,
           "kappa": {f"{a}|{r}": summary(v) for (a, r), v in kap.items()},
           "diff": {f"{a}-{b}|{r}": summary(diff(a, b, r)) for a, b in DIFFS for r in REFS},
           "per_pair": {f"{a}|{r}": dict(zip(keys, map(float, v[0]))) for (a, r), v in kap.items()},
           "accuracy": {a: float(np.mean([ok[(a, k)].mean() for k in keys])) for a in ARMS + ("native",)}}

    f = lambda d: f"{d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
    print(f"error consistency (Cohen's kappa on correctness), mean over {len(keys)} pairs, 95% bootstrap CI")
    print(f"{'':26}{'accuracy':>9}   {'with native':>24}   {'with source':>24}")
    for a in ARMS + ("native",):
        cells = [f(out["kappa"][f"{a}|{r}"]) if a != r else "1 (itself)" for r in REFS]
        print(f"{a:26}{out['accuracy'][a]:>9.3f}   {cells[0]:>24}   {cells[1]:>24}")
    print("\ndifference made by steering")
    for a, b in DIFFS:
        print(f"{a + ' - ' + b:40}" + "".join(f"   {r}: {f(out['diff'][f'{a}-{b}|{r}'])}" for r in REFS))
    (HERE / "results/agreement.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
