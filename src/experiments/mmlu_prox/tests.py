"""Paired significance tests on MMLU-ProX (the boxed runs of pilot.py, all pairs), as for MGSM (../mgsm_crosslingual).

Per subset (stem, hss) and held-out pair (the 99 pairs not used for tuning): an exact McNemar test on per-item success
(correct and in T) of one arm against another on the same questions, Holm-corrected over the 99 pairs; the number of
pairs significantly better / worse. Over the held-out pairs: the mean success of each arm and native (cot in T), the
mean differences of the compared arms, and the gap closed, (arm - instruction) / (native - instruction), each with a
95% percentile CI from a bootstrap over questions (2000 resamples, seed 0, the same questions for every pair and arm
in a resample).

Usage: XL_MODEL=<model> python src/experiments/mmlu_prox/tests.py [--format boxed --max-new 4096 --first-cap 2048]
  reads results/<model>/pilot<suffix>_all_items.json (written by pilot.py analyse --pairs all),
  writes results/<model>/pilot<suffix>_all_tests.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
sys.path.insert(0, str(HERE.parent / "mgsm_crosslingual"))
from analyse import holm, mcnemar  # noqa: E402
from generate_pairs import MODEL, PAIRS, TUNE  # noqa: E402

ARMS = ("instruction", "instruction_en", "instruction_steer_prompt", "recover", "instruction_system")  # those run
COMPARE = (("instruction_steer_prompt", "recover"), ("instruction_steer_prompt", "instruction"),
           ("recover", "instruction"), ("instruction_system", "instruction"),
           ("instruction_steer_prompt", "instruction_system"))  # those whose arms were run
HELD = [f"{s}-{t}" for s, t in PAIRS if (s, t) not in TUNE]
N_BOOT, SEED, ALPHA = 2000, 0, 0.05


def ci(x, boot):
    """Point estimate and 95% percentile interval of the mean over questions, given resampled question indices."""
    return [float(x.mean()), *map(float, np.percentile(x[boot].mean(-1), [2.5, 97.5]))]


def fmt(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="free", choices=("free", "boxed"))
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--first-cap", type=int, default=2048)
    a = ap.parse_args()
    suffix = ("" if a.max_new == 2048 else f"_{a.max_new}") + (f"_from{a.first_cap}" if a.first_cap else "") \
        + ("_boxed" if a.format == "boxed" else "")  # as in pilot.py
    RES = HERE / "results" / MODEL
    items = json.loads((RES / f"pilot{suffix}_all_items.json").read_text())
    out = {"held_out_pairs": HELD, "n_boot": N_BOOT, "alpha": ALPHA}
    for subset, by_arm in items.items():
        ids = [x[0] for x in by_arm["cot"][f"{HELD[0][-2:]}-{HELD[0][-2:]}"]]

        def ok(arm, pair):
            rows = by_arm[arm][pair]
            assert [x[0] for x in rows] == ids, (subset, arm, pair)
            return np.array([x[1] and x[2] for x in rows], dtype=bool)

        boot = np.random.default_rng(SEED).integers(0, len(ids), (N_BOOT, len(ids)))
        arms = [a for a in ARMS if a in by_arm]
        mean = {arm: np.mean([ok(arm, p) for p in HELD], axis=0) for arm in arms}  # per question, over held-out
        mean["native"] = np.mean([ok("cot", f"{p[-2:]}-{p[-2:]}") for p in HELD], axis=0)
        gap = lambda arm, b: ((mean[arm][b].mean(-1) - mean["instruction"][b].mean(-1))
                              / (mean["native"][b].mean(-1) - mean["instruction"][b].mean(-1)))
        s = out[subset] = {"n_questions": len(ids),
                           "success": {k: ci(v, boot) for k, v in mean.items()},
                           "gap_closed": {arm: [float(gap(arm, slice(None))),
                                                *map(float, np.percentile(gap(arm, boot), [2.5, 97.5]))]
                                          for arm in arms if arm != "instruction"},
                           "compare": {}}
        print(f"\n== {MODEL} {subset} ({len(ids)} questions x {len(HELD)} held-out pairs)")
        for k, c in s["success"].items():
            print(f"  success {k:26s} {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]")
        for arm, c in s["gap_closed"].items():
            print(f"  gap closed {arm:23s} {c[0]:.0%} [{c[1]:.0%}, {c[2]:.0%}]")
        for x, y in ((x, y) for x, y in COMPARE if x in arms and y in arms):
            tests = [mcnemar(ok(y, p).tolist(), ok(x, p).tolist()) for p in HELD]  # (x only, y only, p)
            adj = holm([t[2] for t in tests])
            better = [p for p, t, q in zip(HELD, tests, adj) if q < ALPHA and t[0] > t[1]]
            worse = [p for p, t, q in zip(HELD, tests, adj) if q < ALPHA and t[0] < t[1]]
            raw = sum(t[2] < ALPHA for t in tests)
            d = s["compare"][f"{x} vs {y}"] = {
                "diff_ci": ci(mean[x] - mean[y], boot), "pairs_higher": sum(t[0] > t[1] for t in tests),
                "pairs_lower": sum(t[0] < t[1] for t in tests), "significant_raw": raw,
                "significant_better_holm": better, "significant_worse_holm": worse,
                "per_pair": {p: dict(only_x=t[0], only_y=t[1], p=t[2], p_holm=q)
                             for p, t, q in zip(HELD, tests, adj)}}
            print(f"  {x} - {y}: {fmt(d['diff_ci'])}; higher on {d['pairs_higher']}, lower on {d['pairs_lower']} "
                  f"pairs; McNemar p<.05 on {raw}, after Holm better on {len(better)}, worse on {len(worse)}")
    path = RES / f"pilot{suffix}_all_tests.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("->", path)


if __name__ == "__main__":
    main()
