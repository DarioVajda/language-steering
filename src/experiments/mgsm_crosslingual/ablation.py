"""The ablation of our two design choices against ReCoVeR on cross-lingual MGSM: our method (instruction + steering,
one layer, prompt-only: instruction_steer_prompt) against the arms that each change one thing,
  tokens:  instruction_steer      one layer, all-token (also every generated token);
  layers:  every_steer_prompt     every block but the last, prompt-only (recover.py, XL_ARM=every_steer_prompt);
  baseline: recover               ReCoVeR as published (recover.py);
each at its own setting chosen on dev, on the 99 held-out pairs of test. Per comparison, as in recover.py: the mean
success difference ours - other with its 95% bootstrap CI over the MGSM items (the same resampled items in every
pair), and the held-out pairs significantly better/worse for ours (McNemar on success, p < .05, and after Holm over
the 99 pairs). Also the means of success, accuracy and in-T rate. Login-node work (no model).

Usage: XL_MODEL=<model> python ablation.py   -> results/<model>/ablation.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from analyse import holm, load, mcnemar, score, success  # noqa: E402
from generate_pairs import GEN, KEY, MODEL, PAIRS, RES, TUNE  # noqa: E402
from native_tests import N_BOOT, SEED, ci, fmt  # noqa: E402

HELD = tuple(p for p in PAIRS if p not in TUNE)
OURS = "instruction_steer_prompt"
OTHERS = {"instruction_steer": "tokens: all-token", "every_steer_prompt": "layers: every layer",
          "recover": "baseline: ReCoVeR"}
SHOWN = ("instruction", OURS) + tuple(OTHERS)


def path(arm, p):
    if arm in ("recover", "every_steer_prompt"):
        return GEN / arm / "test" / f"{arm}_{KEY[p]}.jsonl"
    return GEN / "test" / f"{arm}_{KEY[p]}.jsonl"


def main():
    rows = {(a, p): load(path(a, p)) for a in SHOWN for p in HELD}
    ids = [r["id"] for r in rows["instruction", HELD[0]]]
    assert all([r["id"] for r in rs] == ids for rs in rows.values())
    for a in ("recover", "every_steer_prompt"):  # the test runs are at the setting chosen on dev
        c = json.loads((GEN / a / "chosen.json").read_text())
        assert all(r["alpha"] == c["alpha"] for p in HELD for r in rows[a, p]), a
    ok = {k: np.array([success(r) for r in rs], dtype=float) for k, rs in rows.items()}
    sc = {k: score(rs) for k, rs in rows.items()}
    n = len(ids)
    boot = np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))
    item = lambda a: np.mean([ok[a, p] for p in HELD], axis=0)
    out = {"held_out_pairs": [KEY[p] for p in HELD], "n_boot": N_BOOT, "seed": SEED, "ours": OURS,
           "means": {a: {c: float(np.mean([sc[a, p][c] for p in HELD])) for c in ("success", "accuracy", "in_lang")}
                     | {"success_ci": ci(item(a), boot)} for a in SHOWN},
           "comparisons": {}}
    for b, what in OTHERS.items():
        per = [mcnemar([bool(x) for x in ok[b, p]], [bool(x) for x in ok[OURS, p]]) for p in HELD]
        ps = [x[2] for x in per]
        cnt = lambda qs, s: sum(q < .05 and (g - l) * s > 0 for (g, l, _), q in zip(per, qs))
        out["comparisons"][f"{OURS}-{b}"] = dict(
            what=what, diff_ci=ci(item(OURS) - item(b), boot),
            pairs_better_worse=[cnt(ps, 1), cnt(ps, -1)], pairs_better_worse_holm=[cnt(holm(ps), 1), cnt(holm(ps), -1)])
    print(f"\n{MODEL}: cross-lingual MGSM, {len(HELD)} held-out pairs (n = {n} each)")
    print(f"{'':26}{'success [95% CI]':>24}{'accuracy':>10}{'in T':>8}")
    for a in SHOWN:
        m = out["means"][a]
        print(f"{a:26}{fmt(m['success_ci']):>24}{m['accuracy']:>10.3f}{m['in_lang']:>8.3f}")
    print("\nours - other: success difference [95% CI]; held-out pairs better/worse for ours, p < .05 (Holm)")
    for k, d in out["comparisons"].items():
        bw, bh = d["pairs_better_worse"], d["pairs_better_worse_holm"]
        print(f"  {d['what']:22}{k:46}{fmt(d['diff_ci'], '+'):>26}   {bw[0]}/{bw[1]} ({bh[0]}/{bh[1]})")
    (RES / "ablation.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
