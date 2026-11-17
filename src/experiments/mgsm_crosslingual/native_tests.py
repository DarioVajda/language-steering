"""How close the arms come to the native question (the question asked in T), on test.

Per held-out pair: an exact McNemar test on per-item success of the arm against the native question (the same 200
items), Holm-corrected over the 99 held-out pairs of each arm; pairs significantly below / above native. Per target
language T (the 9 held-out pairs into T, which all share the native run of T) and over all held-out pairs: the mean
success difference arm - native with a 95% CI from a bootstrap over items (2000 resamples, the same items for every
pair and arm in a resample, as in agreement.py). Arms: the instruction, instruction + prompt-only steering, and the
instruction in English + prompt-only steering, at the settings chosen on dev.

Also the 95% CIs (same bootstrap) of the headline means over the held-out pairs: native, and for each of the arms above
and the two controls (random direction, wrong-target vector), its mean success and its difference to the instruction.

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/native_tests.py   # -> results/<model>/native_tests.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from analyse import holm, load, mcnemar, success  # noqa: E402
from generate_pairs import GEN, KEY, LANGS, PAIRS, RES, TUNE  # noqa: E402

ARMS = ("instruction", "instruction_steer_prompt", "instruction_en_steer_prompt")
CONTROLS = ("instruction_steer_prompt_random", "instruction_steer_prompt_wrong")
HELD = tuple(p for p in PAIRS if p not in TUNE)
N_BOOT, SEED = 2000, 0


def ci(diff, boot):
    """Point estimate and 95% percentile interval of the mean over items, given resampled item indices."""
    return [float(diff.mean()), *map(float, np.percentile(diff[boot].mean(-1), [2.5, 97.5]))]


def fmt(c, sign=""):
    return f"{c[0]:{sign}.3f} [{c[1]:{sign}.3f}, {c[2]:{sign}.3f}]"


def main():
    T = GEN / "test"
    nat_rows = {L: load(T / f"native_{L}.jsonl") for L in LANGS}
    ids = [r["id"] for r in nat_rows[LANGS[0]]]
    nat = {L: np.array([success(r) for r in rows], dtype=float) for L, rows in nat_rows.items()}
    assert all([r["id"] for r in rows] == ids for rows in nat_rows.values())
    ok = {}
    for arm in ARMS + CONTROLS:
        for p in HELD:
            rows = load(T / f"{arm}_{KEY[p]}.jsonl")
            assert [r["id"] for r in rows] == ids
            ok[arm, p] = np.array([success(r) for r in rows], dtype=float)
    boot = np.random.default_rng(SEED).integers(0, len(ids), (N_BOOT, len(ids)))

    out = {"arms": ARMS, "held_out_pairs": [KEY[p] for p in HELD], "n_boot": N_BOOT, "per_arm": {}}
    mean = lambda arm: np.mean([ok[arm, p] for p in HELD], axis=0)  # per item, over the held-out pairs
    out["means"] = {"native": ci(np.mean([nat[p[1]] for p in HELD], axis=0), boot), **{
        arm: {"success_ci": ci(mean(arm), boot), "minus_instruction_ci": ci(mean(arm) - mean("instruction"), boot)}
        for arm in ARMS + CONTROLS}}
    for arm in ARMS:
        pairs = {}
        for p in HELD:
            g, l, pv = mcnemar([bool(x) for x in nat[p[1]]], [bool(x) for x in ok[arm, p]])
            pairs[KEY[p]] = dict(success=float(ok[arm, p].mean()), native=float(nat[p[1]].mean()),
                                 gained=int(g), lost=int(l), p=pv)
        for d, q in zip(pairs.values(), holm([d["p"] for d in pairs.values()])):
            d["p_holm"] = q
        count = lambda k, s: sum(d[k] < 0.05 and (d["gained"] - d["lost"]) * s > 0 for d in pairs.values())
        by_target = {}
        for L in LANGS:
            ps = [p for p in HELD if p[1] == L]
            diff = np.mean([ok[arm, p] for p in ps], axis=0) - nat[L]
            by_target[L] = dict(success=float(np.mean([ok[arm, p].mean() for p in ps])), native=float(nat[L].mean()),
                                diff_ci=ci(diff, boot),
                                below_above_holm=[sum(pairs[KEY[p]]["p_holm"] < 0.05 and pairs[KEY[p]]["lost"] >
                                                      pairs[KEY[p]]["gained"] for p in ps),
                                                  sum(pairs[KEY[p]]["p_holm"] < 0.05 and pairs[KEY[p]]["gained"] >
                                                      pairs[KEY[p]]["lost"] for p in ps)])
        diff = np.mean([ok[arm, p] - nat[p[1]] for p in HELD], axis=0)
        out["per_arm"][arm] = {
            "success": float(np.mean([ok[arm, p].mean() for p in HELD])),
            "native": float(np.mean([nat[p[1]].mean() for p in HELD])),
            "diff_ci": ci(diff, boot),
            "below_above": {k: [count(k, -1), count(k, 1)] for k in ("p", "p_holm")},
            "by_target": by_target, "pairs": pairs}

    print(f"held-out pairs below / above native (McNemar p < 0.05, uncorrected [Holm over {len(HELD)}]); "
          f"success - native, mean [95% CI over items]")
    print(f"headline means, 95% CI over items: native {fmt(out['means']['native'])}")
    for arm in ARMS + CONTROLS:
        x = out["means"][arm]
        print(f"  {arm}: {fmt(x['success_ci'])}, minus instruction {fmt(x['minus_instruction_ci'], '+')}")
    for arm, x in out["per_arm"].items():
        b, d = x["below_above"], x["diff_ci"]
        print(f"\n{arm}: {x['success']:.3f} vs native {x['native']:.3f}, {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]; "
              f"below / above {b['p'][0]} / {b['p'][1]} [{b['p_holm'][0]} / {b['p_holm'][1]}]")
        for L, t in x["by_target"].items():
            d = t["diff_ci"]
            print(f"  into {L}: {t['success']:.3f} vs {t['native']:.3f}  {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]  "
                  f"pairs below / above (Holm) {t['below_above_holm'][0]} / {t['below_above_holm'][1]}")
    (RES / "native_tests.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
