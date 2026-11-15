"""Score the cross-lingual MGSM generations, choose the steering settings on dev, compare the arms on test.

Scoring is that of ../mgsm_steering/analyse.py (imported): success = correct AND in the wanted language (T, or S
for the source reference). Selection (dev, per steering arm): the (layer, alpha) with the highest success rate
averaged over the 11 tuning pairs; ties go to the smaller alpha, then the earlier layer. The instruction
variant (INSTRUCTION_ARMS) is chosen the same way, ties to the first (the instruction), -> chosen_instruction.json;
if it is not the instruction, it is added to the test comparison. The controls (CONTROL_ARMS) are also tested
against instruction_steer_prompt (vs_steer). Test: exact McNemar
test on per-item success within each pair, against the instruction; means over the tuning pairs, the 99
held-out pairs and all 110, and per source and per target language over the held-out pairs. Held-out pairs better /
worse (better_worse, better_worse_vs_steer): p < 0.05, uncorrected (p) and Holm-corrected over the 99 held-out pairs
of each comparison (p_holm, stored for the held-out pairs).

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/analyse_pairs.py --select   # dev -> results/<model>/generations/chosen.json
       XL_MODEL=<model> python src/experiments/mgsm_crosslingual/analyse_pairs.py            # all -> results/<model>/summary.json
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from analyse import holm, load, mcnemar, score, success  # noqa: E402
from generate_pairs import CONTROL_ARMS, GEN, RES, INSTRUCTION_ARMS, KEY, LANGS, PAIRS, STEER_ARMS, TUNE, units  # noqa: E402

BASE_ARMS = ("source", "native", "instruction") + STEER_ARMS + CONTROL_ARMS
COLS = ("success", "accuracy", "in_lang", "english", "capped")
HELD = tuple(p for p in PAIRS if p not in TUNE)


def wait_for(stage):
    """Wait until every unit of the stage has its output (units may still be running in other workers)."""
    while missing := [n for n in units(stage) if not (GEN / stage / f"{n}.jsonl").exists()]:
        print(f"waiting for {len(missing)} {stage} units", flush=True)
        time.sleep(30)


def correct(ds):
    """Add p_holm to the comparisons ds (one per held-out pair): Holm over the pairs."""
    for d, q in zip(ds, holm([d["p"] for d in ds])):
        d["p_holm"] = q


def better_worse(ds):
    """Number of comparisons significantly better / worse, uncorrected (p) and Holm-corrected (p_holm)."""
    return {k: [sum(d[k] < 0.05 and d["gained"] > d["lost"] for d in ds), sum(d[k] < 0.05 and d["gained"] < d["lost"] for d in ds)]
            for k in ("p", "p_holm")}


def by_pair(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[KEY[(r["src"], r["tgt"])]].append(r)
    return g


def select():
    """{steering arm: {layer, alpha, mean_success}} chosen on dev, the dev grids, the dev baselines and the chosen
    instruction variant; prints the dev scores."""
    wait_for("dev")
    dev = {f.stem: by_pair(load(f)) for f in sorted((GEN / "dev").glob("*.jsonl"))}
    keys = [KEY[p] for p in TUNE]
    print("dev, success / accuracy / in the wanted language per tuning pair")
    print(f"{'':18}" + "".join(f"{k:>16}" for k in keys))
    base = {}
    for arm in ("source", "native") + INSTRUCTION_ARMS:
        s = {k: score(dev[arm][k]) for k in keys}
        base[arm] = sum(s[k]["success"] for k in keys) / len(keys)
        print(f"{arm[:18]:18}" + "".join(f"  {s[k]['success']:.2f} {s[k]['accuracy']:.2f} {s[k]['in_lang']:.2f}"
                                    for k in keys) + f"   mean success {base[arm]:.3f}")
    chosen, grid = {}, {}
    for arm in STEER_ARMS:
        means = {}
        for name, g in dev.items():
            r0 = next(iter(g.values()))[0]
            if r0["arm"] == arm:
                means[(r0["layer"], r0["alpha"])] = sum(score(g[k])["success"] for k in keys) / len(keys)
        (layer, alpha) = min(means, key=lambda k: (-means[k], k[1], k[0]))
        chosen[arm] = dict(layer=layer, alpha=alpha, mean_success=means[(layer, alpha)])
        grid[arm] = [dict(layer=l, alpha=a, mean_success=m) for (l, a), m in sorted(means.items())]
        print(f"\n{arm}: mean dev success over tuning pairs (rows: layer, columns: alpha)")
        alphas = sorted({a for _, a in means})
        print(f"{'':6}" + "".join(f"{a:>7}" for a in alphas))
        for l in sorted({l for l, _ in means}):
            print(f"{l:>6}" + "".join(f"{means[(l, a)]:>7.3f}" for a in alphas))
        print(f"chosen: layer {layer}, alpha {alpha} ({means[(layer, alpha)]:.3f})")
    (GEN / "chosen.json").write_text(json.dumps(chosen, indent=1) + "\n")
    variant = min(INSTRUCTION_ARMS, key=lambda a: (-base[a], INSTRUCTION_ARMS.index(a)))
    print(f"\ninstruction variant: {variant} ({base[variant]:.3f})")
    (GEN / "chosen_instruction.json").write_text(json.dumps(
        dict(arm=variant, mean_success={a: base[a] for a in INSTRUCTION_ARMS}), indent=1) + "\n")
    return chosen, grid, base, variant


def mean(test, pairs, arm, c):
    return sum(test[KEY[p]][arm][c] for p in pairs) / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    if args.select:
        select()
        return
    out = {"tuning_pairs": [KEY[p] for p in TUNE]}
    out["chosen"], out["dev_grid"], out["dev_baselines"], variant = select()
    out["instruction_variant"] = variant
    extra = (variant,) if variant != "instruction" else ()
    ARMS = BASE_ARMS + extra
    wait_for("test")
    T = GEN / "test"
    src = {L: load(T / f"source_{L}.jsonl") for L in LANGS}
    nat = {L: load(T / f"native_{L}.jsonl") for L in LANGS}
    test = out["test"] = {}
    print(f"\ntest success per pair (n = 200; * McNemar p < 0.05 against the instruction; T: tuning pair)")
    print(f"{'':8}" + "".join(f"{a[:14]:>16}" for a in ARMS))
    for p in PAIRS:
        k = KEY[p]
        rows = {"source": src[p[0]], "native": nat[p[1]]}
        rows.update({arm: load(T / f"{arm}_{k}.jsonl") for arm in ("instruction",) + STEER_ARMS + CONTROL_ARMS + extra})
        base = rows["instruction"]
        res = test[k] = {}
        cells = []
        for arm in ARMS:
            d = res[arm] = score(rows[arm])
            mark = " "
            if arm not in ("instruction", "source"):
                assert [r["id"] for r in rows[arm]] == [r["id"] for r in base]
                n01, n10, pv = mcnemar([success(r) for r in base], [success(r) for r in rows[arm]])
                d.update(gained=n01, lost=n10, p=pv)
            if arm in CONTROL_ARMS:
                isp = [success(r) for r in rows["instruction_steer_prompt"]]
                n01, n10, pv = mcnemar(isp, [success(r) for r in rows[arm]])
                d["vs_steer"] = dict(gained=n01, lost=n10, p=pv)
                mark = "*" if pv < 0.05 else " "
            if arm in STEER_ARMS:
                d.update(layer=out["chosen"][arm]["layer"], alpha=out["chosen"][arm]["alpha"])
            cells.append(f"{d['success']:>15.2f}{mark}")
        print(f"{k:6}{'T' if p in TUNE else ' ':2}" + "".join(cells))

    tested = [arm for arm in ARMS if arm not in ("instruction", "source")]
    for arm in tested:
        correct([test[KEY[p]][arm] for p in HELD])
    for arm in CONTROL_ARMS:
        correct([test[KEY[p]][arm]["vs_steer"] for p in HELD])
    out["better_worse"] = {arm: better_worse([test[KEY[p]][arm] for p in HELD]) for arm in tested}
    out["better_worse_vs_steer"] = {arm: better_worse([test[KEY[p]][arm]["vs_steer"] for p in HELD]) for arm in CONTROL_ARMS}

    groups = {"tuning": TUNE, "held_out": HELD, "all": PAIRS}
    out["means"] = {g: {arm: {c: mean(test, ps, arm, c) for c in COLS} for arm in ARMS} for g, ps in groups.items()}
    for g, ps in groups.items():
        print(f"\n== mean over the {len(ps)} {g} pairs")
        print(f"{'':26}" + "".join(f"{c:>9}" for c in COLS) + "   held-out pairs better / worse than instruction (Holm)")
        for arm in ARMS:
            sig = ""
            if g == "held_out" and arm in tested:
                bw = out["better_worse"][arm]
                sig = f"   {bw['p'][0]} / {bw['p'][1]}   ({bw['p_holm'][0]} / {bw['p_holm'][1]})"
            print(f"{arm:26}" + "".join(f"{out['means'][g][arm][c]:>9.3f}" for c in COLS) + sig)

    print("\ncontrols against instruction + prompt-only steering, held-out pairs better / worse (Holm)")
    for arm, bw in out["better_worse_vs_steer"].items():
        print(f"{arm:26}   {bw['p'][0]} / {bw['p'][1]}   ({bw['p_holm'][0]} / {bw['p_holm'][1]})")

    for side, i in (("source", 0), ("target", 1)):
        out[f"by_{side}"] = {L: {arm: sum(test[KEY[p]][arm]["success"] for p in HELD if p[i] == L) / 9
                                 for arm in ARMS} for L in LANGS}
        print(f"\n== mean held-out success by {side} language (9 pairs each)")
        print(f"{'':6}" + "".join(f"{a[:14]:>16}" for a in ARMS))
        for L in LANGS:
            print(f"{L:6}" + "".join(f"{out[f'by_{side}'][L][arm]:>16.3f}" for arm in ARMS))
    (RES / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
