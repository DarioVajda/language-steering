"""Score the cross-lingual MGSM generations, choose the steering settings on dev, compare the arms on test.

Scoring is that of ../mgsm_steering/analyse.py (imported): success = correct AND in the wanted language (T, or S
for the source reference). Selection (dev, per steering arm): the (layer, alpha) with the highest success rate
averaged over the 10 pairs; ties go to the smaller alpha, then the earlier layer. Test: exact McNemar test on
per-item success within each pair, against the instruction. The pilot (instruction on all 100 pairs, dev) is
reported only.

Usage: python src/experiments/mgsm_crosslingual/analyse_pairs.py --select   # dev -> results/generations/chosen.json
       python src/experiments/mgsm_crosslingual/analyse_pairs.py            # all -> results/summary.json
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from analyse import load, mcnemar, score, success  # noqa: E402
from generate_pairs import GEN, KEY, LANGS, PAIRS, SOURCES, STEER_ARMS, units  # noqa: E402

ARMS = ("source", "native", "instruction") + STEER_ARMS
COLS = ("success", "accuracy", "in_lang", "english", "capped")


def wait_for(stage):
    """Wait until every unit of the stage has its output (units may still be running in other workers)."""
    while missing := [n for n in units(stage) if not (GEN / stage / f"{n}.jsonl").exists()]:
        print(f"waiting for {len(missing)} {stage} units", flush=True)
        time.sleep(30)


def by_pair(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[KEY[(r["src"], r["tgt"])]].append(r)
    return g


def pilot():
    """{S-T: score} of the instruction on dev for all 100 pairs; prints the success and in-T matrices."""
    wait_for("pilot")
    g = by_pair([r for s in SOURCES for r in load(GEN / "pilot" / f"{s}.jsonl")])
    res = {k: score(v) for k, v in g.items()}
    for c in ("success", "in_lang", "accuracy"):
        print(f"\npilot (instruction, dev): {c}; rows S, columns T")
        print(f"{'':4}" + "".join(f"{t:>6}" for t in LANGS))
        for s in SOURCES:
            print(f"{s:4}" + "".join(f"{res[f'{s}-{t}'][c]:>6.2f}" if t != s else f"{'':>6}" for t in LANGS))
    return res


def select():
    """{steering arm: {layer, alpha, mean_success}} chosen on dev, and the dev grids; prints the dev scores."""
    wait_for("dev")
    dev = {f.stem: by_pair(load(f)) for f in sorted((GEN / "dev").glob("*.jsonl"))}
    keys = [KEY[p] for p in PAIRS]
    print("dev, success / accuracy / in the wanted language per pair")
    print(f"{'':12}" + "".join(f"{k:>16}" for k in keys))
    base = {}
    for arm in ("source", "native", "instruction"):
        s = {k: score(dev[arm][k]) for k in keys}
        base[arm] = sum(s[k]["success"] for k in keys) / len(keys)
        print(f"{arm:12}" + "".join(f"  {s[k]['success']:.2f} {s[k]['accuracy']:.2f} {s[k]['in_lang']:.2f}"
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
        print(f"\n{arm}: mean dev success over pairs (rows: layer, columns: alpha)")
        alphas = sorted({a for _, a in means})
        print(f"{'':6}" + "".join(f"{a:>7}" for a in alphas))
        for l in sorted({l for l, _ in means}):
            print(f"{l:>6}" + "".join(f"{means[(l, a)]:>7.3f}" for a in alphas))
        print(f"chosen: layer {layer}, alpha {alpha} ({means[(layer, alpha)]:.3f})")
    (GEN / "chosen.json").write_text(json.dumps(chosen, indent=1) + "\n")
    return chosen, grid, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    if args.select:
        select()
        return
    out = {"pairs": [KEY[p] for p in PAIRS], "pilot": pilot()}
    out["chosen"], out["dev_grid"], out["dev_baselines"] = select()
    wait_for("test")
    T = GEN / "test"
    out["test"] = {}
    for p in PAIRS:
        k = KEY[p]
        rows = {arm: load(T / f"{arm}_{k}.jsonl") for arm in ARMS}
        base = rows["instruction"]
        print(f"\n== {k}  (test, n = {len(base)})")
        print(f"{'':26}" + "".join(f"{c:>9}" for c in COLS) + f"{'tokens':>8}   success vs instruction (McNemar)")
        res = out["test"][k] = {}
        for arm in ARMS:
            d = res[arm] = score(rows[arm])
            cmp = ""
            if arm not in ("instruction", "source"):
                assert [r["id"] for r in rows[arm]] == [r["id"] for r in base]
                n01, n10, pv = mcnemar([success(r) for r in base], [success(r) for r in rows[arm]])
                d.update(gained=n01, lost=n10, p=pv)
                cmp = f"   +{n01} / -{n10}, p = {pv:.3f}"
            if arm in STEER_ARMS:
                d.update(layer=out["chosen"][arm]["layer"], alpha=out["chosen"][arm]["alpha"])
            print(f"{arm:26}" + "".join(f"{d[c]:>9.2f}" for c in COLS) + f"{d['median_tokens']:>8}{cmp}")
    print("\n== mean over pairs (source: in S)")
    for arm in ARMS:
        print(f"{arm:26}" + "".join(f"{sum(out['test'][KEY[p]][arm][c] for p in PAIRS) / len(PAIRS):>9.3f}"
                                    for c in COLS))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
