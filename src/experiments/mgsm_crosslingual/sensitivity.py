"""Sensitivity of steering to the layer and alpha, from the dev grids already stored (no generation).

Cross-lingual MGSM: for every model and steering arm, the mean dev success over the 11 tuning pairs at every
(layer, alpha) of results/<model>/summary.json (written by analyse_pairs.py): the chosen cell, how many cells come within
0.02 of it and how many beat the instruction, and, at the chosen alpha, the success per layer and the share of the
chosen cell's gain over the instruction that the worst layer keeps (only where that gain is at least MIN_GAIN, as the
share is meaningless when the chosen cell barely beats the instruction).
LCB (../language_confusion/results/<model>/summary.json, all-token): the mean dev LPR and degenerate rate over the
six languages at every (layer, alpha).

These are dev numbers, so the chosen cell is the maximum of the grid and slightly optimistic; the test results at the
chosen settings are in the READMEs.

Usage: python src/experiments/mgsm_crosslingual/sensitivity.py   # -> results/sensitivity.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LCB = HERE.parent / "language_confusion" / "results"
MODELS = ("llama-3.1-8b", "qwen3-8b", "qwen3-32b", "gemma3-12b")
NEAR = 0.02
MIN_GAIN = 0.05


def mgsm(model):
    s = json.loads((HERE / "results" / model / "summary.json").read_text())
    base = s["dev_baselines"]
    out = {"instruction": base["instruction"], "native": base["native"], "arms": {}}
    for arm, cells in s["dev_grid"].items():
        grid = {(c["layer"], c["alpha"]): c["mean_success"] for c in cells}
        ch = s["chosen"][arm]
        best, alpha = ch["mean_success"], ch["alpha"]
        layers = sorted({l for l, _ in grid})
        row = {l: grid[(l, alpha)] for l in layers}
        gain = best - base["instruction"]
        out["arms"][arm] = {
            "chosen": {"layer": ch["layer"], "alpha": alpha, "success": best},
            "n_cells": len(grid),
            "near_best": sum(v >= best - NEAR for v in grid.values()),
            "above_instruction": sum(v > base["instruction"] for v in grid.values()),
            "at_chosen_alpha": row,
            "worst_layer_share_of_gain": (min(row.values()) - base["instruction"]) / gain if gain >= MIN_GAIN else None,
        }
    return out


def lcb(model):
    s = json.loads((LCB / model / "summary.json").read_text())
    langs = sorted(s)
    out = {"languages": langs,
           "instruction": sum(next(c["lpr"] for c in s[L]["dev_grid"] if c["cond"] == "instruction")
                              for L in langs) / len(langs), "arms": {}}
    for cond in ("steer", "instruction_steer"):
        cells = {}
        for L in langs:
            for c in s[L]["dev_grid"]:
                if c["cond"] == cond:
                    cells.setdefault((c["layer"], c["alpha"]), []).append(c)
        out["arms"][cond] = [{"layer": l, "alpha": a,
                              "lpr": sum(c["lpr"] for c in cs) / len(cs),
                              "degenerate": sum(c["degenerate"] for c in cs) / len(cs)}
                             for (l, a), cs in sorted(cells.items()) if len(cs) == len(langs)]
    return out


def main():
    res = {"mgsm": {m: mgsm(m) for m in MODELS}, "lcb": {m: lcb(m) for m in MODELS}, "near": NEAR}
    for m, r in res["mgsm"].items():
        print(f"\n== MGSM dev, {m}: instruction {r['instruction']:.3f}, native {r['native']:.3f}")
        for arm, a in r["arms"].items():
            ch = a["chosen"]
            share = a["worst_layer_share_of_gain"]
            print(f"{arm:28} chosen l {ch['layer']}, alpha {ch['alpha']}: {ch['success']:.3f}; "
                  f"within {NEAR} {a['near_best']}/{a['n_cells']}, above instruction {a['above_instruction']}/{a['n_cells']}; "
                  f"per layer at alpha {ch['alpha']}: " + " ".join(f"{v:.3f}" for v in a["at_chosen_alpha"].values())
                  + ("" if share is None else f"; worst layer keeps {share:.0%} of the gain"))
    for m, r in res["lcb"].items():
        print(f"\n== LCB dev (all-token), {m}: mean LPR over {len(r['languages'])} languages, instruction "
              f"{r['instruction']:.2f}; degenerate in brackets where above 0.05")
        for cond, cells in r["arms"].items():
            alphas = sorted({c["alpha"] for c in cells})
            print(f"{cond:18}" + "".join(f"{a:>12}" for a in alphas))
            for l in sorted({c["layer"] for c in cells}):
                row = {c["alpha"]: c for c in cells if c["layer"] == l}
                print(f"  {l:>3}{'':13}" + "".join(
                    f"{row[a]['lpr']:6.2f}" + (f"({row[a]['degenerate']:.2f})" if row[a]["degenerate"] > 0.05 else " " * 6)
                    if a in row else " " * 12 for a in alphas))
    (HERE / "results" / "sensitivity.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
