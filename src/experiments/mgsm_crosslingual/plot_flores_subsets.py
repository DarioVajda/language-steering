"""The paper's FLORES-size figure: mean held-out success of instruction + prompt-only steering against the number of
FLORES sentences the vector is computed from (1, 3, 10, 100: mean over the three draws, bar = their range; 997: the
main test run), per model, with the instruction alone as a dashed line. Reads results/<model>/flores_subsets.json
(flores_subsets.py); writes ../../../paper/figures/flores_subsets.pdf. Light work: runs on the login node.

    python src/experiments/mgsm_crosslingual/plot_flores_subsets.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "paper" / "figures" / "flores_subsets.pdf"
MODELS = {"llama-3.1-8b": "Llama-3.1-8B", "qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B",
          "gemma3-12b": "Gemma-3-12B"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, in this fixed order
MARKERS = ["o", "s", "^", "D"]


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.03, 2.25))
    for (m, name), c, mk in zip(MODELS.items(), COLORS, MARKERS):
        d = json.loads((HERE / "results" / m / "flores_subsets.json").read_text())
        ns = d["sizes"] + [997]
        draws = [[d["draws"][f"n{n}_s{s}"]["success"] for s in d["seeds"]] for n in d["sizes"]]
        y = [100 * sum(v) / len(v) for v in draws] + [100 * d["n_997"]["success"]]
        lo = [100 * min(v) for v in draws] + [y[-1]]
        hi = [100 * max(v) for v in draws] + [y[-1]]
        ax.axhline(100 * d["instruction"]["success"], color=c, lw=1, ls=(0, (3, 2)), alpha=0.8, zorder=1)
        ax.errorbar(ns, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], color=c, lw=1.6,
                    marker=mk, ms=4.5, mec="white", mew=0.7, capsize=0, elinewidth=1, zorder=3, label=name)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 100, 997], ["1", "3", "10", "100", "997"])
    ax.minorticks_off()
    ax.set_xlim(0.75, 1500)
    ax.set_ylim(30, 85)
    ax.set_xlabel("FLORES sentence pairs used for the vector")
    ax.set_ylabel("Success (%)")
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6, length=2.5, color="#555555")
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#777777", lw=1, ls=(0, (3, 2))))
    l.append("Instruction only")
    ax.legend(h, l, ncol=3, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
