"""How strong is each language's signal in single-word states? (Is English a near-zero "default"?)

For every language L and layer, on all segmented words of the test half (last-token states),
with mu_L, mu and u_L = (mu_L - mu) / |mu_L - mu| fitted on the fit half's token means:
  offset      |mu_L - mu| / rms|h - mu| over L's words: the language offset relative to
              the typical size of a centred word state
  gamma_share mean gamma_L^2 / mean |h - mu|^2 over L's words, gamma_L = (h - mu).u_L: the
              share of a word's centred state along its own language direction
  snr         mean / std of gamma_L over L's words
  dprime      (mean gamma_L on L's words - mean gamma_L on other languages' words) / pooled
              std: how well gamma_L separates L from the rest
  id_acc      fraction of L's words whose argmax over L' of (h - mu).u_L' is L

Outputs: results/lang_signal.json and lang_signal.png.
Usage (GPU job, via slurm/submit.sh):
    slurm/submit.sh -J lang_signal -- python src/experiments/word_level_analysis/lang_signal.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
COLORS = {"en": "#e34948", "zh": "#2a78d6", "fr": "#eb6834", "de": "#1baf7a", "ru": "#eda100",
          "es": "#e87ba4", "sw": "#008300", "th": "#6250d6"}


def test_rows(records, langs, row_start):
    """Per language: state rows of all its words in test-half statements."""
    return {L: np.concatenate([np.arange(row_start[a, b], row_start[a, b] + len(r["words"][L]))
                               for a, r in enumerate(records) if r["half"] == "test"])
            for b, L in enumerate(langs)}


def layer_metrics(x, rows, mu_L, mu, langs):
    u = (mu_L - mu) / (mu_L - mu).norm(dim=1, keepdim=True)
    h = {L: x[torch.from_numpy(rows[L]).to(x.device)].double() - mu for L in langs}
    g = {L: h[L] @ u.T for L in langs}                                   # [n_L, K] projections
    out = {}
    for b, L in enumerate(langs):
        own = g[L][:, b]
        other = torch.cat([g[M][:, b] for M in langs if M != L])
        sq = (h[L] * h[L]).sum(1)
        pooled = ((own.var() + other.var()) / 2).sqrt()
        out[L] = {
            "offset": float((mu_L[b] - mu).norm() / sq.mean().sqrt()),
            "offset_abs": float((mu_L[b] - mu).norm()),
            "gamma_share": float((own ** 2).mean() / sq.mean()),
            "snr": float(own.mean() / own.std()),
            "dprime": float((own.mean() - other.mean()) / pooled),
            "id_acc": float((g[L].argmax(1) == b).double().mean()),
        }
    return out


def plot(results, langs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = np.arange(len(results))
    n = len(results) - 1
    ticks = [0, 4, 8, 12, 16, 20, 24, 28, n]
    specs = [("offset", "|mu_L - mu| / rms|h - mu|"), ("gamma_share", "share of |h - mu|^2 along u_L"),
             ("dprime", "d' of gamma_L: own vs other languages"), ("id_acc", "language ID accuracy (argmax projection)")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
    for ax, (key, title) in zip(axes, specs):
        for L in langs:
            ax.plot(layers, [r[L][key] for r in results], color=COLORS[L], lw=2.4 if L == "en" else 1.5, label=L)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(ticks, ["emb" if t == 0 else f"L{t}" for t in ticks])
    axes[-1].axhline(1 / len(langs), color="#8a8a85", lw=0.8, ls="--")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=HERE / "states" / "llama-3.1-8b-instruct")
    parser.add_argument("--alignments", type=Path, default=HERE / "results" / "alignments_simalign.json")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    al = json.loads(args.alignments.read_text())
    langs, records = al["langs"], al["records"]
    words = np.load(args.states / "words.npz")
    tok = np.load(args.states / "token_sums.npz")
    rows = test_rows(records, langs, words["row_start"])
    states = np.load(args.states / "last.npy", mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_fit = tok["counts"][:, :, 0].sum(1)
    print("test words: " + ", ".join(f"{L} {len(r)}" for L, r in rows.items()), flush=True)

    results = []
    for layer in range(states.shape[0]):
        x = torch.from_numpy(np.array(states[layer])).to(device)
        mu_L = torch.from_numpy(tok["sums"][:, :, 0, layer].sum(1) / n_fit[:, None]).to(device)
        results.append(layer_metrics(x, rows, mu_L, mu_L.mean(0), langs))
        print(f"layer {layer:>2}: " + " | ".join(
            f"{L} off {m['offset']:.2f} share {m['gamma_share']:.3f} d' {m['dprime']:.1f} id {m['id_acc']:.2f}"
            for L, m in results[-1].items()), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lang_signal.json").write_text(json.dumps({"langs": langs, "layers": results}, indent=1) + "\n")
    plot(results, langs, args.out / "lang_signal.png")
    print(f"wrote {args.out / 'lang_signal.json'} and lang_signal.png")


if __name__ == "__main__":
    main()
