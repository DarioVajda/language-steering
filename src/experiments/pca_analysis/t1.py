"""EXP1 / T1: do translations differ only inside a shared (K-1)-dimensional language subspace?

For every statement, its K language versions are centred on their own mean. By Hypothesis 1
(README) the meaning and the global mean cancel, so every residual should lie in
span{u^L}, the (K-1)-dimensional span of the global language directions.

Per layer (index 0 = embedding output, i = output of block i; the last one is after the
final RMSNorm), with all language directions and PCs fitted on the fit half and every
metric measured on the test half:

  span_energy     fraction of residual energy ||r||^2 inside span{u^L}       (main metric)
  pc_var          fraction explained by the top K-1 PCs of the fit-half residuals
  pc_var_insample same, but with PCs of the test residuals themselves (upper bound)
  pc_overlap      mean cos^2 of the principal angles between those fit PCs and span{u^L}
  random_energy   fraction inside a random (K-1)-dim subspace (mean of 5 draws; ~ (K-1)/d)
  mismatched_*    span_energy / pc_var when, within each benchmark, every language's
                  statements are shuffled independently, so the K vectors that are centred
                  together are not translations (meaning does not cancel)

Each metric is computed with all dimensions ("all") and with the massive-activation
dimensions removed ("no_massive"; dimensions whose mean |activation| on the fit half is
> 20x the median dimension's, cf. Sun et al., 2024). Also reported: span_energy per
benchmark, and a cross-domain matrix (u^L fitted on benchmark A, evaluated on B).

Outputs: results/t1.json and results/t1.png next to this script.
Usage (GPU job, see CLAUDE.md):
    slurm/submit.sh -J t1 -- python src/experiments/pca_analysis/t1.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_statements

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASSIVE_RATIO = 20
N_RANDOM = 5
SEED = 0


def residuals(x):
    """[n, K, d] -> per-statement centred residuals, flattened to [n * K, d]."""
    return (x - x.mean(1, keepdim=True)).reshape(-1, x.shape[-1])


def language_basis(x):
    """Orthonormal basis [K-1, d] of span{u^L}, from states x [n, K, d]."""
    mu_lang = x.mean(0)
    directions = mu_lang - mu_lang.mean(0)
    u = directions / directions.norm(dim=1, keepdim=True)
    # The K directions sum to zero before normalisation, so they span K-1 dimensions.
    return torch.linalg.svd(u, full_matrices=False).Vh[: x.shape[1] - 1]


def energy(r, basis):
    """Fraction of the energy of the rows of r inside the row space of the orthonormal `basis`."""
    return float((r @ basis.T).square().sum() / r.square().sum())


def top_pcs(r, k):
    return torch.linalg.svd(r, full_matrices=False).Vh[:k]


def overlap(a, b):
    """Mean cos^2 of the principal angles between two k-dim orthonormal bases."""
    return float((a @ b.T).square().sum() / a.shape[0])


def mismatch(x, benchmarks, gen):
    """Shuffle statements independently per language, within each benchmark."""
    out = x.clone()
    for b in np.unique(benchmarks):
        idx = torch.from_numpy(np.flatnonzero(benchmarks == b)).to(x.device)
        for k in range(x.shape[1]):
            out[idx, k] = x[idx[torch.randperm(len(idx), generator=gen, device="cpu").to(x.device)], k]
    return out


def rows(x, mask):
    """Select the statements of x where the numpy bool `mask` is true."""
    return x[torch.from_numpy(mask).to(x.device)]


def massive_dims(x_fit):
    scale = x_fit.abs().mean((0, 1))
    return torch.nonzero(scale > MASSIVE_RATIO * scale.median()).flatten()


def layer_metrics(fit, test, test_bench, gen):
    k = fit.shape[1] - 1
    basis = language_basis(fit)
    r_test = residuals(test)
    pcs = top_pcs(residuals(fit), k)
    rand = [torch.linalg.qr(torch.randn(fit.shape[-1], k, generator=gen, dtype=fit.dtype).to(fit.device)).Q.T
            for _ in range(N_RANDOM)]
    r_mis = residuals(mismatch(test, test_bench, gen))
    return {
        "span_energy": energy(r_test, basis),
        "pc_var": energy(r_test, pcs),
        "pc_var_insample": energy(r_test, top_pcs(r_test, k)),
        "pc_overlap": overlap(pcs, basis),
        "random_energy": float(np.mean([energy(r_test, q) for q in rand])),
        "mismatched_span_energy": energy(r_mis, basis),
        "mismatched_pc_var": energy(r_mis, pcs),
        "span_energy_by_benchmark": {
            b: energy(residuals(rows(test, test_bench == b)), basis) for b in np.unique(test_bench)
        },
    }


def cross_domain(fit, fit_bench, test, test_bench):
    """span_energy with u^L fitted on the fit half of benchmark A and evaluated on the test half of B."""
    benchmarks = list(np.unique(fit_bench))
    bases = {a: language_basis(rows(fit, fit_bench == a)) for a in benchmarks}
    return {a: {b: energy(residuals(rows(test, test_bench == b)), bases[a]) for b in benchmarks}
            for a in benchmarks}


def run(states, half, bench, device):
    fit_idx, test_idx = np.flatnonzero(half == "fit"), np.flatnonzero(half == "test")
    gen = torch.Generator().manual_seed(SEED)
    results = []
    for layer in range(states.shape[2]):
        x = torch.from_numpy(states[:, :, layer]).to(device, torch.float64)
        fit, test = x[fit_idx], x[test_idx]
        massive = massive_dims(fit)
        keep = torch.ones(x.shape[-1], dtype=torch.bool, device=device)
        keep[massive] = False
        entry = {"layer": layer, "massive_dims": massive.tolist()}
        for variant, cols in [("all", slice(None)), ("no_massive", keep)]:
            f, t = fit[..., cols], test[..., cols]
            entry[variant] = layer_metrics(f, t, bench[test_idx], gen)
            entry[variant]["cross_domain"] = cross_domain(f, bench[fit_idx], t, bench[test_idx])
        results.append(entry)
        a = entry["all"]
        print(f"layer {layer:>2}: span_energy {a['span_energy']:.3f} (mismatched {a['mismatched_span_energy']:.3f}, "
              f"random {a['random_energy']:.4f}), pc_var {a['pc_var']:.3f}, pc_overlap {a['pc_overlap']:.3f}, "
              f"massive dims {entry['massive_dims']}", flush=True)
    return results


def plot(results, n_layers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua, gray = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
    layers = np.arange(len(results))
    # Only the all-dimensions variant: removing massive dimensions changes nothing visible.
    get = lambda key: [r["all"][key] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ax = axes[0]
    ax.plot(layers, get("span_energy"), color=blue, lw=2, label="translations")
    ax.plot(layers, get("mismatched_span_energy"), color=orange, lw=2, label="mismatched (control)")
    ax.plot(layers, get("random_energy"), color=gray, lw=2, ls="--", label="random subspace")
    ax.set_ylabel("fraction")
    ax.set_title("Residual energy inside span{u^L}")
    ax = axes[1]
    ax.plot(layers, get("pc_var"), color=blue, lw=2, label="top-7 PC variance, translations")
    ax.plot(layers, get("mismatched_pc_var"), color=orange, lw=2, label="top-7 PC variance, mismatched")
    ax.plot(layers, get("pc_overlap"), color=aqua, lw=2, label="PC overlap with span{u}")
    ax.set_title("Top-(K-1) principal components (fit half)")
    ticks = [0, 4, 8, 12, 16, 20, 24, 28, n_layers]
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(ticks, ["emb" if t == 0 else f"L{t}" for t in ticks])
        ax.set_xlabel(f"layer (L{n_layers} is after the final RMSNorm)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=ROOT / "data/hidden_states/llama-3.1-8b-instruct.npz")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    z = np.load(args.states)
    statements = {s.id: s for s in load_statements(max_per_benchmark=None)}
    meta = [statements[i] for i in z["ids"]]
    half = np.array([s.half for s in meta])
    bench = np.array([s.benchmark for s in meta])
    states = z["states"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{states.shape} from {args.states.name} ({z['model']}, {z['dtype'] if 'dtype' in z else '?'}), "
          f"fit {int((half == 'fit').sum())} / test {int((half == 'test').sum())} statements, device {device}")

    results = run(states, half, bench, device)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"states": str(args.states.relative_to(ROOT)), "model": str(z["model"]),
               "langs": z["langs"].tolist(), "layers": results}
    (args.out / "t1.json").write_text(json.dumps(summary, indent=1) + "\n")
    plot(results, states.shape[2] - 1, args.out / "t1.png")
    print(f"wrote {args.out / 't1.json'} and t1.png")


if __name__ == "__main__":
    main()
