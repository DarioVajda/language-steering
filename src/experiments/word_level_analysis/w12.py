"""EXP1 / W1 and W2: how well do the global language directions describe single words?

Source S = English, target T = each other language. For an aligned word pair c (align.py
"pairs") with states h_S, h_T at layer l, the local difference is d = h_T - h_S. Fitted on
the fit half (all benchmarks pooled):
  mu_L  mean state of all non-BOS tokens of language L        mu = mean over L of mu_L
  u_L   (mu_L - mu) / |mu_L - mu|
  v     mu_T - mu_S   (the constant-shift direction)          w = u_T - u_S (README method)

W1, per pair: cos(d, v), cos(d, w), and span = fraction of |d|^2 inside span{u_L}.
  Controls: "mismatched" (T word of a random pair of the same English word type, from
  another statement) and random unit directions (mean |cos|).
W2, per pair: oracle scales alpha* = d.v / |v|^2 and beta* = d.w / |w|^2, and the
  README prediction gamma_S = (h_S - mu).u_S (from the source state alone).
  For a scale p (alpha* or beta*) and a prediction q, the error of steering by q along z
  is |d - q z|^2 = |d|^2 - 2 q p |z|^2 + q^2 |z|^2, so relative to a constant a = mean p
  (fit half):
    headroom  = |z|^2 sum (p - a)^2 / sum |d|^2   share of |d|^2 a per-word scale can remove
                                                   beyond the best constant shift
    captured  = 1 - sum (p - q)^2 / sum (p - a)^2  share of that headroom the prediction gets
                                                   (1 = oracle, 0 = no better than constant)
  with q = gamma_S ("readme"), k gamma_S ("scaled", k fitted), c + k gamma_S ("affine")
  and k' gamma_S shuffled within word type ("shuffled", the null).
  Null: all of this with h_T replaced by the mismatched word ("*_mismatched").
  Also: Pearson/Spearman corr(p, gamma_S), within word type, partial on |h_S - mu|,
  OLS slope of p on gamma_S (README: 1 for beta*), corr(gamma_S, gamma_T), corr with
  token position, CV of alpha*.
Metrics on the test half; 95% CIs from a bootstrap over test statements.

Outputs: results/w12[_variant].json, w1[_variant].png, w2[_variant].png.
Usage (GPU job, via slurm/submit.sh):
    slurm/submit.sh -J w12 -- python src/experiments/word_level_analysis/w12.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEED = 0
N_BOOT = 200
N_RANDOM = 5
TYPES = ("content", "entity", "number", "math", "punct")


def build_pairs(records, langs, row_start, clean, method, clean_only):
    """Per target language: rows of S and T words, English type, statement index, half."""
    out = {}
    for b, L in enumerate(langs):
        if L == "en":
            continue
        cols = {k: [] for k in ("s", "t", "type", "stmt", "fit")}
        for a, r in enumerate(records):
            for i, j in r["pairs"][method][L]:
                s, t = row_start[a, 0] + i, row_start[a, b] + j
                if clean_only and not (clean[s] and clean[t]):
                    continue
                for k, x in zip(cols, (s, t, TYPES.index(r["types"][i]), a, r["half"] == "fit")):
                    cols[k].append(x)
        out[L] = {k: np.array(x) for k, x in cols.items()}
    return out


def mismatch(p, rng):
    """For each pair, the T row of a random pair with the same English type from another statement."""
    t = p["t"].copy()
    for ty in np.unique(p["type"]):
        idx = np.flatnonzero(p["type"] == ty)
        pick = rng.choice(idx, len(idx))
        for _ in range(20):
            same = p["stmt"][pick] == p["stmt"][idx]
            if not same.any():
                break
            pick[same] = rng.choice(idx, same.sum())
        t[idx] = p["t"][pick]
    return t


def boot_weights(stmt, rng):
    """[N_BOOT, n] multiplicity of each pair when test statements are resampled."""
    uniq, inv = np.unique(stmt, return_inverse=True)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=N_BOOT)
    return counts[:, inv].astype(np.float64)


def wmean(W, x):
    return (W @ x) / W.sum(1)


def wpearson(W, x, y):
    mx, my = wmean(W, x), wmean(W, y)
    cov = wmean(W, x * y) - mx * my
    return cov / np.sqrt((wmean(W, x * x) - mx ** 2) * (wmean(W, y * y) - my ** 2))


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")


def spearman(x, y):
    return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def partial(x, y, z):
    """corr(x, y) after regressing both on [1, z]."""
    A = np.stack([np.ones_like(z), z], 1)
    res = lambda t: t - A @ np.linalg.lstsq(A, t, rcond=None)[0]
    return pearson(res(x), res(y))


def ci(W, f):
    lo, hi = np.nanpercentile(f(W), [2.5, 97.5])
    return [float(lo), float(hi)]


def scale_metrics(p, g, g_shuf, z2, dd, fit, W, types):
    """W2 metrics for the oracle scale p along a direction with |z|^2 = z2."""
    pf, gf, gsf = p[fit], g[fit], g_shuf[fit]
    a = pf.mean()
    k = (gf * pf).sum() / (gf * gf).sum()
    ks = (gsf * pf).sum() / (gsf * gsf).sum()
    k_aff, c_aff = np.polyfit(gf, pf, 1)
    t = ~fit
    pt, gt, dt = p[t], g[t], dd[t]
    preds = {"readme": gt, "scaled": k * gt, "affine": c_aff + k_aff * gt, "shuffled": ks * g_shuf[t]}
    base = (pt - a) ** 2
    out = {
        "mean": float(pt.mean()), "cv": float(pt.std() / abs(pt.mean())), "constant_fit": float(a), "k_fit": float(k),
        "headroom": float(z2 * base.sum() / dt.sum()),
        "headroom_ci": ci(W, lambda W: z2 * (W @ base) / (W @ dt)),
        "pearson": pearson(pt, gt), "pearson_ci": ci(W, lambda W: wpearson(W, pt, gt)),
        "spearman": spearman(pt, gt),
        "slope": float(np.polyfit(gt, pt, 1)[0]),
        "pearson_by_type": {TYPES[ty]: pearson(pt[types == ty], gt[types == ty]) for ty in np.unique(types)},
    }
    for name, q in preds.items():
        err = (pt - q) ** 2
        out[f"captured_{name}"] = float(1 - err.sum() / base.sum())
        out[f"captured_{name}_ci"] = ci(W, lambda W: 1 - (W @ err) / (W @ base))
    return out


def lang_metrics(x, p, t_mis, mu, mu_L, u, basis, b, rand, rng):
    dev, f64 = x.device, torch.float64
    rows = lambda r: x[torch.from_numpy(r).to(dev)].to(f64)
    # Pairs with identical states (same tokens after an identical prefix, e.g. a sentence-
    # initial name; at layer 0 any shared token) have d = 0 and are left out.
    same = ((rows(p["t"]) - rows(p["s"])).norm(dim=1) == 0).cpu().numpy()
    p = {k: a[~same] for k, a in p.items()}
    t_mis = t_mis[~same]
    hs, ht, hm = rows(p["s"]), rows(p["t"]), rows(t_mis)
    d, dm = ht - hs, hm - hs
    v, w = mu_L[b] - mu_L[0], u[b] - u[0]
    np_ = lambda t: t.cpu().numpy()
    dd, dv, dw = np_((d * d).sum(1)), np_(d @ v), np_(d @ w)
    v2, w2 = float(v @ v), float(w @ w)
    norm = np.sqrt(dd)
    cos_v, cos_w = dv / (norm * np.sqrt(v2)), dw / (norm * np.sqrt(w2))
    span = np_((d @ basis.T).square().sum(1)) / dd
    ddm = (dm * dm).sum(1)
    mis_cos_v = np_(dm @ v / (ddm.sqrt() * v.norm()))
    mis_cos_w = np_(dm @ w / (ddm.sqrt() * w.norm()))
    mis_span = np_((dm @ basis.T).square().sum(1) / ddm)
    rand_cos = np_((d @ rand.T).abs() / (d.norm(dim=1, keepdim=True)))
    g = np_((hs - mu) @ u[0])
    g_t = np_((ht - mu) @ u[b])
    hs_norm = np_((hs - mu).norm(dim=1))

    fit, t = p["fit"], ~p["fit"]
    types = p["type"]
    g_shuf = g.copy()
    for ty in np.unique(types):  # shuffle within (half, type)
        for h in (fit, t):
            idx = np.flatnonzero((types == ty) & h)
            g_shuf[idx] = g[rng.permutation(idx)]
    W = boot_weights(p["stmt"][t], rng)
    tt = types[t]
    alpha, beta = dv / v2, dw / w2
    # Null for W2: the same metrics with h_T replaced by the mismatched word. Any
    # correlation left there comes from h_S alone (gamma_S also enters d = h_T - h_S).
    ddm_np = np_(ddm)
    beta_mis, alpha_mis = np_(dm @ w) / w2, np_(dm @ v) / v2
    return {
        "n_test": int(t.sum()), "n_identical_dropped": int(same.sum()),
        "w1": {
            "cos_v": float(cos_v[t].mean()), "cos_v_ci": ci(W, lambda W: wmean(W, cos_v[t])),
            "cos_w": float(cos_w[t].mean()), "cos_w_ci": ci(W, lambda W: wmean(W, cos_w[t])),
            "cos_w_quartiles": np.percentile(cos_w[t], [25, 50, 75]).tolist(),
            "cos_w_positive": float((cos_w[t] > 0).mean()),
            "span": float(span[t].mean()),
            "mismatched_cos_v": float(mis_cos_v[t].mean()), "mismatched_cos_w": float(mis_cos_w[t].mean()),
            "mismatched_cos_w_ci": ci(W, lambda W: wmean(W, mis_cos_w[t])),
            "mismatched_span": float(mis_span[t].mean()),
            "random_abs_cos": float(rand_cos[t].mean()),
            "by_type": {TYPES[ty]: {"n": int((tt == ty).sum()), "cos_v": float(cos_v[t][tt == ty].mean()),
                                    "cos_w": float(cos_w[t][tt == ty].mean()), "span": float(span[t][tt == ty].mean())}
                        for ty in np.unique(tt)},
        },
        "w2": {
            "alpha": scale_metrics(alpha, g, g_shuf, v2, dd, fit, W, tt),
            "beta": scale_metrics(beta, g, g_shuf, w2, dd, fit, W, tt),
            "beta_partial_norm": partial(beta[t], g[t], hs_norm[t]),
            "alpha_partial_norm": partial(alpha[t], g[t], hs_norm[t]),
            "beta_mismatched": scale_metrics(beta_mis, g, g_shuf, w2, ddm_np, fit, W, tt),
            "alpha_mismatched": scale_metrics(alpha_mis, g, g_shuf, v2, ddm_np, fit, W, tt),
            "gamma_s_t_pearson": pearson(g[t], g_t[t]),
            "gamma_s_mean": float(g[t].mean()), "gamma_t_mean": float(g_t[t].mean()),
            "alpha_position_pearson": pearson(alpha[t], p["position"][t]),
            "beta_position_pearson": pearson(beta[t], p["position"][t]),
        },
    }


def run(states, pairs, mis, sums, counts, langs, device):
    gen = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    K, d = len(langs), states.shape[-1]
    n_fit = counts[:, :, 0].sum(1)                                   # fit half, all benchmarks
    results = []
    for layer in range(states.shape[0]):
        x = torch.from_numpy(np.asarray(states[layer])).to(device)    # float16, cast per use
        mu_L = torch.from_numpy(sums[:, :, 0, layer].sum(1) / n_fit[:, None]).to(device)
        mu = mu_L.mean(0)
        u = (mu_L - mu) / (mu_L - mu).norm(dim=1, keepdim=True)
        basis = torch.linalg.svd(u, full_matrices=False).Vh[:K - 1]
        rand = torch.randn(N_RANDOM, d, generator=gen, dtype=torch.float64).to(device)
        rand = rand / rand.norm(dim=1, keepdim=True)
        entry = {"layer": layer, "langs": {}}
        for b, L in enumerate(langs):
            if L != "en":
                entry["langs"][L] = lang_metrics(x, pairs[L], mis[L], mu, mu_L, u, basis, b, rand, rng)
        results.append(entry)
        m = lambda f: np.mean([f(e) for e in entry["langs"].values()])
        print(f"layer {layer:>2}: cos(d,w) {m(lambda e: e['w1']['cos_w']):.3f} (mismatched "
              f"{m(lambda e: e['w1']['mismatched_cos_w']):.3f}), cos(d,v) {m(lambda e: e['w1']['cos_v']):.3f}, "
              f"span {m(lambda e: e['w1']['span']):.3f} | beta: corr {m(lambda e: e['w2']['beta']['pearson']):.3f}, "
              f"slope {m(lambda e: e['w2']['beta']['slope']):.2f}, headroom {m(lambda e: e['w2']['beta']['headroom']):.3f}, "
              f"captured {m(lambda e: e['w2']['beta']['captured_scaled']):.3f} "
              f"(shuffled {m(lambda e: e['w2']['beta']['captured_shuffled']):.3f})", flush=True)
        del x
    return results


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GRAY = "#8a8a85"


def plot(results, path_w1, path_w2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = list(results[0]["langs"])
    layers = np.arange(len(results))
    n = len(results) - 1
    ticks = [0, 4, 8, 12, 16, 20, 24, 28, n]
    get = lambda L, f: [f(r["langs"][L]) for r in results]
    macro = lambda f: [np.mean([f(e) for e in r["langs"].values()]) for r in results]

    def panels(path, specs, ncols):
        nrows = -(-len(specs) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
        for ax, (title, f, null, hline) in zip(axes.flat, specs):
            for L, c in zip(langs, COLORS):
                ax.plot(layers, get(L, f), color=c, lw=1.6, label=L)
            if null is not None:
                ax.plot(layers, macro(null[1]), color=GRAY, lw=1.6, ls="--", label=null[0])
            if hline is not None:
                ax.axhline(hline, color=GRAY, lw=0.8)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(ticks, ["emb" if t == 0 else f"L{t}" for t in ticks])
        for ax in axes.flat[len(specs):]:
            ax.axis("off")
        axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    panels(path_w1, [
        ("W1: mean cos(d, w),  w = u_T - u_S", lambda e: e["w1"]["cos_w"],
         ("mismatched (mean over T)", lambda e: e["w1"]["mismatched_cos_w"]), 0),
        ("W1: mean cos(d, v),  v = mu_T - mu_S", lambda e: e["w1"]["cos_v"],
         ("mismatched (mean over T)", lambda e: e["w1"]["mismatched_cos_v"]), 0),
        ("W1: fraction of |d|^2 in span{u_L}", lambda e: e["w1"]["span"],
         ("mismatched (mean over T)", lambda e: e["w1"]["mismatched_span"]), None),
    ], 3)
    panels(path_w2, [
        ("W2: corr(beta*, gamma_S)", lambda e: e["w2"]["beta"]["pearson"],
         ("mismatched h_T (mean over T)", lambda e: e["w2"]["beta_mismatched"]["pearson"]), 0),
        ("W2: OLS slope of beta* on gamma_S (README: 1)", lambda e: e["w2"]["beta"]["slope"],
         ("mismatched h_T (mean over T)", lambda e: e["w2"]["beta_mismatched"]["slope"]), 1),
        ("W2: headroom along w (share of |d|^2)", lambda e: e["w2"]["beta"]["headroom"], None, None),
        ("W2: captured headroom, k*gamma_S along w", lambda e: e["w2"]["beta"]["captured_scaled"],
         ("gamma shuffled within type (mean over T)", lambda e: e["w2"]["beta"]["captured_shuffled"]), 0),
        ("W2: corr(alpha*, gamma_S)  (along v)", lambda e: e["w2"]["alpha"]["pearson"],
         ("mismatched h_T (mean over T)", lambda e: e["w2"]["alpha_mismatched"]["pearson"]), 0),
        ("W2: captured headroom, k*gamma_S along v", lambda e: e["w2"]["alpha"]["captured_scaled"],
         ("gamma shuffled within type (mean over T)", lambda e: e["w2"]["alpha"]["captured_shuffled"]), 0),
    ], 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=HERE / "states" / "llama-3.1-8b-instruct")
    parser.add_argument("--alignments", type=Path, default=HERE / "results" / "alignments_simalign.json")
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    parser.add_argument("--method", choices=["inter", "itermax", "mwmf"], default="inter")
    parser.add_argument("--clean-only", action="store_true", help="only pairs whose two words own their tokens")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    al = json.loads(args.alignments.read_text())
    langs, records = al["langs"], al["records"]
    words = np.load(args.states / "words.npz")
    tok = np.load(args.states / "token_sums.npz")
    assert list(tok["langs"]) == langs
    pairs = build_pairs(records, langs, words["row_start"], words["clean"], args.method, args.clean_only)
    rng = np.random.default_rng(SEED)
    for p in pairs.values():
        p["position"] = words["position"][p["s"]]
    mis = {L: mismatch(p, rng) for L, p in pairs.items()}
    states = np.load(args.states / f"{args.pool}.npy", mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{states.shape} {args.pool} states, method {args.method}, clean_only {args.clean_only}; test pairs: "
          + ", ".join(f"{L} {int((~p['fit']).sum())}" for L, p in pairs.items()), flush=True)

    results = run(states, pairs, mis, tok["sums"], tok["counts"], langs, device)
    variant = "".join([f"_{args.pool}" if args.pool != "last" else "", f"_{args.method}" if args.method != "inter" else "",
                       "_clean" if args.clean_only else ""])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"w12{variant}.json").write_text(json.dumps(
        {"states": str(args.states.relative_to(HERE)), "pool": args.pool, "method": args.method,
         "clean_only": args.clean_only, "layers": results}, indent=1) + "\n")
    plot(results, args.out / f"w1{variant}.png", args.out / f"w2{variant}.png")
    print(f"wrote {args.out / f'w12{variant}.json'}, w1{variant}.png, w2{variant}.png")


if __name__ == "__main__":
    main()
