"""KL divergence to the native question and top-1 agreement, by position in the native response (kl.py output).

For each arm and position bucket (response token 1, 2-10, 11-50, 51 onward): the per-token KL(P_native || P_arm)
and top-1 agreement, averaged over all tokens of the bucket in all 110 pairs (token-weighted). 95% CIs from a
bootstrap over the 200 test items (2000 resamples, seed 0, the same items for every pair and arm), as in
agreement.py. Also per source language S and per target language T (no CIs; for inspection).

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/analyse_kl.py   # -> results/<model>/kl.json (after kl.sh)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate_pairs import GEN, LANGS, RES  # noqa: E402
from kl import ARMS  # noqa: E402

BUCKETS = ((0, 1), (1, 10), (10, 50), (50, None))
NAMES = ("1", "2-10", "11-50", "51+")
N_BOOT, SEED = 2000, 0


def main():
    # sums[arm, T, item, S, bucket, (kl, top1, n)]; S indexes LANGS (zero where S == T)
    ids, sums = None, None
    for ti, t in enumerate(LANGS):
        d = torch.load(GEN / "kl" / f"{t}.pt")
        ids = ids or sorted(d)
        assert sorted(d) == ids, t
        if sums is None:
            sums = np.zeros((len(ARMS), len(LANGS), len(ids), len(LANGS), len(BUCKETS), 3))
        for ii, i in enumerate(ids):
            r = d[i]
            for j, (a, s) in enumerate(r["rows"][1:]):
                for b, (lo, hi) in enumerate(BUCKETS):
                    k, tp = r["kl"][j, lo:hi], r["top1"][j, lo:hi]
                    sums[ARMS.index(a), ti, ii, LANGS.index(s), b] = (k.sum().item(), tp.sum().item(), len(k))
    n = len(ids)
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([np.arange(n)[None], rng.integers(0, n, (N_BOOT, n))])  # row 0: the data itself
    per_item = sums.sum((1, 3))                      # [arm, item, bucket, 3], summed over pairs
    boot = per_item[:, idx].sum(2)                   # [arm, 1 + N_BOOT, bucket, 3]
    kl, top1 = boot[..., 0] / boot[..., 2], boot[..., 1] / boot[..., 2]

    def summ(x):  # [1 + N_BOOT] -> estimate and CI
        lo, hi = np.percentile(x[1:], [2.5, 97.5])
        return dict(mean=float(x[0]), ci=[float(lo), float(hi)])

    out = {"n_items": n, "n_pairs": len(LANGS) * (len(LANGS) - 1), "n_boot": N_BOOT, "seed": SEED,
           "buckets": NAMES, "tokens": {nm: int(per_item[0, :, b, 2].sum()) for b, nm in enumerate(NAMES)},
           "kl": {a: {nm: summ(kl[ai, :, b]) for b, nm in enumerate(NAMES)} for ai, a in enumerate(ARMS)},
           "top1": {a: {nm: summ(top1[ai, :, b]) for b, nm in enumerate(NAMES)} for ai, a in enumerate(ARMS)}}
    for side, axis in (("by_source", 1), ("by_target", 3)):  # sum out the other language and the items
        s = sums.sum((axis, 2))                            # [arm, language, bucket, 3]
        out[side] = {L: {a: {"kl": [float(x) for x in s[ai, li, :, 0] / s[ai, li, :, 2]],
                             "top1": [float(x) for x in s[ai, li, :, 1] / s[ai, li, :, 2]]}
                         for ai, a in enumerate(ARMS)} for li, L in enumerate(LANGS)}

    print("KL to native (nats) / top-1 agreement, token-weighted over all pairs, by response position")
    print(f"{'':26}" + "".join(f"{nm:>16}" for nm in NAMES))
    for a in ARMS:
        print(f"{a:26}" + "".join(f"{out['kl'][a][nm]['mean']:>9.2f} / {out['top1'][a][nm]['mean']:.2f}"
                                  for nm in NAMES))
    (RES / "kl.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
