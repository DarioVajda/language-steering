"""Precompute per-language mean states for language steering, per benchmark, on the compute split.

Following Kirtane & Huang (2026, Eqs. 1-4), each compute item i yields one sample per language:
item i followed by k-1 = 5 items drawn uniformly with replacement from the compute split, all
formatted as demonstrations (task.format_demo) and joined into one text. The sample's items are
the same in every language. Its state at each layer is the mean over its tokens (BOS excluded,
no instruction or chat template). The mean over samples is stored per language and layer, and

    v_(S->T)(t) = mu_T(t) - mu_S(t)

which for S = en is the paper's v(t) = mean_i (h_i^T(t) - h_i^en(t)). Samples containing an item
whose translation into L is flagged are dropped for L only, so mu_L may average over slightly
fewer samples than mu_en.

Output: data/language_vectors/<model>/<benchmark>.npz with
  mu        float32 [n_langs, n_layers + 1, d]   (layer indexing as in model.py)
  n         int     [n_langs]                    samples averaged
  langs, model, k, seed

Usage (GPU job, see CLAUDE.md):
    slurm/submit.sh -J vectors -- python src/icl/vectors.py --benchmarks xnli mgsm msvamp
"""

import argparse
import zlib
from pathlib import Path

import numpy as np
import torch

from tasks import K_SHOT, ROOT, TASKS

OUT = ROOT / "data" / "language_vectors"
SEED = 0


def slug(model):
    return model.split("/")[-1].lower()


def path(model, benchmark):
    return OUT / slug(model) / f"{benchmark}.npz"


def item_rng(seed, *keys):
    """A generator determined by the seed and the given string keys (e.g. an item id)."""
    return np.random.default_rng([seed, *(zlib.crc32(str(k).encode()) for k in keys)])


def samples(task, k=K_SHOT, seed=SEED):
    """Compute-split item tuples: item i then k-1 items drawn with replacement."""
    pool = [it for it in task.items("compute") if "en" not in it.flags]
    return [[it] + [pool[j] for j in item_rng(seed, it.id).integers(len(pool), size=k - 1)] for it in pool]


class LanguageVectors:
    def __init__(self, file):
        data = np.load(file)
        self.langs = tuple(data["langs"])
        self.mu = torch.from_numpy(data["mu"])
        self.n = dict(zip(self.langs, data["n"]))

    @classmethod
    def load(cls, model, benchmark):
        return cls(path(model, benchmark))

    def direction(self, source, target, layer):
        """v_(S->T) at `layer`: float32 [d]."""
        return self.mu[self.langs.index(target), layer] - self.mu[self.langs.index(source), layer]


def compute(task, runner, k=K_SHOT, seed=SEED, batch_size=16, limit=None):
    groups = samples(task, k, seed)[:limit]
    mu, n = [], []
    for L in task.langs:
        keep = [g for g in groups if not any(L in it.flags for it in g)]
        print(f"{task.name}/{L}: {len(keep)} samples")
        states = runner.mean_states(["\n\n".join(task.format_demo(it, L) for it in g) for g in keep], batch_size)
        mu.append(states.double().mean(0).float())
        n.append(len(keep))
    return torch.stack(mu).numpy(), np.array(n)


def main():
    from model import Runner
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--benchmarks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="only the first N samples (for quick tests)")
    args = parser.parse_args()

    runner = Runner(args.model, dtype=getattr(torch, args.dtype))
    for b in args.benchmarks:
        task = TASKS[b]()
        mu, n = compute(task, runner, batch_size=args.batch_size, limit=args.limit)
        out = path(args.model, b)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, mu=mu, n=n, langs=np.array(task.langs), model=args.model, k=K_SHOT, seed=SEED)
        print(f"saved {mu.shape} -> {out}")


if __name__ == "__main__":
    main()
