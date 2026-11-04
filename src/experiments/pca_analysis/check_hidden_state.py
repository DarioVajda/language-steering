"""Sanity checks for hidden_state.py, meant to run on a GPU node (via slurm/submit.sh).

1. Padding invariance: a text's mean states are the same alone and inside a padded batch.
2. bf16 vs fp32: relative error of the bf16 mean states, per layer.
3. Throughput: seconds per 1000 texts for a few batch sizes (bf16).

Usage: python src/experiments/pca_analysis/check_hidden_state.py [--model ...] [--n 64]
"""

import argparse
import time

import numpy as np
import torch

from data import LANGS, load_statements
from hidden_state import load_model, mean_states


def per_text(texts, model, tokenizer):
    return torch.cat([mean_states([t], model, tokenizer) for t in texts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n", type=int, default=64, help="texts used in checks 1 and 2")
    args = parser.parse_args()

    statements = load_statements()
    texts = [s.text[L] for s in statements for L in LANGS]
    rng = np.random.default_rng(0)
    sample = [texts[i] for i in rng.choice(len(texts), args.n, replace=False)]
    print(f"device: {torch.cuda.get_device_name(0)}, {len(texts)} texts in the dataset")

    model, tokenizer = load_model(args.model, dtype=torch.float32)
    single = per_text(sample, model, tokenizer)
    batched = mean_states(sample, model, tokenizer)
    err = ((batched - single).norm(dim=-1) / single.norm(dim=-1)).max()
    print(f"1. padding invariance (fp32, batch of {args.n}): max relative error {err:.2e}")
    del model
    torch.cuda.empty_cache()

    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    bf16 = per_text(sample, model, tokenizer)
    rel = ((bf16 - single).norm(dim=-1) / single.norm(dim=-1)).mean(0)
    layers = list(range(0, rel.numel(), 4)) + [rel.numel() - 1]
    print("2. bf16 vs fp32, mean relative error by layer: "
          + ", ".join(f"L{i} {rel[i]:.1e}" for i in layers))

    mean_states(texts[:16], model, tokenizer)  # warm-up
    for bs in [16, 64, 128]:
        chunk = texts[:4 * bs]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(0, len(chunk), bs):
            mean_states(chunk[i:i + bs], model, tokenizer)
        torch.cuda.synchronize()
        sec = (time.perf_counter() - start) / len(chunk) * 1000
        print(f"3. batch size {bs:>3}: {sec:.1f} s / 1000 texts, peak memory "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
