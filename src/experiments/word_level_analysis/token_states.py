"""EXP1 word-level hidden states: the state of every segmented word, at every layer.

Every statement text is run through the model on its own (raw text, no chat template,
fp32), exactly as in pca_analysis/hidden_state.py. A word (from align.py) owns the tokens
whose character span overlaps it; Llama's tokens carry their leading space, which is not
counted as overlap. For each word two vectors are kept per layer:
  last  the state of its last token (the main representation, EXP1.md)
  mean  the mean over its tokens (robustness check)
All words are stored, not only the aligned ones, so other aligners can reuse the states
as long as the segmentation is unchanged.

Also stored are the sums and counts of the states of all non-BOS tokens per (language,
benchmark, half), from which the global token means mu_L (EXP1.md, Setup) are computed.

Output directory (default: states/<model> next to this script):
  last.npy, mean.npy   float16 [n_layers + 1, n_words, d]  (computed in fp32; one layer is
                       contiguous, so np.load(..., mmap_mode="r")[l] reads only layer l)
  words.npz            per-word metadata, rows in alignment order (records, LANGS, words):
                         row_start [n_records, n_langs]  row of word 0 of each text
                         n_tokens  tokens owned by the word
                         position  token position of its last token (BOS = 0)
                         last_id   token id of its last token
                         clean     the word's tokens lie inside it (no token shared with a
                                   neighbouring word, e.g. Thai tokens that cross a
                                   PyThaiNLP boundary)
  token_sums.npz       sums float64 [n_langs, n_benchmarks, 2 (fit, test), n_layers + 1, d],
                       counts [n_langs, n_benchmarks, 2], langs, benchmarks
  meta.json            model, alignment file, shapes

Usage (GPU job, see CLAUDE.md):
    slurm/submit.sh -J token_states -t 02:00:00 -- python src/experiments/word_level_analysis/token_states.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pca_analysis"))
from data import BENCHMARKS, load_statements  # noqa: E402
from hidden_state import load_model  # noqa: E402

HALVES = ("fit", "test")


def word_tokens(words, offsets, text):
    """For each word (word, start, end): (indices of the tokens it owns, clean)."""
    out = []
    for _, ws, we in words:
        # A token's span starts at its leading space, which is not part of any word.
        own = [k for k, (s, e) in enumerate(offsets) if s < we and e > ws]
        clean = all(offsets[k][1] <= we and (offsets[k][0] >= ws or text[offsets[k][0]:ws].isspace()) for k in own)
        out.append((own, clean))
    return out


@torch.inference_mode()
def forward(texts, model, tokenizer):
    """Hidden states [B, n_layers + 1, T, d] (fp32, on device), offsets and attention mask."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping").tolist()
    enc = enc.to(model.device)
    hidden = torch.stack(model(**enc, output_hidden_states=True).hidden_states, 1).float()
    return hidden, offsets, enc.attention_mask, enc.input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--alignments", type=Path, default=HERE / "results" / "alignments_simalign.json")
    parser.add_argument("--out", type=Path, default=HERE / "states" / "llama-3.1-8b-instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    al = json.loads(args.alignments.read_text())
    langs, records = al["langs"], al["records"]
    statements = {s.id: s for s in load_statements(max_per_benchmark=None)}
    model, tokenizer = load_model(args.model, dtype=torch.float32)
    n_layers, d = model.config.num_hidden_layers + 1, model.config.hidden_size

    # One text per (record, language); its words occupy consecutive rows.
    texts, words, row_start = [], [], np.zeros((len(records), len(langs)), dtype=np.int64)
    for a, r in enumerate(records):
        for b, L in enumerate(langs):
            row_start[a, b] = sum(len(w) for w in words)
            texts.append(statements[r["id"]].text[L])
            words.append(r["words"][L])
    n_words = sum(len(w) for w in words)
    print(f"{len(texts)} texts, {n_words} words, {n_layers} layers -> "
          f"{2 * n_layers * n_words * d * 2 / 2**30:.1f} GiB", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    last = np.lib.format.open_memmap(args.out / "last.npy", "w+", np.float16, (n_layers, n_words, d))
    mean = np.lib.format.open_memmap(args.out / "mean.npy", "w+", np.float16, (n_layers, n_words, d))
    n_tok, position, last_id = (np.zeros(n_words, dtype=np.int64) for _ in range(3))
    clean = np.zeros(n_words, dtype=bool)
    sums = torch.zeros(len(langs), len(BENCHMARKS), 2, n_layers, d, dtype=torch.float64, device=model.device)
    counts = np.zeros((len(langs), len(BENCHMARKS), 2), dtype=np.int64)
    max_abs = 0.0

    lengths = [len(ids) for ids in tokenizer(texts).input_ids]
    order = np.argsort(lengths, kind="stable")[::-1]
    for start in range(0, len(order), args.batch_size):
        batch = order[start:start + args.batch_size]
        hidden, offsets, mask, ids = forward([texts[k] for k in batch], model, tokenizer)
        for t, k in enumerate(batch):
            a, b = divmod(int(k), len(langs))
            r = records[a]
            keep = mask[t].bool() & (ids[t] != tokenizer.bos_token_id)
            key = (b, BENCHMARKS.index(r["benchmark"]), HALVES.index(r["half"]))
            sums[key] += hidden[t][:, keep].double().sum(1)
            counts[key] += int(keep.sum())
            owned = word_tokens(words[k], offsets[t], texts[k])
            assert all(own for own, _ in owned), f"word without tokens in {r['id']} {langs[b]}"
            idx = torch.tensor([own[-1] for own, _ in owned], device=hidden.device)
            w_last = hidden[t][:, idx]                                            # [layers, n, d]
            w_mean = torch.stack([hidden[t][:, own].mean(1) for own, _ in owned], 1)
            max_abs = max(max_abs, float(w_last.abs().max()), float(w_mean.abs().max()))
            rows = slice(row_start[a, b], row_start[a, b] + len(owned))
            last[:, rows] = w_last.half().cpu().numpy()
            mean[:, rows] = w_mean.half().cpu().numpy()
            n_tok[rows] = [len(own) for own, _ in owned]
            position[rows] = [own[-1] for own, _ in owned]
            last_id[rows] = ids[t][idx].tolist()
            clean[rows] = [c for _, c in owned]
        print(f"\r{min(start + args.batch_size, len(order))}/{len(order)} texts", end="", flush=True)
    print()

    assert max_abs < 6e4, f"max |state| {max_abs} overflows float16"
    last.flush()
    mean.flush()
    np.savez(args.out / "words.npz", row_start=row_start, n_tokens=n_tok, position=position,
             last_id=last_id, clean=clean)
    np.savez(args.out / "token_sums.npz", sums=sums.cpu().numpy(), counts=counts,
             langs=np.array(langs), benchmarks=np.array(BENCHMARKS))
    (args.out / "meta.json").write_text(json.dumps({
        "model": args.model, "alignments": str(args.alignments.relative_to(HERE)), "dtype": "float32 -> float16",
        "shape": [n_layers, n_words, d], "langs": langs, "max_abs": max_abs}, indent=1) + "\n")
    print(f"max |state| {max_abs:.1f}; wrote {args.out}")
    for b, L in enumerate(langs):
        rows = np.concatenate([np.arange(row_start[a, b], row_start[a, b] + len(r["words"][L]))
                               for a, r in enumerate(records)])
        print(f"  {L}: {len(rows)} words, {n_tok[rows].mean():.2f} tokens/word, clean {clean[rows].mean():.1%}")


if __name__ == "__main__":
    main()
