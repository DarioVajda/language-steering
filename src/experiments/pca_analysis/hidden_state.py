"""Mean-pooled hidden states of the EXP1 statements, per layer and language.

For every statement and language, the text is run through the model on its own (raw
text, no chat template) and the hidden states of every layer are averaged over its
tokens. The BOS token is fed to the model but excluded from the mean: it acts as an
attention sink with massive activations (Sun et al., 2024) that would dominate it.

Layer 0 is the embedding output and layer i the output of block i. In Hugging Face
Llama models the last entry is taken after the final RMSNorm.

Output (.npz):
  states    float32 [n_statements, n_langs, n_layers + 1, d]
  n_tokens  int     [n_statements, n_langs]   tokens averaged (BOS excluded)
  ids, langs, model, dtype (the model dtype; float32 by default)

Usage:
    python src/experiments/pca_analysis/hidden_state.py \
        --model meta-llama/Llama-3.1-8B-Instruct --out data/hidden_states/llama-3.1-8b-instruct.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from data import LANGS, MAX_PER_BENCHMARK, load_statements

ROOT = Path(__file__).resolve().parents[3]


def load_model(name, device=None, dtype=torch.float32):
    """Load a Hugging Face model (without its LM head) and its tokenizer."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def mean_states(texts, model, tokenizer):
    """Mean hidden state of each text at every layer: float32 [len(texts), n_layers + 1, d]."""
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    mask = enc.attention_mask.clone()
    if tokenizer.bos_token_id is not None:
        mask[enc.input_ids == tokenizer.bos_token_id] = 0
    mask = mask.unsqueeze(-1).float()
    hidden = model(**enc, output_hidden_states=True).hidden_states
    means = [(h.float() * mask).sum(1) / mask.sum(1) for h in hidden]
    return torch.stack(means, 1).cpu()


def process_dataset(statements, model, tokenizer, langs=LANGS, batch_size=16, out_path=None):
    """Mean states of every (statement, language) pair; optionally saved to `out_path` (.npz).

    Returns (states, n_tokens) with shapes [n_statements, n_langs, n_layers + 1, d] and
    [n_statements, n_langs]. Texts are batched in order of length to reduce padding."""
    pairs = [(i, j) for i in range(len(statements)) for j in range(len(langs))]
    texts = [statements[i].text[langs[j]] for i, j in pairs]
    lengths = [len(ids) for ids in tokenizer(texts).input_ids]
    order = np.argsort(lengths, kind="stable")[::-1]  # longest first: fail fast on memory

    n_layers, d = model.config.num_hidden_layers + 1, model.config.hidden_size
    states = np.zeros((len(statements), len(langs), n_layers, d), dtype=np.float32)
    n_tokens = np.zeros((len(statements), len(langs)), dtype=np.int64)
    bos = int(tokenizer.bos_token_id is not None)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        means = mean_states([texts[k] for k in batch], model, tokenizer).numpy()
        for k, m in zip(batch, means):
            i, j = pairs[k]
            states[i, j], n_tokens[i, j] = m, lengths[k] - bos
        print(f"\r{min(start + batch_size, len(order))}/{len(order)} texts", end="", flush=True)
    print()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, states=states, n_tokens=n_tokens, dtype=str(model.dtype), ids=np.array([s.id for s in statements]),
                 langs=np.array(langs), model=model.config._name_or_path)
        print(f"saved {states.shape} -> {out_path}")
    return states, n_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-per-benchmark", type=int, default=MAX_PER_BENCHMARK)
    parser.add_argument("--limit", type=int, help="only the first N statements (for quick tests)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="model dtype (bf16 states differ from fp32 by ~0.5%% of their norm)")
    args = parser.parse_args()

    statements = load_statements(max_per_benchmark=args.max_per_benchmark)[:args.limit]
    model, tokenizer = load_model(args.model, dtype=getattr(torch, args.dtype))
    process_dataset(statements, model, tokenizer, batch_size=args.batch_size, out_path=args.out)


if __name__ == "__main__":
    main()
