"""Output-language control on MKQA: generate responses for every condition.

Computes en->T language vectors from the compute split, then generates greedy responses for
  plain        English question + English format instruction                  (lang "en" only)
  instruction  the plain prompt + " Answer in <T>."
  native       question and format instruction both in T
  steer        the plain prompt, with alpha * v_(en->T)(l) added at the given layers
Two stages:
  single  plain / instruction / native / single-layer grid, on the 60 test items
  multi   5-layer and all-32-layer grids, on the 60 test + 100 val items
Each stage generates all items of one configuration in a single batch, so the outputs match
the pilot runs exactly when run as in the README.

Usage: python src/experiments/output_language/generate.py --stage single --langs en fr hu
"""
import argparse
import gzip
import json
import zlib
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MKQA = ROOT / "data/raw_data/mkqa/mkqa.jsonl.gz"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CODE = {"en": "en", "fr": "fr", "zh": "zh_cn", "ru": "ru", "hu": "hu"}  # MKQA language codes
NAME = {"fr": "French", "zh": "Chinese", "ru": "Russian", "hu": "Hungarian"}
FMT = {
    "en": "Explain briefly in 1-3 sentences, then give the final answer in \\boxed{}.",
    "fr": "Explique brièvement en 1 à 3 phrases, puis donne la réponse finale dans \\boxed{}.",
    "zh": "请用1到3句话简要解释，然后将最终答案写在\\boxed{}中。",
    "ru": "Кратко объясни в 1–3 предложениях, затем дай окончательный ответ в \\boxed{}.",
    "hu": "Röviden magyarázd el 1–3 mondatban, majd add meg a végső választ \\boxed{} formában.",
}
N_COMPUTE, N_TEST, N_VAL = 300, 60, 100
MAX_NEW = 200
GRIDS = {  # stage -> list of (layers, alphas)
    "single": [((l,), (0.5, 1, 2, 3)) for l in (8, 12, 16, 20, 24)],
    "multi": [((8, 12, 16, 20, 24), (0.1, 0.2, 0.3, 0.5)), (tuple(range(1, 33)), (0.02, 0.05, 0.1, 0.25))],
}


def load_splits():
    """Answerable MKQA items with an answer in every target language, in a fixed hash order."""
    keep = []
    for line in gzip.open(MKQA, "rt"):
        r = json.loads(line)
        a = r["answers"]["en"][0]
        if a["type"] in ("unanswerable", "long_answer") or not a.get("text"):
            continue
        if not all(r["answers"][CODE[L]] and r["answers"][CODE[L]][0].get("text") for L in NAME):
            continue
        keep.append(r)
    keep.sort(key=lambda r: zlib.crc32(str(r["example_id"]).encode()))
    c, t = N_COMPUTE, N_COMPUTE + N_TEST
    return {"compute": keep[:c], "test": keep[c:t], "val": keep[t:t + N_VAL]}


def question(r, L):
    return r["query"] if L == "en" else r["queries"][CODE[L]]


class Model:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.pad_token, self.tok.padding_side = self.tok.eos_token, "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        self.blocks = self.model.model.layers

    @torch.no_grad()
    def mean_states(self, texts, bs=50):
        """[n_layers + 1, d]: mean over tokens (BOS excluded) of each layer's output, then over texts."""
        acc = torch.zeros(len(self.blocks) + 1, self.model.config.hidden_size, device="cuda")
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
            hs = self.model(**b, output_hidden_states=True).hidden_states
            m = b.attention_mask.clone()
            m[torch.arange(m.shape[0]), m.argmax(1)] = 0  # first real token is BOS
            m = m.unsqueeze(-1).float()
            for l, h in enumerate(hs):
                acc[l] += ((h.float() * m).sum(1) / m.sum(1)).sum(0)
        return acc / len(texts)

    @torch.no_grad()
    def generate(self, texts, steer=None):
        """Greedy chat responses. steer = {layer: vector} is added to the output of each listed
        block (1-indexed) at every position from the start of the user message onward, and at
        every generated token."""
        tok = self.tok
        ps = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
              for t in texts]
        pre = [len(tok(p.split(t)[0], add_special_tokens=False).input_ids) for p, t in zip(ps, texts)]
        b = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        start = ((b.attention_mask == 0).sum(1) + torch.tensor(pre, device="cuda")).unsqueeze(1)
        handles = []
        for layer, vec in (steer or {}).items():
            def hook(mod, args, out, vec=vec):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] > 1:  # prefill
                    m = (torch.arange(h.shape[1], device=h.device).unsqueeze(0) >= start).unsqueeze(-1).to(h.dtype)
                else:  # decoding step
                    m = torch.ones_like(h[..., :1])
                h = h + m * vec.to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            handles.append(self.blocks[layer - 1].register_forward_hook(hook))
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                g = self.model.generate(**b, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                                        pad_token_id=tok.eos_token_id)
        finally:
            for h_ in handles:
                h_.remove()
        return [tok.decode(x[b.input_ids.shape[1]:], skip_special_tokens=True) for x in g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=GRIDS, required=True)
    ap.add_argument("--langs", nargs="+", choices=["en", *NAME], required=True, help="'en' = the plain condition")
    ap.add_argument("--out", type=Path, default=HERE / "results/generations")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sp = load_splits()
    items = sp["test"] if args.stage == "single" else sp["test"] + sp["val"]
    split = ["test"] * N_TEST + (["val"] * N_VAL if args.stage == "multi" else [])
    en_prompts = [f"{question(r, 'en')}\n\n{FMT['en']}" for r in items]
    M = Model()
    targets = [L for L in args.langs if L != "en"]
    mu = {L: M.mean_states([question(r, L) for r in sp["compute"]]) for L in ["en", *targets]}

    for L in args.langs:
        rows = []

        def record(cond, resps, **kw):
            for r, s, resp in zip(items, split, resps):
                rows.append(dict(id=r["example_id"], split=s, lang=L, cond=cond, **kw, response=resp,
                                 answers=r["answers"][CODE[L]], answers_en=r["answers"]["en"]))
            print(args.stage, L, cond, kw, "|", resps[0][:100].replace("\n", " "), flush=True)

        if L == "en":
            if args.stage == "single":
                record("plain", M.generate(en_prompts))
        else:
            if args.stage == "single":
                record("instruction", M.generate([f"{p} Answer in {NAME[L]}." for p in en_prompts]))
                record("native", M.generate([f"{question(r, L)}\n\n{FMT[L]}" for r in items]))
            for layers, alphas in GRIDS[args.stage]:
                for a in alphas:
                    steer = {l: a * (mu[L][l] - mu["en"][l]) for l in layers}
                    record("steer", M.generate(en_prompts, steer), layers=list(layers), alpha=a)
        if rows:
            with open(args.out / f"{args.stage}_{L}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
