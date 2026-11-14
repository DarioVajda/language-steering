"""Instruction vs language steering on the cross-lingual Language Confusion Benchmark, one target language.

The model is set by the environment variable LCB_MODEL, a key of analyse.MODELS (llama-3.1-8b:
Llama-3.1-8B-Instruct; qwen3-8b, qwen3-32b: Qwen3-8B, Qwen3-32B, thinking off); its outputs go to
results/<LCB_MODEL>/.

Conditions (the benchmark prompt is the only user message; greedy, at most 100 new tokens as in the benchmark):
  instruction        the prompt as is (English request + "Write in <T>." or similar)
  steer              the prompt with its language instruction removed, + alpha * v_(en->T)(l)
  instruction_steer  the prompt as is, + alpha * v_(en->T)(l)
v_(en->T)(l) = mu_T(l) - mu_en(l), the mean output of block l over the tokens of the 997 FLORES-200 dev
sentences in T and in English, without the first (the attention sink: Llama's BOS, Qwen's first word). The
vector is added at every position from the start of the user message onward and at every generated token. With --prompt-only, it is added to the prompt only
(from the start of the user message to the end of the prompt), so generated tokens are not steered; the
alpha grid then extends to 6, since the vector is applied once.

One run does everything for one language, in a single job:
  1. dev: instruction, and both steering conditions over the grid layer x alpha;
  2. per steering condition, the dev setting chosen by analyse.select;
  3. test: instruction and both steering conditions at their chosen settings.
Writes results/<model>/generations/<T>_{dev,test}.jsonl; an existing <T>_dev.jsonl is reused.
With --universal, step 1 must have been run for every language: test is run instead at the setting shared
by all languages (analyse.select_universal), with its own instruction run, into <T>_test_universal.jsonl.
--prompt-only runs write the same files with the suffix _prompt (<T>_dev_prompt.jsonl, ...). Each configuration is generated as one call over all of
its prompts in length-sorted batches with a fixed token budget, so a run is deterministic.

The --pilot flag instead generates the instruction condition for all 14 languages on dev
(results/<model>/generations/pilot_dev.jsonl), the run used to choose the languages.

Usage: LCB_MODEL=<model> python src/experiments/language_confusion/generate.py --lang ja
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

import analyse
import data

GEN = analyse.GEN
FLORES = data.ROOT / "data/raw_data/flores/flores200_dataset.tar.gz"
FLORES_CODE = {"en": "eng_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans", "ru": "rus_Cyrl", "hi": "hin_Deva",
               "tr": "tur_Latn", "ar": "arb_Arab"}
LANGS = ("ja", "zh", "ru", "hi", "tr", "ar")
MAX_NEW = 100
MAX_BATCH, TOKEN_BUDGET = 64, 96_000  # prompts per batch, padded prompt tokens per batch
LAYERS, ALPHAS = analyse.MODELS[analyse.MODEL]["layers"], (0.25, 0.5, 1, 2, 3)
ALPHAS_PROMPT = ALPHAS + (4, 6)


def flores(lang):
    with tarfile.open(FLORES) as t:
        f = t.extractfile(f"./flores200_dataset/dev/{FLORES_CODE[lang]}.dev")
        return [l.rstrip("\n") for l in io.TextIOWrapper(f, encoding="utf-8")]


class Model:
    def __init__(self):
        name = analyse.MODELS[analyse.MODEL]["hf"]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.pad_token, self.tok.padding_side = self.tok.eos_token, "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
        self.blocks = self.model.model.layers

    @torch.no_grad()
    def mean_states(self, texts, bs=50):
        """[n_layers + 1, d]: mean over tokens of each layer's output, then over texts. The first token (the
        attention sink: Llama's BOS; Qwen3 adds no BOS, so the first word) is excluded."""
        acc = torch.zeros(len(self.blocks) + 1, self.model.config.hidden_size, device="cuda")
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
            hs = self.model(**b, output_hidden_states=True).hidden_states
            m = b.attention_mask.clone()
            m[torch.arange(m.shape[0]), m.argmax(1)] = 0  # the first real token (the sink)
            m = m.unsqueeze(-1).float()
            for l, h in enumerate(hs):
                acc[l] += ((h.float() * m).sum(1) / m.sum(1)).sum(0)
        return acc / len(texts)

    @torch.no_grad()
    def generate(self, texts, steer=None, prompt_only=False):
        """Greedy chat responses. steer = {layer: vector} is added to the output of each listed block
        (1-indexed) at every position from the start of the user message onward and at every generated token,
        or with prompt_only up to the end of the prompt only. Qwen3's thinking is off (enable_thinking=False; Llama's
        template ignores it)."""
        tok = self.tok
        ps = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for t in texts]
        ids = [tok(p, add_special_tokens=False).input_ids for p in ps]
        pre = [len(tok(p.split(t)[0], add_special_tokens=False).input_ids) for p, t in zip(ps, texts)]
        order = sorted(range(len(texts)), key=lambda i: (len(ids[i]), i))
        out = [None] * len(texts)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and j - i < MAX_BATCH and (j - i + 1) * len(ids[order[j]]) <= TOKEN_BUDGET:
                j += 1
            batch = order[i:j]
            n = max(len(ids[k]) for k in batch)
            x = torch.tensor([[tok.pad_token_id] * (n - len(ids[k])) + ids[k] for k in batch], device="cuda")
            m = torch.tensor([[0] * (n - len(ids[k])) + [1] * len(ids[k]) for k in batch], device="cuda")
            start = torch.tensor([n - len(ids[k]) + pre[k] for k in batch], device="cuda").unsqueeze(1)
            handles = []
            for layer, vec in (steer or {}).items():
                def hook(mod, args, o, vec=vec, start=start):
                    h = o[0] if isinstance(o, tuple) else o
                    if h.shape[1] > 1:  # prefill
                        w = (torch.arange(h.shape[1], device=h.device).unsqueeze(0) >= start).unsqueeze(-1).to(h.dtype)
                    elif prompt_only:  # decoding step, not steered
                        return o
                    else:  # decoding step
                        w = torch.ones_like(h[..., :1])
                    h = h + w * vec.to(h.dtype)
                    return (h, *o[1:]) if isinstance(o, tuple) else h
                handles.append(self.blocks[layer - 1].register_forward_hook(hook))
            try:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    g = self.model.generate(input_ids=x, attention_mask=m, max_new_tokens=MAX_NEW, do_sample=False,
                                            temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
            finally:
                for h_ in handles:
                    h_.remove()
            for k, row in zip(batch, g):
                out[k] = tok.decode(row[n:], skip_special_tokens=True)
            i = j
        return out


def run(M, items, cond, steer_vecs=None, layer=None, alpha=None, prompt_only=False):
    texts = [data.strip_instruction(r["prompt"], r["lang"]) if cond == "steer" else r["prompt"] for r in items]
    steer = {layer: alpha * steer_vecs[layer]} if layer else None
    resps = M.generate(texts, steer, prompt_only)
    rows = [dict(r, cond=cond, layer=layer, alpha=alpha, prompt_only=prompt_only, input=t, response=s)
            for r, t, s in zip(items, texts, resps)]
    print(f"{items[0]['lang']} {items[0]['split']} {cond} layer={layer} alpha={alpha} prompt_only={prompt_only} | "
          f"LPR {analyse.rates([analyse.judge(r['response'], r['lang']) for r in rows], items[0]['lang'])['lpr']:.2f} | "
          f"{resps[0][:80]!r}", flush=True)
    return rows


def write(rows, path):
    """Written to a temporary file, then renamed, so that a file that exists is complete (run.sh waits for them)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lang", choices=LANGS)
    g.add_argument("--pilot", action="store_true")
    ap.add_argument("--universal", action="store_true")
    ap.add_argument("--prompt-only", action="store_true")
    args = ap.parse_args()
    po, sfx = args.prompt_only, "_prompt" if args.prompt_only else ""
    GEN.mkdir(parents=True, exist_ok=True)
    M = Model()

    if args.pilot:
        items = [dict(r, split="dev") for r in data.load("dev")]
        write(run(M, items, "instruction"), GEN / "pilot_dev.jsonl")
        return

    L = args.lang
    mu_en, mu_t = M.mean_states(flores("en")), M.mean_states(flores(L))
    vecs = {l: mu_t[l] - mu_en[l] for l in LAYERS}
    dev = [dict(r, split="dev") for r in data.load("dev", [L])]
    test = [dict(r, split="test") for r in data.load("test", [L])]

    if args.universal:
        chosen = analyse.select_universal(
            {l: analyse.dev_scores([json.loads(x) for x in open(GEN / f"{l}_dev{sfx}.jsonl")]) for l in LANGS})
        print("universal setting:", chosen, flush=True)
        rows = run(M, test, "instruction")
        for cond, (layer, a) in chosen.items():
            rows += run(M, test, cond, vecs, layer, a, po)
        write(rows, GEN / f"{L}_test_universal{sfx}.jsonl")
        return

    if (GEN / f"{L}_dev{sfx}.jsonl").exists():  # a finished dev grid is reused (e.g. after a failed test stage)
        rows = [json.loads(l) for l in open(GEN / f"{L}_dev{sfx}.jsonl")]
    else:
        rows = run(M, dev, "instruction")
        for cond in ("steer", "instruction_steer"):
            for layer in LAYERS:
                for a in ALPHAS_PROMPT if po else ALPHAS:
                    rows += run(M, dev, cond, vecs, layer, a, po)
        write(rows, GEN / f"{L}_dev{sfx}.jsonl")

    chosen = analyse.select(analyse.dev_scores(rows))
    print("chosen on dev:", chosen, flush=True)
    rows = run(M, test, "instruction")
    for cond, (layer, a) in chosen.items():
        rows += run(M, test, cond, vecs, layer, a, po)
    write(rows, GEN / f"{L}_test{sfx}.jsonl")


if __name__ == "__main__":
    main()
