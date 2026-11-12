"""Instruction vs language steering on MGSM (zero-shot, Llama-3.1-8B-Instruct), 10 target languages.

The question is the only user message; greedy decoding, at most 2048 new tokens. Arms, for target T:
  english                   the English question (the reference; target language English)
  native                    the question in T
  instruction               the English question + "\n\nAnswer in <T>."
  steer                     the English question, + alpha * v_(en->T)(l)
  instruction_steer         the instruction prompt, + alpha * v_(en->T)(l)
  steer_prompt, instruction_steer_prompt   the same, but steering the prompt only
v_(en->T)(l) = mu_T(l) - mu_en(l), the mean output of block l over the tokens (BOS excluded) of the 997
FLORES-200 dev sentences in T and in English. All-token steering adds the vector at every position from the
start of the user message onward and at every generated token; prompt-only steering adds it from the start
of the user message to the end of the prompt only.

Split: the 250 MGSM items sorted by crc32(id); the first 50 are dev, the other 200 test, the same items in
every language. Stages (each skips work whose output already exists):
  vectors   the steering vectors of every language at the grid layers -> results/generations/vectors.pt
  dev       english, native and instruction, and each steering arm over layer x alpha, on dev; a unit is
            one configuration over all 10 languages -> results/generations/dev/<unit>.jsonl
  test      every arm on test, the steering arms at the setting chosen on dev (analyse.py --select ->
            results/generations/chosen.json); a unit is one arm in one language -> .../test/<unit>.jsonl
Several workers can share a stage (--worker k): each claims the next free unit by creating a file under
results/generations/claims/<job id>/, so units are spread over GPUs as they free up. A unit's output does
not depend on which worker ran it: its prompts are batched the same way every time (sorted by length, at
most MAX_BATCH per batch). In a batch, a row leaves the batch and the KV cache once it has finished.

Usage: see README.md (one job runs every stage).
"""
import argparse
import io
import json
import os
import sys
import tarfile
import zlib
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src/icl"))
import model as icl_model  # noqa: E402  (registers the "sdpa_gqa_decode" attention)
from tasks import MGSM  # noqa: E402

GEN = HERE / "results/generations"
FLORES = ROOT / "data/raw_data/flores/flores200_dataset.tar.gz"
FLORES_CODE = {"en": "eng_Latn", "bn": "ben_Beng", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
               "ja": "jpn_Jpan", "ru": "rus_Cyrl", "sw": "swh_Latn", "te": "tel_Telu", "th": "tha_Thai",
               "zh": "zho_Hans"}
NAMES = {"bn": "Bengali", "de": "German", "es": "Spanish", "fr": "French", "ja": "Japanese", "ru": "Russian",
         "sw": "Swahili", "te": "Telugu", "th": "Thai", "zh": "Chinese"}
LANGS = tuple(NAMES)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW, MAX_BATCH = 2048, 128
N_DEV = 50
LAYERS, ALPHAS = (8, 12, 16, 20, 24), (0.25, 0.5, 1, 2, 3)
STEER_ARMS = ("steer", "instruction_steer", "steer_prompt", "instruction_steer_prompt")


def split():
    """{"dev": [row], "test": [row]}: the MGSM rows sorted by crc32(id), the first N_DEV are dev."""
    rows = sorted(MGSM().rows, key=lambda r: (zlib.crc32(r["id"].encode()), r["id"]))
    return {"dev": rows[:N_DEV], "test": rows[N_DEV:]}


def prompt(arm, row, lang):
    q_en = row["text"]["en"]["question"]
    if arm == "native":
        return row["text"][lang]["question"]
    if arm.startswith("instruction"):
        return f"{q_en}\n\nAnswer in {NAMES[lang]}."
    return q_en  # english, steer, steer_prompt


def flores(lang):
    with tarfile.open(FLORES) as t:
        f = t.extractfile(f"./flores200_dataset/dev/{FLORES_CODE[lang]}.dev")
        return [l.rstrip("\n") for l in io.TextIOWrapper(f, encoding="utf-8")]


class Model:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.pad_token, self.tok.padding_side = self.tok.eos_token, "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        self.model.set_attn_implementation("sdpa_gqa_decode")
        self.blocks = self.model.model.layers
        eos = self.model.generation_config.eos_token_id
        self.eos = torch.tensor(eos if isinstance(eos, list) else [eos], device="cuda")

    @torch.inference_mode()
    def mean_states(self, texts, bs=50):
        """[n_layers + 1, d]: mean over tokens (BOS excluded) of each layer's output, then over texts."""
        acc = torch.zeros(len(self.blocks) + 1, self.model.config.hidden_size, device="cuda")
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
            pos = (b.attention_mask.cumsum(1) - 1).clamp(min=0)
            hs = self.model(**b, position_ids=pos, output_hidden_states=True).hidden_states
            m = b.attention_mask.clone()
            m[torch.arange(m.shape[0]), m.argmax(1)] = 0  # first real token is BOS
            m = m.unsqueeze(-1).float()
            for l, h in enumerate(hs):
                acc[l] += ((h.float() * m).sum(1) / m.sum(1)).sum(0)
        return acc / len(texts)

    @torch.inference_mode()
    def generate(self, texts, layer=None, vecs=None, prompt_only=False):
        """Greedy chat responses [(text, n_tokens, capped)]. With a layer, row i adds vecs[i] to the output of
        that block (1-indexed) from the start of its user message onward, and at every generated token unless
        prompt_only."""
        tok = self.tok
        ps = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
              for t in texts]
        ids = [tok(p, add_special_tokens=False).input_ids for p in ps]
        pre = [len(tok(p[:p.index(t)], add_special_tokens=False).input_ids) for p, t in zip(ps, texts)]
        order = sorted(range(len(texts)), key=lambda i: (len(ids[i]), i))
        out = [None] * len(texts)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for s in range(0, len(order), MAX_BATCH):
                b = order[s:s + MAX_BATCH]
                v = torch.stack([vecs[i] for i in b]).to("cuda", torch.bfloat16)[:, None] if layer else None
                for i, r in zip(b, self._batch([ids[i] for i in b], [pre[i] for i in b], layer, v, prompt_only)):
                    out[i] = r
        return out

    def _batch(self, ids, pre, layer, vec, prompt_only):
        n = max(map(len, ids))
        x = torch.tensor([[self.tok.pad_token_id] * (n - len(r)) + r for r in ids], device="cuda")
        mask = torch.tensor([[0] * (n - len(r)) + [1] * len(r) for r in ids], device="cuda")
        start = torch.tensor([n - len(r) + p for r, p in zip(ids, pre)], device="cuda")[:, None]
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        st = {"vec": vec}  # the steering vectors of the rows still generating

        def hook(mod, args, o):
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] > 1:  # prefill
                h = h + (torch.arange(h.shape[1], device="cuda")[None] >= start)[..., None].to(h.dtype) * st["vec"]
            elif not prompt_only:  # decoding step
                h = h + st["vec"]
            return (h, *o[1:]) if isinstance(o, tuple) else h

        handle = self.blocks[layer - 1].register_forward_hook(hook) if layer else None
        try:
            o = self.model(input_ids=x, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
            cache, logits = o.past_key_values, o.logits[:, -1]
            B = len(ids)
            active = torch.arange(B, device="cuda")
            tokens = torch.full((B, MAX_NEW), self.tok.pad_token_id, dtype=torch.long, device="cuda")
            n_tok = torch.zeros(B, dtype=torch.long, device="cuda")
            done = torch.zeros(B, dtype=torch.bool, device="cuda")
            for step in range(MAX_NEW):
                nxt = logits.argmax(-1)
                tokens[active, step] = nxt
                fin = torch.isin(nxt, self.eos)
                n_tok[active] += (~fin).long()
                done[active[fin]] = True
                if fin.all() or step == MAX_NEW - 1:
                    break
                if fin.any():  # finished rows leave the batch and the KV cache
                    keep = (~fin).nonzero().squeeze(1)
                    cache.batch_select_indices(keep)
                    active, nxt, mask, pos = active[keep], nxt[keep], mask[keep], pos[keep]
                    if st["vec"] is not None:
                        st["vec"] = st["vec"][keep]
                mask = torch.cat([mask, mask.new_ones(len(mask), 1)], 1)
                pos = pos[:, -1:] + 1
                o = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                               past_key_values=cache, use_cache=True)
                cache, logits = o.past_key_values, o.logits[:, -1]
        finally:
            if handle:
                handle.remove()
        res = []
        for row, k, d in zip(tokens.tolist(), n_tok.tolist(), done.tolist()):
            res.append((self.tok.decode(row[:k], skip_special_tokens=True).strip(), k, not d))
        return res


def units(stage):
    """{unit name: (arm, layer, alpha, [(row, lang)])} of a stage."""
    sp = split()
    if stage == "dev":
        items = lambda: [(r, L) for L in LANGS for r in sp["dev"]]
        u = {"english": ("english", None, None, [(r, "en") for r in sp["dev"]]),
             "native": ("native", None, None, items()), "instruction": ("instruction", None, None, items())}
        for arm in STEER_ARMS:
            for l in LAYERS:
                for a in ALPHAS:
                    u[f"{arm}_L{l}_a{a}"] = (arm, l, a, items())
        return u
    chosen = json.loads((GEN / "chosen.json").read_text())
    u = {"english": ("english", None, None, [(r, "en") for r in sp["test"]])}
    for L in LANGS:
        for arm in ("native", "instruction") + STEER_ARMS:
            l, a = (chosen[arm]["layer"], chosen[arm]["alpha"]) if arm in STEER_ARMS else (None, None)
            u[f"{arm}_{L}"] = (arm, l, a, [(r, L) for r in sp["test"]])
    return u


def run_unit(M, vecs, arm, layer, alpha, items):
    texts = [prompt(arm, r, L) for r, L in items]
    po = arm.endswith("_prompt")
    v = [alpha * vecs[L][layer] for _, L in items] if layer else None
    res = M.generate(texts, layer, v, po)
    task = MGSM()
    return [dict(id=r["id"], lang=L, arm=arm, layer=layer, alpha=alpha, prompt_only=po, answer=r["answer"],
                 pred=task.parse(s), input=t, response=s, n_tokens=k, capped=c)
            for (r, L), t, (s, k, c) in zip(items, texts, res)]


def claim(stage, name):
    d = GEN / "claims" / os.environ.get("SLURM_JOB_ID", "local") / stage
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / name, os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("vectors", "dev", "test"))
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    M = Model()
    if args.stage == "vectors":
        if not (GEN / "vectors.pt").exists():
            mu = {L: M.mean_states(flores(L)) for L in ("en",) + LANGS}
            torch.save({L: {l: (mu[L][l] - mu["en"][l]).cpu() for l in LAYERS} for L in LANGS}, GEN / "vectors.pt")
        return
    vecs = torch.load(GEN / "vectors.pt")
    out = GEN / args.stage
    out.mkdir(exist_ok=True)
    for name, (arm, layer, alpha, items) in units(args.stage).items():
        if (out / f"{name}.jsonl").exists() or not claim(args.stage, name):
            continue
        rows = run_unit(M, vecs, arm, layer, alpha, items)
        tmp = out / f".{name}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.rename(out / f"{name}.jsonl")
        print(f"{args.stage} {name}: acc {sum(r['pred'] is not None and abs(r['pred'] - r['answer']) < 1e-6 for r in rows) / len(rows):.2f}"
              f" capped {sum(r['capped'] for r in rows)} | {rows[0]['response'][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
