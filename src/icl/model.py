"""Model wrapper for the ICL experiments: prompt encoding, steering hooks, forward passes and
greedy generation.

A prompt is a list of segments (instruction, demonstrations, query). Each segment is tagged with
its role and language. The segments are joined into one user turn and wrapped in the chat
template; the model's answer is the assistant turn (Kirtane & Huang, 2026, App. H). Every token is
labelled with the role and language of the segment its first character belongs to, or with the
language of a `spans` entry of that segment covering the character (e.g. a demonstration's English
chain of thought inside a demonstration in another language). Template
tokens and the separators between segments have role "template". A steering function can
therefore act on any chosen set of positions (see `Batch.mask` and steering.py).

Steering is applied to the prompt only: the hooks are active during the forward pass over the
prompt, and generated tokens attend to the steered (cached) states but are not steered themselves.

Generation speed (outputs are unchanged up to bf16 rounding, the same size as from changing the
batch): (1) decode steps use `sdpa_gqa_decode`, which attends with the grouped query heads folded
into the query axis instead of copying the KV cache to every query head (`repeat_kv`, which
transformers' sdpa does whenever there is a padding mask); (2) a row leaves the batch, and the KV
cache, once it has finished. Prompt passes (prefill, forward-mode scoring, mean states) use sdpa
unchanged.

Layer indexing: layer t (1..n_layers) is the residual stream after decoder block t, and layer 0
is the embedding output. This is `hidden_states[t]` of Hugging Face models, except for the last
layer, where `hidden_states` is taken after the final norm and the hooks here are not.
"""

import bisect
import re
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

ROLES = ("template", "instruction", "demo", "query")
SEP = "\n\n"


def sdpa_gqa_decode(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """sdpa, except that a single-token (decode) query attends to the grouped KV heads directly:
    query heads kv*g .. kv*g+g-1 of KV head kv become g query positions, so the cache is not repeated."""
    B, H, q, d = query.shape
    Hkv = key.shape[1]
    if q != 1 or H == Hkv:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kwargs)
    mask = attention_mask[:, :, :, :key.shape[2]] if attention_mask is not None else None   # [B, 1, 1, T]
    out = torch.nn.functional.scaled_dot_product_attention(query.reshape(B, Hkv, H // Hkv, d), key, value,
                                                           attn_mask=mask, scale=scaling)
    return out.reshape(B, H, 1, d).transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa_gqa_decode", sdpa_gqa_decode)
AttentionMaskInterface.register("sdpa_gqa_decode", sdpa_mask)   # the masks of "sdpa"



def cut_at_stop(text, stop):
    """Index of the first stop string that follows some number, else None. A stop string before any
    number is the model restating the question (e.g. sw "I'll translate it: \nQuestion: ..."), not a
    new item, so cutting there would drop the whole solution."""
    for m in sorted((m for s in stop for m in re.finditer(re.escape(s), text)), key=lambda m: m.start()):
        if re.search(r"\d", text[:m.start()]):
            return m.start()
    return None

@dataclass(frozen=True)
class Segment:
    role: str               # "instruction", "demo" or "query"
    text: str
    lang: str | None = None
    spans: tuple = ()       # (start, end, lang) character ranges of `text` in another language than `lang`


@dataclass(frozen=True)
class Prompt:
    segments: tuple
    target: str             # language of the query


@dataclass
class Batch:
    """A left-padded batch of encoded prompts. `role` indexes ROLES; `lang` indexes `langs`, -1 for none."""
    input_ids: torch.Tensor       # [B, T]
    attention_mask: torch.Tensor  # [B, T]
    role: torch.Tensor            # [B, T]
    lang: torch.Tensor            # [B, T]
    langs: tuple
    prompts: list

    @property
    def position_ids(self):
        return (self.attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    def mask(self, positions):
        """Bool [B, T] mask of a position set (the configurations of Kirtane & Huang, Table 10):
        "demos" all demonstration tokens, "boundary" the last demonstration token, "query" the
        test item's tokens, "prompt" every prompt token."""
        if positions == "prompt":
            return self.attention_mask.bool()
        if positions in ("demos", "boundary"):
            demo = self.role == ROLES.index("demo")
            if positions == "demos":
                return demo
            last = demo.shape[1] - 1 - demo.flip(1).int().argmax(1)
            out = torch.zeros_like(demo)
            out[torch.arange(len(out)), last] = demo.any(1)
            return out
        if positions == "query":
            return self.role == ROLES.index("query")
        raise ValueError(f"unknown positions {positions!r}")


class Runner:
    def __init__(self, name, dtype=torch.bfloat16, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
        self.model.set_attn_implementation("sdpa_gqa_decode")
        self.blocks = self.model.model.layers
        self.n_layers = len(self.blocks)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = torch.tensor(eos if isinstance(eos, list) else [eos], device=device)
        self._eos = set(self.eos_ids.tolist())
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else int(self.eos_ids[0])

    @property
    def device(self):
        return self.model.device

    # ---------------------------------------------------------------- encoding

    def _encode_one(self, prompt, langs):
        content = SEP.join(s.text for s in prompt.segments)
        full = self.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        offset = full.index(content)
        starts, ends, pos = [], [], offset
        for s in prompt.segments:
            starts.append(pos)
            ends.append(pos + len(s.text))
            pos += len(s.text) + len(SEP)
        enc = self.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        role, lang = [], []
        for a, _ in enc.offset_mapping:
            i = bisect.bisect_right(starts, a) - 1
            seg = prompt.segments[i] if i >= 0 and a < ends[i] else None
            L = seg.lang if seg else None
            if seg:
                L = next((sl for s0, s1, sl in seg.spans if s0 <= a - starts[i] < s1), L)
            role.append(ROLES.index(seg.role) if seg else 0)
            lang.append(langs.index(L) if L else -1)
        return enc.input_ids, role, lang

    def encode(self, prompts, langs):
        rows = [self._encode_one(p, langs) for p in prompts]
        T = max(len(r[0]) for r in rows)
        def pad(xs, value):
            return torch.tensor([[value] * (T - len(x)) + list(x) for x in xs], device=self.device)
        ids = pad([r[0] for r in rows], self.pad_id)
        mask = pad([[1] * len(r[0]) for r in rows], 0)
        return Batch(ids, mask, pad([r[1] for r in rows], 0), pad([r[2] for r in rows], -1), tuple(langs), list(prompts))

    def _batches(self, prompts, langs, batch_size):
        """(indices, Batch) chunks, longest prompts first (fail fast on memory, little padding)."""
        lengths = [sum(len(s.text) for s in p.segments) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            yield idx, self.encode([prompts[i] for i in idx], langs)

    # ---------------------------------------------------------------- steering

    @contextmanager
    def steered(self, steering, batch):
        """Apply `steering(layer, h, batch) -> h` to the output of each block in `steering.layers`."""
        handles = []
        for t in (steering.layers if steering is not None else ()):
            assert 1 <= t <= self.n_layers, f"cannot steer layer {t}"
            def hook(module, args, output, t=t):
                if isinstance(output, tuple):
                    return (steering(t, output[0], batch), *output[1:])
                return steering(t, output, batch)
            handles.append(self.blocks[t - 1].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ---------------------------------------------------------------- forward passes

    @torch.inference_mode()
    def next_token_logprobs(self, prompts, langs, steering=None, token_ids=None, batch_size=16):
        """Log-probabilities of the first answer token: float32 [len(prompts), V], or
        [len(prompts), len(token_ids)] restricted to `token_ids` (still normalised over V)."""
        out = [None] * len(prompts)
        for idx, batch in self._batches(prompts, langs, batch_size):
            with self.steered(steering, batch):
                logits = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                                    position_ids=batch.position_ids, logits_to_keep=1).logits[:, -1]
            lp = logits.float().log_softmax(-1)
            if token_ids is not None:
                lp = lp[:, token_ids]
            for i, row in zip(idx, lp.cpu()):
                out[i] = row
        return torch.stack(out)

    @torch.inference_mode()
    def generate(self, prompts, langs, steering=None, max_new_tokens=512, stop=(), batch_size=16):
        """Greedy continuations of `prompts`, cut at EOS or the first stop string that
        follows a number (`cut_at_stop`): one dict per prompt
        with "response", "n_tokens" (tokens generated until EOS, the stop string, which is checked
        every 16 tokens, or the cap) and "capped" (True if generation stopped at `max_new_tokens`).
        Attention runs without the cuDNN backend: it keeps an execution plan in host memory for every new
        (batch size, length) shape, and decoding (growing length, finished rows dropped) creates new shapes
        at nearly every step, so host memory grew by tens of GB per hour until the job was killed.
        Flash / memory-efficient attention gives the same accuracy (bf16 noise) and is ~3x faster here."""
        out = [None] * len(prompts)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for idx, batch in self._batches(prompts, langs, batch_size):
                for i, r in zip(idx, self._generate_batch(batch, steering, max_new_tokens, stop)):
                    out[i] = r
        return out

    def _generate_batch(self, batch, steering, max_new_tokens, stop):
        mask, pos = batch.attention_mask, batch.position_ids
        with self.steered(steering, batch):
            out = self.model(input_ids=batch.input_ids, attention_mask=mask, position_ids=pos,
                             use_cache=True, logits_to_keep=1)
        B = len(mask)
        cache, logits = out.past_key_values, out.logits[:, -1]
        active = torch.arange(B, device=self.device)       # rows still generating, in the cache's order
        tokens = torch.full((B, max_new_tokens), self.pad_id, dtype=torch.long, device=self.device)
        n_tokens = torch.zeros(B, dtype=torch.long, device=self.device)
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        steps = 0
        for step in range(max_new_tokens):
            nxt = logits.argmax(-1)
            tokens[active, step] = nxt
            n_tokens[active] += 1
            steps = step + 1
            fin = torch.isin(nxt, self.eos_ids)
            if stop and step % 16 == 15:
                texts = self._decode(tokens[active, :steps])
                fin |= torch.tensor([cut_at_stop(t, stop) is not None for t in texts], device=self.device)
            done[active[fin]] = True
            if fin.all():
                break
            if fin.any():                                   # finished rows leave the batch and the cache
                keep = (~fin).nonzero().squeeze(1)
                cache.batch_select_indices(keep)
                active, nxt, mask, pos = active[keep], nxt[keep], mask[keep], pos[keep]
            mask = torch.cat([mask, mask.new_ones(len(mask), 1)], 1)
            pos = pos[:, -1:] + 1
            out = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                             past_key_values=cache, use_cache=True)
            cache, logits = out.past_key_values, out.logits[:, -1]
        texts = self._decode(tokens[:, :steps])
        texts = [t[:cut_at_stop(t, stop)] for t in texts]
        return [{"response": t.strip(), "n_tokens": n, "capped": not d}
                for t, n, d in zip(texts, n_tokens.tolist(), done.tolist())]

    def _decode(self, tokens):
        rows = []
        for row in tokens.tolist():
            cut = next((i for i, t in enumerate(row) if t in self._eos), len(row))
            rows.append(self.tokenizer.decode(row[:cut], skip_special_tokens=True))
        return rows

    @torch.inference_mode()
    def mean_states(self, texts, batch_size=16):
        """Mean residual stream of each raw text (no chat template) at every layer 0..n_layers:
        float32 [len(texts), n_layers + 1, d]. BOS is fed to the model but excluded from the mean,
        since as an attention sink it has massive activations (Sun et al., 2024)."""
        enc = self.tokenizer(texts)
        order = sorted(range(len(texts)), key=lambda i: -len(enc.input_ids[i]))
        out = torch.zeros(len(texts), self.n_layers + 1, self.model.config.hidden_size)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            T = max(len(enc.input_ids[i]) for i in idx)
            ids = torch.tensor([[self.pad_id] * (T - len(enc.input_ids[i])) + enc.input_ids[i] for i in idx], device=self.device)
            attn = torch.tensor([[0] * (T - len(enc.input_ids[i])) + [1] * len(enc.input_ids[i]) for i in idx], device=self.device)
            keep = attn.clone()
            keep[ids == self.tokenizer.bos_token_id] = 0
            keep = keep.unsqueeze(-1).float()
            states = []
            hooks = [self.model.model.embed_tokens.register_forward_hook(lambda m, a, o: states.append(o))]
            hooks += [b.register_forward_hook(lambda m, a, o: states.append(o[0] if isinstance(o, tuple) else o))
                      for b in self.blocks]
            try:
                self.model.model(input_ids=ids, attention_mask=attn, position_ids=(attn.cumsum(-1) - 1).clamp(min=0))
            finally:
                for h in hooks:
                    h.remove()
            means = torch.stack([(h.float() * keep).sum(1) / keep.sum(1) for h in states], 1)
            out[idx] = means.cpu()
            print(f"\r{min(start + batch_size, len(order))}/{len(order)} texts", end="", flush=True)
        print()
        return out
