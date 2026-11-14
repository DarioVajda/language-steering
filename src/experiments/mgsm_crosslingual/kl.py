"""Teacher-forced comparison with the native question: on the native response, how close is each arm's
next-token distribution to the native question's?

For a target T and a test item, the text is the native arm's saved greedy response (re-tokenised, followed by
the end-of-turn token unless it hit the token limit). Each prompt below is followed by that same text, and the model's
next-token distribution is taken at every response position:
  native                       the question in T (the reference distribution P)
  source, steer_prompt         the question in S, without and with prompt-only steering   } for each of the
  instruction, instruction_steer_prompt   the instruction prompt, without and with it     } 10 sources S != T
The steering arms use the settings chosen on dev (generations/chosen.json); as in generation, prompt-only
steering adds alpha * v_(S->T)(l) from the start of the user message to the end of the prompt, and not on the
response. All 110 pairs, the 200 test items.

Saved per token (nothing is aggregated here; the metrics are computed from these files):
  kl     KL(P_native || P_arm)      rkl    KL(P_arm || P_native)
  top1   argmax P_arm == argmax P_native
  lp     log P(response token) under each row, the native row (row 0) included
-> results/<model>/generations/kl/<T>.pt: {item id: {"n", "capped", "rows": [(arm, S)], "kl", "rkl", "top1", "lp"}}

A (T, item) group is the 41 sequences that share a response. Groups are sorted by length and packed into
batches of at most TOKENS padded tokens, so a batch's output does not depend on which worker ran it. Workers
claim batches as in generate_pairs.py; `merge` collects them into one file per T.

Usage: see kl.sh (one `run` process per GPU, then `merge`; helper jobs only run).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate import MODELS, Model, chat, split  # noqa: E402
from generate_pairs import GEN, LANGS, MODEL, claim, prompt  # noqa: E402

ARMS = ("source", "steer_prompt", "instruction", "instruction_steer_prompt")
OUT = GEN / "kl"
TOKENS = 128 * 1024  # padded tokens per batch
CHUNK = 128          # response positions per vocabulary-sized computation


def groups(tok):
    """One dict per (T, test item): the shared response ids and, per row, its prompt ids, the steering span
    [start, len(prompt)) and its steering vector spec (layer, alpha, S, T) or None."""
    chosen = json.loads((GEN / "chosen.json").read_text())
    items = {r["id"]: r for r in split()["test"]}
    eot = tok.convert_tokens_to_ids(MODELS[MODEL]["eot"])
    out = []
    for t in LANGS:
        for nat in map(json.loads, open(GEN / "test" / f"native_{t}.jsonl")):
            r = items[nat["id"]]
            target = tok(nat["response"], add_special_tokens=False).input_ids + ([] if nat["capped"] else [eot])
            rows, ids, start, steer = [("native", None)], [], [], [None]
            rows += [(a, s) for s in LANGS if s != t for a in ARMS]
            steer += [(chosen[a]["layer"], chosen[a]["alpha"], s, t) if a in chosen else None for a, s in rows[1:]]
            for a, s in rows:
                x = prompt(a, r, s, t)
                p, = chat(tok, [x])
                ids.append(tok(p, add_special_tokens=False).input_ids)
                start.append(len(tok(p[:p.index(x)], add_special_tokens=False).input_ids))
            out.append(dict(t=t, id=nat["id"], capped=nat["capped"], target=target, rows=rows, ids=ids,
                            start=start, steer=steer, width=max(map(len, ids)) + len(target)))
    return out


def batches(gs):
    """Lists of group indices: groups sorted by padded length, packed while rows * width <= TOKENS."""
    order = sorted(range(len(gs)), key=lambda i: (gs[i]["width"], i))
    out, cur, rows = [], [], 0
    for i in order:
        n = len(gs[i]["rows"])
        if cur and (rows + n) * gs[i]["width"] > TOKENS:  # sorted, so this group is the widest so far
            out.append(cur)
            cur, rows = [], 0
        cur.append(i)
        rows += n
    out.append(cur)
    return out


@torch.inference_mode()
def run_batch(M, mu, gs):
    tok, d = M.tok, M.model.config.hidden_size
    seqs = [ids + g["target"] for g in gs for ids in g["ids"]]
    plen = [len(ids) for g in gs for ids in g["ids"]]
    start = [s for g in gs for s in g["start"]]
    steer = [s for g in gs for s in g["steer"]]
    B, L = len(seqs), max(map(len, seqs))
    x = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in seqs], device="cuda")
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda")
    pos = torch.arange(L, device="cuda")[None].expand(B, L)
    span = ((pos >= torch.tensor(start, device="cuda")[:, None])
            & (pos < torch.tensor(plen, device="cuda")[:, None]))[..., None]  # prompt-only steering positions
    handles = []

    def adder(v):
        def hook(mod, args, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h + span.to(h.dtype) * v
            return (h, *o[1:]) if isinstance(o, tuple) else h
        return hook

    for layer in sorted({st[0] for st in steer if st}):
        v = torch.zeros(B, d)
        for i, st in enumerate(steer):
            if st and st[0] == layer:
                _, a, s, t = st
                v[i] = a * (mu[t][layer] - mu[s][layer])
        handles.append(M.blocks[layer - 1].register_forward_hook(adder(v.to("cuda", torch.bfloat16)[:, None])))
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            h = M.model.model(input_ids=x, attention_mask=mask, position_ids=pos).last_hidden_state
    finally:
        for hd in handles:
            hd.remove()

    out, o = [], 0
    for g in gs:
        n, R = len(g["rows"]), len(g["target"])
        idx = torch.tensor(plen[o:o + n], device="cuda")[:, None] - 1 + torch.arange(R, device="cuda")[None]
        H = h[o:o + n].gather(1, idx[..., None].expand(-1, -1, d))  # [n, R, d]: states that predict the response
        tgt = torch.tensor(g["target"], device="cuda")
        res = {k: [] for k in ("kl", "rkl", "top1", "lp")}
        for j in range(0, R, CHUNK):
            lp = torch.log_softmax(M.model.lm_head(H[:, j:j + CHUNK]).float(), -1)  # [n, c, V]
            ln, la = lp[0], lp[1:]
            res["kl"].append((ln.exp() * (ln - la)).sum(-1))
            res["rkl"].append((la.exp() * (la - ln)).sum(-1))
            res["top1"].append(la.argmax(-1) == ln.argmax(-1))
            res["lp"].append(lp.gather(-1, tgt[None, j:j + CHUNK, None].expand(n, -1, 1))[..., 0])
            del lp, ln, la
        out.append(dict(t=g["t"], id=g["id"], n=R, capped=g["capped"], rows=g["rows"],
                        **{k: torch.cat(v, 1).cpu() for k, v in res.items()}))
        o += n
    return out


def run():
    M = Model(MODELS[MODEL]["hf"])
    mu = torch.load(GEN / "vectors.pt")
    gs = groups(M.tok)
    bs = batches(gs)
    (OUT / "batches").mkdir(parents=True, exist_ok=True)
    print(f"{len(gs)} groups in {len(bs)} batches", flush=True)
    for i, b in enumerate(bs):
        f = OUT / "batches" / f"{i:05d}.pt"
        if f.exists() or not claim("kl", f"{i:05d}"):
            continue
        t0 = time.time()
        res = run_batch(M, mu, [gs[k] for k in b])
        torch.save(res, OUT / "batches" / f".{i:05d}.pt.tmp")
        os.rename(OUT / "batches" / f".{i:05d}.pt.tmp", f)
        kl = torch.cat([r["kl"].mean(1) for r in res]).mean().item()
        print(f"batch {i}: {len(b)} groups, width {gs[b[-1]]['width']}, {time.time() - t0:.1f}s, mean kl {kl:.3f}",
              flush=True)


def merge():
    """Waits until every batch is written (helper jobs may still be running some), then merges."""
    n = len(batches(groups(AutoTokenizer.from_pretrained(MODELS[MODEL]["hf"]))))
    while len(files := sorted((OUT / "batches").glob("[0-9]*.pt"))) < n:
        time.sleep(30)
    per = {t: {} for t in LANGS}
    for f in files:
        for r in torch.load(f):
            per[r.pop("t")][r.pop("id")] = r
    n_test = len(split()["test"])
    assert all(len(v) == n_test for v in per.values()), {t: len(v) for t, v in per.items()}
    for t, v in per.items():
        torch.save(v, OUT / f".{t}.pt.tmp")
        os.rename(OUT / f".{t}.pt.tmp", OUT / f"{t}.pt")
    for f in files:
        f.unlink()
    (OUT / "batches").rmdir()
    print(f"merged {len(files)} batches into {OUT}/<T>.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "merge"))
    {"run": run, "merge": merge}[ap.parse_args().stage]()
