"""Where is the language of the first answer token decided? A logit lens on the last prompt position, for the
instruction and instruction + prompt-only steering, on the items and pairs of the forced-start experiment
(prefill.py).

Logit lens: the output of block l (1-indexed, after the steering vector where it is added) at the last prompt
position, passed through the model's final norm and unembedding, read as a next-token distribution P_l. At the
last block this is the model's own prediction (checked against its logits). For each item and pair three
candidate first tokens are scored at every layer:
  t_T  the first token of the native arm's test response (the question in T), the k = 1 start T of prefill.py
  t_S  the first token of the source arm's test response (the question in S), the k = 1 start S
  t_E  the first token of the native English response
recorded as log P_l(t_T), log P_l(t_S), log P_l(t_E) and d_l = log P_l(t_T) - log P_l(t_S), with the top-1 token
of every layer. Steering uses the setting chosen on dev (chosen.json), added at its layer from the start of the
user message to the end of the prompt, as in generation.

Summary (analyse): for each arm and group of pairs, the mean of each measure per layer, over pairs and items, with a
95% bootstrap interval over items (2000 resamples). Groups: the held-out pairs, the failing ones (instruction
success < 0.10 on test, as in prefill.py) and the rest. Items where t_T = t_S (e.g. an answer starting with a
number in both) are left out, since the two tokens cannot be told apart; log P_l(t_E) is averaged over the pairs
without English only. Also: the top tokens of each steering vector, W_U (g * v) with g the final norm's weights,
and how often the last layer's top-1 token is the first token of the prefill k = 0 run's continuation,
re-tokenised (a check).

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/logit_lens.py run       # GPU
       XL_MODEL=<model> python src/experiments/mgsm_crosslingual/logit_lens.py analyse   # -> results/<model>/logit_lens.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate import MODELS, Model, chat  # noqa: E402
from generate_pairs import GEN, KEY, LANGS, MODEL, RES, prompt, vector  # noqa: E402
from prefill import ARMS, FAIL, HELD, OUT as PREFILL, SEED, items, starts  # noqa: E402

OUT = GEN / "logit_lens"
BATCH = 64
TOP = 20  # top tokens listed per steering vector


@torch.inference_mode()
def lens(M, texts, layer=None, vecs=None):
    """Per prompt: [n_blocks, vocab] log-probabilities of the logit lens at the last prompt position, and the
    model's own last-position log-probabilities. With a layer, prompt i adds vecs[i] to the output of that block
    from the start of its user message on (prompt-only steering)."""
    tok, model = M.tok, M.model
    ps = chat(tok, texts)
    ids = [tok(p, add_special_tokens=False).input_ids for p in ps]
    pre = [len(tok(p[:p.index(t)], add_special_tokens=False).input_ids) for p, t in zip(ps, texts)]
    for s in range(0, len(ids), BATCH):
        b = range(s, min(s + BATCH, len(ids)))
        n = max(len(ids[i]) for i in b)
        x = torch.tensor([[tok.pad_token_id] * (n - len(ids[i])) + ids[i] for i in b], device="cuda")
        mask = (torch.arange(n, device="cuda")[None] >= torch.tensor([n - len(ids[i]) for i in b],
                                                                    device="cuda")[:, None]).long()
        start = torch.tensor([n - len(ids[i]) + pre[i] for i in b], device="cuda")[:, None]
        v = torch.stack([vecs[i] for i in b]).to("cuda", torch.bfloat16)[:, None] if layer else None
        last = []

        def hook(l):
            def f(mod, args, o):
                h = o[0] if isinstance(o, tuple) else o
                if l == layer:
                    on = torch.arange(n, device="cuda")[None] >= start
                    h = h + on[..., None].to(h.dtype) * v
                last.append(h[:, -1])
                return (h, *o[1:]) if isinstance(o, tuple) else h
            return f

        handles = [blk.register_forward_hook(hook(l)) for l, blk in enumerate(M.blocks, 1)]
        try:
            o = model(input_ids=x, attention_mask=mask, position_ids=(mask.cumsum(1) - 1).clamp(min=0),
                      logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        logp = model.lm_head(model.model.norm(torch.stack(last))).float().log_softmax(-1)  # [blocks, B, vocab]
        own = o.logits[:, -1].float().log_softmax(-1)
        for j in range(len(b)):
            yield logp[:, j], own[j]


def run():
    M = Model(MODELS[MODEL]["hf"])
    mu = torch.load(GEN / "vectors.pt")
    chosen = json.loads((GEN / "chosen.json").read_text())
    OUT.mkdir(exist_ok=True)
    rows = items()
    first = lambda st, s, t: {i: x[0] for i, x in starts(M.tok, st, s, t).items()}  # noqa: E731
    eng = first("T", None, "en")
    todo = [(r, s, t) for s, t in HELD for r in rows]
    ref = {(s, t): (first("T", s, t), first("S", s, t)) for s, t in HELD}
    for arm in ARMS:
        if (OUT / f"{arm}.jsonl").exists():
            continue
        layer, alpha = (chosen[arm]["layer"], chosen[arm]["alpha"]) if arm in chosen else (None, None)
        vs = [vector(arm, mu, layer, alpha, r, s, t)[0] for r, s, t in todo] if layer else None
        texts = [prompt(arm, r, s, t) for r, s, t in todo]
        out, gap = [], 0.0
        for (r, s, t), x, (logp, own) in zip(todo, texts, lens(M, texts, layer, vs)):
            gap = max(gap, (logp[-1] - own).abs().max().item())
            tt, ts, te = ref[s, t][0][r["id"]], ref[s, t][1][r["id"]], eng[r["id"]]
            out.append(dict(id=r["id"], src=s, tgt=t, arm=arm, layer=layer, alpha=alpha, input=x,
                            t_T=M.tok.decode([tt]), t_S=M.tok.decode([ts]), t_E=M.tok.decode([te]), same=tt == ts,
                            logp_T=logp[:, tt].tolist(), logp_S=logp[:, ts].tolist(), logp_E=logp[:, te].tolist(),
                            top1_ids=logp.argmax(-1).tolist(),
                            top1=[M.tok.decode([i]) for i in logp.argmax(-1).tolist()]))
        print(f"{arm}: {len(out)} prompts; last layer vs the model's logits: max |diff| {gap:.2e}", flush=True)
        tmp = OUT / f".{arm}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        tmp.rename(OUT / f"{arm}.jsonl")
    # the top tokens of each steering vector: W_U (g * v), the final norm's scale without its normalisation
    W, g = M.model.lm_head.weight, M.model.model.norm.weight
    c = chosen["instruction_steer_prompt"]
    top = {}
    for s, t in HELD:
        v = (c["alpha"] * (mu[t][c["layer"]] - mu[s][c["layer"]])).to("cuda", W.dtype)
        top[KEY[s, t]] = [M.tok.decode([i]) for i in (W @ (g * v)).float().topk(TOP).indices.tolist()]
    (OUT / "vector_tokens.json").write_text(json.dumps(top, ensure_ascii=False, indent=1) + "\n")


def stat(rows, f, ids, boot):
    """Per layer: [mean, 2.5%, 97.5%] of f(row) (a list over layers), and the number of rows; the mean over pairs
    per item, then over items, with the bootstrap over items (boot: resamples of positions in ids)."""
    per = {}
    for r in rows:
        per.setdefault(r["id"], []).append(f(r))
    its = [i for i in ids if i in per]
    m = torch.tensor([[sum(c) / len(c) for c in zip(*per[i])] for i in its])  # [items, layers]
    pos = {i: k for k, i in enumerate(its)}
    ms = torch.stack([m[torch.tensor([pos[ids[j]] for j in b if ids[j] in pos])].mean(0) for b in boot])
    lo, hi = ms.quantile(0.025, 0), ms.quantile(0.975, 0)
    return [[m.mean(0)[l].item(), lo[l].item(), hi[l].item()] for l in range(m.shape[1])], len(rows)


def groups():
    """{group: set of pairs}: the held-out pairs, the failing ones and the rest."""
    summary = json.loads((RES / "summary.json").read_text())["test"]
    failing = {p for p in HELD if summary[KEY[p]]["instruction"]["success"] < FAIL}
    return {"held_out": set(HELD), "failing": failing, "rest": set(HELD) - failing}


def bootstrap(ids):
    rng = random.Random(SEED)
    return [[rng.randrange(len(ids)) for _ in ids] for _ in range(2000)]


def analyse():
    gs = groups()
    failing = gs["failing"]
    ids = [r["id"] for r in items()]
    boot = bootstrap(ids)
    data = {arm: [json.loads(l) for l in open(OUT / f"{arm}.jsonl")] for arm in ARMS}
    n_layers = len(data[ARMS[0]][0]["logp_T"])
    measures = {"logp_T": lambda r: r["logp_T"], "logp_S": lambda r: r["logp_S"], "logp_E": lambda r: r["logp_E"],
                "d": lambda r: [a - b for a, b in zip(r["logp_T"], r["logp_S"])]}

    out = dict(n_items=len(ids), layers=list(range(1, n_layers + 1)), n_pairs={g: len(p) for g, p in gs.items()},
               failing=sorted(KEY[p] for p in failing), arms={})
    for arm, rows in data.items():
        out["arms"][arm] = {}
        for g, ps in gs.items():
            keep = [r for r in rows if (r["src"], r["tgt"]) in ps and not r["same"]]
            res = {}
            for m, f in measures.items():
                sub = [r for r in keep if "en" not in (r["src"], r["tgt"])] if m == "logp_E" else keep
                res[m], res[f"n_{m}"] = stat(sub, f, ids, boot)
            res["n_same_excluded"] = sum(1 for r in rows if (r["src"], r["tgt"]) in ps and r["same"])
            out["arms"][arm][g] = res
    # check: the last layer's top-1 token against the first token the prefill k = 0 run generated
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[MODEL]["hf"])
    agree = {}
    for arm in ARMS:
        gen = {}
        for s in LANGS:
            f = PREFILL / f"{arm}_T_k0_{s}.jsonl"
            for r in map(json.loads, open(f)):
                gen[r["id"], r["src"], r["tgt"]] = r["continuation"]
        hits = [tok(gen[r["id"], r["src"], r["tgt"]], add_special_tokens=False).input_ids[:1] == r["top1_ids"][-1:]
                for r in data[arm]]
        agree[arm] = sum(hits) / len(hits)
    out["last_top1_matches_generation"] = agree
    (RES / "logit_lens.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"{len(ids)} items; pairs {out['n_pairs']}; last-layer top-1 = generated first token: {agree}")
    for g in gs:
        print(f"== {g}: per layer, mean d = log P(t_T) - log P(t_S) | log P(t_T) | log P(t_S) | log P(t_E)")
        for arm in ARMS:
            a = out["arms"][arm][g]
            print(f"-- {arm} (n = {a['n_d']}, {a['n_same_excluded']} with t_T = t_S left out)")
            for l in range(n_layers):
                if (l + 1) % 2 == 0 or l == 0:
                    print(f"  {l + 1:>2} {a['d'][l][0]:7.2f} [{a['d'][l][1]:6.2f}, {a['d'][l][2]:6.2f}]"
                          f" | {a['logp_T'][l][0]:7.2f} | {a['logp_S'][l][0]:7.2f} | {a['logp_E'][l][0]:7.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "analyse"))
    {"run": run, "analyse": analyse}[ap.parse_args().stage]()
