"""The logit lens of logit_lens.py with each language scored by a set of tokens rather than one: the top K first
tokens of the model's own answer to the same item asked in that language (the native prompt, question in L, no
instruction; its next-token distribution at the last prompt position).

For each item, pair (S, T), arm and layer l: log P_l(set_T), log P_l(set_S) and, for pairs without English,
log P_l(set_E), each the log of the summed probability of its set under the logit lens at layer l. A token in more
than one of the sets compared (T and S, and E where it is scored) is left out of all of them, so that no token
counts for two languages; how many are left out is recorded, as is the share of the native distribution that each
set covers (its K tokens before any are left out).

Besides the two arms of the forced start, the native prompt itself (question in T, no instruction) is scored the
same way, as a reference. At the last layer its T set is by construction its own top K, so only the earlier layers
compare like with like. Arms already in OUT are not rerun, and the native top K are reused if saved, so that adding
an arm leaves the earlier ones unchanged.

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/logit_lens_top5.py run       # GPU
       XL_MODEL=<model> python src/experiments/mgsm_crosslingual/logit_lens_top5.py analyse   # -> results/<model>/logit_lens_top5.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate import MODELS, Model  # noqa: E402
from generate_pairs import GEN, KEY, LANGS, MODEL, RES, prompt, vector  # noqa: E402
from logit_lens import OUT, bootstrap, groups, lens, stat  # noqa: E402
from prefill import ARMS, HELD, items  # noqa: E402

K = 5
LENS_ARMS = ARMS + ("native",)


def run():
    M = Model(MODELS[MODEL]["hf"])
    mu = torch.load(GEN / "vectors.pt")
    chosen = json.loads((GEN / "chosen.json").read_text())
    OUT.mkdir(exist_ok=True)
    rows = items()
    # the native top K per item and language
    saved = OUT / f"top{K}_native.json"
    if saved.exists():
        top = {(n["id"], n["lang"]): (n["ids"], n["mass"]) for n in json.loads(saved.read_text())}
    else:
        native = [(r, L) for L in LANGS for r in rows]
        top = {}
        for (r, L), (_, own) in zip(native, lens(M, [prompt("native", r, None, L) for r, L in native])):
            p, i = own.exp().topk(K)
            top[r["id"], L] = (i.tolist(), p.sum().item())
        saved.write_text(json.dumps(
            [dict(id=i, lang=L, tokens=[M.tok.decode([t]) for t in ts], ids=ts, mass=m) for (i, L), (ts, m) in top.items()],
            ensure_ascii=False, indent=0) + "\n")
    todo = [(r, s, t) for s, t in HELD for r in rows]
    for arm in LENS_ARMS:
        if (OUT / f"top{K}_{arm}.jsonl").exists():
            continue
        layer, alpha = (chosen[arm]["layer"], chosen[arm]["alpha"]) if arm in chosen else (None, None)
        vs = [vector(arm, mu, layer, alpha, r, s, t)[0] for r, s, t in todo] if layer else None
        texts = [prompt(arm, r, s, t) for r, s, t in todo]
        out = []
        for (r, s, t), (logp, _) in zip(todo, lens(M, texts, layer, vs)):
            scored = {"T": t, "S": s} | ({} if "en" in (s, t) else {"E": "en"})
            sets = {k: set(top[r["id"], L][0]) for k, L in scored.items()}
            shared = {x for k in sets for x in sets[k] if sum(x in v for v in sets.values()) > 1}
            row = dict(id=r["id"], src=s, tgt=t, arm=arm, n_shared=len(shared))
            for k, v in sets.items():
                keep = sorted(v - shared)
                row[f"logm_{k}"] = logp[:, keep].logsumexp(-1).tolist() if keep else None
            out.append(row)
        print(f"{arm}: {len(out)} prompts", flush=True)
        tmp = OUT / f".top{K}_{arm}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in out))
        tmp.rename(OUT / f"top{K}_{arm}.jsonl")


def analyse():
    gs = groups()
    ids = [r["id"] for r in items()]
    boot = bootstrap(ids)
    native = json.loads((OUT / f"top{K}_native.json").read_text())
    data = {arm: [json.loads(l) for l in open(OUT / f"top{K}_{arm}.jsonl")] for arm in LENS_ARMS}
    out = dict(k=K, n_items=len(ids), n_pairs={g: len(p) for g, p in gs.items()},
               native_mass={L: sum(n["mass"] for n in native if n["lang"] == L) / len(ids) for L in LANGS}, arms={})
    for arm, rows in data.items():
        out["arms"][arm] = {}
        for g, ps in gs.items():
            sub = [r for r in rows if (r["src"], r["tgt"]) in ps]
            res = dict(mean_shared=sum(r["n_shared"] for r in sub) / len(sub))
            for k in "TSE":
                have = [r for r in sub if r.get(f"logm_{k}") is not None]
                res[k], res[f"n_{k}"] = stat(have, lambda r: r[f"logm_{k}"], ids, boot)
            out["arms"][arm][g] = res
    (RES / f"logit_lens_top{K}.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"top {K}; share of the native distribution in its top {K}: "
          + ", ".join(f"{L} {m:.2f}" for L, m in out["native_mass"].items()))
    for g in gs:
        a = out["arms"]
        print(f"== {g}: per layer, typical P(set) per million: target | question | English, for "
              + " ; ".join(f"{arm} ({a[arm][g]['mean_shared']:.2f} shared tokens per row)" for arm in LENS_ARMS))
        for l in range(len(a[ARMS[0]][g]["T"])):
            print(f"  {l + 1:>2} " + "   ".join(" | ".join(f"{1e6 * math.exp(a[arm][g][k][l][0]):>9,.1f}" for k in "TSE")
                                           for arm in LENS_ARMS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "analyse"))
    {"run": run, "analyse": analyse}[ap.parse_args().stage]()
