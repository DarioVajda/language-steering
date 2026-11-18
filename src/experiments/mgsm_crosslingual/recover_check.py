"""Checks of our ReCoVeR reimplementation (Model.generate(recover=...), ../mgsm_steering/generate.py) against the
authors' code, github.com/hSterz/recover @ 4c6eb4d (cloned outside the repo; --repo), for one model:

  hooks      the same token ids through the model with their Steer hooks (Steer.naive_steer, bs = 1) and with ours:
             the logits at every prompt position and at STEPS teacher-forced decoding steps must match exactly, for
             alpha in {0.2, 2} x restore_norm off/on. Their code reads the decoder layers at model.model.layers and
             expects a block to return a tuple; a shim gives it the decoder's blocks and wraps the output, nothing else.
             Prompt-only: our prompt-only hooks against theirs on the prompt and no steering after it.
  batched    greedy generation (NEW tokens) of N prompts with our batched decoding (left padding, rows leaving the
             batch) against a plain bs = 1 greedy loop with their hooks, steered and unsteered: how many prompts give
             the same tokens (the unsteered rate is the baseline for numerical differences of batching).
  vectors    their vector collection (collect.py's _aggregate over right-padded 512-token batches, their BOS rule)
             against Model.mean_states on the same FLORES sentences: the cosine of mu_T - mu_S at every hidden state.
  lcb        (stage lcb) their cross-lingual LCB setting with our implementation: every prompt of the 14 languages (dev
             and test), their system prompt "You are a helpful assistant.", 100 new tokens (greedy here), vectors from
             the 997 FLORES-200 dev sentences, ReCoVeR at alpha 0.2 and 1 with norm restore and 0.2 without, against
             no steering; LPR (the LCB scorer, ../language_confusion/analyse.py), mean over languages. Their paper:
             Llama-3.1-8B-Instruct 91.0 -> 97.9.

Usage: XL_MODEL=<model> python recover_check.py --repo <clone of hSterz/recover> [--stages hooks,batched,vectors,lcb]
       -> results/<model>/recover_check_<stages>.json
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
import generate as base  # noqa: E402
from generate import chat  # noqa: E402
from generate_pairs import MODEL, MODELS, RES, Model, flores, prompt, split  # noqa: E402

PAIRS = (("en", "zh"), ("de", "ja"), ("sw", "fr"), ("th", "ru"))
N_FLORES, STEPS, NEW, N = 64, 8, 64, 8


def their_steer(repo, M, mu, alpha, restore, src, tgt, tmp):
    """Their Steer object, steering src -> tgt, with the vectors mu written in their format (one .npy per language,
    [n_layers + 1, 1, d] float32)."""
    sys.path.insert(0, str(Path(repo) / "recover"))
    from steering import Steer  # noqa: E402

    tmp.mkdir(parents=True, exist_ok=True)
    for L, v in mu.items():
        np.save(tmp / f"{L}.npy", v.float().cpu().numpy()[:, None])
    cfg = M.model.config.get_text_config()
    shim = types.SimpleNamespace(model=types.SimpleNamespace(layers=M.blocks), config=cfg)
    s = Steer(shim, path=str(tmp), arithmetic="naive", remove_content=True, anchor=src, alpha=alpha, beta=1.0,
              scaling_mode="norm", restore_norm=restore)
    orig = s.naive_steer  # a block returns a tensor here, a tuple in the transformers they used

    def adapted(module, args, output, **kw):
        return orig(module, args, output, **kw) if isinstance(output, tuple) else orig(module, args, (output,), **kw)[0]

    s.naive_steer = adapted
    s.set_lang(tgt, anchor=src)
    return s


def our_hooks(M, mu, alpha, restore, src, tgt, n_prompt, prompt_only, steer=None):
    """Our hooks with our vector mu_T - mu_S or, given their Steer object, with their vector (r_T - r_S of their
    centred vectors, subtracted on the GPU as they do): the same up to float32 rounding."""
    if steer is None:
        v = (mu[tgt] - mu[src]).to("cuda", torch.float32)[None]
    else:
        r = lambda L: torch.tensor(steer.vectors[L][:, 0], device="cuda")
        v = (r(tgt) - 1.0 * r(src))[None]
    st = {"vec": v / torch.linalg.norm(v, dim=-1, keepdim=True) * alpha}
    one = lambda x: torch.tensor([[x]], device="cuda")
    return M.recover_hooks(st, one(1), one(n_prompt), prompt_only, restore)


@torch.inference_mode()
def forced(M, ids, tokens=None, steps=STEPS, between=None):
    """Logits [n_prompt + steps, vocab] of a bs = 1 prefill and teacher-forced (tokens) or greedy decoding steps;
    between() is called after the prefill. Returns the logits and the tokens fed."""
    o = M.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
    cache, out, fed = o.past_key_values, [o.logits[0].float()], []
    if between:
        between()
    for k in range(steps):
        nxt = int(out[-1][-1].argmax()) if tokens is None else tokens[k]
        fed.append(nxt)
        o = M.model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        out.append(o.logits[0].float())
    return torch.cat(out), fed


def check_hooks(repo, M, mu, items, tmp):
    res = []
    for (r, s, t) in items:
        text = prompt("instruction", r, s, t)
        ids = M.tok(chat(M.tok, [text])[0], add_special_tokens=False).input_ids
        for alpha in (0.2, 2.0):
            for restore in (False, True):
                st = their_steer(repo, M, mu, alpha, restore, s, t, tmp)
                ref, fed = forced(M, ids)
                st.clear()
                ours = {}
                for name, sv in (("ours", None), ("their_vector", st)):
                    hs = our_hooks(M, mu, alpha, restore, s, t, len(ids), False, sv)
                    ours[name], _ = forced(M, ids, fed)
                    for h in hs:
                        h.remove()
                # prompt-only: their hooks on the prefill only, against our prompt-only hooks
                st = their_steer(repo, M, mu, alpha, restore, s, t, tmp)
                ref_po, fed_po = forced(M, ids, between=st.clear)
                ours_po = {}
                for name, sv in (("ours", None), ("their_vector", st)):
                    hs = our_hooks(M, mu, alpha, restore, s, t, len(ids), True, sv)
                    ours_po[name], _ = forced(M, ids, fed_po)
                    for h in hs:
                        h.remove()
                plain, _ = forced(M, ids, fed)
                d = dict(pair=f"{s}-{t}", alpha=alpha, restore_norm=restore,
                         steering_effect=(ref - plain).abs().max().item())
                for name in ours:
                    d[f"max_abs_diff_{name}"] = (ref - ours[name]).abs().max().item()
                    d[f"max_abs_diff_prompt_only_{name}"] = (ref_po - ours_po[name]).abs().max().item()
                res.append(d)
                print(f"hooks {s}-{t} alpha {alpha} restore {restore}: max |their - our logits| "
                      f"{d['max_abs_diff_ours']:.3g} (prompt-only {d['max_abs_diff_prompt_only_ours']:.3g}); with "
                      f"their vector {d['max_abs_diff_their_vector']:.3g} "
                      f"({d['max_abs_diff_prompt_only_their_vector']:.3g}); steering moves them by up to "
                      f"{d['steering_effect']:.3g}", flush=True)
    return res


def check_batched(repo, M, mu, items, tmp):
    base.MAX_NEW = NEW
    texts = [prompt("instruction", r, s, t) for r, s, t in items]
    out = {}
    for name, rec in (("unsteered", None), ("recover_a1_r1", dict(alpha=1.0, restore_norm=True)),
                      ("recover_a0.2_r0", dict(alpha=0.2, restore_norm=False))):
        if rec:
            ours = M.generate(texts, vecs=[mu[t] - mu[s] for _, s, t in items], recover=rec)
        else:
            ours = M.generate(texts)
        same, first = 0, []
        for (r, s, t), x, (txt, k, c) in zip(items, texts, ours):
            ids = M.tok(chat(M.tok, [x])[0], add_special_tokens=False).input_ids
            st = their_steer(repo, M, mu, rec["alpha"], rec["restore_norm"], s, t, tmp) if rec else None
            _, toks = forced(M, ids, steps=NEW)
            if st:
                st.clear()
            ref = M.tok.decode(toks[:next((i for i, y in enumerate(toks) if y in M.eos.tolist()), len(toks))],
                               skip_special_tokens=True).strip()
            same += ref == txt
            first.append(next((i for i, (a, b) in enumerate(zip(ref, txt)) if a != b), min(len(ref), len(txt))))
        out[name] = dict(same=same, n=len(items), first_diff_chars=first)
        print(f"batched {name}: {same}/{len(items)} prompts give the same {NEW} tokens; first differing character "
              f"{first}", flush=True)
    base.MAX_NEW = 2048
    return out


@torch.inference_mode()
def check_vectors(repo, M, langs):
    stub = types.ModuleType("datasets")  # collect.py imports load_dataset for its main() only
    stub.load_dataset = None
    sys.modules.setdefault("datasets", stub)
    sys.path.insert(0, str(Path(repo) / "collect_language_vectors"))
    from collect import _aggregate  # noqa: E402

    tok = M.tok
    pad = tok.pad_token_id
    theirs, ours = {}, {}
    for L in langs:
        texts = flores(L)[:N_FLORES]
        acc = None
        for i in range(0, len(texts), 4):  # their defaults: batch 4, padding to max_length 512, right padding
            tok.padding_side = "right"
            b = tok(texts[i:i + 4], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            tok.padding_side = "left"
            hs = M.model(input_ids=b.input_ids.cuda(), attention_mask=b.attention_mask.cuda(),
                         output_hidden_states=True).hidden_states
            state = torch.stack([x.float().cpu() for x in hs], dim=1)  # [B, n_layers + 1, 512, d]
            s = torch.stack([_aggregate(state[j], b.attention_mask[j]) for j in range(len(state))]).sum(0)
            acc = s if acc is None else acc + s
        theirs[L] = acc[:, 0] / len(texts)
        ours[L] = M.mean_states(texts).cpu()
    assert tok.padding_side == "left" and tok.pad_token_id == pad
    res = {}
    for s, t in PAIRS:
        a, b = theirs[t] - theirs[s], ours[t] - ours[s]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        res[f"{s}-{t}"] = dict(min_cosine=cos.min().item(), mean_cosine=cos.mean().item(),
                               argmin=int(cos.argmin()))
        print(f"vectors {s}-{t}: cosine of their mu_T - mu_S to ours over the hidden states: min {cos.min():.6f} "
              f"(at {int(cos.argmin())}), mean {cos.mean():.6f}", flush=True)
    return res


def check_lcb(M):
    os.environ["LCB_MODEL"] = MODEL
    sys.path.insert(0, str(HERE.parent / "language_confusion"))
    import data as lcb  # noqa: E402
    import analyse as lcb_analyse  # noqa: E402  (../language_confusion/analyse.py, not mgsm_steering's)

    code = {"fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "pt": "por_Latn", "it": "ita_Latn",
            "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans", "ar": "arb_Arab", "tr": "tur_Latn",
            "hi": "hin_Deva", "ru": "rus_Cyrl", "id": "ind_Latn", "vi": "vie_Latn", "en": "eng_Latn"}
    base.FLORES_CODE.update(code)
    mu = {L: M.mean_states(base.flores(L)).cpu() for L in code}
    rows = lcb.load("dev") + lcb.load("test")
    sysmsg = ["You are a helpful assistant."] * len(rows)
    base.MAX_NEW, batch = 100, base.MAX_BATCH
    base.MAX_BATCH = 32  # some LCB prompts are long: 128 rows of them do not fit an 80 GB GPU at prefill
    out = {}
    for name, rec in (("none", None), ("a0.2_r1", dict(alpha=0.2, restore_norm=True)),
                      ("a1_r1", dict(alpha=1.0, restore_norm=True)), ("a0.2_r0", dict(alpha=0.2, restore_norm=False))):
        texts = [r["prompt"] for r in rows]
        res = (M.generate(texts, system=sysmsg, vecs=[mu[r["lang"]] - mu["en"] for r in rows], recover=rec) if rec
               else M.generate(texts, system=sysmsg))
        gen = [dict(r, response=x) for r, (x, k, c) in zip(rows, res)]
        per = {L: lcb_analyse.score([g for g in gen if g["lang"] == L])["lpr"] for L in lcb.LANGS}
        out[name] = dict(mean_lpr=sum(per.values()) / len(per), per_lang=per)
        print(f"lcb {name}: mean LPR {out[name]['mean_lpr']:.3f} | " +
              " ".join(f"{L} {v:.2f}" for L, v in per.items()), flush=True)
    base.MAX_NEW, base.MAX_BATCH = 2048, batch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stages", default="hooks,batched,vectors", help="any of hooks, batched, vectors, lcb")
    a = ap.parse_args()
    M = Model(MODELS[MODEL]["hf"])
    langs = sorted({L for p in PAIRS for L in p})
    mu = {L: M.mean_states(flores(L)[:N_FLORES]).cpu() for L in langs}
    dev = split()["dev"]
    items = [(dev[i], *PAIRS[i % len(PAIRS)]) for i in range(N)]
    tmp = RES / "generations" / "recover" / "check_vectors"
    out = {"model": MODEL, "repo_commit": os.popen(f"git -C {a.repo} rev-parse HEAD").read().strip(),
           "stages": a.stages}
    st = a.stages.split(",")
    if "hooks" in st:
        out["hooks"] = check_hooks(a.repo, M, mu, items[:len(PAIRS)], tmp)
    if "batched" in st:
        out["batched"] = check_batched(a.repo, M, mu, items, tmp)
    if "vectors" in st:
        out["vectors"] = check_vectors(a.repo, M, langs)
    if "lcb" in st:
        out["lcb"] = check_lcb(M)
    (RES / f"recover_check_{a.stages.replace(',', '_')}.json").write_text(json.dumps(out, indent=1) + "\n")
    if "hooks" in st:
        ok = all(h["max_abs_diff_their_vector"] == 0 and h["max_abs_diff_prompt_only_their_vector"] == 0
                 for h in out["hooks"])
        print(f"\nhooks identical to theirs: {ok}", flush=True)


if __name__ == "__main__":
    main()
