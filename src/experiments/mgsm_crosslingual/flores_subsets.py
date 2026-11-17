"""How many FLORES sentences the steering vector needs: instruction + prompt-only steering with vectors from n of the
997 FLORES-200 dev sentences, on the test items of the 99 held-out pairs.

For a pair (S, T), a draw (seed) takes n sentence indices, random.Random(f"{seed}|{S}|{T}").sample(range(997), 100)[:n],
so the sizes are nested (the n sentences are the first n of the 100), and each pair has its own sentences. mu_S and mu_T are the means of
Model.mean_states over those sentences in S and in T (the same sentences, translations of each other), and
v_(S->T) = mu_T - mu_S, used as it is (not normalised). Everything else is the test of instruction_steer_prompt in
generate_pairs.py: its chosen layer and alpha (generations/chosen.json, chosen with the 997-sentence vectors, not
retuned), prompts, decoding and run_unit, all imported unchanged. n = 997 is not rerun: it is the existing test
output (generations/test/instruction_steer_prompt_<S>-<T>.jsonl). generate_pairs.py and its outputs are untouched.

Stages:
  run      the units, one per (n, seed, held-out pair) -> results/<model>/generations/flores_subsets/n<n>_s<seed>/
           <S>-<T>.jsonl, and beside it <S>-<T>.vec.json: the indices, and the cosine and norm ratio of the vector
           to the 997-sentence one at the chosen layer. Workers share the units through claim files, as in
           generate_pairs.py; a unit's output does not depend on which worker ran it. --defer-to <job id>,... makes a
           worker also claim each unit in those jobs' claim directories, so that a second job can share the units
           of a running one without any unit running twice.
  analyse  per (n, seed): held-out mean success, accuracy and in-language rate, pairs below 0.30, McNemar per pair
           against the instruction and against n = 997 (Holm over the 99 pairs), mean cosine and norm ratio; per n:
           the mean and range over the seeds, the per-pair spread over the seeds, and how far each (pair, draw) is
           from n = 997 -> results/<model>/flores_subsets.json

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/flores_subsets.py run|analyse   (flores_subsets.sh)
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate_pairs import GEN, KEY, MODEL, MODELS, PAIRS, RES, TUNE, Model, claim, flores, run_unit, split  # noqa: E402

ARM = "instruction_steer_prompt"
SIZES, SEEDS, N_FLORES = (1, 3, 10, 100), (0, 1, 2), 997
HELD = tuple(p for p in PAIRS if p not in TUNE)
OUT = GEN / "flores_subsets"


def indices(seed, src, tgt, n):
    return sorted(random.Random(f"{seed}|{src}|{tgt}").sample(range(N_FLORES), max(SIZES))[:n])


def units():
    """[(name, n, seed, (src, tgt))], ordered by seed, then n, so that each draw completes before the next."""
    return [(f"n{n}_s{seed}/{KEY[p]}", n, seed, p) for seed in SEEDS for n in SIZES for p in HELD]


def claim_from(jobs, name):
    """Claim the unit in each of the given jobs' claim directories (O_EXCL, as claim() does in its own), so that
    their workers skip it; False if one of them has it already."""
    for j in jobs:
        d = GEN / "claims" / j / "flores_subsets"
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(d / name, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return False
    return True


def run(defer=()):
    chosen = json.loads((GEN / "chosen.json").read_text())[ARM]
    layer, alpha = chosen["layer"], chosen["alpha"]
    full = torch.load(GEN / "vectors.pt")
    test = split()["test"]
    M = Model(MODELS[MODEL]["hf"])
    text = {}
    for name, n, seed, (s, t) in units():
        out = OUT / f"{name}.jsonl"
        c = name.replace("/", "_")
        if out.exists() or not claim_from(defer, c) or not claim("flores_subsets", c):
            continue
        for L in (s, t):
            text.setdefault(L, flores(L))
            assert len(text[L]) == N_FLORES
        idx = indices(seed, s, t, n)
        mu = {L: {layer: M.mean_states([text[L][i] for i in idx])[layer].cpu()} for L in (s, t)}
        v, v_full = (mu[t][layer] - mu[s][layer]).float(), (full[t][layer] - full[s][layer]).float()
        stats = dict(n=n, seed=seed, src=s, tgt=t, layer=layer, alpha=alpha, indices=idx,
                     cosine=torch.nn.functional.cosine_similarity(v, v_full, dim=0).item(),
                     norm_ratio=(v.norm() / v_full.norm()).item())
        rows = run_unit(M, mu, ARM, layer, alpha, [(r, s, t) for r in test])
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / f"{KEY[(s, t)]}.vec.json").write_text(json.dumps(stats) + "\n")
        tmp = out.parent / f".{out.name}.tmp"
        tmp.write_text("".join(json.dumps(dict(r, n_flores=n, seed=seed), ensure_ascii=False) + "\n" for r in rows))
        tmp.rename(out)
        acc = sum(r["pred"] is not None and abs(r["pred"] - r["answer"]) < 1e-6 for r in rows) / len(rows)
        print(f"{name}: acc {acc:.2f} capped {sum(r['capped'] for r in rows)} cos {stats['cosine']:.3f} "
              f"norm {stats['norm_ratio']:.2f}", flush=True)


def analyse():
    from analyse import holm, load, mcnemar, score, success  # noqa: E402  (../mgsm_steering/analyse.py; loads fastText)

    while missing := [u for u in units() if not (OUT / f"{u[0]}.jsonl").exists()]:
        print(f"waiting for {len(missing)} units", flush=True)
        time.sleep(60)
    T = GEN / "test"
    keys = [KEY[p] for p in HELD]
    ref = {k: {a: load(T / f"{a}_{k}.jsonl") for a in ("instruction", ARM)} for k in keys}
    ok = {k: {a: [success(r) for r in rows] for a, rows in d.items()} for k, d in ref.items()}
    full = {k: score(ref[k][ARM]) for k in keys}
    out = {"arm": ARM, "chosen": json.loads((GEN / "chosen.json").read_text())[ARM], "sizes": SIZES, "seeds": SEEDS,
           "held_out_pairs": keys, "n_997": {
               "success": statistics.mean(full[k]["success"] for k in keys),
               "accuracy": statistics.mean(full[k]["accuracy"] for k in keys),
               "in_lang": statistics.mean(full[k]["in_lang"] for k in keys),
               "below_0.30": sum(full[k]["success"] < 0.30 for k in keys)},
           "instruction": {"success": statistics.mean(score(ref[k]["instruction"])["success"] for k in keys)},
           "draws": {}, "per_n": {}}
    per_pair = {}
    for n in SIZES:
        for seed in SEEDS:
            d = OUT / f"n{n}_s{seed}"
            pairs, cmp = {}, {"vs_instruction": [], "vs_997": []}
            for k in keys:
                rows = load(d / f"{k}.jsonl")
                assert [r["id"] for r in rows] == [r["id"] for r in ref[k][ARM]]
                vec = json.loads((d / f"{k}.vec.json").read_text())
                sc = score(rows)
                b = [success(r) for r in rows]
                pairs[k] = dict(success=sc["success"], accuracy=sc["accuracy"], in_lang=sc["in_lang"],
                                capped=sc["capped"], cosine=vec["cosine"], norm_ratio=vec["norm_ratio"])
                for c, a in (("vs_instruction", "instruction"), ("vs_997", ARM)):
                    g, l, p = mcnemar(ok[k][a], b)
                    pairs[k][c] = dict(gained=g, lost=l, p=p)
                    cmp[c].append(pairs[k][c])
                per_pair.setdefault(n, {}).setdefault(k, []).append(sc["success"])
            for c, ds in cmp.items():
                for x, q in zip(ds, holm([x["p"] for x in ds])):
                    x["p_holm"] = q
            m = lambda f: statistics.mean(pairs[k][f] for k in keys)
            out["draws"][f"n{n}_s{seed}"] = {
                "success": m("success"), "accuracy": m("accuracy"), "in_lang": m("in_lang"), "capped": m("capped"),
                "cosine": m("cosine"), "norm_ratio": m("norm_ratio"),
                "min_cosine": min(pairs[k]["cosine"] for k in keys),
                "below_0.30": sum(pairs[k]["success"] < 0.30 for k in keys),
                **{f"better_worse_{c}": [sum(x[q] < 0.05 and x["gained"] > x["lost"] for x in ds)
                                         for q in ("p", "p_holm")] + [sum(x[q] < 0.05 and x["gained"] < x["lost"]
                                                                          for x in ds) for q in ("p", "p_holm")]
                   for c, ds in cmp.items()},
                "pairs": pairs}
        ds = [out["draws"][f"n{n}_s{seed}"] for seed in SEEDS]
        spread = [max(v) - min(v) for v in per_pair[n].values()]
        off = [abs(s - full[k]["success"]) for k, v in per_pair[n].items() for s in v]
        out["per_n"][str(n)] = {
            "success_mean": statistics.mean(x["success"] for x in ds),
            "success_range": [min(x["success"] for x in ds), max(x["success"] for x in ds)],
            "below_0.30": [x["below_0.30"] for x in ds],
            "cosine_mean": statistics.mean(x["cosine"] for x in ds),
            "norm_ratio_mean": statistics.mean(x["norm_ratio"] for x in ds),
            "per_pair_range_mean": statistics.mean(spread), "per_pair_range_median": statistics.median(spread),
            "per_pair_range_max": max(spread),
            "abs_diff_997_median": statistics.median(off), "draws_off_997_by_0.10": sum(x > 0.10 for x in off)}
    print(f"{MODEL}, {ARM} (layer {out['chosen']['layer']}, alpha {out['chosen']['alpha']}), {len(keys)} held-out "
          f"pairs: instruction {out['instruction']['success']:.3f}, n = 997 {out['n_997']['success']:.3f} "
          f"({out['n_997']['below_0.30']} pairs below 0.30)")
    print("better/worse: p < 0.05 uncorrected [Holm], as gained > lost / lost > gained")
    for name, x in out["draws"].items():
        bi, b9 = x["better_worse_vs_instruction"], x["better_worse_vs_997"]
        print(f"{name:8} success {x['success']:.3f} acc {x['accuracy']:.3f} in T {x['in_lang']:.3f} capped "
              f"{x['capped']:.3f} below 0.30 {x['below_0.30']:2}  vs instr {bi[0]}/{bi[2]} [{bi[1]}/{bi[3]}]  "
              f"vs 997 {b9[0]}/{b9[2]} [{b9[1]}/{b9[3]}]  cos {x['cosine']:.3f} (min {x['min_cosine']:.3f}) "
              f"norm {x['norm_ratio']:.2f}")
    for n, x in out["per_n"].items():
        print(f"n = {n}: success {x['success_mean']:.3f} [{x['success_range'][0]:.3f}, {x['success_range'][1]:.3f}] "
              f"over the seeds; per-pair range over the seeds mean {x['per_pair_range_mean']:.3f}, median "
              f"{x['per_pair_range_median']:.3f}, max {x['per_pair_range_max']:.3f}; per pair and draw, |success - "
              f"n = 997| median {x['abs_diff_997_median']:.3f}, > 0.10 in {x['draws_off_997_by_0.10']} of "
              f"{len(keys) * len(SEEDS)}")
    (RES / "flores_subsets.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "analyse"))
    ap.add_argument("--defer-to", default="", help="run: comma-separated job ids whose claims to respect")
    a = ap.parse_args()
    if a.stage == "run":
        run(tuple(j for j in a.defer_to.split(",") if j))
    else:
        analyse()


if __name__ == "__main__":
    main()
