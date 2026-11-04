"""Evaluate one ICL method on one benchmark: tune on val if the method has hyperparameters, then test.

  1. If the method has a grid (methods.py), every config (the fallback first, then the grid) is
     scored on the val split for every target language and seed. The best config is chosen on
     mean val accuracy, over all languages (--tune-scope global) or for each language on its own
     (--tune-scope per_lang, as in Kirtane & Huang's main results). Ties go to the earlier config,
     so the fallback (no steering) wins unless a config beats it strictly. Val results are
     appended to val.jsonl as they finish, and a restarted job skips the ones already there (only
     those scored with the same token cap; records without one were scored with the task's default).
     With --shard I/N, a job scores only every N-th config (I, I+N, ...), writes them to
     val_<I>of<N>.jsonl and stops; an unsharded run afterwards reads all val*.jsonl files.
     With --val-stage1 F, tuning has two stages (a race): (1) every config is scored on a seeded
     random fraction F of the val items (the same items in every language); (2) the configs whose
     stage-1 score is within --race-z paired standard errors of the best one, and the fallback,
     are scored on the remaining items, and the config is chosen among them on the full val
     split. Configs that are clearly worse after stage 1 are not scored further. A sharded run
     does its share of stage 1 and, once stage 1 is complete (shards run again), of stage 2.
  2. The chosen config (or {} without a grid) is run on the test split. --config JSON skips
     tuning and runs that config, e.g. one chosen before, on new seeds.

Outputs, in src/icl/results/<model>/<benchmark>/<method>[_<tag>]/:
  val.jsonl              one line per (config, language, seed[, stage]): accuracy, per-item correctness
                         and, when generating, the token cap and which responses hit it
  tuning.json            val accuracy of every config (stage 1 only for configs dropped by the race),
                         the chosen config per language and, with --val-stage1, the kept configs
  test_predictions.jsonl one line per (item, language, seed); when generating, with the response,
                         its length in tokens and whether it hit --max-new-tokens
  summary.json           chosen configs and test accuracy per language and on average

Usage (GPU job, via slurm/submit.sh):
    slurm/submit.sh -J icl_steer_xnli -t 12:00:00 -- python src/icl/run.py --benchmark xnli --method steer
    python src/icl/run.py --benchmark xnli --method steer --grid layer=10 alpha=1,3 --limit 20   # quick test
    # the same tuning on 8 GPUs: 8 shards, then a run that selects the config and tests it
    for i in 0 1 2 3 4 5 6 7; do slurm/submit.sh -- python src/icl/run.py ... --shard $i/8; done
    slurm/submit.sh --dependency afterok:<shard job ids> -- python src/icl/run.py ...
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from methods import METHODS
from model import Runner
from tasks import TASKS
from vectors import LanguageVectors, item_rng, slug

HERE = Path(__file__).resolve().parent


def key(config):
    return json.dumps(config, sort_keys=True)


def parse_grid(specs):
    """["layer=5,10", "positions=demos"] -> {"layer": [5, 10], "positions": ["demos"]}."""
    def value(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return {name: [value(v) for v in values.split(",")]
            for name, values in (s.split("=", 1) for s in specs)}


def accuracy(records):
    return float(np.mean([r["correct"] for r in records])) if records else float("nan")


def split_val(items, langs, fraction):
    """Stage-1 and stage-2 val items per language: a seeded random `fraction` of the item ids
    (the same ids in every language) and the rest."""
    ids = sorted({it.id for L in langs for it in items[L]})
    perm = item_rng(0, "val_stage1").permutation(len(ids))
    first = {ids[j] for j in perm[:round(fraction * len(ids))]}
    return ({L: [it for it in items[L] if it.id in first] for L in langs},
            {L: [it for it in items[L] if it.id not in first] for L in langs})


def race(configs, correct, langs, seeds, z, fallback):
    """Configs kept after stage 1: the best one by mean accuracy over languages, those whose paired
    difference to it is above -z standard errors, and the fallback. `correct(c, L)` is the
    [seeds, items] stage-1 correctness of config c in language L."""
    def score(c):
        return np.mean([correct(c, L).mean() for L in langs])
    top = configs[0]
    for c in configs[1:]:
        if score(c) > score(top):
            top = c
    kept = []
    for c in configs:
        d = {L: (correct(c, L) - correct(top, L)).mean(0) for L in langs}   # per item, averaged over seeds
        diff = np.mean([d[L].mean() for L in langs])
        se = np.sqrt(sum(d[L].var(ddof=1) / len(d[L]) for L in langs if len(d[L]) > 1)) / len(langs)
        if c in (top, fallback) or diff + z * se >= 0:
            kept.append(c)
    return kept


def tune(method, items, langs, seeds, mode, scope, out, shard=None, stage1=None, z=2.0):
    """Val accuracy of every config per language, and the chosen config per language.
    With stage1 = F, a two-stage race (see the module docstring): returns the val accuracy (full val
    for the configs kept after stage 1, stage 1 only for the others), the chosen config per
    language and the kept configs. With shard = (I, N), only scores configs I, I+N, ... (of the
    configs that the current stage needs) and returns None."""
    cache_file = out / (f"val_{shard[0]}of{shard[1]}.jsonl" if shard else "val.jsonl")
    cache = {}
    cap = method.max_new_tokens if mode == "generate" else None
    default_cap = method.task.max_new_tokens if mode == "generate" else None   # of records without "max_new_tokens"

    def load():
        for file in sorted(out.glob("val*.jsonl")):
            for line in file.read_text().splitlines():
                r = json.loads(line)
                if r.get("max_new_tokens", default_cap) == cap:
                    cache[key(r["config"]), r["lang"], r["seed"], r.get("stage", 0)] = r["correct"]
    load()
    configs = method.configs()

    def score_all(cs, stage, stage_items):
        for c_i, config in enumerate(cs):
            if shard and c_i % shard[1] != shard[0]:
                continue
            for L in langs:
                for seed in seeds:
                    k = key(config), L, seed, stage
                    if k in cache:
                        continue
                    t0 = time.time()
                    records = method.run(stage_items[L], L, config, seed, mode)
                    cache[k] = [int(r["correct"]) for r in records]
                    with open(cache_file, "a") as f:
                        f.write(json.dumps({"config": config, "lang": L, "seed": seed, "stage": stage,
                                            "n": len(records), "acc": accuracy(records), "correct": cache[k],
                                            "max_new_tokens": cap,
                                            "n_capped": sum(r.get("capped", False) for r in records),
                                            "capped": [int(r.get("capped", False)) for r in records]}) + "\n")
                    print(f"[stage {stage}, {c_i + 1}/{len(cs)}] {k[0]} {L} seed {seed}: {accuracy(records):.3f} "
                          f"(n={len(records)}, {time.time() - t0:.0f}s)", flush=True)

    def complete(cs, stage):
        return all((key(c), L, seed, stage) in cache for c in cs for L in langs for seed in seeds)

    def correct(c, L, stages):
        return np.array([sum((cache[key(c), L, seed, st] for st in stages), []) for seed in seeds], dtype=float)

    if stage1 is None:
        score_all(configs, 0, items)
        if shard:
            return None
        kept, stages = configs, {key(c): (0,) for c in configs}
    else:
        items1, items2 = split_val(items, langs, stage1)
        score_all(configs, 1, items1)
        if shard:
            load()
            if not complete(configs, 1):
                return None
        kept = race(configs, lambda c, L: correct(c, L, (1,)), langs, seeds, z, method.fallback)
        print(f"stage 1 done: {len(kept)} of {len(configs)} configs kept: {[key(c) for c in kept]}", flush=True)
        score_all(kept, 2, items2)
        if shard:
            return None
        stages = {key(c): (1, 2) if c in kept else (1,) for c in configs}
    val = {key(c): {L: float(correct(c, L, stages[key(c)]).mean()) for L in langs} for c in configs}

    def best(score):
        top = kept[0]
        for c in kept[1:]:
            if score(c) > score(top):
                top = c
        return top
    if scope == "global":
        g = best(lambda c: np.mean(list(val[key(c)].values())))
        chosen = {L: g for L in langs}
    else:
        chosen = {L: best(lambda c, L=L: val[key(c)][L]) for L in langs}
    return val, chosen, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--benchmark", required=True, choices=list(TASKS))
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--langs", nargs="+", help="target languages (default: all of the benchmark's languages except en)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="demonstration sampling seeds")
    parser.add_argument("--test-seeds", nargs="+", type=int, help="seeds for the test split (default: --seeds)")
    parser.add_argument("--mode", choices=["generate", "forward"], help="default: the task's (generate)")
    parser.add_argument("--tune-scope", choices=["global", "per_lang"], default="global")
    parser.add_argument("--grid", nargs="+", metavar="NAME=V1,V2", help="override (part of) the method's grid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, help="generation cap (default: the task's)")
    parser.add_argument("--val-stage1", type=float, metavar="F",
                        help="tune in two stages, the first on this fraction of the val items (see above)")
    parser.add_argument("--race-z", type=float, default=2.0, help="standard errors for keeping a config after stage 1")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--limit", type=int, help="only the first N items per split and language (quick tests)")
    parser.add_argument("--config", type=json.loads, metavar="JSON", help="skip tuning and test this config")
    parser.add_argument("--shard", metavar="I/N", help="only score every N-th config on val, starting at I, then stop")
    parser.add_argument("--tag", help="suffix of the output directory, e.g. to keep runs with other settings apart")
    args = parser.parse_args()

    task = TASKS[args.benchmark]()
    langs = args.langs or [L for L in task.langs if L != "en"]
    mode = args.mode or task.mode
    Method = METHODS[args.method]
    runner = Runner(args.model, dtype=getattr(torch, args.dtype))
    vectors = LanguageVectors.load(args.model, args.benchmark) if Method.uses_vectors else None
    method = Method(task, runner, vectors, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    if args.grid:
        method.grid = {**method.grid, **parse_grid(args.grid)}

    tag = args.tag or (f"{args.limit}items" if args.limit else None)
    out = HERE / "results" / slug(args.model) / args.benchmark / (method.name + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    test_seeds = args.test_seeds or args.seeds
    setup = {"model": args.model, "benchmark": args.benchmark, "method": method.name, "mode": mode,
             "langs": langs, "seeds": args.seeds, "test_seeds": test_seeds, "k": method.k, "limit": args.limit,
             "max_new_tokens": method.max_new_tokens if mode == "generate" else None, "batch_size": args.batch_size}
    print(json.dumps(setup), flush=True)

    if args.config is not None:
        print(f"fixed config {key(args.config)}: running on test", flush=True)
        chosen = {L: args.config for L in langs}
    elif method.grid:
        val_items = {L: task.items("val", L)[:args.limit] for L in langs}
        shard = tuple(map(int, args.shard.split("/"))) if args.shard else None
        tuned = tune(method, val_items, langs, args.seeds, mode, args.tune_scope, out, shard, args.val_stage1, args.race_z)
        if shard:
            print(f"shard {args.shard} done: run without --shard to select the config and test", flush=True)
            return
        val, chosen, kept = tuned
        race_info = ({"val_stage1": args.val_stage1, "race_z": args.race_z, "kept": kept}
                     if args.val_stage1 is not None else {})
        (out / "tuning.json").write_text(json.dumps(
            {**setup, "grid": method.grid, "fallback": method.fallback, "tune_scope": args.tune_scope,
             **race_info, "val": val, "chosen": chosen}, indent=1))
        print("chosen:", json.dumps(chosen), flush=True)
    else:
        print("no hyperparameters to tune: running on test", flush=True)
        chosen = {L: {} for L in langs}

    records, test = [], {}
    for L in langs:
        items = task.items("test", L)[:args.limit]
        rs = [r for seed in test_seeds for r in method.run(items, L, chosen[L], seed, mode)]
        test[L] = accuracy(rs)
        records += rs
        capped = sum(r.get("capped", False) for r in rs)
        print(f"test {L} {key(chosen[L])}: {test[L]:.3f} (n={len(items)} x {len(test_seeds)} seeds"
              + (f", {capped} capped" if mode == "generate" else "") + ")", flush=True)
    with open(out / "test_predictions.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    summary = {**setup, "chosen": chosen, "test": test, "test_mean": float(np.mean(list(test.values())))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"test mean: {summary['test_mean']:.3f} -> {out}")


if __name__ == "__main__":
    main()
