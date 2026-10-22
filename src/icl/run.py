"""Evaluate one ICL method on one benchmark: tune on val if the method has hyperparameters, then test.

  1. If the method has a grid (methods.py), every config (the fallback first, then the grid) is
     scored on the val split for every target language and seed. The best config is chosen on
     mean val accuracy, over all languages (--tune-scope global) or for each language on its own
     (--tune-scope per_lang, as in Kirtane & Huang's main results). Ties go to the earlier config,
     so the fallback (no steering) wins unless a config beats it strictly. Val results are
     appended to val.jsonl as they finish, and a restarted job skips the ones already there.
     With --shard I/N, a job scores only every N-th config (I, I+N, ...), writes them to
     val_<I>of<N>.jsonl and stops; an unsharded run afterwards reads all val*.jsonl files.
  2. The chosen config (or {} without a grid) is run on the test split. --config JSON skips
     tuning and runs that config, e.g. one chosen before, on new seeds.

Outputs, in src/icl/results/<model>/<benchmark>/<method>[_<tag>]/:
  val.jsonl              one line per (config, language, seed): accuracy and per-item correctness
  tuning.json            val accuracy of every config and the chosen config per language
  test_predictions.jsonl one line per (item, language, seed)
  summary.json           chosen configs and test accuracy per language and on average

Usage (GPU job, see CLAUDE.md):
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
from vectors import LanguageVectors, slug

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


def tune(method, items, langs, seeds, mode, scope, out, shard=None):
    """Val accuracy of every config per language, and the chosen config per language.
    With shard = (I, N), only scores configs I, I+N, ... and returns None."""
    cache_file = out / (f"val_{shard[0]}of{shard[1]}.jsonl" if shard else "val.jsonl")
    cache = {}
    for file in sorted(out.glob("val*.jsonl")):
        for line in file.read_text().splitlines():
            r = json.loads(line)
            cache[key(r["config"]), r["lang"], r["seed"]] = r["acc"]
    configs = method.configs()
    val = {}
    for c_i, config in enumerate(configs):
        if shard and c_i % shard[1] != shard[0]:
            continue
        val[key(config)] = {}
        for L in langs:
            accs = []
            for seed in seeds:
                if (key(config), L, seed) not in cache:
                    t0 = time.time()
                    records = method.run(items[L], L, config, seed, mode)
                    cache[key(config), L, seed] = accuracy(records)
                    with open(cache_file, "a") as f:
                        f.write(json.dumps({"config": config, "lang": L, "seed": seed, "n": len(records),
                                            "acc": cache[key(config), L, seed],
                                            "correct": [int(r["correct"]) for r in records]}) + "\n")
                    print(f"[{c_i + 1}/{len(configs)}] {key(config)} {L} seed {seed}: "
                          f"{cache[key(config), L, seed]:.3f} ({time.time() - t0:.0f}s)", flush=True)
                accs.append(cache[key(config), L, seed])
            val[key(config)][L] = float(np.mean(accs))
    if shard:
        return None

    def best(score):
        top = configs[0]
        for c in configs[1:]:
            if score(c) > score(top):
                top = c
        return top
    if scope == "global":
        g = best(lambda c: np.mean(list(val[key(c)].values())))
        chosen = {L: g for L in langs}
    else:
        chosen = {L: best(lambda c, L=L: val[key(c)][L]) for L in langs}
    return val, chosen


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
    method = Method(task, runner, vectors, batch_size=args.batch_size)
    if args.grid:
        method.grid = {**method.grid, **parse_grid(args.grid)}

    tag = args.tag or (f"{args.limit}items" if args.limit else None)
    out = HERE / "results" / slug(args.model) / args.benchmark / (method.name + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    test_seeds = args.test_seeds or args.seeds
    setup = {"model": args.model, "benchmark": args.benchmark, "method": method.name, "mode": mode,
             "langs": langs, "seeds": args.seeds, "test_seeds": test_seeds, "k": method.k, "limit": args.limit}
    print(json.dumps(setup), flush=True)

    if args.config is not None:
        print(f"fixed config {key(args.config)}: running on test", flush=True)
        chosen = {L: args.config for L in langs}
    elif method.grid:
        val_items = {L: task.items("val", L)[:args.limit] for L in langs}
        shard = tuple(map(int, args.shard.split("/"))) if args.shard else None
        tuned = tune(method, val_items, langs, args.seeds, mode, args.tune_scope, out, shard)
        if shard:
            print(f"shard {args.shard} done: run without --shard to select the config and test", flush=True)
            return
        val, chosen = tuned
        (out / "tuning.json").write_text(json.dumps(
            {**setup, "grid": method.grid, "fallback": method.fallback, "tune_scope": args.tune_scope,
             "val": val, "chosen": chosen}, indent=1))
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
        print(f"test {L} {key(chosen[L])}: {test[L]:.3f} (n={len(items)} x {len(test_seeds)} seeds)", flush=True)
    with open(out / "test_predictions.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    summary = {**setup, "chosen": chosen, "test": test, "test_mean": float(np.mean(list(test.values())))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"test mean: {summary['test_mean']:.3f} -> {out}")


if __name__ == "__main__":
    main()
