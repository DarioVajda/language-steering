"""Does the start of the answer set its language? The instruction and instruction + prompt-only steering, with
the first k tokens of the answer forced (prefilled) and the rest generated as usual.

The forced start is in T or in S:
  T   the first k tokens of the native arm's test response to the same item (the question in T)
  S   the first k tokens of the source arm's test response to the same item (the question in S, no instruction)
re-tokenised from the saved responses; k in K = (0, 1, 4, 16). k = 0 (nothing forced) is generated here too, for
start T only (start S at k = 0 is the same run), so that every cell comes from the same run and hardware. Prompt-only
steering uses the setting chosen on dev (chosen.json) and, as in generation, covers the prompt only: the forced
tokens are part of the answer and are not steered. The forced tokens count towards the 2048 answer tokens.

Items: N_ITEMS of the 200 test items, drawn once with a fixed seed, the same for every pair, arm, start and k;
the 99 held-out pairs. A unit is one arm, start language, k and source S (its 9 held-out targets)
-> results/<model>/generations/prefill/<arm>_<T|S>_k<k>_<S>.jsonl; each row has the answer (forced start +
continuation) and the continuation alone.

Scoring (analyse) is that of ../mgsm_steering/analyse.py, except that the language is judged on the continuation
only, so that the forced tokens cannot count as being in T; correctness is judged on the whole answer. Reported,
for each arm, start and k: success, accuracy and in-T rate, as means over the held-out pairs and over the held-out
pairs where the instruction fails (success < 0.10 on all 200 test items, from summary.json, i.e. defined before this
experiment), each with a 95% bootstrap interval over the items (the same items are in every pair, so they are
resampled together).

Usage: XL_MODEL=<model> python src/experiments/mgsm_crosslingual/prefill.py run       # one per GPU
       XL_MODEL=<model> python src/experiments/mgsm_crosslingual/prefill.py analyse   # -> results/<model>/prefill.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from generate import MODELS, Model, split  # noqa: E402
from generate_pairs import GEN, KEY, LANGS, MODEL, PAIRS, RES, TUNE, claim, prompt, vector  # noqa: E402
from tasks import MGSM  # noqa: E402

ARMS = ("instruction", "instruction_steer_prompt")
STARTS = ("T", "S")
K = (0, 1, 4, 16)
N_ITEMS, SEED = 50, 0
HELD = tuple(p for p in PAIRS if p not in TUNE)  # as in analyse_pairs.py
OUT = GEN / "prefill"
FAIL = 0.10  # instruction success below this on test: a failing pair


def items():
    """The N_ITEMS test rows, in test order."""
    test = split()["test"]
    keep = set(random.Random(SEED).sample([r["id"] for r in test], N_ITEMS))
    return [r for r in test if r["id"] in keep]


def units():
    """{unit name: (arm, start, k, S)}; k = 0 only once per arm (as start T)."""
    return {f"{a}_{st}_k{k}_{s}": (a, st, k, s) for a in ARMS for st in STARTS for k in K for s in LANGS
            if k or st == "T"}


def starts(tok, st, s, t):
    """{item id: token ids of the start of the reference answer}: native (in T) for start T, source (in S) for S."""
    f = GEN / "test" / (f"native_{t}.jsonl" if st == "T" else f"source_{s}.jsonl")
    return {r["id"]: tok(r["response"], add_special_tokens=False).input_ids for r in map(json.loads, open(f))}


def run_unit(M, mu, chosen, arm, st, k, s):
    rows = items()
    todo = [(r, s, t) for t in LANGS if (s, t) in HELD for r in rows]
    ref = {t: starts(M.tok, st, s, t) for t in LANGS if (s, t) in HELD}
    fixed = [ref[t][r["id"]][:k] for r, _, t in todo]
    layer, alpha = (chosen[arm]["layer"], chosen[arm]["alpha"]) if arm in chosen else (None, None)
    vs = [vector(arm, mu, layer, alpha, r, s_, t)[0] for r, s_, t in todo] if layer else None
    texts = [prompt(arm, r, s_, t) for r, s_, t in todo]
    res = M.generate(texts, layer, vs, True, prefix=fixed)
    task = MGSM()
    return [dict(id=r["id"], src=s_, tgt=t, lang=t, arm=arm, start=st, k=k, layer=layer, alpha=alpha,
                 answer=r["answer"], pred=task.parse(out), input=x, forced=M.tok.decode(f), response=out,
                 continuation=cont, n_tokens=n, capped=c)
            for (r, s_, t), x, f, (out, n, c, cont) in zip(todo, texts, fixed, res)]


def run():
    M = Model(MODELS[MODEL]["hf"])
    mu = torch.load(GEN / "vectors.pt")
    chosen = json.loads((GEN / "chosen.json").read_text())
    OUT.mkdir(exist_ok=True)
    for name, (arm, st, k, s) in units().items():
        if (OUT / f"{name}.jsonl").exists() or not claim("prefill", name):
            continue
        rows = run_unit(M, mu, chosen, arm, st, k, s)
        tmp = OUT / f".{name}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.rename(OUT / f"{name}.jsonl")
        print(f"prefill {name}: {len(rows)} rows | {rows[0]['forced']!r} -> {rows[0]['continuation'][:60]!r}",
              flush=True)


def analyse():
    from analyse import correct, judge, load  # ../mgsm_steering/analyse.py (loads fastText)
    while missing := [n for n in units() if not (OUT / f"{n}.jsonl").exists()]:  # other workers may still run
        print(f"waiting for {len(missing)} units", flush=True)
        time.sleep(30)
    ids = [r["id"] for r in items()]
    summary = json.loads((RES / "summary.json").read_text())["test"]
    failing = [p for p in HELD if summary[KEY[p]]["instruction"]["success"] < FAIL]
    # per (arm, start, k): {pair: {item id: (correct, in T)}}; the k = 0 run is the point k = 0 of both starts
    cells = {}
    for name, (arm, st, k, s) in units().items():
        for r in load(OUT / f"{name}.jsonl"):
            v = (correct(r), judge(r["continuation"], r["lang"])[0])
            for st_ in (STARTS if k == 0 else (st,)):
                cells.setdefault((arm, st_, k), {}).setdefault((r["src"], r["tgt"]), {})[r["id"]] = v
    measures = {"success": lambda c, l: c and l, "accuracy": lambda c, l: c, "in_lang": lambda c, l: l}
    rng = random.Random(SEED)
    boot = [[rng.randrange(len(ids)) for _ in ids] for _ in range(2000)]

    def stat(cell, pairs, f):
        """Mean over pairs and items, and the 95% bootstrap interval over items."""
        per = [sum(f(*cell[p][i]) for p in pairs) / len(pairs) for i in ids]  # per item, mean over pairs
        ms = sorted(sum(per[j] for j in b) / len(b) for b in boot)
        return [sum(per) / len(per), ms[int(0.025 * len(ms))], ms[int(0.975 * len(ms)) - 1]]

    groups = {"held_out": HELD, "failing": failing}
    out = dict(n_items=len(ids), items=ids, n_pairs={g: len(ps) for g, ps in groups.items()}, failing=[KEY[p] for p in failing],
               cells=[])
    for (arm, st, k), cell in sorted(cells.items(), key=lambda x: (ARMS.index(x[0][0]), x[0][1], x[0][2])):
        assert set(cell) == set(HELD) and all(len(v) == len(ids) for v in cell.values()), (arm, st, k)
        out["cells"].append(dict(arm=arm, start=st, k=k, **{g: {m: stat(cell, ps, f) for m, f in measures.items()}
                                                            for g, ps in groups.items()}))
    (RES / "prefill.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(ids)} items; {len(HELD)} held-out pairs, {len(failing)} failing (instruction success < {FAIL})")
    for g in groups:
        print(f"== {g}: success [95% CI] / accuracy / in T")
        for c in out["cells"]:
            m = c[g]
            print(f"{c['arm']:26} {c['start']} k={c['k']:<3} {m['success'][0]:.3f} [{m['success'][1]:.3f}, "
                  f"{m['success'][2]:.3f}]  {m['accuracy'][0]:.3f}  {m['in_lang'][0]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "analyse"))
    {"run": run, "analyse": analyse}[ap.parse_args().stage]()
