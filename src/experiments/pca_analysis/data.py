"""Load the parallel statements used in EXP1.

A *statement* is one piece of source text together with its translations into all EXP1
languages: an MGSM/MSVAMP question, or an XNLI premise or hypothesis. Each XNLI premise
is shared by 3 hypotheses and is returned once.

Selection (see EXP1.md, Setup):
  * MGSM and MSVAMP: split "compute". XNLI: split "analysis".
  * A row is dropped if any requested language is in its "flags" (broken translation).
    For XNLI, a premise is dropped if any of its rows is flagged.
  * Each statement is assigned to the "fit" or "test" half with a fixed seed. XNLI
    premises and their hypotheses always land in the same half.
  * By default at most 250 statements per benchmark are kept (125 fit / 125 test,
    seeded), so XNLI does not dominate pooled analyses. MGSM has only 83.

Usage:
    from data import load_statements
    statements = load_statements(half="fit")
    statements[0].text["zh"]

    python src/experiments/pca_analysis/data.py   # prints counts and an example
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed_data"
LANGS = ("en", "zh", "fr", "de", "ru", "es", "sw", "th")
BENCHMARKS = ("mgsm", "msvamp", "xnli")
SPLIT = {"mgsm": "compute", "msvamp": "compute", "xnli": "analysis"}
SEED = 0
MAX_PER_BENCHMARK = 250


@dataclass(frozen=True)
class Statement:
    id: str          # "mgsm-0012", "xnli-validation-0036/hypothesis", "xnli-premise-0004"
    benchmark: str
    field: str       # "question", "premise" or "hypothesis"
    half: str        # "fit" or "test"
    text: dict       # lang -> text


def load_rows(benchmark):
    with open(PROC / f"{benchmark}.jsonl", encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if r["split"] == SPLIT[benchmark]]


def flagged(row, langs):
    return any(L in row.get("flags", {}) for L in langs)


def halves(keys, rng):
    """Map each key to "fit" or "test", half of them each (fit gets the extra one)."""
    keys = sorted(set(keys))
    order = rng.permutation(len(keys))
    return {keys[i]: "fit" if rank < (len(keys) + 1) // 2 else "test" for rank, i in enumerate(order)}


def math_statements(benchmark, langs, rng):
    rows = [r for r in load_rows(benchmark) if not flagged(r, langs)]
    half = halves([r["id"] for r in rows], rng)
    return [Statement(r["id"], benchmark, "question", half[r["id"]], {L: r["text"][L]["question"] for L in langs})
            for r in rows]


def xnli_statements(langs, rng):
    rows = load_rows("xnli")
    groups = {}
    for r in rows:
        groups.setdefault(r["text"]["en"]["premise"], []).append(r)
    groups = {p: g for p, g in groups.items() if not any(flagged(r, langs) for r in g)}
    half = halves(groups, rng)
    out = []
    for k, (premise, group) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0]["id"])):
        texts = {L: {r["text"][L]["premise"] for r in group} for L in langs}
        assert all(len(t) == 1 for t in texts.values()), f"premise translations differ: {premise!r}"
        out.append(Statement(f"xnli-premise-{k:04d}", "xnli", "premise", half[premise],
                             {L: t.pop() for L, t in texts.items()}))
        out += [Statement(f"{r['id']}/hypothesis", "xnli", "hypothesis", half[premise],
                          {L: r["text"][L]["hypothesis"] for L in langs}) for r in group]
    return out


def subsample(statements, n, rng):
    """A random subset of n statements (all if fewer), in their original order."""
    if len(statements) <= n:
        return statements
    keep = set(rng.choice(len(statements), size=n, replace=False))
    return [s for i, s in enumerate(statements) if i in keep]


def load_statements(benchmarks=BENCHMARKS, langs=LANGS, half=None, max_per_benchmark=MAX_PER_BENCHMARK):
    """EXP1 statements of the given benchmarks, parallel in `langs`.

    `half` is "fit", "test" or None (both). `max_per_benchmark` caps the number of
    statements per benchmark (fit gets half rounded up, test the rest), so that no
    benchmark dominates pooled analyses; None keeps everything."""
    assert "en" in langs
    out = []
    for b in benchmarks:
        rng = np.random.default_rng(SEED)
        statements = xnli_statements(langs, rng) if b == "xnli" else math_statements(b, langs, rng)
        if max_per_benchmark is not None:
            caps = {"fit": (max_per_benchmark + 1) // 2, "test": max_per_benchmark // 2}
            statements = [s for h, cap in caps.items()
                          for s in subsample([s for s in statements if s.half == h], cap, rng)]
        out += statements
    return [s for s in out if half is None or s.half == half]


def main():
    statements = load_statements()
    counts = Counter((s.benchmark, s.field, s.half) for s in statements)
    print("| benchmark | field | fit | test |\n|---|---|---|---|")
    for b, f in sorted({(b, f) for b, f, _ in counts}):
        print(f"| {b} | {f} | {counts[b, f, 'fit']} | {counts[b, f, 'test']} |")
    print(f"total: {len(statements)} statements x {len(LANGS)} languages")
    s = statements[0]
    print(f"\nexample {s.id} ({s.half}):")
    for L, t in s.text.items():
        print(f"  {L}: {t[:80]}")


if __name__ == "__main__":
    main()
