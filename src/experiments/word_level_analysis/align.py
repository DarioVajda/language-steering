"""EXP1 word alignment: find concepts, i.e. English words aligned one-to-one in all languages.

Steps (see docs/plans/EXP1.md, Setup, "Word alignment"):
  1. Segment every text into words with character spans. Han runs (zh) are segmented
     with jieba and Thai runs (th) with PyThaiNLP (newmm); everything else is split by a
     regex into numbers ("1,500", "3.5"), words (with inner "-" or "'": "4-mal",
     "d'habitude") and single punctuation / symbol characters.
  2. Align every English-L sentence pair with SimAlign (Jalili Sabet et al., 2020) on
     mBERT layer 8, subword level, with all three of its methods: "inter" (Argmax, the
     high-precision one), "itermax" and "mwmf".
  3. A pair (i, j) is kept for the English-L pair if English word i is aligned to exactly
     one L word j and j to no English word but i. A punctuation word may only be paired
     with punctuation: Thai rarely writes "." or ",", so they get forced onto Thai words.
  4. A concept is an English word i that has a kept pair in every other language. Tests
     on one language pair (W1-W3, T5) can use the pairs instead, which are not limited
     by the worst-aligned language (Thai).

Each English word is labelled "number", "math" (+ - * / = % $ ...), "punct", "entity"
(capitalised and not sentence-initial: a crude named-entity heuristic) or "content".

Output (.json), one record per statement:
  id, benchmark, field, half,
  words    {lang: [[word, start, end], ...]}   character spans into Statement.text[lang]
  types    [type of each English word]
  align    {method: {lang: [[i_en, j_lang], ...]}}   raw SimAlign output
  pairs    {method: {lang: [[i_en, j_lang], ...]}}   kept one-to-one pairs (step 3)
  concepts {method: [[word index in each of LANGS], ...]}

Usage (GPU job, via slurm/submit.sh):
    slurm/submit.sh -J align -- python src/experiments/word_level_analysis/align.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import regex

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pca_analysis"))
from data import LANGS, MAX_PER_BENCHMARK, load_statements  # noqa: E402

METHODS = ("inter", "itermax", "mwmf")
TOKEN = regex.compile(
    r"(?P<han>[\p{Han}]+)|(?P<thai>[\p{Thai}]+)"
    r"|\d+(?:[.,:]\d+)*|[^\W\p{Han}\p{Thai}]+(?:[-'’][^\W\p{Han}\p{Thai}]+)*|[^\w\s]")
MATH = set("+-−*/=%$€£×÷<>^")


def segment(text, lang):
    """[(word, start, end)] for `text`, with Han / Thai runs split by the language's segmenter."""
    out = []
    for m in TOKEN.finditer(text):
        script = m.lastgroup
        if script is None or (script, lang) not in {("han", "zh"), ("thai", "th")}:
            out.append((m.group(), m.start(), m.end()))
            continue
        if script == "han":
            import jieba
            pieces = jieba.cut(m.group())
        else:
            from pythainlp.tokenize import word_tokenize
            pieces = word_tokenize(m.group(), engine="newmm", keep_whitespace=False)
        pos = m.start()
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            start = text.index(p, pos)
            out.append((p, start, start + len(p)))
            pos = start + len(p)
    return out


def word_type(words, i):
    w = words[i][0]
    if regex.search(r"\d", w):
        return "number"
    if not regex.search(r"\w", w):
        return "math" if w in MATH else "punct"
    sentence_start = i == 0 or words[i - 1][0] in {".", "?", "!", ":", '"', "“"}
    return "entity" if w[0].isupper() and not sentence_start else "content"


def one_to_one(pairs):
    """{i: j} for the pairs (i, j) in which both i and j occur exactly once."""
    ci, cj = Counter(i for i, _ in pairs), Counter(j for _, j in pairs)
    return {i: j for i, j in pairs if ci[i] == 1 and cj[j] == 1}


def kept_pairs(align, words, types):
    """Step 3: one-to-one pairs, with punctuation only paired with punctuation."""
    return [[i, j] for i, j in sorted(one_to_one(align).items())
            if types[i] != "punct" or not regex.search(r"\w", words[j][0])]


def concepts(n_en, pairs, langs):
    maps = {L: dict(pairs[L]) for L in langs if L != "en"}
    return [[i if L == "en" else maps[L][i] for L in langs]
            for i in range(n_en) if all(i in m for m in maps.values())]


def max_weight_match(sim):
    """SimAlign's "mwmf" without networkx (pure Python, ~10 s per statement): all similarities
    are positive, so the max-cardinality max-weight bipartite matching is an assignment."""
    from scipy.optimize import linear_sum_assignment
    out = np.zeros_like(sim)
    out[linear_sum_assignment(sim, maximize=True)] = 1
    return out


def align_statements(statements, aligner, langs):
    tokenize = aligner.embed_loader.tokenizer.tokenize
    records = []
    for k, s in enumerate(statements):
        words = {L: segment(s.text[L], L) for L in langs}
        # SimAlign silently mis-maps subwords to words if a word has no mBERT subwords.
        words = {L: [w for w in ws if tokenize(w[0])] for L, ws in words.items()}
        en = [w for w, _, _ in words["en"]]
        align = {m: {} for m in METHODS}
        for L in langs:
            if L == "en":
                continue
            result = aligner.get_word_aligns(en, [w for w, _, _ in words[L]])
            for m in METHODS:
                align[m][L] = [list(p) for p in result[m]]
        types = [word_type(words["en"], i) for i in range(len(en))]
        pairs = {m: {L: kept_pairs(align[m][L], words[L], types) for L in align[m]} for m in METHODS}
        records.append({
            "id": s.id, "benchmark": s.benchmark, "field": s.field, "half": s.half,
            "words": {L: [list(w) for w in ws] for L, ws in words.items()},
            "types": types,
            "align": align,
            "pairs": pairs,
            "concepts": {m: concepts(len(en), pairs[m], langs) for m in METHODS},
        })
        print(f"\r{k + 1}/{len(statements)} statements", end="", flush=True)
    print()
    return records


def summary(records, langs):
    """Concept yield: overall, by benchmark, by word type, and per language (one-to-one rate)."""
    n_en = sum(len(r["types"]) for r in records)
    print(f"\n{len(records)} statements, {n_en} English words")
    for m in METHODS:
        n = sum(len(r["concepts"][m]) for r in records)
        print(f"\n== {m}: {n} concepts ({n / n_en:.1%} of English words)")
        by_bench = Counter()
        for r in records:
            by_bench[r["benchmark"], "words"] += len(r["types"])
            by_bench[r["benchmark"], "concepts"] += len(r["concepts"][m])
        print("  by benchmark: " + ", ".join(
            f"{b} {by_bench[b, 'concepts']} ({by_bench[b, 'concepts'] / by_bench[b, 'words']:.0%})"
            for b in sorted({b for b, _ in by_bench})))
        all_types = Counter(t for r in records for t in r["types"])
        kept = Counter(r["types"][c[0]] for r in records for c in r["concepts"][m])
        print("  by type: " + ", ".join(f"{t} {kept[t]}/{all_types[t]}" for t in sorted(all_types)))
        rate = {L: sum(len(r["pairs"][m][L]) for r in records) / n_en for L in langs if L != "en"}
        print("  English words with a kept pair, per language: " + ", ".join(f"{L} {v:.0%}" for L, v in rate.items()))
        halves = Counter(r["half"] for r in records for _ in r["concepts"][m])
        print(f"  fit {halves['fit']} / test {halves['test']}")


def show(records, langs, method="inter", n=20):
    """Print random content-word concepts to eyeball the alignment quality."""
    print(f"\nrandom content-word concepts ({method}):")
    pool = [(r, c) for r in records for c in r["concepts"][method] if r["types"][c[0]] in ("content", "entity")]
    for i in np.random.default_rng(0).choice(len(pool), min(n, len(pool)), replace=False):
        r, c = pool[i]
        print(f"  {r['benchmark']:>6}: " + " | ".join(r["words"][L][j][0] for L, j in zip(langs, c)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results" / "alignments_simalign.json")
    parser.add_argument("--max-per-benchmark", type=int, default=MAX_PER_BENCHMARK)
    parser.add_argument("--limit", type=int, help="only the first N statements (for quick tests)")
    args = parser.parse_args()

    import torch
    from simalign import SentenceAligner

    statements = load_statements(max_per_benchmark=args.max_per_benchmark)[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SentenceAligner.get_max_weight_match = staticmethod(max_weight_match)
    aligner = SentenceAligner(model="google-bert/bert-base-multilingual-cased", token_type="bpe", matching_methods="mai", device=device)
    records = align_statements(statements, aligner, LANGS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"langs": list(LANGS), "aligner": "simalign/mbert-layer8",
                                    "records": records}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    summary(records, LANGS)
    show(records, LANGS)


if __name__ == "__main__":
    main()
