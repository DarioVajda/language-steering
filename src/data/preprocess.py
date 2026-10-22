"""Convert the raw benchmarks into one parallel JSONL file per benchmark.

Every output row is one item with its text in all available languages:

    {"id": "mgsm-0000", "split": "compute", "answer": 18,
     "text": {"en": {"question": "..."}, "zh": {"question": "..."}, ...}}

XNLI rows carry {"premise", "hypothesis"} per language and a "label" field
instead of "answer". In MSVAMP, the English text is truecased and detokenised
(see truecase.py); the original lowercased source is kept in "en_question_original".
Rows with probably broken translations also carry
"flags": {lang: [reasons]} (see `add_quality_flags`). Items are parallel by index in all three raw datasets
(checked in `check_parallel`).

Splits mirror the sizes in Kirtane & Huang (2026), Table 14. Their exact item
IDs are not public, so these are a fixed-seed random partition:
  * MGSM (250 test items)    -> compute 83 / val 83 / test 84
  * MSVAMP (1000 test items) -> compute 333 / val 333 / test 334
  * XNLI test (5010 items)   -> random 1000-item subsample -> ~333 / ~333 / ~334,
                                split by premise so no premise crosses splits;
                                the remaining 4010 test items get split "unused".
  * XNLI validation (2490)   -> split "analysis" (disjoint from all ICL splits;
                                intended for EXP1).

Outputs (data/processed_data/):
  mgsm.jsonl, mgsm_exemplars.jsonl, msvamp.jsonl, xnli.jsonl

Usage: python src/data/preprocess.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from truecase import match_svamp, proper_nouns, truecase

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_data"
OUT = ROOT / "data" / "processed_data"
SEED = 0

MGSM_LANGS = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MSVAMP_LANGS = {
    "en": "English", "bn": "Bengali", "de": "German", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "ru": "Russian", "sw": "Swahili", "th": "Thai", "zh": "Chinese",
}
XNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def three_way_split(n, rng):
    """Assign compute/val/test to n items, with sizes floor(n/3), floor(n/3), rest."""
    k = n // 3
    names = np.array(["compute"] * k + ["val"] * k + ["test"] * (n - 2 * k))
    return names[rng.permutation(n).argsort()]


def number(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def add_quality_flags(rows):
    """Flag translations that are probably broken (machine-translation failures).

    A (item, language) pair is flagged when its character-length ratio to English
    is an outlier for that language (< 0.4x or > 2.5x the language's median ratio:
    truncated or hallucinated output), or when it is a degenerate repetition loop
    (a single word making up > 50% of >= 10 words). Flags are stored as
    row["flags"] = {lang: [reasons]}; rows are kept, filtering is left to the user.
    """
    def joined(row, lang):
        return " ".join(row["text"][lang].values())

    for lang in rows[0]["text"]:
        if lang == "en":
            continue
        ratios = np.array([len(joined(r, lang)) / len(joined(r, "en")) for r in rows])
        median = np.median(ratios)
        for r, ratio in zip(rows, ratios):
            words = joined(r, lang).split()
            reasons = []
            if ratio < 0.4 * median:
                reasons.append("too_short")
            if ratio > 2.5 * median:
                reasons.append("too_long")
            if len(words) >= 10 and max(Counter(words).values()) / len(words) > 0.5:
                reasons.append("repetition")
            if reasons:
                r.setdefault("flags", {})[lang] = reasons
    return rows


def write_jsonl(rows, path):
    rows = add_quality_flags(rows)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>5} rows -> {path.relative_to(ROOT)}")


def check_parallel(name, answers_by_lang):
    ref = answers_by_lang["en"]
    for lang, ans in answers_by_lang.items():
        assert len(ans) == len(ref) and list(ans) == list(ref), f"{name}: {lang} not parallel with en"


def mgsm(rng):
    test = {L: pd.read_parquet(RAW / f"mgsm/{L}/test-00000-of-00001.parquet") for L in MGSM_LANGS}
    check_parallel("mgsm", {L: d.answer_number for L, d in test.items()})
    splits = three_way_split(len(test["en"]), rng)
    rows = [
        {
            "id": f"mgsm-{i:04d}",
            "split": splits[i],
            "answer": number(test["en"].answer_number[i]),
            "text": {L: {"question": test[L].question[i]} for L in MGSM_LANGS},
        }
        for i in range(len(test["en"]))
    ]
    write_jsonl(rows, OUT / "mgsm.jsonl")

    # The 8 official chain-of-thought exemplars (MGSM "train" split), kept verbatim,
    # including their localised "Question:" / "Step-by-Step Answer:" prefixes.
    train = {L: pd.read_parquet(RAW / f"mgsm/{L}/train-00000-of-00001.parquet") for L in MGSM_LANGS}
    check_parallel("mgsm exemplars", {L: d.answer_number for L, d in train.items()})
    rows = [
        {
            "id": f"mgsm-exemplar-{i}",
            "answer": number(train["en"].answer_number[i]),
            "text": {L: {"question": train[L].question[i], "answer": train[L].answer[i]} for L in MGSM_LANGS},
        }
        for i in range(len(train["en"]))
    ]
    write_jsonl(rows, OUT / "mgsm_exemplars.jsonl")


def msvamp(rng):
    raw = {}
    for L, name in MSVAMP_LANGS.items():
        with open(RAW / f"msvamp/test_{name}.json", encoding="utf-8") as f:
            raw[L] = [json.loads(line) for line in f]
    check_parallel("msvamp", {L: [r["response"] for r in rs] for L, rs in raw.items()})
    # Non-English files also carry the English source in "query"; it must match en.
    for L, rs in raw.items():
        assert all(r.get("query", r["m_query"]) == e["m_query"] for r, e in zip(rs, raw["en"])), L
    original = [r["m_query"] for r in raw["en"]]
    with open(RAW / "svamp/SVAMP.json", encoding="utf-8") as f:
        svamp = json.load(f)
    references, names = match_svamp(original, svamp), proper_nouns(svamp)
    truecased = [truecase(q, ref, names)[0] for q, (ref, _) in zip(original, references)]
    splits = three_way_split(len(raw["en"]), rng)
    rows = [
        {
            "id": f"msvamp-{i:04d}",
            "split": splits[i],
            "answer": number(raw["en"][i]["response"]),
            "equation": raw["en"][i]["equation"],
            "en_question_original": original[i],
            "text": {L: {"question": truecased[i] if L == "en" else raw[L][i]["m_query"]} for L in MSVAMP_LANGS},
        }
        for i in range(len(raw["en"]))
    ]
    write_jsonl(rows, OUT / "msvamp.jsonl")


def xnli(rng):
    rows = []
    for raw_split in ["validation", "test"]:
        df = pd.read_parquet(RAW / f"xnli/all_languages/{raw_split}-00000-of-00001.parquet")
        n = len(df)
        if raw_split == "validation":
            splits = np.array(["analysis"] * n)
        else:
            # Each premise has 3 hypotheses; split by premise so that no premise is
            # shared between compute/val/test (sizes are then only approximately 1/3).
            splits = np.array(["unused"] * n, dtype=object)
            subsample = np.sort(rng.choice(n, size=1000, replace=False))
            premises = df.premise.map(lambda p: p["en"]).to_numpy()
            groups = rng.permutation(np.unique(premises[subsample]))
            counts = pd.Series(premises[subsample]).value_counts()[groups].cumsum().to_numpy()
            group_split = np.where(counts <= 333, "compute", np.where(counts <= 666, "val", "test"))
            lookup = dict(zip(groups, group_split))
            splits[subsample] = [lookup[premises[i]] for i in subsample]
        for i, r in enumerate(df.itertuples()):
            hyp = dict(zip(r.hypothesis["language"], r.hypothesis["translation"]))
            assert set(hyp) == set(r.premise), f"xnli {raw_split}[{i}]: language sets differ"
            rows.append({
                "id": f"xnli-{raw_split}-{i:04d}",
                "split": splits[i],
                "label": XNLI_LABELS[r.label],
                "text": {L: {"premise": r.premise[L], "hypothesis": hyp[L]} for L in sorted(r.premise)},
            })
    write_jsonl(rows, OUT / "xnli.jsonl")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mgsm(np.random.default_rng(SEED))
    msvamp(np.random.default_rng(SEED))
    xnli(np.random.default_rng(SEED))


if __name__ == "__main__":
    main()
