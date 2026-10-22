"""Basic statistics of the processed benchmarks.

Prints markdown tables and writes data/processed_data/stats.json.

Per benchmark and language:
  * chars      - mean characters per item (all text fields joined)
  * words      - mean whitespace-separated words (meaningless for zh/ja/th, which
                 do not use spaces)
  * len/en     - mean per-item character-length ratio to the English version
  * same_as_en - share of items whose text is identical to English (untranslated)
  * digits_ok  - share of items whose multiset of numbers (written in digits,
                 any Unicode script) equals the English one; a rough translation
                 quality check, deflated where English spells numbers out ("three")
  * dup        - number of duplicated texts within the language
  * flagged    - items flagged as probably broken by preprocess.add_quality_flags

Usage: python src/data/analyze.py
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed_data"
EXP1_LANGS = ["en", "zh", "fr", "de", "ru", "es", "sw", "th"]

THOUSANDS_SEP = re.compile(r"(?<=\d)[,.\s  '](?=\d{3}(?!\d))")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def load(name):
    with open(PROC / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def joined(item, lang):
    return " ".join(item["text"][lang].values())


def numbers(s):
    s = "".join(str(unicodedata.decimal(c)) if c.isdigit() and not c.isascii() else c for c in s)
    s = THOUSANDS_SEP.sub("", s)
    return Counter(float(m.replace(",", ".")) for m in NUMBER.findall(s))


def language_stats(items, lang):
    texts = [joined(it, lang) for it in items]
    en = [joined(it, "en") for it in items]
    return {
        "items": len(texts),
        "chars": float(np.mean([len(t) for t in texts])),
        "words": float(np.mean([len(t.split()) for t in texts])),
        "len_ratio_en": float(np.mean([len(t) / len(e) for t, e in zip(texts, en)])),
        "same_as_en": float(np.mean([t == e for t, e in zip(texts, en)])) if lang != "en" else None,
        "digits_ok": float(np.mean([numbers(t) == numbers(e) for t, e in zip(texts, en)])) if lang != "en" else None,
        "dup": len(texts) - len(set(texts)),
        "flagged": sum(lang in it.get("flags", {}) for it in items),
    }


def fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.1f}" if isinstance(x, float) else str(x)


def table(name, stats):
    print(f"\n### {name}\n")
    print("| lang | items | chars | words | len/en | same_as_en | digits_ok | dup | flagged |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lang, s in stats.items():
        star = "*" if lang in EXP1_LANGS else ""
        print(f"| {lang}{star} | {s['items']} | {fmt(s['chars'])} | {fmt(s['words'])} | {s['len_ratio_en']:.2f} "
              f"| {fmt(s['same_as_en'], True)} | {fmt(s['digits_ok'], True)} | {s['dup']} | {s['flagged']} |")


def main():
    out = {}
    for name in ["mgsm", "msvamp", "xnli"]:
        items = load(name)
        if name == "xnli":
            items = [it for it in items if it["split"] != "unused"]
        langs = list(items[0]["text"])
        out[name] = {
            "splits": dict(Counter(it["split"] for it in items)),
            "languages": {L: language_stats(items, L) for L in langs},
        }
        if name == "xnli":
            out[name]["labels"] = {
                sp: dict(Counter(it["label"] for it in items if it["split"] == sp))
                for sp in ["analysis", "compute", "val", "test"]
            }
        else:
            answers = [it["answer"] for it in items]
            out[name]["answers"] = {
                "min": min(answers), "median": float(np.median(answers)), "max": max(answers),
                "non_integer": sum(not float(a).is_integer() for a in answers),
            }
        table(name, out[name]["languages"])
        print(f"\nsplits: {out[name]['splits']}")
        print(f"labels: {out[name]['labels']}" if name == "xnli" else f"answers: {out[name]['answers']}")

    msvamp = load("msvamp")
    for key, texts in [("original", [it["en_question_original"] for it in msvamp]),
                       ("truecased", [it["text"]["en"]["question"] for it in msvamp])]:
        lower = float(np.mean([q == q.lower() for q in texts]))
        spaced = float(np.mean([bool(re.search(r"\w [.,?!]( |$)", q)) for q in texts]))
        out["msvamp"][f"en_{key}"] = {"lowercase_share": lower, "spaced_punct_share": spaced}
        print(f"\nMSVAMP English ({key}): {100 * lower:.1f}% fully lowercase, "
              f"{100 * spaced:.1f}% with space-separated punctuation")

    with open(PROC / "stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n(* = EXP1 language) stats -> {(PROC / 'stats.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
