"""Score generations with the Language Confusion Benchmark metrics, choose steering settings, compare conditions.

A port of the benchmark's compute_metrics.py (for-ai/language-confusion@0cfdbb4a), same rules:
  * strip punctuation, split into lines, keep lines with >= 5 tokens (jieba for zh, fugashi for ja,
    whitespace otherwise); a response with no such line is skipped;
  * line-level pass (LPR): fastText lid.176.bin labels every kept line as T with probability > 0.3;
  * word-level pass (WPR, non-Latin-script languages only): among line-level passes, no token is an
    English word from the benchmark's word list (lower-case, > 3 letters).
A language's LPR and WPR are the means over its three sources, as in the benchmark. Our own measures:
  * English: the response falls back to English (every kept line, or its first kept line, is English);
    this separates real confusion from line-level errors caused by quoted names, tables or code;
  * degenerate: no kept line, a substring of >= 4 characters repeated >= 4 times in a row (a loop), or a
    thinking tag (<think> or </think>: Qwen3 runs with thinking off, so its prompt ends with an empty thinking
    block, and strong prompt-only steering can make it reopen one and reason in the answer);
  * thinking: the share of responses with a thinking tag (counted in degenerate);
  * drift: among responses whose first kept line is T, the share with a later English line.
Selection (dev, per steering condition): among layer x alpha settings whose degenerate rate is at most the
instruction's + 0.05, the highest LPR; ties go to the lower degenerate rate, then the smaller alpha, then
the earlier layer. The universal setting, shared by all languages, applies the same rule to the mean LPR and
degenerate rate over the languages, among settings within the degenerate limit in every language.
Test comparison with the instruction: paired over prompts, exact McNemar test on line-level passes.

The model is set by the environment variable LCB_MODEL, as in generate.py; results are read from and written to
results/<LCB_MODEL>/.

Usage: LCB_MODEL=<model> python src/experiments/language_confusion/analyse.py   # main comparison -> summary.json
       LCB_MODEL=<model> python src/experiments/language_confusion/analyse.py --prompt-only   # -> summary_prompt.json
       LCB_MODEL=<model> python src/experiments/language_confusion/analyse.py --pilot  # 14 languages -> summary_pilot.json
"""
import argparse
import collections
import functools
import json
import math
import os
import re
import string
import sys
from pathlib import Path

import fasttext

import data

# Hugging Face id and steering-layer grid (1-indexed blocks, about 25-75% of the depth; the grids of
# ../mgsm_steering/generate.py) of each model.
MODELS = {"llama-3.1-8b": dict(hf="meta-llama/Llama-3.1-8B-Instruct", layers=(8, 12, 16, 20, 24)),
          "qwen3-8b": dict(hf="Qwen/Qwen3-8B", layers=(8, 12, 16, 20, 24, 28)),
          "qwen3-32b": dict(hf="Qwen/Qwen3-32B", layers=(16, 24, 32, 40, 48))}
MODEL = os.environ.get("LCB_MODEL")
if MODEL not in MODELS:
    sys.exit(f"set LCB_MODEL to one of {', '.join(MODELS)}")
RESULTS = Path(__file__).resolve().parent / "results" / MODEL
GEN = RESULTS / "generations"
LOOP = re.compile(r"(.{4,}?)\1{3,}", re.S)
MAX_EXTRA_DEGENERATE = 0.05
NON_LATIN = ("ar", "hi", "ja", "ko", "ru", "zh")

fasttext.FastText.eprint = lambda *a, **k: None  # silence the load_model deprecation notice
LID = fasttext.load_model(str(data.ROOT / "data/raw_data/fasttext_bin/lid.176.bin"))
EN_WORDS = {w for w in (l.strip() for l in open(data.ROOT / "data/raw_data/lcb_words/words")) if w.islower() and len(w) > 3}


def normalize(text):
    text = text.split("\nQ:")[0].strip()
    return text.translate(str.maketrans("", "", string.punctuation)).replace("—", " ").replace("،", "")


@functools.cache
def tokenizer(lang):
    if lang == "zh":
        import jieba
        jieba.setLogLevel(60)
        return lambda s: list(jieba.cut(s))
    if lang == "ja":
        from fugashi import Tagger
        t = Tagger("-O wakati -b 50000")
        return lambda s: t.parse(s).split()
    return str.split


def langid(line):
    # what FastText.predict does (it appends "\n"), minus its NumPy-2-incompatible np.array(..., copy=False)
    p, label = LID.f.predict(line + "\n", 1, 0.0, "strict")[0]
    return label.removeprefix("__label__") if p > 0.3 else "unknown"


def judge(response, lang):
    """None (skipped: no line of >= 5 tokens), 'line' (line-level error), 'word' (word-level error) or 'pass'."""
    lines = normalize(response).split("\n")
    toks = [tokenizer(lang)(l) for l in lines]
    kept = [(l, t) for l, t in zip(lines, toks) if len(t) >= 5]
    if not kept:
        return None
    if any(langid(l) != lang for l, _ in kept):
        return "line"
    if any(w.strip() in EN_WORDS for _, t in kept for w in t):
        return "word"
    return "pass"


def english(response, lang):
    lines = [l for l in normalize(response).split("\n") if len(tokenizer(lang)(l)) >= 5]
    labels = [langid(l) for l in lines]
    return bool(labels) and labels[0] == "en"  # covers "all lines English" too


def drift(response, lang):
    """None if the first kept line is not T; else whether a later kept line is English."""
    lines = [l for l in normalize(response).split("\n") if len(tokenizer(lang)(l)) >= 5]
    labels = [langid(l) for l in lines]
    return any(x == "en" for x in labels[1:]) if labels and labels[0] == lang else None


def thinking(response):
    return "<think>" in response or "</think>" in response


def degenerate(response, lang):
    return judge(response, lang) is None or bool(LOOP.search(response)) or thinking(response)


def rates(verdicts, lang):
    v = [x for x in verdicts if x is not None]
    d = dict(n=len(verdicts), skipped=len(verdicts) - len(v), lpr=sum(x != "line" for x in v) / max(1, len(v)))
    if lang in NON_LATIN:
        ok = [x for x in v if x != "line"]
        d["wpr"] = sum(x == "pass" for x in ok) / max(1, len(ok))
    return d


def score(rows):
    """All measures for rows of one condition and language (LPR and WPR as means over sources)."""
    L = rows[0]["lang"]
    by = {s: rates([judge(r["response"], L) for r in rows if r["source"] == s], L) for s in data.SOURCES}
    d = {k: sum(by[s][k] for s in data.SOURCES) / len(data.SOURCES) for k in ("lpr", "wpr") if k in by["okapi"]}
    d.update(n=len(rows), english=sum(english(r["response"], L) for r in rows) / len(rows),
             degenerate=sum(degenerate(r["response"], L) for r in rows) / len(rows),
             thinking=sum(thinking(r["response"]) for r in rows) / len(rows),
             drift=(lambda d: sum(d) / max(1, len(d)))([x for r in rows if (x := drift(r["response"], L)) is not None]),
             lpr_by_source={s: by[s]["lpr"] for s in data.SOURCES})
    return d


def configs(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["cond"], r.get("layer"), r.get("alpha"))].append(r)
    return g


def dev_scores(rows):
    """{(cond, layer, alpha): score} for the dev rows of one language."""
    return {k: score(v) for k, v in configs(rows).items()}


def select(scores):
    """{steering condition: (layer, alpha)} chosen on the dev scores of one language."""
    return select_universal({None: scores})


def select_universal(scores_by_lang):
    """{steering condition: (layer, alpha)} shared by the languages {lang: dev scores}."""
    S = list(scores_by_lang.values())
    mean = lambda k, m: sum(s[k][m] for s in S) / len(S)
    within = lambda k: all(s[k]["degenerate"] <= s[("instruction", None, None)]["degenerate"] + MAX_EXTRA_DEGENERATE
                           for s in S)
    chosen = {}
    for cond in ("steer", "instruction_steer"):
        keys = [k for k in S[0] if k[0] == cond]
        ok = [k for k in keys if within(k)] or [min(keys, key=lambda k: mean(k, "degenerate"))]
        best = min(ok, key=lambda k: (-mean(k, "lpr"), mean(k, "degenerate"), k[2], k[1]))
        chosen[cond] = (best[1], best[2])
    return chosen


def passes(r):
    v = judge(r["response"], r["lang"])
    return None if v is None else v != "line"


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired booleans (None pairs dropped); returns (b_only, a_only, p)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n01, n10 = sum((not x) and y for x, y in pairs), sum(x and not y for x, y in pairs)
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p


def pilot():
    rows = [json.loads(l) for l in open(GEN / "pilot_dev.jsonl")]
    out = {}
    print("instruction, dev: LPR overall and per source, WPR for non-Latin scripts, English fallback, degenerate")
    print(f"{'':4}{'LPR':>6}" + "".join(f"{s:>17}" for s in data.SOURCES) + f"{'WPR':>6}{'Eng':>6}{'degen':>7}")
    for L in data.LANGS:
        d = out[L] = score([r for r in rows if r["lang"] == L])
        wpr = f"{d['wpr']:.2f}" if "wpr" in d else "-"
        print(f"{L:4}{d['lpr']:>6.2f}" + "".join(f"{d['lpr_by_source'][s]:>17.2f}" for s in data.SOURCES)
              + f"{wpr:>6}{d['english']:>6.2f}{d['degenerate']:>7.2f}")
    (RESULTS / "summary_pilot.json").write_text(json.dumps(out, indent=1) + "\n")


def compare(test, title):
    """Score the test conditions {(cond, layer, alpha): rows} of one language and test each against the instruction."""
    out = {}
    base = test[("instruction", None, None)]
    print(f"\n== {title}  (test, n = {len(base)})")
    print(f"{'':19}{'LPR':>6}{'complex':>9}{'WPR':>6}{'English':>9}{'degen':>7}   vs instruction (McNemar)")
    for k, rows in sorted(test.items(), key=lambda kv: ["instruction", "steer", "instruction_steer"].index(kv[0][0])):
        d = out[k[0]] = score(rows)
        cmp = ""
        if k[0] != "instruction":
            assert [r["id"] for r in rows] == [r["id"] for r in base]
            n01, n10, p = mcnemar([passes(r) for r in base], [passes(r) for r in rows])
            d.update(layer=k[1], alpha=k[2], gained=n01, lost=n10, p=p)
            cmp = f"   +{n01} / -{n10}, p = {p:.3f}   (layer {k[1]}, alpha {k[2]})"
        wpr = f"{d['wpr']:.2f}" if "wpr" in d else "-"
        print(f"{k[0]:19}{d['lpr']:>6.2f}{d['lpr_by_source']['complex_prompts']:>9.2f}{wpr:>6}"
              f"{d['english']:>9.2f}{d['degenerate']:>7.2f}{cmp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--prompt-only", action="store_true")
    args = ap.parse_args()
    if args.pilot:
        return pilot()
    sfx = "_prompt" if args.prompt_only else ""
    langs = sorted(f.name.split("_")[0] for f in GEN.glob(f"*_test{sfx}.jsonl"))
    dev = {L: dev_scores([json.loads(l) for l in open(GEN / f"{L}_dev{sfx}.jsonl")]) for L in langs}
    universal = select_universal(dev)
    print("universal setting:", universal)
    out = {}
    for L in langs:
        res = out[L] = {"chosen": {c: dict(layer=l, alpha=a) for c, (l, a) in select(dev[L]).items()},
                        "dev_grid": [dict(cond=k[0], layer=k[1], alpha=k[2], **{m: v[m] for m in ("lpr", "english", "degenerate", "thinking")})
                                     for k, v in dev[L].items()]}
        res["test"] = compare(configs([json.loads(l) for l in open(GEN / f"{L}_test{sfx}.jsonl")]), L)
        if (GEN / f"{L}_test_universal{sfx}.jsonl").exists():
            res["test_universal"] = compare(configs([json.loads(l) for l in open(GEN / f"{L}_test_universal{sfx}.jsonl")]),
                                            f"{L}, universal setting")
    (RESULTS / f"summary{sfx}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
