"""Score MGSM generations, choose the steering settings on dev, compare the arms on test.

Main measure, "success": the answer is correct AND the response is in the target language T.
  * correct: tasks.MGSM.parse (the number after the last "Final answer:", else the last number; locale-aware,
    native digits) equals the gold answer;
  * in T: every character that is a digit, punctuation or symbol becomes a space (equations and amounts carry
    no language), then
      - fastText lid.176.bin labels the whole response (its lines joined) as T with probability > 0.3, and
      - for T != English, no line of >= 5 words is labelled English with probability > 0.5 (a switch to English
        part-way). Words come from jieba (zh), fugashi (ja), pythainlp (th), else whitespace, where a word
        must have >= 2 letters (so variable names and units like "x" or "h" do not make a line).
    Line-level labels alone (as in the Language Confusion Benchmark) are not used: fastText often gives short
    Swahili lines a wrong or low-confidence label, so it rejected correct Swahili answers.
Also reported: accuracy, in-T rate, English (the whole response labelled English), capped (hit 2048 tokens).
Selection (dev, per steering arm): the (layer, alpha) with the highest success rate averaged over the 10
languages; ties go to the smaller alpha, then the earlier layer.
Test comparison with the instruction: paired over items within a language, exact McNemar test on success.
Both steps first wait until every unit of their stage has been generated (by the workers of any job).

Usage: python src/experiments/mgsm_steering/analyse.py --select   # dev -> results/generations/chosen.json
       python src/experiments/mgsm_steering/analyse.py            # test -> results/summary.json
"""
import argparse
import collections
import functools
import json
import math
import time
import unicodedata

import fasttext

from generate import GEN, HERE, LANGS, ROOT, STEER_ARMS, units

fasttext.FastText.eprint = lambda *a, **k: None  # silence the load_model deprecation notice
LID = fasttext.load_model(str(ROOT / "data/raw_data/fasttext_bin/lid.176.bin"))
ARMS = ("native", "instruction") + STEER_ARMS


@functools.cache
def words(lang):
    if lang == "zh":
        import jieba
        jieba.setLogLevel(60)
        return lambda s: [w for w in jieba.cut(s) if w.strip()]
    if lang == "ja":
        from fugashi import Tagger
        t = Tagger("-O wakati -b 50000")
        return lambda s: t.parse(s).split()
    if lang == "th":
        from pythainlp.tokenize import word_tokenize
        return lambda s: [w for w in word_tokenize(s) if w.strip()]
    return lambda s: [w for w in s.split() if len(w) >= 2]


def lid(text):
    """(label, probability); what FastText.predict does, minus its NumPy-2-incompatible array copy."""
    p, label = LID.f.predict(text + "\n", 1, 0.0, "strict")[0]
    return label.removeprefix("__label__"), p


def letters(line):
    return "".join(" " if unicodedata.category(ch)[0] in "NPS" else ch for ch in line)


@functools.cache
def judge(response, lang):
    """(in T, whole response labelled English)."""
    lines = [" ".join(letters(l).split()) for l in response.split("\n")]
    lines = [l for l in lines if l]
    if not lines:
        return False, False
    label, p = lid(" ".join(lines))
    in_t = label == lang and p > 0.3
    if in_t and lang != "en":
        in_t = not any(len(words(lang)(l)) >= 5 and (x := lid(l))[0] == "en" and x[1] > 0.5 for l in lines)
    return in_t, label == "en" and p > 0.3


def correct(r):
    return r["pred"] is not None and abs(r["pred"] - r["answer"]) < 1e-6


def success(r):
    return correct(r) and judge(r["response"], r["lang"])[0]


def score(rows):
    n = len(rows)
    j = [judge(r["response"], r["lang"]) for r in rows]
    return dict(n=n, success=sum(map(success, rows)) / n, accuracy=sum(map(correct, rows)) / n,
                in_lang=sum(x[0] for x in j) / n, english=sum(x[1] for x in j) / n,
                capped=sum(r["capped"] for r in rows) / n,
                median_tokens=sorted(r["n_tokens"] for r in rows)[n // 2])


def load(path):
    return [json.loads(l) for l in open(path)]


def by_lang(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["lang"]].append(r)
    return g


def wait_for(stage):
    """Wait until every unit of the stage has its output (units may still be running in other jobs)."""
    while missing := [n for n in units(stage) if not (GEN / stage / f"{n}.jsonl").exists()]:
        print(f"waiting for {len(missing)} {stage} units", flush=True)
        time.sleep(30)


def select():
    """{steering arm: {layer, alpha, mean_success}} chosen on dev; prints the dev scores."""
    wait_for("dev")
    dev = {f.stem: by_lang(load(f)) for f in sorted((GEN / "dev").glob("*.jsonl"))}
    print("dev, success / accuracy / in T per language")
    print(f"{'':12}" + "".join(f"{L:>16}" for L in LANGS))
    for arm in ("native", "instruction"):
        s = {L: score(dev[arm][L]) for L in LANGS}
        print(f"{arm:12}" + "".join(f"  {s[L]['success']:.2f} {s[L]['accuracy']:.2f} {s[L]['in_lang']:.2f}" for L in LANGS))
    e = score(dev["english"]["en"])
    print(f"english: accuracy {e['accuracy']:.2f}, in English {e['in_lang']:.2f}")
    chosen, grid = {}, {}
    for arm in STEER_ARMS:
        means = {}
        for name, g in dev.items():
            r0 = next(iter(g.values()))[0]
            if r0["arm"] == arm:
                means[(r0["layer"], r0["alpha"])] = sum(score(g[L])["success"] for L in LANGS) / len(LANGS)
        (layer, alpha) = min(means, key=lambda k: (-means[k], k[1], k[0]))
        chosen[arm] = dict(layer=layer, alpha=alpha, mean_success=means[(layer, alpha)])
        grid[arm] = [dict(layer=l, alpha=a, mean_success=m) for (l, a), m in sorted(means.items())]
        print(f"\n{arm}: mean dev success over languages (rows: layer, columns: alpha)")
        alphas = sorted({a for _, a in means})
        print(f"{'':6}" + "".join(f"{a:>7}" for a in alphas))
        for l in sorted({l for l, _ in means}):
            print(f"{l:>6}" + "".join(f"{means[(l, a)]:>7.3f}" for a in alphas))
        print(f"chosen: layer {layer}, alpha {alpha} ({means[(layer, alpha)]:.3f})")
    (GEN / "chosen.json").write_text(json.dumps(chosen, indent=1) + "\n")
    return chosen, grid


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired booleans; returns (b_only, a_only, p)."""
    n01, n10 = sum((not x) and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b))
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    if args.select:
        select()
        return
    chosen, grid = select()
    wait_for("test")
    T = GEN / "test"
    out = {"chosen": chosen, "dev_grid": grid, "english": score(load(T / "english.jsonl")), "test": {}}
    print(f"\n== test; english: accuracy {out['english']['accuracy']:.2f}")
    cols = ("success", "accuracy", "in_lang", "english", "capped")
    for L in LANGS:
        rows = {arm: load(T / f"{arm}_{L}.jsonl") for arm in ARMS}
        base = rows["instruction"]
        print(f"\n== {L}  (test, n = {len(base)})")
        print(f"{'':26}" + "".join(f"{c:>9}" for c in cols) + f"{'tokens':>8}   success vs instruction (McNemar)")
        res = out["test"][L] = {}
        for arm in ARMS:
            d = res[arm] = score(rows[arm])
            cmp = ""
            if arm != "instruction":
                assert [r["id"] for r in rows[arm]] == [r["id"] for r in base]
                n01, n10, p = mcnemar([success(r) for r in base], [success(r) for r in rows[arm]])
                d.update(gained=n01, lost=n10, p=p)
                cmp = f"   +{n01} / -{n10}, p = {p:.3f}"
            if arm in STEER_ARMS:
                d.update(layer=chosen[arm]["layer"], alpha=chosen[arm]["alpha"])
            print(f"{arm:26}" + "".join(f"{d[c]:>9.2f}" for c in cols) + f"{d['median_tokens']:>8}{cmp}")
    print("\n== mean over languages")
    for arm in ARMS:
        print(f"{arm:26}" + "".join(f"{sum(out['test'][L][arm][c] for L in LANGS) / len(LANGS):>9.3f}" for c in cols))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
