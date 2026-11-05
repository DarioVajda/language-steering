"""Score the generations: language fidelity (fastText) and accuracy against the localized answer.

  fidelity  share of responses whose explanation (text before \\boxed) fastText labels as T
  accuracy  the normalized \\boxed{} content contains the localized answer or one of its aliases
Selection:
  single layer   alpha = 1 (the strength that switches the language at every layer), the layer
                 with the best test accuracy (chosen on test, so slightly optimistic)
  multi layer    per variant, the alpha with the best val accuracy among those with val
                 fidelity >= 0.95 (best val fidelity if none qualifies); reported on test
Prints the tables and writes results/summary.json.

Usage: python src/experiments/output_language/analyse.py
"""
import collections
import json
import re
import unicodedata
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GEN = HERE / "results/generations"
LANGS = ("fr", "hu", "zh", "ru")
VARIANTS = {"5 layers": [8, 12, 16, 20, 24], "all 32 layers": list(range(1, 33))}

fasttext.FastText.eprint = lambda *a, **k: None  # silence the load_model deprecation notice
LID = fasttext.load_model(str(ROOT / "data/raw_data/fasttext/lid.176.ftz"))


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[\s\.,;:!?\"'«»“”()\[\]]+", " ", s).strip()


def boxed(s):
    m = re.findall(r"\\boxed\{([^{}]*)\}", s)
    return m[-1] if m else None


def language(s):
    expl = s.split("\\boxed")[0].replace("\n", " ").strip()
    if not expl:
        return "none"
    # what FastText.predict does (it appends "\n"), minus its NumPy-2-incompatible np.array(..., copy=False)
    return LID.f.predict(expl + "\n", 1, 0.0, "strict")[0][1].replace("__label__", "")


def correct(s, answers):
    b = boxed(s)
    if b is None:
        return 0
    b = norm(b)
    aliases = [norm(x) for a in answers for x in [a.get("text") or ""] + (a.get("aliases") or []) if x]
    return int(any(x and x in b for x in aliases))


def score(rows, L):
    n = len(rows)
    langs = collections.Counter(language(r["response"]) for r in rows)
    return dict(n=n, fidelity=langs[L] / n,
                accuracy=sum(correct(r["response"], r["answers"]) for r in rows) / n,
                no_box=sum(boxed(r["response"]) is None for r in rows) / n,
                languages=dict(langs.most_common(3)))


def main():
    rows = [json.loads(l) for f in sorted(GEN.glob("*.jsonl")) for l in open(f)]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["lang"], r["cond"], tuple(r.get("layers", ())), r.get("alpha"), r["split"])].append(r)
    plain = groups[("en", "plain", (), None, "test")]
    out = {"plain": dict(score(plain, "en"), accuracy_en=sum(correct(r["response"], r["answers_en"]) for r in plain) / len(plain)),
           "languages": {}}
    print(f"plain (English question, English answers): accuracy {out['plain']['accuracy_en']:.2f}")

    for L in LANGS:
        res = {c: score(groups[(L, c, (), None, "test")], L) for c in ("instruction", "native")}
        single = {k[2][0]: score(v, L) for k, v in groups.items()
                  if k[0] == L and k[1] == "steer" and len(k[2]) == 1 and k[3] == 1 and k[4] == "test"}
        best = max(single, key=lambda l: single[l]["accuracy"])
        res["single layer"] = dict(single[best], layer=best, alpha=1)
        grid = []
        for name, layers in VARIANTS.items():
            alphas = sorted(k[3] for k in groups if k[:3] == (L, "steer", tuple(layers)) and k[4] == "val")
            val = {a: score(groups[(L, "steer", tuple(layers), a, "val")], L) for a in alphas}
            test = {a: score(groups[(L, "steer", tuple(layers), a, "test")], L) for a in alphas}
            ok = [a for a in alphas if val[a]["fidelity"] >= 0.95]
            pick = max(ok, key=lambda a: val[a]["accuracy"]) if ok else max(alphas, key=lambda a: val[a]["fidelity"])
            res[name] = dict(test[pick], alpha=pick)
            grid += [dict(variant=name, alpha=a, selected=a == pick, val=val[a], test=test[a]) for a in alphas]
        out["languages"][L] = {"test": res, "multi_layer_grid": grid}

        print(f"\n== {L}  (test, n = {res['instruction']['n']})")
        for c, d in res.items():
            extra = f"  layer {d['layer']}" if "layer" in d else f"  alpha {d['alpha']}" if "alpha" in d else ""
            print(f"{c:14} fidelity {d['fidelity']:.2f}  accuracy {d['accuracy']:.2f}  no box {d['no_box']:.2f}{extra}")
        for g in grid:
            v, t = g["val"], g["test"]
            print(f"  {g['variant']:14} alpha {g['alpha']:<5}{'*' if g['selected'] else ' '} "
                  f"val fid {v['fidelity']:.2f} acc {v['accuracy']:.2f} | test fid {t['fidelity']:.2f} "
                  f"acc {t['accuracy']:.2f} no box {t['no_box']:.2f}")
    (HERE / "results/summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
