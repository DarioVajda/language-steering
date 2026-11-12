"""Cross-lingual instruction vs language steering on MGSM: the question in a source language S, the answer
wanted in a target language T (zero-shot, Llama-3.1-8B-Instruct), for all 110 ordered pairs of the 11 MGSM
languages (English included, as any other language).

The arms are those of ../mgsm_steering (en -> T) one-to-one, with English replaced by S:
  source                    the question in S (the reference; target language S)
  native                    the question in T
  instruction               the S question + "\n\n" + "Answer in <T>." written in S (INSTRUCTION[S][T])
  steer                     the S question, + alpha * v_(S->T)(l)
  instruction_steer         the instruction prompt, + alpha * v_(S->T)(l)
  steer_prompt, instruction_steer_prompt   the same, but steering the prompt only
v_(S->T)(l) = mu_T(l) - mu_S(l), mu_L(l) the mean output of block l over the tokens (BOS excluded) of the 997
FLORES-200 dev sentences in L. Model, decoding (greedy, at most 2048 new tokens), split (50 dev / 200 test
items), grid and steering are those of ../mgsm_steering/generate.py, imported from there.

Tuning pairs (TUNE): 11 pairs drawn before any generation, with a fixed seed, uniformly among the assignments
in which every language is the source of exactly one pair and the target of exactly one, and no pair is tuned
in both directions. The steering settings are chosen on their dev items; the other 99 pairs are held out.

Stages (each skips work whose output already exists):
  vectors   mu_L of the 11 languages at the grid layers -> results/generations/vectors.pt
  dev       source, native and instruction, and each steering arm over layer x alpha, on the dev items of the
            tuning pairs; a unit is one configuration over the 11 pairs -> results/generations/dev/<unit>.jsonl
  test      every arm on the test items of all 110 pairs, the steering arms at the setting chosen on dev
            (analyse_pairs.py --select -> results/generations/chosen.json); a unit is one arm on one pair, or
            for source and native (which depend only on S, resp. T) one language -> .../test/<unit>.jsonl
Workers share a stage through claim files, as in ../mgsm_steering.

Usage: see README.md (run.sh runs every stage).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
import generate as base  # noqa: E402  (../mgsm_steering/generate.py)
from generate import ALPHAS, LAYERS, NAMES, STEER_ARMS, Model, flores, split  # noqa: E402
from tasks import MGSM  # noqa: E402

GEN = HERE / "results/generations"
LANGS = ("en",) + base.LANGS
PAIRS = tuple((s, t) for s in LANGS for t in LANGS if t != s)
KEY = {p: f"{p[0]}-{p[1]}" for p in PAIRS}


def draw_tuning(seed=0):
    """Rejection sampling: a uniform shuffle, kept if no language maps to itself or back to its own source."""
    rng = random.Random(seed)
    while True:
        p = list(LANGS)
        rng.shuffle(p)
        m = dict(zip(LANGS, p))
        if all(m[a] != a and m[m[a]] != a for a in LANGS):
            return tuple(m.items())


TUNE = draw_tuning()

# The name of language T written in language S: CLDR 47 (cldr-localenames-full, languages.json of S).
# Russian needs the prepositional case after "на" ("на немецком языке"); суахили and телугу do not decline.
NAME_IN = {
    "en": {"en": "English", **NAMES},
    "bn": {"en": "ইংরেজি", "de": "জার্মান", "es": "স্প্যানিশ", "fr": "ফরাসি", "ja": "জাপানি", "ru": "রুশ",
           "sw": "সোয়াহিলি", "te": "তেলুগু", "th": "থাই", "zh": "চীনা"},
    "de": {"en": "Englisch", "bn": "Bengalisch", "es": "Spanisch", "fr": "Französisch", "ja": "Japanisch",
           "ru": "Russisch", "sw": "Suaheli", "te": "Telugu", "th": "Thailändisch", "zh": "Chinesisch"},
    "es": {"en": "inglés", "bn": "bengalí", "de": "alemán", "fr": "francés", "ja": "japonés", "ru": "ruso",
           "sw": "suajili", "te": "telugu", "th": "tailandés", "zh": "chino"},
    "fr": {"en": "anglais", "bn": "bengali", "de": "allemand", "es": "espagnol", "ja": "japonais", "ru": "russe",
           "sw": "swahili", "te": "télougou", "th": "thaï", "zh": "chinois"},
    "ja": {"en": "英語", "bn": "ベンガル語", "de": "ドイツ語", "es": "スペイン語", "fr": "フランス語", "ru": "ロシア語",
           "sw": "スワヒリ語", "te": "テルグ語", "th": "タイ語", "zh": "中国語"},
    "ru": {"en": "английском языке", "bn": "бенгальском языке", "de": "немецком языке", "es": "испанском языке",
           "fr": "французском языке", "ja": "японском языке", "sw": "языке суахили", "te": "языке телугу",
           "th": "тайском языке", "zh": "китайском языке"},
    "sw": {"en": "Kiingereza", "bn": "Kibengali", "de": "Kijerumani", "es": "Kihispania", "fr": "Kifaransa",
           "ja": "Kijapani", "ru": "Kirusi", "te": "Kitelugu", "th": "Kithai", "zh": "Kichina"},
    "te": {"en": "ఇంగ్లీష్", "bn": "బంగ్లా", "de": "జర్మన్", "es": "స్పానిష్", "fr": "ఫ్రెంచ్", "ja": "జపనీస్",
           "ru": "రష్యన్", "sw": "స్వాహిలి", "th": "థాయ్", "zh": "చైనీస్"},
    "th": {"en": "อังกฤษ", "bn": "บังกลา", "de": "เยอรมัน", "es": "สเปน", "fr": "ฝรั่งเศส", "ja": "ญี่ปุ่น",
           "ru": "รัสเซีย", "sw": "สวาฮีลี", "te": "เตลูกู", "zh": "จีน"},
    "zh": {"en": "英语", "bn": "孟加拉语", "de": "德语", "es": "西班牙语", "fr": "法语", "ja": "日语", "ru": "俄语",
           "sw": "斯瓦希里语", "te": "泰卢固语", "th": "泰语"},
}
# "Answer in <T>." in S (imperative; the English one is that of ../mgsm_steering).
TEMPLATE = {"en": "Answer in {}.", "bn": "{} ভাষায় উত্তর দিন।", "de": "Antworte auf {}.", "es": "Responde en {}.",
            "fr": "Réponds en {}.", "ja": "{}で答えてください。", "ru": "Отвечай на {}.", "sw": "Jibu kwa {}.",
            "te": "{} భాషలో సమాధానం ఇవ్వండి.", "th": "ตอบเป็นภาษา{}", "zh": "请用{}回答。"}
INSTRUCTION = {s: {t: TEMPLATE[s].format(n) for t, n in NAME_IN[s].items() if t != s} for s in LANGS}


def prompt(arm, row, src, tgt):
    if arm == "native":
        return row["text"][tgt]["question"]
    q = row["text"][src]["question"]
    if arm.startswith("instruction"):
        return f"{q}\n\n{INSTRUCTION[src][tgt]}"
    return q  # source, steer, steer_prompt


def units(stage):
    """{unit name: (arm, layer, alpha, [(row, src, tgt)])} of a stage (src is None for native, tgt for source)."""
    sp = split()
    if stage == "dev":
        items = lambda: [(r, s, t) for s, t in TUNE for r in sp["dev"]]
        u = {arm: (arm, None, None, items()) for arm in ("source", "native", "instruction")}
        for arm in STEER_ARMS:
            for l in LAYERS:
                for a in ALPHAS:
                    u[f"{arm}_L{l}_a{a}"] = (arm, l, a, items())
        return u
    chosen = json.loads((GEN / "chosen.json").read_text())
    u = {}
    for L in LANGS:
        u[f"source_{L}"] = ("source", None, None, [(r, L, None) for r in sp["test"]])
        u[f"native_{L}"] = ("native", None, None, [(r, None, L) for r in sp["test"]])
    for p in PAIRS:
        for arm in ("instruction",) + STEER_ARMS:
            l, a = (chosen[arm]["layer"], chosen[arm]["alpha"]) if arm in STEER_ARMS else (None, None)
            u[f"{arm}_{KEY[p]}"] = (arm, l, a, [(r, *p) for r in sp["test"]])
    return u


def run_unit(M, mu, arm, layer, alpha, items):
    texts = [prompt(arm, r, s, t) for r, s, t in items]
    po = arm.endswith("_prompt")
    v = [alpha * (mu[t][layer] - mu[s][layer]) for _, s, t in items] if layer else None
    res = M.generate(texts, layer, v, po)
    task = MGSM()
    # lang: the language the answer is wanted in (S for the source reference, else T)
    return [dict(id=r["id"], src=s, tgt=t, lang=s if arm == "source" else t, arm=arm, layer=layer, alpha=alpha,
                 prompt_only=po, answer=r["answer"], pred=task.parse(out), input=x, response=out, n_tokens=k,
                 capped=c)
            for (r, s, t), x, (out, k, c) in zip(items, texts, res)]


def claim(stage, name):
    d = GEN / "claims" / os.environ.get("SLURM_JOB_ID", "local") / stage
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / name, os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("vectors", "dev", "test"))
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    M = Model()
    if args.stage == "vectors":
        if not (GEN / "vectors.pt").exists():
            mu = {L: M.mean_states(flores(L)) for L in LANGS}
            tmp = GEN / ".vectors.pt.tmp"
            torch.save({L: {l: mu[L][l].cpu() for l in LAYERS} for L in LANGS}, tmp)
            tmp.rename(GEN / "vectors.pt")
        return
    mu = torch.load(GEN / "vectors.pt")
    out = GEN / args.stage
    out.mkdir(exist_ok=True)
    for name, (arm, layer, alpha, items) in units(args.stage).items():
        if (out / f"{name}.jsonl").exists() or not claim(args.stage, name):
            continue
        rows = run_unit(M, mu, arm, layer, alpha, items)
        tmp = out / f".{name}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.rename(out / f"{name}.jsonl")
        acc = sum(r["pred"] is not None and abs(r["pred"] - r["answer"]) < 1e-6 for r in rows) / len(rows)
        print(f"{args.stage} {name}: acc {acc:.2f} capped {sum(r['capped'] for r in rows)}"
              f" | {rows[0]['response'][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
