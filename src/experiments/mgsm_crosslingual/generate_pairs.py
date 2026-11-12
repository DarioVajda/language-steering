"""Cross-lingual instruction vs language steering on MGSM: the question in a source language S, the answer
wanted in a target language T (zero-shot, Llama-3.1-8B-Instruct).

The arms are those of ../mgsm_steering (en -> T) one-to-one, with English replaced by S:
  source                    the question in S (the reference; target language S)
  native                    the question in T
  instruction               the S question + "\n\n" + "Answer in <T>." written in S (INSTRUCTION[S][T])
  steer                     the S question, + alpha * v_(S->T)(l)
  instruction_steer         the instruction prompt, + alpha * v_(S->T)(l)
  steer_prompt, instruction_steer_prompt   the same, but steering the prompt only
v_(S->T)(l) = mu_T(l) - mu_S(l), the mean output of block l over the tokens (BOS excluded) of the 997
FLORES-200 dev sentences. Model, decoding (greedy, at most 2048 new tokens), split (50 dev / 200 test items),
grid and steering are those of ../mgsm_steering/generate.py, imported from there.

Pairs: 10 drawn uniformly without replacement, with a fixed seed and before any generation, from the 100 pairs
with S one of the 10 non-English MGSM languages and T any other of the 11 (English included).

Stages (each skips work whose output already exists):
  pilot     the instruction arm on dev for all 100 pairs (a failure map; not used to choose anything);
            a unit is one source language -> results/generations/pilot/<S>.jsonl
  vectors   v_(S->T) of the 10 pairs at the grid layers -> results/generations/vectors.pt
  dev       source, native and instruction, and each steering arm over layer x alpha, on dev; a unit is one
            configuration over the 10 pairs -> results/generations/dev/<unit>.jsonl
  test      every arm on test, the steering arms at the setting chosen on dev (analyse_pairs.py --select ->
            results/generations/chosen.json); a unit is one arm on one pair -> .../test/<unit>.jsonl
Workers share a stage through claim files, as in ../mgsm_steering.

Usage: see README.md (one job runs every stage).
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
from generate import ALPHAS, LAYERS, STEER_ARMS, Model, flores, split  # noqa: E402
from tasks import MGSM  # noqa: E402

GEN = HERE / "results/generations"
LANGS = ("en",) + base.LANGS
SOURCES = base.LANGS
ALL_PAIRS = [(s, t) for s in SOURCES for t in LANGS if t != s]
PAIRS = tuple(random.Random(0).sample(ALL_PAIRS, 10))
KEY = {p: f"{p[0]}-{p[1]}" for p in ALL_PAIRS}

# The name of language T written in language S: CLDR 47 (cldr-localenames-full, languages.json of S).
# Russian needs the prepositional case after "на" ("на немецком языке"); суахили and телугу do not decline.
NAME_IN = {
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
# "Answer in <T>." in S (imperative, as in the English instruction).
TEMPLATE = {"bn": "{} ভাষায় উত্তর দিন।", "de": "Antworte auf {}.", "es": "Responde en {}.", "fr": "Réponds en {}.",
            "ja": "{}で答えてください。", "ru": "Отвечай на {}.", "sw": "Jibu kwa {}.", "te": "{} భాషలో సమాధానం ఇవ్వండి.",
            "th": "ตอบเป็นภาษา{}", "zh": "请用{}回答。"}
INSTRUCTION = {s: {t: TEMPLATE[s].format(n) for t, n in NAME_IN[s].items()} for s in SOURCES}


def prompt(arm, row, src, tgt):
    q = row["text"][src]["question"]
    if arm == "native":
        return row["text"][tgt]["question"]
    if arm.startswith("instruction"):
        return f"{q}\n\n{INSTRUCTION[src][tgt]}"
    return q  # source, steer, steer_prompt


def units(stage):
    """{unit name: (arm, layer, alpha, [(row, src, tgt)])} of a stage."""
    sp = split()
    if stage == "pilot":
        return {s: ("instruction", None, None, [(r, s, t) for t in LANGS if t != s for r in sp["dev"]])
                for s in SOURCES}
    if stage == "dev":
        items = lambda: [(r, s, t) for s, t in PAIRS for r in sp["dev"]]
        u = {arm: (arm, None, None, items()) for arm in ("source", "native", "instruction")}
        for arm in STEER_ARMS:
            for l in LAYERS:
                for a in ALPHAS:
                    u[f"{arm}_L{l}_a{a}"] = (arm, l, a, items())
        return u
    chosen = json.loads((GEN / "chosen.json").read_text())
    u = {}
    for p in PAIRS:
        for arm in ("source", "native", "instruction") + STEER_ARMS:
            l, a = (chosen[arm]["layer"], chosen[arm]["alpha"]) if arm in STEER_ARMS else (None, None)
            u[f"{arm}_{KEY[p]}"] = (arm, l, a, [(r, *p) for r in sp["test"]])
    return u


def run_unit(M, vecs, arm, layer, alpha, items):
    texts = [prompt(arm, r, s, t) for r, s, t in items]
    po = arm.endswith("_prompt")
    v = [alpha * vecs[KEY[(s, t)]][layer] for _, s, t in items] if layer else None
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
    ap.add_argument("stage", choices=("pilot", "vectors", "dev", "test"))
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    M = Model()
    if args.stage == "vectors":
        if not (GEN / "vectors.pt").exists():
            mu = {L: M.mean_states(flores(L)) for L in sorted({L for p in PAIRS for L in p})}
            torch.save({KEY[(s, t)]: {l: (mu[t][l] - mu[s][l]).cpu() for l in LAYERS} for s, t in PAIRS},
                       GEN / "vectors.pt")
        return
    vecs = torch.load(GEN / "vectors.pt") if args.stage != "pilot" else None
    out = GEN / args.stage
    out.mkdir(exist_ok=True)
    for name, (arm, layer, alpha, items) in units(args.stage).items():
        if (out / f"{name}.jsonl").exists() or not claim(args.stage, name):
            continue
        rows = run_unit(M, vecs, arm, layer, alpha, items)
        tmp = out / f".{name}.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.rename(out / f"{name}.jsonl")
        acc = sum(r["pred"] is not None and abs(r["pred"] - r["answer"]) < 1e-6 for r in rows) / len(rows)
        print(f"{args.stage} {name}: acc {acc:.2f} capped {sum(r['capped'] for r in rows)}"
              f" | {rows[0]['response'][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
