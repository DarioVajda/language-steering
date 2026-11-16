"""Cross-lingual instruction vs language steering on MGSM: the question in a source language S, the answer
wanted in a target language T (zero-shot), for all 110 ordered pairs of the 11 MGSM languages (English
included, as any other language).

The model is set by the environment variable XL_MODEL, a key of MODELS in ../mgsm_steering/generate.py
(llama-3.1-8b: Llama-3.1-8B-Instruct; qwen3-8b, qwen3-32b: Qwen3-8B, Qwen3-32B, thinking off; gemma3-12b:
Gemma-3-12B-it); its outputs go to results/<XL_MODEL>/. Every script here reads it.

The arms are those of ../mgsm_steering (en -> T) one-to-one, with English replaced by S:
  source                    the question in S (the reference; target language S)
  native                    the question in T
  instruction               the S question + "\n\n" + "Answer in <T>." written in S (INSTRUCTION[S][T])
  instruction_en            the same, with the instruction in English ("Answer in <T>.", ENGLISH[T])
  instruction_system        the S question, with INSTRUCTION[S][T] as a system message
  steer                     the S question, + alpha * v_(S->T)(l)
  instruction_steer         the instruction prompt, + alpha * v_(S->T)(l)
  steer_prompt, instruction_steer_prompt   the same, but steering the prompt only
  instruction_en_steer_prompt              the instruction_en prompt, steering the prompt only
  instruction_target        the S question + "\n\n" + "Answer in <T>." written in T (TARGET[T]; for T = en the
                            instruction_en prompt)
  instruction_first         INSTRUCTION[S][T] + "\n\n" + the S question (the instruction before the question)
  instruction_steer_prompt_random          controls for instruction_steer_prompt, at its chosen setting: per item,
  instruction_steer_prompt_wrong           a random direction of the norm of alpha * v_(S->T)(l), resp.
                                           alpha * v_(S->U)(l) for a language U != S, T drawn uniformly
                                           (seeded by arm, S, T and item id)
v_(S->T)(l) = mu_T(l) - mu_S(l), mu_L(l) the mean output of block l over the tokens of the 997 FLORES-200 dev
sentences in L, without the first (the attention sink: Llama's BOS, Qwen's first word). Decoding (greedy, at
most 2048 new tokens), split (50 dev / 200 test items), alpha grid and steering are those of ../mgsm_steering/generate.py, imported from there; the layer grid is the
model's (MODELS).

Tuning pairs (TUNE): 11 pairs drawn before any generation, with a fixed seed, uniformly among the assignments
in which every language is the source of exactly one pair and the target of exactly one, and no pair is tuned
in both directions. The steering settings are chosen on their dev items; the other 99 pairs are held out.

Stages (each skips work whose output already exists):
  vectors   mu_L of the 11 languages at the grid layers -> results/<model>/generations/vectors.pt
  dev       source, native, the three instruction variants, and each steering arm over layer x alpha, on the
            dev items of the tuning pairs; a unit is one configuration over the 11 pairs
            -> results/<model>/generations/dev/<unit>.jsonl
  test      source, native, instruction and the steering arms on the test items of all 110 pairs, the steering
            arms at the setting chosen on dev (analyse_pairs.py --select -> generations/chosen.json)
            and the two controls at that of instruction_steer_prompt,
            and the instruction variant chosen on dev if it is not the instruction (-> chosen_instruction.json);
            a unit is one arm on one pair, or for source and native (which depend only on S, resp. T) one
            language -> .../test/<unit>.jsonl
  prompts   the two prompt variants PROMPT_ARMS (instruction_target, instruction_first) on the test items of all
            110 pairs, untuned and compared directly with the instruction (prompt_variants.py); run separately,
            after test -> .../test/<unit>.jsonl
The two variants are prompt baselines without steering. instruction_en was chosen on dev for Llama, so instruction
+ prompt-only steering is also tuned and tested with it (instruction_en_steer_prompt), for every model.
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
from generate import ALPHAS, MODELS, NAMES, STEER_ARMS, Model, flores, split  # noqa: E402
from tasks import MGSM  # noqa: E402

MODEL = os.environ.get("XL_MODEL")
if MODEL not in MODELS:
    sys.exit(f"set XL_MODEL to one of {', '.join(MODELS)}")
LAYERS = MODELS[MODEL]["layers"]
RES = HERE / "results" / MODEL
GEN = RES / "generations"
LANGS = ("en",) + base.LANGS
PAIRS = tuple((s, t) for s in LANGS for t in LANGS if t != s)
KEY = {p: f"{p[0]}-{p[1]}" for p in PAIRS}
STEER_ARMS = STEER_ARMS + ("instruction_en_steer_prompt",)
CONTROL_ARMS = ("instruction_steer_prompt_random", "instruction_steer_prompt_wrong")


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
ENGLISH = {t: TEMPLATE["en"].format(n) for t, n in NAME_IN["en"].items()}  # "Answer in <T>." for every T
INSTRUCTION_ARMS = ("instruction", "instruction_en", "instruction_system")  # the first is the default
# Each language's name for itself, as the templates take it: "Answer in <T>." written in T.
SELF_NAME = {"en": "English", "bn": "বাংলা", "de": "Deutsch", "es": "español", "fr": "français", "ja": "日本語",
             "ru": "русском языке", "sw": "Kiswahili", "te": "తెలుగు", "th": "ไทย", "zh": "中文"}
TARGET = {t: TEMPLATE[t].format(n) for t, n in SELF_NAME.items()}
PROMPT_ARMS = ("instruction_target", "instruction_first")


def prompt(arm, row, src, tgt):
    """The user message."""
    if arm == "native":
        return row["text"][tgt]["question"]
    q = row["text"][src]["question"]
    if arm == "instruction_target":
        return f"{q}\n\n{TARGET[tgt]}"
    if arm == "instruction_first":
        return f"{INSTRUCTION[src][tgt]}\n\n{q}"
    if arm.startswith("instruction_en"):
        return f"{q}\n\n{ENGLISH[tgt]}"
    if arm.startswith("instruction") and arm != "instruction_system":
        return f"{q}\n\n{INSTRUCTION[src][tgt]}"
    return q  # source, steer, steer_prompt, instruction_system


def system(arm, src, tgt):
    """The system message, or None."""
    return INSTRUCTION[src][tgt] if arm == "instruction_system" else None


def units(stage):
    """{unit name: (arm, layer, alpha, [(row, src, tgt)])} of a stage (src is None for native, tgt for source)."""
    sp = split()
    if stage == "dev":
        items = lambda: [(r, s, t) for s, t in TUNE for r in sp["dev"]]
        u = {arm: (arm, None, None, items()) for arm in ("source", "native") + INSTRUCTION_ARMS}
        for arm in STEER_ARMS:
            for l in LAYERS:
                for a in ALPHAS:
                    u[f"{arm}_L{l}_a{a}"] = (arm, l, a, items())
        return u
    if stage == "prompts":
        return {f"{arm}_{KEY[p]}": (arm, None, None, [(r, *p) for r in sp["test"]]) for arm in PROMPT_ARMS for p in PAIRS}
    chosen = json.loads((GEN / "chosen.json").read_text())
    u = {}
    for L in LANGS:
        u[f"source_{L}"] = ("source", None, None, [(r, L, None) for r in sp["test"]])
        u[f"native_{L}"] = ("native", None, None, [(r, None, L) for r in sp["test"]])
    for p in PAIRS:
        for arm in ("instruction",) + STEER_ARMS:
            l, a = (chosen[arm]["layer"], chosen[arm]["alpha"]) if arm in STEER_ARMS else (None, None)
            u[f"{arm}_{KEY[p]}"] = (arm, l, a, [(r, *p) for r in sp["test"]])
        for arm in CONTROL_ARMS:
            l, a = chosen["instruction_steer_prompt"]["layer"], chosen["instruction_steer_prompt"]["alpha"]
            u[f"{arm}_{KEY[p]}"] = (arm, l, a, [(r, *p) for r in sp["test"]])
    variant = json.loads((GEN / "chosen_instruction.json").read_text())["arm"]
    if variant != "instruction":
        for p in PAIRS:
            u[f"{variant}_{KEY[p]}"] = (variant, None, None, [(r, *p) for r in sp["test"]])
    return u


def vector(arm, mu, layer, alpha, row, src, tgt):
    """The vector added for one item, and the wrong target U (for the wrong-target control, else None)."""
    v = alpha * (mu[tgt][layer] - mu[src][layer])
    if arm not in CONTROL_ARMS:
        return v, None
    rng = random.Random(f"{arm}|{src}|{tgt}|{row['id']}")
    if arm.endswith("_wrong"):
        u = rng.choice([L for L in LANGS if L not in (src, tgt)])
        return alpha * (mu[u][layer] - mu[src][layer]), u
    d = torch.randn(v.shape, generator=torch.Generator().manual_seed(rng.getrandbits(63)))
    return (d * v.float().norm() / d.norm()).to(v.dtype), None


def run_unit(M, mu, arm, layer, alpha, items):
    texts = [prompt(arm, r, s, t) for r, s, t in items]
    po = arm.endswith("_prompt") or arm in CONTROL_ARMS
    vs = [vector(arm, mu, layer, alpha, r, s, t) for r, s, t in items] if layer else [(None, None)] * len(items)
    v = [x for x, _ in vs] if layer else None
    res = M.generate(texts, layer, v, po, [system(arm, s, t) for _, s, t in items])
    task = MGSM()
    # lang: the language the answer is wanted in (S for the source reference, else T)
    return [dict(id=r["id"], src=s, tgt=t, lang=s if arm == "source" else t, arm=arm, layer=layer, alpha=alpha,
                 prompt_only=po, answer=r["answer"], pred=task.parse(out), system=system(arm, s, t), wrong_target=u,
                 input=x, response=out, n_tokens=k, capped=c)
            for (r, s, t), x, (_, u), (out, k, c) in zip(items, texts, vs, res)]


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
    ap.add_argument("stage", choices=("vectors", "dev", "test", "prompts"))
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    M = Model(MODELS[MODEL]["hf"])
    if args.stage == "vectors":
        if not (GEN / "vectors.pt").exists():
            mu = {L: M.mean_states(flores(L)) for L in LANGS}
            tmp = GEN / ".vectors.pt.tmp"
            torch.save({L: {l: mu[L][l].cpu() for l in LAYERS} for L in LANGS}, tmp)
            tmp.rename(GEN / "vectors.pt")
        return
    mu = torch.load(GEN / "vectors.pt")
    out = GEN / ("test" if args.stage == "prompts" else args.stage)
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
