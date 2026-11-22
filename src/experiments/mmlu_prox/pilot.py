"""Pilot of cross-lingual MMLU-ProX: does the instruction fail to set the answer language outside maths?

The question and its options are in a source language S, with zero-shot chain-of-thought prompting; the answer is
wanted in a target language T. Two subsets of the MMLU-ProX test set (data/raw_data/mmlu_prox, the 11 MGSM
languages), the same questions in every language:
  stem   physics, chemistry, engineering, biology, computer science: 40 questions each (200); maths is left out
  hss    humanities and social sciences, with health, business and "other": law, economics, psychology, business,
         health, philosophy, history, other: 25 questions each (200)
Per category, the questions are sorted by crc32 of their question_id and the first ones are taken (no RNG).

Arms (the user message; options lettered A, B, ... in the dataset's order):
  cot          question + "\\n\\n" + options + "\\n\\n" + COT[S] (the answer is wanted in S)
  instruction  the same + " " + INSTRUCTION[S][T] ("Answer in <T>." written in S, as in ../mgsm_crosslingual)
  instruction_steer_prompt   the instruction prompt, + alpha * v_(S->T)(l) on the prompt only: the FLORES vectors
               and the (layer, alpha) chosen on MGSM dev in ../mgsm_crosslingual (generations/vectors.pt, chosen.json),
               reused without tuning on MMLU-ProX
  instruction_en  the cot prompt + " " + "Answer in <T>." in English (ENGLISH[T])
  instruction_system  the cot prompt, with INSTRUCTION[S][T] as a system message, as in ../mgsm_crosslingual
               (Gemma-3's chat template has no system role and puts it at the start of the user turn); run only
               for Gemma-3-12B, the model whose prompt variant chosen on MGSM dev it is (--arms)
  recover      ReCoVeR (Sterz et al. 2025) as in ../mgsm_crosslingual/recover.py: the instruction prompt, alpha *
               (r_T - r_S)(j) / |.| added at every block and token, with the norm restore, at the alpha chosen on
               MGSM dev (generations/recover/), reused without tuning, like ours
COT[S] is "Let's think step by step." in S, as MMLU-ProX writes it in its own worked examples (validation split).
Pairs: the 11 tuning pairs of ../mgsm_crosslingual (TUNE). Greedy decoding (../mgsm_steering/generate.py), at
most --max-new new tokens: 2048 there, but Bengali and Telugu answers that finish are about 2.5 times as long in
Llama-3.1 tokens as those of the other languages (median about 1150 against 450), so 2048 cuts many of them;
8192 gives them the room 2048 gives the others. The model is XL_MODEL, as in ../mgsm_crosslingual.

--first N keeps only the first N questions of each category (a prefix of the same order), as a small test: its
outputs go to generations/pilot[_<max new>]_first<N>/ and pilot[_<max new>]_first<N>.json, and units already
generated on the whole sample (pilot[_<max new>]/) are read from there, restricted to those questions.

--format boxed adds FORMAT[S] ("Finish your answer with \\boxed{X}, where X is the letter of the correct option.",
in S) after COT[S] in every arm, and the letter is the last \\boxed{X} (none: wrong; a letter in T's script, with no
Latin box, is read through TRANSLIT); outputs get the suffix _boxed.
--pool generates all units of an arm in one call (for small samples, e.g. with --first).

--pairs all runs the instruction and steering arms on all 110 ordered pairs instead of the 11 tuning pairs (the
same unit files; the analysis adds the tuning and the 99 held-out pairs) -> pilot[...]_all.json.

--first-cap C generates in two stages instead of one pass: stage 1 generates at most C new tokens, stage 2 continues
every response capped there from its own C tokens (forced, as a prefix: greedy decoding is the same continuation,
up to the rounding of recomputing those tokens in one pass) to --max-new in all, the capped responses of all units
pooled into full batches. A batch decodes until its last response stops, and most capped responses are repetition
loops that run to the cap, so in one pass a single loop keeps a nearly empty batch decoding to --max-new; pooled,
they share full batches. Stage 1 -> generations/pilot_<max new>_from<C>/stage1/, 2 -> stage2/<arm>_<chunk>, merged into the
units by analyse.

Stages:
  run       every task not yet done: the units (cot in one S, or one arm on one pair; 400 questions) and, staged,
            the chunks of capped stage-1 responses of each arm whose units are all done; workers of any number of
            jobs share the tasks through heartbeat claims (generations/.../claims/shared/), --arms limits them to
            some arms, and a --helper worker leaves when nothing is left to take
            -> results/<model>/generations/pilot[_<max new>]/<unit>.jsonl (no suffix for 2048)
  analyse   merge the stages, then the letter, accuracy, in-language rate, success, per subset and arm
            -> results/<model>/pilot[_<max new>].json, and the per-question scores for tests.py -> pilot[...]_items.json

Usage: XL_MODEL=<model> python pilot.py run | analyse [options]   (pilot.sh runs both)
"""
import argparse
import collections
import functools
import json
import os
import re
import sys
import threading
import time
import uuid
import zlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
sys.path.insert(0, str(HERE.parent / "mgsm_crosslingual"))
import generate  # noqa: E402  (MAX_NEW per task in run_unit and run_chunk, MAX_BATCH per batch in fit)
from generate import MODELS  # noqa: E402
from generate_pairs import ENGLISH, INSTRUCTION, KEY, LANGS, MODEL, PAIRS, TUNE, vector  # noqa: E402

DATA = ROOT / "data/raw_data/mmlu_prox"
RES = HERE / "results" / MODEL
GEN = RES / "generations" / "pilot"  # set in main for --max-new
OUT = RES / "pilot.json"
SUBSETS = {"stem": (("physics", "chemistry", "engineering", "biology", "computer science"), 40),
           "hss": (("law", "economics", "psychology", "business", "health", "philosophy", "history", "other"), 25)}
# "Let's think step by step." as in MMLU-ProX's validation cot_content (checked in check_cot).
COT = {"en": "Let's think step by step.", "bn": "আসুন ধাপে ধাপে চিন্তা করি।", "de": "Denken wir Schritt für Schritt nach.",
       "es": "Pensemos paso a paso.", "fr": "Réfléchissons étape par étape.", "ja": "一歩一歩考えていきましょう。",
       "ru": "Давайте подумаем шаг за шагом.", "sw": "Hebu tufikiria hatua kwa hatua.", "te": "దశలవారీగా ఆలోచిద్దాం.",
       "th": "มาคิดทีละขั้นตอนกัน", "zh": "让我们一步一步地思考。"}
# --format boxed: the answer format, after COT[S] in every arm, written in S like the rest of the prompt (in English
# for every S it would itself pull the answer towards English, as "Answer in <T>." in English does): "Finish your
# answer with \boxed{X}, where X is the letter of the correct option." The second-person form of each language
# follows that of INSTRUCTION[S] (du in German, tu in French, ty in Russian). The letter is then the last \boxed{X}.
FORMAT = {"en": r"Finish your answer with \boxed{X}, where X is the letter of the correct option.",
          "bn": r"আপনার উত্তর \boxed{X} দিয়ে শেষ করুন, যেখানে X হলো সঠিক বিকল্পের অক্ষর।",
          "de": r"Beende deine Antwort mit \boxed{X}, wobei X der Buchstabe der richtigen Option ist.",
          "es": r"Termina tu respuesta con \boxed{X}, donde X es la letra de la opción correcta.",
          "fr": r"Termine ta réponse par \boxed{X}, où X est la lettre de la bonne option.",
          "ja": r"回答の最後に \boxed{X} と書いてください。X は正しい選択肢の文字です。",
          "ru": r"Закончи ответ записью \boxed{X}, где X — буква правильного варианта.",
          "sw": r"Malizia jibu lako kwa \boxed{X}, ambapo X ni herufi ya chaguo sahihi.",
          "te": r"మీ సమాధానాన్ని \boxed{X}తో ముగించండి, ఇక్కడ X సరైన ఎంపిక యొక్క అక్షరం.",
          "th": r"จบคำตอบด้วย \boxed{X} โดยที่ X คือตัวอักษรของตัวเลือกที่ถูกต้อง",
          "zh": r"请在回答的最后写出 \boxed{X}，其中 X 是正确选项的字母。"}
BOXED_FORMAT = False  # --format boxed
POOL = False  # --pool
LETTERS = "ABCDEFGHIJ"
ARMS = ("cot", "instruction", "instruction_steer_prompt", "instruction_en", "recover", "instruction_system")
DEFAULT_ARMS = ARMS[:-1]  # run: instruction_system only when asked for (--arms)
STEER = ("instruction_steer_prompt",)
XL = HERE.parent / "mgsm_crosslingual/results" / MODEL / "generations"
FIRST = None  # --first
STAGED, FIRST_CAP, MAX_NEW = False, None, 2048  # --first-cap, --max-new
RUN = TUNE  # --pairs: the pairs of the instruction and steering arms
FULL = GEN  # set in main: the whole-sample generations


def table(lang, split="test"):
    return pd.read_parquet(next((DATA / lang).glob(f"{split}-*.parquet")))


def check_cot():
    """Every COT[L] opens most of the worked examples of L."""
    for L in LANGS:
        v = table(L, "validation").cot_content
        n = sum(c.strip().removeprefix("A:").strip().startswith(COT[L]) for c in v)
        assert n >= len(v) * 0.8, (L, n, len(v))


@functools.cache
def sample():
    """[{id, category, subset, answer, text: {L: (question, [options])}}], the same questions in every language."""
    t = {L: table(L).set_index("question_id") for L in LANGS}
    en = t["en"]
    for L in LANGS:
        assert (t[L].index == en.index).all() and (t[L].answer == en.answer).all()
    rows = []
    for subset, (cats, k) in SUBSETS.items():
        for c in cats:
            ids = sorted(en.index[en.category == c], key=lambda i: zlib.crc32(str(i).encode()))[:k]
            assert len(ids) == k
            ids = ids[:FIRST]
            for i in ids:
                opts = [o for o in en.loc[i, [f"option_{j}" for j in range(10)]] if isinstance(o, str)]
                text = {}
                for L in LANGS:
                    r = t[L].loc[i]
                    o = [r[f"option_{j}"] for j in range(len(opts))]
                    assert all(isinstance(x, str) and x.strip() for x in o) and r.question.strip(), (L, i)
                    text[L] = (r.question.strip(), [x.strip() for x in o])
                assert LETTERS.index(en.loc[i, "answer"]) < len(opts)
                rows.append(dict(id=int(i), category=c, subset=subset, answer=en.loc[i, "answer"], text=text))
    return rows


def prompt(arm, row, src, tgt):
    q, opts = row["text"][src]
    body = q + "\n\n" + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts)) + "\n\n" + COT[src]
    if BOXED_FORMAT:
        body += " " + FORMAT[src]
    if arm in ("cot", "instruction_system"):
        return body
    return f"{body} {ENGLISH[tgt] if arm == 'instruction_en' else INSTRUCTION[src][tgt]}"


def system(arm, src, tgt):
    """The system message, or None."""
    return INSTRUCTION[src][tgt] if arm == "instruction_system" else None


def units(arms=None):
    """{unit: (arm, src, tgt)}: cot in each language (tgt = src), the other arms on each pair of RUN, for the given
    arms (all by default)."""
    arms = arms or ARMS
    u = {f"cot_{L}": ("cot", L, L) for L in LANGS} if "cot" in arms else {}
    u.update({f"{a}_{KEY[p]}": (a, *p) for a in arms if a != "cot" for p in RUN})
    return u


STALE = 15 * 60  # a claim not refreshed for this long is a dead worker's


class Claim:
    """An exclusive claim on a task, kept fresh by a heartbeat thread while held, shared by every worker of every
    job (as in ../mgsm_crosslingual/recover.py), so that helper jobs can join; a dead worker's claim is taken over."""

    def __init__(self, name):
        self.path, self.stop = GEN / "claims" / "shared" / name, threading.Event()

    def take(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > STALE:
            try:  # whoever renames it first takes the task over
                self.path.rename(self.path.with_name(f"{self.path.name}.stale.{uuid.uuid4().hex[:8]}"))
            except FileNotFoundError:
                return False
            print(f"taking over the stale claim {self.path.name}", flush=True)
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return False
        threading.Thread(target=self._beat, daemon=True).start()
        return True

    def running(self):
        return self.path.exists() and time.time() - self.path.stat().st_mtime <= STALE

    def _beat(self):
        while not self.stop.wait(60):
            self.path.touch()

    def release(self):
        self.stop.set()


@functools.cache
def setting(arm):
    """generate's keyword arguments for the arm, besides vecs, and (layer, alpha) for the output: our arms at the
    (layer, alpha) chosen on MGSM dev, ReCoVeR at its alpha chosen
    on MGSM dev (../mgsm_crosslingual/recover.py), all blocks, all tokens, with the norm restore."""
    if arm in STEER:
        ch = json.loads((XL / "chosen.json").read_text())[arm]
        return dict(layer=ch["layer"], prompt_only=True), (ch["layer"], ch["alpha"])
    if arm == "recover":
        ch = json.loads((XL / "recover/chosen.json").read_text())
        return dict(recover=dict(alpha=ch["alpha"], restore_norm=ch["restore_norm"])), ("all", ch["alpha"])
    return {}, (None, None)


@functools.cache
def means(arm):
    import torch
    return torch.load(XL / ("recover/vectors.pt" if arm == "recover" else "vectors.pt"))


def vecs(rows):
    """The vector of each row ({id, arm, src, tgt}; one arm), or None for an unsteered arm."""
    arm = rows[0]["arm"]
    kw, (layer, alpha) = setting(arm)
    if arm == "recover":  # [n_layers + 1, d]: r_T - r_S at every hidden state
        return [means(arm)[r["tgt"]] - means(arm)[r["src"]] for r in rows]
    if arm in STEER:  # as ../mgsm_crosslingual
        return [vector(arm, means(arm), layer, alpha, r, r["src"], r["tgt"])[0] for r in rows]
    return None


def fit(M, texts, frac=0.6):
    """Set generate.MAX_BATCH to the most rows whose KV cache at full length (the longest prompt + the cap) takes
    at most frac of the GPU memory free now (the rest: the prefill's activations, the allocator's slack), at most 512.
    A decoding step costs about the same for few rows as for many (it is bound by the weights' reads and Python),
    so the fewer batches the better."""
    import torch
    c = M.model.config.get_text_config()
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    per_token = 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * 2  # K and V, bf16
    longest = max(len(M.tok(t, add_special_tokens=False).input_ids) for t in texts) + 64  # + the chat template
    torch.cuda.empty_cache()  # free = what this process has cached but not used, too
    free = torch.cuda.mem_get_info()[0]
    generate.MAX_BATCH = max(1, min(512, int(frac * free / (per_token * (longest + generate.MAX_NEW)))))
    return generate.MAX_BATCH


def write(path, rows):
    """Atomic, and never over an existing file (two workers of the same task: the first one's output stays)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    try:
        os.link(tmp, path)
    except FileExistsError:
        pass
    tmp.unlink()


def load(path):
    return [json.loads(l) for l in open(path)]


def done(name):
    """The unit's generation (stage 1 when staged) or its final file exists."""
    return any((d / f"{name}.jsonl").exists() for d in ((GEN / "stage1") if STAGED else GEN, GEN, FULL))


CHUNK = 256  # capped responses per claimable task of stage 2


def chunks(arm):
    """The arm's capped stage-1 responses as [(unit, row index)] chunks (all its stage-1 units must exist)."""
    todo = [(n, k) for n in units((arm,)) for k, r in enumerate(load(GEN / "stage1" / f"{n}.jsonl")) if r["capped"]]
    return [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]


def tasks(arms):
    """[(task, output path, what)] that can run now: every unit not done (stage 1 when staged); when staged, the
    chunks of every arm whose stage-1 units are all done; and whether some arm's chunks are not known yet."""
    us = units(arms)
    out = [(n, (GEN / "stage1" if STAGED else GEN) / f"{n}.jsonl", ("unit", n)) for n in us if not done(n)]
    if POOL:  # one task per arm: its units not done, in one generate call (written by the last unit's path)
        out = [(f"pool_{arm}", ts[-1][1], ("units", [t[0] for t in ts]))
               for arm in arms if (ts := [t for t in out if us[t[0]][0] == arm])]
    waiting = False
    if STAGED:
        for arm in arms:
            names = [n for n, (a, _, _) in us.items() if a == arm]
            if not all(done(n) for n in names):
                waiting = True
            elif not all((GEN / f"{n}.jsonl").exists() for n in names):
                out += [(f"{arm}_{c:04d}", GEN / "stage2" / f"{arm}_{c:04d}.jsonl", ("chunk", ch))
                        for c, ch in enumerate(chunks(arm))]
    return [t for t in out if not t[1].exists()], waiting


def run_unit(M, name, path):
    run_units(M, [name])


def run_units(M, names):
    """Generate units of one arm (one, or with --pool all of them in one call) and write each unit's file."""
    us = units()
    arm = us[names[0]][0]
    assert all(us[n][0] == arm for n in names)
    rows = [dict(r, arm=arm, src=us[n][1], tgt=us[n][2], unit=n) for n in names for r in sample()]
    texts = [prompt(arm, r, r["src"], r["tgt"]) for r in rows]
    kw, (layer, alpha) = setting(arm)
    generate.MAX_NEW = FIRST_CAP or MAX_NEW
    print(",".join(names), "batch", fit(M, texts, 0.5 if arm == "recover" else 0.6), flush=True)
    sys_ = [system(arm, r["src"], r["tgt"]) for r in rows]
    res = M.generate(texts, vecs=vecs(rows), return_ids=True, system=sys_, **kw)
    out = [dict(id=r["id"], category=r["category"], subset=r["subset"], src=r["src"], tgt=r["tgt"], lang=r["tgt"],
                arm=arm, layer=layer, alpha=alpha, answer=r["answer"], n_options=len(r["text"][r["src"]][1]),
                input=x, response=o, n_tokens=k, capped=c) | ({"system": s} if s else {}) | ({"ids": ids} if STAGED and c else {})
           for r, x, s, (o, k, c, ids) in zip(rows, texts, sys_, res)]
    for n in names:  # the last one last: its file marks the pooled task done
        rs = [o for o, r in zip(out, rows) if r["unit"] == n]
        write((GEN / "stage1" if STAGED else GEN) / f"{n}.jsonl", rs)
        print(f"{n}: capped {sum(r['capped'] for r in rs)} | {rs[0]['response'][:80]!r}", flush=True)


def run_chunk(M, name, path, todo):
    """Continue capped stage-1 responses of one arm from their stage-1 tokens (forced: not steered by the
    prompt-only arms, steered as at every generated token by ReCoVeR) to MAX_NEW new tokens in all."""
    rows = [load(GEN / "stage1" / f"{n}.jsonl")[k] for n, k in todo]
    kw = setting(rows[0]["arm"])[0]
    generate.MAX_NEW = MAX_NEW
    print(name, "batch", fit(M, [r["input"] for r in rows], 0.5 if rows[0]["arm"] == "recover" else 0.6), flush=True)
    res = M.generate([r["input"] for r in rows], vecs=vecs(rows), prefix=[r["ids"] for r in rows],
                     system=[r.get("system") for r in rows], **kw)
    # n_tokens: the stage-1 tokens + the continuation's
    write(path, [dict(unit=n, index=k, response=o, n_tokens=len(r["ids"]) + m, capped=cp)
                 for (n, k), r, (o, m, cp, _) in zip(todo, rows, res)])
    print(f"{name}: {len(rows)} continued, still capped {sum(x[2] for x in res)}", flush=True)


def run(arms, helper):
    """Take free tasks until none is left. The main workers then wait for the others' running tasks (taking over
    any whose worker died); a helper leaves when nothing is left for it to take."""
    from generate import Model
    check_cot()
    M = None
    while True:
        todo, waiting = tasks(arms)
        for name, path, (kind, what) in todo:
            c = Claim(name)
            if path.exists() or not c.take():
                continue
            try:
                M = M or Model(MODELS[MODEL]["hf"])
                if kind == "chunk":
                    run_chunk(M, name, path, what)
                else:
                    run_units(M, [what] if kind == "unit" else what)
            finally:
                c.release()
            break
        else:
            if not todo and not waiting:
                return
            if helper and not waiting and all(Claim(n).running() for n, _, _ in todo):
                return
            time.sleep(60)


def merge(arms):
    """Staged only: the final units of every arm whose capped stage-1 responses are all continued, the stage-1
    responses with the capped ones replaced by their continuations, from any stage-2 file (the per-arm chunks, and
    the pooled chunk*.jsonl of the first three arms' run). A continuation longer than MAX_NEW tokens (some ran to
    8192 before the cap was set to 4096) is cut to MAX_NEW and capped: greedy decoding makes its first MAX_NEW
    tokens those of a run to MAX_NEW, and whether a response finishes within MAX_NEW is exact (n_tokens); only the
    cut text is re-tokenized, which matters just for the language of capped responses (scored wrong anyway)."""
    if not STAGED:
        return
    cont = {}
    for f in sorted((GEN / "stage2").glob("*.jsonl")):
        for x in load(f):
            assert (x["unit"], x["index"]) not in cont, (f, x["unit"], x["index"])
            cont[x["unit"], x["index"]] = x
    tok = None
    for arm in arms:
        names = list(units((arm,)))
        if all((GEN / f"{n}.jsonl").exists() for n in names) or not all((GEN / "stage1" / f"{n}.jsonl").exists() for n in names):
            continue
        rows = {n: load(GEN / "stage1" / f"{n}.jsonl") for n in names}
        capped = {(n, k) for n, rs in rows.items() for k, r in enumerate(rs) if r["capped"]}
        if not capped <= cont.keys():
            continue
        for n, k in capped:
            r, x = rows[n][k], cont[n, k]
            # the continuation is generated from r["ids"] itself; this only checks that it is matched to its row.
            # Stage-1 text cut inside a multi-byte character decodes it as U+FFFD, which the continuation's text
            # (decoded with the bytes that follow) has as the real character, so compare the text before it only
            # (Gemma-3-12B, boxed, instruction_en_de-ja row 218: a degenerate response, capped at 4096 either way)
            assert "ids" in r and x["response"].startswith(r["response"][:50].split("�")[0])
            r.update(response=x["response"], n_tokens=x["n_tokens"], capped=x["capped"])
            if r["n_tokens"] > MAX_NEW:
                if tok is None:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(MODELS[MODEL]["hf"])
                ids = tok(r["response"], add_special_tokens=False).input_ids[:MAX_NEW]
                r.update(response=tok.decode(ids, skip_special_tokens=True).strip(), n_tokens=MAX_NEW, capped=True)
        for n, rs in rows.items():
            write(GEN / f"{n}.jsonl", [{k: v for k, v in r.items() if k != "ids"} for r in rs])


# The chosen letter. Llama states it in free form, in any of the languages ("La respuesta correcta es J.",
# "因此，答案是 J. 泰加林。", "Kwa hivyo, jibu sahihi ni chaguo J."), often after going through the options one by
# one ("J. ...: incorrect"), so a letter opening a line is not taken as the answer. In order:
#   boxed  the last \boxed{X}
#   cue    the last answer word (CUES, in any of the 11 languages) and the letter after it, within 40 characters
#          (several different letters there, e.g. "The answer is: G, J", is "multi" and counts as wrong)
#   text   no letter after the last answer word, but exactly one option's text is quoted after it, e.g.
#          "Jibu la mwisho ni 0.0783 k mole/hr." (option text alone, with no answer word, is not used: the
#          responses often end by going through the options)
#   last   no answer word but exactly one letter in the final paragraph, which is not itself an option of a list
#          ("综上所述，最可能的原因是G. ...", "...政治理論（G）です。", "...— вариант J."); "A" or "I" followed by a
#          lowercase word ("A higher price", "I think") is an English word there, not a letter
# else no letter (counts as wrong). Cyrillic and full-width look-alikes of A-J are read as A-J.
CUES = ("answer", "antwort", "respuesta", "réponse", "reponse", "ответ", "jibu", "উত্তর", "సమాధానం", "జవాబు",
        "คำตอบ", "答え", "正解", "答案", "答")
CUE = re.compile("|".join(map(re.escape, CUES)), re.I)
TOKEN = re.compile(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z'’])")
BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])\)?")
LOOKALIKE = str.maketrans("АВСЕНІ" + "ＡＢＣＤＥＦＧＨＩＪ", "ABCEHI" + "ABCDEFGHIJ")


# LaTeX maths ($$...$$, $...$, \[...\], \(...\)), removed before the language judge: its command names and units
# (\times, \frac, \text{J mol}) are Latin words that the judge reads as English lines, so STEM answers heavy in maths
# fail to count as in T (Qwen3-8B, the native question in STEM: 0.83 in T with them, 0.95 without; zh 0.67 -> 0.89)
MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)


def norm(t):
    return " ".join(re.sub(r"[\W_]+", " ", t.lower()).split())


# A box with no letter A-J but a letter of T's script: models steered into T box the option letter in T's script
# (Qwen3-32B, ReCoVeR into ru: 28% of responses). It is read as the Latin letter it transliterates: Cyrillic by ISO 9
# (Г = G, Д = D, Х = H, Й = J), and the letters whose ISO 9 letter is not one of A-J (В v, Ж ž, З z, К k) by their
# place in the alphabet (А Б В Г Д Е Ж З И К = A-J: В = C, Ж = G, З = H, К = J); Bengali and Telugu letters by the
# Latin letter they write (গ ga = G, জ ja = J, হ ha = H, ই i = I; Telugu ఐ, the name of I). Others count as no answer.
# Checked against the gold answers (Qwen3-32B, Gemma-3-12B): each reads about as often right as the Latin boxes.
LOCAL = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([^\W\d_A-Za-z][^\W\d_]{0,2})\s*[).]?\s*\}")
TRANSLIT = dict(zip("АБГДЕИЙФХ", "ABGDEIJFH")) | dict(zip("ВЖЗК", "CGHJ")) | dict(zip("গজহই", "GJHI")) | {"ఐ": "I"}


def letter(response, options):
    """(letter or None, how): how is boxed, cue, text, last, multi or None."""
    r = response.translate(LOOKALIKE)
    if m := list(BOXED.finditer(r)):
        return m[-1].group(1), "boxed"
    cues = list(CUE.finditer(r))
    if cues:
        first = TOKEN.search(r[cues[-1].end():][:40])
        if first:
            # what follows the letter on its line: another letter listed right after it ("G, J", "D, E na G")
            rest = re.split(r"[\n。]", r[cues[-1].end() + first.end():], maxsplit=1)[0]
            if re.match(r"\s*([,，、/]|(and|or|und|y|o|et|ou|и|или|na|au|和|或)\s)\s*\(?[A-J]\)?(?![A-Za-z])", rest):
                return None, "multi"
            return first.group(1), "cue"
    if cues:
        t = norm(r[cues[-1].end():])
        hits = [k for k, o in enumerate(options) if len(norm(o)) >= 4 and norm(o) in t]
        hits = [k for k in hits if not any(k != h and norm(options[k]) in norm(options[h]) for h in hits)]
        if len(hits) == 1:
            return LETTERS[hits[0]], "text"
    else:
        last = [b for b in re.split(r"\n\s*\n", r) if b.strip()][-1].strip()
        if not re.match(r"[*\-\s]*\(?[A-J][.):：]", last):
            found = {m.group(1) for m in TOKEN.finditer(last)
                     if not (m.group(1) in "AI" and re.match(r"\s+[a-zà-ÿ]", last[m.end():]))}
            if len(found) == 1:
                return found.pop(), "last"
    return None, None


def analyse():
    merge(ARMS)
    from analyse import judge, letters  # ../mgsm_steering/analyse.py (fastText; the MGSM judge)
    opts = {r["id"]: r for r in sample()}
    # the units generated so far (the steering arm may not be run on the whole sample)
    files = [p for name in units() if (p := next((d / f"{name}.jsonl" for d in (GEN, FULL)
                                                 if (d / f"{name}.jsonl").exists()), None))]
    rows = [r for p in files for l in open(p) if (r := json.loads(l))["id"] in opts]

    def judged(r, strip):
        """The MGSM judge on the response without its LaTeX maths (MATH); with strip, lines that are mostly a quoted
        option in S are removed too."""
        resp = MATH.sub(" ", r["response"])
        if strip:
            o = [" ".join(letters(MATH.sub(" ", x)).split()) for x in opts[r["id"]]["text"][r["src"]][1]]
            o = [x for x in o if len(x) >= 8]
            keep = []
            for l in resp.split("\n"):
                t = " ".join(letters(l).split())
                if not any(x in t and len(x) >= 0.5 * len(t) for x in o):
                    keep.append(l)
            resp = "\n".join(keep)
        return judge(resp, r["lang"])

    out = {}
    for r in rows:
        # a capped response has no final answer: any letter in it is from the middle of the reasoning
        boxes = BOXED.findall(r["response"])
        r["boxes"] = len(set(boxes))  # distinct letters boxed
        if r["capped"]:
            r["pred"], r["parse"] = None, "capped"
        elif BOXED_FORMAT:  # the format asked for, and nothing else (its letter in T's script: TRANSLIT)
            local = LOCAL.findall(r["response"])
            local = TRANSLIT.get(local[-1]) if local and not boxes else None
            r["pred"], r["parse"] = ((boxes[-1], "boxed") if boxes else (local, "local") if local else
                                     (None, "local_unread") if LOCAL.search(r["response"]) else (None, None))
        else:
            r["pred"], r["parse"] = letter(r["response"], opts[r["id"]]["text"][r["src"]][1])
        r["correct"] = r["pred"] == r["answer"]
        r["in_lang"], r["english"] = judged(r, False)
        r["in_lang_stripped"] = judged(r, True)[0]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["subset"], r["arm"], f"{r['src']}-{r['tgt']}")].append(r)
        g[(r["subset"], r["arm"], "all")].append(r)
        if r["arm"] != "cot":  # the pairs the steering setting was tuned on (on MGSM) and the other 99
            g[(r["subset"], r["arm"], "tune" if (r["src"], r["tgt"]) in TUNE else "held_out")].append(r)
    for (subset, arm, pair), rs in sorted(g.items()):
        n = len(rs)
        s = dict(n=n, accuracy=sum(r["correct"] for r in rs) / n,
                 in_lang=sum(r["in_lang"] for r in rs) / n,
                 success=sum(r["correct"] and r["in_lang"] for r in rs) / n,
                 in_lang_stripped=sum(r["in_lang_stripped"] for r in rs) / n,
                 english=sum(r["english"] for r in rs) / n, capped=sum(r["capped"] for r in rs) / n,
                 unparsed=sum(r["pred"] is None and not r["capped"] for r in rs) / n, text=sum(r["parse"] == "text" for r in rs) / n,
                 last=sum(r["parse"] == "last" for r in rs) / n,
                 local=sum(r["parse"] == "local" for r in rs) / n,
                 local_unread=sum(r["parse"] == "local_unread" for r in rs) / n,
                 boxed=sum(r["boxes"] > 0 for r in rs if not r["capped"]) / max(1, sum(not r["capped"] for r in rs)),
                 several_boxed=sum(r["boxes"] > 1 for r in rs) / n,
                 median_tokens=sorted(r["n_tokens"] for r in rs)[n // 2])
        out.setdefault(subset, {}).setdefault(arm, {})[pair] = s
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    # per item, for the paired tests (tests.py): subset -> arm -> pair -> [[id, correct, in T], ...] by id
    items = {}
    for (subset, arm, pair), rs in g.items():
        if "-" in pair:
            items.setdefault(subset, {}).setdefault(arm, {})[pair] = sorted(
                [r["id"], bool(r["correct"]), bool(r["in_lang"])] for r in rs)
    OUT.with_name(OUT.stem + "_items.json").write_text(json.dumps(items) + "\n")
    for subset in out:
        print(f"\n{subset}: arm, pair: success / accuracy / in-lang (options stripped) / English / capped / "
              f"unparsed (not capped) / by option text / by last paragraph / boxed (not capped) / several boxed")
        for arm in (a for a in ARMS if a in out[subset]):
            for pair, s in out[subset][arm].items():
                print(f"  {arm:11s} {pair:6s} {s['success']:.3f} / {s['accuracy']:.3f} / {s['in_lang']:.3f} "
                      f"({s['in_lang_stripped']:.3f}) / {s['english']:.3f} / {s['capped']:.3f} / "
                      f"{s['unparsed']:.3f} / {s['text']:.3f} / {s['last']:.3f} / {s['boxed']:.3f} / {s['several_boxed']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "analyse"))
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--first", type=int)
    ap.add_argument("--first-cap", type=int)
    ap.add_argument("--pairs", choices=("tune", "all"), default="tune")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="run: comma-separated arms (default all but instruction_system)")
    ap.add_argument("--format", choices=("free", "boxed"), default="free",
                    help="boxed: ask for the answer as \\boxed{X} and score the last one (outputs *_boxed)")
    ap.add_argument("--pool", action="store_true", help="run: all units of an arm in one generate call")
    ap.add_argument("--helper", action="store_true", help="run: leave when nothing is left to take")
    a = ap.parse_args()
    STAGED, FIRST_CAP, MAX_NEW = a.first_cap is not None, a.first_cap, a.max_new
    BOXED_FORMAT, POOL = a.format == "boxed", a.pool
    suffix = ("" if a.max_new == 2048 else f"_{a.max_new}") + (f"_from{a.first_cap}" if STAGED else "")
    suffix += "_boxed" if BOXED_FORMAT else ""  # before FULL: never reuse generations of the other prompt
    FIRST, FULL = a.first, RES / "generations" / f"pilot{suffix}"
    suffix += f"_first{a.first}" if a.first else ""
    RUN = PAIRS if a.pairs == "all" else TUNE
    GEN, OUT = RES / "generations" / f"pilot{suffix}", RES / f"pilot{suffix}{'_all' if a.pairs == 'all' else ''}.json"
    arms = tuple(a.arms.split(","))
    assert set(arms) <= set(ARMS), arms
    run(arms, a.helper) if a.stage == "run" else analyse()
