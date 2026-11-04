"""Benchmarks for the ICL experiments: items, demonstrations, prompt text and scoring.

Each task reads its parallel JSONL from data/processed_data/ (see src/data/README.md) and
uses the compute / val / test splits of Kirtane & Huang (2026, Table 14):
  * compute: demonstrations (XNLI) and language vectors (all benchmarks);
  * val:     hyperparameter tuning;
  * test:    final evaluation.

Prompt formats follow Kirtane & Huang (2026, App. H). Field names ("Premise:", "Question:")
and labels are always English; only the item text changes language. Demonstrations:
  * XNLI:  6 items from the compute split, with English labels.
  * MGSM / MSVAMP: 6 of the 8 official MGSM exemplars, with their English chain of thought
    (CoT is English in every condition, as in the paper). Kirtane & Huang draw English demos
    from GSM8K train instead, which has no translations for the oracle condition. The CoT of a
    demonstration in another language is labelled as English (`demo_spans`), so per-token
    language steering (mfs_steer) treats it as the English text it is.

Usage:
    from tasks import TASKS
    task = TASKS["xnli"]()
    task.items("val", lang="zh")
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed_data"
K_SHOT = 6


@dataclass(frozen=True)
class Item:
    id: str
    text: dict              # lang -> {"question": ...} or {"premise": ..., "hypothesis": ...}
    answer: object          # int (math) or label (XNLI)
    flags: frozenset = frozenset()   # languages whose translation is broken
    cot: str | None = None  # English chain of thought (MGSM exemplars only)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Task:
    name: str
    instruction: str
    max_new_tokens: int
    stop: tuple = ()        # strings that end a generation (the model starting a new item)
    labels: tuple = ()      # closed label set, for scoring with a single forward pass
    mode = "generate"       # default scoring mode

    def __init__(self):
        self.rows = read_jsonl(PROC / f"{self.name}.jsonl")
        self.langs = tuple(self.rows[0]["text"])

    def items(self, split, lang=None):
        """Items of a split; with `lang`, those whose `lang` translation is not flagged."""
        items = [Item(r["id"], r["text"], r.get("label", r.get("answer")), frozenset(r.get("flags", {})))
                 for r in self.rows if r["split"] == split]
        return [it for it in items if lang is None or lang not in it.flags]

    def demo_pool(self):
        """Candidate demonstrations, unflagged in every language."""
        return [it for it in self.items("compute") if not it.flags]

    def format_demo(self, item, lang):
        raise NotImplementedError

    def demo_spans(self, item, lang):
        """(start, end, lang) character ranges of format_demo(item, lang) in another language than
        `lang` (see model.Segment.spans). Default: none, the whole demonstration is in `lang`."""
        return ()

    def format_query(self, item, lang):
        raise NotImplementedError

    def parse(self, response):
        raise NotImplementedError

    def is_correct(self, pred, item):
        return pred is not None and pred == item.answer


class XNLI(Task):
    name = "xnli"
    instruction = ("You are a helpful assistant that performs natural language inference. "
                   "Given a premise and a hypothesis, decide whether the hypothesis is entailed by the premise "
                   "(entailment), contradicts it (contradiction), or neither (neutral). "
                   "Answer with only: entailment, neutral, or contradiction.")
    max_new_tokens = 8
    labels = ("entailment", "neutral", "contradiction")

    def format_demo(self, item, lang):
        return f"{self.format_query(item, lang)} {item.answer}"

    def format_query(self, item, lang):
        t = item.text[lang]
        return f"Premise: {t['premise']}\nHypothesis: {t['hypothesis']}\nLabel:"

    def parse(self, response):
        """First word of the response, if it is a label (Kirtane & Huang, App. H)."""
        words = re.findall(r"[a-z]+", response.lower())
        return words[0] if words and words[0] in self.labels else None


class Math(Task):
    instruction = ("You are a helpful assistant that solves math word problems step by step. "
                   "Show your reasoning clearly and end with 'Final answer: <number>'.")
    max_new_tokens = 512
    stop = ("\nQuestion:",)

    def demo_pool(self):
        """The 8 MGSM CoT exemplars: localised question prefix dropped, English CoT kept."""
        pool = []
        for r in read_jsonl(PROC / "mgsm_exemplars.jsonl"):
            text = {L: {"question": re.sub(r"^[^:：]{1,40}[:：]\s*", "", t["question"])}
                    for L, t in r["text"].items() if L in self.langs}
            cot = re.sub(r"^Step-by-Step Answer:\s*", "", r["text"]["en"]["answer"])
            cot = re.sub(r"\s*The answer is [^.]*\.\s*$", "", cot)
            pool.append(Item(r["id"], text, r["answer"], cot=cot))
        return pool

    def format_demo(self, item, lang):
        """Demo with CoT; compute items have no CoT and get only the final answer (used for vectors)."""
        return "".join(self._demo_parts(item, lang))

    def _demo_parts(self, item, lang):
        """format_demo split into (question, English CoT, final answer)."""
        answer = f"\nAnswer: {item.cot}" if item.cot else ""
        return f"Question: {item.text[lang]['question']}", answer, f"\nFinal answer: {item.answer}"

    def demo_spans(self, item, lang):
        """The English chain of thought ("\nAnswer: ..."), when the demonstration is in another language."""
        question, cot, _ = self._demo_parts(item, lang)
        return ((len(question), len(question) + len(cot), "en"),) if cot and lang != "en" else ()

    def format_query(self, item, lang):
        return f"Question: {item.text[lang]['question']}\nAnswer:"

    # A number in any locale: groups of three split by ",", "." or a space ("1,234", "1.234.567",
    # "6 277"), then an optional decimal part ("26,00", "2.5"). \d also matches native digits (৩২).
    SEP = "[    ]"
    NUM = re.compile(rf"-?\d+(?:(?:,|\.|{SEP})\d{{3}}(?!\d))*(?:[.,]\d+)?")
    MARK = re.compile(r"Final answer\s*[:：]\s*[*$€£]*\s*")

    @classmethod
    def to_float(cls, tok):
        """Locale-aware: "1,234" and "1.234.567" are thousands, "26,00" and "12,8" decimal commas."""
        t = re.sub(cls.SEP, "", tok)
        if "," in t and "." in t:                    # the later mark is the decimal one
            t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
        elif "," in t:
            t = t.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", t) else t.replace(",", ".")
        elif t.count(".") > 1:
            t = t.replace(".", "")
        try:
            return float(t)
        except ValueError:
            return None

    def parse(self, response):
        """The number right after the last "Final answer:", else the last number in the response.
        Locale-aware, so de/es/fr/ru number formats ("26,00", "6 277") are not misread."""
        marks = list(self.MARK.finditer(response))
        if marks and (m := self.NUM.match(response, marks[-1].end())):
            return self.to_float(m.group())
        nums = self.NUM.findall(response)
        return self.to_float(nums[-1]) if nums else None

    def is_correct(self, pred, item):
        return pred is not None and abs(pred - item.answer) < 1e-6


class MGSM(Math):
    name = "mgsm"


class MSVAMP(Math):
    name = "msvamp"


TASKS = {t.name: t for t in (XNLI, MGSM, MSVAMP)}
