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
    from GSM8K train instead, which has no translations for the oracle condition.

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
        answer = f"\nAnswer: {item.cot}" if item.cot else ""
        return f"Question: {item.text[lang]['question']}{answer}\nFinal answer: {item.answer}"

    def format_query(self, item, lang):
        return f"Question: {item.text[lang]['question']}\nAnswer:"

    def parse(self, response):
        """The number after the last "Final answer:", else the last number in the response."""
        nums = re.findall(r"Final answer:\s*\$?\s*(-?[\d,]*\.?\d+)", response) or re.findall(r"-?[\d,]*\.?\d+", response)
        if not nums:
            return None
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None

    def is_correct(self, pred, item):
        return pred is not None and abs(pred - item.answer) < 1e-6


class MGSM(Math):
    name = "mgsm"


class MSVAMP(Math):
    name = "msvamp"


TASKS = {t.name: t for t in (XNLI, MGSM, MSVAMP)}
