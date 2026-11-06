"""Cross-lingual part of the Language Confusion Benchmark (Marchisio et al., 2024).

Each of the three sources (okapi, sharegpt, complex_prompts) has the same ~100 English prompts for
every target language, differing only in the language named in the instruction ("Write in French.").
Items are split by a hash of (source, row), so dev and test hold the same prompts in every language:
  dev   crc32 % 3 == 0   (about a third; used to choose languages and steering settings)
  test  the rest
"""
import csv
import io
import re
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LCB = ROOT / "data/raw_data/lcb/test_sets.zip"
SOURCES = ("okapi", "sharegpt", "complex_prompts")
LANGS = ("fr", "de", "es", "pt", "it", "ja", "ko", "zh", "ar", "tr", "hi", "ru", "id", "vi")


def split_of(source, row):
    return "dev" if zlib.crc32(f"{source}:{row}".encode()) % 3 == 0 else "test"


def load(split, langs=LANGS):
    """[{id, source, row, lang, prompt, location, phrasing}] for the given split."""
    out = []
    with zipfile.ZipFile(LCB) as z:
        for src in SOURCES:
            for L in langs:
                with z.open(f"test_sets/crosslingual/{src}/{L}.csv") as f:
                    for row, r in enumerate(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))):
                        if split_of(src, row) == split:
                            out.append(dict(id=f"{src}:{row}", source=src, row=row, lang=L, prompt=r["prompt"],
                                            location=r["location"], phrasing=r["phrasing"]))
    return out


NAMES = {"fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "ja": "Japanese",
         "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "tr": "Turkish", "hi": "Hindi", "ru": "Russian",
         "id": "Indonesian", "vi": "Vietnamese"}
# words allowed in a sentence that is only the language instruction ("Make sure to respond in Russian.")
_INSTR_WORDS = {"write", "reply", "respond", "answer", "anwser", "use", "make", "sure", "to", "your", "you", "should",
                "output", "be", "all", "of", "this", "response", "please", "the", "written", "in", "using", "for",
                "summary"}


def strip_instruction(prompt, lang):
    """The prompt without its language instruction: a sentence that is only the instruction is removed
    whole ("Use Russian."), otherwise the phrase inside the task is ("Write an essay in Russian about ...")."""
    n = re.escape(NAMES[lang])
    for m in re.finditer(r"(?:^|(?<=[.?!:\"”\n]))([ \t]*)((?:[A-Za-z]+ ){0,6})" + n + r"((?: for your [a-z]+)?)[ \t]*(?:[.!]|(?=\n)|$)", prompt):
        if all(w.lower() in _INSTR_WORDS for w in (m.group(2) + m.group(3)).split()):
            return (prompt[:m.start()] + prompt[m.end():]).strip()
    s, k = re.subn(r"[ \t]+(?:and (?:be )?)?(?:written in|using|in|use)[ \t]+" + n + r"\b", "", prompt, count=1)
    if not k:
        raise ValueError(f"no language instruction found: {prompt[:100]!r}")
    return s.strip()
