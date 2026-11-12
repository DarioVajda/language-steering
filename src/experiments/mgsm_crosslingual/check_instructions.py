"""Check the 100 instruction strings INSTRUCTION[S][T] ("Answer in <T>." written in S).

For each string:
  * back-translation S -> English with NLLB-200 3.3B (greedy); it passes if it names T (or a common alternative
    name) and says answer/respond/reply;
  * the English instruction "Answer in <T>." translated English -> S with the same model, for comparison;
  * the fastText lid.176.bin label of the string (informative only: short strings are often mislabelled).
Writes results/instructions.json; README.md lists the strings and back-translations.

Usage: python src/experiments/mgsm_crosslingual/check_instructions.py
"""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mgsm_steering"))
from analyse import lid  # noqa: E402
from generate import FLORES_CODE, NAMES  # noqa: E402
from generate_pairs import INSTRUCTION, LANGS, SOURCES  # noqa: E402

NLLB = "facebook/nllb-200-3.3B"
EN_NAME = {"en": "English", **NAMES}
ALIASES = {"bn": ("bengali", "bangla"), "sw": ("swahili", "kiswahili"), "zh": ("chinese", "mandarin")}


def translate(tok, model, texts, src, tgt):
    tok.src_lang = FLORES_CODE[src]
    b = tok(texts, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**b, forced_bos_token_id=tok.convert_tokens_to_ids(FLORES_CODE[tgt]), num_beams=1,
                         do_sample=False, max_new_tokens=32)
    return tok.batch_decode(out, skip_special_tokens=True)


@torch.inference_mode()
def main():
    tok = AutoTokenizer.from_pretrained(NLLB)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB, dtype=torch.bfloat16).cuda().eval()
    res, n_fail = {}, 0
    for s in SOURCES:
        ts = [t for t in LANGS if t != s]
        back = translate(tok, model, [INSTRUCTION[s][t] for t in ts], s, "en")
        fwd = translate(tok, model, [f"Answer in {EN_NAME[t]}." for t in ts], "en", s)
        res[s] = {}
        for t, b, f in zip(ts, back, fwd):
            low = b.lower()
            ok = (any(n in low for n in ALIASES.get(t, (EN_NAME[t].lower(),)))
                  and re.search(r"\b(answer|respond|reply)", low) is not None)
            n_fail += not ok
            label, p = lid(INSTRUCTION[s][t])
            res[s][t] = dict(instruction=INSTRUCTION[s][t], back_translation=b, nllb_from_english=f, passes=ok,
                             fasttext=label, fasttext_p=round(p, 3))
            print(f"{'ok  ' if ok else 'FAIL'} {s}->{t}  {INSTRUCTION[s][t]!r:40} -> {b!r:40}  "
                  f"NLLB en->{s}: {f!r}  fastText {label} {p:.2f}", flush=True)
    print(f"{n_fail} of {sum(map(len, res.values()))} back-translations fail")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/instructions.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
