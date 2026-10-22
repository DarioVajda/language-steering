"""Restore casing and detokenise the lowercased, pre-tokenised MSVAMP English text.

MSVAMP's English source ("julia played tag with 18 kids on monday . she ...") is a
lowercased, tokenised copy of SVAMP (Patel et al., 2021). The original SVAMP release
is cased, but 58/1000 MSVAMP items were edited relative to it, so SVAMP text cannot
simply replace MSVAMP text. Instead, SVAMP is used only as a *casing reference*:

1. Match each MSVAMP item to its SVAMP item (exact match on lowercased alphanumerics,
   else the most similar unused SVAMP item).
2. Align the two token sequences (difflib) and copy SVAMP's casing onto every MSVAMP
   token whose lowercase form matches. Words are never changed, only their case.
3. Rules on top: capitalise sentence starts, "i", weekdays, unambiguous month
   names (SVAMP itself writes "monday"; "may"/"march" are left alone), and, for
   tokens that could not be aligned in step 2 (MSVAMP edits), words that SVAMP never
   writes in lowercase (names such as "Adam").
4. Detokenise ("monday ." -> "monday.", "rachel 's" -> "rachel's", "$ 3" -> "$3").

The procedure is deterministic given the pinned SVAMP commit (see download.py).
"""

import difflib
import re

TOKEN = re.compile(r"\w+|[^\w\s]")
ALWAYS_CAPITALISED = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "april", "june", "july", "august", "september",
    "october", "november", "december",
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def svamp_text(item):
    """SVAMP Body + Question. When the Body has no final punctuation the two form one
    sentence ("... on each pack How much ..."), so the Question is not capitalised."""
    body, question = item["Body"].strip(), item["Question"].strip()
    if not body.endswith((".", "?", "!")):
        question = question[:1].lower() + question[1:]
    return body + " " + question


def proper_nouns(svamp):
    """Lowercased forms of the words that SVAMP never writes in lowercase."""
    capitalised, lowercased = set(), set()
    for item in svamp:
        for t in TOKEN.findall(svamp_text(item)):
            if t[:1].isalpha():
                (lowercased if t[:1].islower() else capitalised).add(t.lower())
    return capitalised - lowercased


def match_svamp(queries, svamp):
    """Return, for each MSVAMP query, the matching SVAMP text and whether the match was exact."""
    by_key = {norm(svamp_text(s)): s for s in svamp}
    matches = [by_key.get(norm(q)) for q in queries]
    used = {s["ID"] for s in matches if s is not None}
    unused = [s for s in svamp if s["ID"] not in used]
    out = []
    for q, s in zip(queries, matches):
        if s is not None:
            out.append((svamp_text(s), True))
            continue
        best = max(unused, key=lambda u: difflib.SequenceMatcher(None, norm(q), norm(svamp_text(u))).ratio())
        unused.remove(best)
        out.append((svamp_text(best), False))
    return out


def detokenise(s):
    s = re.sub(r"\s+([.,?!;:%)])", r"\1", s)
    s = re.sub(r"([($])\s+", r"\1", s)
    s = re.sub(r"\s+('s|'re|'ll|'ve|'m|'d|n't)\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d) ([.,]) (?=\d)", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def truecase(query, reference, names=frozenset()):
    """Case `query` like `reference`; return (text, n_tokens, n_tokens_cased_from_reference).
    `names` is the output of `proper_nouns`."""
    q_spans = [(m.start(), m.end()) for m in TOKEN.finditer(query)]
    q_tok = [query[a:b] for a, b in q_spans]
    r_tok = TOKEN.findall(reference)
    cased = [t.lower() for t in q_tok]
    sm = difflib.SequenceMatcher(None, cased, [t.lower() for t in r_tok], autojunk=False)
    aligned = [False] * len(cased)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            cased[i1:i2] = r_tok[j1:j2]
            aligned[i1:i2] = [True] * (i2 - i1)
    sentence_start = True
    for k, t in enumerate(cased):
        if t == "i":
            cased[k] = "I"
        name = not aligned[k] and t.lower() in names
        if t.lower() in ALWAYS_CAPITALISED or name or (sentence_start and t[:1].isalpha()):
            cased[k] = t[:1].upper() + t[1:]
        if t[:1].isalnum():
            sentence_start = False
        elif t in {".", "?", "!"}:
            sentence_start = True
    out, last = [], 0
    for (a, b), t in zip(q_spans, cased):
        out.append(query[last:a] + t)
        last = b
    return detokenise("".join(out) + query[last:]), len(q_tok), sum(aligned)
