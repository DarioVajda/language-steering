# Data

Three parallel benchmarks used by [Kirtane and Huang (2026)](../../README.md#kirtane-2026): MGSM, MSVAMP and XNLI.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # from the project root
.venv/bin/python src/data/download.py     # -> data/raw_data/       (~24 MB, plus ~13 MB for MKQA and fastText; pinned in manifest.json; or name sources, e.g. `download.py mkqa`)
.venv/bin/python src/data/preprocess.py   # -> data/processed_data/ (one parallel JSONL per benchmark)
.venv/bin/python src/data/analyze.py      # -> stats tables on stdout + data/processed_data/stats.json
```

| Benchmark | HF source | Items | Languages | Kind |
|---|---|---|---|---|
| MGSM | `juletxara/mgsm` | 250 (+ 8 CoT exemplars) | en, bn, de, es, fr, ja, ru, sw, te, th, zh | Human translations of GSM8K |
| MSVAMP | `Mathoctopus/MSVAMP` | 1000 | en, bn, de, es, fr, ja, ru, sw, th, zh | **Machine** translations of SVAMP |
| XNLI | `facebook/xnli` | 2490 validation + 5010 test | 15 (ar, bg, de, el, en, es, fr, hi, ru, sw, th, tr, ur, vi, zh) | Human translations of MultiNLI |
| SVAMP | GitHub `arkilpatel/SVAMP` @ `78e72768` | 1000 | en | Original cased SVAMP, used only to truecase MSVAMP English |
| MKQA | GitHub `apple/ml-mkqa` @ `f964fe1b` | 10,000 | 26, incl. en, fr, hu, ru, zh_cn | Human translations of Natural Questions queries with localized answers; used raw by [output_language](../experiments/output_language/README.md) |
| fastText `lid.176.ftz` | `dl.fbaipublicfiles.com` (SHA-256 in manifest) | – | 176 | Language-ID model, not a benchmark; used by output_language |

## Format

One row per item, with the text in every language. Items are parallel by index in all three raw datasets, and `preprocess.py` asserts this. Example rows, shortened to three languages:

```jsonc
// mgsm.jsonl
{"id": "mgsm-0001", "split": "val", "answer": 3,
 "text": {"en": {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"},
          "zh": {"question": "制作一件袍子需要 2 匹蓝色纤维布料和这个数量一半的白色纤维布料。它一共需要用掉多少匹布料？"},
          "sw": {"question": "Joho hutumia komeo 2 za ufumwele wa buluu na nusu ya kiasi hicho cha ufumwele mweupe. Huwa inatumia jumla ya komeo ngapi?"}}}

// msvamp.jsonl  (English is truecased; the raw lowercased source is kept in en_question_original)
{"id": "msvamp-0000", "split": "val", "answer": 8, "equation": "- number0 number1",
 "en_question_original": "julia played tag with 18 kids on monday . she played tag with 10 kids on tuesday . how many more kids did she play with on monday than on tuesday ?",
 "text": {"en": {"question": "Julia played tag with 18 kids on Monday. She played tag with 10 kids on Tuesday. How many more kids did she play with on Monday than on Tuesday?"},
          "de": {"question": "Julia spielte am Montag mit 18 Kindern Fangen. Sie spielte am Dienstag mit 10 Kindern Fangen. Mit wie vielen Kindern hat sie am Montag mehr gespielt als am Dienstag?"},
          "th": {"question": "จูเลียเล่นเกมแท็กกับเด็ก 18 คนในวันจันทร์ เธอเล่นเกมแท็กกับเด็ก 10 คนในวันอังคาร จูเลียเล่นกับเด็กกี่คนมากกว่าในวันจันทร์?"}}}

// xnli.jsonl
{"id": "xnli-validation-0000", "split": "analysis", "label": "neutral",
 "text": {"en": {"premise": "And he said, Mama, I'm home.", "hypothesis": "He called his mom as soon as the school bus dropped him off."},
          "ru": {"premise": "И он сказал: Мама, я дома.", "hypothesis": "Он позвал маму, как только вышел из школьного автобуса."},
          "zh": {"premise": "他说，妈妈，我回来了。", "hypothesis": "校车把他放下后，他立即给他妈妈打了电话。"}}}

// a flagged MSVAMP row: the Thai translation lost everything but the final question
{"id": "msvamp-0260", ..., "flags": {"th": ["too_short"]},
 "text": {"en": {"question": "The grasshopper the frog and the mouse had a jumping contest. The grasshopper jumped 25 inches. [...] How far did the mouse jump?"},
          "th": {"question": "มวยได้ระยะทางเท่าไหร่?"}}}
```

`mgsm_exemplars.jsonl` contains the 8 official CoT exemplars verbatim, as `{"question", "answer"}` per language. They keep their localised `Question:` / `Step-by-Step Answer:` prefixes.

## MSVAMP English Truecasing

The raw MSVAMP English is a lowercased, pre-tokenised copy of SVAMP, e.g. `julia played tag with 18 kids on monday . she ...`. The translations are normally cased. Left as is, English vs. target-language hidden states would also differ in casing and tokenisation, and that difference would leak into the language directions.

**Why SVAMP can't simply replace the text:** the original cased SVAMP release can't be used as a drop-in replacement. Only 942 of 1000 items match it exactly after lowercasing. The other 58 were edited by MSVAMP (e.g. "his hair" became "her hair", "the same number of pages" became "450 pages"), and the translations follow the MSVAMP wording.

**What `truecase.py` does instead:** it uses SVAMP only as a **casing reference**, and never changes a word.

1. Match each item to SVAMP, exactly or else by the most similar unused item.
2. Align the tokens and copy SVAMP's casing. This covers 99.7% of tokens.
3. Apply rules on top:
   * capitalise sentence starts, "I", weekdays and months;
   * for unaligned tokens only, capitalise words SVAMP never lowercases (names).
4. Detokenise (`monday .` → `Monday.`, `rachel 's` → `Rachel's`).

The steps are deterministic given the pinned SVAMP commit. `preprocess.py` asserts that the alphanumeric content of every item is unchanged, and the raw text stays in `en_question_original` for exact ICL replication.

Residual quirks come from the source text, not from truecasing:
* typos ("wtih", "larence county");
* run-on sentences where SVAMP joins Body and Question without punctuation ("If there are 739 girls how many boys …");
* a few MSVAMP edits that dropped words ("… have 2 apples. Has 6 apples more than …").

## Splits

The split sizes follow Kirtane & Huang (Table 14). Their exact item IDs are not public, so the splits here are a seed-0 random partition.

| Benchmark | compute | val | test | other |
|---|---|---|---|---|
| MGSM | 83 | 83 | 84 | – |
| MSVAMP | 333 | 333 | 334 | – |
| XNLI (1000-item test subsample) | 333 | 333 | 334 | 4010 test items → `unused`; 2490 validation items → `analysis` (for EXP1) |

Each XNLI premise comes with 3 hypotheses. The XNLI split is therefore made **by premise**, so that no premise appears in more than one of compute, val, test and analysis. A plain item-level split would have shared 33 premises between compute and test.

## Brief Analysis

Statistics for the EXP1 languages are below; `analyze.py` prints all languages. The columns are:

* **chars** – mean characters per item.
* **len/en** – mean ratio of character length to the English version.
* **digits_ok** – share of items whose numbers written in digits match English exactly.
* **flagged** – items whose translation is probably broken (length outlier or repetition loop).

| lang | MGSM chars | MGSM len/en | MGSM digits_ok | MSVAMP chars | MSVAMP digits_ok | MSVAMP flagged | XNLI chars | XNLI digits_ok | XNLI flagged |
|---|---|---|---|---|---|---|---|---|---|
| en | 240.7 | 1.00 | – | 162.2 | – | – | 159.5 | – | – |
| de | 270.9 | 1.13 | 98.0% | 178.9 | 98.8% | 1 | 181.6 | 97.3% | 0 |
| es | 257.0 | 1.07 | 98.0% | 170.8 | 99.3% | 0 | 174.6 | 98.1% | 0 |
| fr | 266.0 | 1.11 | 93.2% | 187.6 | 99.3% | 0 | 185.0 | 98.5% | 0 |
| ru | 246.7 | 1.02 | 97.6% | 162.9 | 98.6% | 4 | 166.8 | 97.1% | 0 |
| sw | 265.5 | 1.11 | 98.8% | 171.8 | 98.7% | 1 | 158.2 | 97.0% | 0 |
| th | 213.5 | 0.90 | 96.8% | 139.5 | 95.3% | 6 | 144.7 | 97.9% | 0 |
| zh | 85.2 | 0.36 | 81.2% | 52.9 | 98.2% | 0 | 50.7 | 93.4% | 1 |

**Takeaways:**

* **Size:** all texts are short, typically 30–50 English words per item. The EXP1 word pool is roughly 250 + 1000 + 2490 parallel items per language.
* **Translation quality:** MGSM and XNLI are clean, since they are human-translated. Their `digits_ok` gaps are mostly benign:
  * English spells out numbers ("three") that zh and ja write as digits.
  * French writes times as "13 h", and Japanese uses units like 8万.

  MSVAMP is machine-translated and has real failures. **9 bn, 6 th, 4 ru, 1 de and 1 sw items are flagged**, e.g. degenerate "ভেসে ভেসে ভেসে…" loops and truncation down to the final question. XNLI has 7 flagged ur items and 1 zh item. Rows are kept, and filtering on `flags` is up to the consumer.
* **The MSVAMP English source is lowercased and pre-tokenised** in 100% of items. It is truecased during preprocessing (see above). For exact ICL replication, use `en_question_original`.
* **Scripts without spaces:** zh, ja and th have essentially no whitespace words (MSVAMP zh averages 1.0 "word" per item). They need a word segmenter (jieba, PyThaiNLP) before alignment.
* **MSVAMP is templated:** SVAMP contains many variants of one story with different numbers, e.g. msvamp-0260…0272. Near-duplicates therefore cross the compute/val/test splits.
* **Labels and answers:** XNLI labels are balanced, exactly 830 per class in `analysis` and 101–122 per class in the ICL splits. All MGSM and MSVAMP answers are integers.
