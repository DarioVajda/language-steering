# Cross-lingual instruction vs language steering on MGSM

The question is in a source language S and the answer is wanted in a target language T. Can a language vector make Llama-3.1-8B-Instruct **answer in T, where an instruction fails?** The measure is *success*, as in `../mgsm_steering/`: an answer counts only if it is mathematically correct and written in the wanted language.

**Result: here steering helps.** In the en→T experiment the instruction was in T 99% of the time. With the question in S it is in T only 52% of the time: the model mostly ignores the request and answers in S, with its maths intact. On test, the success rate averaged over the 10 pairs is:
* **0.349 with the instruction alone** (accuracy 0.665);
* **0.618 when prompt-only steering is added on top**, significantly better than the instruction in 8 of 10 pairs and worse in none;
* 0.724 for the native question (in T), the ceiling.

## Reproduce

```bash
G=src/experiments/mgsm_crosslingual
slurm/submit.sh -J mgsm_crosslingual -t 12:00:00 --gres gpu:2 --gres-flags=disable-binding -c 8 --mem 128G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- bash -c 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1; G=src/experiments/mgsm_crosslingual; python $G/generate_pairs.py vectors || exit 1; for i in 0 1 2 3; do (CUDA_VISIBLE_DEVICES=$((i / 2)) python $G/generate_pairs.py pilot && CUDA_VISIBLE_DEVICES=$((i / 2)) python $G/generate_pairs.py dev) & done; wait; python $G/analyse_pairs.py --select || exit 1; for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((i / 2)) python $G/generate_pairs.py test & done; wait; python $G/analyse_pairs.py'
slurm/submit.sh -J mgsm_xl_check -t 00:30:00 -c 4 --mem 48G --gres-flags=disable-binding -- \
  bash -c 'HF_HUB_OFFLINE=1 python src/experiments/mgsm_crosslingual/check_instructions.py'
```

One job runs every stage: vectors, the instruction pilot on all 100 pairs, the dev grid, selection, test and the tables. Workers share each stage through claim files, as in `../mgsm_steering/`. The reported run took 2 h 43 min on 2 B200, mostly the dev grid. The second job checks the instruction strings (below).

| File | Contents |
|---|---|
| `generate_pairs.py` | pairs, instruction strings, prompts, vectors, work units; model, split, grid and steering are imported from `../mgsm_steering/generate.py` |
| `analyse_pairs.py` | pilot matrices, selection, McNemar tests, tables; scoring is imported from `../mgsm_steering/analyse.py` |
| `check_instructions.py` | NLLB-200 back-translation of the 100 instruction strings |
| `results/summary.json` | every number below (the generations in `results/generations/` are not committed) |
| `results/instructions.json` | the instruction strings, back-translations and fastText labels |

## Setup

Everything is as in `../mgsm_steering/` (model, greedy decoding with at most 2048 new tokens, the 50 dev / 200 test split, the grid, selection, success and the McNemar test), with English replaced by S.

* **Pairs:** S is one of the 10 non-English MGSM languages; T is any other of the 11. Of these 100 pairs, 10 were drawn uniformly with a fixed seed before any generation: ja→zh, zh→sw, ru→es, bn→ru, fr→es, sw→ja, sw→de, ru→bn, fr→th, sw→bn.
* **Arms (for a pair S→T):**

  | Name | What the model is given |
  |---|---|
  | Source | the question in S, judged in S (reference) |
  | Native | the question in T |
  | Instruction | the S question + "\n\n" + "Answer in T." written in S (baseline) |
  | Steering | the S question, + α·v at layer l |
  | Instruction + steering | the instruction prompt, + α·v at layer l |

  Each steering arm comes in an all-token and a prompt-only version.
* **Vectors:** v_(S→T)(l) = μ_T(l) − μ_S(l) over the FLORES-200 dev sentences.
* **Instruction strings:** the name of T in S from CLDR 47, in a per-language imperative template (Russian takes the prepositional case). NLLB-200 3.3B translates all 100 back into a correct English instruction. The weakest is th→te ("The answer is Telugu."), which appears only in the pilot.

  | Pair | Instruction | Back-translation |
  |---|---|---|
  | ja→zh | 中国語で答えてください。 | Please answer in Chinese. |
  | zh→sw | 请用斯瓦希里语回答。 | Please answer in Swahili. |
  | ru→es | Отвечай на испанском языке. | Answer in Spanish. |
  | bn→ru | রুশ ভাষায় উত্তর দিন। | Answer in Russian. |
  | fr→es | Réponds en espagnol. | Answer in Spanish. |
  | sw→ja | Jibu kwa Kijapani. | Answer in Japanese. |
  | sw→de | Jibu kwa Kijerumani. | Answer in German. |
  | ru→bn | Отвечай на бенгальском языке. | Answer in Bengali. |
  | fr→th | Réponds en thaï. | Answer in Thai. |
  | sw→bn | Jibu kwa Kibengali. | Answer in Bengali. |

## Results

### Pilot: the instruction on all 100 pairs (dev, 50 items each; not used to choose anything)

* **The source language matters most.** From sw the instruction never works (0.00 for every T); from bn, te and th it almost never does; from de, es, fr, ru and zh it mostly does.
* **"Answer in English" fails from every source** (success at most 0.06): the model answers in S.
* Accuracy is 0.36–0.78 in every cell, so the failures are about language, not maths.
* The full matrices (success, accuracy, in T) are in `summary.json` (`pilot`).

### Dev selection (mean success over the 10 pairs; source 0.652, native 0.672, instruction 0.340)

| Arm | Chosen | Dev success |
|---|---|---|
| Steering, all-token | l 16, α 1 | 0.476 |
| Instruction + steering, all-token | l 20, α 0.5 | 0.554 |
| Steering, prompt-only | l 16, α 2 | 0.518 |
| Instruction + steering, prompt-only | l 16, α 1 | 0.578 |

Stronger α than in en→T, where α 0.25 won with the instruction. Near the optimum the layer matters little (neighbouring layers are within 0.01–0.03).

### Test (200 items per pair): success

| | ja→zh | zh→sw | ru→es | bn→ru | fr→es | sw→ja | sw→de | ru→bn | fr→th | sw→bn | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Source (in S) | 0.62 | 0.76 | 0.77 | 0.65 | 0.75 | 0.64 | 0.64 | 0.77 | 0.75 | 0.64 | 0.697 |
| Native | **0.76**↑ | 0.64 | **0.83**↑ | **0.77**↑ | **0.83**↑ | **0.62**↑ | **0.78**↑ | 0.65 | **0.73**↑ | **0.65**↑ | 0.724 |
| Instruction | 0.32 | 0.56 | 0.65 | 0.18 | 0.67 | 0.00 | 0.01 | 0.60 | 0.49 | 0.00 | 0.349 |
| Steering, all-token | **0.64**↑ | **0.42**↓ | **0.80**↑ | **0.56**↑ | **0.77**↑ | **0.45**↑ | **0.55**↑ | **0.24**↓ | **0.61**↑ | **0.04**↑ | 0.507 |
| Instr. + steering, all-token | **0.63**↑ | **0.42**↓ | 0.70 | **0.59**↑ | 0.73 | **0.56**↑ | **0.59**↑ | 0.56 | **0.64**↑ | **0.23**↑ | 0.567 |
| Steering, prompt-only | **0.62**↑ | **0.41**↓ | 0.69 | **0.60**↑ | 0.73 | **0.53**↑ | **0.52**↑ | 0.54 | **0.65**↑ | **0.49**↑ | 0.578 |
| Instr. + steering, prompt-only | **0.59**↑ | 0.55 | **0.76**↑ | **0.67**↑ | 0.71 | **0.58**↑ | **0.56**↑ | **0.69**↑ | **0.66**↑ | **0.41**↑ | 0.618 |

Bold: McNemar p < 0.05 against the instruction (↑ better, ↓ worse).

Mean over pairs of the other measures:

| | accuracy | in the wanted language | capped at 2048 |
|---|---|---|---|
| Source (in S) | 0.706 | 0.990 | 0.011 |
| Native | 0.727 | 0.997 | 0.013 |
| Instruction | 0.665 | 0.518 | 0.017 |
| Steering, all-token | 0.581 | 0.845 | 0.115 |
| Instr. + steering, all-token | 0.604 | 0.931 | 0.053 |
| Steering, prompt-only | 0.594 | 0.953 | 0.095 |
| Instr. + steering, prompt-only | 0.633 | 0.962 | 0.045 |

* **The instruction fails by staying in S.** Its off-target answers are almost all in S (e.g. about 196 of 200 in each sw-source pair). Where it works (zh→sw, fr→es, ru→bn, ru→es: in T 88–93%), steering adds little, and steering alone can hurt.
* **Steering recovers most of the gap where the instruction fails:** sw→ja, sw→de and sw→bn go from 0.00–0.01 to 0.41–0.58. Without the three sw-source pairs, the mean still rises from 0.50 to 0.66.
* **Steering costs accuracy** (0.665 → 0.58–0.63) and lengthens answers; prompt-only steering costs least.
* **One shared setting** suits most pairs but is too weak for sw→bn without the instruction (0.04 with all-token steering).
* **An artefact:** in sw→de, 72 of 200 prompt-only answers with the instruction start with the Swahili word "Umeme" ("electricity") before continuing in German.

### Checks on the judge

* The in-language check looks for English lines, not lines in S. Also rejecting a line of ≥ 5 words labelled S with p > 0.5 changes the test means by at most 0.017, and lowers the instruction most (0.349 → 0.332; prompt-only with instruction 0.618 → 0.615).
* Lines under 5 words are not checked on their own, so a few short S fragments in a T answer can pass (e.g. mgsm-0195 in sw→de). This can only favour the steered arms.
