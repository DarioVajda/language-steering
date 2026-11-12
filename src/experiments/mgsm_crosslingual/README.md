# Cross-lingual instruction vs language steering on MGSM

The question is in a source language S and the answer is wanted in a target language T. Can a language vector make Llama-3.1-8B-Instruct **answer in T, where an instruction fails?** The measure is *success*, as in `../mgsm_steering/`: an answer counts only if it is mathematically correct and written in the wanted language. All 11 MGSM languages (English included, as any other language) are both sources and targets: 110 ordered pairs. The steering settings are chosen on 11 tuning pairs; the other 99 are held out.

**Result: steering helps wherever the instruction fails.** With the question in S, the instruction yields an answer in T only half the time: the model mostly ignores the request and answers in S, with its maths intact. On test, the success rate averaged over the 99 held-out pairs is:
* **0.340 with the instruction alone** (accuracy 0.673);
* **0.634 when prompt-only steering is added on top**, significantly better than the instruction in 59 pairs and worse in 1;
* 0.723 for the native question (in T), the ceiling.

## Reproduce

```bash
slurm/submit.sh -J mgsm_xl_main -t 10:00:00 --gres gpu:2 --gres-flags=disable-binding -c 8 --mem 72G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/run.sh 2 2
# optional helpers on other nodes, joining the main job's queue (<id>: the job id printed above):
slurm/submit.sh -J mgsm_xl_help -t 10:00:00 --gres gpu:4 --gres-flags=disable-binding -c 16 --mem 144G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/run.sh 4 2 <id>
slurm/submit.sh -J mgsm_xl_check -t 00:30:00 -c 4 --mem 48G --gres-flags=disable-binding -- \
  bash -c 'HF_HUB_OFFLINE=1 python src/experiments/mgsm_crosslingual/check_instructions.py'
```

`run.sh <gpus> <workers per gpu> [main job id]` runs every stage in one job: vectors, the dev grid, selection, test and the tables. Workers share each stage through claim files under `results/generations/claims/<main job id>/`, as in `../mgsm_steering/`. A helper (with a main job id) waits for the main job's vectors and selection, takes units from the same queue, and leaves the tables to the main job.

`HF_HUB_CACHE=$PWD/logs/hf_cache` is needed because the shared cache's Llama-3.1-8B-Instruct snapshot lacks its config and tokenizer files. `logs/hf_cache` holds those files, and its weight entries link into the shared snapshot.

The reported run used 7 GPUs: the main job on 2 B300, and helpers on 1 B300 and 4 B200. It took 2 h 08 min, mostly the test stage. As in `../mgsm_steering/`, a unit's output depends on which GPU ran it only through bf16 differences. The third job checks the instruction strings (below).

| File | Contents |
|---|---|
| `generate_pairs.py` | pairs, tuning pairs, instruction strings, prompts, vectors, work units; model, split, grid and steering are imported from `../mgsm_steering/generate.py` |
| `analyse_pairs.py` | selection, McNemar tests, tables; scoring is imported from `../mgsm_steering/analyse.py` |
| `run.sh` | every stage in one Slurm job, optionally shared with helper jobs |
| `check_instructions.py` | NLLB-200 back-translation of the 100 non-English instruction strings |
| `results/summary.json` | every number below, and per pair every arm with its McNemar counts (the generations in `results/generations/` are not committed) |
| `results/instructions.json` | the instruction strings, back-translations and fastText labels |

## Setup

Everything is as in `../mgsm_steering/` (model, greedy decoding with at most 2048 new tokens, the 50 dev / 200 test split, the grid, success and the McNemar test), with English replaced by S.

* **Pairs:** all 110 ordered pairs (S, T), S ≠ T, of en, bn, de, es, fr, ja, ru, sw, te, th, zh.
* **Tuning pairs:** 11 pairs drawn before any generation (`draw_tuning`, seed 0), uniformly among the assignments in which every language is the source of exactly one pair and the target of exactly one, and no two languages are tuned in both directions:

  | S → T | en→sw | bn→ru | de→te | es→de | fr→en | ja→zh | ru→es | sw→ja | te→th | th→bn | zh→fr |
  |---|---|---|---|---|---|---|---|---|---|---|---|

* **Arms (for a pair S→T):**

  | Name | What the model is given |
  |---|---|
  | Source | the question in S, judged in S (reference) |
  | Native | the question in T |
  | Instruction | the S question + "\n\n" + "Answer in T." written in S (baseline) |
  | Steering | the S question, + α·v at layer l |
  | Instruction + steering | the instruction prompt, + α·v at layer l |

  Each steering arm comes in an all-token and a prompt-only version. Source depends only on S and native only on T, so each is generated once per language.
* **Vectors:** v_(S→T)(l) = μ_T(l) − μ_S(l) over the FLORES-200 dev sentences.
* **Instruction strings:** the name of T in S from CLDR 47, in a per-language imperative template (Russian takes the prepositional case). For S = en they are the strings of `../mgsm_steering/`. NLLB-200 3.3B translates all 100 non-English strings back into a correct English instruction.

  | S | Template | S | Template |
  |---|---|---|---|
  | en | Answer in {T}. | ru | Отвечай на {T}. |
  | bn | {T} ভাষায় উত্তর দিন। | sw | Jibu kwa {T}. |
  | de | Antworte auf {T}. | te | {T} భాషలో సమాధానం ఇవ్వండి. |
  | es | Responde en {T}. | th | ตอบเป็นภาษา{T} |
  | fr | Réponds en {T}. | zh | 请用{T}回答。 |
  | ja | {T}で答えてください。 | | |

* **Selection (dev, per steering arm):** the (layer, α) with the highest success averaged over the 11 tuning pairs; ties go to the smaller α, then the earlier layer. The chosen settings are used for all 110 pairs.

## Results

### Dev selection (mean success over the 11 tuning pairs; source 0.662, native 0.656, instruction 0.311)

| Arm | Chosen | Dev success |
|---|---|---|
| Steering, all-token | l 16, α 1 | 0.449 |
| Instruction + steering, all-token | l 16, α 0.5 | 0.589 |
| Steering, prompt-only | l 16, α 2 | 0.485 |
| Instruction + steering, prompt-only | l 16, α 1 | 0.602 |

### Test (200 items per pair): mean success

| | tuning (11) | held-out (99) | all (110) | held-out pairs better / worse |
|---|---|---|---|---|
| Source (in S) | 0.723 | 0.723 | 0.723 | – |
| Native | 0.723 | 0.723 | 0.723 | 71 / 2 |
| Instruction | 0.350 | 0.340 | 0.341 | – |
| Steering, all-token | 0.491 | 0.528 | 0.524 | 58 / 23 |
| Instr. + steering, all-token | 0.640 | 0.602 | 0.606 | 56 / 1 |
| Steering, prompt-only | 0.550 | 0.554 | 0.554 | 57 / 21 |
| Instr. + steering, prompt-only | 0.655 | 0.634 | 0.636 | 59 / 1 |

Better / worse: McNemar p < 0.05 against the instruction. Each language is S once and T once among the tuning pairs, so source and native have the same mean in every column.

Held-out means of the other measures:

| | accuracy | in the wanted language | capped at 2048 |
|---|---|---|---|
| Source (in S) | 0.727 | 0.996 | 0.014 |
| Native | 0.727 | 0.996 | 0.014 |
| Instruction | 0.673 | 0.492 | 0.030 |
| Steering, all-token | 0.570 | 0.892 | 0.149 |
| Instr. + steering, all-token | 0.646 | 0.928 | 0.037 |
| Steering, prompt-only | 0.599 | 0.888 | 0.112 |
| Instr. + steering, prompt-only | 0.655 | 0.963 | 0.041 |

### Held-out success by language (9 pairs each)

| S | source | native | instr. | steer | instr. + steer | steer, PO | instr. + steer, PO |
|---|---|---|---|---|---|---|---|
| en | 0.795 | 0.725 | 0.785 | 0.603 | 0.761 | 0.596 | 0.774 |
| bn | 0.645 | 0.727 | 0.052 | 0.446 | 0.530 | 0.490 | 0.592 |
| de | 0.775 | 0.726 | 0.443 | 0.616 | 0.688 | 0.642 | 0.688 |
| es | 0.830 | 0.706 | 0.507 | 0.605 | 0.683 | 0.640 | 0.701 |
| fr | 0.750 | 0.712 | 0.572 | 0.563 | 0.654 | 0.603 | 0.663 |
| ja | 0.620 | 0.731 | 0.372 | 0.536 | 0.593 | 0.471 | 0.601 |
| ru | 0.770 | 0.706 | 0.373 | 0.557 | 0.623 | 0.599 | 0.664 |
| sw | 0.635 | 0.744 | 0.001 | 0.451 | 0.393 | 0.561 | 0.531 |
| te | 0.650 | 0.731 | 0.062 | 0.379 | 0.471 | 0.402 | 0.474 |
| th | 0.730 | 0.731 | 0.039 | 0.520 | 0.616 | 0.574 | 0.643 |
| zh | 0.755 | 0.717 | 0.538 | 0.532 | 0.613 | 0.522 | 0.643 |

| T | source | native | instr. | steer | instr. + steer | steer, PO | instr. + steer, PO |
|---|---|---|---|---|---|---|---|
| en | 0.712 | 0.795 | 0.013 | 0.648 | 0.663 | 0.588 | 0.683 |
| bn | 0.731 | 0.645 | 0.438 | 0.305 | 0.493 | 0.454 | 0.559 |
| de | 0.706 | 0.775 | 0.302 | 0.452 | 0.635 | 0.418 | 0.643 |
| es | 0.706 | 0.830 | 0.431 | 0.656 | 0.691 | 0.674 | 0.687 |
| fr | 0.717 | 0.750 | 0.315 | 0.678 | 0.702 | 0.672 | 0.696 |
| ja | 0.744 | 0.620 | 0.227 | 0.582 | 0.646 | 0.627 | 0.658 |
| ru | 0.727 | 0.770 | 0.474 | 0.673 | 0.682 | 0.670 | 0.685 |
| sw | 0.725 | 0.635 | 0.436 | 0.414 | 0.510 | 0.426 | 0.528 |
| te | 0.726 | 0.650 | 0.429 | 0.224 | 0.488 | 0.307 | 0.548 |
| th | 0.731 | 0.730 | 0.314 | 0.582 | 0.457 | 0.601 | 0.606 |
| zh | 0.731 | 0.755 | 0.364 | 0.593 | 0.661 | 0.662 | 0.679 |

* **The held-out pairs agree with the tuning pairs.** The arms with the instruction lose 0.02–0.04 on held-out pairs, the arms without it none; the ranking is the same.
* **The instruction fails by staying in S.** Of its 11,044 test answers that fastText does not label T (whole response), 10,723 are labelled S. The source language decides whether it works: from en 0.785, from de, es, fr, ja, ru and zh 0.37–0.57, from bn, sw, te and th 0.00–0.06. "Answer in English" fails from every source (0.013).
* **Steering recovers most of the gap where the instruction fails.** In the 38 pairs where the instruction scores ≤ 0.05, prompt-only steering with the instruction reaches 0.59 (native 0.75). In the 32 pairs where it scores ≥ 0.60, it is level with the instruction (0.72 vs 0.70).
* **Steering without the instruction is risky:** it is worse than the instruction in 21–23 held-out pairs, mostly from English or into te, bn, sw and de.
* **Steering costs accuracy** (0.673 → 0.57–0.66) and lengthens answers; with the instruction, prompt-only steering costs least.
* **An artefact:** in sw→de, 72 of 200 prompt-only answers with the instruction start with the Swahili word "Umeme" ("electricity") before continuing in German. No other sw-source pair shows this.
* **The en-source pairs repeat `../mgsm_steering/`** with the same prompts and items: the instruction scores 0.776 over them here against 0.770 there, native 0.716 against 0.715 (different batching and GPUs).

### Checks on the judge

* The in-language check looks for English lines, not lines in S. In the earlier 10-pair version of this experiment, also rejecting a line of ≥ 5 words labelled S with p > 0.5 changed the test means by at most 0.017 and lowered the instruction most. It has not been applied to this run.
* Lines under 5 words are not checked on their own, so a few short S fragments in a T answer can pass (e.g. mgsm-0195 in sw→de). This can only favour the steered arms.
