# Cross-lingual instruction vs language steering on MGSM

The question is in a source language S and the answer is wanted in a target language T. Can a language vector make an instruction-tuned model **answer in T, where an instruction fails?** The measure is *success*, as in `../mgsm_steering/`: an answer counts only if it is mathematically correct and written in the wanted language. All 11 MGSM languages (English included, as any other language) are both sources and targets: 110 ordered pairs. The steering settings are chosen on 11 tuning pairs; the other 99 are held out. Three models: Llama-3.1-8B-Instruct (the main one), Qwen3-8B and Qwen3-32B (thinking off).

**Result: steering helps wherever the instruction fails, in all three models.** With the question in S, the instruction often yields an answer in S, with the maths intact. Mean test success over the 99 held-out pairs:

| | Llama-3.1-8B | Qwen3-8B | Qwen3-32B |
|---|---|---|---|
| Native question (in T), the ceiling | 0.723 | 0.763 | 0.835 |
| Instruction | 0.340 | 0.412 | 0.694 |
| **Instruction + prompt-only steering** | **0.634** (59 / 1) | **0.711** (69 / 0) | **0.800** (36 / 6) |
| Instruction in English + prompt-only steering | 0.653 (65 / 1) | 0.722 (71 / 1) | 0.817 (39 / 5) |
| Control: random direction, same norm | 0.322 (0 / 10) | 0.402 (4 / 12) | 0.693 (10 / 10) |
| Control: vector towards a wrong language | 0.162 (13 / 53) | 0.301 (23 / 49) | 0.698 (18 / 28) |
| Held-out pairs below 0.30: instruction / instr. + steering | 47 / 2 | 42 / 5 | 13 / 1 |

In brackets: held-out pairs significantly better / worse than the instruction (McNemar p < 0.05). Steering does not change which problems are solved: the error consistency with the native question moves by +0.010 [−0.007, 0.026], +0.004 [−0.010, 0.018] and +0.002 [−0.014, 0.018] (Cohen's κ, below).

## Reproduce

```bash
slurm/submit.sh -J mgsm_xl_main -t 10:00:00 --gres gpu:2 --gres-flags=disable-binding -c 8 --mem 72G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/run.sh llama-3.1-8b 2 2
# optional helpers on other nodes, joining the main job's queue (<id>: the job id printed above):
slurm/submit.sh -J mgsm_xl_help -t 10:00:00 --gres gpu:4 --gres-flags=disable-binding -c 16 --mem 144G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/run.sh llama-3.1-8b 4 2 <id>
# after the main job: teacher-forced KL to the native question (kl.sh <model> <gpus> [main job id], helpers as above)
slurm/submit.sh -J mgsm_xl_kl -t 03:00:00 --gres gpu:1 --gres-flags=disable-binding -c 8 --mem 96G \
  --constraint 'GPU_BRD:B300' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/kl.sh llama-3.1-8b 1
slurm/submit.sh -J mgsm_xl_check -t 00:30:00 -c 4 --mem 48G --gres-flags=disable-binding -- \
  bash -c 'HF_HUB_OFFLINE=1 python src/experiments/mgsm_crosslingual/check_instructions.py'
# robustness and agreement (CPU)
slurm/submit.sh -J mgsm_xl_tables -c 4 --mem 16G --gres=none --constraint= -- \
  bash -c 'export XL_MODEL=llama-3.1-8b; python src/experiments/mgsm_crosslingual/robustness.py && python src/experiments/mgsm_crosslingual/agreement.py'

# Qwen3-8B: the same with qwen3-8b and without HF_HUB_CACHE (shared cache)
# Qwen3-32B: one worker per GPU, on B200/B300
slurm/submit.sh -J mgsm_xl_q32 -t 24:00:00 --gres gpu:8 --gres-flags=disable-binding -c 32 --mem 320G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/run.sh qwen3-32b 8 1
```

`run.sh <model> <gpus> <workers per gpu> [main job id]` runs every stage in one job: vectors, the dev grid, selection, test and the tables. The model is a key of `MODELS` in `../mgsm_steering/generate.py` (`llama-3.1-8b`, `qwen3-8b`, `qwen3-32b`); every script reads it from the environment variable `XL_MODEL` (set by `run.sh` and `kl.sh`) and writes to `results/<model>/`. Workers share each stage through claim files under `results/<model>/generations/claims/<main job id>/`, as in `../mgsm_steering/`. A helper (with a main job id) waits for the main job's vectors and selection, takes units from the same queue, and leaves the tables to the main job.

For Llama, `HF_HUB_CACHE=$PWD/logs/hf_cache` is needed because the shared cache's Llama-3.1-8B-Instruct snapshot lacks its config and tokenizer files. `logs/hf_cache` holds those files, and its weight entries link into the shared snapshot.

The Qwen3-32B checkpoint (revision 9216db5) is also in `logs/hf_cache`, downloaded from the Hub; Qwen3-8B (revision b968826) is in the shared cache. Qwen3-32B runs one worker per GPU (B200/B300 only), the 8B models two.

Runs: Llama on 7 GPUs (2 B300 in the main job, helpers on 1 B300 and 4 B200), 2 h 08 min; Qwen3-8B on 9 B200, 3 h 01 min; Qwen3-32B on 20 B200, 1 h 50 min; mostly the test stage. As in `../mgsm_steering/`, a unit's output depends on which GPU ran it only through bf16 differences. KL: 29 min on 1 B300 (Llama), 37 min on 1 B200 (Qwen3-8B), 18 min on 8 B200 (Qwen3-32B, main job and a helper). `robustness.py`, `agreement.py` and `analyse_kl.py` run on a CPU in under a minute each.

| File | Contents |
|---|---|
| `generate_pairs.py` | pairs, tuning pairs, instruction strings, prompts, vectors, work units; model, split, grid and steering are imported from `../mgsm_steering/generate.py` |
| `analyse_pairs.py` | selection, McNemar tests, tables; scoring is imported from `../mgsm_steering/analyse.py` |
| `run.sh` | every stage in one Slurm job, optionally shared with helper jobs |
| `robustness.py` | held-out pairs with success below 0.10 and 0.30, from `results/<model>/summary.json` -> `results/<model>/robustness.json` |
| `agreement.py` | error consistency (Cohen's κ) with the native and source questions -> `results/<model>/agreement.json` |
| `kl.py`, `kl.sh` | teacher-forced per-token KL and top-1 agreement with the native question, in one Slurm job (-> `results/<model>/generations/kl/`, 0.3–0.5 GB per model, not committed) |
| `analyse_kl.py` | KL and top-1 by response position, and per S and T -> `results/<model>/kl.json` |
| `check_instructions.py` | NLLB-200 back-translation of the 100 non-English instruction strings |
| `results/<model>/summary.json` | every number below, and per pair every arm with its McNemar counts (the generations in `results/<model>/generations/` are not committed) |
| `results/instructions.json` | the instruction strings, back-translations and fastText labels |

## Setup

Everything is as in `../mgsm_steering/` (greedy decoding with at most 2048 new tokens, the 50 dev / 200 test split, the α grid, success and the McNemar test), with English replaced by S. The layer grid is per model: 8–24 (Llama, 32 layers), 8–28 (Qwen3-8B, 36 layers), 16–48 (Qwen3-32B, 64 layers), in steps of 4, 4 and 8.

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
  | Instruction in English | the same, with "Answer in T." in English |
  | Instruction as system message | the S question, with the instruction in S as the system message |
  | Steering | the S question, + α·v at layer l |
  | Instruction + steering | the instruction prompt, + α·v at layer l |
  | Instruction in English + steering | the English-instruction prompt, + α·v at layer l (prompt-only) |
  | Controls | instruction + prompt-only steering at its chosen (l, α), with v replaced, per item, by a random direction of the same norm, or by v_(S→U) for a wrong language U ≠ S, T drawn at random |

  Each steering arm except the English-instruction one comes in an all-token and a prompt-only version. Source depends only on S and native only on T, so each is generated once per language. Of the three instruction variants, only the one with the best dev success goes to test as a baseline (besides the instruction itself).
* **Vectors:** v_(S→T)(l) = μ_T(l) − μ_S(l), μ_L(l) the mean output of block l over the tokens of the 997 FLORES-200 dev sentences in L. The first token of each sentence is left out: it is the attention sink, with a norm 50–100 times that of the others (Llama's BOS; Qwen3 adds no BOS, so its first word). With it, Qwen's vectors are dominated by the sink and break the model at α = 1.
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

## Results: Llama-3.1-8B-Instruct

### Dev selection (mean success over the 11 tuning pairs; source 0.662, native 0.656; instruction 0.311, in English 0.451, as system message 0.167)

| Arm | Chosen | Dev success |
|---|---|---|
| Steering, all-token | l 16, α 1 | 0.449 |
| Instruction + steering, all-token | l 16, α 0.5 | 0.589 |
| Steering, prompt-only | l 16, α 2 | 0.485 |
| Instruction + steering, prompt-only | l 16, α 1 | 0.602 |
| Instruction in English + steering, prompt-only | l 16, α 1 | 0.607 |

### Test (200 items per pair)

| | tuning (11) | held-out (99) | all (110) | held-out pairs better / worse |
|---|---|---|---|---|
| Source (in S) | 0.723 | 0.723 | 0.723 | – |
| Native | 0.723 | 0.723 | 0.723 | 71 / 2 |
| Instruction | 0.350 | 0.340 | 0.341 | – |
| Instruction in English | 0.499 | 0.514 | 0.512 | 50 / 3 |
| Steering, all-token | 0.491 | 0.528 | 0.524 | 58 / 23 |
| Instr. + steering, all-token | 0.640 | 0.602 | 0.606 | 56 / 1 |
| Steering, prompt-only | 0.550 | 0.554 | 0.554 | 57 / 21 |
| Instr. + steering, prompt-only | 0.655 | 0.634 | 0.636 | 59 / 1 |
| Instr. in English + steering, prompt-only | 0.638 | 0.653 | 0.652 | 65 / 1 |
| Control: random direction | 0.319 | 0.322 | 0.321 | 0 / 10 |
| Control: wrong language | 0.141 | 0.162 | 0.160 | 13 / 53 |

Mean success. Better / worse: McNemar p < 0.05 against the instruction. Each language is S once and T once among the tuning pairs, so source and native have the same mean in every column.

Held-out means of the other measures, and the pairs where an arm fails (`robustness.py`; test success below 0.10 and below 0.30, out of 99):

| | accuracy | in the wanted language | capped at 2048 | below 0.10 | below 0.30 |
|---|---|---|---|---|---|
| Source (in S) | 0.727 | 0.996 | 0.014 | 0 | 0 |
| Native | 0.727 | 0.996 | 0.014 | 0 | 0 |
| Instruction | 0.673 | 0.492 | 0.030 | 40 | 47 |
| Instruction in English | 0.678 | 0.761 | 0.034 | 13 | 19 |
| Steering, all-token | 0.570 | 0.892 | 0.149 | 4 | 11 |
| Instr. + steering, all-token | 0.646 | 0.928 | 0.037 | 4 | 7 |
| Steering, prompt-only | 0.599 | 0.888 | 0.112 | 3 | 7 |
| Instr. + steering, prompt-only | 0.655 | 0.963 | 0.041 | 1 | 2 |
| Instr. in English + steering, prompt-only | 0.662 | 0.986 | 0.042 | 0 | 0 |
| Control: random direction | 0.676 | 0.466 | 0.032 | 40 | 50 |
| Control: wrong language | 0.646 | 0.239 | 0.049 | 42 | 83 |

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
* **The gain comes from the direction towards T.** At the same setting, a random direction of the same norm leaves the instruction where it is (0.322 against 0.340), and a vector towards a wrong language U pulls answers into U (0.162, in T 0.239). Instruction + prompt-only steering beats them in 69 and 97 of the 99 held-out pairs and loses to them in none.
* **Writing the instruction in English helps, and steering still adds to it:** 0.514 alone (dev winner of the three variants; as a system message it is worst, 0.167 on dev), 0.653 with prompt-only steering, the best arm, with no held-out pair below 0.30.
* **Steering without the instruction is risky:** it is worse than the instruction in 21–23 held-out pairs, mostly from English or into te, bn, sw and de.
* **Steering costs accuracy** (0.673 → 0.57–0.66) and lengthens answers; with the instruction, prompt-only steering costs least.
* **An artefact:** in sw→de, 72 of 200 prompt-only answers with the instruction start with the Swahili word "Umeme" ("electricity") before continuing in German. No other sw-source pair shows this.
* **The en-source pairs repeat `../mgsm_steering/`** with the same prompts and items: the instruction scores 0.776 over them here against 0.770 there, native 0.716 against 0.715 (different batching and GPUs).

### Closeness to the native question

Does steering make the model behave as when the question is asked in T? Two measures, each comparing an arm with the native question in the same pair; prompt-only steering at the dev settings, without the instruction (source → steer, PO) and with it (instr. → instr. + steer, PO).

**Which items are solved** (`agreement.py`): error consistency, Cohen's κ between two arms' per-item correctness (the number only, any language), per pair, mean over all 110 pairs; 95% CIs from 2000 bootstrap resamples of the 200 items.

| arm | accuracy | κ with native | κ with source |
|---|---|---|---|
| source | 0.727 | 0.273 [0.217, 0.333] | 1 |
| steer, PO | 0.596 | 0.265 [0.222, 0.309] | 0.378 [0.338, 0.416] |
| instr. | 0.673 | 0.296 [0.247, 0.348] | 0.482 [0.442, 0.523] |
| instr. + steer, PO | 0.656 | 0.306 [0.260, 0.355] | 0.418 [0.375, 0.461] |

Steering changes κ with native by −0.008 [−0.035, 0.019] without the instruction and +0.010 [−0.007, 0.026] with it, and κ with source by −0.622 [−0.662, −0.584] and −0.064 [−0.086, −0.043].

**Next-token distributions** (`kl.py`, `analyse_kl.py`): the native arm's saved response is fed after each arm's prompt (teacher forcing), and at every response token the arm's next-token distribution is compared with the native question's: KL(P_native ‖ P_arm) in nats, and whether the two most likely tokens agree (top-1). Token-weighted over all 110 pairs, by position in the response (22k, 198k, 880k and 5.3M arm tokens):

| arm | token 1 | tokens 2–10 | 11–50 | 51+ |
|---|---|---|---|---|
| source | 14.90 / 0.00 | 2.27 / 0.61 | 0.60 / 0.80 | 0.13 / 0.94 |
| steer, PO | 3.26 / 0.26 | 0.77 / 0.74 | 0.39 / 0.84 | 0.10 / 0.95 |
| instr. | 4.71 / 0.20 | 0.89 / 0.74 | 0.42 / 0.83 | 0.11 / 0.95 |
| instr. + steer, PO | 1.79 / 0.40 | 0.67 / 0.77 | 0.39 / 0.84 | 0.10 / 0.95 |

KL / top-1. The 95% bootstrap CIs over items (`results/<model>/kl.json`) are at most ±0.09 nats and ±0.01 top-1; the file also has the breakdown per S and per T (without CIs).

* **The same question in S and in T fails on different items** (κ 0.27), so "behaving like native" is a low, noisy target at the level of items.
* **Steering moves the model away from the source question's errors, not towards the native question's.** Which problems are solved does not measurably change towards native.
* **Steering does bring the next-token distributions closer to native, mostly at the start of the answer**, where the model commits to a language and a way of answering: token-1 KL falls from 14.9 to 3.3 without the instruction and from 4.7 to 1.8 with it. After ~50 tokens every arm is close to native (KL ≈ 0.1, top-1 0.95): the text so far, not the prompt, decides the next token.
* Together: steering acts on how and in which language the model starts to answer, not on which problems it solves.

## Results: Qwen3-8B

Dev (tuning pairs): source 0.738, native 0.740; instruction 0.338, in English 0.375 (tested), as system message 0.273.

| | chosen (dev) | tuning | held-out | held-out better / worse | accuracy | in T | capped | below 0.10 / 0.30 |
|---|---|---|---|---|---|---|---|---|
| Source (in S) | – | 0.763 | 0.763 | – | 0.797 | 0.932 | 0.016 | 0 / 9 |
| Native | – | 0.763 | 0.763 | 71 / 6 | 0.797 | 0.932 | 0.016 | 0 / 9 |
| Instruction | – | 0.385 | 0.412 | – | 0.766 | 0.516 | 0.037 | 30 / 42 |
| Instruction in English | – | 0.401 | 0.449 | 38 / 22 | 0.771 | 0.588 | 0.029 | 17 / 41 |
| Steering, all-token | l 24, α 1 | 0.608 | 0.613 | 58 / 16 | 0.690 | 0.862 | 0.115 | 9 / 12 |
| Instr. + steering, all-token | l 24, α 0.5 | 0.727 | 0.705 | 72 / 0 | 0.738 | 0.941 | 0.056 | 1 / 6 |
| Steering, prompt-only | l 24, α 1 | 0.572 | 0.556 | 55 / 25 | 0.751 | 0.742 | 0.039 | 14 / 19 |
| Instr. + steering, prompt-only | l 28, α 1 | 0.696 | 0.711 | 69 / 0 | 0.741 | 0.945 | 0.053 | 2 / 5 |
| Instr. in English + steering, prompt-only | l 28, α 1 | 0.719 | 0.722 | 71 / 1 | 0.747 | 0.962 | 0.051 | 0 / 0 |
| Control: random direction | l 28, α 1 | 0.370 | 0.402 | 4 / 12 | 0.764 | 0.504 | 0.041 | 30 / 43 |
| Control: wrong language | l 28, α 1 | 0.266 | 0.301 | 23 / 49 | 0.746 | 0.378 | 0.048 | 24 / 51 |

* **The same picture as Llama.** Instruction + prompt-only steering is never significantly worse than the instruction, and beats both controls (in 75 and 92 held-out pairs; never worse).
* **The two prompt-only arms with the instruction chose layer 28, the last of the grid;** layer 24 is within 0.002 on dev (0.676 against 0.678).
* **Qwen3-8B answers Swahili questions in English**: source and native success are 0.15 for sw (accuracy is normal), which is why the native ceiling has 9 held-out pairs below 0.30. For T = sw, the English instruction with steering (0.44) is far above the native question (0.15).
* **Thinking off is not watertight:** a few answers contain a thinking tag (`<think>` or `</think>`, as plain text) and reason inside the answer. On test this is at most 0.6% of answers per arm, highest in the instruction (0.5%) and the two controls (0.6%), and 0.00–0.27% in the steering arms; Qwen3-32B at most 0.08% per arm, Llama never. It does not favour steering, so the answers are scored as they are.
* Closeness to native: κ with native changes by +0.035 [0.012, 0.057] without the instruction and +0.004 [−0.010, 0.018] with it; κ with source by −0.514 and −0.051 [−0.067, −0.034]. KL / top-1 by response position:

  | arm | token 1 | tokens 2–10 | 11–50 | 51+ |
  |---|---|---|---|---|
  | source | 20.97 / 0.01 | 0.98 / 0.77 | 0.63 / 0.83 | 0.14 / 0.95 |
  | steer, PO | 1.44 / 0.63 | 0.67 / 0.80 | 0.60 / 0.84 | 0.14 / 0.95 |
  | instr. | 7.69 / 0.18 | 0.86 / 0.78 | 0.72 / 0.82 | 0.17 / 0.94 |
  | instr. + steer, PO | 3.07 / 0.33 | 0.81 / 0.78 | 0.70 / 0.82 | 0.16 / 0.94 |

## Results: Qwen3-32B

Dev (tuning pairs): source 0.811, native 0.813; instruction 0.611 (tested), in English 0.582, as system message 0.451.

| | chosen (dev) | tuning | held-out | held-out better / worse | accuracy | in T | capped | below 0.10 / 0.30 |
|---|---|---|---|---|---|---|---|---|
| Source (in S) | – | 0.835 | 0.835 | – | 0.850 | 0.980 | 0.003 | 0 / 0 |
| Native | – | 0.835 | 0.835 | 42 / 12 | 0.850 | 0.980 | 0.003 | 0 / 0 |
| Instruction | – | 0.616 | 0.694 | – | 0.854 | 0.802 | 0.002 | 6 / 13 |
| Steering, all-token | l 16, α 1 | 0.606 | 0.616 | 21 / 41 | 0.783 | 0.782 | 0.017 | 13 / 18 |
| Instr. + steering, all-token | l 40, α 0.5 | 0.815 | 0.802 | 37 / 2 | 0.846 | 0.942 | 0.003 | 2 / 2 |
| Steering, prompt-only | l 48, α 2 | 0.675 | 0.660 | 31 / 33 | 0.823 | 0.797 | 0.012 | 10 / 15 |
| Instr. + steering, prompt-only | l 40, α 1 | 0.810 | 0.800 | 36 / 6 | 0.836 | 0.953 | 0.004 | 1 / 1 |
| Instr. in English + steering, prompt-only | l 48, α 1 | 0.831 | 0.817 | 39 / 5 | 0.849 | 0.961 | 0.003 | 0 / 0 |
| Control: random direction | l 40, α 1 | 0.620 | 0.693 | 10 / 10 | 0.840 | 0.816 | 0.003 | 4 / 10 |
| Control: wrong language | l 40, α 1 | 0.682 | 0.698 | 18 / 28 | 0.836 | 0.822 | 0.005 | 2 / 6 |

* **The larger model follows the instruction far more often** (0.694, 13 pairs below 0.30), so there is less to gain; steering with the instruction still adds 0.106 and removes almost all failing pairs. Without the instruction, steering is worse than the instruction alone.
* **Where the instruction already works, steering costs accuracy.** The 6 held-out pairs where instruction + prompt-only steering is significantly worse are en and zh into bn, te and sw, where the instruction scores 0.65–0.98 (above native in en→bn and en→te). There steering stays in T and loses accuracy (e.g. zh→bn 0.92 → 0.79).
* **Both controls stay at the instruction's level** (0.693 and 0.698): at this setting a direction that is not towards T neither helps nor, unlike in the 8B models, drags the answer elsewhere. Instruction + steering beats them in 41 and 51 held-out pairs (worse in 4 and 5).
* **Settings at the edge of the layer grid:** steering, all-token chose layer 16 (the first); steering, prompt-only and the English-instruction arm chose layer 48 (the last). At layer 48 some answers start in English ("Let's calculate …") and switch to T, which costs the English-instruction arm in-T rate in en→de (0.99 → 0.80) and en→es, zh→ja.
* Closeness to native: κ with native changes by +0.057 [0.022, 0.091] without the instruction and +0.002 [−0.014, 0.018] with it; κ with source by −0.476 and −0.039 [−0.060, −0.019]. KL / top-1 by response position:

  | arm | token 1 | tokens 2–10 | 11–50 | 51+ |
  |---|---|---|---|---|
  | source | 18.51 / 0.01 | 1.03 / 0.75 | 0.39 / 0.85 | 0.10 / 0.95 |
  | steer, PO | 2.49 / 0.54 | 0.58 / 0.79 | 0.37 / 0.85 | 0.10 / 0.95 |
  | instr. | 3.53 / 0.32 | 0.64 / 0.78 | 0.44 / 0.84 | 0.12 / 0.94 |
  | instr. + steer, PO | 2.67 / 0.33 | 0.61 / 0.78 | 0.45 / 0.83 | 0.12 / 0.94 |

  With the instruction, steering moves the first token less here than in the 8B models, as the instruction alone already starts most answers in T.

## Checks on the judge

* The in-language check looks for English lines, not lines in S. In the earlier 10-pair version of this experiment, also rejecting a line of ≥ 5 words labelled S with p > 0.5 changed the test means by at most 0.017 and lowered the instruction most. It has not been applied to this run.
* Lines under 5 words are not checked on their own, so a few short S fragments in a T answer can pass (e.g. mgsm-0195 in sw→de). This can only favour the steered arms.
