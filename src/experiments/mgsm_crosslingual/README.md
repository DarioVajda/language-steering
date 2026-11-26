# Cross-lingual instruction vs language steering on MGSM

The question is in a source language S and the answer is wanted in a target language T. Can a language vector make an instruction-tuned model **answer in T, where an instruction fails?** The measure is *success*, as in `../mgsm_steering/`: an answer counts only if it is mathematically correct and written in the wanted language. All 11 MGSM languages (English included, as any other language) are both sources and targets: 110 ordered pairs. The steering settings are chosen on 11 tuning pairs; the other 99 are held out. Four models: Llama-3.1-8B-Instruct (the main one), Qwen3-8B and Qwen3-32B (thinking off), and Gemma-3-12B-it.

**Result: steering helps wherever the instruction fails, in all four models.** With the question in S, the instruction often yields an answer in S, with the maths intact. Mean test success over the 99 held-out pairs:

| | Llama-3.1-8B | Qwen3-8B | Qwen3-32B | Gemma-3-12B |
|---|---|---|---|---|
| Native question (in T), the ceiling | 0.723 | 0.763 | 0.835 | 0.828 |
| Instruction | 0.340 | 0.412 | 0.694 | 0.729 |
| **Instruction + prompt-only steering** | **0.634** (59 / 1 [57 / 0]) | **0.711** (69 / 0 [62 / 0]) | **0.800** (36 / 6 [31 / 4]) | **0.777** (15 / 5 [11 / 1]) |
| Instruction in English + prompt-only steering | 0.653 (65 / 1 [61 / 0]) | 0.722 (71 / 1 [67 / 0]) | 0.817 (39 / 5 [31 / 4]) | 0.773 (17 / 5 [11 / 4]) |
| Baseline: ReCoVeR (all blocks, all tokens) | 0.567 (58 / 10 [52 / 2]) | 0.661 (67 / 0 [55 / 0]) | 0.796 (34 / 1 [28 / 0]) | 0.772 (14 / 3 [11 / 0]) |
| Control: random direction, same norm | 0.322 (0 / 10 [0 / 2]) | 0.402 (4 / 12 [0 / 2]) | 0.693 (10 / 10 [6 / 3]) | 0.517 (0 / 55 [0 / 43]) |
| Control: vector towards a wrong language | 0.162 (13 / 53 [6 / 48]) | 0.301 (23 / 49 [16 / 46]) | 0.698 (18 / 28 [14 / 16]) | 0.654 (6 / 50 [4 / 37]) |
| Held-out pairs below 0.30: instruction / instr. + steering | 47 / 2 | 42 / 5 | 13 / 1 | 4 / 0 |
| Instr. + prompt-only steering − instruction [95% CI] | +.294 [.275, .311] | +.299 [.284, .314] | +.107 [.099, .115] | +.048 [.042, .053] |
| Instr. + prompt-only steering − native [95% CI] | −.089 [−.105, −.074] | −.052 [−.066, −.038] | −.035 [−.046, −.024] | −.051 [−.064, −.039] |

In brackets: held-out pairs significantly better / worse than the instruction (McNemar p < 0.05), and in square brackets the same after a Holm correction over the 99 held-out pairs (each arm and model separately). The 95% CIs are from 2000 bootstrap resamples of the 200 test items (the same resample for every pair and arm). As all arms share the items, the CIs of the means themselves are wider (±.02–.03 for the instruction and ±.03 for steering and native; `results/<model>/native_tests.json`) than those of the paired differences. Steering closes 77%, 85%, 75% and 49% of the gap between the instruction and native, but stays significantly below native in every model (*Tests against the native question*, below). Steering does not change which problems are solved: the error consistency with the native question moves by +0.010 [−0.007, 0.026], +0.004 [−0.010, 0.018], +0.002 [−0.014, 0.018] and +0.001 [−0.011, 0.012] (Cohen's κ, below). The vector needs little text: from 10 parallel FLORES sentences instead of 997 the means are within .002, and from a single sentence within .013 (*Number of FLORES sentences*, below). ReCoVeR, the published method with the same direction at every block and every token, fixes the language as well, but is .067 and .051 below prompt-only steering in Llama and Qwen3-8B (in Llama by losing accuracy, below the instruction's), and ties in Qwen3-32B and Gemma-3-12B (*Baseline: ReCoVeR*, below).

**Outside maths (MMLU-ProX, [../mmlu_prox/](../mmlu_prox/)), with the same layer and α, not retuned:** the instruction fails there as well, and prompt-only steering closes 69–81% of the gap to native in Llama, Qwen3-8B and Qwen3-32B, ahead of ReCoVeR (+.009 to +.070, all CIs above 0). In Gemma the instruction mostly works, and ReCoVeR is slightly ahead (−.006 and −.019).

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
# Llama only: forced start (prefill.sh <model> <gpus> [main job id], helpers as above), then the logit lens
slurm/submit.sh -J xl-llama-prefill -t 04:00:00 --gres gpu:2 --gres-flags=disable-binding -c 8 --mem 64G \
  --constraint 'GPU_BRD:B300|GPU_BRD:B200' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/prefill.sh llama-3.1-8b 2
slurm/submit.sh -J xl-llama-lens -t 01:00:00 --gres gpu:1 --gres-flags=disable-binding -c 8 --mem 64G -- \
  bash -c "export XL_MODEL=llama-3.1-8b HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/logs/hf_cache G=src/experiments/mgsm_crosslingual; \
  for f in logit_lens logit_lens_top5; do python \$G/\$f.py run && python \$G/\$f.py analyse; done"
# Qwen3-8B: the same with XL_MODEL=qwen3-8b and without HF_HUB_CACHE; Qwen3-32B: with XL_MODEL=qwen3-32b, --mem 128G and on B200/B300
# robustness and agreement (CPU)
slurm/submit.sh -J mgsm_xl_tables -c 4 --mem 16G --gres=none --constraint= -- \
  bash -c 'export XL_MODEL=llama-3.1-8b; python src/experiments/mgsm_crosslingual/robustness.py && python src/experiments/mgsm_crosslingual/agreement.py'
# tests against native and the CIs of the headline means, all models (CPU, 3 min)
slurm/submit.sh -J native-tests -c 4 --mem 16G --gres=none --constraint= -- bash -c 'for m in llama-3.1-8b qwen3-8b qwen3-32b gemma3-12b; do XL_MODEL=$m python src/experiments/mgsm_crosslingual/native_tests.py; done'
# layer and alpha sensitivity, all models and LCB, from the stored dev grids (CPU, seconds)
python src/experiments/mgsm_crosslingual/sensitivity.py

# Qwen3-8B: the same with qwen3-8b and without HF_HUB_CACHE (shared cache)
# Qwen3-32B: one worker per GPU, on B200/B300
slurm/submit.sh -J mgsm_xl_q32 -t 24:00:00 --gres gpu:8 --gres-flags=disable-binding -c 32 --mem 320G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- env HF_HUB_CACHE=$PWD/logs/hf_cache bash src/experiments/mgsm_crosslingual/run.sh qwen3-32b 8 1

# Gemma-3-12B-it (shared cache): the main run, the tables, KL and the logit lens in one job; helpers as above
slurm/submit.sh -J xl-gemma -t 24:00:00 --gres gpu:2 --gres-flags=disable-binding -c 16 --mem 160G --constraint GPU_BRD:B200 -- \
  bash -c 'export XL_MODEL=gemma3-12b HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; G=src/experiments/mgsm_crosslingual
  bash $G/run.sh gemma3-12b 2 1 && python $G/robustness.py && python $G/agreement.py && bash $G/kl.sh gemma3-12b 2 && python $G/analyse_kl.py
  for f in logit_lens logit_lens_top5; do python $G/$f.py run && python $G/$f.py analyse; done'

# prompt variants (Qwen3-8B, after its main run): 2 GPUs, 2 workers each, then the comparison
slurm/submit.sh -J xl-q8-prompts -t 10:00:00 --gres gpu:A100_80GB:2 --gres-flags=disable-binding -c 16 --mem 96G \
  --constraint GPU_BRD:A100 -- bash -c 'export XL_MODEL=qwen3-8b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
  G=src/experiments/mgsm_crosslingual; for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((i / 2)) python $G/generate_pairs.py prompts & done
  wait; python $G/prompt_variants.py'
# optional helpers: the same without prompt_variants.py, and with SLURM_JOB_ID=<id> added to the export

# number of FLORES sentences (after the main runs): one GPU per model unless <gpus> says more, then each model's tables
slurm/submit.sh -J flores-subsets -t 24:00:00 --gres gpu:4 --gres-flags=disable-binding -c 32 --mem 256G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- bash src/experiments/mgsm_crosslingual/flores_subsets.sh \
  llama-3.1-8b:2:$PWD/logs/hf_cache qwen3-8b:2 qwen3-32b:1:$PWD/logs/hf_cache gemma3-12b:1
# optional helpers: they take units from the main job's queue (<id>) and leave the tables to it
slurm/submit.sh -J flores-help -t 08:00:00 --gres gpu:1 --gres-flags=disable-binding -c 8 --mem 96G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- bash src/experiments/mgsm_crosslingual/flores_subsets.sh \
  --no-analyse --defer-to <id> qwen3-32b:1:$PWD/logs/hf_cache

# ReCoVeR baseline (after the main runs), all four models in one job, one worker per GPU; LCB's ReCoVeR runs alongside
slurm/submit.sh -J recover -t 01:30:00 --gres gpu:3 --gres-flags=disable-binding -c 12 --mem 160G --constraint GPU_BRD:B300 -- \
  bash -c 'bash src/experiments/mgsm_crosslingual/recover.sh --main gemma3-12b llama-3.1-8b:$PWD/logs/hf_cache qwen3-8b qwen3-32b:$PWD/logs/hf_cache &
  bash src/experiments/language_confusion/recover.sh llama-3.1-8b:$PWD/logs/hf_cache qwen3-8b qwen3-32b:$PWD/logs/hf_cache gemma3-12b; wait'
# optional helpers: recover.sh without --main; checks against the authors' code (a clone of hSterz/recover @ 4c6eb4d):
#   XL_MODEL=<model> python src/experiments/mgsm_crosslingual/recover_check.py --repo <clone> --stages hooks,vectors (or lcb)
# ablation (after the ReCoVeR runs): the every-layer arms, MGSM and LCB, in one job over the free GPUs of 1-3 nodes;
# then the analysis (no GPU): XL_MODEL=<model> python src/experiments/mgsm_crosslingual/ablation.py and
# LCB_MODEL=<model> python src/experiments/language_confusion/ablation.py
slurm/submit.sh -J every_layer -t 05:00:00 --nodes 1-3 --ntasks-per-node 1 --gpus 8 --cpus-per-gpu 4 --mem-per-gpu 40G \
  --constraint GPU_BRD:B200 -- bash src/experiments/mgsm_crosslingual/every_layer.sh llama-3.1-8b:$PWD/logs/hf_cache qwen3-8b
```

`run.sh <model> <gpus> <workers per gpu> [main job id]` runs every stage in one job: vectors, the dev grid, selection, test and the tables. The model is a key of `MODELS` in `../mgsm_steering/generate.py` (`llama-3.1-8b`, `qwen3-8b`, `qwen3-32b`, `gemma3-12b`); every script reads it from the environment variable `XL_MODEL` (set by `run.sh` and `kl.sh`) and writes to `results/<model>/`. Workers share each stage through claim files under `results/<model>/generations/claims/<main job id>/`, as in `../mgsm_steering/`. A helper (with a main job id) waits for the main job's vectors and selection, takes units from the same queue, and leaves the tables to the main job.

For Llama, `HF_HUB_CACHE=$PWD/logs/hf_cache` is needed because the shared cache's Llama-3.1-8B-Instruct snapshot lacks its config and tokenizer files. `logs/hf_cache` holds those files, and its weight entries link into the shared snapshot.

The Qwen3-32B checkpoint (revision 9216db5) is also in `logs/hf_cache`, downloaded from the Hub; Qwen3-8B (revision b968826) is in the shared cache. Gemma-3-12B-it (revision 96b6f1e) is in the shared cache. Qwen3-32B and Gemma-3-12B run one worker per GPU (B200/B300 only), the 8B models two.

Runs: Llama on 7 GPUs (2 B300 in the main job, helpers on 1 B300 and 4 B200), 2 h 08 min; Qwen3-8B on 9 B200, 3 h 01 min; Qwen3-32B on 20 B200, 1 h 50 min; Gemma-3-12B on 4 B200 (2 in the main job, 2 in a helper for its last 3 h), 13 h 01 min, and the prompt-only steering test units again after the judge change (*Checks on the judge*), 2 h 32 min on 2 B200; mostly the test stage. As in `../mgsm_steering/`, a unit's output depends on which GPU ran it only through bf16 differences. KL: 29 min on 1 B300 (Llama), 37 min on 1 B200 (Qwen3-8B), 18 min on 8 B200 (Qwen3-32B, main job and a helper), 14 min on 2 B200 (Gemma-3-12B). `robustness.py`, `agreement.py` and `analyse_kl.py` run on a CPU in under a minute each. Prompt variants (Qwen3-8B): 4 A100 80 GB (main job and a helper, 2 each), 1 h 53 min. Forced start (Llama): 5 GPUs (2 + 2 B300 and a GH200, main job and helpers), 1 h 52 min. ReCoVeR: 1 h 02 min on 5 GPUs (3 B300 in the main job, also running LCB's ReCoVeR, and 2 B200 in a helper for Gemma-3-12B), most of it Gemma's dev grid at α = 128 and 256, where every answer runs to the token limit. Logit lens (both scripts): 13 min on 1 A100 80 GB (Llama), 5 min on 1 B200 (Qwen3-8B), 11 min on 1 B200 (Qwen3-32B), 2.5 min on 2 B200 (Gemma-3-12B, one script per GPU).

| File | Contents |
|---|---|
| `generate_pairs.py` | pairs, tuning pairs, instruction strings, prompts, vectors, work units; model, split, grid and steering are imported from `../mgsm_steering/generate.py` |
| `analyse_pairs.py` | selection, McNemar tests, tables; scoring is imported from `../mgsm_steering/analyse.py` |
| `run.sh` | every stage in one Slurm job, optionally shared with helper jobs |
| `robustness.py` | held-out pairs with success below 0.10 and 0.30, from `results/<model>/summary.json` -> `results/<model>/robustness.json` |
| `sensitivity.py` | success over the layer × α dev grid for every model and arm, and the LCB dev grids -> `results/sensitivity.json` |
| `flores_subsets.py`, `flores_subsets.sh` | instruction + prompt-only steering on test with vectors from 1, 3, 10 or 100 FLORES sentences, three draws each, in one Slurm job, optionally with helpers -> `results/<model>/flores_subsets.json` |
| `native_tests.py` | per-pair McNemar tests against the native question (Holm over the 99 held-out pairs), by T, and bootstrap CIs of the headline means and differences -> `results/<model>/native_tests.json` |
| `agreement.py` | error consistency (Cohen's κ) with the native and source questions -> `results/<model>/agreement.json` |
| `kl.py`, `kl.sh` | teacher-forced per-token KL and top-1 agreement with the native question, in one Slurm job (-> `results/<model>/generations/kl/`, 0.3–0.5 GB per model, not committed) |
| `analyse_kl.py` | KL and top-1 by response position, and per S and T -> `results/<model>/kl.json` |
| `prompt_variants.py` | the instruction in T and the instruction before the question against the instruction, on test -> `results/<model>/prompt_variants.json` |
| `prefill.py`, `prefill.sh` | forced-start experiment, in one Slurm job, optionally with helpers -> `results/<model>/prefill.json` |
| `logit_lens.py` | logit lens at the last prompt position, one token per language -> `results/<model>/logit_lens.json` |
| `logit_lens_top5.py` | the same with each language scored by its native top-5 first tokens -> `results/<model>/logit_lens_top5.json` |
| `recover.py`, `recover.sh` | the ReCoVeR baseline: dev grid over α, selection, test and analysis, all models in one Slurm job, optionally with helpers -> `results/<model>/recover.json` |
| `recover_check.py` | checks of the ReCoVeR reimplementation against the authors' code -> `results/<model>/recover_check_<stages>.json` |
| `every_layer.sh`, `every_layer_check.py` | the every-layer arms of the ablation (`recover.py` with `XL_ARM` / `LCB_ARM`), MGSM and LCB, several models in one Slurm job, after a check of the tuple-of-layers path |
| `ablation.py` | our method against all-token steering, every-layer steering and ReCoVeR, on test -> `results/<model>/ablation.json` |
| `check_instructions.py` | NLLB-200 back-translation of the 100 non-English instruction strings |
| `results/<model>/summary.json` | every number below, and per pair every arm with its McNemar counts (the generations in `results/<model>/generations/` are not committed) |
| `results/instructions.json` | the instruction strings, back-translations and fastText labels |

## Setup

Everything is as in `../mgsm_steering/` (greedy decoding with at most 2048 new tokens, the 50 dev / 200 test split, the α grid, success and the McNemar test), with English replaced by S. Counts of significantly better / worse pairs are given both uncorrected and Holm-corrected over the 99 held-out pairs of each comparison (`p_holm` in `summary.json`). **Statistics, throughout:** a difference between two arms in one pair is an exact McNemar test on the per-item success of the same 200 test items, Holm-corrected over the pairs of the comparison; a mean (or a mean difference) gets a 95% percentile CI from 2000 bootstrap resamples of the items (seed 0). Decoding is greedy and nothing is trained, so the items are the only source of sampling noise; the one exception is the number of FLORES sentences, where the draw of sentences is a second one and three draws are reported. The layer grid is per model: 8–24 (Llama, 32 layers), 8–28 (Qwen3-8B, 36 layers), 16–48 (Qwen3-32B, 64 layers), in steps of 4, 4 and 8, and 12–36 (Gemma-3-12B, 48 layers) in steps of 6. Gemma 3 interleaves five sliding-window (1024-token) attention blocks with one full-attention block; its grid is the full-attention blocks in about 25–75% of its depth.

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
  | Instruction as system message | the S question, with the instruction in S as the system message (Gemma's chat template has no system role and puts it at the start of the user turn, before the question) |
  | Instruction in T (Qwen3-8B only) | the S question + "\n\n" + "Answer in T." written in T, with T's own name for itself (e.g. "Antworte auf Deutsch.") |
  | Instruction first (Qwen3-8B only) | the instruction + "\n\n" + the S question |
  | Steering | the S question, + α·v at layer l |
  | Instruction + steering | the instruction prompt, + α·v at layer l |
  | Instruction in English + steering | the English-instruction prompt, + α·v at layer l (prompt-only) |
  | Controls | instruction + prompt-only steering at its chosen (l, α), with v replaced, per item, by a random direction of the same norm, or by v_(S→U) for a wrong language U ≠ S, T drawn at random |

  Each steering arm except the English-instruction one comes in an all-token and a prompt-only version. The two Qwen3-8B-only prompt variants have no settings and are not selected on dev: both go to test on all 110 pairs, and each is compared directly with the instruction. Source depends only on S and native only on T, so each is generated once per language. Of the three instruction variants, only the one with the best dev success goes to test as a baseline (besides the instruction itself).
* **Vectors:** v_(S→T)(l) = μ_T(l) − μ_S(l), μ_L(l) the mean output of block l over the tokens of the 997 FLORES-200 dev sentences in L. The first token of each sentence is left out: it is the attention sink, with a norm 50–100 times that of the others (Llama's and Gemma's BOS; Qwen3 adds no BOS, so its first word). With it, Qwen's vectors are dominated by the sink and break the model at α = 1.
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
| Native | 0.723 | 0.723 | 0.723 | 71 / 2 [65 / 1] |
| Instruction | 0.350 | 0.340 | 0.341 | – |
| Instruction in English | 0.499 | 0.514 | 0.512 | 50 / 3 [44 / 0] |
| Steering, all-token | 0.491 | 0.528 | 0.524 | 58 / 23 [50 / 19] |
| Instr. + steering, all-token | 0.640 | 0.602 | 0.606 | 56 / 1 [52 / 1] |
| Steering, prompt-only | 0.550 | 0.554 | 0.554 | 57 / 21 [53 / 13] |
| Instr. + steering, prompt-only | 0.655 | 0.634 | 0.636 | 59 / 1 [57 / 0] |
| Instr. in English + steering, prompt-only | 0.638 | 0.653 | 0.652 | 65 / 1 [61 / 0] |
| Control: random direction | 0.319 | 0.322 | 0.321 | 0 / 10 [0 / 2] |
| Control: wrong language | 0.141 | 0.162 | 0.160 | 13 / 53 [6 / 48] |

Mean success. Better / worse: McNemar p < 0.05 against the instruction; in square brackets, Holm-corrected over the 99 held-out pairs. Each language is S once and T once among the tuning pairs, so source and native have the same mean in every column.

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
* **The gain comes from the direction towards T.** At the same setting, a random direction of the same norm leaves the instruction where it is (0.322 against 0.340), and a vector towards a wrong language U pulls answers into U (0.162, in T 0.239). Instruction + prompt-only steering beats them in 69 and 97 of the 99 held-out pairs (61 and 97 after Holm) and loses to them in none.
* **Writing the instruction in English helps, and steering still adds to it:** 0.514 alone (dev winner of the three variants; as a system message it is worst, 0.167 on dev), 0.653 with prompt-only steering, the best arm, with no held-out pair below 0.30.
* **Steering without the instruction is risky:** it is worse than the instruction in 21–23 held-out pairs (13–19 after Holm), mostly from English or into te, bn, sw and de.
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

### The start of the answer and where its language is decided

Two follow-ups, the forced start on Llama only and the logit lens on all four models, on 50 of the 200 test items (drawn once, seed 0) and the 99 held-out pairs, for the instruction and instruction + prompt-only steering at its dev setting (Llama l 16, α 1; for the Qwen models see below). *Failing pairs*: the 40 held-out pairs where the instruction's test success is below 0.10 (from `summary.json`, i.e. fixed before these runs); *working pairs*: the other 59.

**Forced start** (`prefill.py`): the first k tokens of the answer are forced, the rest is generated as usual. The forced start is in T (the native arm's answer to the same item) or in S (the source arm's answer); the language is judged on the generated continuation only, correctness on the whole answer. Success (in T, accuracy), in %, mean over pairs:

| k | instr., T start | instr., S start | instr. + steer, T start | instr. + steer, S start |
|---|---|---|---|---|
| **Failing (40)** | | | | |
| 0 | 2 (3, 70) | 2 (3, 70) | 65 (94, 69) | 65 (94, 69) |
| 1 | 43 (66, 68) | 0 (0, 72) | 67 (99, 68) | 47 (68, 69) |
| 4 | 50 (74, 68) | 0 (0, 72) | 67 (99, 68) | 45 (66, 69) |
| 16 | 54 (80, 68) | 0 (0, 71) | 68 (100, 68) | 42 (63, 69) |
| **Held-out (99)** | | | | |
| 0 | 35 (49, 72) | 35 (49, 72) | 68 (97, 70) | 68 (97, 70) |
| 1 | 59 (84, 71) | 14 (19, 73) | 69 (99, 70) | 61 (87, 70) |
| 4 | 62 (88, 71) | 10 (13, 74) | 69 (99, 69) | 59 (84, 70) |
| 16 | 64 (91, 71) | 7 (9, 74) | 71 (99, 71) | 55 (79, 70) |

k = 0 is one run (nothing forced), shown under both starts. 95% bootstrap CIs over items are in `results/<model>/prefill.json` (about ±0.06 on success).

* **With the instruction, the first token largely decides the language.** In the failing pairs, one forced token in T lifts the answers in T from 3% to 66%, and 16 tokens to 80%; one token in S drops them to 0%.
* **The start is not the whole story.** Even 16 tokens in T leave a fifth of the instruction's answers drifting out of T, while with steering a T start keeps them in T (99–100%). Steering also holds about two thirds of the answers in T after an S start (63–68% in the failing pairs), so it acts beyond the first token.
* **Accuracy barely moves** (68–74% in every cell): the start changes the language, not whether the problem is solved.

**Logit lens** (`logit_lens.py`, `logit_lens_top5.py`): the output of each block l at the last prompt position, passed through the model's final norm and unembedding, read as the next-token distribution P_l that block would predict (at l = 32 it is the model's own; the two agree exactly). Each language is scored by a set of tokens: the 5 most likely first tokens of the model's answer to the same item asked in that language, without an instruction (the native prompt, for T, S and English). These 5 cover 73% (th) to 100% (bn, te) of that distribution. A token in more than one of the sets compared is left out of all of them (0.4 tokens per prompt on average). Typical probability, ×10⁶ (the geometric mean over pairs and items, exp of the mean log P_l(set)); English only over the pairs without English:

| layer | instr.: T | instr.: S | instr.: en | instr. + steer: T | instr. + steer: S | instr. + steer: en |
|---|---|---|---|---|---|---|
| **Failing (40)** | | | | | | |
| 1 | 13 | 12 | 16 | 13 | 12 | 16 |
| 8 | 11 | 5 | 9 | 11 | 5 | 9 |
| 15 | 10 | 4 | 4 | 10 | 4 | 4 |
| 16 | 22 | 6 | 13 | 21 | 6 | 13 |
| 20 | 80 | 13 | 93 | 75 | 11 | 74 |
| 24 | 135 | 20 | 1,387 | 213 | 21 | 797 |
| 28 | 152 | 76 | 2,258 | 1,060 | 19 | 1,778 |
| 29 | 506 | 631 | 1,611 | 4,665 | 33 | 762 |
| 30 | 637 | 2,994 | 784 | 12,074 | 31 | 277 |
| 31 | 415 | 255,504 | 345 | 117,076 | 66 | 378 |
| 32 | 587 | 634,596 | 322 | 402,553 | 128 | 433 |
| **Working (59)** | | | | | | |
| 1 | 11 | 13 | 16 | 11 | 13 | 16 |
| 8 | 6 | 10 | 9 | 6 | 10 | 9 |
| 16 | 12 | 18 | 8 | 11 | 18 | 8 |
| 20 | 46 | 36 | 51 | 42 | 23 | 39 |
| 24 | 62 | 39 | 747 | 71 | 31 | 420 |
| 28 | 176 | 73 | 1,404 | 271 | 44 | 821 |
| 29 | 1,094 | 313 | 1,542 | 1,252 | 82 | 555 |
| 30 | 3,210 | 586 | 513 | 3,708 | 89 | 206 |
| 31 | 56,202 | 1,999 | 417 | 57,076 | 101 | 203 |
| 32 | 225,722 | 5,703 | 598 | 200,737 | 33 | 98 |

Every layer, all held-out pairs together and 95% bootstrap CIs over items are in `results/<model>/logit_lens_top5.json`. `logit_lens.py` does the same with a single token per language (the first token of the native answer; items where T and S start with the same token are left out) and gives the same picture (`results/<model>/logit_lens.json`).

* **Layers 1–15: nothing is decided.** Every language is near 10 in a million. The two arms are identical, since steering starts at layer 16.
* **Layers 17–28: English leads.** Although no one asked for English, it is the most likely of the three in both arms and groups, 3–20 times the target in layers 21–28.
* **Layers 29–32: the language is chosen.** The winning language rises from at most 1 in 1,000 at layer 28 to 20–63% at the output. With the instruction, in the failing pairs S wins (63% at the output, T 0.06%); in the working pairs T wins (23%, S 0.6%). With steering, T wins in both (40% and 20%), and S stays near zero.
* **Steering makes the failing pairs behave like the working ones.** Its effect is not visible as more T right at layer 16 (T 22 vs 21 in a million), but it changes what the last layers do: they pick T instead of S.
* **Averaging the instruction over all held-out pairs is misleading** (T 2%, S 4% at the output): it mixes two behaviours that no single pair shows; with steering the overall mean is a fair summary (T 27%, S 0.006%).
* Together with the forced start: the language is settled in the last layers, before the first token, and the first token then largely fixes the rest. Steering changes that late choice, not the reasoning.
* **The native question as a reference** (question in T, no instruction; added later, same script, `native` arm, 2.5 min on 1 B200; the two arms above unchanged). In the failing pairs, English leads in layers 17–28 here too (up to 7,062 in a million at layer 27, vs 1,543 for T), and T is chosen in the last four layers (T 20,618 at layer 29, 899,628 at the output, where the T set is by construction the native top 5). S, here just another language, stays below 21 in a million throughout. The working pairs look the same. So the English stage is not caused by the instruction, and with steering the last layers behave as natively.

**Logit lens on the Qwen models**, the same scripts with each model's dev setting of instruction + prompt-only steering: Qwen3-8B l 28 of 36, Qwen3-32B l 40 of 64, both α 1. The instruction fails (test success below 0.10) in 30 held-out pairs of Qwen3-8B and only 6 of Qwen3-32B, so the latter's failing group is small. The native top 5 cover 79% (sw) to 100% of the native distribution in Qwen3-8B, and 96–100% in Qwen3-32B. Typical probability ×10⁶ as above, selected layers; every layer is in `results/<model>/logit_lens_top5.json`:

| Qwen3-8B, layer | instr.: T | instr.: S | instr.: en | instr. + steer: T | instr. + steer: S | instr. + steer: en |
|---|---|---|---|---|---|---|
| **Failing (30)** | | | | | | |
| 20 | 4 | 6 | 27 | 4 | 6 | 27 |
| 26 | 65 | 4 | 509 | 65 | 4 | 509 |
| 28 | 24 | 0.7 | 132 | 196 | 0.02 | 41 |
| 32 | 0.09 | 0.02 | 0.6 | 129 | 0.0007 | 2 |
| 34 | 11 | 3,470 | 29 | 19,400 | 0.07 | 51 |
| 35 | 115 | 240,000 | 827 | 139,000 | 1.5 | 849 |
| 36 | 1.6 | 331,000 | 96 | 219,000 | 0.08 | 327 |
| **Working (69)** | | | | | | |
| 20 | 4 | 5 | 14 | 4 | 5 | 14 |
| 26 | 131 | 42 | 1,500 | 131 | 42 | 1,500 |
| 28 | 53 | 16 | 514 | 148 | 0.2 | 17 |
| 32 | 1.4 | 0.1 | 1.4 | 201 | 0.09 | 5 |
| 34 | 3,380 | 145 | 30 | 20,700 | 2.6 | 42 |
| 35 | 65,200 | 2,950 | 529 | 182,000 | 22 | 402 |
| 36 | 69,000 | 495 | 107 | 282,000 | 0.1 | 31 |

| Qwen3-32B, layer | instr.: T | instr.: S | instr.: en | instr. + steer: T | instr. + steer: S | instr. + steer: en |
|---|---|---|---|---|---|---|
| **Failing (6)** | | | | | | |
| 20 | 14 | 5 | 11 | 14 | 5 | 11 |
| 40 | 2.3 | 1.1 | 18 | 5 | 1.0 | 28 |
| 50 | 0.7 | 0.07 | 0.2 | 2.7 | 0.1 | 0.4 |
| 58 | 102 | 0.05 | 4 | 809 | 0.003 | 1.0 |
| 61 | 296 | 94 | 45 | 3,630 | 0.2 | 20 |
| 62 | 143 | 2,080 | 118 | 9,210 | 2.3 | 55 |
| 63 | 11 | 65,700 | 145 | 26,800 | 19 | 150 |
| 64 | 11 | 221,000 | 205 | 197,000 | 14 | 104 |
| **Working (93)** | | | | | | |
| 20 | 12 | 17 | 15 | 12 | 17 | 15 |
| 40 | 5 | 4 | 20 | 9 | 1.1 | 9 |
| 50 | 0.9 | 0.5 | 0.4 | 1.5 | 0.3 | 0.4 |
| 58 | 4 | 0.2 | 11 | 11 | 0.08 | 10 |
| 61 | 315 | 4 | 59 | 1,080 | 0.7 | 60 |
| 62 | 9,230 | 47 | 102 | 21,500 | 2.8 | 88 |
| 63 | 58,800 | 198 | 251 | 119,000 | 4.9 | 198 |
| 64 | 194,000 | 368 | 193 | 235,000 | 0.7 | 65 |

* **The main finding holds in all three models.** The language is chosen in the last 3–4 layers (Qwen3-8B 33–36, Qwen3-32B 61–64). In the failing pairs the instruction lets S win (S 33% and 22% at the output, T near zero), and steering lets T win (22% and 20%, S near zero), as in the working pairs.
* **The English stage is model-dependent.** Qwen3-8B, like Llama, has English clearly ahead in its middle layers (about 18–30, up to 15–25 times T). In Qwen3-32B, English is ahead only briefly (layers 38–44, 2–8 times T); through most of the middle all three languages are about equally (un)likely.
* **In Qwen3-32B's failing pairs, T leads S for most of the depth** (in layers 53–60, T at 40–260 in a million, S below 1), yet S overtakes it in the last 3 layers. The late layers override the target there, as in Llama (where d fell from +0.7 at layer 28 to −8.5 at layer 32 with one token per language).
* **Steering acts at once in Qwen3-8B**: at layer 28 it lifts T about 8 times and drops S about 40 times. In Llama and Qwen3-32B, which steer earlier in the network, its effect grows over the later layers.
* **Qwen3-8B's layers 31–32 are a dip** where the lens reads every language as very unlikely; no conclusion rests on them.
* **Steering also helps Qwen3-8B's working pairs** (T at the output 7% with the instruction, 28% with steering); in Llama and Qwen3-32B the working pairs end similarly with and without it.
* Check: the last layer's top-1 token is the first token of the arm's test response in 90–98% of prompts in every model (`last_top1_matches_generation` in `logit_lens.json`; the misses are mostly tokenisation differences in bn and te, and near-ties).

**Logit lens on Gemma-3-12B**, at its dev setting of instruction + prompt-only steering, l 24 of 48, α 1. The instruction fails in only 2 held-out pairs (sw→th and sw→bn), so the failing group is very small. The native top 5 cover 90% (zh) to 100% of the native distribution. Typical probability ×10⁶ as above, selected layers; every layer is in `results/gemma3-12b/logit_lens_top5.json`:

| Gemma-3-12B, layer | instr.: T | instr.: S | instr.: en | instr. + steer: T | instr. + steer: S | instr. + steer: en |
|---|---|---|---|---|---|---|
| **Failing (2)** | | | | | | |
| 12 | 9 | 1,077 | 10 | 9 | 1,077 | 10 |
| 24 | 35 | 99 | 335 | 46 | 105 | 304 |
| 28 | 84 | 1,022 | 15,501 | 158 | 528 | 10,692 |
| 32 | 64 | 668 | 341,653 | 179 | 330 | 118,481 |
| 36 | 0.4 | 6 | 70,499 | 1.7 | 0.5 | 14,610 |
| 40 | 6 | 126 | 125,372 | 124 | 0.8 | 2,762 |
| 44 | 11 | 53,983 | 34,188 | 272 | 4 | 3,627 |
| 46 | 12 | 334,180 | 863 | 1,818 | 94 | 1,543 |
| 47 | 24 | 443,964 | 1,896 | 112,512 | 131 | 5,240 |
| 48 | 1.2 | 449,776 | 84 | 469,019 | 0.4 | 79 |
| **Working (97)** | | | | | | |
| 12 | 16 | 16 | 8 | 16 | 16 | 8 |
| 24 | 95 | 68 | 271 | 129 | 47 | 259 |
| 28 | 1,140 | 475 | 11,433 | 1,091 | 325 | 8,970 |
| 32 | 1,501 | 188 | 153,616 | 1,747 | 151 | 118,775 |
| 36 | 289 | 1.9 | 11,894 | 334 | 1.4 | 12,860 |
| 40 | 1,380 | 1.7 | 1,893 | 1,668 | 0.9 | 1,827 |
| 44 | 23,142 | 6 | 3,055 | 22,964 | 1.9 | 3,568 |
| 46 | 51,515 | 41 | 804 | 56,545 | 13 | 1,106 |
| 47 | 254,379 | 199 | 1,647 | 273,322 | 24 | 2,321 |
| 48 | 232,362 | 4 | 6 | 264,896 | 0.03 | 14 |

* **The English stage is the strongest of the four models.** In layers 28–35 English is far ahead in both groups and arms (up to 15% of the probability in the working pairs, 10–100 times T, and 34% in the failing ones).
* **The language is chosen over the last 8 or so layers,** more gradually than in the other models: in the working pairs T rises from 0.1% at layer 40 to 23–26% at the output.
* **The failing pairs behave as in the other models.** With the instruction S wins (45% at the output, T about one in a million); with steering T wins (47%) and S is near zero. The effect of steering on T shows from about layer 38, well after the steered layer 24.
* **In the two failing pairs S leads from early on** (layers 11–22: S 100–1,400 in a million, T below 45), while in the working pairs the three languages are level until about layer 21.
* The working pairs end alike with and without steering (T 23% and 26%). With one token per language (`logit_lens.json`) the picture is the same: in the failing pairs d = log P(T) − log P(S) is −2 to −3.5 through layers 16–38 with the instruction and −18.6 at the output; with steering, +19.6.
* Check: the last layer's top-1 token is the first token of the test response in 97.5% (instruction) and 98.2% (steering) of prompts, and the lens at the last layer equals the model's logits exactly (Gemma 3's text model has no final logit soft-capping).

## Results: Qwen3-8B

Dev (tuning pairs): source 0.738, native 0.740; instruction 0.338, in English 0.375 (tested), as system message 0.273.

| | chosen (dev) | tuning | held-out | held-out better / worse | accuracy | in T | capped | below 0.10 / 0.30 |
|---|---|---|---|---|---|---|---|---|
| Source (in S) | – | 0.763 | 0.763 | – | 0.797 | 0.932 | 0.016 | 0 / 9 |
| Native | – | 0.763 | 0.763 | 71 / 6 [65 / 3] | 0.797 | 0.932 | 0.016 | 0 / 9 |
| Instruction | – | 0.385 | 0.412 | – | 0.766 | 0.516 | 0.037 | 30 / 42 |
| Instruction in English | – | 0.401 | 0.449 | 38 / 22 [30 / 18] | 0.771 | 0.588 | 0.029 | 17 / 41 |
| Steering, all-token | l 24, α 1 | 0.608 | 0.613 | 58 / 16 [53 / 12] | 0.690 | 0.862 | 0.115 | 9 / 12 |
| Instr. + steering, all-token | l 24, α 0.5 | 0.727 | 0.705 | 72 / 0 [67 / 0] | 0.738 | 0.941 | 0.056 | 1 / 6 |
| Steering, prompt-only | l 24, α 1 | 0.572 | 0.556 | 55 / 25 [49 / 20] | 0.751 | 0.742 | 0.039 | 14 / 19 |
| Instr. + steering, prompt-only | l 28, α 1 | 0.696 | 0.711 | 69 / 0 [62 / 0] | 0.741 | 0.945 | 0.053 | 2 / 5 |
| Instr. in English + steering, prompt-only | l 28, α 1 | 0.719 | 0.722 | 71 / 1 [67 / 0] | 0.747 | 0.962 | 0.051 | 0 / 0 |
| Control: random direction | l 28, α 1 | 0.370 | 0.402 | 4 / 12 [0 / 2] | 0.764 | 0.504 | 0.041 | 30 / 43 |
| Control: wrong language | l 28, α 1 | 0.266 | 0.301 | 23 / 49 [16 / 46] | 0.746 | 0.378 | 0.048 | 24 / 51 |

* **The same picture as Llama.** Instruction + prompt-only steering is never significantly worse than the instruction, and beats both controls (in 75 and 92 held-out pairs, 71 and 91 after Holm; never worse).
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

### Prompt variants (Qwen3-8B, test)

Mean over the 99 held-out pairs; better / worse: held-out pairs significantly different from the instruction (McNemar, p < 0.05; in square brackets Holm-corrected over the 99). The instruction and instruction + steering rows are the ones above.

| | held-out | held-out better / worse | accuracy | in T | capped | below 0.10 / 0.30 |
|---|---|---|---|---|---|---|
| Instruction (after the question, in S) | 0.412 | – | 0.766 | 0.516 | 0.037 | 30 / 42 |
| Instruction in T | 0.451 | 45 / 26 [32 / 23] | 0.754 | 0.610 | 0.032 | 17 / 36 |
| Instruction first | 0.420 | 21 / 19 [17 / 13] | 0.753 | 0.528 | 0.046 | 31 / 40 |
| Instr. + steering, prompt-only | 0.711 | 69 / 0 [62 / 0] | 0.741 | 0.945 | 0.053 | 2 / 5 |

Held-out success by T (9 pairs each), with better / worse [Holm-corrected over the 99 held-out pairs]:

| T | en | bn | de | es | fr | ja | ru | sw | te | th | zh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Instruction | 0.63 | 0.55 | 0.46 | 0.46 | 0.47 | 0.31 | 0.36 | 0.20 | 0.48 | 0.52 | 0.10 |
| Instruction in T | 0.57 (4 / 4 [3 / 4]) | 0.70 (5 / 2 [4 / 1]) | 0.33 (2 / 5 [1 / 5]) | 0.46 (3 / 2 [2 / 2]) | 0.37 (1 / 5 [1 / 4]) | 0.43 (7 / 1 [5 / 1]) | 0.41 (3 / 3 [3 / 2]) | 0.24 (5 / 0 [2 / 0]) | 0.65 (5 / 0 [3 / 0]) | 0.43 (2 / 4 [2 / 4]) | 0.37 (8 / 0 [6 / 0]) |
| Instruction first | 0.61 (3 / 2 [2 / 2]) | 0.59 (3 / 0 [2 / 0]) | 0.41 (1 / 4 [1 / 3]) | 0.46 (2 / 1 [2 / 1]) | 0.49 (1 / 2 [1 / 1]) | 0.34 (3 / 1 [2 / 0]) | 0.47 (4 / 1 [4 / 1]) | 0.22 (1 / 0 [1 / 0]) | 0.42 (1 / 3 [0 / 2]) | 0.46 (1 / 4 [1 / 2]) | 0.14 (1 / 1 [1 / 1]) |

* **Neither variant closes the gap:** 0.451 and 0.420 against 0.711 with steering; accuracy is unchanged (0.75–0.77), so both act only on the language of the answer.
* **The instruction in T is unreliable rather than better:** it lifts T = zh (0.10 to 0.37), bn, te and ja, and lowers T = de, fr, th and en; 26 held-out pairs are significantly worse (23 after Holm). The losses are in the answer language (T = de: in T 0.60 to 0.44), not in accuracy.
* **Placing the instruction first changes little** (21 better, 19 worse; 17 and 13 after Holm): letting the question tokens attend to the instruction does not help.
* Not run on Llama or Qwen3-32B, since neither variant changes the conclusion. Run on A100 80 GB, the other Qwen3-8B arms on B200; a unit's output depends on the GPU only through bf16 differences.

## Results: Qwen3-32B

Dev (tuning pairs): source 0.811, native 0.813; instruction 0.611 (tested), in English 0.582, as system message 0.451.

| | chosen (dev) | tuning | held-out | held-out better / worse | accuracy | in T | capped | below 0.10 / 0.30 |
|---|---|---|---|---|---|---|---|---|
| Source (in S) | – | 0.835 | 0.835 | – | 0.850 | 0.980 | 0.003 | 0 / 0 |
| Native | – | 0.835 | 0.835 | 42 / 12 [32 / 5] | 0.850 | 0.980 | 0.003 | 0 / 0 |
| Instruction | – | 0.616 | 0.694 | – | 0.854 | 0.802 | 0.002 | 6 / 13 |
| Steering, all-token | l 16, α 1 | 0.606 | 0.616 | 21 / 41 [17 / 28] | 0.783 | 0.782 | 0.017 | 13 / 18 |
| Instr. + steering, all-token | l 40, α 0.5 | 0.815 | 0.802 | 37 / 2 [31 / 0] | 0.846 | 0.942 | 0.003 | 2 / 2 |
| Steering, prompt-only | l 48, α 2 | 0.675 | 0.660 | 31 / 33 [24 / 25] | 0.823 | 0.797 | 0.012 | 10 / 15 |
| Instr. + steering, prompt-only | l 40, α 1 | 0.810 | 0.800 | 36 / 6 [31 / 4] | 0.836 | 0.953 | 0.004 | 1 / 1 |
| Instr. in English + steering, prompt-only | l 48, α 1 | 0.831 | 0.817 | 39 / 5 [31 / 4] | 0.849 | 0.961 | 0.003 | 0 / 0 |
| Control: random direction | l 40, α 1 | 0.620 | 0.693 | 10 / 10 [6 / 3] | 0.840 | 0.816 | 0.003 | 4 / 10 |
| Control: wrong language | l 40, α 1 | 0.682 | 0.698 | 18 / 28 [14 / 16] | 0.836 | 0.822 | 0.005 | 2 / 6 |

* **The larger model follows the instruction far more often** (0.694, 13 pairs below 0.30), so there is less to gain; steering with the instruction still adds 0.106 and removes almost all failing pairs. Without the instruction, steering is worse than the instruction alone.
* **Where the instruction already works, steering costs accuracy.** The 6 held-out pairs (4 after Holm) where instruction + prompt-only steering is significantly worse are en and zh into bn, te and sw, where the instruction scores 0.65–0.98 (above native in en→bn and en→te). There steering stays in T and loses accuracy (e.g. zh→bn 0.92 → 0.79).
* **Both controls stay at the instruction's level** (0.693 and 0.698): at this setting a direction that is not towards T neither helps nor, unlike in the 8B models, drags the answer elsewhere. Instruction + steering beats them in 41 and 51 held-out pairs (worse in 4 and 5; after Holm 30 and 35, worse in 0 and 1).
* **Settings at the edge of the layer grid:** steering, all-token chose layer 16 (the first); steering, prompt-only and the English-instruction arm chose layer 48 (the last). At layer 48 some answers start in English ("Let's calculate …") and switch to T, which costs the English-instruction arm in-T rate in en→de (0.99 → 0.80) and en→es, zh→ja.
* Closeness to native: κ with native changes by +0.057 [0.022, 0.091] without the instruction and +0.002 [−0.014, 0.018] with it; κ with source by −0.476 and −0.039 [−0.060, −0.019]. KL / top-1 by response position:

  | arm | token 1 | tokens 2–10 | 11–50 | 51+ |
  |---|---|---|---|---|
  | source | 18.51 / 0.01 | 1.03 / 0.75 | 0.39 / 0.85 | 0.10 / 0.95 |
  | steer, PO | 2.49 / 0.54 | 0.58 / 0.79 | 0.37 / 0.85 | 0.10 / 0.95 |
  | instr. | 3.53 / 0.32 | 0.64 / 0.78 | 0.44 / 0.84 | 0.12 / 0.94 |
  | instr. + steer, PO | 2.67 / 0.33 | 0.61 / 0.78 | 0.45 / 0.83 | 0.12 / 0.94 |

  With the instruction, steering moves the first token less here than in the 8B models, as the instruction alone already starts most answers in T.

## Results: Gemma-3-12B-it

Dev (tuning pairs): source 0.787, native 0.789; instruction 0.713, in English 0.709, as system message 0.735 (tested).

| | chosen (dev) | tuning | held-out | held-out better / worse | accuracy | in T | capped | below 0.10 / 0.30 |
|---|---|---|---|---|---|---|---|---|
| Source (in S) | – | 0.828 | 0.828 | – | 0.840 | 0.983 | 0.009 | 0 / 0 |
| Native | – | 0.828 | 0.828 | 45 / 2 [28 / 0] | 0.840 | 0.983 | 0.009 | 0 / 0 |
| Instruction | – | 0.757 | 0.729 | – | 0.806 | 0.905 | 0.002 | 2 / 4 |
| Instruction as system message | – | 0.775 | 0.751 | 11 / 8 [9 / 4] | 0.794 | 0.945 | 0.002 | 1 / 1 |
| Steering, all-token | l 12, α 2 | 0.767 | 0.752 | 15 / 12 [11 / 4] | 0.782 | 0.960 | 0.013 | 1 / 1 |
| Instr. + steering, all-token | l 36, α 0.5 | 0.764 | 0.774 | 17 / 4 [11 / 0] | 0.801 | 0.965 | 0.003 | 0 / 0 |
| Steering, prompt-only | l 18, α 2 | 0.773 | 0.741 | 14 / 16 [11 / 8] | 0.811 | 0.911 | 0.019 | 2 / 4 |
| Instr. + steering, prompt-only | l 24, α 1 | 0.768 | 0.777 | 15 / 5 [11 / 1] | 0.805 | 0.964 | 0.002 | 0 / 0 |
| Instr. in English + steering, prompt-only | l 24, α 0.5 | 0.755 | 0.773 | 17 / 5 [11 / 4] | 0.812 | 0.951 | 0.001 | 0 / 0 |
| Control: random direction | l 24, α 1 | 0.610 | 0.517 | 0 / 55 [0 / 43] | 0.661 | 0.712 | 0.063 | 14 / 34 |
| Control: wrong language | l 24, α 1 | 0.662 | 0.654 | 6 / 50 [4 / 37] | 0.803 | 0.810 | 0.004 | 2 / 2 |

* **The instruction mostly works, except from English and Swahili questions.** Its held-out success (0.729, 4 pairs below 0.30) is close to Qwen3-32B's. By source, English questions score 0.549 and Swahili 0.569, the others 0.71–0.82: asked in English, Gemma often answers in English (en→zh 0.20, en→de 0.26), and sw→th and sw→bn fail almost completely (0.00 and 0.02). Instruction + prompt-only steering lifts these four pairs to 0.91, 0.63, 0.76 and 0.63, and S = en to 0.858.
* **Steering does not reach native** (0.777 against 0.828). Answering in a language other than the question's costs accuracy (0.806 with the instruction against 0.840 for source and native), and steering does not recover it (0.805).
* **The instruction as system message** is the dev winner; since Gemma's template has no system role, it puts the instruction before the question. On test it reaches 0.751 (better in 11 held-out pairs, worse in 8): it fixes the English sources (en→zh 0.90, en→de 0.87) but not sw→th (0.00).
* **Both controls clearly hurt,** unlike in Qwen3-32B: a push of this norm in a direction other than T costs Gemma success (random 0.517, worse in 55 pairs; wrong language 0.654, worse in 50), so the gain comes from the direction of the vector. Instruction + prompt-only steering beats them in 57 and 56 held-out pairs and is never worse (49 and 44 after Holm).
* **One held-out pair is significantly worse after Holm:** zh→de (0.80 → 0.69).
* **Steering without the instruction** (0.752 all-token, 0.741 prompt-only) is about as good as the instruction on average, but fails elsewhere. It fixes the four pairs the instruction fails (en→zh 0.88 / 0.86, en→de 0.91 / 0.87, sw→th 0.71 / 0.73, sw→bn 0.61 / 0.72), but leaves some answers in S, mostly between Bengali and Telugu: bn→te 0.02 / 0.00 (instruction 0.79), and with prompt-only steering also te→bn 0.00, bn→sw 0.14 and th→te 0.15. It is worse than the instruction in 12 and 16 held-out pairs (4 and 8 after Holm). With the instruction, steering has no such failures.
* Closeness to native: κ with native changes by +0.035 [0.004, 0.064] without the instruction and +0.001 [−0.011, 0.012] with it; κ with source by −0.463 and −0.005 [−0.018, 0.009]. KL / top-1 by response position:

  | arm | token 1 | tokens 2–10 | 11–50 | 51+ |
  |---|---|---|---|---|
  | source | 19.96 / 0.00 | 1.37 / 0.72 | 0.47 / 0.86 | 0.14 / 0.95 |
  | steer, PO | 2.30 / 0.50 | 0.90 / 0.75 | 0.47 / 0.86 | 0.14 / 0.95 |
  | instr. | 5.72 / 0.26 | 1.55 / 0.72 | 0.63 / 0.86 | 0.21 / 0.94 |
  | instr. + steer, PO | 4.93 / 0.28 | 1.50 / 0.72 | 0.63 / 0.86 | 0.20 / 0.94 |

  With the instruction, steering moves the first token much less towards native than in the other models (KL 5.72 → 4.93), as the instruction already gets the language right in most pairs.
* Gemma ends most answers with an English template line ("Final Answer: The final answer is $\boxed{18}$"), which the judge does not score (below).

## Sensitivity to layer and α (dev)

`sensitivity.py`, from the dev grids above (mean success over the 11 tuning pairs, 550 answers per cell, standard error about ±.02). Instruction + prompt-only steering at α = 1, by layer:

| model | instruction | layers | success at α = 1 | grid cells above the instruction |
|---|---|---|---|---|
| Llama-3.1-8B | .311 | 8, 12, **16**, 20, 24 | .478, .573, **.602**, .538, .542 | 23/25 |
| Qwen3-8B | .338 | 8, 12, 16, 20, 24, **28** | .605, .615, .653, .671, .676, **.678** | 28/30 |
| Qwen3-32B | .611 | 16, 24, 32, **40**, 48 | .755, .760, .758, **.793**, .780 | 20/25 |
| Gemma-3-12B | .713 | 12, 18, **24**, 30, 36 | .702, .713, **.738**, .696, .698 | 5/25 |

* **α matters more than the layer.** α = 1 is the best α, or within .03 of it, at every layer of every model; α ≤ 0.5 does too little and α = 3 too much (Llama, layer 8: .478 at α = 1, .182 at α = 3).
* **Every tested layer helps, but not equally.** The grids cover about 25–75% of the depth. Within them the worst layer keeps 57% of the chosen setting's gain over the instruction in Llama (layer 8) and 79% in both Qwen models; Qwen3-8B improves with depth, Qwen3-32B is nearly flat.
* **Without the instruction, steering is narrower:** success is near 0 at α ≤ 0.5 (the language does not change), and the best cell is usually alone within .02.
* **Gemma's grid says little:** its instruction already works on the tuning pairs (.713 against native .789), so all cells are within .69–.74. Its gains on test come from pairs that fail with the instruction and are not among the tuning pairs.
* These are dev numbers and the chosen cell is the maximum of the grid, so it is slightly optimistic; the test results are above.

On LCB (`../language_confusion/`, all-token, mean over its six languages) the layer matters even less: with the instruction at α = 1, all layers are within .03 of each other in every model (.95–.98 Llama, .95–.96 Qwen, .93–.94 Gemma), and α ≥ 2 makes many answers degenerate (at α = 2 up to .61 of them in Llama and .41 in Qwen3-8B).

## Number of FLORES sentences for the vector (test)

`flores_subsets.py`: instruction + prompt-only steering on the 99 held-out pairs (200 test items each), with v(S→T) = μ_T − μ_S computed from n of the 997 FLORES-200 dev sentences instead of all of them, the same n sentences in S and in T (translations of each other), and the vector used as it is (not normalised). Each pair draws its own sentences, three draws (seeds) per n, nested (the n sentences are the first n of the 100). The layer and α are those chosen with the 997-sentence vectors (not retuned); n = 997 is the test run above.

Mean held-out success, mean over the three draws [range]:

| model | instruction | n = 1 | n = 3 | n = 10 | n = 100 | n = 997 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | .340 | .628 [.626, .630] | .633 [.630, .638] | .633 [.632, .634] | .635 [.632, .638] | .634 |
| Qwen3-8B | .412 | .707 [.703, .712] | .710 [.709, .711] | .710 [.708, .712] | .711 [.710, .713] | .711 |
| Qwen3-32B | .694 | .787 [.785, .791] | .796 [.790, .799] | .799 [.797, .801] | .801 [.799, .803] | .800 |
| Gemma-3-12B | .729 | .771 [.768, .775] | .775 [.771, .779] | .777 [.776, .777] | .777 [.776, .778] | .777 |

Held-out pairs significantly worse than n = 997 (McNemar, Holm over the 99 pairs), per draw; and per pair and draw, how many of the 297 are more than .10 from n = 997:

| model | n = 1 | n = 3 | n = 10 | n = 100 |
|---|---|---|---|---|
| Llama-3.1-8B | 0, 2, 0; 3 | 0, 1, 1; 3 | 0, 0, 0; 0 | 0, 0, 0; 0 |
| Qwen3-8B | 0, 0, 2; 2 | 0, 0, 0; 0 | 0, 0, 0; 0 | 0, 0, 0; 0 |
| Qwen3-32B | 1, 2, 3; 9 | 1, 0, 0; 1 | 1, 0, 0; 0 | 0, 0, 0; 0 |
| Gemma-3-12B | 4, 0, 1; 8 | 0, 0, 1; 1 | 0, 0, 1; 1 | 0, 0, 0; 0 |

Mean cosine of the vector to the 997-sentence one at the chosen layer, and mean norm ratio:

| model | n = 1 | n = 3 | n = 10 | n = 100 |
|---|---|---|---|---|
| Llama-3.1-8B | .73, 1.39 | .88, 1.15 | .96, 1.05 | .996, 1.00 |
| Qwen3-8B | .85, 1.19 | .94, 1.08 | .98, 1.03 | .998, 1.00 |
| Qwen3-32B | .72, 1.43 | .86, 1.22 | .94, 1.08 | .993, 1.01 |
| Gemma-3-12B | .79, 1.27 | .91, 1.12 | .97, 1.06 | .997, 1.01 |

* **Ten sentences give the same result as 997** in every model: the means are within .002, at most one pair per draw is worse after Holm, and no (pair, draw) is more than .10 away except one Gemma pair (zh→de, .47 in one draw).
* **One sentence keeps almost all of the gain.** The mean drops by .004–.013 (most in Qwen3-32B), and every draw is still better than the instruction in about as many pairs as n = 997 (after Holm: 54–57 Llama, 62–65 Qwen3-8B, 23–27 Qwen3-32B, 11–12 Gemma; worse in 0, 0, 2–5 and 1–2).
* **What small n costs is reliability in a few pairs, not the typical pair.** The median spread of a pair over the three draws is .030–.045 at n = 1 against .015–.025 at n = 100, and the median distance of a (pair, draw) from n = 997 is .015–.025 against .010–.015; even at n = 100, where the vector is nearly identical (cosine > .99), a pair moves by about .01, as a slightly different vector flips a few items. But 2–9 of the 297 (pair, draw) at n = 1 are more than .10 off (22 over the four models, 19 of them worse, mostly into de, te, zh, sw and ru; the worst: Gemma zh→de .17 against .69 at n = 997, Qwen3-32B sw→de .17 against .42 and sw→ru .17 against .32). By n = 3 this is down to 0–3, with single bad draws left (Qwen3-32B th→ru .10 against .90, Gemma zh→de .23).
* **Fewer sentences give longer, noisier vectors** (E‖v_n‖² = ‖v‖² + noise), so at n = 1 the effective α is 1.2–1.4 on average and up to 3.4 for single pairs; a single sentence can even point away from the 997-sentence vector (one Gemma pair, cosine −0.83). Normalising the vectors to the 997-sentence norm might remove part of the n = 1 tail; it was not tried, as the means barely move.
* The sentences are matched translations (FLORES is parallel), so the content cancels in μ_T − μ_S; how much unmatched text in S and T would need is not tested.

Runs (all B200 except as noted, one worker per GPU for Qwen3-32B and Gemma-3-12B, two for the 8B models): n = 10 and 100, one 2-GPU job per two models (one GPU each), 13 h 15 min (Llama, Qwen3-32B) and 11 h 17 min (Qwen3-8B, Gemma-3-12B); n = 1 and 3, 5–6 h per model with helpers (main job + helpers: Llama 1 + 2 GPUs, Qwen3-8B 1 + 1, Gemma-3-12B 1 + 1, Qwen3-32B 2 + 2, of which 98 of its 594 units on 2 B300). `analyse`: 8 min for all four models on a CPU.

## Baseline: ReCoVeR (test)

ReCoVeR (Findings of EMNLP 2025), its training-free variant, as published: the same direction as ours, α · normalize(μ_T − μ_S), added at every block, at every position except the first token and at every generated token, each block's output rescaled to its unsteered norm. It runs with the instruction prompt in S. Its code ([github.com/hSterz/recover](https://github.com/hSterz/recover) @ 4c6eb4d) has no licence, so it is reimplemented (`Model.generate(recover=...)` in `../mgsm_steering/generate.py`) and checked against it (`recover_check.py`, below). The only change to their setup is that α is chosen on dev, like our settings: α ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16}, doubled further while the best α is the largest, by the highest mean success over the 11 tuning pairs (ties to the smaller α). One α for all pairs: 0.1 (Llama), 0.5 (Qwen3-8B), 1 (Qwen3-32B), 16 (Gemma-3-12B), none at the edge of the grid. The dev optimum is narrow: in Llama α = 0.2 gives 0.293 and α ≥ 0.5 gives 0, through degenerate answers; Gemma, whose mid-layer hidden norm is about 50,000 (Llama's about 6), falls from 0.722 at α = 16 to 0.660 at 32 and 0.28 at 64.

Mean test success over the 99 held-out pairs (accuracy, in-T rate, answers that hit the 2048-token limit):

| | Llama-3.1-8B | Qwen3-8B | Qwen3-32B | Gemma-3-12B |
|---|---|---|---|---|
| Instruction | 0.340 (.673, .492, .030) | 0.412 (.766, .516, .037) | 0.694 (.854, .802, .002) | 0.729 (.806, .905, .002) |
| ReCoVeR | 0.567 (.579, .959, .076) | 0.661 (.747, .860, .042) | 0.796 (.838, .944, .003) | 0.772 (.793, .974, .002) |
| Instruction + all-token steering (ours) | 0.602 (.646, .928, .037) | 0.705 (.738, .941, .056) | 0.802 (.846, .942, .003) | 0.774 (.801, .965, .003) |
| **Instruction + prompt-only steering (ours)** | **0.634** (.655, .963, .041) | **0.711** (.741, .945, .053) | **0.800** (.836, .953, .004) | **0.777** (.805, .964, .002) |
| ReCoVeR − prompt-only steering [95% CI] | −.067 [−.076, −.059] | −.051 [−.058, −.043] | −.004 [−.010, +.001] | −.005 [−.011, +.002] |
| Held-out pairs ReCoVeR better / worse than prompt-only steering | 3 / 37 [2 / 8] | 7 / 19 [2 / 13] | 8 / 8 [2 / 3] | 3 / 7 [1 / 1] |
| ReCoVeR − all-token steering [95% CI] | −.036 [−.045, −.027] | −.044 [−.050, −.038] | −.006 [−.011, −.001] | −.002 [−.008, +.004] |

McNemar per pair, Holm over the 99 pairs in square brackets; bootstrap CIs as in the headline table (`results/<model>/recover.json`).

* **ReCoVeR fixes the language about as well as our steering** (in-T .86–.97), and is significantly better than the instruction in most failing pairs (52, 55, 28 and 11 pairs after Holm).
* **It is below prompt-only steering in the two models where the instruction fails most**, by .067 (Llama) and .051 (Qwen3-8B), and ties in Qwen3-32B and Gemma-3-12B. In Llama the loss is in the maths: its accuracy falls below the instruction's (.579 against .673; prompt-only steering .655), and twice as many answers run to the token limit (.076 against .041). Steering every block at every generated token over a long chain of thought is what our method avoids: one layer, the prompt only.
* Error consistency with the native question: ReCoVeR − prompt-only steering −.016 [−.032, +.002], +.010 [−.008, +.028], +.013 [−.002, +.028], +.004 [−.012, +.020] (Cohen's κ).

**Checks of the reimplementation** (`recover_check.py`, against their code):
* Hooks: the same tokens through their `Steer` hooks and ours (Llama and Qwen3-8B, 4 pairs, α ∈ {0.2, 2} × norm restore off / on, every prompt position and 8 teacher-forced steps). With the same vector the logits are identical except in 4 of 32 settings (at most 0.5, bf16, against a steering effect of 9–50); prompt-only identical in all.
* Vectors: their collection code against ours, cosine of μ_T − μ_S ≥ 0.9999 at every hidden state (Qwen3-8B, Qwen3-32B, Gemma-3-12B, 4 pairs).
* Their LCB setting (Llama, 14 languages, 100 tokens, their system prompt; greedy here, they sample): LPR 0.916 without steering and 0.975 with α = 0.2 and norm restore, against 91.0 and 97.9 in their paper.

Not tested: ReCoVeR+ (their trained variant). They tuned for 100-token answers, not for 2048-token chains of thought; the comparison is between the two methods, not of single factors (layers, positions; see the next section); of their models only Llama is used here.

## Ablation: one layer and prompt-only (test)

Our method (instruction + prompt-only steering at one layer) differs from ReCoVeR in which layers are steered and in whether the generated tokens are. Each choice is tested by changing only it, everything else as in our method, against our method at its own dev setting:

* **Tokens:** instruction + all-token steering at one layer (the existing arm).
* **Layers:** instruction + prompt-only steering at every block (`recover.py` with `XL_ARM=every_steer_prompt`). Block j adds α · (μ_T − μ_S)(j), the raw difference of the mean hidden states after block j (not normalised, no norm restore), from the start of the user message to the end of the prompt. The last block is left out: its stored hidden state is after the final norm. α ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2} (the additions sum over the blocks, so α is far smaller than at one layer), extended by halvings or doublings while the best is at an edge, chosen like ReCoVeR's α: the highest mean dev success over the 11 tuning pairs. α = 0.05 in both models, not at the edge (dev 0.600 Llama, 0.658 Qwen3-8B; α = 0.1 gives 0.545 and 0.555, α ≥ 0.5 at most 0.13).

Before the runs, a check (`every_layer_check.py`, Llama): with one layer given as a tuple, `Model.generate` gives the same tokens as the single-layer path of our runs (16/16 prompts, all-token and prompt-only). Llama-3.1-8B and Qwen3-8B; one job on 8 B200 GPUs (`every_layer.sh`, 1 h 15 min with the LCB runs); the analysis is `ablation.py` (`results/<model>/ablation.json`).

Mean test success over the 99 held-out pairs [95% CI] (accuracy, in-T rate):

| | Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| Instruction | 0.340 (.673, .492) | 0.412 (.766, .516) |
| **Ours: one layer, prompt-only** | **0.634** [.598, .667] (.655, .963) | **0.711** [.684, .736] (.741, .945) |
| Tokens: one layer, all-token | 0.602 [.568, .633] (.646, .928) | 0.705 [.678, .729] (.738, .941) |
| Layers: every layer, prompt-only | 0.634 [.598, .666] (.655, .966) | 0.703 [.675, .729] (.721, .965) |
| ReCoVeR | 0.567 [.531, .600] (.579, .959) | 0.661 [.635, .683] (.747, .860) |

Ours − the other arm: mean success difference [95% CI]; held-out pairs where ours is significantly better / worse (McNemar p < .05; after Holm over the 99 pairs in square brackets):

| ours − | Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| all-token (tokens) | +.031 [+.024, +.038]; 14 / 2 [8 / 0] | +.007 [+.001, +.012]; 6 / 7 [3 / 0] |
| every layer (layers) | +.000 [−.006, +.007]; 7 / 4 [2 / 3] | +.008 [+.001, +.015]; 11 / 7 [3 / 4] |
| ReCoVeR | +.067 [+.059, +.076]; 37 / 3 [8 / 2] | +.051 [+.043, +.058]; 19 / 7 [13 / 2] |

* **Prompt-only is better than all-token** in both models: clearly in Llama (+.031, mostly through the in-T rate, .963 against .928), slightly in Qwen3-8B (+.007, CI just above 0). It never steers the answer itself.
* **Steering every layer gains nothing:** a tie in Llama, and in Qwen3-8B slightly worse (−.008): more answers in T (.965 against .945) but lower accuracy (.721 against .741). One layer is enough, and the smaller intervention.
* **Both choices together account for the gap to ReCoVeR.** Llama, every layer *and* all-token steering with our vector (`XL_ARM=every_steer`, α = 0.05): success 0.585 (accuracy .588, in T .991, 10.8% of the answers at the token limit), close to ReCoVeR's 0.567 (.579, .959, 7.6%) and −.049 [−.057, −.041] below our method. Steering every block at every generated token costs the maths, with or without ReCoVeR's normalisation. Qwen3-8B's run of this arm was stopped before test, as not needed for the ablation; its partial test files in `generations/every_steer/` are not used.

The new arms ran on B200 GPUs, our original arms on B200 and B300 (`run.sh` above); comparisons are paired over items, not over runs.

## Tests against the native question (test)

`native_tests.py`: per held-out pair, an exact McNemar test of each arm against the native question (asked in T, the same 200 items), Holm-corrected over the 99 pairs; per T and overall, the mean difference arm − native with a 95% bootstrap CI over items.

Held-out pairs significantly below / above native, after Holm (uncorrected in brackets):

| | Llama-3.1-8B | Qwen3-8B | Qwen3-32B | Gemma-3-12B |
|---|---|---|---|---|
| Instruction | 65 / 1 (71 / 2) | 65 / 3 (71 / 6) | 32 / 5 (42 / 12) | 28 / 0 (45 / 2) |
| Instruction + prompt-only steering | 23 / 0 (53 / 4) | 20 / 9 (32 / 16) | 13 / 2 (22 / 14) | 21 / 1 (38 / 5) |
| Instruction in English + prompt-only steering | 19 / 0 (46 / 5) | 18 / 10 (34 / 17) | 11 / 1 (21 / 10) | 24 / 0 (39 / 5) |

* **Steering comes close to native but does not reach it.** Its mean is .035–.089 below native, with CIs that exclude 0 in every model, and 13–23 held-out pairs are significantly below native after Holm (against 28–65 with the instruction).
* **Where steering is above native, the native run is weak.** 8 of Qwen3-8B's 9 such pairs are into sw, where its native question scores 0.150 (steering 0.397). The others are en→te (Qwen3-8B, Gemma-3-12B), en→fr and en→ja (Qwen3-32B).
* **Pairs into English are the ones most often below native** (3, 5, 3 and 9 of 9 after Holm), where native is highest (0.80–0.97).
* **Japanese: steering matches native, it does not beat it.** Into ja, instruction + prompt-only steering is above native in three models (Llama 0.658 against 0.620, +.038 [−.019, +.098]; Qwen3-8B +.012 [−.042, +.067]; Gemma-3-12B +.028 [−.020, +.078]), but the CI excludes 0 only in Qwen3-32B (+.049 [+.001, +.094]), and no pair into ja is significantly below native in any model.

## Checks on the judge

* The in-language check looks for English lines, not lines in S. In the earlier 10-pair version of this experiment, also rejecting a line of ≥ 5 words labelled S with p > 0.5 changed the test means by at most 0.017 and lowered the instruction most. It has not been applied to this run.
* Lines under 5 words are not checked on their own, so a few short S fragments in a T answer can pass (e.g. mgsm-0195 in sw→de). This can only favour the steered arms.
* **Gemma's answer template is not scored.** Gemma-3 often ends an answer, even one written entirely in T, with the English line "Final Answer: The final answer is $\boxed{18}$". Lines matching `^[\s*#]*Final Answer[\s*:]*The final answer is.*$` (`ANSWER_LINE` in `../mgsm_steering/analyse.py`) are removed before the language check; with them, that line of ≥ 5 English words failed answers that were otherwise in T. No Llama or Qwen response in any experiment of this project contains the line, so the rule changes none of their verdicts. For Gemma it only turns failures into successes (4,647 responses, none the other way). It changed one dev pick, prompt-only steering (l 12, α 3 → l 18, α 2), whose test units were rerun. All-token steering ties exactly on dev (401 / 550 at l 12 and l 18, α 2) and keeps l 12 by the earlier-layer rule; the selection rounds the means, so that float summation order cannot break such ties.
