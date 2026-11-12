# Instruction vs language steering on MGSM

On MGSM, does a language vector make Llama-3.1-8B-Instruct **answer in the target language without losing accuracy?** The measure is *success*: an answer counts only if it is mathematically correct and written in the target language T. The setup follows the Language Confusion Benchmark experiment (`../language_confusion/`), now with checkable answers.

**Result: steering adds little.** The instruction already yields answers in T 99% of the time, so there is almost nothing to fix. On test, the success rate averaged over 10 languages is:
* **0.770 with the instruction alone;**
* 0.786 when prompt-only steering is added on top, significant only in Spanish;
* 0.763 when all-token steering is added on top.

**Steering alone is worse.** Its mean is 0.60 in both versions:
* in de, es, fr, ru and ja it roughly matches the instruction;
* it fails where the vector is poor: in Bengali it steers the model into Hindi, in Telugu the answers degenerate, and in Swahili the model often falls back to English.

The native question is also worse than the instruction: 0.715 vs 0.770. The model reasons better from the English question.

## Reproduce

```bash
G=src/experiments/mgsm_steering
slurm/submit.sh -J mgsm_steering -t 03:00:00 --gres gpu:3 --gres-flags=disable-binding -c 12 --mem 160G \
  --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- bash -c 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; G=src/experiments/mgsm_steering; python $G/generate.py vectors || exit 1; for i in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$((i / 2)) python $G/generate.py dev & done; wait; python $G/analyse.py --select || exit 1; for i in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$((i / 2)) python $G/generate.py test & done; wait; python $G/analyse.py'
```

One job runs every stage: vectors, the dev grid, selection, test and the tables. Two workers share each GPU. Workers take units (one configuration) from a shared queue of claim files under `results/generations/claims/<job id>/`. Helper jobs on other nodes can join by exporting the main job's `SLURM_JOB_ID` and running the same `dev` and `test` workers. `analyse.py` waits until every unit of a stage exists.

The reported run used 9 GPUs: 4 B200, 3 B300 and 2 A100-80GB. It took about 45 min, mostly the dev grid. The batching of a unit's prompts is fixed, so a unit's output depends on which GPU ran it only through bf16 differences.

| File | Contents |
|---|---|
| `generate.py` | dev/test split, prompts, FLORES-200 vectors, steered generation (a row leaves the batch when it finishes), work units |
| `analyse.py` | correctness, the in-language check, success, selection, McNemar test, tables |
| `results/summary.json` | every number below (the per-item generations in `results/generations/` are not committed) |

## Setup

* **Data:** MGSM, 250 grade-school problems in each of bn, de, es, fr, ja, ru, sw, te, th and zh, plus the English originals. MGSM has no dev split. The items are sorted by crc32(id): the **first 50 are dev**, the other **200 are test**. The same items fall in each split for every language.
* **Prompting:** the question is the only user message. It is zero-shot, with no CoT instruction: the model reasons step by step by default (median 150–600 tokens). Decoding is greedy, with at most 2048 new tokens.
* **Arms (for target T):**

  | Name | What the model is given |
  |---|---|
  | English | the English question (reference) |
  | Native | the question in T |
  | Instruction | the English question + "\n\nAnswer in T." (baseline) |
  | Steering | the English question, + α·v at layer l |
  | Instruction + steering | the instruction prompt, + α·v at layer l |

  Each steering arm comes in two versions:
  * *all-token*: v is added from the start of the user message, during prefill and at every generated token;
  * *prompt-only*: v is added to the prompt only.
* **Vectors:** v_(en→T)(l) = μ_T(l) − μ_en(l), the mean output of block l over the tokens (BOS excluded) of the 997 FLORES-200 dev sentences, as in the LCB experiment.
* **Success = correct ∧ in T.**
  * *Correct:* the number after the last "Final answer:", else the last number. Parsing is locale-aware and handles native digits (`src/icl/tasks.py`). It must equal the gold answer.
  * *In T:* digits, punctuation and symbols become spaces. Then fastText `lid.176.bin` must label the whole response as T with p > 0.3. For T ≠ English, no line of ≥ 5 words may be labelled English with p > 0.5.
  * Why not the line rule alone: LCB's rule (every line of ≥ 5 tokens labelled T) rejected 15 of 40 Swahili pilot answers that were in Swahili, because fastText mislabels short Swahili lines. The rule here accepted all 180 pilot answers in en, de, ja, bn and sw, and still catches a switch to English partway through.
  * No separate degeneration limit: a degenerate answer is wrong, so it already counts as a failure.
* **Grid and selection (dev):**
  * grid: l ∈ {8, 12, 16, 20, 24} × α ∈ {0.25, 0.5, 1, 2, 3}, for each of the 4 steering arms;
  * one setting per arm, shared by all languages: the highest dev success averaged over the 10 languages (500 answers);
  * ties go to the smaller α, then the earlier layer.
* **Test comparison:** an exact McNemar test on per-item success within each language, paired against the instruction.

## Results

### Dev selection (mean success over the 10 languages; instruction 0.730, native 0.640)

| Arm | Chosen | Dev success | Best of the grid at α ≥ 1 |
|---|---|---|---|
| Steering, all-token | l 8, α 1 | 0.550 | — |
| Instruction + steering, all-token | l 16, α 0.25 | 0.716 | 0.562 (l 8/12, α 1) |
| Steering, prompt-only | l 8, α 1 | 0.574 | — |
| Instruction + steering, prompt-only | l 20, α 0.25 | 0.736 | 0.730 (l 20, α 1) |

* Without the instruction, α 0.25 scores 0: the model answers in English.
* At α ≥ 2 the all-token answers degenerate. For example, at l 8, α 2, 190 of 500 hit the token cap.
* With the instruction, the weakest α is best, so steering is at most neutral.
* The full grids are in `summary.json` (`dev_grid`).

### Test (200 items per language): success

| | bn | de | es | fr | ja | ru | sw | te | th | zh | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Native | **0.64**↓ | 0.78 | 0.83 | 0.75 | **0.62**↓ | 0.77 | 0.64 | 0.65 | 0.73 | 0.76 | 0.715 |
| Instruction | 0.77 | 0.79 | 0.78 | 0.81 | 0.77 | 0.79 | 0.69 | 0.71 | 0.79 | 0.82 | 0.770 |
| Steering, all-token | **0.01**↓ | 0.76 | 0.79 | 0.80 | **0.67**↓ | 0.80 | **0.48**↓ | **0.26**↓ | **0.71**↓ | **0.74**↓ | 0.602 |
| Instr. + steering, all-token | 0.75 | 0.78 | 0.81 | 0.82 | 0.77 | 0.76 | **0.62**↓ | 0.76 | 0.78 | 0.80 | 0.763 |
| Steering, prompt-only | **0.01**↓ | **0.69**↓ | 0.76 | 0.78 | 0.72 | 0.76 | **0.55**↓ | **0.20**↓ | **0.71**↓ | 0.80 | 0.596 |
| Instr. + steering, prompt-only | 0.78 | 0.81 | **0.83**↑ | 0.80 | 0.79 | 0.81 | 0.70 | 0.71 | 0.82 | 0.82 | 0.786 |

Bold: McNemar p < 0.05 against the instruction (↑ better, ↓ worse). English questions answered in English score 0.80.

Mean over languages of the other measures:

| | accuracy | in T | capped at 2048 |
|---|---|---|---|
| Native | 0.719 | 0.996 | 0.012 |
| Instruction | 0.776 | 0.993 | 0.024 |
| Steering, all-token | 0.669 | 0.859 | 0.074 |
| Instr. + steering, all-token | 0.773 | 0.988 | 0.015 |
| Steering, prompt-only | 0.712 | 0.805 | 0.044 |
| Instr. + steering, prompt-only | 0.791 | 0.991 | 0.021 |

* **The instruction already solves the language problem on MGSM.** It is in T 99% of the time, in every language except Swahili (0.94). Success is therefore limited by the arithmetic, and steering can only cost accuracy. With the instruction and a weak α (0.25), it costs nothing. The prompt-only variant is even +1.6 points on average, but it is significant only in es (+13 / −2).
* **Steering alone works in high-resource languages and fails in others:**
  * in de, es, fr and ru it is within noise of the instruction;
  * **Bengali:** the en→bn vector at l 8 sends the model into **Hindi** (179 of 200 answers in both versions), so success is 0.01 even though half the answers are correct;
  * **Telugu:** all-token answers degenerate (44% hit the cap, with broken characters). Prompt-only answers are often in English or Hindi;
  * **Swahili:** the model falls back to English or mixes it in.
  * The single shared setting (l 8, α 1) is too weak for some languages and too strong for others.
* **Native questions** hurt accuracy compared with English questions plus an instruction. The gap is significant in bn and ja.
