# Output-language control by steering (Experiment 2 pilot)

Can a language vector make Llama-3.1-8B-Instruct **answer in another language than it was asked in**, and does it do this better than simply asking for that language? This is the pilot from [EXP2.md](../../../docs/plans/EXP2.md), run on a small scale with French, Hungarian, Chinese and Russian.

**Result: steering switches the language reliably, but on MKQA it never beats the plain instruction "Answer in T."** The instruction already gets the language right 97–100% of the time and keeps more answers correct, so the task leaves steering no room. By the plan's stopping rule we did not build the full experiment (KL, 1000 test items) on MKQA.

## Reproduce

```bash
.venv/bin/pip install -r requirements.txt                 # adds fasttext-wheel
.venv/bin/python src/data/download.py mkqa fasttext       # -> data/raw_data/{mkqa,fasttext}/ (pinned in manifest.json)
G=src/experiments/output_language/generate.py
slurm/submit.sh -J outlang_single -- bash -c "python $G --stage single --langs en fr hu & python $G --stage single --langs zh ru; wait"
slurm/submit.sh -J outlang_multi  -- bash -c "python $G --stage multi --langs fr hu & python $G --stage multi --langs zh ru; wait"
.venv/bin/python src/experiments/output_language/analyse.py   # -> tables on stdout + results/summary.json
```

Each stage takes about 3–7 minutes on one B200. The outputs were reproduced response for response on a B200; another GPU type may change a few greedy decodes.

| File | Contents |
|---|---|
| `generate.py` | MKQA splits, prompts, language vectors, steering hook, generation for all conditions |
| `analyse.py` | Language fidelity, accuracy, configuration selection |
| `results/summary.json` | Every number below (the per-item generations in `results/generations/` are not committed) |

## Setup

* **Data:** MKQA, items with a short answer in English and in all four languages (6758 of 10,000), ordered by a hash of the item ID. The first 300 are the *compute* split (vectors), the next 60 *test*, the next 100 *val*.
* **Prompt:** the question, then "Explain briefly in 1-3 sentences, then give the final answer in \boxed{}." Greedy decoding, at most 200 new tokens.
* **Conditions:**

  | Name | What the model is given |
  |---|---|
  | Plain | the English prompt; reference only |
  | Instruction | the English prompt + " Answer in *T*." (baseline 1) |
  | Native | question and format instruction both translated into *T* (baseline 2) |
  | Steering | the English prompt with nothing about *T*; α·v_(en→T)(l) is added to the output of layer l at every token of the user message and every generated token |

* **Vectors:** v_(en→T)(l) = μ_T(l) − μ_en(l), the mean layer-l state over the tokens of the 300 compute questions (BOS excluded), in *T* and in English.
* **Steering grids:**
  * single layer: l ∈ {8, 12, 16, 20, 24} × α ∈ {0.5, 1, 2, 3}, test only;
  * 5 layers at once (8, 12, 16, 20, 24): α ∈ {0.1, 0.2, 0.3, 0.5};
  * all 32 layers at once: α ∈ {0.02, 0.05, 0.1, 0.25}.

  A vector added early stays in the residual stream, so the multi-layer pushes add up; hence the smaller α.
* **Measures:**
  * *in language*: share of explanations (the text before `\boxed`) that fastText `lid.176` labels as *T*;
  * *correct*: share whose normalized `\boxed{}` content contains MKQA's localized answer or one of its aliases.
* **Selection:**
  * single layer: α = 1 (the only strength that switches the language at every layer without breaking the output) and the layer with the best test accuracy, so this row is slightly optimistic;
  * multi layer: the α with the best val accuracy among those with ≥ 95% val fidelity, then reported on test.

## Results

Test, 60 items per language: share correct, with the share in the target language in brackets. Plain English is correct 57% of the time (against the English answers).

| | French | Hungarian | Chinese | Russian |
|---|---|---|---|---|
| Instruction | **45%** (100%) | **43%** (100%) | **17%** (97%) | **27%** (100%) |
| Native question | 38% (100%) | 22% (100%) | 7% (100%) | 23% (100%) |
| Single layer, α = 1 | 42% (100%), l = 16 | 30% (100%), l = 12 | 15% (100%), l = 20 | 22% (100%), l = 8 |
| 5 layers | 43% (97%), α = 0.2 | 15% (100%), α = 0.3 | 15% (100%), α = 0.2 | 25% (100%), α = 0.2 |
| All 32 layers | 38% (100%), α = 0.05 | 27% (100%), α = 0.05 | 12% (97%), α = 0.05 | 22% (100%), α = 0.05 |

1. **Steering alone switches the language.** With no instruction in the prompt, the right strength gives 97–100% answers in *T* at every layer tried.
2. **It works like a switch.** Below some strength the answer stays in English (single layer α = 0.5 at layers 16–24; α = 0.1 and 0.02 for the multi-layer variants); one step above, it is fully in *T*; stronger still, the model stops giving a boxed answer and accuracy collapses (at the highest strengths Hungarian turns into repeated syllables). There is no setting that switches the language and keeps English-level accuracy.
3. **Several layers do not help.** 5 layers ties the instruction in French, Chinese and Russian (within 2 points, far below the ±12-point noise of 60 items) and is clearly worse in Hungarian; all 32 layers is never better than 5.
4. **Asking natively is not a ceiling.** The native question is the least accurate condition in every language, most of all in Hungarian.

**Caveats:** 60 test items; fastText on 1–3 sentence explanations; localized-answer matching is harsh for Chinese.

**Deviations from [EXP2.md](../../../docs/plans/EXP2.md):** a pilot, so 60 test and 100 val items instead of 1000 and 200; fastText instead of GlotLID; 200 instead of 256 new tokens; no KL measurement.

**Next:** steering can only win where the instruction fails. The next step is an instruction-only pilot in settings expected to pull answers back to English (an English context passage, long answers, the instruction far from the end), moving on to steering only where the instruction gets the language right less than about 85–90% of the time.
