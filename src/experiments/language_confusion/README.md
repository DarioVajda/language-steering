# Instruction vs language steering on the Language Confusion Benchmark

Does a language vector make an instruction-tuned model **answer in the requested language more reliably than the instruction itself?** We use the cross-lingual part of the Language Confusion Benchmark (LCB; Marchisio et al., 2024): English prompts that ask for a reply in another language ("Write in Japanese."). The model sometimes ignores the request and answers in English, especially for long prompts. Four models: Llama-3.1-8B-Instruct (the main one), Qwen3-8B and Qwen3-32B (thinking off), and Gemma-3-12B-it.

**Result: yes, wherever the instruction fails.** On held-out test prompts, steering raises the line-level pass rate from 0.83–0.94 to 0.90–1.00 in Japanese, Chinese, Russian, Hindi and Arabic, and removes nearly all English fallbacks. Turkish, where the instruction already scores 0.97, is unchanged. Adding the vector on top of the instruction is the most consistent arm. Qwen3-8B, whose instruction fails more often (0.77–0.91), gains significantly in all six languages with all-token steering, also after a Holm correction over the languages. Qwen3-32B follows the instruction better (0.87–0.97) and gains less: all-token steering is significant in five languages (three after Holm), and no prompt-only gain survives the correction. Gemma-3-12B-it almost never follows the instruction (LPR .07–.35; 63–89% of its answers start in English, often with a preamble such as "Okay, here's an essay on Machine Learning in Japanese, …"). Steering raises it to .82–1.00 in all six languages, significant after Holm in every mode and setting. LCB only checks the language, though. By hand, steered answers show more code-mixing inside words and, in Turkish and Hindi, some loss of fluency. A benchmark with checkable answers is still needed to measure quality.

## Reproduce

```bash
.venv/bin/pip install -r requirements.txt                          # adds fugashi, unidic-lite (Japanese tokenizer for the scorer)
.venv/bin/python src/data/download.py lcb lcb_words fasttext_bin flores   # -> data/raw_data/ (pinned in manifest.json)
# Llama-3.1-8B-Instruct, as it was run (the commands now need LCB_MODEL=llama-3.1-8b; results in results/llama-3.1-8b/):
G=src/experiments/language_confusion/generate.py
slurm/submit.sh -J lcb_instr_dev -- python $G --pilot          # instruction only, all 14 languages, dev (language choice)
for L in ja zh ru hi tr ar; do                                 # one language per GPU; the runs are independent
  slurm/submit.sh -J lcb_$L -t 01:30:00 -- env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $G --lang $L
done
slurm/submit.sh -J lcb_universal --gres gpu:2 -- bash -c "for L in ja zh ru hi tr ar; do python $G --lang \$L --universal; done"   # after all dev grids; shared setting on test
.venv/bin/python src/experiments/language_confusion/analyse.py --pilot   # -> pilot table + results/summary_pilot.json
.venv/bin/python src/experiments/language_confusion/analyse.py           # -> test tables + results/summary.json
# prompt-only steering: the same pipeline with --prompt-only (files *_prompt.jsonl), two languages per GPU
slurm/submit.sh -J lcb_prompt_only --gres gpu:3 -- bash -c 'for s in "" --universal; do i=0; for L in ja zh ru hi tr ar; do CUDA_VISIBLE_DEVICES=$((i / 2)) python src/experiments/language_confusion/generate.py --lang $L --prompt-only $s & i=$((i + 1)); done; wait; done'
.venv/bin/python src/experiments/language_confusion/analyse.py --prompt-only   # -> results/summary_prompt.json

# Qwen3-8B and Qwen3-32B (thinking off): everything above in one job per model, run.sh <model> <gpus> <workers per gpu>
R=src/experiments/language_confusion/run.sh
slurm/submit.sh -J lcb-q8-main -t 06:00:00 --gres gpu:2 --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- bash $R qwen3-8b 2 2
slurm/submit.sh -J lcb-q32-main -t 12:00:00 --gres gpu:4 --constraint 'GPU_BRD:B200|GPU_BRD:B300' -- bash $R qwen3-32b 4 1
slurm/submit.sh -J lcb-gemma -t 12:00:00 --gres gpu:2 --constraint GPU_BRD:B200 -- bash $R gemma3-12b 2 2
# optional helpers on other nodes join the same queue: ... -- bash $R <model> <gpus> <workers per gpu> <main job id>
```

The model is set by `LCB_MODEL` (a key of `MODELS` in `analyse.py`: `llama-3.1-8b`, `qwen3-8b`, `qwen3-32b`, `gemma3-12b`); each model has its own layer grid, the one used in cross-lingual MGSM, and writes to `results/<model>/`. `run.sh` hands the runs (pilot; per language all-token and prompt-only; the universal runs once all dev grids of a mode exist) to workers through a claim queue, then writes the three summaries. Qwen3-32B is loaded from the project's own Hugging Face cache (`HF_HUB_CACHE=logs/hf_cache`, revision `9216db5`, downloaded from the Hub); Gemma-3-12B-it (revision `96b6f1e`) from the shared cache. Gemma's run took 1 h 34 min on 4 B200 (2 in the main job, 2 in a helper).

Each language run does everything in one job: vectors, the dev grid, selection, and test. It takes about 25 minutes on one A100. On a 40 GB A100 the Turkish test stage ran out of memory without `expandable_segments`. ja, zh, ru, hi and ar ran on A100s. Turkish ran its dev grid on an A100 and its test stage on a B200, because an existing `<T>_dev.jsonl` is reused. Another GPU type may change a few greedy decodes. For example, the zh dev instruction run differs from the pilot by 2 of 102 prompts.

| File | Contents |
|---|---|
| `data.py` | LCB loading, dev/test split, removal of the language instruction |
| `generate.py` | FLORES-200 vectors, steering hook, generation of all conditions, dev grid → selection → test |
| `analyse.py` | The LCB metrics (port of the official scorer), extra measures, selection rule, McNemar test, tables |
| `run.sh` | One job per model: every run below through a claim queue, then the tables |
| `results/<model>/summary.json`, `summary_prompt.json`, `summary_pilot.json` | Every number below (the per-item generations in `results/<model>/generations/` are not committed) |

## Setup

* **Data:** the LCB cross-lingual test sets (repo `for-ai/language-confusion` @ `0cfdbb4a`). There are three sources: okapi (100 prompts, median 15 words), sharegpt (100, 18 words) and complex_prompts (99, 159 words). The prompts are identical across languages apart from the language name, and the instruction is placed at the start, middle or end of the prompt. LCB has no split. We split by a hash of (source, row): **dev** is 102 prompts, **test** is 197. The same prompts fall in each split for every language.
* **Languages:** a 14-language instruction-only pilot on dev (below) picked the languages where the instruction fails most: ja, zh, ru, hi and ar. tr is a lower-resource control where the instruction works.
* **Conditions:** the prompt is the only user message. Decoding is greedy with at most 100 new tokens, as in the benchmark. Qwen3 runs with thinking off (`enable_thinking=False`, which ends the prompt with an empty thinking block).

  | Name | What the model is given |
  |---|---|
  | Instruction | the LCB prompt as is (baseline) |
  | Steering | the prompt with its language instruction removed (e.g. "Write in Russian." or "… in Russian"), + α·v at layer l |
  | Instruction + steering | the prompt as is, + α·v at layer l |

* **Vectors:** v_(en→T)(l) = μ_T(l) − μ_en(l). This is the mean output of block l over the tokens (the first, BOS for Llama and Gemma, excluded) of the 997 FLORES-200 dev sentences, as raw text without the chat template. The vector is added at every position from the start of the user message and at every generated token. LCB has no target-language prompts, so the vectors come from out-of-domain parallel text.
* **Measures:**
  * *LPR* (LCB's line-level pass rate): fastText `lid.176.bin` labels every line of ≥ 5 tokens as T, with p > 0.3.
  * *WPR* (word-level, non-Latin scripts only): a line-level pass contains no English word from LCB's list.
  * Both are means over the three sources, as in LCB. `analyse.py` ports the official `compute_metrics.py`.
  * Three measures of our own: *English*, the share of responses whose first scored line is English; *degenerate*, the share with no scorable line, with a substring of ≥ 4 characters repeated ≥ 4 times in a row, or with a thinking tag (`<think>` or `</think>`: strong prompt-only steering can make Qwen3 reopen thinking and reason inside the answer; *thinking* is reported separately); and *drift*, among responses whose first scored line is in T, the share with a later English line.
* **Grid and selection (dev):** l × α ∈ {0.25, 0.5, 1, 2, 3}, separately for both steering conditions, with the layer grid of cross-lingual MGSM: l ∈ {8, 12, …, 24} (Llama, 32 layers), {8, 12, …, 28} (Qwen3-8B, 36 layers), {16, 24, …, 48} (Qwen3-32B, 64 layers), {12, 18, …, 36} (Gemma-3-12B, 48 layers: its full-attention blocks). The rule picks the highest LPR among settings whose degenerate rate is at most the instruction's + 0.05. Ties go to lower degeneration, then smaller α, then the earlier layer. The chosen settings are then run once on test.
* **Test comparison:** an exact McNemar test on per-prompt line-level passes, paired against the instruction. † marks a p-value still below .05 after a Holm correction over the six languages (each model, condition and setting separately; `p_holm` in the summaries).

## Results: Llama-3.1-8B-Instruct

### Pilot: the instruction alone, all 14 languages (dev, 102 prompts)

| | fr | de | es | pt | it | ja | ko | zh | ar | tr | hi | ru | id | vi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LPR | .93 | .95 | .95 | .93 | .97 | .83 | .91 | .84 | .91 | .94 | .89 | .89 | .83 | .93 |
| LPR, complex prompts | .80 | .88 | .86 | .86 | .91 | .66 | .79 | .62 | .79 | .91 | .71 | .74 | .77 | .85 |
| English | .01 | .01 | .03 | .02 | .00 | .10 | .06 | .08 | .06 | .01 | .06 | .07 | .01 | .01 |
| LPR, Qwen3-8B | .91 | .89 | .91 | .87 | .91 | .85 | .85 | .82 | .89 | .86 | .91 | .91 | .73 | .87 |
| LPR, Qwen3-32B | .90 | .92 | .92 | .92 | .94 | .86 | .92 | .87 | .95 | .92 | .95 | .95 | .76 | .92 |
| LPR, Gemma-3-12B | .17 | .15 | .15 | .13 | .18 | .15 | .11 | .08 | .10 | .38 | .22 | .09 | .31 | .17 |
| English, Gemma-3-12B | .80 | .81 | .79 | .84 | .77 | .81 | .86 | .84 | .87 | .59 | .74 | .87 | .55 | .80 |

The instruction fails mostly on the long complex prompts. The languages were chosen on Llama's pilot and kept for Qwen and Gemma; Gemma fails in all 14, mostly by answering in English; for Qwen3-8B, Turkish is not an easy language (.86). Indonesian's low LPR is largely fastText labelling it as Malay, not English fallback.

### Test (197 prompts per language)

LPR, with the gained / lost prompts against the instruction and the McNemar p in brackets:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .86 | .83 | .94 | .92 | .93 | .97 |
| Steering | .94 (+26/−9, p = .006†) | .90 (+23/−9, p = .02†) | **.99** (+10/−1, p = .01†) | **1.00** (+15/−0, p < .001†) | **1.00** (+13/−0, p < .001†) | .96 (+2/−4, p = .69) |
| Instruction + steering | **.97** (+28/−5, p < .001†) | **.91** (+19/−3, p = .001†) | .98 (+11/−3, p = .06) | **1.00** (+16/−0, p < .001†) | .99 (+12/−0, p < .001†) | **.98** (+3/−1, p = .63) |
| *chosen l, α: steering* | 8, 1 | 24, 1 | 12, 1 | 12, 1 | 12, 1 | 8, 1 |
| *chosen l, α: instr. + steering* | 16, 1 | 20, 0.5 | 12, 2 | 16, 1 | 8, 1 | 24, 1 |

The other measures, given as instruction → steering → instruction + steering:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| LPR, complex prompts | .77 → .84 → .97 | .69 → .77 → .81 | .87 → 1.00 → .97 | .78 → 1.00 → 1.00 | .85 → 1.00 → .98 | .92 → .92 → .95 |
| English | .09 → .00 → .00 | .06 → .01 → .03 | .05 → .00 → .00 | .06 → .00 → .00 | .05 → .00 → .00 | .02 → .01 → .01 |
| WPR | .96 → .92 → .95 | .95 → .89 → .93 | .95 → .92 → .95 | .97 → .98 → .98 | .91 → .90 → .91 | – |
| Degenerate | .01 → .03 → .01 | .00 → .00 → .01 | .01 → .04 → .04 | .00 → .02 → .00 | .03 → .03 → .03 | .01 → .01 → .03 |

1. **Steering fixes the instruction's failures.** Steering alone is significant in all five languages except tr, and instruction + steering in four (ru: p = .06); all of these hold after Holm. English fallbacks drop from 5–9% to 0–3%, and the largest gains are on the long complex prompts (hi .78 → 1.00, ja .77 → .97).
2. **Instruction + steering is the most consistent arm.** It is within 0.01 of the best LPR in every language and loses the fewest prompts. Steering alone loses 9 prompts in ja and zh.
3. **No gain where the instruction already works.** In Turkish all three arms agree within 2 points.
4. **Steering acts like a switch (dev grid).** Without the instruction, α ≤ 0.5 leaves the answer in English (LPR 0.00–0.35). α = 1 switches it at every layer (0.78–1.00). From α = 2, degeneration grows quickly, especially from layer 16 on, and at α = 3 most settings exceed 50%. Every chosen setting has α ≤ 2.
5. **Word-level mixing rises slightly under steering alone** (WPR −0.03 to −0.06 in ja, zh and ru).

### One setting for all languages (test)

The same selection rule, applied to the mean dev LPR and degenerate rate over the six languages and requiring the degenerate limit in every language, picks **l = 12, α = 1 for steering** and **l = 24, α = 1 for instruction + steering**. These were run once on test (`--universal`), with a fresh instruction run in the same job. That job ran on B300s, so the instruction differs from the table above by up to 0.01.

LPR, with the gained / lost prompts against the instruction and the McNemar p in brackets:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .85 | .83 | .94 | .91 | .93 | .97 |
| Steering (12, 1) | .95 (+27/−7, p = .001†) | .90 (+21/−7, p = .01†) | .99 (+9/−1, p = .02†) | 1.00 (+17/−0, p < .001†) | .99 (+14/−1, p = .001†) | .96 (+2/−4, p = .69) |
| Instruction + steering (24, 1) | .95 (+26/−7, p = .001†) | .92 (+24/−5, p = .001†) | 1.00 (+10/−0, p = .002†) | 1.00 (+16/−0, p < .001†) | 1.00 (+13/−0, p < .001†) | .98 (+3/−1, p = .63) |

* **Per-language tuning buys almost nothing.** Every LPR is within 0.02 of the per-language setting, and the same comparisons are significant, also after Holm. The dev grids are flat: at α = 1, layers 8–16 are within about 2 dev prompts of the best in nearly every language.
* **Steering at (12, 1) is the safer default.** Instruction + steering at layer 24 keeps LPR at 1.00 in hi and ar, but degeneration rises to .06–.07 and WPR drops (hi .98 → .87, ar .91 → .86).

### Prompt-only steering (test)

The vector is added only from the start of the user message to the end of the prompt (`--prompt-only`). The first generated token is still predicted from a steered position, but later tokens are not steered. Everything else is the same pipeline (dev grid, per-language and shared selection, test), with α extended to {…, 4, 6}. That job ran on H100s, with its own instruction run.

LPR, all-token → prompt-only; brackets give the prompt-only McNemar p against the instruction when it is ≥ .05. After Holm, steering stays significant in ja and hi (and worse in tr), instruction + steering only in ja and hi (ar, ru and zh are not):

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction (prompt-only job) | .85 | .83 | .94 | .91 | .93 | .97 |
| Steering | .94 → **1.00** | .90 → .88 (p = .21) | .99 → .99 (p = .06) | 1.00 → .98 | 1.00 → .94 (p = .61) | .96 → .92 (worse, p = .007) |
| Instruction + steering | .97 → .97 | .91 → .89 (p = .10) | .98 → .98 | 1.00 → .99 | .99 → .97 | .98 → .95 (p = .27) |
| *chosen l, α (prompt-only): steering* | 12, 3 | 12, 3 | 12, 2 | 12, 2 | 16, 2 | 8, 2 |
| *chosen l, α (prompt-only): instr. + steering* | 12, 2 | 16, 3 | 20, 3 | 16, 2 | 24, 4 | 8, 3 |

* **Once an answer starts in T, it stays there.** Drift is 0–4% for prompt-only and 0–2% for all-token steering, so steering the generated tokens is not what keeps 100-token answers in T.
* **A single push needs to be stronger, and the working range is narrower.** With steering alone at α = 1, dev LPR is only 0.80–0.96. α ≥ 4 degenerates, and α = 6 breaks every language.
* **The shared setting fails for steering alone.** Larger α breaks the degenerate limit in some language, so the rule picks l = 8, α = 1. That setting gives no significant gain on test and lowers hi (.89) and tr (.91). The shared instruction + steering setting (l = 16, α = 2) gives .90–.99, significant in ja, zh, hi and ar (after Holm in ja, hi and ar).
* **Less code-mixing only in ja and zh:** WPR under steering alone rises from .92 to .96 (ja) and from .89 to .95 (zh); it does not change in ru, hi or ar, and degeneration is similar.

All-token steering is therefore the more robust choice on LCB, which scores only the language (see *All-token or prompt-only* below). Prompt-only matches it only with per-language tuning, and is worse in ar and tr.

**Hand check:** we read 20 test prompts (ja, zh, ru, hi, tr), all three arms: 2 random prompts per language, 1 where steering alone gained and 1 where it lost.
* On the random prompts, all arms were fluent and on topic, and wrong facts appeared equally often in every arm.
* Of the 5 gains, 3 were real: romanized Russian and Hindi became native script, and an English JSON analysis became Turkish. 2 were scorer artifacts: song titles translated, and a truncated list line.
* Of the 5 losses, 1 was an artifact (all-Chinese JSON). The others were mild code-mixing, a short loop, and one partly English Turkish answer.
* Steering-specific problems: code-mixing inside words ("Machine学习", "Quantум", Cyrillic "крок" in a Chinese story), more spelling errors in Turkish, and occasional drift in meaning (a Hindi metaphor).

## Results: Qwen3-8B and Qwen3-32B

The same pipeline (`run.sh`): languages, split, α grids, selection rule and test, with each model's layer grid. The runs used B200 and B300 GPUs. Each test file holds the instruction run and the steering runs of one process, so every comparison is paired on one GPU; across files the instruction runs differ by at most one prompt (Qwen3-8B tr in the prompt-only shared-setting file: .87). Test, 197 prompts per language; LPR, with the gained / lost prompts against the instruction and the McNemar p in brackets.

#### Qwen3-8B, per-language settings

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .88 | .77 | .91 | .90 | .89 | .88 |
| Steering, all-token | .97 (+22/−5, p = .002†) | .90 (+36/−11, p < .001†) | 1.00 (+16/−0, p < .001†) | .98 (+16/−3, p = .004†) | 1.00 (+17/−0, p < .001†) | .95 (+17/−3, p = .003†) |
| Instruction + steering, all-token | .97 (+21/−4, p < .001†) | .92 (+36/−7, p < .001†) | 1.00 (+16/−0, p < .001†) | .98 (+17/−2, p < .001†) | .99 (+21/−0, p < .001†) | .97 (+18/−3, p = .001†) |
| Steering, prompt-only | .92 (+17/−9, p = .169) | .85 (+36/−20, p = .044) | .99 (+15/−1, p < .001†) | .96 (+18/−7, p = .043) | .94 (+20/−11, p = .150) | .95 (+17/−5, p = .017) |
| Instruction + steering, prompt-only | .96 (+20/−5, p = .004†) | .90 (+37/−12, p < .001†) | 1.00 (+14/−0, p < .001†) | .95 (+15/−7, p = .134) | .99 (+19/−0, p < .001†) | .96 (+18/−5, p = .011†) |
| *chosen l, α, all-token: steering; instr. + steering* | 16, 2; 20, 2 | 16, 2; 16, 3 | 12, 1; 20, 1 | 16, 1; 16, 1 | 12, 2; 20, 2 | 8, 1; 16, 1 |
| *chosen l, α, prompt-only: steering; instr. + steering* | 8, 2; 8, 1 | 16, 6; 16, 6 | 8, 4; 8, 2 | 8, 2; 12, 4 | 8, 4; 20, 6 | 12, 3; 8, 3 |

#### Qwen3-8B, one setting for all languages

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .88 | .77 | .91 | .90 | .89 | .88 |
| Steering, all-token (16, 1) | .93 (+18/−9, p = .122) | .90 (+35/−10, p < .001†) | .97 (+15/−4, p = .019) | .99 (+16/−2, p = .001†) | .99 (+21/−3, p < .001†) | .93 (+15/−7, p = .134) |
| Instruction + steering, all-token (16, 1) | .95 (+20/−7, p = .019†) | .89 (+30/−7, p < .001†) | .99 (+16/−1, p < .001†) | .99 (+17/−1, p < .001†) | .99 (+21/−1, p < .001†) | .97 (+18/−3, p = .001†) |
| Steering, prompt-only (12, 2) | .94 (+17/−7, p = .064) | .72 (+25/−34, p = .298) | .97 (+15/−4, p = .019) | .95 (+15/−7, p = .134) | .94 (+17/−9, p = .169) | .86 (+16/−17, p = 1.00) |
| Instruction + steering, prompt-only (12, 2) | .95 (+18/−6, p = .023†) | .86 (+30/−13, p = .014†) | .99 (+14/−0, p < .001†) | .99 (+17/−1, p < .001†) | .97 (+17/−2, p < .001†) | .95 (+20/−6, p = .009†) |

Qwen3-8B, the other measures on test (per-language settings), instruction → steering → instruction + steering:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| LPR, complex prompts, all-token | .77 → .92 → .97 | .61 → .84 → .75 | .80 → 1.00 → 1.00 | .76 → 1.00 → .97 | .71 → 1.00 → .98 | .72 → .89 → .96 |
| LPR, complex prompts, prompt-only | .77 → .84 → .94 | .61 → .91 → .86 | .80 → 1.00 → 1.00 | .76 → .92 → .91 | .71 → .98 → .97 | .72 → .92 → .97 |
| English, all-token | .08 → .00 → .00 | .10 → .02 → .01 | .07 → .00 → .00 | .09 → .01 → .00 | .09 → .00 → .00 | .09 → .04 → .02 |
| English, prompt-only | .08 → .04 → .01 | .10 → .09 → .04 | .07 → .00 → .00 | .09 → .03 → .05 | .09 → .03 → .00 | .09 → .03 → .00 |
| WPR, all-token | .90 → .94 → .96 | .93 → .94 → .97 | .95 → .93 → .96 | .96 → .95 → .98 | .97 → .98 → .98 | – |
| WPR, prompt-only | .90 → .84 → .92 | .93 → .97 → .98 | .95 → .95 → .97 | .96 → 1.00 → 1.00 | .97 → .97 → .99 | – |
| Degenerate, all-token | .00 → .02 → .04 | .01 → .01 → .02 | .02 → .00 → .02 | .02 → .04 → .03 | .01 → .05 → .01 | .04 → .02 → .04 |
| Degenerate, prompt-only | .00 → .01 → .00 | .01 → .01 → .02 | .02 → .00 → .02 | .02 → .04 → .03 | .01 → .01 → .01 | .04 → .02 → .03 |

#### Qwen3-32B, per-language settings

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .89 | .87 | .94 | .97 | .96 | .93 |
| Steering, all-token | .95 (+15/−5, p = .041) | .95 (+19/−4, p = .003†) | 1.00 (+11/−0, p < .001†) | .95 (+6/−10, p = .454) | 1.00 (+8/−0, p = .008†) | .97 (+10/−2, p = .039) |
| Instruction + steering, all-token | .95 (+19/−6, p = .015) | .93 (+15/−4, p = .019) | .99 (+9/−0, p = .004†) | 1.00 (+6/−0, p = .031) | .98 (+7/−1, p = .070) | .97 (+8/−0, p = .008†) |
| Steering, prompt-only | .89 (+15/−14, p = 1.00) | .92 (+14/−5, p = .064) | .98 (+8/−1, p = .039) | .94 (+6/−12, p = .238) | .94 (+3/−8, p = .227) | .93 (+8/−8, p = 1.00) |
| Instruction + steering, prompt-only | .93 (+14/−5, p = .064) | .93 (+16/−5, p = .027) | .96 (+4/−1, p = .375) | .98 (+6/−3, p = .508) | .96 (+7/−6, p = 1.00) | .96 (+9/−4, p = .267) |
| *chosen l, α, all-token: steering; instr. + steering* | 16, 3; 40, 2 | 24, 2; 16, 3 | 16, 2; 16, 2 | 40, 1; 24, 1 | 32, 1; 16, 1 | 16, 2; 24, 2 |
| *chosen l, α, prompt-only: steering; instr. + steering* | 24, 2; 24, 4 | 16, 4; 16, 3 | 16, 2; 16, 2 | 16, 2; 40, 4 | 32, 2; 16, 1 | 16, 4; 24, 6 |

#### Qwen3-32B, one setting for all languages

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .89 | .87 | .94 | .97 | .96 | .93 |
| Steering, all-token (24, 2) | .96 (+16/−2, p = .001†) | .95 (+19/−4, p = .003†) | .99 (+11/−2, p = .022) | .97 (+5/−5, p = 1.00) | 1.00 (+8/−0, p = .008†) | .97 (+10/−2, p = .039) |
| Instruction + steering, all-token (24, 1) | .91 (+10/−6, p = .454) | .90 (+13/−8, p = .383) | .97 (+6/−0, p = .031) | 1.00 (+6/−0, p = .031) | 1.00 (+8/−1, p = .039) | .97 (+8/−1, p = .039) |
| Steering, prompt-only (16, 2) | .94 (+16/−5, p = .027) | .87 (+13/−14, p = 1.00) | .98 (+8/−1, p = .039) | .94 (+6/−12, p = .238) | .92 (+6/−14, p = .115) | .92 (+7/−10, p = .629) |
| Instruction + steering, prompt-only (16, 2) | .91 (+12/−7, p = .359) | .90 (+11/−6, p = .332) | .96 (+4/−1, p = .375) | .98 (+3/−2, p = 1.00) | 1.00 (+8/−1, p = .039) | .96 (+7/−2, p = .180) |

Qwen3-32B, the other measures on test (per-language settings), instruction → steering → instruction + steering:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| LPR, complex prompts, all-token | .78 → .87 → .94 | .71 → .95 → .86 | .90 → 1.00 → 1.00 | .90 → 1.00 → 1.00 | .88 → 1.00 → .98 | .85 → .95 → .95 |
| LPR, complex prompts, prompt-only | .78 → .86 → .88 | .71 → .86 → .89 | .90 → .97 → .97 | .90 → .93 → 1.00 | .88 → .97 → .98 | .85 → .90 → .93 |
| English, all-token | .03 → .00 → .00 | .05 → .01 → .01 | .03 → .00 → .00 | .03 → .02 → .00 | .03 → .00 → .00 | .03 → .01 → .01 |
| English, prompt-only | .03 → .04 → .01 | .05 → .01 → .01 | .03 → .01 → .01 | .03 → .03 → .01 | .03 → .03 → .01 | .03 → .04 → .01 |
| WPR, all-token | .96 → .98 → .97 | .92 → .95 → .98 | .94 → .95 → .98 | .96 → .97 → .98 | .98 → .94 → .94 | – |
| WPR, prompt-only | .96 → .94 → .98 | .92 → .98 → .98 | .94 → .92 → .95 | .96 → .95 → .97 | .98 → .95 → .92 | – |
| Degenerate, all-token | .01 → .02 → .02 | .01 → .01 → .01 | .03 → .02 → .02 | .01 → .02 → .01 | .03 → .02 → .02 | .02 → .02 → .01 |
| Degenerate, prompt-only | .01 → .04 → .01 | .01 → .01 → .01 | .03 → .01 → .03 | .01 → .05 → .01 | .03 → .03 → .02 | .02 → .02 → .03 |

* **Qwen3-8B: the instruction fails more often than in Llama** (LPR .77–.91, English fallbacks 7–10%, Turkish included). All-token steering is significant in all six languages, with and without the instruction, and English falls to ≤ .04. Prompt-only steering with the instruction is significant in five (hi: p = .13), also after Holm; without it in four, after Holm only in ru.
* **Qwen3-32B follows the instruction better** (.87–.97, English 3–5%), so there is less to fix. All-token steering is significant in five languages (not hi, .95 against .97), with the instruction also in five (ar: p = .07); after Holm, three (ru, zh, ar) and two (ru, tr). Prompt-only steering with the instruction is significant only in zh; without it only in ru, and it loses more prompts than it gains in hi and ar (not significant). No prompt-only comparison survives Holm: the gains are a few prompts out of 197, where the instruction already passes .87–.97.
* **Shared setting:** for Qwen3-8B, instruction + steering stays significant in all six languages in both modes, also after Holm. Prompt-only steering without the instruction fails again (zh .72, tr .86), as in Llama. For Qwen3-32B, shared all-token steering (24, 2) is significant in five languages (three after Holm: ja, zh, ar); shared instruction + steering is weaker than per-language (ja, zh not significant; none after Holm).
* **The gains are largest on the long complex prompts**, as in Llama (Qwen3-8B zh .61 → .75–.91, Qwen3-32B zh .71 → .86–.95).
* **Leaked thinking** is at most .01 at every chosen setting and at most .03 anywhere in Qwen3-32B's dev grid. It matters for strong prompt-only steering of Qwen3-8B: before thinking tags counted as degenerate, the rule chose α = 6 in ja, and up to 123 of 197 test answers reopened thinking and reasoned inside the answer, which LPR passes.
* **Settings at the edge of the grid:** α = 6, the largest, in Qwen3-8B prompt-only zh (both arms) and ar (instruction + steering), and in Qwen3-32B prompt-only tr (instruction + steering). Qwen3-32B often chose layer 16, the lowest of its grid.
* Drift is ≤ .04 in every arm, so once an answer starts in T it stays there, as in Llama.

## Results: Gemma-3-12B-it

The same pipeline (`run.sh`), with the layer grid {12, 18, 24, 30, 36}, on B200 GPUs. The instruction runs are identical in every test file. Test, 197 prompts per language; LPR, with the gained / lost prompts against the instruction and the McNemar p in brackets.

#### Gemma-3-12B, per-language settings

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .13 | .07 | .09 | .19 | .12 | .35 |
| Steering, all-token | 1.00 (+170/−0, p < .001†) | .91 (+168/−1, p < .001†) | 1.00 (+177/−0, p < .001†) | 1.00 (+160/−0, p < .001†) | 1.00 (+172/−0, p < .001†) | .98 (+124/−1, p < .001†) |
| Instruction + steering, all-token | .97 (+167/−2, p < .001†) | .91 (+168/−1, p < .001†) | 1.00 (+178/−0, p < .001†) | .99 (+159/−1, p < .001†) | .99 (+173/−0, p < .001†) | .99 (+127/−0, p < .001†) |
| Steering, prompt-only | .91 (+156/−2, p < .001†) | .82 (+151/−3, p < .001†) | .98 (+172/−0, p < .001†) | .99 (+158/−0, p < .001†) | .98 (+170/−1, p < .001†) | .98 (+118/−2, p < .001†) |
| Instruction + steering, prompt-only | .93 (+159/−1, p < .001†) | .85 (+155/−1, p < .001†) | .98 (+173/−0, p < .001†) | .98 (+157/−0, p < .001†) | .99 (+171/−0, p < .001†) | .96 (+123/−3, p < .001†) |
| *chosen l, α, all-token: steering; instr. + steering* | 24, 2; 18, 3 | 24, 3; 18, 3 | 12, 2; 12, 2 | 12, 3; 24, 1 | 12, 2; 12, 2 | 30, 1; 12, 3 |
| *chosen l, α, prompt-only: steering; instr. + steering* | 12, 2; 12, 2 | 12, 4; 12, 4 | 12, 4; 12, 2 | 12, 2; 12, 2 | 12, 3; 18, 2 | 12, 6; 18, 2 |

#### Gemma-3-12B, one setting for all languages

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .13 | .07 | .09 | .19 | .12 | .35 |
| Steering, all-token (12, 3) | .99 (+169/−0, p < .001†) | .88 (+163/−2, p < .001†) | 1.00 (+177/−0, p < .001†) | 1.00 (+160/−0, p < .001†) | 1.00 (+172/−0, p < .001†) | .96 (+125/−3, p < .001†) |
| Instruction + steering, all-token (18, 3) | .97 (+167/−2, p < .001†) | .91 (+168/−1, p < .001†) | 1.00 (+178/−0, p < .001†) | 1.00 (+159/−0, p < .001†) | .99 (+173/−1, p < .001†) | .99 (+128/−0, p < .001†) |
| Steering, prompt-only (12, 6) | .92 (+155/−3, p < .001†) | .87 (+159/−1, p < .001†) | .95 (+170/−2, p < .001†) | .99 (+154/−2, p < .001†) | .95 (+165/−5, p < .001†) | .98 (+118/−2, p < .001†) |
| Instruction + steering, prompt-only (12, 4) | .92 (+157/−2, p < .001†) | .85 (+155/−1, p < .001†) | .98 (+173/−0, p < .001†) | .98 (+157/−1, p < .001†) | .98 (+171/−2, p < .001†) | .97 (+125/−1, p < .001†) |

Gemma-3-12B, the other measures on test (per-language settings), instruction → steering → instruction + steering:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| LPR, complex prompts, all-token | .30 → 1.00 → .95 | .20 → .88 → .86 | .24 → 1.00 → 1.00 | .31 → 1.00 → 1.00 | .33 → 1.00 → .98 | .42 → .97 → .98 |
| LPR, complex prompts, prompt-only | .30 → .91 → .91 | .20 → .75 → .80 | .24 → .97 → .95 | .31 → .97 → .97 | .33 → .97 → .98 | .42 → 1.00 → .95 |
| English, all-token | .83 → .00 → .00 | .88 → .00 → .01 | .89 → .00 → .00 | .78 → .00 → .00 | .87 → .00 → .00 | .63 → .01 → .00 |
| English, prompt-only | .83 → .00 → .00 | .88 → .04 → .02 | .89 → .01 → .01 | .78 → .00 → .00 | .87 → .00 → .00 | .63 → .01 → .01 |
| WPR, all-token | .56 → .98 → .97 | .26 → .98 → .97 | .56 → .98 → .99 | .75 → .95 → .98 | .62 → .99 → .99 | – |
| WPR, prompt-only | .56 → .97 → .95 | .26 → 1.00 → .98 | .56 → .96 → .94 | .75 → .96 → .95 | .62 → .97 → .97 | – |
| Degenerate, all-token | .00 → .00 → .01 | .00 → .01 → .01 | .01 → .02 → .01 | .00 → .01 → .00 | .01 → .03 → .01 | .00 → .03 → .01 |
| Degenerate, prompt-only | .00 → .01 → .01 | .00 → .00 → .01 | .01 → .05 → .02 | .00 → .01 → .01 | .01 → .03 → .02 | .00 → .07 → .02 |

* **Gemma ignores the instruction far more often than the other models:** LPR .07–.35, and 63–89% of the answers start in English, typically with an English preamble ("Okay, here's an essay on Machine Learning in Japanese, …"). Turkish, the control where Llama's instruction works, fails too (.35).
* **Steering fixes it in every language, mode and setting:** LPR .82–1.00, English ≤ .04, every comparison p < .001 and significant after Holm, with at most 5 prompts lost.
* **Word-level mixing falls as well** (WPR .26–.75 → .94–1.00).
* **Prompt-only is a little weaker than all-token**, as in the other models (zh .82–.87 against .88–.91, ja .91–.93 against .97–1.00), mostly on the long complex prompts.
* **One setting is enough:** every shared-setting LPR is within .03 of the per-language one. Most chosen settings use layer 12, the lowest of the grid; α = 6, the largest, in tr (prompt-only steering) and in the shared prompt-only setting.
* Degeneration is at most .07 (tr, prompt-only steering) and drift at most .07.

## All-token or prompt-only

On LCB all-token steering keeps the language at least as well as prompt-only steering in every model, by a wide margin without the instruction. With the instruction, the two are within .03 of each other in every language for Llama and Qwen, and within .06 for Gemma (zh). LCB scores only the language of a 100-token answer, so it cannot show what steering the generated tokens costs. Cross-lingual MGSM ([../mgsm_crosslingual/](../mgsm_crosslingual/)) can: at matched settings, all-token steering loses accuracy over long answers, partly through loops and answers that run to the token limit.

## Sensitivity to layer and α (dev)

From `../mgsm_crosslingual/sensitivity.py` (all-token, mean over the six languages, 102 prompts each). With the instruction at α = 1, all layers are within .03 of each other in every model (.95–.98 Llama, .95–.96 Qwen3-8B and Qwen3-32B, .93–.94 Gemma), above the instruction (.89, .87, .92, .17) with at most .05 degenerate answers. α ≥ 2 makes many answers degenerate (up to .61 of them in Llama, .41 in Qwen3-8B, .11 in Qwen3-32B at α = 2; more at α = 3), which the selection rule excludes. Gemma needs α ≥ 1, as its instruction barely works (α = 0.5 gives .46–.56). Without the instruction, α ≤ 0.5 leaves the answers in English (LPR ≤ .23).

## Caveats

* LCB measures only the language, not quality or correctness.
* The main settings are tuned per language on dev (a shared setting does as well, above).
* fastText is unreliable on code, JSON and quoted English names.
* The hand check covers Llama only.

**Deviations from LCB:** our dev/test split (LCB has none); 6 of its 14 languages; LCPR (the harmonic mean of LPR and WPR) is not reported.

Correctness under steering, against the instruction and a native-language baseline, is measured in [../mgsm_crosslingual/](../mgsm_crosslingual/).
