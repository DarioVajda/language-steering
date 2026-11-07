# Instruction vs language steering on the Language Confusion Benchmark

Does a language vector make Llama-3.1-8B-Instruct **answer in the requested language more reliably than the instruction itself?** We use the cross-lingual part of the Language Confusion Benchmark (LCB; Marchisio et al., 2024): English prompts that ask for a reply in another language ("Write in Japanese."). The model sometimes ignores the request and answers in English, especially for long prompts.

**Result: yes, wherever the instruction fails.** On held-out test prompts, steering raises the line-level pass rate from 0.83–0.94 to 0.90–1.00 in Japanese, Chinese, Russian, Hindi and Arabic, and removes nearly all English fallbacks. Turkish, where the instruction already scores 0.97, is unchanged. Adding the vector on top of the instruction is the most consistent arm. LCB only checks the language, though. By hand, steered answers show more code-mixing inside words and, in Turkish and Hindi, some loss of fluency. A benchmark with checkable answers is still needed to measure quality.

## Reproduce

```bash
.venv/bin/pip install -r requirements.txt                          # adds fugashi, unidic-lite (Japanese tokenizer for the scorer)
.venv/bin/python src/data/download.py lcb lcb_words fasttext_bin flores   # -> data/raw_data/ (pinned in manifest.json)
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
```

Each language run does everything in one job: vectors, the dev grid, selection, and test. It takes about 25 minutes on one A100. On a 40 GB A100 the Turkish test stage ran out of memory without `expandable_segments`. ja, zh, ru, hi and ar ran on A100s. Turkish ran its dev grid on an A100 and its test stage on a B200, because an existing `<T>_dev.jsonl` is reused. Another GPU type may change a few greedy decodes. For example, the zh dev instruction run differs from the pilot by 2 of 102 prompts.

| File | Contents |
|---|---|
| `data.py` | LCB loading, dev/test split, removal of the language instruction |
| `generate.py` | FLORES-200 vectors, steering hook, generation of all conditions, dev grid → selection → test |
| `analyse.py` | The LCB metrics (port of the official scorer), extra measures, selection rule, McNemar test, tables |
| `results/summary.json`, `results/summary_prompt.json`, `results/summary_pilot.json` | Every number below (the per-item generations in `results/generations/` are not committed) |

## Setup

* **Data:** the LCB cross-lingual test sets (repo `for-ai/language-confusion` @ `0cfdbb4a`). There are three sources: okapi (100 prompts, median 15 words), sharegpt (100, 18 words) and complex_prompts (99, 159 words). The prompts are identical across languages apart from the language name, and the instruction is placed at the start, middle or end of the prompt. LCB has no split. We split by a hash of (source, row): **dev** is 102 prompts, **test** is 197. The same prompts fall in each split for every language.
* **Languages:** a 14-language instruction-only pilot on dev (below) picked the languages where the instruction fails most: ja, zh, ru, hi and ar. tr is a lower-resource control where the instruction works.
* **Conditions:** the prompt is the only user message. Decoding is greedy with at most 100 new tokens, as in the benchmark.

  | Name | What the model is given |
  |---|---|
  | Instruction | the LCB prompt as is (baseline) |
  | Steering | the prompt with its language instruction removed (e.g. "Write in Russian." or "… in Russian"), + α·v at layer l |
  | Instruction + steering | the prompt as is, + α·v at layer l |

* **Vectors:** v_(en→T)(l) = μ_T(l) − μ_en(l). This is the mean output of block l over the tokens (BOS excluded) of the 997 FLORES-200 dev sentences, as raw text without the chat template. The vector is added at every position from the start of the user message and at every generated token. LCB has no target-language prompts, so the vectors come from out-of-domain parallel text.
* **Measures:**
  * *LPR* (LCB's line-level pass rate): fastText `lid.176.bin` labels every line of ≥ 5 tokens as T, with p > 0.3.
  * *WPR* (word-level, non-Latin scripts only): a line-level pass contains no English word from LCB's list.
  * Both are means over the three sources, as in LCB. `analyse.py` ports the official `compute_metrics.py`.
  * Three measures of our own: *English*, the share of responses whose first scored line is English; *degenerate*, the share with no scorable line or with a substring of ≥ 4 characters repeated ≥ 4 times in a row; and *drift*, among responses whose first scored line is in T, the share with a later English line.
* **Grid and selection (dev):** l ∈ {8, 12, 16, 20, 24} × α ∈ {0.25, 0.5, 1, 2, 3}, separately for both steering conditions. The rule picks the highest LPR among settings whose degenerate rate is at most the instruction's + 0.05. Ties go to lower degeneration, then smaller α, then the earlier layer. The chosen settings are then run once on test.
* **Test comparison:** an exact McNemar test on per-prompt line-level passes, paired against the instruction.

## Results

### Pilot: the instruction alone, all 14 languages (dev, 102 prompts)

| | fr | de | es | pt | it | ja | ko | zh | ar | tr | hi | ru | id | vi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LPR | .93 | .95 | .95 | .93 | .97 | .83 | .91 | .84 | .91 | .94 | .89 | .89 | .83 | .93 |
| LPR, complex prompts | .80 | .88 | .86 | .86 | .91 | .66 | .79 | .62 | .79 | .91 | .71 | .74 | .77 | .85 |
| English | .01 | .01 | .03 | .02 | .00 | .10 | .06 | .08 | .06 | .01 | .06 | .07 | .01 | .01 |

The instruction fails mostly on the long complex prompts. Indonesian's low LPR is largely fastText labelling it as Malay, not English fallback.

### Test (197 prompts per language)

LPR, with the gained / lost prompts against the instruction and the McNemar p in brackets:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction | .86 | .83 | .94 | .92 | .93 | .97 |
| Steering | .94 (+26/−9, p = .006) | .90 (+23/−9, p = .02) | **.99** (+10/−1, p = .01) | **1.00** (+15/−0, p < .001) | **1.00** (+13/−0, p < .001) | .96 (+2/−4, p = .69) |
| Instruction + steering | **.97** (+28/−5, p < .001) | **.91** (+19/−3, p = .001) | .98 (+11/−3, p = .06) | **1.00** (+16/−0, p < .001) | .99 (+12/−0, p < .001) | **.98** (+3/−1, p = .63) |
| *chosen l, α: steering* | 8, 1 | 24, 1 | 12, 1 | 12, 1 | 12, 1 | 8, 1 |
| *chosen l, α: instr. + steering* | 16, 1 | 20, 0.5 | 12, 2 | 16, 1 | 8, 1 | 24, 1 |

The other measures, given as instruction → steering → instruction + steering:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| LPR, complex prompts | .77 → .84 → .97 | .69 → .77 → .81 | .87 → 1.00 → .97 | .78 → 1.00 → 1.00 | .85 → 1.00 → .98 | .92 → .92 → .95 |
| English | .09 → .00 → .00 | .06 → .01 → .03 | .05 → .00 → .00 | .06 → .00 → .00 | .05 → .00 → .00 | .02 → .01 → .01 |
| WPR | .96 → .92 → .95 | .95 → .89 → .93 | .95 → .92 → .95 | .97 → .98 → .98 | .91 → .90 → .91 | – |
| Degenerate | .01 → .03 → .01 | .00 → .00 → .01 | .01 → .04 → .04 | .00 → .02 → .00 | .03 → .03 → .03 | .01 → .01 → .03 |

1. **Steering fixes the instruction's failures.** Steering alone is significant in all five languages except tr, and instruction + steering in four (ru: p = .06). English fallbacks drop from 5–9% to 0–3%, and the largest gains are on the long complex prompts (hi .78 → 1.00, ja .77 → .97).
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
| Steering (12, 1) | .95 (+27/−7, p = .001) | .90 (+21/−7, p = .01) | .99 (+9/−1, p = .02) | 1.00 (+17/−0, p < .001) | .99 (+14/−1, p = .001) | .96 (+2/−4, p = .69) |
| Instruction + steering (24, 1) | .95 (+26/−7, p = .001) | .92 (+24/−5, p = .001) | 1.00 (+10/−0, p = .002) | 1.00 (+16/−0, p < .001) | 1.00 (+13/−0, p < .001) | .98 (+3/−1, p = .63) |

* **Per-language tuning buys almost nothing.** Every LPR is within 0.02 of the per-language setting, and the same comparisons are significant. The dev grids are flat: at α = 1, layers 8–16 are within about 2 dev prompts of the best in nearly every language.
* **Steering at (12, 1) is the safer default.** Instruction + steering at layer 24 keeps LPR at 1.00 in hi and ar, but degeneration rises to .06–.07 and WPR drops (hi .98 → .87, ar .91 → .86).

### Prompt-only steering (test)

The vector is added only from the start of the user message to the end of the prompt (`--prompt-only`). The first generated token is still predicted from a steered position, but later tokens are not steered. Everything else is the same pipeline (dev grid, per-language and shared selection, test), with α extended to {…, 4, 6}. That job ran on H100s, with its own instruction run.

LPR, all-token → prompt-only; brackets give the prompt-only McNemar p against the instruction when it is ≥ .05:

| | ja | zh | ru | hi | ar | tr |
|---|---|---|---|---|---|---|
| Instruction (prompt-only job) | .85 | .83 | .94 | .91 | .93 | .97 |
| Steering | .94 → **1.00** | .90 → .88 (p = .21) | .99 → .99 (p = .06) | 1.00 → .98 | 1.00 → .94 (p = .61) | .96 → .92 (worse, p = .007) |
| Instruction + steering | .97 → .97 | .91 → .89 (p = .10) | .98 → .98 | 1.00 → .99 | .99 → .97 | .98 → .95 (p = .27) |
| *chosen l, α (prompt-only): steering* | 12, 3 | 12, 3 | 12, 2 | 12, 2 | 16, 2 | 8, 2 |
| *chosen l, α (prompt-only): instr. + steering* | 12, 2 | 16, 3 | 20, 3 | 16, 2 | 24, 4 | 8, 3 |

* **Once an answer starts in T, it stays there.** Drift is 0–4% for prompt-only and 0–2% for all-token steering, so steering the generated tokens is not what keeps 100-token answers in T.
* **A single push needs to be stronger, and the working range is narrower.** With steering alone at α = 1, dev LPR is only 0.80–0.96. α ≥ 4 degenerates, and α = 6 breaks every language.
* **The shared setting fails for steering alone.** Larger α breaks the degenerate limit in some language, so the rule picks l = 8, α = 1. That setting gives no significant gain on test and lowers hi (.89) and tr (.91). The shared instruction + steering setting (l = 16, α = 2) gives .90–.99, significant in ja, zh, hi and ar.
* **Less code-mixing only in ja and zh:** WPR under steering alone rises from .92 to .96 (ja) and from .89 to .95 (zh); it does not change in ru, hi or ar, and degeneration is similar.

All-token steering is therefore the more robust choice on LCB. Prompt-only matches it only with per-language tuning, and is worse in ar and tr.

**Hand check:** we read 20 test prompts (ja, zh, ru, hi, tr), all three arms: 2 random prompts per language, 1 where steering alone gained and 1 where it lost.
* On the random prompts, all arms were fluent and on topic, and wrong facts appeared equally often in every arm.
* Of the 5 gains, 3 were real: romanized Russian and Hindi became native script, and an English JSON analysis became Turkish. 2 were scorer artifacts: song titles translated, and a truncated list line.
* Of the 5 losses, 1 was an artifact (all-Chinese JSON). The others were mild code-mixing, a short loop, and one partly English Turkish answer.
* Steering-specific problems: code-mixing inside words ("Machine学习", "Quantум", Cyrillic "крок" in a Chinese story), more spelling errors in Turkish, and occasional drift in meaning (a Hindi metaphor).

**Caveats:**
* LCB measures only the language, not quality or correctness.
* The main settings are tuned per language on dev (a shared setting does as well, above).
* fastText is unreliable on code, JSON and quoted English names.
* One model.

**Deviations from LCB:** our dev/test split (LCB has none); 6 of its 14 languages; LCPR (the harmonic mean of LPR and WPR) is not reported.

**Next:** measure whether steering preserves correctness on a task with checkable answers, against both the instruction and a native-language baseline.
