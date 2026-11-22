# Cross-lingual instruction vs language steering on MMLU-ProX

The second benchmark of `../mgsm_crosslingual/`, outside maths: does the instruction fail to set the answer language there too, and does the language vector, tuned on MGSM, fix it? The question and its ten options are in a source language S, with zero-shot chain of thought; the answer is wanted in a target language T. *Success*, as on MGSM: the answer counts only if the letter is correct and the response is in T (the MGSM fastText judge). All 11 MGSM languages are both sources and targets (110 ordered pairs): the 11 tuning pairs of `../mgsm_crosslingual/` and the 99 held-out ones. Nothing is tuned on MMLU-ProX: the layer and α of our steering and ReCoVeR's α are those chosen on MGSM dev.

**Result: the instruction fails outside maths as well, and prompt-only steering closes 69–81% of the gap to the native question in Llama-3.1-8B, Qwen3-8B and Qwen3-32B. It beats ReCoVeR there, by +3.4 to +7.0 points in the 8B models and by about +1 in Qwen3-32B.** In Gemma-3-12B the instruction already mostly works (0.84–0.91 in T). Steering then adds little (+1.2 / +2.0 points), and ReCoVeR is slightly better (−0.6 / −1.9), as on MGSM, where the two tied.

Mean success over the 99 held-out pairs [95% CI], with the gap to native closed in brackets:

| | Llama-3.1-8B | Qwen3-8B | Qwen3-32B | Gemma-3-12B |
|---|---|---|---|---|
| **STEM (no maths)** | | | | |
| Native question (in T), the ceiling † | 0.287 | 0.559 | 0.701 | 0.512 |
| Instruction | 0.099 | 0.032 | 0.498 | 0.461 |
| Instruction in English | 0.146 (25%) | 0.030 (0%) | 0.539 (20%) | 0.462 (3%) |
| **Instruction + prompt-only steering** | **0.229** [.199, .261] (69%) | **0.457** [.420, .496] (81%) | **0.644** [.601, .684] (72%) | 0.473 [.427, .522] (23%) |
| Baseline: ReCoVeR (all blocks, all tokens) | 0.192 [.167, .219] (50%) | 0.387 [.356, .419] (67%) | 0.633 [.590, .672] (66%) | **0.492** [.446, .541] (61%) |
| **Humanities, social sciences, other** | | | | |
| Native question (in T), the ceiling † | 0.232 | 0.405 | 0.560 | 0.471 |
| Instruction | 0.087 | 0.031 | 0.376 | 0.420 |
| Instruction in English | 0.119 (22%) | 0.022 (−3%) | 0.412 (20%) | 0.427 (13%) |
| **Instruction + prompt-only steering** | **0.198** [.174, .225] (77%) | **0.328** [.293, .366] (79%) | **0.506** [.461, .552] (70%) | 0.440 [.395, .490] (39%) |
| Baseline: ReCoVeR (all blocks, all tokens) | 0.164 [.144, .186] (53%) | 0.281 [.251, .312] (67%) | 0.497 [.452, .542] (65%) | **0.446** [.402, .496] (51%) |

† The native question is one unit per language (S = T); its mean over the held-out pairs is the mean over the 11 target languages.

**Scoring:** Llama-3.1-8B is scored on its free-form answers. The other three models are asked to end with `\boxed{X}` and are scored by the box (see *Answer format*). Gap closed = (arm − instruction) / (native − instruction). For Gemma, whose instruction gap is only .05, these percentages are noisy: the CI of ours runs from 10% to 38% in STEM and from 27% to 57% in humanities.

Paired tests on success over the 99 held-out pairs: the mean difference [95% CI]; then the number of pairs significantly better / worse (McNemar p < .05, Holm over the 99 pairs):

| | Llama-3.1-8B | Qwen3-8B | Qwen3-32B | Gemma-3-12B |
|---|---|---|---|---|
| **STEM** | | | | |
| Ours − instruction | +.130 [+.112, +.149]; 63 / 0 | +.426 [+.390, +.461]; 91 / 0 | +.146 [+.132, +.160]; 35 / 0 | +.012 [+.005, +.019]; 8 / 3 |
| ReCoVeR − instruction | +.093 [+.079, +.108]; 55 / 7 | +.355 [+.326, +.385]; 73 / 0 | +.135 [+.121, +.148]; 31 / 0 | +.031 [+.024, +.039]; 9 / 0 |
| Ours − ReCoVeR | +.037 [+.028, +.046]; 7 / 1 | +.070 [+.061, +.080]; 27 / 9 | +.011 [+.003, +.019]; 6 / 2 | −.019 [−.026, −.013]; 2 / 5 |
| **Humanities etc.** | | | | |
| Ours − instruction | +.112 [+.097, +.127]; 59 / 1 | +.297 [+.265, +.330]; 90 / 0 | +.130 [+.115, +.145]; 36 / 0 | +.020 [+.014, +.026]; 6 / 2 |
| ReCoVeR − instruction | +.077 [+.066, +.089]; 49 / 4 | +.249 [+.223, +.277]; 70 / 0 | +.120 [+.106, +.134]; 34 / 0 | +.026 [+.019, +.033]; 5 / 1 |
| Ours − ReCoVeR | +.034 [+.025, +.044]; 5 / 1 | +.047 [+.038, +.056]; 25 / 5 | +.009 [+.003, +.016]; 5 / 0 | −.006 [−.012, +.000]; 4 / 4 |

* **Steering never loses to the instruction in the three models where the instruction fails.** After Holm, steering is better in 35–91 held-out pairs per subset and worse in at most 1. In Gemma it is better in 6–8 pairs and worse in 2–3, the same pairs where it loses to ReCoVeR (below).
* **Against ReCoVeR, the ranking follows MGSM's.** On MGSM, ReCoVeR minus prompt-only steering was −.067 (Llama), −.051 (Qwen3-8B), −.004 (Qwen3-32B) and −.005 (Gemma). Here ours is clearly ahead in the 8B models, slightly ahead in Qwen3-32B, and slightly behind in Gemma.
* **The English instruction is no fix:** it closes at most a quarter of the gap in any model.

## Setup

* **Data:** the MMLU-ProX test set (`li-lab/MMLU-ProX`, Xuan et al. 2025, MIT; revision in `data/raw_data/manifest.json`), the same questions in all 11 languages. Two subsets of 200 questions each:
  * **STEM:** physics, chemistry, engineering, biology and computer science, 40 each. Maths is left out, to stay away from MGSM.
  * **Humanities etc.:** law, economics, psychology, business, health, philosophy, history and other, 25 each.

  Per category the questions are sorted by crc32 of their id and the first ones taken (no RNG).
* **Prompt:** the question, the options lettered A, B, …, then "Let's think step by step." written in S, as MMLU-ProX writes it in its own worked examples. For Qwen3-8B, Qwen3-32B and Gemma-3-12B, one more sentence in S follows it: "Finish your answer with \boxed{X}, where X is the letter of the correct option." (`--format boxed`, in every arm).
* **Arms** (`pilot.py`):
  * **native** (`cot`): the question in T, no instruction.
  * **instruction:** "Answer in <T>." written in S, as on MGSM.
  * **instruction in English:** the same instruction written in English.
  * **instruction + prompt-only steering (ours):** the instruction prompt plus α · v_(S→T) at layer l on the prompt tokens only. The FLORES vectors and (l, α) come from MGSM dev: Llama (16, 1), Qwen3-8B (28, 1), Qwen3-32B (40, 1), Gemma-3-12B (24, 1).
  * **ReCoVeR** (Sterz et al. 2025, as published): the instruction prompt; α (r_T − r_S) / |·| added at every block and every token, with the norm restore, at the α chosen on MGSM dev: Llama 0.1, Qwen3-8B 0.5, Qwen3-32B 1, Gemma-3-12B 16.
* **Decoding:** greedy, in two stages:
  * **Stage 1:** at most 2048 new tokens.
  * **Stage 2:** every answer capped in stage 1 is continued from its own tokens to 4096 in all, the capped answers of all pairs pooled into full batches.

  Most capped answers are repetition loops, so pooling saves most of the decode time. Bengali and Telugu answers that finish are about 2.5 times as long in Llama tokens as the others, and the 4096 cap leaves them room. An answer still capped at 4096 counts as wrong.
* **Scoring:**
  * **Free-form (Llama; `letter` in `pilot.py`):** the letter comes from a `\boxed{}` if there is one. Otherwise it is taken from right after the last answer cue ("the answer is" and its translations), else from an option's text after that cue, else, when there is no cue, from the last paragraph. An answer that names several letters, or none, counts as wrong.
  * **Boxed (the other three models):** the letter of the last `\boxed{X}`, and nothing else; no box counts as wrong. A box whose letter is written in T's script (`\boxed{Г}`) is read by a fixed transliteration table (`TRANSLIT`), but only when the answer has no Latin box. See *Answer format* for the table.
  * **Language:** the MGSM fastText judge, run on the response with its LaTeX maths removed (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`). Left in, command names and units such as `\times`, `\frac` and `\text{J mol}` read as English lines, and the judge then rejects STEM answers heavy in maths. For Qwen3-8B's native question in STEM (free-form) that gave 0.830 in T instead of 0.950; for Llama, which writes less LaTeX, 0.937 instead of 0.943. On MGSM the same change moves the in-T rates by at most .007 (a sample of 20 test pairs per arm and model), so the MGSM tables stand. Judged with the options quoted in S removed as well, the in-T rates change by at most .017, so the tables do not remove them.
* **Statistics** (`tests.py`), as on MGSM:
  * an exact McNemar test per held-out pair on per-question success, with Holm over the 99 pairs;
  * 95% percentile bootstrap CIs (2000 resamples over the 200 questions of a subset, seed 0, the same resampled questions for every pair and arm) for the means, the differences and the gap closed.

## Llama-3.1-8B in detail (free-form)

Held-out means (native: the 11 languages).

| | STEM: success / accuracy / in T / capped at 4096 | Humanities etc.: success / accuracy / in T / capped at 4096 |
|---|---|---|
| Native † | 0.287 / 0.305 / 0.94 / 0.23 | 0.232 / 0.233 / 0.99 / 0.07 |
| Instruction | 0.099 / 0.272 / 0.34 / 0.24 | 0.087 / 0.217 / 0.35 / 0.11 |
| Instruction in English | 0.146 / 0.282 / 0.52 / 0.24 | 0.119 / 0.225 / 0.50 / 0.11 |
| **Instruction + prompt-only steering** | **0.229** / 0.248 / 0.90 / 0.28 | **0.198** / 0.205 / 0.95 / 0.15 |
| ReCoVeR | 0.192 / 0.201 / 0.91 / 0.37 | 0.164 / 0.167 / 0.94 / 0.24 |

* **The instruction fails as on MGSM.**
  * With an English source it mostly works: 0.71–0.98 in T.
  * Otherwise the answer tends to stay in S: bn→X is at most 0.17 in T, and X→en at most 0.04, except zh→en in STEM (0.30).
  * Its accuracy is intact (0.272 against native's 0.305 in STEM), so the loss is almost all language.
* **Steering fixes the language at a small cost in accuracy:** −.024 (STEM) and −.012 (humanities). It also caps a few more answers than the instruction (+.04).
* **ReCoVeR fixes the language about as well, but loses more accuracy:** −.071 and −.050. It also caps the most answers: 0.37 in STEM and 0.24 in humanities.
* **The English instruction helps only half the time.**
* **The tuning pairs agree with the held-out ones:** steering 0.228 / 0.206 against 0.229 / 0.198. The MGSM settings carry over without overfitting the tuning pairs.
* **Answers without a letter:** in 7–12% (STEM) and 15–21% (humanities) of all answers, no letter is found although the answer finished. The rate is similar in every arm (STEM .072–.119, humanities .153–.211; ours and ReCoVeR slightly higher than the instruction, native highest in humanities), so it lowers absolute accuracy more than it changes the comparison.
* About a fifth of STEM answers stay capped at 4096 even for the native question (0.23; humanities 0.07), mostly repetition loops.

## Qwen3-8B in detail (boxed)

Thinking off, as on MGSM (no `<think>` blocks in any answer). Held-out means:

| | STEM: success / accuracy / in T / capped at 4096 | Humanities etc.: success / accuracy / in T / capped at 4096 |
|---|---|---|
| Native † | 0.559 / 0.575 / 0.96 / 0.12 | 0.405 / 0.410 / 0.98 / 0.08 |
| Instruction | 0.032 / 0.567 / 0.05 / 0.11 | 0.031 / 0.413 / 0.06 / 0.08 |
| Instruction in English | 0.030 / 0.583 / 0.05 / 0.09 | 0.022 / 0.424 / 0.05 / 0.06 |
| **Instruction + prompt-only steering** | **0.457** / 0.548 / 0.83 / 0.15 | **0.328** / 0.387 / 0.85 / 0.13 |
| ReCoVeR | 0.387 / 0.541 / 0.68 / 0.14 | 0.281 / 0.383 / 0.67 / 0.13 |

* **The instruction fails almost completely:** 5–6% of its answers are in T, and the answers stay in S. Accuracy is untouched (0.567 against native's 0.575).
* **Steering fixes most of the language at a small cost in accuracy:** −.019 (STEM) and −.026 (humanities).
* **ReCoVeR loses about as much accuracy but fixes less of the language:** 0.67–0.68 in T, against 0.83–0.85.
* **The box sentence changes the instruction arm, not the steered ones.** Under the free-form prompt the instruction gave 0.23 (STEM) and 0.30 (humanities) in T; with the box sentence it gives 0.05 and 0.06. The steered arms barely move: ours 0.450 / 0.315 free-form against 0.457 / 0.328 boxed, ReCoVeR 0.388 / 0.265 against 0.387 / 0.281. So the gap closed is higher under the boxed prompt (81% / 79%, against 78% / 76% free-form), but ours − ReCoVeR is the same under both: +.062 / +.050 free-form, +.070 / +.047 boxed. The free-form run is in `results/qwen3-8b/pilot_4096_from2048_all.json`.
* **The tuning pairs agree with the held-out ones:** steering 0.456 / 0.313 against 0.457 / 0.328.
* **Where ReCoVeR wins (after Holm):** 9 pairs in STEM and 5 in humanities, mostly into Japanese (en, de, es, fr, th → ja) and Swahili into German or Spanish. Ours wins in 27 and 25 pairs.

## Qwen3-32B in detail (boxed)

Thinking off. Held-out means:

| | STEM: success / accuracy / in T / capped at 4096 | Humanities etc.: success / accuracy / in T / capped at 4096 |
|---|---|---|
| Native † | 0.701 / 0.716 / 0.97 / 0.05 | 0.560 / 0.567 / 0.98 / 0.02 |
| Instruction | 0.498 / 0.705 / 0.69 / 0.05 | 0.376 / 0.549 / 0.66 / 0.03 |
| Instruction in English | 0.539 / 0.704 / 0.76 / 0.05 | 0.412 / 0.549 / 0.75 / 0.04 |
| **Instruction + prompt-only steering** | **0.644** / 0.687 / 0.93 / 0.05 | **0.506** / 0.536 / 0.93 / 0.04 |
| ReCoVeR | 0.633 / 0.698 / 0.89 / 0.05 | 0.497 / 0.544 / 0.89 / 0.04 |

* **The instruction fails in a third of the answers** (0.66–0.69 in T), with accuracy close to native's. As on MGSM, the larger model fails less often than the 8B models, but it still fails.
* **Steering raises the in-T rate to 0.93 at a small cost in accuracy** (−.018 and −.013). ReCoVeR keeps a little more accuracy and fixes a little less of the language (0.89). Net: ours +.011 / +.009, both CIs above 0, with 6 / 5 pairs better and 2 / 0 worse after Holm.
* **The tuning pairs:** steering 0.663 / 0.497 against 0.644 / 0.506 held-out.
* Few answers are capped (about 0.05 or fewer) or unscored (≤ 0.6%).

## Gemma-3-12B in detail (boxed)

Held-out means:

| | STEM: success / accuracy / in T / capped at 4096 | Humanities etc.: success / accuracy / in T / capped at 4096 |
|---|---|---|
| Native † | 0.512 / 0.547 / 0.92 / 0.03 | 0.471 / 0.474 / 0.99 / 0.01 |
| Instruction | 0.461 / 0.529 / 0.84 / 0.02 | 0.420 / 0.456 / 0.91 / 0.01 |
| Instruction in English | 0.462 / 0.531 / 0.85 / 0.02 | 0.427 / 0.459 / 0.93 / 0.01 |
| Instruction + prompt-only steering | 0.473 / 0.510 / 0.90 / 0.02 | 0.440 / 0.450 / 0.98 / 0.01 |
| **ReCoVeR** | **0.492** / 0.522 / 0.92 / 0.02 | **0.446** / 0.453 / 0.98 / 0.01 |

* **The instruction mostly works:** 0.84 (STEM) and 0.91 (humanities) in T. The gap to native is only .05, as on MGSM, where Gemma's instruction failed least of the four models.
* **Both steering methods close part of that small gap.** Steering is better than the instruction in 8 / 6 pairs after Holm and worse in 3 / 2. ReCoVeR is better in 9 / 5 and worse in 0 / 1.
* **ReCoVeR's lead over ours comes from five pairs** (ours − ReCoVeR in success points, the mean of the two subsets):
  * **Into English (te→en −21.0, sw→en −19.3, bn→en −9.0): the box is dropped.** Our arm leaves out the box in 44%, 44% and 30% of these answers, against 3–8% for ReCoVeR and 11–37% for the instruction. Its answers are in English, and its accuracy among the boxed answers is the same as ReCoVeR's. The loss comes from the format, not from the reasoning: the steered prompt makes Gemma end with a restatement instead of the box.
  * **Into German (th→de −17.5, te→de −15.0): the answer drifts into English.** Only 0.63 and 0.61 of our answers are in T, against 0.97 for ReCoVeR. Prompt-only steering cannot correct a drift during generation; ReCoVeR, which also steers the generated tokens, can.

  Requests for the question (the model asking for it instead of answering) are equally rare in every arm. Without these five pairs, ours − ReCoVeR is −.010 (STEM) and +.000 (humanities).
* **The tuning pairs:** steering 0.492 / 0.458 against 0.473 / 0.440 held-out.

## Answer format and scoring (appendix material)

**Why the scoring differs between models.** The free-form parser misses some answers: for Gemma 13–23% (it ends by restating the options, which the parser cannot read), for Qwen3-32B a share 3.5 points higher in our arm than in ReCoVeR's (humanities), as large as their difference. Asking for `\boxed{X}` removes that, but only for a model that keeps the format in every arm. A pilot (`boxed_pilot.sh`: the first 3 questions per category, native in every language and the other arms on the 11 tuning pairs, 429 answers per arm) checked this before the full runs. Share of finished answers with a box, STEM / humanities:

| | Native | Instruction | Instruction in English | Ours | ReCoVeR |
|---|---|---|---|---|---|
| Llama-3.1-8B | .83 / .86 | .83 / .82 | .81 / .84 | .66 / .69 | .69 / .76 |
| Qwen3-8B | 1.00 / 1.00 | .99 / 1.00 | 1.00 / 1.00 | .99 / 1.00 | .99 / .99 |
| Gemma-3-12B | .97 / .98 | .93 / .97 | .96 / .99 | .93 / .98 | .93 / .94 |

(Qwen3-32B did not finish in the pilot; in its full run 96–100% of finished answers are boxed in every arm.)

**Llama is scored free-form because of this pilot, not because of its outcome.** In Llama the steered arms drop the box about twice as often as the unsteered ones: 31–34% of finished answers in our arm and 24–31% in ReCoVeR's, against 14–19% in the native and instruction arms. Scoring by the box alone would then count a format failure that steering causes as a wrong answer, and so bias the comparison against both steering arms. Its free-form parser misses a similar share in every arm (above). For the other three models the box is kept in 94–100% of the answers of every arm, so the box is the cleaner score. The decision was made on the pilot, before the full boxed runs. Llama was not rerun boxed.

**Boxes in T's script.** Some answers write the letter in T's script: `\boxed{Г}`, `\boxed{গ}`. When an answer has no Latin box, the letter is read through a fixed table (`TRANSLIT` in `pilot.py`):
* Cyrillic by ISO 9 (А Б Г Д Е И Й Ф Х → A B G D E I J F H), and В Ж З К by their position in the alphabet (→ C G H J);
* Bengali গ জ হ ই → G J H I, Telugu ఐ → I.

Each mapping was checked against the gold answers of the questions it occurs in. The table was fixed before the significance tests. Shares of answers read this way, ours / ReCoVeR: Qwen3-32B 2.0% / 3.4% (humanities) and 1.3% / 2.3% (STEM); Gemma ≤ 0.1% / ≤ 1.5%; Qwen3-8B ≤ 0.1% / ≤ 0.8%. A box in another script that the table cannot read: at most 0.2% of answers in any arm. An earlier attempt to read such letters through the option text resolved only 14–16% of them and was dropped.

**Unscored answers (boxed models).** Finished answers without a box: Qwen3-8B and Qwen3-32B ≤ 1.0% in every arm; Gemma 2.0–3.7%, except our arm in STEM (5.9%; the pairs into English above).

## Not run

* **The instruction in English + prompt-only steering:** dropped (2026-11-19); the bare English instruction stays as a prompt baseline.
* **The random-direction control:** dropped. On MGSM it was at or below the instruction in all four models (`../mgsm_crosslingual/README.md`), so the gain comes from the language direction; no need to rerun it on every benchmark.
* **The instruction written in T:** not used.
* **Llama-3.1-8B with the box sentence:** not run (see *Answer format*).
* **No chain of thought:** checked and not used. On 52 questions (Llama, native and the 11 tuning pairs), the answers without "Let's think step by step." are only about a third shorter: median 342 and 398 tokens against 590 and 604 with it, and 10% and 17% still capped at 2048. With the instruction, 6% of them are under 20 tokens, mostly a bare letter that the language judge can't place.

## Reproduce

```bash
# Llama-3.1-8B, free-form (its shared Hugging Face snapshot is incomplete: a local cache)
slurm/submit.sh -J mmlu-prox-allpairs -t 40:00:00 --gres gpu:4 --gres-flags=disable-binding -c 16 --mem 128G \
  --constraint GPU_BRD:B300 -- bash src/experiments/mmlu_prox/pilot.sh llama-3.1-8b $PWD/logs/hf_cache \
  --max-new 4096 --first-cap 2048 --pairs all
# the other three models, boxed, one after another (the analysis and the tests after each)
slurm/submit.sh -J mmlu-prox-models -t 72:00:00 --gres gpu:4 --gres-flags=disable-binding -c 16 --mem 128G \
  --constraint GPU_BRD:B300 -- bash src/experiments/mmlu_prox/models.sh
# optional helpers, joining the tasks of the job above: the same with HELPER=1 (no analysis), e.g.
slurm/submit.sh -J mmlu-prox-models-helper -t 72:00:00 --gres gpu:2 --gres-flags=disable-binding -c 8 --mem 128G \
  --constraint GPU_BRD:B300 -- env HELPER=1 bash src/experiments/mmlu_prox/models.sh
# tests (no GPU; Llama without --format boxed)
XL_MODEL=<model> python src/experiments/mmlu_prox/tests.py --format boxed --max-new 4096 --first-cap 2048
# the format pilot
slurm/submit.sh -J boxed-pilot -t 06:00:00 --gres gpu:4 --gres-flags=disable-binding -c 16 --mem 128G \
  --constraint GPU_BRD:B300 -- bash src/experiments/mmlu_prox/boxed_pilot.sh
```

How the runs actually went:
* **Llama** took a few restarts on its way to these settings. Some stage-2 continuations ran to 8192 tokens before the cap was set to 4096. The analysis cuts them back to 4096 and marks them capped: with greedy decoding, their first 4096 tokens are those of a run to 4096. Its output directory is `generations/pilot_8192_from2048`, linked as `pilot_4096_from2048`.
* **The other three models** were first run free-form (`models.sh` at the time). Each was then rerun boxed:
  * Gemma: in the same jobs, plus helpers;
  * Qwen3-32B and Qwen3-8B: through a queue job (`slurm/queue.sh`, `enqueue.sh`) and a helper.

  The free-form results of Qwen3-32B and Gemma are not used; Qwen3-8B's are kept for the prompt comparison above.
* **Cost** of the Qwen3-8B boxed run: about 21 B200-hours and 11 B300-hours.

## Files

| File | What |
|---|---|
| `pilot.py` | prompts, arms, generation (shared claims, two stages), scoring and the per-pair means |
| `tests.py` | McNemar with Holm, bootstrap CIs, gap closed |
| `pilot.sh`, `models.sh`, `boxed_pilot.sh` | Slurm drivers (above) |
| `results/<model>/pilot_4096_from2048[_boxed]_all.json` | per subset, arm and pair (and tuning / held-out / all): success, accuracy, in T, capped, unscored, box shares, median length |
| `results/<model>/pilot_4096_from2048[_boxed]_all_tests.json` | the tests and CIs above, with the per-pair McNemar counts |
| `results/<model>/pilot_4096_from2048_boxed_first3.json` | the format pilot (Llama, Qwen3-8B, Gemma) |
| `results/qwen3-8b/pilot_4096_from2048_all.json` | Qwen3-8B free-form (the prompt comparison) |

Not committed (regenerable with `pilot.py analyse`): the generations (`results/<model>/generations/`), the per-question scores that `tests.py` reads (`*_items.json`), and the superseded free-form results of Qwen3-32B and Gemma.
