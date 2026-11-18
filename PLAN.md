# Plan: paper and project report

The original proposal is in [README.md](README.md); the experiment plans written for it are in [docs/plans/](docs/plans/). This file records where the project now stands, what the paper and the report will say, and what is left to do.

## Central claim

> Instruction prompts often fail to control the output language across languages; with Llama-3.1-8B-Instruct, "answer in T" fails almost completely (success < 0.10) in 40 of 99 held-out source-target pairs on MGSM. A single training-free language vector, added to the prompt, fixes this (1 of 99 pairs fail) without changing which problems the model can solve.

Everything else supports, delimits or motivates this claim:

* **Where steering helps:** wherever the instruction fails (LCB cross-lingual; MGSM with a non-English question).
* **Where it does not:** where the instruction already works (MKQA, MGSM asked in English) and in multilingual ICL (no headroom on math; on XNLI the gain is not language-specific).
* **Why the per-token method of the proposal was dropped:** the language decomposition holds for sentence averages, not for single tokens.
* **What steering changes:** the start of the answer (its language), not the reasoning (κ, teacher-forced KL, forced starts), through the choice of language made in the last layers (logit lens).

## Paper vs report

| | Paper | Project report |
|---|---|---|
| Scope | The central claim, defended against every obvious objection | The whole project, from the proposal to the final result |
| Hypothesis tests (T1, W1-W2) | One paragraph: why a constant vector, not a per-token scale | Full section: hypotheses, tests, negative result, consequences |
| ICL (Kirtane & Huang replication) | Short motivation section, or appendix | Full section, with the stronger English baseline and the controls |
| MKQA, same-language MGSM | One sentence each: the boundary of where steering helps | Short sections |
| LCB | Main result 1 | Full section |
| Cross-lingual MGSM | Main result 2, with the mechanism (κ, KL) | Full section |
| Per-language tables | Appendix | Main text or appendix |

### Paper outline (working)

1. Introduction: output-language control, why instructions fail, the claim.
2. Related work: language confusion and its mitigation, language-specific representations and neurons, activation steering, multilingual reasoning.
3. Method: language vectors from FLORES-200 means, prompt-only vs all-token steering, selection on dev pairs only.
4. Setup: models, benchmarks (LCB, MGSM in 110 pairs), success = correct and in T, judge and its checks.
5. Results: instruction vs steering vs native; failure counts; controls; prompt baselines; second model.
6. Analysis: what steering changes (κ, KL by position), by resource level, Japanese.
7. Where steering does not help: MKQA, MGSM from English, multilingual ICL; the token-level geometry.
8. Limitations, conclusion.

### Report outline (working)

1. Introduction and motivation (as in the proposal).
2. Background: representation geometry, steering, Kirtane & Huang (2026).
3. The proposed method and its hypotheses (from the proposal).
4. Testing the hypotheses (T1, W1-W2): sentence level holds, token level does not.
5. Language steering for multilingual ICL: protocol, stronger baseline, results, controls.
6. The pivot: output-language control (MKQA pilot, MGSM from English).
7. Language Confusion Benchmark.
8. Cross-lingual MGSM: results, mechanism, resource patterns.
9. Discussion, limitations, future work.

## Results already in hand (Llama-3.1-8B-Instruct)

| Result | Where | In paper | In report |
|---|---|---|---|
| T1: language subspace at sentence level | `src/experiments/pca_analysis/` | 1 paragraph | yes |
| W1-W2: no per-token structure | `src/experiments/word_level_analysis/` | 1 paragraph | yes |
| ICL: no headroom on math, non-specific gain on XNLI | `src/icl/RESULTS.md` | motivation / appendix | yes |
| MKQA pilot: instruction already works | `src/experiments/output_language/` | 1 sentence | yes |
| MGSM from English: instruction already works | `src/experiments/mgsm_steering/` | 1 sentence | yes |
| LCB: steering fixes language confusion | `src/experiments/language_confusion/` | main | yes |
| Cross-lingual MGSM: success, failure counts, κ, KL, by resource | `src/experiments/mgsm_crosslingual/` | main | yes |

## To do

Priority: **P1** needed for a credible paper, **P2** strengthens it, **P3** only if time allows.

### Experiments

- [x] **P1 Second model on cross-lingual MGSM:** Qwen3-8B (thinking off), and Qwen3-32B for scale. The model is a parameter (`XL_MODEL`); each model has its own layer grid (about 25-75% of its depth); same items, tuning pairs, α grid and judge; all analyses rerun. Results in `src/experiments/mgsm_crosslingual/README.md`. The sink token is now left out of the vector means for every model.
- [x] **P1 Second model on LCB:** Qwen3-8B and Qwen3-32B (thinking off), with the same settings and each model's MGSM layer grid; one job per model (`run.sh`). Results in `src/experiments/language_confusion/README.md`.
- [x] **P1 Steering controls in cross-lingual MGSM**, at the chosen setting of instruction + prompt-only steering, on all test pairs: a norm-matched random direction and a wrong-target vector v(S→U), U ≠ S, T, each drawn per item.
- [x] **P1 Stronger prompt baselines.** The instruction in English and as a system message (the dev winner goes to test), and instruction in English + steering. On Qwen3-8B only: the instruction written in T (0.451 held-out success; better in 45 pairs, worse in 26, depending on T) and the instruction before the question (0.420), against 0.412 for the instruction and 0.711 with steering; not extended to the other models, as neither changes the conclusion. For the paper: one sentence in the main body (or rows marked "Qwen3-8B only" in the prompt-baselines table), details in the appendix. Dropped: demonstrations answered in T (they assume the target-language data the method is meant to do without); translate-then-ask (NLLB), which changes the task and is close to the native ceiling.
- [x] **P2 Sensitivity:** success over the layer × α dev grid, all four models, MGSM and LCB (`src/experiments/mgsm_crosslingual/sensitivity.py`). α ≈ 1 works across models and both tasks; every tested layer (about 25–75% of the depth) helps, the worst keeping 57–79% of the chosen gain on MGSM, and on LCB all layers are within .03; α ≥ 2 degrades or degenerates; steering without the instruction is narrower; Gemma's MGSM grid is flat, as its instruction works on the tuning pairs. Results in `src/experiments/mgsm_crosslingual/README.md`; the heatmap figure is part of the P1 Figures item.
- [x] **P2 Number of FLORES sentences for the vectors:** instruction + prompt-only steering on test with vectors from 1, 3, 10 or 100 of the 997 FLORES sentences (matched in S and T, not normalised, three draws each, the chosen layer and α not retuned), all four models (`src/experiments/mgsm_crosslingual/flores_subsets.py`). Ten sentences give the same result as 997 (means within .002, at most one pair per draw worse after Holm); one sentence keeps almost all of the gain (−.004 to −.013), but 1–3% of (pair, draw) are then more than .10 worse, which is gone by n = 10. Results in `src/experiments/mgsm_crosslingual/README.md`.
- [x] **P2 Interpretability:** a forced-start experiment (Llama) and a logit lens at the last prompt position (all four models), for the instruction and instruction + prompt-only steering. With the instruction, the first answer token largely sets the language; the language is chosen in the last 3–4 layers, in Llama and Qwen3-8B after a stage where English leads (in Qwen3-32B only briefly, in Gemma-3-12B strongest, with the choice spread over its last 8 layers); in the failing pairs the instruction lets S win there, steering lets T win. Results in `src/experiments/mgsm_crosslingual/README.md`.
- [x] **P3 Third model family:** Gemma-3-12B-it on cross-lingual MGSM (all arms except the prompt variants and the forced start) and on LCB, with its full-attention blocks 12–36 as the layer grid. On MGSM the instruction fails less than in the 8B models (0.729 held-out, 4 pairs below 0.30, from English and Swahili questions); instruction + prompt-only steering gives 0.777, better in 15 held-out pairs and worse in 5 (11 and 1 after Holm), with no pair below 0.30, and both controls are clearly worse. On LCB the instruction fails almost always (LPR .07–.35, English preambles) and steering gives .82–1.00, significant after Holm in every language. The judge no longer scores Gemma's English answer template line (no effect on the other models). Steering without the instruction is close to the instruction (0.752 / 0.741) and leaves some Bengali–Telugu answers in S; κ with native moves by +0.001 with the instruction. Results in both READMEs.
- [ ] **P1 ReCoVeR baseline** (`src/experiments/mgsm_crosslingual/recover.py`, `recover.sh`, `recover_check.py`): the published method, whose vector has the same direction as ours ([LITERATURE.md](LITERATURE.md), *ReCoVeR*), in its training-free variant, reimplemented in `Model.generate(recover=...)`. Their code (github.com/hSterz/recover @ 4c6eb4d) has no licence, so it is not copied. The method adds α · normalize(μ_T − μ_S) at every block, at every position except the first token, and at every generated token, with an optional norm restore. Both arms use the instruction prompt in S:
  - `recover`: all blocks, every token, i.e. ReCoVeR as published;
  - `recover_prompt`: all blocks, prompt tokens only.

  Steps:
  - A. Check the hooks against their `naive_steer` (logits on the same tokens, prompt and teacher-forced decoding), and our batched decoding against bs = 1.
  - B. Check our vector construction against their `_aggregate`, by the cosine at every hidden state.
  - C. Dev grid: α ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16} × restore on/off, on the 11 tuning pairs × 50 dev items.
  - D. Selection by our rule: highest mean dev success, with ties going to the smaller α, then restore off. Flag the result if the chosen α is at the edge of the grid.
  - E. Test on the 99 held-out pairs × 200 items.
  - F. Analysis:
    - success, accuracy, in-T rate and capped rate;
    - McNemar + Holm against `instruction_steer_prompt`, `instruction_steer`, `instruction` and native;
    - bootstrap CIs;
    - κ with native.
    Output goes to `results/<model>/recover.json`.
  - G. Fidelity check: Llama cross-lingual LCB in their setting (100 tokens, their system prompt; greedy here). The paper reports 91.0 → 97.9 LPR.

  All four models run in one main job plus helpers, sharing the work through claim files.

  Caveats to state in the paper:
  - no ReCoVeR+ (the trained variant);
  - a long-CoT regime (2048 tokens) they never tuned for (they used 100 tokens);
  - greedy decoding where they sample;
  - the comparison is between methods, not single factors (layers × positions);
  - small differences in vector collection (padding);
  - only Llama overlaps with their models.
  - Gemma: both arms chose α = 16, the top of the grid (2026-11-18). The grid is in their absolute units, and Gemma's mid-layer hidden norm is about 50,000 (Llama's about 6), so α = 16 is a small fraction of the state. It still steers (on MGSM +.046 over the instruction; on LCB the LPR rises from .15 at α ≤ 4 to about .9 at α = 16), but the best α may lie above the grid. We kept the grid, by decision; state this as a limitation, not as a tie.

  ITLC is an optional second baseline.
- [ ] **P1 Second benchmark, outside maths: MMLU-ProX** (Xuan et al., arXiv 2503.10497; `li-lab/MMLU-ProX` on Hugging Face, MIT). 11,829 parallel questions per language, 10 options, all 11 MGSM languages (so the full 110 pairs), a 70-item validation split. Non-maths categories only (law, economics, psychology, business, health, philosophy, history, other: about 5,900 questions), a fixed sample of a few hundred per language, chain of thought ("reason step by step, answer in T"); correctness is the letter, success = correct and the reasoning in T. Arms as in MGSM, the layer and α from MGSM, not retuned. Steps:
  1. **Pilot first:** the instruction only (no steering), all four models, a handful of pairs, about 300 questions, to check that the failure appears outside maths.
  2. **Language-ID check:** the options are in S, so quoted option text can make reasoning in T look like S. Measure it in the pilot; if it matters, strip quoted options or classify per sentence.
  3. The full run, only if the pilot shows the failure (otherwise the paper's claim narrows to reasoning-heavy tasks).
  Replaces MSVAMP, which adds robustness but no breadth (the same kind of problems as MGSM). Fallback, if reviewers question MGSM specifically: MSVAMP (`data/processed_data/msvamp.jsonl`; 1000 items, 90 pairs, no te; machine-translated, some items flagged in `src/data/README.md`).
- [ ] **P2 Instruction written in T, all models** (so far Qwen3-8B only, above): cheap, the prompt baseline reviewers expect (CLaS-Bench's baseline II).
- [ ] **P3 "Reason in English, answer in T":** a possible route to beating native in low-resource targets; a separate experiment.

### Analysis and statistics

- [x] **P1** Multiple-comparison correction (Holm) for the per-pair McNemar tests: over the 99 held-out pairs (MGSM), the 6 languages (LCB) and the 10 languages (MGSM from English), stored as `p_holm` and reported next to the uncorrected counts in each README. The main claim holds: instruction + prompt-only steering is better in 57 / 62 / 31 / 11 held-out pairs and worse in 0 / 0 / 4 / 1 (Llama / Qwen3-8B / Qwen3-32B / Gemma-3-12B). What weakens: Qwen3-32B on LCB (no prompt-only gain survives) and the single es gain on MGSM from English.
- [x] **P2** Per-pair tests against native, Holm-corrected over the 99 held-out pairs, all four models (`src/experiments/mgsm_crosslingual/native_tests.py`). Instruction + prompt-only steering stays below native (−.089 / −.052 / −.035 / −.051, all CIs exclude 0; 23 / 20 / 13 / 21 pairs below after Holm, against 65 / 65 / 32 / 28 for the instruction), closing 49–85% of the gap; the pairs above native are mostly where native is weak (Qwen3-8B into sw). Japanese: steering matches native (above in three models, significant only in Qwen3-32B), so the paper says "matches", not "beats".
- [x] **P2** One consistent statement of CIs and tests: McNemar per pair with Holm over the pairs of a comparison, and 95% bootstrap CIs over the 200 items (2000 resamples) for means and mean differences; stated in the Setup of the cross-lingual MGSM README, with CIs added to its headline table. For the paper: the same sentence in Setup, and CIs on the headline means and differences; no further tests.

### Writing

- [ ] **P1** Decide where the paper and report live (LaTeX in the repo, or Overleaf) and the target venue and format.
- [ ] **P1** New title (the current one refers to ICL and a per-token method).
- [ ] **P1** Related work, every reference checked against the source: language confusion (Marchisio et al., 2024) and its mitigations; language-specific neurons and representations; activation steering and representation engineering; Kirtane & Huang (2026); multilingual reasoning (Shi et al., 2023; Schut et al., 2025; Wendler et al., 2024).
  The search is done: [LITERATURE.md](LITERATURE.md) (positioning, closest papers, baselines, paragraph plan), with the detailed lists in `docs/related_work/`. The vector itself is published (ReCoVeR, Findings of EMNLP 2025, has the same direction; ITLC, CLaS-Bench), so the paper is pitched as the cross-lingual reasoning evaluation and analysis, not a new method. Still to do: the final bibliography, each entry checked again (LITERATURE.md, *To check*).
- [ ] **P1** Reframe the title, introduction and contributions around the cross-lingual failure (LITERATURE.md, *Framing*); LCB becomes supporting evidence.
- [ ] **P1** Figures: success per pair (heatmap), failure counts, KL by position, layer × α grid.
- [ ] **P2** Report sections 3-6 can be written now from existing results.
- [ ] **P2** Update the TODO list in [README.md](README.md) to point here.

### Open decisions

- [ ] Deadlines for the report and the paper.
- [ ] Target venue and page limit; report length limit.
