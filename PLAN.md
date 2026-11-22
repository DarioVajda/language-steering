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
4. Setup: models, benchmarks (LCB, MGSM and MMLU-ProX in 110 pairs), success = correct and in T, judge and its checks.
5. Results: instruction vs steering vs native; failure counts; controls; prompt baselines; ReCoVeR and the ablation; four models; MMLU-ProX.
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

## Results already in hand (Llama-3.1-8B-Instruct; the later ones on all four models)

| Result | Where | In paper | In report |
|---|---|---|---|
| T1: language subspace at sentence level | `src/experiments/pca_analysis/` | 1 paragraph | yes |
| W1-W2: no per-token structure | `src/experiments/word_level_analysis/` | 1 paragraph | yes |
| ICL: no headroom on math, non-specific gain on XNLI | `src/icl/RESULTS.md` | motivation / appendix | yes |
| MKQA pilot: instruction already works | `src/experiments/output_language/` | 1 sentence | yes |
| MGSM from English: instruction already works | `src/experiments/mgsm_steering/` | 1 sentence | yes |
| LCB: steering fixes language confusion | `src/experiments/language_confusion/` | main | yes |
| Cross-lingual MGSM: success, failure counts, κ, KL, by resource | `src/experiments/mgsm_crosslingual/` | main | yes |
| MMLU-ProX: the failure and the fix outside maths, against ReCoVeR | `src/experiments/mmlu_prox/` | main | yes |

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
- [x] **P1 ReCoVeR baseline** (`src/experiments/mgsm_crosslingual/recover.py`, `recover.sh`, `recover_check.py`; `src/experiments/language_confusion/recover.py`, `recover.sh`): the published method, whose vector has the same direction as ours ([LITERATURE.md](LITERATURE.md), *ReCoVeR*), in its training-free variant, reimplemented in `Model.generate(recover=...)`. Their code (github.com/hSterz/recover @ 4c6eb4d) has no licence, so it is not copied. As published: α · normalize(μ_T − μ_S) at every block, at every position except the first token and at every generated token, with norm restore, on the instruction prompt in S. The only change is that α is chosen on dev (α ∈ {0.05, …, 16}, doubled while the best is the largest); one α for all pairs on MGSM, one shared setting on LCB (per-language as a secondary result). No other variants, and our arms are not rerun.
  - Checks against their code: hook logits identical with the same vector (up to bf16 in 4 of 32 settings), vector cosine ≥ 0.9999 at every hidden state, their Llama LCB setting 0.916 → 0.975 LPR (paper 91.0 → 97.9).
  - MGSM (99 held-out pairs), success: ReCoVeR 0.567 / 0.661 / 0.796 / 0.772 (Llama / Qwen3-8B / Qwen3-32B / Gemma; α 0.1 / 0.5 / 1 / 16, none at the grid edge), against prompt-only steering 0.634 / 0.711 / 0.800 / 0.777: −.067 and −.051 (CIs exclude 0) in the 8B models, ties in the other two. In Llama ReCoVeR loses accuracy (.579, below the instruction's .673) and doubles the answers that hit the token limit.
  - LCB (shared setting), mean LPR: ReCoVeR .975 / .958 / .980 / .976 (α 0.1 / 0.5 / 2 / 64), our all-token .976 / .965 / .958 / .979; no language differs significantly from either of our arms after Holm.
  - Results in both READMEs (*Baseline: ReCoVeR*). Runs: 1 h 02 min, 3 B300 + 2 B200, everything in one job plus a helper.

  Caveats to state in the paper:
  - no ReCoVeR+ (the trained variant);
  - a long-CoT regime (2048 tokens) they never tuned for (they used 100 tokens);
  - greedy decoding where they sample;
  - the comparison is between methods, not single factors (layers × positions);
  - small differences in vector collection (padding);
  - only Llama overlaps with their models.

  ITLC is an optional second baseline.
- [x] **P1 Second benchmark, outside maths: MMLU-ProX** (Xuan et al., arXiv 2503.10497; `li-lab/MMLU-ProX`, MIT; `src/experiments/mmlu_prox/`). 200 STEM questions (no maths) and 200 humanities / social sciences / other, the same in all 11 MGSM languages, all 110 pairs, zero-shot chain of thought, 4096 tokens; success = correct letter and the answer in T. The layer and α of our steering and ReCoVeR's α are those of MGSM dev, not retuned. Llama is scored on its free-form answers. The other three models are asked for `\boxed{X}` and scored by it: a format pilot showed that they box in every arm, while Llama's steered arms drop the box twice as often (appendix only). Tests as on MGSM (McNemar with Holm over the 99 held-out pairs, bootstrap CIs; `tests.py`).
  - Held-out success, ours / ReCoVeR / instruction (STEM; humanities): Llama .229 / .192 / .099 (.198 / .164 / .087), Qwen3-8B .457 / .387 / .032 (.328 / .281 / .031), Qwen3-32B .644 / .633 / .498 (.506 / .497 / .376), Gemma .473 / .492 / .461 (.440 / .446 / .420).
  - Ours closes 69–81% of the gap to native in Llama and the Qwen models, and is better than the instruction in 35–91 pairs per subset after Holm (worse in at most 1).
  - Ours − ReCoVeR: +.037 / +.034 (Llama), +.070 / +.047 (Qwen3-8B), +.011 / +.009 (Qwen3-32B), all CIs above 0; Gemma −.019 / −.006, where the instruction already mostly works (0.84–0.91 in T). Gemma's deficit comes from five pairs: into English our arm drops the box, into German the answer drifts into English during generation.
  - So the ranking against ReCoVeR follows MGSM's (ahead in the 8B models, tied or close in the other two). For the paper: the headline table and the tests in the main body; the per-model format, the pilot and the transliteration of non-Latin boxes in the appendix.
  - MSVAMP is no longer needed (it adds robustness, not breadth).
- [x] **P2 Ablation: one layer and prompt-only** (`src/experiments/mgsm_crosslingual/ablation.py`, `every_layer.sh`; `src/experiments/language_confusion/ablation.py`), Llama and Qwen3-8B. Each of the two choices that separate our method from ReCoVeR is changed alone, everything else kept.
  - MGSM, ours − the other arm: all-token steering +.031 / +.007, every-layer prompt-only steering +.000 / +.008, ReCoVeR +.067 / +.051. Every layer *and* all tokens with our vector (Llama) comes close to ReCoVeR (0.585 against 0.567), so the two choices together account for the gap.
  - LCB: every layer within .01 of one layer; prompt-only .013–.018 below all-token, as LCB scores only the language.
  - Results in both READMEs (*Ablation: one layer and prompt-only*).
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
