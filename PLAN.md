# Plan: paper and project report

The original proposal is in [README.md](README.md); the experiment plans written for it are in [docs/plans/](docs/plans/). This file records where the project now stands, what the paper and the report will say, and what is left to do.

## Central claim

> Instruction prompts often fail to control the output language across languages; with Llama-3.1-8B-Instruct, "answer in T" fails almost completely (success < 0.10) in 40 of 99 held-out source-target pairs on MGSM. A single training-free language vector, added to the prompt, fixes this (1 of 99 pairs fail) without changing which problems the model can solve.

Everything else supports, delimits or motivates this claim:

* **Where steering helps:** wherever the instruction fails (LCB cross-lingual; MGSM with a non-English question).
* **Where it does not:** where the instruction already works (MKQA, MGSM asked in English) and in multilingual ICL (no headroom on math; on XNLI the gain is not language-specific).
* **Why the per-token method of the proposal was dropped:** the language decomposition holds for sentence averages, not for single tokens.
* **What steering changes:** the start of the answer (its language), not the reasoning (κ and teacher-forced KL).

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

- [ ] **P1 Second model on cross-lingual MGSM.** Qwen2.5-7B-Instruct (different language balance: strong zh/ja, weaker sw/te).
  - [ ] Check that `mgsm_steering/generate.py` does not assume Llama (32 layers, block access, chat template, `<|eot_id|>`); make the model a parameter.
  - [ ] Express the layer grid in relative depth so it carries over between models.
  - [ ] Same items, tuning pairs, grid, judge; one job with one process per GPU.
  - [ ] Rerun `analyse_pairs.py`, `agreement.py`, `kl.py`, `analyse_kl.py`, `robustness.py`.
- [ ] **P1 Second model on LCB**, with the same settings.
- [ ] **P1 Steering controls in cross-lingual MGSM**, at the chosen settings, on the test pairs: a norm-matched random vector (3 draws) and a wrong-target vector v(S→U), U ≠ S, T. Same definitions as the ICL controls.
- [ ] **P1 Stronger prompt baselines** (decide which): the instruction in a system prompt; the instruction after the question; one or two demonstrations answered in T; translate the question into T (NLLB), then ask natively.
- [ ] **P2 Sensitivity:** success over the layer × α dev grid (data exists, needs a figure); number of FLORES sentences used for the vectors (e.g. 10, 100, 997).
- [ ] **P2 Interpretability, one analysis:** where the language of the first answer token is decided, across layers (logit lens, or projection on the language direction), for instruction vs instruction + steering, in pairs where the instruction fails.
- [ ] **P3 Third model** (e.g. Gemma-2-9B-it), mainly if the second model disagrees with Llama.
- [ ] **P3 "Reason in English, answer in T":** a possible route to beating native in low-resource targets; a separate experiment.

### Analysis and statistics

- [ ] **P1** Multiple-comparison correction (Holm) for the per-pair McNemar tests.
- [ ] **P2** Per-pair tests against native (high-resource S into each T), Holm-corrected; the Japanese result.
- [ ] **P2** One consistent statement of CIs and tests across all experiments.

### Writing

- [ ] **P1** Decide where the paper and report live (LaTeX in the repo, or Overleaf) and the target venue and format.
- [ ] **P1** New title (the current one refers to ICL and a per-token method).
- [ ] **P1** Related work, every reference checked against the source: language confusion (Marchisio et al., 2024) and its mitigations; language-specific neurons and representations; activation steering and representation engineering; Kirtane & Huang (2026); multilingual reasoning (Shi et al., 2023; Schut et al., 2025; Wendler et al., 2024).
- [ ] **P1** Figures: success per pair (heatmap), failure counts, KL by position, layer × α grid.
- [ ] **P2** Report sections 3-6 can be written now from existing results.
- [ ] **P2** Update the TODO list in [README.md](README.md) to point here.

### Open decisions

- [ ] Deadlines for the report and the paper.
- [ ] Target venue and page limit; report length limit.
- [ ] Which prompt baselines (above).
- [ ] Which second (and third) model.
