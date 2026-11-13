# Experiment 2: Output-Language Control by Steering

**Goal:** Test one thing: whether language steering can make the model **answer in a different language than it was asked in**, and whether it does so better than simply telling the model which language to use. This is TODO item 7 in the [README](../../README.md).

The experiment is deliberately small: one task, four languages, four conditions and three measurements.

**Status:** the pilot is done. On MKQA the instruction already reaches 97–100% fidelity, so the stopping rule applied and the full experiment was not run. Code and results: [src/experiments/output_language](../../src/experiments/output_language/README.md).

## Setup

* **Model:** Llama-3.1-8B-Instruct ([Grattafiori et al., 2024](#grattafiori-2024)).
* **Task:** MKQA ([Longpre et al., 2021](#longpre-2021)), open-domain factual questions that are parallel across 26 languages, with answers localized per language and lists of answer aliases.
  * Keep every item that has an answer, and drop unanswerable and long-answer items. Do **not** filter on whether the localized answer equals the English one; that split is tracked as a diagnostic (see [Metrics](#metrics)).
  * **Splits, by item ID:**
    * *compute*: 300 items, used for the steering vectors;
    * *val*: 200 items, used for tuning;
    * *test*: 1000 items, used for every number reported.

    All splits are identical across languages.
* **Target languages:** French, Chinese, Russian and Hungarian; the source language is always English.
  * French is officially supported by Llama 3.1, so it is the control, where an instruction should already work.
  * Chinese, Russian and Hungarian are officially unsupported. They differ in script (Latin, Han, Cyrillic) and in resource level, with Hungarian at the low end.
* **Response format:** the same in every condition.
  * The model gives a short explanation of 1–3 sentences, then the final answer in `\boxed{…}`. This marker is language-neutral and easy to parse.
  * The format instruction is written in the language of the question: English for the English-question conditions, and translated (and checked) for the native condition. This keeps English text out of the native prompt.
* **Decoding:** greedy, with `max_new_tokens = 256`. There are no demonstrations, so one run per condition suffices.

## Conditions

| | Question | Instruction | Steering | Role |
|---|---|---|---|---|
| **Plain** | English | none | none | reference: the fully English end of the KL scale |
| **Naive** | English | "Answer in *T*." | none | baseline 1 |
| **Native** | *T* | none | none | baseline 2: hypothetical ceiling |
| **LS** | English | none | toward *T* | the method |

LS gets no instruction, so that the effect of steering is isolated.

### Steering

* **Vectors:** $\mathbf{v}_{(\text{en}\to T)}(l) = \mu_T(l) - \mu_{\text{en}}(l)$, computed as in `src/icl/vectors.py` (mean state over tokens of the questions in the compute split). They are recomputed for MKQA for all four languages, not reused from XNLI or MGSM, since there is no Hungarian vector and in-domain vectors work better.
* **Where:** $\mathbf{h}_p \leftarrow \mathbf{h}_p + \alpha\,\mathbf{v}(l)$ at one layer $l$, for every question token **and every generated token**. The generated tokens are the new part, because the ICL harness only steers prompt positions.
* **Tuning, on val, per language:** search the grid layer ∈ {8, 12, 16, 20, 24} × α ∈ {0.5, 1, 2, 3}.
  * Pick the configuration with the highest fidelity whose accuracy is at most 1 point below Plain.
  * If no configuration qualifies, report that rather than relaxing the constraint.

## Metrics

Exactly three, per language and condition, on test.

1. **Fidelity:** the share of responses whose explanation is in *T*, according to GlotLID ([Kargaran et al., 2023](#kargaran-2023)). The explanation is the text before `\boxed{}`.
2. **Accuracy:** whether the normalized content of `\boxed{}` contains any alias of the *localized* answer.
   * **Diagnostic only, not a headline number:** accuracy is also reported separately on items whose localized answer equals the English one and on items where it differs. On the "equal" items an English answer still scores correct, so the "differs" items are where accuracy also tests language. If steering helps, the gain over Naive should be larger there.
3. **KL to native:**
   * **How it's computed:** take Native's greedy responses as the reference. Teacher-force each one under every other condition, with steering active on the response tokens for LS, as in generation. At every response position, compute $D_{\mathrm{KL}}\big(p_{\text{Native}} \,\Vert\, p_{\text{cond}}\big)$ over the full vocabulary. The response tokens are identical across conditions, so only the prompt (and steering) differs.
   * **Summary:** the gap closed, $1 - \overline{\mathrm{KL}}_{\text{cond}} / \overline{\mathrm{KL}}_{\text{Plain}}$, where 0 means it behaves as if asked in English and 1 as if asked natively.
   * **One plot:** mean KL against response position for Plain, Naive and LS. It shows whether Naive starts in *T* and then drifts, and whether LS holds.

**Statistics:** 95% CIs from a paired bootstrap over items (2000 resamples), and the differences LS − Naive per language.

## Deliverables

1. One table: fidelity and accuracy for each language × condition, with LS − Naive and its CI.
2. One bar chart: the share of the KL gap closed by Naive and LS, per language.
3. One plot: KL against response position.
4. Appendix only: the accuracy split by whether the answers are equal or differ.

## Order of work

1. **Data:** load MKQA, build the splits, and report item counts and the share of "equal"/"differs" items per language.
2. **Pilot:** run Plain and Naive on val only. If Naive already reaches about 95% fidelity in every language, there is no headroom, so stop and reconsider before building anything else.
3. **Steering:** compute the vectors, add steering on generated tokens, and tune on val.
4. **Test:** run all four conditions and the teacher-forced KL pass.

**Compute:** test generation is 1000 items × 4 languages × 3 conditions, plus Plain once, ≈ 13k short generations, which is well under an hour on one B300. Tuning is 20 configurations × 200 items × 4 languages. Everything fits in one or two jobs.

## References

* <a id="grattafiori-2024"></a>Grattafiori, A., et al. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*.
* <a id="kargaran-2023"></a>Kargaran, A. H., Imani, A., Yvon, F., & Schütze, H. (2023). GlotLID: Language Identification for Low-Resource Languages. *Findings of EMNLP 2023*.
* <a id="longpre-2021"></a>Longpre, S., Lu, Y., & Daiber, J. (2021). MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering. *TACL 9*.
