# Experiment 1: Testing the Language Decomposition Hypothesis

**Goal:** Check, using forward passes only, whether hidden states follow the decomposition from the [README](README.md):

$$
\mathbf{h}^L_c(l) = \mu_l + \mathbf{a}_c(l) + \gamma^L_c \cdot \mathbf{u}^L_l, \qquad \mathbf{a}_c(l) \perp \mathbf{u}^L_l, \quad \Vert \mathbf{u}^L_l \Vert = 1
$$

Here $c$ is a concept (an aligned word). $\mathbf{a}_c$ is its meaning, shared across languages. $\gamma^L_c$ is its per-token language scale.

The notation generalises the README's:
* $\mathbf{a}_c$ corresponds to $\mathbf{h}^{\text{abs}}_t$, and $\mathbf{u}^L$ to the normalised $\mathbf{v}^L$.
* $\gamma$ may depend on the language, since T3 tests whether the README's single $\gamma$ holds.

The tests map to the README as follows:
* T1, T3 and T4 test [Hypothesis 1](README.md#hyp-decomposition).
* T2 and W5 test [Hypothesis 2](README.md#hyp-orthogonality).
* T5 and W2 test the per-token scale $\alpha^{(S,T)}(\mathbf{h}^S_t)$ that the method relies on.

The experiment also checks:

* how well the precomputed global directions describe *individual words*;
* whether a per-token scale reconstructs target-language states better than a constant shift.

Results are indicative only and come before any downstream ICL experiments.

## Setup

* **Model:** Llama-3.1-8B-Instruct ([Grattafiori et al., 2024](#grattafiori-2024)). Hidden states are taken from every layer.
* **Languages:** English plus the 7 languages common to all three benchmarks used by [Kirtane and Huang (2026)](#kirtane-2026): Chinese, French, German, Russian, Spanish, Swahili and Thai. This covers Latin, Cyrillic, Chinese and Thai scripts and one low-resource language, and every language can be compared with its downstream steering results.
* **Data (in-domain, parallel across languages):** source text only, i.e. questions, or premises and hypotheses. Answers and reasoning are not used. In-domain data is used because [Kirtane and Huang (2026)](#kirtane-2026) found FLORES-based vectors worse than benchmark-format vectors.

  * **Items:** use the processed data in `data/processed_data/` (see [`src/data/README.md`](src/data/README.md)):
    * `split == "compute"` for MGSM ([Shi et al., 2023](#shi-2023); 83 items) and MSVAMP ([Chen et al., 2023](#chen-2023); 333);
    * `split == "analysis"` for XNLI ([Conneau et al., 2018](#conneau-2018); 2490 validation items, with no premise shared with the ICL splits).
  * **Filtering:** drop any (item, language) pair listed in `flags`; these are broken machine translations. MSVAMP English is used in its truecased form.
  * **Fit/test halves:** items are split into a *fit* half, used for global estimates, and a *test* half, used for all metrics, **by item ID**.
* **Word alignment:** align every English–$L$ sentence pair with SimAlign ([Jalili Sabet et al., 2020](#jalilisabet-2020)) or awesome-align ([Dou & Neubig, 2021](#dou-2021)). Both use an external multilingual encoder (mBERT), so the alignment is independent of the model being analysed.
  * Keep only English words aligned one-to-one in all 7 languages. Each such word is one concept $c$.
  * Chinese and Thai are first split into words with [jieba](https://github.com/fxsjy/jieba) and [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp).
  * Each word is represented by its last subword token, taken in sentence context. Mean-pooling over the word's subwords serves as a robustness check.
  * Run both aligners and report how sensitive the results are to the choice.
* **Hygiene:** exclude BOS, chat-template and attention-sink positions. Report results with and without the massive-activation / rogue dimensions ([Sun et al., 2024](#sun-2024); [Timkey & van Schijndel, 2021](#timkey-2021)).
* **Global estimates (fit split):**
  * $\mu_L$ is the mean over tokens of language $L$.
  * $\mu_l = \frac{1}{K}\sum_L \mu_L$.
  * $\mathbf{u}^L_l = (\mu_L - \mu_l)/\Vert \mu_L - \mu_l \Vert$.
  * $\mathbf{v}^{(S,T)}_l = \mu_T - \mu_S$.
  * $\gamma^L_c = (\mathbf{h}^L_c - \mu_l) \cdot \mathbf{u}^L_l$.

## Tests

Every test is reported **per layer**. The decomposition is expected to fit best in middle layers, where the latent concept space is found ([Wendler et al., 2024](#wendler-2024); [Schut et al., 2025](#schut-2025)).

| # | Hypothesis | Measurement | Null / control |
|---|---|---|---|
| **T1** | Meaning cancels between translations | Centre each concept's $K$ states across languages, then run PCA on the pooled residuals. Report the variance explained by the top $K{-}1$ components and their overlap with $\text{span}\{\mathbf{u}^L\}$ (cf. [Chang et al., 2022](#chang-2022); [Libovický et al., 2020](#libovicky-2020)). | Mismatched (non-translation) pairs; random $k$-dimensional subspaces ($\approx k/d$) |
| **T2** | Orthogonality, $\mathbf{a}_c \perp \mathbf{u}^L$ | Two-way ANOVA (concept $\times$ language) on the projections $(\mathbf{h}^{L'}_c - \mu_l)\cdot\mathbf{u}^L_l$. The **concept main effect** should be small: a large one means meaning leaks onto the language axis. | Projections onto random unit directions |
| **T3** | One scalar per token | Build the matrix $\Gamma_{c,L} = \gamma^L_c$. Measure how close it is to rank 1 ($\gamma_c \cdot s_L$) and the cross-language correlation of $\gamma_c$. The README additionally assumes $\gamma^S_c = \gamma^T_c$. | Shuffled concept labels |
| **T4** | Additivity / path independence | Compare $\Vert \textbf{steer}_{(U,T)}(\textbf{steer}_{(S,U)}(\mathbf{h}^S_c)) - \textbf{steer}_{(S,T)}(\mathbf{h}^S_c) \Vert$ and the round trip $\Vert \textbf{steer}_{(T,S)}(\textbf{steer}_{(S,T)}(\mathbf{h}^S_c)) - \mathbf{h}^S_c \Vert$, both relative to $\Vert \mathbf{h}^S_c - \mu_l \Vert$. | — |
| **T5** | **Per-token $\gamma$ beats a constant shift** | Reconstruction error $\Vert \hat{\mathbf{h}}^T_c - \mathbf{h}^T_c \Vert$ and cosine similarity for: **(a)** no steering, **(b)** $\mathbf{h}^S_c + \alpha\,\mathbf{v}^{(S,T)}$ with $\alpha$ fitted on the fit split, **(c)** the README method, **(d)** (c) with $\gamma$ shuffled across tokens, **(e)** (c) with a fitted global multiplier $\beta$, **(f)** (c) with $\mathbf{u}^S, \mathbf{u}^T$ replaced by random unit directions. | (a), (d) and (f) |
| **T6** | Language-agnostic tokens need less steering | $\gamma$ and $\Vert \mathbf{h}^L - \mathbf{h}^{L'} \Vert$ for tokens that are *identical* across languages (digits, `=`, punctuation, named entities) vs. content words. | Content words |

## Word-Level Analysis

This section asks how well the precomputed global quantities describe each word and how much words differ within one language pair. For every concept $c$ and pair $(S,T)$, with $S$ = English, define the local difference

$$
\mathbf{d}_c = \mathbf{h}^T_c - \mathbf{h}^S_c
$$

and compute, on the test split:

* **W1 – Direction agreement:** the distribution of $\cos(\mathbf{d}_c, \mathbf{v}^{(S,T)})$. If it is low, no method that steers along $\mathbf{v}$, constant or per-token, can capture the difference.
* **W2 – Oracle per-word scale:** $\alpha^*_c = \mathbf{d}_c \cdot \mathbf{v}^{(S,T)} / \Vert \mathbf{v}^{(S,T)} \Vert^2$, the best scale for that word along the global direction.
  * Its spread across words (coefficient of variation) is the **headroom** for any per-token method. A small spread means a constant $\alpha$ is enough.
  * The correlation between $\alpha^*_c$ and the README prediction $\gamma^S_c$, which is computed from $\mathbf{h}^S_c$ alone, is the **key test** of whether the $\gamma$ formula predicts each word's scale.
* **W3 – Residual:** the fraction of $\Vert \mathbf{d}_c \Vert^2$ left after removing (i) a constant-scale component along $\mathbf{v}$ and (ii) a per-word-scale component along $\mathbf{v}$.
* **W4 – Per-word language directions:** $\mathbf{u}^L_c = \mathbf{h}^L_c - \frac{1}{K}\sum_{L'} \mathbf{h}^{L'}_c$. Report its cosine with the global $\mathbf{u}^L$ and the spread of these vectors across words.
* **W5 – Per-word meaning:** $\mathbf{a}_c = \frac{1}{K}\sum_{L} \mathbf{h}^L_c - \mu_l$. Report the projection $\mathbf{a}_c \cdot \mathbf{u}^L$, which checks orthogonality word by word, relative to $\Vert \mathbf{a}_c \Vert$.

All of W1–W5 are broken down by **word type** (content words, digits / math symbols, punctuation, named entities), **frequency**, **language pair** and **layer**, to locate where the variance comes from.

## Decision Criteria

* **Proceed to downstream ICL** if all three hold at the candidate steering layers:
  * T1 and T2 support the decomposition.
  * W2 shows both non-trivial headroom and a positive correlation between $\alpha^*_c$ and $\gamma^S_c$.
  * T5(c) or T5(e) beats (b), (d) and (f), using a paired bootstrap over concepts.
* **Geometry-only project** if T1–T4 and W1–W5 are informative but per-token scaling shows no gain.
* **Rethink** if T1 shows no shared low-rank language structure at any layer.

## Limitations

* A good representation-level fit need not improve ICL accuracy. [Kirtane and Huang (2026)](#kirtane-2026) choose $\alpha=3$ on XNLI, which suggests over-steering helps.
* Word-alignment and segmentation noise (especially for Chinese and Thai), word-order and position effects, and tokenisation bias in which words survive the one-to-one filter. MGSM is small (83 compute items), so it is pooled with MSVAMP and XNLI.
* $\mu_l$, and hence $\gamma$, depends on which languages are pooled. Only differences such as $\mu_T - \mu_S$ are invariant to that choice.

## References

* <a id="chang-2022"></a>**Chang, T. A., Tu, Z., & Bergen, B. K.** (2022). The Geometry of Multilingual Language Model Representations. *Proceedings of EMNLP 2022*. <https://arxiv.org/abs/2205.10964>

* <a id="chen-2023"></a>**Chen, N., Zheng, Z., Wu, N., Gong, M., Zhang, D., & Li, J.** (2023). *Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations*. <https://arxiv.org/abs/2310.20246>

* <a id="conneau-2018"></a>**Conneau, A., Lample, G., Rinott, R., Williams, A., Bowman, S. R., Schwenk, H., & Stoyanov, V.** (2018). XNLI: Evaluating Cross-lingual Sentence Representations. *Proceedings of EMNLP 2018*. <https://arxiv.org/abs/1809.05053>

* <a id="dou-2021"></a>**Dou, Z.-Y., & Neubig, G.** (2021). Word Alignment by Fine-tuning Embeddings on Parallel Corpora. *Proceedings of EACL 2021*. <https://arxiv.org/abs/2101.08231>

* <a id="grattafiori-2024"></a>**Grattafiori, A., et al.** (2024). *The Llama 3 Herd of Models*. <https://arxiv.org/abs/2407.21783>

* <a id="jalilisabet-2020"></a>**Jalili Sabet, M., Dufter, P., Yvon, F., & Schütze, H.** (2020). SimAlign: High Quality Word Alignments Without Parallel Training Data Using Static and Contextualized Embeddings. *Findings of EMNLP 2020*. <https://arxiv.org/abs/2004.08728>

* <a id="kirtane-2026"></a>**Kirtane, N., & Huang, K.-H.** (2026). *Language Steering for Multilingual In-Context Learning*. <https://arxiv.org/abs/2602.02326>

* <a id="libovicky-2020"></a>**Libovický, J., Rosa, R., & Fraser, A.** (2020). On the Language Neutrality of Pre-trained Multilingual Representations. *Findings of EMNLP 2020*. <https://arxiv.org/abs/1911.03310>

* <a id="schut-2025"></a>**Schut, L., Gal, Y., & Farquhar, S.** (2025). *Do Multilingual LLMs Think In English?* <https://arxiv.org/abs/2502.15603>

* <a id="shi-2023"></a>**Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., Das, D., & Wei, J.** (2023). Language Models are Multilingual Chain-of-Thought Reasoners. *Proceedings of ICLR 2023*. <https://arxiv.org/abs/2210.03057>

* <a id="sun-2024"></a>**Sun, M., Chen, X., Kolter, J. Z., & Liu, Z.** (2024). *Massive Activations in Large Language Models*. <https://arxiv.org/abs/2402.17762>

* <a id="timkey-2021"></a>**Timkey, W., & van Schijndel, M.** (2021). All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality. *Proceedings of EMNLP 2021*. <https://arxiv.org/abs/2109.04404>

* <a id="wendler-2024"></a>**Wendler, C., Veselovsky, V., Monea, G., & West, R.** (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. *Proceedings of ACL 2024*. <https://arxiv.org/abs/2402.10588>
