# Project Proposal

### **Title:** Mathematically Grounded Method for Language Steering in Multilingual In-Context Learning

**Author:** Dario Vajda

**Institution:** University of Cambridge, MPhil in Advanced Computer Science

## TODO

1. ✅ Test the hypotheses —> a pivot is required (the hypotheses hold for sentence-level representations, but don't work on token level)
2. ✅ Fix the evaluation protocol + Get the baseline results – english few-shot examples, MFS (multilingual few-shot), constant vector steering, oracle (with the target language few-shot examples)
    * should decide on which layers to steer in
    * should decide on the hyperparameter (scale, etc.)
3. ✅ Test vector steering ablation with (1) multiple random scale-matched vectors, or (2) other language vectors, and attribute the performance gains/gaps.
4. Do the same for the new MFS+steering method (for each example we use the corresponding language vector to our target)
5. All above tests are done on XNLI —> extend to the other two benchmarks
6. Extend everything to 1-2 more LLMs.
7. Quantitatively evaluate the language-steering effects (both for regular steering and random steering)
    * zero-shot language forcing – ask in english, steer to $T$ and see how the model answers
    * compare the representations/distributions (target language input vs english with steering) – measure $D_{KL}$ and $|h_{steer}-h_{oracle}|/|h_{base}-h_{oracle}|$.
    * **logic tax** – if the output language is consistently right (the one we steer towards), measure how much the performance degrades (if at all).
    * cross-lingual instruction following – test whether it can follow instructions more consistently with english prompt and language steering than it does in the target language natively (best case scenario).

## Motivation

This project is inspired by recent findings about LLM interpretability. Specifically, the finding that multilingual LLMs tend to reason in English, as demonstrated in [Schut et al. (2025)](#schut-2025). Additionally, they suggest that it is highly likely that the model's internal representations can be cleanly decomposed into two distinct parts: the first representing the abstract meaning of the given concept, and the second encoding the language. Further, following [Ethayarajh (2019)](#ethayarajh-2019), this project relies on the observation that LLM internal representations are usually clustered around a massive global mean.

First, this project will validate some of the properties suggested by those previous works and test [Hypotheses 1](#hyp-decomposition) and [2](#hyp-orthogonality). Upon concluding the preliminary interpretability work, this project will focus on evaluating a mathematically grounded method for language steering in multilingual in-context learning, which heavily relies on the previous findings.

> <a id="hyp-decomposition"></a>**Hypothesis 1 (Decomposition):** In principle, all internal representations of a multilingual LLM can be broken down into three distinct parts:
>
> $$
> \mathbf{h}^L_t(l) = \mu_l + \mathbf{h}^{\text{abs}}_t(l) + \gamma \cdot \mathbf{v}^L_l
> $$
>
> where $\mu_l$ is the global mean representation of the LLM, which does not carry any meaning, $\mathbf{h}^{\text{abs}}_t(l)$ is the component which encodes the abstract meaning of the given concept, and $\mathbf{v}^L_l$ is the normalised direction vector encoding the given language $L$ with scale $\gamma$. This hypothesis is a combination of different findings in previous research.

> <a id="hyp-orthogonality"></a>**Hypothesis 2 (Orthogonality):** The specific directions used for encoding the language ($\mathbf{v}^L_l$) and the abstract meaning of concepts ($\mathbf{h}^{\text{abs}}_t(l)$) are close to being orthogonal to each other. Meaning, $\langle \mathbf{v}^L_l, \mathbf{h}^{\text{abs}}_t(l) \rangle \approx 0$.

[Hypothesis 2](#hyp-orthogonality) is supported by fundamental geometric properties of high-dimensional spaces and empirical findings in mechanistic interpretability. First, due to the blessing of dimensionality in vast latent spaces ($d \ge 4096$), any two independently learned features naturally tend toward near-orthogonality. Second, the linear representation hypothesis suggests that neural networks actively map disentangled, independent features, such as semantic payload and language identity, into orthogonal directions to minimize computational interference during forward passes ([Elhage et al., 2022](#elhage-2022); [Park et al., 2023](#park-2023)). Finally, the empirical success of concept erasure techniques, such as Iterative Null-space Projection ([Ravfogel et al., 2020](#ravfogel-2020)), provides concrete evidence for this separability. These methods successfully project specific attributes out of representations by zeroing out orthogonal subspaces without degrading the underlying abstract semantics, a mechanism that mathematically requires the target attribute and the semantic core to be largely orthogonal to begin with.

Finally, we will use this insight to make an inference-time modification to existing LLMs by using RepE (Representation Engineering) ([Bartoszcze et al., 2025](#bartoszcze-2025); [Zou et al., 2023](#zou-2023)). Our method, explained and derived in [The Proposed Method](#the-proposed-method-mathematical-derivation), is based on the idea that different tokens may require adjustments of different magnitudes in order to be maximally effective. Our method would be evaluated on a wide variety of languages on the following three benchmarks: MGSM ([Shi et al., 2023](#shi-2023)), XNLI ([Conneau et al., 2018](#conneau-2018)), and MSVAMP ([Chen et al., 2023](#chen-2023)). The closest related work to our method would be [Kirtane and Huang (2026)](#kirtane-2026), which will be used as a baseline, as it primarily differs in the fact that they apply the language steering uniformly to all tokens.

Our method is based on the idea that different tokens may require different amounts of steering, so we set it up as follows:

$$
\mathbf{h}^T_t(l) = \mathbf{h}^S_t(l) + \alpha^{(S,T)}(\mathbf{h}^S_t(l)) \cdot \mathbf{v}^{(S,T)}_l
$$

where $\alpha^{(S,T)}(\mathbf{h}^S_t(l))$ is a hidden-state-dependent scale factor.

## The Proposed Method (Mathematical Derivation)

Following our proposed language steering method, we want to find a mapping:

$$
\textbf{steer}_{(S,T)}: \mathbf{h}^S_t(l) \rightarrow \mathbf{h}^T_t(l)
$$

From our hypothesis, we know that the function must be of this shape:

$$
\textbf{steer}_{(S,T)}(\mathbf{h}^S_t(l)) = \mathbf{h}^S_t(l) + \alpha^{(S,T)}(\mathbf{h}^S_t(l)) \cdot \mathbf{v}^{(S,T)}_l
$$

The unknowns are: the function $\alpha^{(S,T)}(\mathbf{h}^S_t(l))$ and vector $\mathbf{v}^{(S,T)}_l$.

Consider the form of the input and output vectors:

$$
\mathbf{h}^S_t(l) = \mu_l + \mathbf{h}^{\text{abs}}_t(l) + \gamma \cdot \mathbf{v}^S_l
$$

$$
\mathbf{h}^T_t(l) = \mu_l + \mathbf{h}^{\text{abs}}_t(l) + \gamma \cdot \mathbf{v}^T_l
$$

To isolate $\gamma$, we mean-center the hidden state before computing the scalar product. Using [Hypothesis 2](#hyp-orthogonality) ($\mathbf{h}^{\text{abs}}_t(l) \cdot \mathbf{v}^S_l = 0$):

$$
(\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l = (\mathbf{h}^{\text{abs}}_t(l) + \gamma \cdot \mathbf{v}^S_l) \cdot \mathbf{v}^S_l = \mathbf{h}^{\text{abs}}_t(l) \cdot \mathbf{v}^S_l + \gamma \cdot \mathbf{v}^S_l \cdot \mathbf{v}^S_l = 0 + \gamma \cdot 1 = \gamma
$$

We can completely remove the source language component from a given hidden state by computing:

$$
\mathbf{h}^S_t(l) - ((\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l) \cdot \mathbf{v}^S_l = \mu_l + \mathbf{h}^{\text{abs}}_t(l) + \gamma \cdot \mathbf{v}^S_l - \gamma \cdot \mathbf{v}^S_l = \mu_l + \mathbf{h}^{\text{abs}}_t(l)
$$

Finally, we only need to add the appropriate direction for the target language:

$$
\mathbf{h}^S_t(l) - ((\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l) \cdot \mathbf{v}^S_l + ((\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l) \cdot \mathbf{v}^T_l = \mu_l + \mathbf{h}^{\text{abs}}_t(l) + \gamma \cdot \mathbf{v}^T_l = \mathbf{h}^T_t(l)
$$

Rewriting the function gives:

$$
\textbf{steer}_{(S,T)}(\mathbf{h}^S_t(l)) = \mathbf{h}^S_t(l) - ((\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l) \cdot \mathbf{v}^S_l + ((\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l) \cdot \mathbf{v}^T_l
$$

$$
\textbf{steer}_{(S,T)}(\mathbf{h}^S_t(l)) = \mathbf{h}^S_t(l) + ((\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l) \cdot (\mathbf{v}^T_l - \mathbf{v}^S_l)
$$

Therefore, we find that:

$$
\boxed{
\alpha^{(S,T)}(\mathbf{h}^S_t(l)) = ((\mathbf{h}^S_t(l) - \mu_l) \cdot \mathbf{v}^S_l)
\quad \text{and} \quad
\mathbf{v}^{(S,T)}_l = (\mathbf{v}^T_l - \mathbf{v}^S_l)
}
$$

## Finding the Language-Specific Directions

Instead of contrasting source language $S$ directly with target language $T$, we could contrast language $S$ against the concept of "language" itself.

* **Step 1:** Run a diverse dataset of sentences through the model in many different languages.

* **Step 2:** Calculate the average hidden state for each language.

* **Step 3:** Average all those language centroids together to find the "Global Multilingual Mean" ($\mu_{global}$). This point represents the abstract, language-agnostic center of the model's semantic space.

* **Step 4:** Define the global source language vector as $\mathbf{v}^S = \mu_{S} - \mu_{global}$, and the target language vector as $\mathbf{v}^T = \mu_{T} - \mu_{global}$.

## Experiments

The experiments which would be conducted in this study would include:

* **I.** Verifying the validity of our two hypotheses on the word-level scale by extracting word-pairs in parallel corpora by either using SimAlign ([Jalili Sabet et al., 2020](#jalilisabet-2020)) or awesome-align ([Dou & Neubig, 2021](#dou-2021)). See [docs/plans/EXP1.md](docs/plans/EXP1.md) for the detailed plan.

* **II.** Evaluating the method proposed in [The Proposed Method](#the-proposed-method-mathematical-derivation) on the aforementioned benchmarks and data.

* **III.** Conducting a thorough ablation study, which includes a method applying a constant-scale steering vector on the input sequence (similar to [Kirtane and Huang, 2026](#kirtane-2026)), using a random language direction, etc. This experiment will help attribute the performance of our method to the appropriate mechanisms.

## References

* <a id="bartoszcze-2025"></a>**Bartoszcze, L., Munshi, S., Sukidi, B., Yen, J., Yang, Z., Williams-King, D., Le, L., Asuzu, K., & Maple, C.** (2025). *Representation Engineering for Large-Language Models: Survey and Research Challenges*. <https://arxiv.org/abs/2502.17601>

* <a id="chen-2023"></a>**Chen, N., Zheng, Z., Wu, N., Gong, M., Zhang, D., & Li, J.** (2023). *Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations*. <https://arxiv.org/abs/2310.20246>

* <a id="conneau-2018"></a>**Conneau, A., Lample, G., Rinott, R., Williams, A., Bowman, S. R., Schwenk, H., & Stoyanov, V.** (2018). XNLI: Evaluating Cross-lingual Sentence Representations. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*. <https://arxiv.org/abs/1809.05053>

* <a id="dou-2021"></a>**Dou, Z.-Y., & Neubig, G.** (2021). Word Alignment by Fine-tuning Embeddings on Parallel Corpora. *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)*. <https://arxiv.org/abs/2101.08231>

* <a id="elhage-2022"></a>**Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., et al.** (2022). *Toy Models of Superposition*. <https://arxiv.org/abs/2209.10652>

* <a id="ethayarajh-2019"></a>**Ethayarajh, K.** (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 55–65.

* <a id="jalilisabet-2020"></a>**Jalili Sabet, M., Dufter, P., Yvon, F., & Schütze, H.** (2020). SimAlign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 1627–1643. <https://arxiv.org/abs/2004.08728>

* <a id="kirtane-2026"></a>**Kirtane, N., & Huang, K.-H.** (2026). *Language Steering for Multilingual In-Context Learning*. <https://arxiv.org/abs/2602.02326>

* <a id="park-2023"></a>**Park, K., Choe, Y. J., & Veitch, V.** (2023). *The Linear Representation Hypothesis and the Geometry of Large Language Models*. <https://arxiv.org/abs/2311.03658>

* <a id="ravfogel-2020"></a>**Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y.** (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 7237–7256.

* <a id="schut-2025"></a>**Schut, L., Gal, Y., & Farquhar, S.** (2025). *Do Multilingual LLMs Think In English?* <https://arxiv.org/abs/2502.15603>

* <a id="shi-2023"></a>**Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., Das, D., & Wei, J.** (2023). Language Models are Multilingual Chain-of-Thought Reasoners. *The Eleventh International Conference on Learning Representations (ICLR 2023)*. <https://arxiv.org/abs/2210.03057>

* <a id="zou-2023"></a>**Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D.** (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. <https://arxiv.org/abs/2310.01405>
