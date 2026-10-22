# ICL evaluation

Evaluation harness for TODO item 2 in the [project README](../../README.md): the English few-shot, multilingual few-shot (MFS), constant-steering and oracle baselines of [Kirtane & Huang (2026)](../../README.md#kirtane-2026), with the same compute / val / test splits.

```bash
# 1. language vectors, per benchmark, from the compute split -> data/language_vectors/<model>/<benchmark>.npz
slurm/submit.sh -J icl_vectors -- python src/icl/vectors.py
# 2. one method on one benchmark: tune on val if needed, then test -> src/icl/results/<model>/<benchmark>/<method>/
slurm/submit.sh -J icl_en_xnli -- python src/icl/run.py --benchmark xnli --method en
slurm/submit.sh -J icl_steer_xnli -t 24:00:00 -- python src/icl/run.py --benchmark xnli --method steer
# the same, with the val grid split over 8 GPUs (--shard I/N), then a job that selects the config and tests it
for i in $(seq 0 7); do slurm/submit.sh -J icl_steer_xnli_s$i -t 02:00:00 -- python src/icl/run.py --benchmark xnli --method steer --shard $i/8; done
slurm/submit.sh -J icl_steer_xnli --dependency afterok:<shard job ids> -- python src/icl/run.py --benchmark xnli --method steer
```

| File | Contents |
|---|---|
| `tasks.py` | Benchmarks: splits, demonstration pool, prompt text, answer parsing |
| `model.py` | `Runner`: prompt encoding with per-token role and language, steering hooks, forward pass and greedy generation |
| `steering.py` | Steering functions `(layer, h, batch) -> h`; `AddVector` is the constant shift |
| `methods.py` | `Method` base class and one subclass per method: demonstration languages, steering, hyperparameter grid |
| `vectors.py` | Precomputes per-language mean states; `LanguageVectors.direction(S, T, layer)` = μ_T − μ_S |
| `run.py` | Harness: tunes on val when the method has a grid, logs the choice, then runs test |

**Adding a method:** subclass `Method`. Set `demo_langs` (the language of each demonstration) and, if it steers, `steering(config, target)`, `grid` and `fallback`. Then register it in `METHODS`. A new kind of intervention is a new `Steering` subclass. It can read `batch.mask(...)`, `batch.role` and `batch.lang` to decide which tokens to change. For example, MFS + steering would add v_(L_p→T) at each demonstration token p, according to that token's own language.

## Protocol

These follow Kirtane & Huang (2026) unless marked as a deviation.

* **Target languages:** every language of the benchmark except English (XNLI 14, MGSM 10, MSVAMP 9). Hyperparameters are chosen on the val accuracy averaged over all of them (`--tune-scope global`).

* **Prompt:** instruction, then 6 demonstrations, then the query, as one user turn in the chat template. The model's response is the assistant turn. Field names and labels are English.
* **Demonstrations:**
  * XNLI: compute-split items.
  * MGSM and MSVAMP: 6 of the 8 MGSM exemplars, with English CoT in every condition.
  * The demonstration items depend only on (seed, query item), so all methods see the same items and their results are paired.
  * MFS draws each demonstration's language uniformly from the benchmark's languages other than the target, English included.
  * *Deviation:* the paper takes English demonstrations from GSM8K train or MNLI train. Our pools are parallel, so the oracle can use the same items translated.
* **Vectors** (Eqs. 1–4): for each compute item, one 6-demonstration sample is made per language: the item plus 5 items drawn with replacement, with the same items in every language. Each sample is mean-pooled over its tokens (BOS excluded) at every layer, and v(t) = μ_T(t) − μ_en(t).
  * *Deviation:* compute items have no CoT, so math samples contain only Question and Final answer.
* **Steering** (Eq. 6): h ← h + α·v at one layer t, the output of block t. It is applied to the prompt pass only, so generated tokens are not steered.
  * Grid: layer ∈ {5, 10, 15, 20, 25, 30} × α ∈ {0.5, 1, 2, 3} × positions ∈ {demos, boundary, query, prompt} (the paper's on_fewshot, after_fewshot, on_question, entire). That is 96 configs, plus the unsteered fallback.
  * A config is chosen only if it beats the fallback on val.
  * Selection uses the mean over languages by default (`--tune-scope global`); `per_lang` reproduces the paper's main tables.
* **MFS + steering** (`mfs_steer`, TODO 4): MFS demonstrations, and every token of demonstration p gets h ← h + α·v_(L_p→T) = α·(μ_T − μ_(L_p)) at one layer, with L_p that demonstration's own language. English demonstrations therefore get v_(en→T), as in steer. Instruction, query and template tokens are not steered.
  * Grid: layer ∈ {5, 10, 15, 20, 25, 30} × α ∈ {0.5, 1, 2, 3}, demos only. The fallback α = 0 is plain MFS.
  * Controls, same grid: `mfs_random` replaces each row v_(L_p→T) by a random direction of the same norm, one per (draw, layer, L_p) and shared across targets. `mfs_wrong` replaces it by v_(L_p→L′), rescaled to the norm of v_(L_p→T), with L′ drawn per (draw, target) as in `wrong`; demonstrations already in L′ are not steered, and `steer_lang` fixes L′ for the transfer matrix. `mfs_uniform` gives every demonstration v_(en→T).
* **Scoring:** greedy generation by default.
  * XNLI is scored on the first word of the response. `--mode forward` instead takes the most likely label by its first token in a single forward pass, and keeps the label log-probabilities (for TODO 4).
  * Math is scored on the number after the last "Final answer:".
* **Model dtype:** bf16. Vectors are accumulated in fp32.
