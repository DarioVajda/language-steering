# Results: language steering for multilingual ICL (Llama-3.1-8B-Instruct)

The setup and methods are described in [README.md](README.md). Each accuracy is the test accuracy, averaged first over seeds 0–4 for each item and then over the languages with equal weight (MGSM has 10 languages, MSVAMP 9, XNLI 14). Brackets give 95% CIs from a paired bootstrap with 2000 resamples, stratified by language. Δ is the paired difference from the base method: en for steer, mfs for mfs_steer.

## Math: MGSM and MSVAMP

Generation is capped at 2048 tokens. Numbers are read with the locale-aware parser and responses are cut with the digit-aware stop rule.

| Method | MGSM | MSVAMP |
|---|---|---|
| en | .712 [.686, .738] | .737 [.724, .750] |
| mfs | .695 [.670, .720] | .730 [.717, .743] |
| oracle | .709 [.683, .735] | .740 [.727, .753] |
| steer | .711, Δ −.000 [−.009, +.008] | .738, Δ +.001 [−.005, +.007] |
| mfs_steer | .682, Δ −.013 [−.025, −.001] | .727, Δ −.004 [−.009, +.001] |
| steer controls (random, wrong, wrong_transfer) | .701–.718 | .733–.744 |
| mfs_steer controls (random, wrong, fixed, transfer)¹ | .675–.704 | .726–.735 |

¹ Excludes mfs_wrong_transfer_te on MGSM (.611; see below).

**Findings**

* **Steering has no effect on math accuracy.** steer matches en on both benchmarks, and every control (random direction, wrong-language vector, transfer) lands within about a point of its base method.
* **There is no headroom to recover.** English demos already do as well as target-language demos: oracle − en is −.003 [−.017, +.012] on MGSM and +.003 [−.005, +.010] on MSVAMP.
* **mfs_steer is slightly worse than mfs on MGSM**, by 1.3 points. The loss roughly matches its higher capped share (5.3% vs 3.1%). On MSVAMP the difference is not significant.
* **Steering changes the response language, but accuracy does not follow.** By fastText, mfs_steer cuts English responses on MGSM from 14% to 2%, with no gain in accuracy.

**Sanity checks**

* **Capped responses:** 4.3% on MGSM and 5.0% on MSVAMP, over all runs. About 99% of them are repetition loops rather than reasoning that got cut off. A few of the rest are counting runs, which are also degenerate.
* **Outlier:** on MGSM, mfs_wrong_transfer_te (every query steered with the Telugu vector) drops Swahili to .22, with 73% of its responses capped. No other run comes close.
* **Fallback:** tuning for MGSM mfs_wrong_draw2 chose α = 0, so that run is identical to mfs. The mfs_wrong control on MGSM therefore has two real draws, not three.
* **Response language:** the language ID is a script and stopword heuristic. It agrees with fastText on 95% of responses.
  * en and steer: 94–98% of responses are in the target language.
  * oracle: drifts into English, most on MSVAMP, with Russian worst at 42% English and Swahili at 24%.
  * mfs: about 14% English, mostly in bn, te, ja and zh.

## Comparison: XNLI

These runs use generate mode with 5 seeds. For en, mfs, oracle and steer, the seed-0 directory is merged with its `_seeds1-4` directory. Δ is measured against en.

| Method | XNLI |
|---|---|
| en | .598 [.588, .609] |
| oracle | .626 [.616, .637] |
| steer | .619, Δ +.021 [+.015, +.025] |
| wrong (3 draws) | .614–.617, Δ +.016 to +.019 |
| random (3 draws) | .603–.607, Δ +.005 to +.009 |
| mfs / mfs_steer | .624 / .627 |
| mfs_steer controls (random, wrong) | .624–.633 |

* **Most of steering's gain is not specific to the language.** A wrong-language vector recovers about 80% of steer's +2.1 points. Most single-language transfer vectors (wrong_transfer_*, .617–.624, apart from de and sw) match steer outright.
* **mfs_steer does not beat its controls.** The random and wrong-language controls score as high or higher.
