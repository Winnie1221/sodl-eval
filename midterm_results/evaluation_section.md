# Evaluation/Analysis

## Protocol

We evaluate every model with the same sampling protocol that was used to label problem difficulty, so that numbers are directly comparable across models and with the difficulty bins. For each test problem we draw n independent samples (temperature 0.6, top-p 0.95, at most 4,096 new tokens) with the same prompt template, and grade the last `\boxed{}` answer against the reference with math-verify. We use n = 16 for the base model and, to fit the compute budget before the check-in, n = 8 for the SFT checkpoints. The sampling seed (1) differs from the seed used for difficulty labeling (0), so the base-model numbers are fresh draws rather than the draws that defined the bins.

The test set has four parts of 200 problems each: the held-out test splits of the easy, medium and hard bins, plus 200 problems drawn at random from the never-solved bin. None of these problems are used for training.

From the number of correct samples c out of n for each problem we compute the unbiased pass@k estimator of Chen et al. (2021), pass@k = 1 − C(n−c, k)/C(n, k), averaged over the problems in a bin, for k = 1, 2, 4, 8. We report pass@8 as the main metric and pass@1 as the measure of single-sample accuracy. Pass@1 reflects how reliably a model solves a problem, while pass@8 reflects how many problems are within reach of the model at all; comparing the two tells us whether training sharpens the distribution on problems the model could already solve or extends the set of solvable problems. We also record the mean response length and the fraction of samples cut off at the token limit.

We evaluate two SFT models, one trained on the medium bin and one on the hard bin, both on solutions written by Qwen 8B. Validation loss is lowest after the first epoch (step 60), stays close to that minimum until step 120 and rises clearly in the third epoch (Figure 3, left), so the final checkpoints are overfit. We therefore evaluate checkpoints 60 and 120 and use checkpoint 120 as the main SFT result; the reason is given below.

## Results

**Table 1.** pass@1 and pass@8 on the four test bins. SFT models are at checkpoint 120.

| Test bin | Base pass@1 | SFT-medium pass@1 | SFT-hard pass@1 | Base pass@8 | SFT-medium pass@8 | SFT-hard pass@8 |
|---|---|---|---|---|---|---|
| easy   | 0.797 | 0.840 | 0.833 | 1.000 | 0.985 | 0.970 |
| medium | 0.397 | 0.431 | 0.418 | 0.930 | 0.840 | 0.850 |
| hard   | 0.126 | 0.172 | 0.191 | 0.593 | 0.525 | 0.590 |
| never  | 0.023 | 0.060 | 0.060 | 0.136 | 0.250 | 0.275 |

**Table 2.** Mean response length in tokens and fraction of samples truncated at 4,096 tokens.

| Test bin | Base | SFT-medium ckpt 60 | SFT-medium ckpt 120 | SFT-hard ckpt 60 | SFT-hard ckpt 120 |
|---|---|---|---|---|---|
| easy   | 496 / 1.3%   | 663 / 3.3%   | 648 / 2.4%   | 770 / 5.6%   | 667 / 2.8%   |
| medium | 949 / 6.6%   | 1641 / 19.4% | 1528 / 14.0% | 1842 / 24.6% | 1593 / 14.9% |
| hard   | 1147 / 11.5% | 2259 / 36.6% | 2173 / 29.9% | 2475 / 41.6% | 2132 / 27.3% |
| never  | 1166 / 11.2% | 2486 / 42.4% | 2362 / 33.9% | 2662 / 46.8% | 2366 / 32.3% |

Figure 1 shows the pass@k curves, Figure 2 compares pass@1 and pass@8 across models, and Figure 3 compares the checkpoints.

**The base model reproduces the difficulty bins.** With fresh samples the base model has pass@1 of 0.80 / 0.40 / 0.13 / 0.02 on the easy / medium / hard / never bins, so the bins measure a stable property of the problems rather than noise in the labeling draws. The never bin is not strictly unsolvable: 13.6% of these problems are solved within 8 samples (19.5% within 16), so "never" should be read as "solve rate below roughly 1/16".

**SFT raises single-sample accuracy on every bin.** At checkpoint 120 both SFT models have higher pass@1 than the base model on all four bins. The relative gain grows with difficulty: about +4 points on easy and medium, +5 to +7 points on hard (0.126 to 0.172 and 0.191), and the never bin goes from 0.023 to 0.060.

**SFT lowers pass@8 on problems the base model could already solve, and raises it on problems it could not.** On the easy and medium bins pass@8 drops (medium: 0.930 to 0.840 and 0.850), on the hard bin it is roughly unchanged for SFT-hard (0.593 to 0.590) and lower for SFT-medium (0.525), and on the never bin it roughly doubles (0.136 to 0.250 and 0.275). The pass@k curves of the SFT models therefore start above the base curve at k = 1 and cross below it between k = 2 and k = 8 on the easy, medium and hard bins (Figure 1). This is the pattern expected if SFT concentrates probability on one way of solving each problem: single-sample accuracy goes up, but the samples are less diverse, so extra samples help less. The never bin is the exception, where SFT brings new problems within reach: 32 (SFT-medium) and 35 (SFT-hard) of the never problems that the base model failed in all 16 samples are solved at least once in 8 samples.

**Training on hard problems is slightly better on hard test problems, but the gap is within noise.** SFT-hard is ahead of SFT-medium on the hard bin (pass@1 0.191 vs 0.172, pass@8 0.590 vs 0.525) and on the never bin (pass@8 0.275 vs 0.250), while SFT-medium is marginally ahead on easy and medium pass@1. With 200 problems per bin the standard error of pass@8 is about 0.035, so these differences are suggestive and not yet significant. At this point the difficulty of the SFT training problems has a much smaller effect than SFT itself.

**SFT doubles the response length, and truncation explains part of the lost coverage.** The SFT models imitate the long solutions of the Qwen 8B teacher: mean length on the hard bin goes from 1,147 to about 2,150 tokens, and 27% to 30% of hard-bin samples hit the 4,096-token limit before producing an answer (11.5% for the base model). A truncated sample is graded as wrong, so part of the drop in pass@8 on the easy, medium and hard bins is a budget effect rather than a reasoning effect. We will separate the two in the final report by re-grading with a larger token limit.

**Checkpoint choice matters more than validation loss suggests.** Checkpoints 60 and 120 have nearly the same validation loss (0.212 vs 0.213 for SFT-hard, 0.213 vs 0.217 for SFT-medium), but checkpoint 120 is better on every bin for both models, for both pass@1 and pass@8 (for example SFT-hard on the hard bin: pass@1 0.155 to 0.191, pass@8 0.520 to 0.590), and its responses are shorter and less often truncated (hard bin: 41.6% to 27.3% for SFT-hard). At checkpoint 60, pass@1 on the medium bin is still below the base model (0.36 vs 0.40). Token-level validation loss is therefore not a reliable proxy for sampled accuracy in this setting, and we select checkpoints by pass@k on held-out problems.

## Limitations

The SFT checkpoints are evaluated with n = 8 and the base model with n = 16; both give unbiased pass@k estimates for k ≤ 8, but the SFT estimates have higher variance. Runs of the final checkpoints (step 180) with n = 16 were interrupted by the cluster before finishing and are not reported. All numbers come from a single sampling seed.

## Next steps

1. Evaluate the final checkpoints and repeat with n = 16, so that pass@16 and the full checkpoint trajectory (60 / 120 / 180) are available.
2. Re-grade with a larger token limit, and report accuracy on non-truncated samples, to separate the length effect from the reasoning effect.
3. Train the RL models on the medium and hard bins and evaluate them with the same protocol. The main comparison is whether RL shows the same pattern as SFT (pass@1 up, pass@8 down on solvable problems) or keeps coverage.
4. Add per-problem transition analysis (problems gained and lost relative to the base model) at equal n, and a diversity measure over the sampled answers, to test directly whether SFT reduces sample diversity.
