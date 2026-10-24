# Pre-registered analysis plan: anomaly detection on SECOM

Written 2026-10-23, committed before the pass/fail label column was read into memory or summarized in this repo.

## 1. What was looked at before this plan was written

Only the following were read:
- `secom.names`, the dataset's own documentation. It states "1567 examples, 591 features, 104 fails". That fail count is label-derived, so the author knows the overall fail rate (104 / 1,567 = 6.6%). The split of fails between train and test was not computed.
- The timestamp column of `secom_labels.data` (column 2), read with `usecols=[1]`. Label column 1 was never loaded.
- The shape of `secom.data` (1,567 x 590), and two counts over all 1,567 rows: 28 columns are more than 50% missing and 116 are constant. These were descriptive. The drop rules in section 5 are fitted on train rows only and will give different counts.

What this blinding does and does not cover: it is procedural. The author has likely seen published SECOM results before, and those mostly report weak signal. That is why section 3 states a null as the expected outcome.

## 2. Data

- **Source.** UCI Machine Learning Repository, dataset 179 (McCann and Johnston, donated 2008-11-19).
  - URL: `https://archive.ics.uci.edu/static/public/179/secom.zip`
  - Stored in `data/raw/` (gitignored). The analysis must check these SHA-256 hashes first and stop on a mismatch:
    - `secom.data` (5,389,983 bytes): `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c`
    - `secom_labels.data` (40,638 bytes): `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03`
- **Unit.** One row is one production entity (a run) with 590 anonymized sensor or process measurements. Missing values are the string `NaN`.
- **Label.** In-house line test result: `-1` pass, `1` fail. (`secom.names` writes the pass code as ".1"; the file uses -1.) Fail is the positive class.
- **Timestamps.** 2008-07-19 11:55 to 2008-10-17 06:07, 86 calendar days, non-decreasing in file order, 33 duplicated timestamps.

## 3. Questions and hypotheses

The business question: if a fab can afford to inspect only a fixed fraction of runs, does ranking runs by an anomaly score from the sensor data catch more failing runs than inspecting at random?

- **Q1 (primary).** An unsupervised Isolation Forest, trained only on passing runs from the earlier period, ranks later failing runs above later passing runs better than chance.
- **Q2.** A supervised L2 logistic regression does better than chance.
- **Q3 (headline comparison).** The Isolation Forest and the logistic regression differ. Two-sided, with no favoured direction.
- **Q4.** The Isolation Forest beats a naive univariate control-chart rule (section 6, M4). This asks whether a multivariate model adds anything over what a fab already does.
- **Secondary detector.** PCA reconstruction error (the SPE or Q statistic, the standard multivariate process-monitoring score) does better than chance.

**Stated expectation.** Published SECOM baselines use random 10-fold cross-validation and report balanced error rates around 33% to 40%. A temporal split removes the optimism of random splits. I expect test PR-AUC between 0.06 and 0.20 for every model and a null on most contrasts. A null is a reportable result.

## 4. Split (evaluation protocol)

Temporal holdout, never random.
- **Train:** timestamp < 2008-09-24 00:00:00. 1,046 runs (66.8%), 2008-07-19 to 2008-09-23.
- **Test:** timestamp >= 2008-09-24 00:00:00. 521 runs (33.2%), 2008-09-24 to 2008-10-17.
- The cutoff is at midnight, so no calendar day is split across train and test.
- The test set is scored once, after every model and preprocessing parameter is frozen. No threshold, hyperparameter or feature choice is made on test rows.
- Hyperparameter selection, where any is done (M3 only), uses forward-chaining folds inside train.

## 5. Preprocessing (fitted on train rows only)

Applied in this order by one `Preprocessor` object whose `fit` sees only the rows it is given:
1. Drop every feature that is more than 50% missing in the fitting rows.
2. Impute remaining missing values with the fitting-row median of that feature.
3. Drop every feature whose fitting-row standard deviation is 0 after imputation (constant features).
4. Standardize with the fitting-row mean and standard deviation.

The preprocessor is fitted on all train rows (it does not use labels). For M3's inner cross-validation it is refitted inside each inner training fold. A pytest test must fail if any fitted preprocessing parameter changes when test rows are altered.

## 6. Models (hyperparameters fixed here, not tuned on test)

Every model outputs a score where higher means "more likely to fail".

- **M1 Isolation Forest (primary detector).** `n_estimators=500`, `max_samples=256`, `max_features=1.0`, `random_state=20261023`. Fitted on train runs labelled pass. Score = minus `score_samples`.
- **M2 PCA-SPE (secondary detector).** PCA on standardized train passing runs, keeping the smallest number of components that explains at least 90% of variance. Score = squared reconstruction error.
- **M3 L2 logistic regression (supervised baseline).** `lbfgs`, `max_iter=5000`, no class weighting (so probabilities stay on the base-rate scale). C is chosen from {0.0001, 0.001, 0.01, 0.1, 1} by mean average precision over `TimeSeriesSplit(n_splits=4)` inside train, with the preprocessor refitted in each fold. Folds whose validation part has no fails are skipped. Ties go to the smaller C. Refit on all train rows with the chosen C. Score = predicted probability of fail.
- **M4 naive control-chart rule.** For each retained feature, Shewhart limits at mean plus or minus 3 SD of the train passing runs (in standardized units). Score = number of features outside their limits.
- **Reference: no-skill.** A random ranking. Its expected PR-AUC equals the test fail rate and its expected recall at budget k equals k / n. For probabilities, the no-skill forecast is the train fail rate as a constant.

## 7. Metrics (test set)

- **Primary: PR-AUC**, computed as average precision (`sklearn.metrics.average_precision_score`).
- **Co-primary: recall at a 5% inspection budget.** Flag the top k = ceil(0.05 n) test runs by score (k = 27 when n = 521). Recall = fails flagged / all fails. Ties straddling the cutoff get fractional credit (the expected value under random tie-breaking).
- **Calibration (M3 vs the constant train fail rate):** Brier score, Brier skill score (BSS = 1 minus Brier_M3 / Brier_naive), calibration in the large (mean predicted minus observed fail rate), and ECE with 10 equal-frequency bins (descriptive only).
- **Descriptive only:** ROC-AUC and precision at the 5% budget.

## 8. Inference and multiplicity

- **Paired bootstrap** over test runs: B = 2,000, `numpy.random.default_rng(20261023)`. Every model is scored on the same resampled indices. Resamples with zero fails are redrawn, and the number of redraws is reported.
- **Delta CI:** 95% percentile interval of the bootstrap deltas.
- **p-value:** two-sided Wald, z = observed delta / bootstrap SD of the delta.
- **Primary family (12 tests), Holm at familywise alpha = 0.05.** Six contrasts, each on PR-AUC and on recall at 5%:
  - C1: M1 IF minus no-skill
  - C2: M3 logistic minus no-skill
  - C3: M1 IF minus M3 logistic
  - C4: M1 IF minus M4 control-chart rule
  - C5: M4 control-chart rule minus no-skill
  - C6: M2 PCA-SPE minus no-skill
- **Calibration family (2 tests), Holm:** Brier_M3 minus Brier_naive, and calibration in the large of M3 (against 0).
- **Sensitivity (not a claim):** day-block bootstrap. Resample test calendar days with replacement (same number of days), keep every run in a drawn day, and recompute the 12 primary CIs and p-values. This checks whether within-day dependence makes the run-level bootstrap too narrow.
- **Reporting rule.** Every contrast goes in one table with the estimate, 95% CI, raw p and Holm-adjusted p, significant or not.

## 9. Feature-engineering and pipeline layer (descriptive, does not feed the models)

- **Data contract checks** (stop on failure): hashes match, 1,567 rows, 590 numeric feature columns, labels in {-1, 1}, timestamps parse and are non-decreasing.
- **Missingness profile:** per-feature missing rate in train and test, the number of features removed by each drop rule, and the per-run count of missing values over time.
- **Drift of sensor means over time:** for every retained feature, the standardized mean shift (test mean minus train mean) / train SD; the share of features with |shift| > 0.5; and weekly means of the five most shifted features. Feature drift reads test features but not test labels and is not used to fit or select anything.

## 10. Decision rules and what would falsify each claim

- **Q1, Q2, Q4 and the secondary detector:** a contrast is supported only if its Holm-adjusted p < 0.05 and its 95% CI excludes 0. Otherwise the claim is not supported at this power. A significant negative delta is reported as worse than the comparator.
- **Q3:** "unsupervised beats supervised" needs C3 positive and significant on PR-AUC; "supervised beats unsupervised" needs it negative and significant. Anything else is "no detectable difference", which is not the same as equivalence.
- **Operational rule, fixed in advance.** A model is "worth piloting" only if its recall at 5% has a lower 95% bootstrap bound above 0.05 (beats random inspection) and a point estimate of at least 0.15 (three times random).
- **Calibration rule.** M3's probabilities are usable as probabilities only if BSS > 0 with the Brier-delta CI excluding 0. Otherwise M3 is used as a ranking only.
- **What would falsify the expected null.** Any model with a PR-AUC lower 95% bound above 0.20.
- **Leakage stop rule.** Any test PR-AUC above 0.50 or recall at 5% above 0.50 triggers a leakage audit before the number is reported as real.

## 11. Power, roughly

The test fail count is unknown at plan time. At the documented overall rate of 6.6%, about 34 of 521 test runs fail. With 34 fails, recall at 5% moves in steps of about 0.03. Under the null its SE is about sqrt(0.05 x 0.95 / 34) = 0.037. Detecting it at the Holm worst case (alpha = 0.05 / 12, two-sided) with 80% power needs a true recall near 0.22, about four times random. PR-AUC intervals will likely be 0.10 or more wide. Small real effects will be missed; that is a design limit, not a finding.

## 12. Not done

No oversampling (SMOTE or similar), no feature selection beyond the two drop rules, no tuning of M1, M2 or M4, no threshold tuning on test, no other model families (no gradient boosting), no other budgets than 5%. Anything else that gets run is labelled post hoc in the report and is still reported.

## 13. Known risks and leakage paths

- **Preprocessing leakage.** Controlled by the train-only `fit` and the leakage test (section 5).
- **Tuning leakage.** Only M3 is tuned, and only on forward-chaining folds inside train.
- **Drift.** The test period may differ from train. That is the realistic deployment setting, and section 9 measures it, but it can push every model toward chance.
- **Label delay.** In production a fail label arrives after the run. This analysis assumes every train label is known at the cutoff.
- **Dependence.** Runs close in time may be correlated. The run-level bootstrap may be too narrow; the day-block sensitivity checks this.
- **Anonymized features.** Sensor names are hidden, so findings cannot be tied to physical equipment.
- **One holdout.** One temporal split is one realization. Results may change with a different cutoff; no other cutoff is tried.
- **Forking paths.** The split, models, hyperparameter grid, seeds, B, metrics and families are all fixed above. Any deviation is listed below with its reason.

## Deviations

None yet. Deviations are appended here after the plan commit, dated, with the text above left unchanged.

### 2026-10-23, after the first full run (plan commit 3679a5a)

1. **C2 recall at 5% is a bootstrap boundary artifact, reported as the rule says.** The logistic model caught 0 of 27 test fails in its top 27 runs. Every bootstrap resample therefore also gives it recall 0, and the plan defined the no-skill recall as its expectation k / n, which barely varies across resamples. The bootstrap SD of the delta is 0.004, so the Wald p is about 2e-33 and the pre-registered rule marks C2 "supported" as worse than random. I do not believe it. Neither side of the contrast carries the sampling variance of random inspection itself. Post hoc check: under random inspection of 27 of 521 runs with 27 fails, catching 0 has probability 0.23, and the exact two-sided hypergeometric p is 0.39. The honest reading is "no detectable difference from random", and the plan's construction of the no-skill recall comparator was a design flaw.
2. **Post hoc exact test for every model's recall at 5%.** Added `random_inspection_p` (hypergeometric, two-sided) for all four models, in `outputs/posthoc_exact_recall_and_glitch.csv`. It is labelled post hoc and is not part of either Holm family.
3. **Post hoc robust drift.** The three largest mean shifts (f140 at 233 SD, f275 at 184 SD, f004 at 23 SD) each come from about four extreme test readings (up to 9,999); the test medians match the train medians. Added `median_shift_sd` (test median minus train median, train SD units) next to the pre-registered mean shift, and a third figure panel of weekly medians for the five largest median shifts. The pre-registered mean-shift columns and panel are unchanged.
4. **Post hoc glitch check.** Counted test runs with any retained feature more than 10 train SDs from the train mean (66 of 521, 4 of them fails), and how many of each model's top 27 are such runs. Labelled post hoc; it explains the detector results but is not a test.
5. **Logistic C landed on the grid edge.** Inner forward-chaining CV chose C = 0.0001, the smallest value in the pre-registered grid. The grid was not extended, because the plan fixed it. At that C the model is heavily shrunk toward the base rate, which is why its Brier score is almost the naive one.
6. **Calibration rule fires, but the effect is negligible.** The Brier delta CI excludes 0 (Holm p = 0.049) and BSS = 0.0026, so the pre-registered rule says the probabilities are usable. A 0.26% Brier improvement over a constant is not practically useful. Calibration in the large is also off: mean prediction 7.1% against an observed 5.2%, because the fail rate fell from 7.4% in train to 5.2% in test.
7. **Drop-rule counts on train.** On train rows, 24 features were dropped for more than 50% missing and 122 as constant (the plan-stage all-row counts were 28 and 116, as section 1 expected). 444 features were retained.
8. **Additions that are not analyses.** Two figures, `outputs/summary.json`, and `outputs/logistic_inner_cv.csv`. Peak RAM of the full run was 222 MB.

### 2026-10-23, correction to the "supported" rule (after the first run)

9. **C2 recall "supported" was a code bug, fixed at the root.** `primary_table` in `src/detect.py` set `supported` from Holm p and CI alone, so C2 recall at 5% (logistic 0.000 vs no-skill 0.052, i.e. worse than random) was marked supported on a degenerate Wald p of 2e-33. Two changes, both applied to the primary and day-block tables:
   - **Direction.** A contrast is now supported only if Holm p < 0.05, the CI excludes 0, *and* the delta has the sign section 3 hypothesises: positive for C1, C2, C4, C5, C6; either sign for C3 (two-sided, section 10). A significant negative delta is "worse than the comparator", not support.
   - **Constant reference.** The no-skill recall is the constant k / n, so the four recall-vs-no-skill rows (C1, C2, C5, C6) now take their raw p from the exact two-sided hypergeometric random-inspection test of deviation 2 (column `test` = `exact_hypergeom`) instead of the bootstrap Wald p. These exact p-values enter the 12-test Holm family. An exact binomial test on the hits (27 trials, p = 27/521) gives nearly the same values (0.40, 1.00, 0.65 for 0, 1, 2 hits), so the choice does not matter here.
   - **What moved.** Raw p for recall C1 0.695 -> 1.000, C2 2e-33 -> 0.387, C5 0.670 -> 0.644, C6 0.623 -> 1.000 (day-block table: 0.710, 1.5e-27, 0.783, 0.608 -> the same exact values). C2 Holm p 2e-32 -> 1.0 and `supported` True -> False in both tables. Every other estimate, CI, Holm p, model metric, calibration test, figure and post hoc number is unchanged. No primary contrast is supported. Deltas and bootstrap CIs are still reported as before; the C2 recall CI stays degenerate (-0.052, -0.052) and should not be read as precision.
   - **Test.** `test_significant_delta_in_wrong_direction_is_not_supported` in `tests/test_detect.py`.

### 2026-10-23, correction to the no-skill PR-AUC reference (after results were seen)

10. **The no-skill PR-AUC reference was wrong, corrected post hoc.** Section 6 says a random ranking's expected PR-AUC equals the test fail rate (0.052). It does not when fails are rare: the exact expected average precision of a random ranking of n = 521 runs with m = 27 fails is (H_n + (n - H_n)(m - 1)/(n - 1)) / n = 0.0625, where H_n is the n-th harmonic number. An independent reviewer found this, and a simulation of 20,000 random rankings gave a mean AP of 0.0625. Against the fail rate, the four PR-AUC-vs-no-skill contrasts (C1, C2, C5, C6) showed excesses of +0.006 to +0.013; those excesses were an artifact of the wrong reference. This correction was made after the results were seen. It is more conservative than the original reference, and it changes no conclusion (nothing was supported before or after).
   - **Test.** The four PR-AUC-vs-no-skill rows now use a one-sided label-permutation test (column `test` = `label_perm`): the model's scores are held fixed, the test labels are permuted 10,000 times (`numpy.random.default_rng(20261023)`), and p = (1 + number of permuted APs >= observed AP) / (1 + 10,000). Ties in the scores are kept, so the reference is right for the 3-sigma rule count too. The reference (column `ref`) is the mean permuted AP: IF 0.0628, logistic 0.0628, 3-sigma rule 0.0618, PCA-SPE 0.0627. Delta = observed AP minus that reference. The CI is the paired bootstrap of the model's AP minus the exact random-ranking expectation for each resample. These p-values stay in the same 12-test Holm family. C3 and C4 keep the paired-bootstrap Wald p; the four recall-vs-no-skill rows keep the exact hypergeometric p of deviation 9.
   - **What moved (primary table).** C1 PR-AUC delta +0.011 -> +0.0002, CI (-0.003, 0.050) -> (-0.014, 0.040), raw p 0.423 -> 0.372. C2 +0.008 -> -0.003, CI (-0.0003, 0.030) -> (-0.011, 0.020), p 0.317 -> 0.466. C5 +0.013 -> +0.003, CI (-0.003, 0.073) -> (-0.013, 0.062), p 0.498 -> 0.295. C6 +0.006 -> -0.005, CI (-0.004, 0.032) -> (-0.015, 0.022), p 0.545 -> 0.530. Holm p stays 1.0 for all 12; none supported. In the day-block table the same four CIs moved to (-0.014, 0.044), (-0.011, 0.019), (-0.015, 0.098), (-0.014, 0.020); the C2 day-block CI no longer excludes 0 (it was 0.0005 to 0.030). The no-skill row of `model_metrics.csv` now reports PR-AUC 0.0625 (the random-ranking expectation; CI 0.045 to 0.081) instead of 0.052 (CI 0.035 to 0.071). Model APs, recall rows, calibration tests and post hoc outputs are unchanged. The PR-curve figure's dashed line is relabelled "prevalence", which is the precision baseline of a random ranking, not its expected AP.
   - **Reading.** Every model's PR-AUC (0.058 to 0.065) is within 0.006 of what a random ranking scores, and none is significantly above it; IF is +0.0002 above it, the 3-sigma rule +0.003, and PCA-SPE and logistic are below it.
   - **Test.** `test_random_scorer_ap_is_not_supported_and_reference_exceeds_prevalence` in `tests/test_detect.py`.
   - **Note on deviation 9.** Deviation 9 was also a post hoc change made after results were seen, and it changed the test used for four rows inside the pre-registered primary family (and the "supported" rule), not only a diagnostic outside it.
