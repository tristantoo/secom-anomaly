# Report: anomaly detection on SECOM under a temporal split

Every number below is read from `outputs/` (run of 2026-10-23, after the C2 and no-skill PR-AUC corrections). Plan: `docs/PLAN.md`, committed as 3679a5a before labels were read.

## Result

**A null.** On the 521 later test runs (27 fails, fail rate 0.052), no model ranks failing runs detectably better than random inspection. PR-AUC ranges from 0.058 (PCA-SPE) to 0.065 (3-sigma rule count) against 0.062 to 0.063 for a random ranking (label-permutation reference), so the models sit within 0.006 of random and two fall below it. Every one of the 12 pre-registered primary contrasts has Holm-adjusted p = 1.0. At a 5% inspection budget (27 runs) the models catch 0 to 2 of the 27 fails, where random inspection expects 1.4. No model meets the pre-registered "worth piloting" rule. No PR-AUC lower bound reaches 0.20, so the expected null is not falsified, and the leakage stop rule did not fire. The anomaly detectors mostly flag runs with sensor glitches, not failing runs (post hoc). This is the result the plan predicted (section 3). The design can detect only large effects (section 11), so "no detectable effect" does not mean "no effect".

## Evaluation protocol

- **Split.** Temporal holdout at 2008-09-24 00:00. Train: 1,046 runs (77 fails, rate 0.074), 2008-07-19 to 2008-09-23. Test: 521 runs (27 fails, rate 0.052), 2008-09-24 to 2008-10-17, 24 calendar days. The test set was scored once, after every parameter was frozen.
- **Preprocessing, train rows only.** Drop features more than 50% missing (24 features), median impute, drop constant features (122), standardize; 444 features retained. A pytest test fails if test rows change any fitted parameter.
- **Models.** M1 Isolation Forest and M2 PCA-SPE (122 components, 90% variance), both fitted on passing train runs. M3 L2 logistic regression, with C chosen by forward-chaining `TimeSeriesSplit(4)` inside train; it chose C = 0.0001, the edge of the grid. M4 counts the features outside the 3-sigma limits of passing train runs. A no-skill reference completes the set.
- **Metrics.** PR-AUC (average precision) is primary. Recall at a 5% budget (k = 27) is co-primary. Brier, BSS and calibration in the large are reported for M3.
- **Inference.** Paired run-level bootstrap, B = 2,000, seed 20261023, 0 zero-fail redraws, with 95% percentile CIs. The model-vs-model rows (C3, C4) use the Wald p from the bootstrap SD. The four recall-vs-no-skill rows use the exact hypergeometric random-inspection p (see the C2 correction). The four PR-AUC-vs-no-skill rows use a one-sided label-permutation test, 10,000 permutations, seed 20261023 (see the no-skill PR-AUC correction). Holm is applied over the 12-test primary family and, separately, over the 2-test calibration family. A day-block bootstrap is the sensitivity check.

## Model metrics (test set, 95% paired bootstrap CIs)

| Model | PR-AUC | 95% CI | Recall at 5% | 95% CI | Fails caught (of 27) | ROC-AUC |
|---|---|---|---|---|---|---|
| No-skill | 0.062 (random-ranking expectation) | 0.045 to 0.081 | 0.052 (k/n) | constant | 1.4 expected | 0.500 |
| Isolation Forest | 0.063 | 0.039 to 0.110 | 0.037 | 0.000 to 0.125 | 1 | 0.546 |
| PCA-SPE | 0.058 | 0.037 to 0.093 | 0.037 | 0.000 to 0.097 | 1 | 0.541 |
| L2 logistic | 0.060 | 0.039 to 0.092 | 0.000 | 0.000 to 0.000 | 0 | 0.585 |
| 3-sigma rule count | 0.065 | 0.040 to 0.130 | 0.074 | 0.000 to 0.191 | 2 | 0.538 |

ROC-AUC is descriptive only. The no-skill expected catch is 27 x 27 / 521. The no-skill PR-AUC is the exact expected average precision of a random ranking of 521 runs with 27 fails (0.0625), which exceeds the fail rate 0.052.

## Primary contrasts (12 tests, Holm)

Delta = metric(a) minus metric(b); for PR-AUC against no-skill, b is the model's mean AP over 10,000 label permutations (IF 0.0628, logistic 0.0628, 3-sigma rule 0.0618, PCA-SPE 0.0627). The hypothesised sign is + for every contrast except C3, which is two-sided.

| Contrast | Metric | Delta | 95% CI | Raw p | Test | Holm p | Supported |
|---|---|---|---|---|---|---|---|
| C1 IF - no-skill | PR-AUC | +0.0002 | -0.014 to 0.040 | 0.372 | label permutation | 1.0 | no |
| C2 logistic - no-skill | PR-AUC | -0.003 | -0.011 to 0.020 | 0.466 | label permutation | 1.0 | no |
| C3 IF - logistic | PR-AUC | +0.003 | -0.023 to 0.041 | 0.835 | bootstrap Wald | 1.0 | no |
| C4 IF - 3-sigma rule | PR-AUC | -0.002 | -0.038 to 0.022 | 0.884 | bootstrap Wald | 1.0 | no |
| C5 3-sigma rule - no-skill | PR-AUC | +0.003 | -0.013 to 0.062 | 0.295 | label permutation | 1.0 | no |
| C6 PCA-SPE - no-skill | PR-AUC | -0.005 | -0.015 to 0.022 | 0.530 | label permutation | 1.0 | no |
| C1 IF - no-skill | Recall at 5% | -0.015 | -0.052 to 0.073 | 1.000 | exact hypergeometric | 1.0 | no |
| C2 logistic - no-skill | Recall at 5% | -0.052 | -0.052 to -0.052 | 0.387 | exact hypergeometric | 1.0 | no |
| C3 IF - logistic | Recall at 5% | +0.037 | 0.000 to 0.125 | 0.329 | bootstrap Wald | 1.0 | no |
| C4 IF - 3-sigma rule | Recall at 5% | -0.037 | -0.154 to 0.033 | 0.430 | bootstrap Wald | 1.0 | no |
| C5 3-sigma rule - no-skill | Recall at 5% | +0.022 | -0.052 to 0.139 | 0.644 | exact hypergeometric | 1.0 | no |
| C6 PCA-SPE - no-skill | Recall at 5% | -0.015 | -0.052 to 0.045 | 1.000 | exact hypergeometric | 1.0 | no |

The C2 recall CI is degenerate: logistic recall is 0 in every resample and no-skill recall is a constant. It is not evidence of precision.

## Significance

- **Primary family:** nothing is significant. Q1 (IF better than chance), Q2 (logistic better than chance), Q4 (IF beats the control-chart rule) and the secondary detector (PCA-SPE) are not supported at this power. Q3 (IF vs logistic) shows no detectable difference, which is not the same as equivalence.
- **Day-block sensitivity:** the conclusion is the same. All 12 Holm p-values are 1.0 and none is supported. Every day-block CI includes 0 except the degenerate C2 recall one.
- **Calibration family (M3):** Brier delta -0.00013 (CI -0.00024 to -0.00001, Holm p = 0.049), BSS = 0.0026. The pre-registered rule marks the probabilities "usable", but a 0.26% Brier improvement over a constant is not practically useful. Calibration in the large is +0.019 (CI 0.0002 to 0.037, Holm p = 0.049): the mean prediction is 0.071 against an observed 0.052, because the fail rate fell from train to test. ECE (10 bins, descriptive) = 0.041.
- **Operational rule:** no model is worth piloting. The rule needs a recall-at-5% lower bound above 0.05 and a point estimate of at least 0.15; the best point estimate is 0.074.

## What would have falsified the null

The pre-registered test (plan section 10) was any model with a PR-AUC lower 95% bound above 0.20. The highest lower bound is 0.040 (3-sigma rule). A primary contrast with Holm p < 0.05, a CI excluding 0 and the hypothesised sign would have supported its question; none did. A test PR-AUC or recall above 0.50 would have triggered a leakage audit. The maximum PR-AUC is 0.065 and the maximum recall is 0.074.

## Correction: C2 recall was wrongly marked "supported"

The first run marked C2 recall at 5% (logistic minus no-skill) `supported=True` with Wald p = 2e-33. Logistic recall was 0, worse than random, and the p came from a degenerate bootstrap: the no-skill recall is the constant k / n and the logistic recall is 0 in nearly every resample, so the SD of the delta was 0.004. The fix is in `src/detect.py` (`primary_table`) and is logged as deviation 9 in `docs/PLAN.md`:

1. A contrast is now supported only if Holm p < 0.05, the CI excludes 0, and the delta has the hypothesised sign (+ for C1, C2, C4, C5, C6; either sign for C3).
2. Recall contrasts against the constant no-skill reference use the exact two-sided hypergeometric random-inspection test. Catching 0 of 27 fails in 27 of 521 runs has exact p = 0.387.

What moved: raw p for recall C1, C2, C5 and C6 went from 0.695, 2e-33, 0.670 and 0.623 to 1.000, 0.387, 0.644 and 1.000. The C2 recall Holm p went from 2e-32 to 1.0 and supported went from True to False, in both the primary and the day-block tables. All other estimates, CIs, Holm p-values, model metrics, calibration tests, figures and post hoc outputs are byte-identical. A new test, `test_significant_delta_in_wrong_direction_is_not_supported`, now guards the rule.

## Correction: the no-skill PR-AUC reference was wrong

The first two runs compared PR-AUC with a no-skill value equal to the test fail rate, 0.052. A random ranking's expected average precision is higher when fails are rare: exactly 0.0625 for 521 runs with 27 fails (an independent reviewer found this; 20,000 simulated random rankings gave 0.0625). The reported PR-AUC excesses of +0.006 to +0.013 over no-skill were an artifact of that reference. The fix is in `src/detect.py` (`perm_ap`, `expected_random_ap`, `primary_table`) and is logged as deviation 10 in `docs/PLAN.md`. It was made after the results were seen, and it is more conservative.

1. The four PR-AUC-vs-no-skill rows (C1, C2, C5, C6) use a one-sided label-permutation test: scores fixed, labels permuted 10,000 times, p = (1 + permuted APs >= observed) / 10,001. The reference is the mean permuted AP and delta = observed AP minus it. They stay in the 12-test Holm family.
2. The no-skill row of the model table reports the random-ranking expectation, 0.0625.

What moved: PR-AUC deltas C1 +0.011 -> +0.0002, C2 +0.008 -> -0.003, C5 +0.013 -> +0.003, C6 +0.006 -> -0.005; raw p 0.423, 0.317, 0.498, 0.545 -> 0.372, 0.466, 0.295, 0.530; CIs as in the table above. No-skill PR-AUC 0.052 (CI 0.035 to 0.071) -> 0.062 (CI 0.045 to 0.081). The C2 PR-AUC day-block CI, which had just excluded 0 (0.0005 to 0.030), is now -0.011 to 0.019. Holm p-values, "supported" flags, model APs, recall rows, calibration tests and post hoc outputs are unchanged. A new test, `test_random_scorer_ap_is_not_supported_and_reference_exceeds_prevalence`, guards the reference.

## Post hoc diagnostics (not part of any Holm family)

- 66 of 521 test runs have a retained feature more than 10 train SDs from the train mean, and 4 of them are fails. Glitch runs in each model's top 27: PCA-SPE 27, logistic 16, IF 14, 3-sigma rule 13.
- The three largest mean shifts (f140, f275, f004) come from a few extreme test readings, and their median shifts are near 0. 60 of 444 features have a mean shift above 0.5 train SD in absolute value; 54 do by median shift.

## Limits

This is one temporal split with 27 test fails, so recall moves in steps of 0.037. The plan estimated that 80% power at the Holm worst case needs a true recall near 0.22, about four times random. The sensors are anonymized, so no result can be tied to equipment. The logistic C sits on the edge of the grid, and the grid was not extended.

## Posting liveness

Conditional build check, checked 2026-10-23 by the orchestrator: GM #203, req JR-202621756, Workday API `canApply=true`, posted 18 days ago; Guardian #200, Koch Avature job 195465, page shows "Apply now". Both live.
