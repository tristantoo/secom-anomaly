import numpy as np
import pytest
from sklearn.metrics import average_precision_score

import detect as D


def test_recall_at_exact_and_tied():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 2)  # n = 20, k = ceil(0.05 * 20) = 1, 4 fails
    s = np.arange(20, 0, -1).astype(float)            # top score is a fail
    assert D.recall_at(y, s) == pytest.approx(1 / 4)
    s_tied = np.ones(20)                               # all tied: expected recall = k / n
    assert D.recall_at(y, s_tied) == pytest.approx(1 / 20)
    y2, s2 = np.array([1, 0, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0])
    assert D.recall_at(y2, s2, frac=0.25) == pytest.approx(0.5)  # k = 1, two tied at top, one is a fail


def test_noskill_constant_score_ap_equals_prevalence():
    y = np.array([0] * 90 + [1] * 10)
    assert average_precision_score(y, np.zeros(100)) == pytest.approx(0.1)


def test_holm_known_values():
    np.testing.assert_allclose(D.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert D.holm([0.9, 0.8]).max() <= 1.0


def test_ece_zero_when_calibrated_and_positive_when_not():
    y = np.array([0, 1] * 50)
    assert D.ece(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert D.ece(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_day_sampler_keeps_whole_days():
    days = np.repeat(["a", "b", "c"], [2, 3, 5])
    idx = D.day_sampler(days)(np.random.default_rng(0))
    drawn = days[idx]
    for d in np.unique(drawn):
        assert (drawn == d).sum() % (days == d).sum() == 0


def test_paired_bootstrap_identical_models_give_zero_delta():
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.1).astype(int)
    s = rng.random(200)
    scores = {m: s for m in D.MODELS}
    boot, _ = D.bootstrap(y, scores, 0.1, D.run_sampler(200), 50, np.random.default_rng(2))
    obs = D.metrics(y, scores, 0.1)
    t = D.primary_table(obs, boot, y, {m: D.perm_ap(y, s, 50) for m in D.MODELS})
    same = t[t.contrast.isin(["C3", "C4"])]
    assert (same.delta == 0).all() and (same.ci_lo == 0).all() and (same.ci_hi == 0).all()
    assert not same.supported.any()


def test_significant_delta_in_wrong_direction_is_not_supported():
    rng = np.random.default_rng(5)
    y = (rng.random(2000) < 0.2).astype(int)
    noise = rng.random(2000)
    scores = {"iforest": -y + 0.1 * noise, "pca_spe": y + 0.1 * noise,  # anti-predictive vs. near-perfect
              "logistic": noise, "spc_rule": noise}
    boot, _ = D.bootstrap(y, scores, 0.2, D.run_sampler(2000), 200, np.random.default_rng(6))
    perm = {m: D.perm_ap(y, s, 1000) for m, s in scores.items()}
    t = D.primary_table(D.metrics(y, scores, 0.2), boot, y, perm).set_index(["contrast", "metric"])
    assert t.loc[("C1", "recall_at_5pct"), "p_holm"] < 0.05  # two-sided exact test: significant, but worse
    for met in ("pr_auc", "recall_at_5pct"):
        assert t.loc[("C1", met), "delta"] < 0
        assert not t.loc[("C1", met), "supported"]       # worse than chance is never support
        assert t.loc[("C6", met), "supported"]           # significant and in the hypothesised direction
    assert t.loc[("C3", "pr_auc"), "supported"]          # C3 is two-sided: a significant negative delta counts
    assert t.loc[("C1", "recall_at_5pct"), "test"] == "exact_hypergeom"


def test_random_scorer_ap_is_not_supported_and_reference_exceeds_prevalence():
    rng = np.random.default_rng(7)
    y = np.zeros(521, int)
    y[rng.choice(521, 27, replace=False)] = 1          # rare fails, as in the SECOM test set
    scores = {m: rng.random(521) for m in D.MODELS}
    perm = {m: D.perm_ap(y, s, 2000) for m, s in scores.items()}
    assert perm["iforest"].mean() > y.mean() + 0.008    # random-ranking AP ~0.0625 > prevalence 0.0518
    assert perm["iforest"].mean() == pytest.approx(D.expected_random_ap(521, 27), abs=0.002)
    boot, _ = D.bootstrap(y, scores, 0.05, D.run_sampler(521), 200, np.random.default_rng(8))
    t = D.primary_table(D.metrics(y, scores, 0.05), boot, y, perm)
    ap_rows = t[(t.metric == "pr_auc") & (t.b == "ap_noskill")]
    assert (ap_rows.test == "label_perm").all() and not ap_rows.supported.any()


def test_bootstrap_redraws_zero_fail_samples():
    y = np.zeros(50, int)
    y[0] = 1
    scores = {m: np.arange(50.0) for m in D.MODELS}
    boot, redraws = D.bootstrap(y, scores, 0.02, D.run_sampler(50), 30, np.random.default_rng(3))
    assert len(boot) == 30 and redraws > 0


def test_spc_rule_counts_out_of_limit_features():
    Z_pass = np.random.default_rng(4).normal(size=(500, 3))
    f = D.fit_spc_rule(Z_pass)
    assert f(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, -10.0]])).tolist() == [0.0, 2.0]


def test_random_inspection_p_matches_hypergeometric():
    from scipy import stats
    assert D.random_inspection_p(0, 521, 27, 27) > 0.2  # catching none by chance is common
    assert D.random_inspection_p(10, 521, 27, 27) < 1e-6
    assert D.random_inspection_p(1, 10, 2, 2) == pytest.approx(stats.hypergeom(10, 2, 2).pmf([1, 2]).sum())
