import numpy as np
import pandas as pd
import pytest

import detect as D
import pipeline as P


def synthetic(n=240, p=12, seed=0):
    """Time-ordered toy data shaped like SECOM: some NaNs, one constant column, ~10% fails."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i:03d}" for i in range(p)])
    X["f000"] = 5.0
    X.loc[rng.random(n) < 0.7, "f001"] = np.nan
    X.loc[rng.random(n) < 0.05, "f002"] = np.nan
    y = (rng.random(n) < 0.1).astype(int)
    X.loc[y == 1, "f003"] += 2.0
    t = pd.Series(pd.date_range("2008-09-01", periods=n, freq="4h"))
    return X, y, t


def test_preprocessor_drops_imputes_and_standardizes():
    X, _, _ = synthetic()
    prep = P.Preprocessor().fit(X)
    assert prep.dropped_missing_ == ["f001"]
    assert prep.dropped_constant_ == ["f000"]
    Z = prep.transform(X)
    assert Z.shape == (len(X), X.shape[1] - 2)
    assert not np.isnan(Z).any()
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-12)


def test_temporal_split_is_strictly_ordered():
    _, _, t = synthetic()
    train = P.temporal_split(t, cutoff=pd.Timestamp("2008-09-20"))
    assert t[train].max() < t[~train].min()
    assert 0 < train.sum() < len(t)


def corrupt_test_rows(X, y, train):
    """Wreck every test feature and flip every test label."""
    Xc, yc = X.copy(), y.copy()
    Xc.loc[~train, :] = 1e6
    Xc.loc[~train, "f002"] = np.nan  # would push f002 over 50% missing if test rows were counted
    yc[~train] = 1 - yc[~train]
    return Xc, yc


def test_leakage_preprocessing_never_sees_test_rows():
    X, y, t = synthetic()
    train = P.temporal_split(t, cutoff=pd.Timestamp("2008-09-30"))
    Xc, _ = corrupt_test_rows(X, y, train)
    a, b = P.Preprocessor().fit(X[train]).params(), P.Preprocessor().fit(Xc[train]).params()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    # Control: the same fit on all rows does move, so the test above is not vacuous.
    c = P.Preprocessor().fit(Xc).params()
    assert not np.array_equal(a["mean"], c["mean"]) or a["dropped_missing"] != c["dropped_missing"]


def test_leakage_end_to_end_train_scores_ignore_test_rows_and_labels():
    X, y, t = synthetic()
    train = P.temporal_split(t, cutoff=pd.Timestamp("2008-09-30"))
    Xc, yc = corrupt_test_rows(X, y, train)
    prep_a, s_a, info_a = D.fit_all(X, y, train)
    prep_b, s_b, info_b = D.fit_all(Xc, yc, train)
    for key, v in prep_a.params().items():
        np.testing.assert_array_equal(v, prep_b.params()[key], err_msg=key)
    for m in D.MODELS:
        np.testing.assert_allclose(s_a[m][train], s_b[m][train], err_msg=m)
    assert info_a["logistic_C"] == info_b["logistic_C"]
    assert info_a["pca_components"] == info_b["pca_components"]


def test_contract_rejects_bad_labels_and_unordered_time():
    X = pd.DataFrame(np.zeros((P.N_ROWS, P.N_FEATURES)))
    t = pd.Series(pd.date_range("2008-07-19", periods=P.N_ROWS, freq="h"))
    good = pd.Series(np.where(np.arange(P.N_ROWS) % 10 == 0, 1, -1))
    P.check_contract(X, good, t)
    with pytest.raises(ValueError):
        P.check_contract(X, good.replace(-1, 0), t)
    with pytest.raises(ValueError):
        P.check_contract(X, good, t[::-1].reset_index(drop=True))
    with pytest.raises(ValueError):
        P.check_contract(X.iloc[:, :-1], good, t)


def test_drift_profile_detects_shifted_feature():
    X, _, t = synthetic()
    train = P.temporal_split(t, cutoff=pd.Timestamp("2008-09-30"))
    X.loc[~train, "f005"] += 3.0
    prep = P.Preprocessor().fit(X[train])
    d = P.drift_profile(X, train, prep)
    assert d.feature.iloc[0] == "f005"
    assert d.shift_sd.iloc[0] > 2.0


def test_median_shift_ignores_a_few_glitch_values():
    X, _, t = synthetic()
    train = P.temporal_split(t, cutoff=pd.Timestamp("2008-09-30"))
    X.loc[np.flatnonzero(~train)[:2], "f006"] = 9999.0
    d = P.drift_profile(X, train, P.Preprocessor().fit(X[train])).set_index("feature")
    assert d.loc["f006", "shift_sd"] > 10
    assert abs(d.loc["f006", "median_shift_sd"]) < 0.5
