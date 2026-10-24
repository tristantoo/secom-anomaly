"""Data layer for docs/PLAN.md: load, contract checks, temporal split, train-only preprocessing, drift profile."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HASHES = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
}
N_ROWS, N_FEATURES = 1567, 590
CUTOFF = pd.Timestamp("2008-09-24 00:00:00")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(raw_dir):
    """Hash-gate the raw files, then return features X (DataFrame), label y (1 = fail) and timestamps t."""
    for name, h in HASHES.items():
        got = sha256(raw_dir / name)
        if got != h:
            raise ValueError(f"{name}: SHA-256 {got} != {h}")
    X = pd.read_csv(raw_dir / "secom.data", sep=" ", header=None)
    X.columns = [f"f{i:03d}" for i in range(X.shape[1])]
    lab = pd.read_csv(raw_dir / "secom_labels.data", sep=" ", header=None, names=["label", "ts"])
    t = pd.to_datetime(lab.ts, format="%d/%m/%Y %H:%M:%S")
    check_contract(X, lab.label, t)
    return X, (lab.label == 1).astype(int).to_numpy(), t


def check_contract(X, label, t):
    """Data contract from PLAN section 9. Raises on the first violation."""
    if X.shape != (N_ROWS, N_FEATURES):
        raise ValueError(f"shape {X.shape} != {(N_ROWS, N_FEATURES)}")
    if not all(np.issubdtype(d, np.number) for d in X.dtypes):
        raise ValueError("non-numeric feature column")
    if not set(np.unique(label)) <= {-1, 1}:
        raise ValueError(f"labels outside {{-1, 1}}: {set(np.unique(label))}")
    if len(t) != N_ROWS or t.isna().any() or not t.is_monotonic_increasing:
        raise ValueError("timestamps missing or not non-decreasing")


def temporal_split(t, cutoff=CUTOFF):
    """Boolean train mask: timestamp strictly before the cutoff. Test is its complement."""
    return (pd.Series(t) < cutoff).to_numpy()


class Preprocessor(BaseEstimator, TransformerMixin):
    """PLAN section 5: drop >max_missing features, median-impute, drop constants, standardize.

    Every parameter is computed from the rows passed to fit and nothing else.
    """

    def __init__(self, max_missing=0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        miss = X.isna().mean()
        self.dropped_missing_ = list(miss.index[miss > self.max_missing])
        kept = X.drop(columns=self.dropped_missing_)
        med = kept.median()
        filled = kept.fillna(med)
        sd = filled.std(ddof=0)
        self.dropped_constant_ = list(sd.index[sd == 0])
        self.columns_ = [c for c in kept.columns if c not in self.dropped_constant_]
        self.median_ = med[self.columns_]
        self.mean_ = filled[self.columns_].mean()
        self.scale_ = sd[self.columns_]
        return self

    def transform(self, X):
        Z = pd.DataFrame(X)[self.columns_].fillna(self.median_)
        return ((Z - self.mean_) / self.scale_).to_numpy()

    def params(self):
        """Every fitted quantity, for the leakage test."""
        return dict(dropped_missing=self.dropped_missing_, dropped_constant=self.dropped_constant_,
                    median=self.median_.to_numpy(), mean=self.mean_.to_numpy(), scale=self.scale_.to_numpy())


def missingness_profile(X, train):
    """Per-feature missing rate in train and test."""
    return pd.DataFrame({"feature": X.columns,
                         "missing_train": X[train].isna().mean().to_numpy(),
                         "missing_test": X[~train].isna().mean().to_numpy()})


def drift_profile(X, train, prep):
    """Standardized mean shift (test mean - train mean) / train SD for every retained feature.

    Uses the train-fitted preprocessor, so the shift is in train SD units after train-median imputation.
    median_shift_sd is the same contrast on medians (post hoc, see PLAN deviations): it ignores a few glitch values.
    """
    Z = prep.transform(X)
    shift = Z[~train].mean(axis=0) - Z[train].mean(axis=0)
    med_shift = np.median(Z[~train], axis=0) - np.median(Z[train], axis=0)
    out = pd.DataFrame({"feature": prep.columns_, "shift_sd": shift, "median_shift_sd": med_shift,
                        "missing_train": X.loc[train, prep.columns_].isna().mean().to_numpy(),
                        "missing_test": X.loc[~train, prep.columns_].isna().mean().to_numpy()})
    return out.reindex(out.shift_sd.abs().sort_values(ascending=False).index).reset_index(drop=True)
