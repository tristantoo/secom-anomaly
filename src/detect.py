"""Models, metrics and paired bootstrap for docs/PLAN.md sections 6 to 8."""
import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from pipeline import Preprocessor

SEED = 20261023
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
BUDGET = 0.05
N_PERM = 10_000
MODELS = ["iforest", "pca_spe", "logistic", "spc_rule"]
CONTRASTS = [  # (id, a, b, hypothesised sign of metric(a) - metric(b); 0 = two-sided, PLAN sections 3 and 10)
    ("C1", "iforest", "noskill", 1),
    ("C2", "logistic", "noskill", 1),
    ("C3", "iforest", "logistic", 0),
    ("C4", "iforest", "spc_rule", 1),
    ("C5", "spc_rule", "noskill", 1),
    ("C6", "pca_spe", "noskill", 1),
]


# ---------- models: each returns a function from standardized (or raw, for logistic) rows to a score ----------

def fit_iforest(Z_pass, seed=SEED):
    m = IsolationForest(n_estimators=500, max_samples=256, max_features=1.0, random_state=seed).fit(Z_pass)
    return lambda Z: -m.score_samples(Z)


def fit_pca_spe(Z_pass, var=0.90):
    m = PCA(n_components=var, svd_solver="full").fit(Z_pass)
    score = lambda Z: ((Z - m.inverse_transform(m.transform(Z))) ** 2).sum(axis=1)  # noqa: E731
    return score, m.n_components_


def fit_spc_rule(Z_pass, k=3.0):
    """Count of features outside mean +/- k SD of passing train runs.

    A feature that is constant across passing runs has SD 0, so any departure from that value counts.
    """
    mu, sd = Z_pass.mean(axis=0), Z_pass.std(axis=0)
    return lambda Z: (np.abs(Z - mu) > k * sd).sum(axis=1).astype(float)


def logistic_pipe(C):
    return Pipeline([("prep", Preprocessor()), ("lr", LogisticRegression(C=C, max_iter=5000))])


def fit_logistic(X_train, y_train, grid=C_GRID, n_splits=4):
    """Choose C by mean AP over forward-chaining folds inside train (prep refitted per fold), then refit.

    X_train must be in time order. Folds with no fails in either part are skipped. Ties go to the smaller C.
    """
    rows = []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X_train)):
        if y_train[tr].sum() == 0 or y_train[va].sum() == 0:
            continue
        for C in grid:
            p = logistic_pipe(C).fit(X_train.iloc[tr], y_train[tr]).predict_proba(X_train.iloc[va])[:, 1]
            rows.append(dict(fold=fold, C=C, ap=average_precision_score(y_train[va], p),
                             n_val=len(va), fails_val=int(y_train[va].sum())))
    cv = pd.DataFrame(rows)
    mean_ap = cv.groupby("C").ap.mean()
    best = min(grid)
    for C in sorted(grid):
        if mean_ap[C] > mean_ap[best]:
            best = C
    model = logistic_pipe(best).fit(X_train, y_train)
    return (lambda X: model.predict_proba(X)[:, 1]), best, cv


def fit_all(X, y, train):
    """Fit preprocessing and all four models on train rows only, then score every row.

    Nothing here reads X[~train] or y[~train] except to apply already-fitted models.
    """
    prep = Preprocessor().fit(X[train])
    Z = prep.transform(X)
    Z_pass = Z[train & (y == 0)]
    f_pca, k = fit_pca_spe(Z_pass)
    f_lr, C, cv = fit_logistic(X[train], y[train])
    scores = {"iforest": fit_iforest(Z_pass)(Z), "pca_spe": f_pca(Z),
              "logistic": f_lr(X), "spc_rule": fit_spc_rule(Z_pass)(Z)}
    return prep, scores, dict(pca_components=int(k), logistic_C=C, cv=cv)


# ---------- metrics ----------

def recall_at(y, s, frac=BUDGET):
    """Recall among the top ceil(frac * n) scores; ties straddling the cutoff get fractional credit."""
    y, s = np.asarray(y), np.asarray(s, float)
    k = math.ceil(frac * len(s))
    v = np.sort(s)[::-1][k - 1]
    above, tied = s > v, s == v
    caught = y[above].sum() + (k - above.sum()) / tied.sum() * y[tied].sum()
    return float(caught / y.sum())


def precision_at(y, s, frac=BUDGET):
    k = math.ceil(frac * len(s))
    return recall_at(y, s, frac) * np.sum(y) / k


def random_inspection_p(caught, n, fails, k):
    """Post hoc: exact two-sided p that inspecting k of n runs at random catches `caught` of `fails` fails.

    Hypergeometric null; two-sided by summing every outcome no more likely than the observed one.
    """
    d = stats.hypergeom(n, fails, k)
    pmf = d.pmf(np.arange(min(fails, k) + 1))
    return float(min(1.0, pmf[pmf <= d.pmf(caught) * (1 + 1e-7)].sum()))


def expected_random_ap(n, m):
    """Exact expected AP of a uniformly random ranking of n runs with m fails: (H_n + (n - H_n)(m - 1)/(n - 1)) / n.

    Larger than the prevalence m / n when fails are rare (521 runs, 27 fails: 0.0625 vs 0.0518).
    """
    h = (1.0 / np.arange(1, n + 1)).sum()
    return float((h + (n - h) * (m - 1) / (n - 1)) / n)


def perm_ap(y, s, n_perm=N_PERM, seed=SEED):
    """AP of fixed scores s under n_perm random permutations of the labels (null: s says nothing about y).

    Ties in s are kept, so the null reference is right for coarse scores such as the rule count.
    """
    rng = np.random.default_rng(seed)
    return np.array([average_precision_score(rng.permutation(y), s) for _ in range(n_perm)])


def ece(y, p, bins=10):
    """Expected calibration error with equal-frequency bins."""
    order = np.argsort(p, kind="stable")
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(order, bins)))


def metrics(y, scores, p_naive):
    """All per-sample statistics the plan needs, for one (possibly resampled) test set."""
    out = {"ap_noskill": expected_random_ap(len(y), int(y.sum())), "rec_noskill": math.ceil(BUDGET * len(y)) / len(y)}
    for m, s in scores.items():
        out[f"ap_{m}"] = average_precision_score(y, s)
        out[f"rec_{m}"] = recall_at(y, s)
    p = scores["logistic"]
    out["brier_logistic"] = np.mean((p - y) ** 2)
    out["brier_naive"] = np.mean((p_naive - y) ** 2)
    out["citl_logistic"] = p.mean() - y.mean()
    return out


# ---------- bootstrap ----------

def run_sampler(n):
    return lambda rng: rng.integers(0, n, n)


def day_sampler(days):
    """Resample calendar days with replacement (same count) and keep every run in each drawn day."""
    days = np.asarray(days)
    groups = [np.flatnonzero(days == d) for d in np.unique(days)]
    return lambda rng: np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def bootstrap(y, scores, p_naive, sampler, B, rng):
    """B paired resamples: every model is scored on the same indices. Zero-fail draws are redrawn."""
    rows, redraws = [], 0
    while len(rows) < B:
        i = sampler(rng)
        if y[i].sum() == 0:
            redraws += 1
            continue
        rows.append(metrics(y[i], {m: s[i] for m, s in scores.items()}, p_naive))
    return pd.DataFrame(rows), redraws


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def contrast_row(name, obs, boot, a, b):
    d, dist = obs[a] - obs[b], (boot[a] - boot[b]).to_numpy()
    sd = dist.std(ddof=1)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p = float(2 * stats.norm.sf(abs(d) / sd)) if sd > 0 else float(d == 0)
    return dict(contrast=name, a=a, b=b, delta=d, ci_lo=lo, ci_hi=hi, boot_sd=sd, p_raw=p)


def primary_table(obs, boot, y, perm):
    """The 12-test primary family (6 contrasts x {PR-AUC, recall at 5%}), Holm-adjusted together.

    Recall against no-skill compares with a constant (k / n), so the bootstrap SD of the delta is degenerate;
    those rows use the exact random-inspection (hypergeometric) p instead (PLAN deviations). PR-AUC against
    no-skill uses a one-sided label-permutation test: perm[model] holds that model's AP under label permutations
    (perm_ap), the reference is their mean and p = (1 + #perm AP >= observed) / (1 + #perms) (PLAN deviation 10).
    The CI on those rows is the paired bootstrap of AP minus the random-ranking expectation per resample.
    A contrast is supported only if Holm p < 0.05, the CI excludes 0, and the delta has the hypothesised sign.
    """
    n, fails = len(y), int(np.sum(y))
    k = math.ceil(BUDGET * n)
    rows = []
    for met, pre in (("pr_auc", "ap"), ("recall_at_5pct", "rec")):
        for cid, a, b, sign in CONTRASTS:
            r = dict(metric=met, **contrast_row(cid, obs, boot, f"{pre}_{a}", f"{pre}_{b}"), sign=sign, test="wald_boot",
                     ref=obs[f"{pre}_{b}"])
            if pre == "ap" and b == "noskill":  # every such contrast is hypothesised +, so one-sided upward
                null, ap = perm[a], obs[f"ap_{a}"]
                r.update(delta=ap - null.mean(), ref=null.mean(), test="label_perm",
                         p_raw=float((1 + (null >= ap).sum()) / (1 + len(null))))
            if pre == "rec" and b == "noskill":
                r.update(p_raw=random_inspection_p(round(obs[f"rec_{a}"] * fails), n, fails, k), test="exact_hypergeom")
            rows.append(r)
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_raw)
    right_way = (t.sign == 0) | (np.sign(t.delta) == t.sign)
    t["supported"] = (t.p_holm < 0.05) & ((t.ci_lo > 0) | (t.ci_hi < 0)) & right_way
    return t


def calibration_table(obs, boot):
    """Calibration family: Brier(M3) - Brier(naive), and calibration in the large of M3 against 0."""
    boot = boot.assign(zero=0.0)
    obs = dict(obs, zero=0.0)
    t = pd.DataFrame([contrast_row("brier_delta", obs, boot, "brier_logistic", "brier_naive"),
                      contrast_row("citl", obs, boot, "citl_logistic", "zero")])
    t["p_holm"] = holm(t.p_raw)
    return t
