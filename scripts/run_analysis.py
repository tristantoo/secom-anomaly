"""Run the pre-registered analysis in docs/PLAN.md, in order. Writes outputs/."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import detect as D  # noqa: E402
import pipeline as P  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "outputs"
B = 2000
LABEL = {"iforest": "Isolation Forest", "pca_spe": "PCA-SPE", "logistic": "L2 logistic",
         "spc_rule": "3-sigma rule count", "noskill": "no-skill"}
COLOR = {"iforest": "#2a78d6", "pca_spe": "#7a5bc7", "logistic": "#eb6834", "spc_rule": "#3a9a5b"}
INK, MUTED = "#1f1f1e", "#8a897f"


def pipeline_layer(X, t, train, prep):
    """PLAN section 9: missingness and drift, descriptive only."""
    miss = P.missingness_profile(X, train)
    miss.to_csv(OUT / "missingness_by_feature.csv", index=False)
    drift = P.drift_profile(X, train, prep)
    drift.to_csv(OUT / "drift_by_feature.csv", index=False)
    per_run = X.isna().sum(axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].scatter(t, per_run, s=6, color=MUTED)
    ax[0].plot(t, per_run.rolling(50, center=True).median(), color=INK, lw=1.5, label="rolling median (50 runs)")
    ax[0].set_ylabel("missing values per run")
    ax[0].legend(frameon=False, fontsize=8)
    week = t.dt.to_period("W").dt.start_time
    Z = pd.DataFrame(prep.transform(X), columns=prep.columns_)
    for f in drift.feature.head(5):
        ax[1].plot(Z[f].groupby(week).mean(), marker="o", ms=3, lw=1.2, label=f)
    ax[1].set_ylabel("weekly mean (train SD units)")
    ax[1].legend(frameon=False, fontsize=8, ncol=5, title="five largest mean shifts", title_fontsize=8)
    robust = drift.reindex(drift.median_shift_sd.abs().sort_values(ascending=False).index).feature.head(5)
    for f in robust:  # post hoc panel, PLAN deviations
        ax[2].plot(Z[f].groupby(week).median(), marker="o", ms=3, lw=1.2, label=f)
    ax[2].set_ylabel("weekly median (train SD units)")
    ax[2].legend(frameon=False, fontsize=8, ncol=5, title="five largest median shifts (post hoc)", title_fontsize=8)
    for a in ax:
        a.axvline(P.CUTOFF, color=INK, ls="--", lw=1)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title("Sensor data quality and drift; dashed line = train/test cutoff", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_missingness_drift.png", dpi=150)
    plt.close(fig)
    return dict(
        features_raw=X.shape[1],
        dropped_missing_gt_50pct=len(prep.dropped_missing_),
        dropped_constant=len(prep.dropped_constant_),
        features_retained=len(prep.columns_),
        cells_missing_pct_train=float(X[train].isna().mean().mean() * 100),
        cells_missing_pct_test=float(X[~train].isna().mean().mean() * 100),
        features_abs_shift_gt_0_5_sd=int((drift.shift_sd.abs() > 0.5).sum()),
        share_abs_shift_gt_0_5_sd=float((drift.shift_sd.abs() > 0.5).mean()),
        median_abs_shift_sd=float(drift.shift_sd.abs().median()),
        posthoc_features_abs_median_shift_gt_0_5_sd=int((drift.median_shift_sd.abs() > 0.5).sum()),
        posthoc_top5_median_shift=drift.reindex(drift.median_shift_sd.abs().sort_values(ascending=False).index)
        .head(5)[["feature", "median_shift_sd", "shift_sd"]].round(3).to_dict("records"),
    )


def posthoc(y, scores, Z_test, k):
    """Post hoc diagnostics (PLAN deviations): exact random-inspection test and the glitch-run check."""
    glitch = (np.abs(Z_test) > 10).any(axis=1)
    rows = []
    for m, s in scores.items():
        top = np.argsort(-s, kind="stable")[:k]
        caught = int(round(D.recall_at(y, s) * y.sum()))
        rows.append(dict(model=m, caught_in_top_k=caught, k=k, fails=int(y.sum()),
                         exact_p_vs_random=D.random_inspection_p(caught, len(y), int(y.sum()), k),
                         glitch_runs_in_top_k=int(glitch[top].sum())))
    t = pd.DataFrame(rows)
    return t, dict(test_runs_with_any_abs_z_gt_10=int(glitch.sum()), fails_among_them=int(y[glitch].sum()))


def model_table(y, scores, p_naive):
    # no-skill PR-AUC = exact expected AP of a random ranking (not the prevalence, which it exceeds; deviation 10)
    rows = [dict(model="noskill", pr_auc=D.expected_random_ap(len(y), int(y.sum())), recall_at_5pct=np.ceil(0.05 * len(y)) / len(y),
                 precision_at_5pct=y.mean(), roc_auc=0.5)]
    for m, s in scores.items():
        rows.append(dict(model=m, pr_auc=D.average_precision_score(y, s), recall_at_5pct=D.recall_at(y, s),
                         precision_at_5pct=D.precision_at(y, s), roc_auc=roc_auc_score(y, s)))
    t = pd.DataFrame(rows)
    p = scores["logistic"]
    calib = dict(brier_logistic=float(np.mean((p - y) ** 2)), brier_naive=float(np.mean((p_naive - y) ** 2)),
                 ece_logistic_10bin=D.ece(y, p), mean_pred_logistic=float(p.mean()), observed_rate=float(y.mean()),
                 naive_rate=float(p_naive))
    calib["brier_skill_score"] = 1 - calib["brier_logistic"] / calib["brier_naive"]
    return t, calib


def add_ci(table, boot):
    for col, pre in (("pr_auc", "ap"), ("recall_at_5pct", "rec")):
        lo, hi = zip(*[np.percentile(boot[f"{pre}_{m}"], [2.5, 97.5]) for m in table.model])
        table[f"{col}_lo"], table[f"{col}_hi"] = lo, hi
    return table


def figures(y, scores, prim):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for m, s in scores.items():
        pr, rc, _ = precision_recall_curve(y, s)
        ax[0].step(rc, pr, where="post", color=COLOR[m], lw=1.5,
                   label=f"{LABEL[m]} (AP {D.average_precision_score(y, s):.3f})")
    ax[0].axhline(y.mean(), color=MUTED, ls="--", lw=1, label=f"prevalence ({y.mean():.3f})")
    ax[0].set(xlabel="recall", ylabel="precision", title="Test precision-recall curves", ylim=(0, 1))
    ax[0].legend(frameon=False, fontsize=8)
    t = prim.iloc[::-1].reset_index(drop=True)
    ax[1].errorbar(t.delta, t.index, xerr=[t.delta - t.ci_lo, t.ci_hi - t.delta], fmt="o", color=INK,
                   ecolor=MUTED, capsize=2, ms=4)
    ax[1].set_yticks(t.index, [f"{r.contrast} {r.metric}: {LABEL[r.a.split('_', 1)[1]]} - {LABEL[r.b.split('_', 1)[1]]}"
                               for r in t.itertuples()], fontsize=7)
    ax[1].axvline(0, color=INK, lw=1)
    ax[1].set(xlabel="delta with 95% paired bootstrap CI", title="Primary family (Holm, 12 tests)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pr_curves_and_contrasts.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    X, y, t = P.load(RAW)                                       # hash gate + data contract
    train = P.temporal_split(t)
    test = ~train
    split = dict(cutoff=str(P.CUTOFF), n_train=int(train.sum()), n_test=int(test.sum()),
                 fails_train=int(y[train].sum()), fails_test=int(y[test].sum()),
                 fail_rate_train=float(y[train].mean()), fail_rate_test=float(y[test].mean()),
                 train_span=[str(t[train].min()), str(t[train].max())],
                 test_span=[str(t[test].min()), str(t[test].max())])
    print("split:", split)

    prep, scores, info = D.fit_all(X, y, train)
    info["cv"].to_csv(OUT / "logistic_inner_cv.csv", index=False)
    pipe = pipeline_layer(X, t, train, prep)
    print("pipeline layer:", pipe)

    yt = y[test]
    st = {m: s[test] for m, s in scores.items()}
    p_naive = y[train].mean()
    obs = D.metrics(yt, st, p_naive)
    rng = np.random.default_rng(D.SEED)
    boot, redraws = D.bootstrap(yt, st, p_naive, D.run_sampler(len(yt)), B, rng)
    days = t[test].dt.date.to_numpy()
    boot_day, redraws_day = D.bootstrap(yt, st, p_naive, D.day_sampler(days), B, rng)

    models, calib = model_table(yt, st, p_naive)
    models = add_ci(models, boot)
    models.to_csv(OUT / "model_metrics.csv", index=False)
    perm = {m: D.perm_ap(yt, s) for m, s in st.items()}
    prim = D.primary_table(obs, boot, yt, perm)
    prim.to_csv(OUT / "primary_contrasts.csv", index=False)
    sens = D.primary_table(obs, boot_day, yt, perm)
    sens.to_csv(OUT / "sensitivity_day_block_contrasts.csv", index=False)
    cal = D.calibration_table(obs, boot)
    cal.to_csv(OUT / "calibration_contrasts.csv", index=False)
    figures(yt, st, prim)
    ph, glitch = posthoc(yt, st, prep.transform(X[test]), int(np.ceil(0.05 * len(yt))))
    ph.to_csv(OUT / "posthoc_exact_recall_and_glitch.csv", index=False)

    # Decision rules, PLAN section 10.
    rec = models.set_index("model")
    pilot = {m: bool(rec.loc[m, "recall_at_5pct_lo"] > 0.05 and rec.loc[m, "recall_at_5pct"] >= 0.15)
             for m in D.MODELS}
    leak_flag = bool((models.pr_auc > 0.5).any() or (models.recall_at_5pct > 0.5).any())
    null_falsified = bool((models[models.model != "noskill"].pr_auc_lo > 0.20).any())
    brier_row = cal.set_index("contrast").loc["brier_delta"]
    probs_usable = bool(calib["brier_skill_score"] > 0 and brier_row.ci_hi < 0)

    summary = dict(split=split, pipeline=pipe, pca_components=info["pca_components"], logistic_C=info["logistic_C"],
                   permutation=dict(n_perm=D.N_PERM, seed=D.SEED,
                                    mean_ap={m: float(v.mean()) for m, v in perm.items()}),
                   bootstrap=dict(B=B, seed=D.SEED, redraws_run=redraws, redraws_day=redraws_day,
                                  n_test_days=int(len(np.unique(days)))),
                   models=models.round(4).to_dict("records"), calibration=calib,
                   primary=prim.round(4).to_dict("records"), calibration_tests=cal.round(4).to_dict("records"),
                   posthoc=dict(exact_recall=ph.round(4).to_dict("records"), glitch=glitch),
                   sensitivity_day_block_supported=sens[sens.supported][["contrast", "metric"]].to_dict("records"),
                   decisions=dict(worth_piloting=pilot, leakage_stop_rule_triggered=leak_flag,
                                  expected_null_falsified=null_falsified, logistic_probs_usable=probs_usable,
                                  primary_supported=prim[prim.supported][["contrast", "metric"]].to_dict("records")))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(models.round(3).to_string(index=False))
    print(prim.round(4).to_string(index=False))
    print(sens[["contrast", "metric", "ci_lo", "ci_hi", "p_holm", "supported"]].round(4).to_string(index=False))
    print(cal.round(4).to_string(index=False))
    print("calibration:", calib)
    print("decisions:", summary["decisions"])
    print(ph.round(4).to_string(index=False), glitch)


if __name__ == "__main__":
    main()
