# secom-anomaly

A pre-registered, time-split evaluation of anomaly detectors on the UCI SECOM semiconductor data: 1,567 production runs, 590 sensor measurements, a pass/fail line test and a timestamp.

The question: if a fab can inspect only 5% of runs, does ranking runs by a sensor-based anomaly score catch more failing runs than inspecting at random?

The analysis covers:
- data contract checks, a missingness profile and sensor drift over time
- train-only preprocessing, with a leakage test that fails if test rows reach the fit
- Isolation Forest and PCA reconstruction error (trained on passing runs only), L2 logistic regression, a 3-sigma control-chart rule, and a no-skill reference
- PR-AUC, recall at a 5% inspection budget and calibration, with paired bootstrap CIs, label-permutation and exact hypergeometric tests against random, and Holm correction

**Result: a null.** On the 521 later test runs (27 fails), every model's PR-AUC sits between 0.058 and 0.065, within 0.006 of the 0.062 to 0.063 a random ranking scores, and two of the four fall below it. That reference comes from a label-permutation test; the fail rate 0.052 understates it when fails are rare. No difference survives correction. The anomaly detectors mostly flag runs with sensor glitches, not failing runs. Details in `docs/PLAN.md` (Deviations).

- Plan, committed before labels were read: `docs/PLAN.md`
- Tables, figures, `summary.json`: `outputs/`
- Written report: `docs/REPORT.md`

**Posting liveness (conditional build check), checked 2026-10-23 by the orchestrator.** GM #203, req JR-202621756: Workday API returns `canApply=true`, posted 18 days ago. Guardian #200, Koch Avature job 195465: page shows "Apply now". Both live.

## Reproduce

```bash
uv sync
mkdir -p data/raw
curl -L -o data/raw/secom.zip "https://archive.ics.uci.edu/static/public/179/secom.zip"
unzip -o data/raw/secom.zip -d data/raw
uv run python scripts/run_analysis.py   # stops if either SHA-256 differs from docs/PLAN.md section 2
uv run pytest -q
```

Check the order of the pre-registration with `git log --reverse --format='%h %ad %s' --date=iso`.

## Authorship

Built with AI coding assistance (Claude Code). I set the scope and questions and reviewed the methods, code and results.
