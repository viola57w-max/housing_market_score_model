# Southern California ZIP Market Score Model

> **Start here:** [Read the modeling summary](dashboards/modeling-summary.html).
>
> When GitHub Pages is enabled from the repository root, the project homepage
> automatically opens this interactive summary.

This project calibrates a 0–100 housing-market strength score from ZIP-level
price, demand/supply, and market-speed measures. It produces latest-period 2025
scores for Southern California ZIP codes, extends the fitted method across the
available history, and evaluates stability and distribution shift.

## Project workflow

1. `notebooks/1_EDA.ipynb` prepares the latest-period calibration population
   and the full-period scoring populations.
2. `notebooks/2_modeling.ipynb` compares a compact OLS baseline with a
   sign-compliant LassoCV model, scores the latest period and full history, and
   evaluates stability.
3. `dashboards/modeling-summary.html` is the primary project report, including
   methodology, model results, validation, and interactive charts.
4. `notebooks/local_run/` contains portable scripts for data processing and
   scoring without notebook-only diagnostics.

## Main outputs

- `scores_latest_select_zips_eval.csv`: latest-period evaluation for ZIPs with
  reference scores.
- `scores_latest_all_zips.csv`: latest-period scores for all eligible ZIPs.
- `scores_all_periods.csv`: unadjusted scores for every existing ZIP-period row.
- `scores_all_periods_adjusted.csv`: sensitivity output using reference-based
  predictor trimming and a fitted-Normal score overlay.

## Reproducibility

Create an environment with Python 3.9 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The portable scripts expect their input files to be located in the same folder
as the scripts. See the module docstrings in `notebooks/local_run/` for the exact
input and output contracts.

## Important limitations

- The reference score has no historical timestamp. Calibration therefore uses
  each ZIP's most recent available observation within calendar year 2025.
- Genuine historical reference scores are unavailable; historical predictions
  are evaluated for stability and economic sensibility rather than historical
  predictive accuracy.
- The adjusted full-period score is a sensitivity/contingency result and does
  not replace the unadjusted champion score.
- This is a ZIP-level housing-market model, not an individual borrower credit
  score. Any lending use would require separate legal, fair-lending, bias, and
  model-governance review.

## GitHub Pages

In the repository settings, enable GitHub Pages from the `main` branch and the
repository root. The root `index.html` redirects visitors to the interactive
modeling summary.

## Data and licensing

Confirm that every dataset and generated score file is authorized for public
redistribution before making the repository public. Add restricted files to
`.gitignore` when redistribution is not permitted. No open-source license is
asserted by this repository until ownership and redistribution terms are
confirmed.
