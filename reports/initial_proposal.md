# Initial Proposal After EDA + Cleansing

Source notebook: `notebooks/01_eda_cleansing.ipynb`

## What we learned quickly

- Dataset size is **78,025 rows x 19 columns**.
- `Opportunity Result` is imbalanced: about **22.6% Won** vs **77.4% Loss**.
- Most clients are in reacquisition-like condition (`Revenue From Client Past Two Years (USD) = 0 (No business)`), but these have much lower win rate than clients with recent revenue.
- `Competitor Type` has missing values (~11.9%); other fields are mostly complete.
- `Opportunity Amount USD` is right-skewed with high-tail values; practical winsorization/log transform is appropriate for modeling stability.

## Cleansing choices applied (common sense)

1. Text normalization on categorical columns (`strip`, keep canonical values).
2. Missing `Competitor Type` filled with `Unknown` (explicitly modeled, not dropped).
3. Strategic 2-phase segment created:
   - `Reacquisition` for `0 (No business)`
   - `Engagement/Upselling` for all other recent-revenue bands
4. `target_win` binary target created from `Opportunity Result`.
5. Outlier-aware features created:
   - P99-capped amount/stage-change/time variants
   - `log_opportunity_amount` for amount modeling.

## Proposed modeling track (next)

## 1) Win/loss classifier

- Candidate models: regularized logistic regression (baseline), LightGBM/XGBoost (non-linear).
- Evaluation: PR-AUC, ROC-AUC, F1 at operating threshold, calibration curve.
- Business output: `P(win)` to support process prioritization.

## 2) Amount prediction model

- Train on `log_opportunity_amount` and transform back for expected amount.
- Candidate models: ElasticNet baseline + gradient boosting regressor.
- Evaluation: MAE, RMSE, MAPE by client/revenue segments.

## 3) Decision simulation

- **Process prioritization:** rank opportunities by expected value = `P(win) * predicted_amount` with scenario shifts in days-to-close.
- **Channel optimization:** run what-if allocation by `Route To Market` and segment to compare expected revenue lift.

## Recommended next deliverables

1. `notebooks/02_modeling_classification.ipynb`
2. `notebooks/03_modeling_amount.ipynb`
3. `notebooks/04_simulation_prioritization_channel.ipynb`
4. `reports/modeling_results.md` with metrics + feature importance + actions.
