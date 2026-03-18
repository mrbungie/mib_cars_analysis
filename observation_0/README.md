# Observation 0

Parallel-level validation and maturity diagnostics for the amount target.

## ID Validation

- Notebook: `00_observation0_basic_validation.ipynb`
- Report: `00_observation0_basic_validation.md`
- Key outputs:
  - `observation0_opportunity_duplicates.csv`
  - `observation0_duplicate_state_analysis.csv`
  - `observation0_duplicate_state_sample.csv`
  - `observation0_exact_repeat_rows.csv`
  - `observation0_state_change_rows.csv`
  - `observation0_duplicate_split_summary.csv`

## Amount Maturity Diagnostics

- Notebook: `01_amount_maturity_thresholds.ipynb`
- Report: `01_amount_maturity_thresholds.md`
- Methods included:
  1. Cross-sectional predictability (`R^2` plateau by age bucket)
  2. Round-number heuristic decay by age bucket (strict 10k/50k roundness)
  3. Probability-driven maturity (`CV` stabilization by `P(win)` bins) with forward-stability guardrails
- Primary maturity flag (`is_mature`) derived from Method 1 (age bucket ≥ 61‑90) is now available in the notebook outputs.
- Key outputs:
  - `observation0_amount_method1_r2_mae_by_age_bucket.csv`
  - `observation0_amount_method2_roundpct_by_age_bucket.csv`
  - `observation0_amount_method3_cv_by_pwin_bin.csv` (includes `rolling_range_3` and `forward_range` stability diagnostics)
  - `observation0_amount_maturity_threshold_summary.csv`
