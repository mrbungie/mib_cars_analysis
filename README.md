# MIB Cars Analysis

Project workspace for sales funnel analytics and predictive modeling on `cars.xlsx`.

## Slides structure

- Proposal outline: [`SLIDES_PROPOSAL.md`](SLIDES_PROPOSAL.md)

## Core tasks to do

1. **Classification task (win/loss):**
   - Predict `Opportunity Result` (win vs loss).
   - Focus on funnel, client segmentation, and process variables.

2. **Amount prediction task (deal amount):**
   - Predict the deal amount target (amount/deal-size perspective).
   - Primary numeric target: `Opportunity Amount USD`.
   - Secondary business view: `Deal Size Category (USD)`.

## Delivery format and workflow

- Deliver analysis in **Jupyter notebooks** (`notebooks/`) plus **Markdown summaries** (`reports/`).
- Start with **Exploratory Data Analysis (EDA)**:
  - Funnel phase x client segment insights
  - Target exploration for both tasks
  - Variable distributions and relationship checks
- Continue with **data cleansing based on common sense** after reviewing columns:
  - Missing values
  - Category normalization
  - Type conversion and invalid values
  - Outlier detection and treatment rationale
- End each cycle with an updated **proposal/recommendation** in Markdown.

## Working cadence

- Render notebooks regularly after meaningful progress.
- Commit and push in incremental checkpoints (EDA baseline, cleansing pass, modeling baseline, simulations/proposal updates).

## Notebook sequence

1. `notebooks/01_eda_cleansing.ipynb` - baseline EDA + cleansing
2. `notebooks/02_variable_eda.ipynb` - variable-by-variable deep profile
3. `notebooks/03_monotonicity_targets.ipynb` - monotonicity checks vs win/loss and binned amount targets
4. `notebooks/04_woe_iv_winloss.ipynb` - WoE/IV analysis for win/loss target
5. `notebooks/05_optbinning_amount.ipynb` - continuous optbinning and explained variance for amount target


## Observation zero

- Notebook: `../observation_0/00_observation0_basic_validation.ipynb`
- Markdown summary: `../observation_0/00_observation0_basic_validation.md`
- Amount maturity notebook: `../observation_0/01_amount_maturity_thresholds.ipynb`
- Amount maturity report: `../observation_0/01_amount_maturity_thresholds.md`
