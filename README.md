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
