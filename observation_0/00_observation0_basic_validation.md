# Observation 0 - Basic ID Validation

## Check performed

Validate whether `Opportunity Number` behaves as a unique row identifier.

## Result

- Total rows: **78,025**
- Unique `Opportunity Number` values: **77,829**
- Is unique (`rows == unique_ids`): **No**
- Rows participating in duplicates: **379**
- Distinct duplicated IDs: **183**
- Maximum repetition of a single ID: **4**

## Interpretation

`Opportunity Number` is **not** a strict primary key in this dataset.
Modeling and process analytics should treat it as a business/opportunity reference that may appear multiple times (e.g., updates, process events, or repeated registrations).

## Artifact

- Full duplicated-ID list: `data/processed/observation0_opportunity_duplicates.csv`
