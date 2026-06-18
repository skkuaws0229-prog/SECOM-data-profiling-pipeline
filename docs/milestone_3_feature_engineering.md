# Milestone 3: Feature Engineering Pipeline

Milestone 3 creates a reproducible feature engineering pipeline for the UCI SECOM manufacturing dataset.

This milestone uses:

- Real SECOM raw files from `data/raw/secom.data` and `data/raw/secom_labels.data`
- Milestone 2 feature quality decisions from `reports/secom_feature_quality_summary.csv`

## Scope

This milestone prepares train/test feature matrices and labels. It does not implement ML modeling, RAG, agents, APIs, dashboards, Docker, or fabricated SECOM feature meanings.

## Feature Selection Rules

The generated and metadata columns are never used as model features:

- `sample_id`
- `label`
- `timestamp`

Sensor features with `recommended_action == "drop_candidate"` are excluded.

Sensor features with these actions are included:

- `keep`
- `review`

Review features are included so the next feature engineering/modeling milestone can make an explicit downstream decision rather than silently discarding them here.

## Train/Test Split

The split is stratified by `label`:

- `test_size = 0.2`
- `random_state = 42`
- stratify target: `label`

## Missing Value Handling

Missing values are handled with median imputation.

The median imputer is fit only on `X_train`, then applied to both `X_train` and `X_test`. This prevents data leakage from the test set into preprocessing decisions.

## Outputs

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
data/processed/feature_set_metadata.json
reports/secom_feature_engineering_report.md
```
