# Milestone 3.5: Modeling Master Table & Dataset Lineage

Milestone 3.5 creates a canonical `modeling_master_table` for the SECOM dataset. It is a row-level modeling artifact that keeps sample lineage, split assignment, binary labels, preprocessing versions, and selected imputed sensor features together in one table.

## Why a Modeling Master Table Is Needed

The split files from Milestone 3 are useful for training code because they provide separate `X_train`, `X_test`, `y_train`, and `y_test` files. Those files are compact and model-ready, but they separate features from row lineage.

The modeling master table keeps each sample in one canonical row so downstream analysis can trace predictions back to:

- `sample_id`
- timestamp
- original SECOM label
- binary defect label
- train/test split
- feature set and preprocessing versions

This makes the dataset easier to audit before moving into threshold analysis, false positive review, false negative review, and model error analysis.

## Split Files vs. Master Table

Milestone 3 split files:

- Optimized for model fitting and evaluation
- Store train/test matrices and labels separately
- Do not keep every lineage column beside each feature row

Milestone 3.5 modeling master table:

- Stores train and test rows in one table
- Includes lineage columns first
- Preserves selected, imputed anonymized sensor features
- Supports row-level joins with predictions and future error-analysis outputs

## Lineage Columns

The master table starts with these columns:

```text
sample_id
timestamp
label_original
label_binary
split
feature_set_version
quality_rule_version
preprocessing_version
imputation_strategy
```

After those lineage columns, the table appends selected imputed `sensor_*` features.

## Feature Selection

Feature selection uses the Milestone 2 quality summary at `reports/secom_feature_quality_summary.csv`.

Included:

- `recommended_action == "keep"`
- `recommended_action == "review"`

Excluded:

- `recommended_action == "drop_candidate"`

The anonymized SECOM sensor features are not assigned fabricated process meanings.

## Label Conversion

Original SECOM labels are converted explicitly:

- `-1` -> `0` normal
- `1` -> `1` defect

Both `label_original` and `label_binary` are retained for auditability.

## Train-Only Median Imputation

The train/test split is created with:

- `test_size = 0.2`
- `random_state = 42`
- stratification by binary label

Median values are fit on train rows only. Those train medians are then applied to both train and test rows. This prevents leakage from test rows into preprocessing decisions.

## Support for Threshold and Error Analysis

Future modeling stages can join prediction outputs back to the master table by `sample_id`. This supports:

- Threshold analysis across defect probability scores
- False positive review
- False negative review
- Error analysis by split, label, or sensor feature values
- Reproducible reporting of which feature set and preprocessing rules created the dataset

## Outputs

```text
data/processed/modeling_master_table.csv
data/processed/modeling_master_metadata.json
reports/secom_modeling_master_table_report.md
```

`modeling_master_table.csv` is generated locally and ignored by Git.
