# Milestone 12 - ST-AWFD Retrospective Baseline Models

Milestone 12 trains separate retrospective baseline models for ST-AWFD D1 and D2 using only the MaterialID-level master tables created in M11.

## Scope

Inputs:

```text
data/processed/st_awfd_d1/material_master.csv
data/processed/st_awfd_d2/material_master.csv
```

D1 and D2 are run as separate pipelines. This milestone does not merge D1/D2, merge with SECOM, use raw event CSVs, change master tables, or impute values into the persisted master tables.

The models are full-process retrospective baselines. They are useful for establishing a first modeling reference point after a material has completed the process, not for operational early-warning decisions.

## Validation

Before training, the pipeline verifies:

- Required lineage columns exist.
- `material_id` is unique.
- `label_binary` contains only `0` and `1`.
- `official_split` contains only `train` and `test`.
- Train and test MaterialIDs do not overlap.
- Train and test splits both contain samples.

Feature columns are selected dynamically from numeric columns only. Metadata, labels, split columns, path columns, and JSON columns are excluded.

## Preprocessing

Median imputation is fit only on official train rows and then applied to train/test rows.

Logistic Regression uses a `StandardScaler` fit only on official train rows. Random Forest does not use scaling.

## Models

- Logistic Regression with `class_weight="balanced"` and `max_iter=2000`
- Random Forest with `n_estimators=300`, `class_weight="balanced"`, `random_state=42`, and `n_jobs=-1`

## Evaluation

Evaluation uses the official test split only and the default threshold `0.50`.

Metrics include:

- Accuracy
- Precision
- Recall
- F1
- F2
- Average precision / PR-AUC
- ROC-AUC
- True positives, false positives, true negatives, false negatives

If the official test split contains only one class, probability metrics that require both classes are written as null with a warning instead of fabricated values.

If the official train split contains only one class, supervised Logistic Regression and Random Forest training is skipped for that dataset/model family. The pipeline still writes the requested report structure, prediction file, metadata, and joblib artifact, but model metrics are null with a warning. This is preferable to fabricating a two-class model from data that does not contain both classes in the official training split.

## Limitations

Default threshold `0.50` is not an operationally approved threshold. Class imbalance can make accuracy misleading, and false negatives require separate threshold and cost analysis in the next milestone.

D1/D2 results must not be directly treated as the same production environment.

## Outputs

```text
reports/st_awfd_d1_baseline_metrics.csv
reports/st_awfd_d1_baseline_confusion_matrix.csv
reports/st_awfd_d1_baseline_model_report.md
reports/st_awfd_d1_baseline_predictions.csv
data/processed/st_awfd_d1/baseline_feature_metadata.json
models/st_awfd_d1/logistic_regression.joblib
models/st_awfd_d1/random_forest.joblib

reports/st_awfd_d2_baseline_metrics.csv
reports/st_awfd_d2_baseline_confusion_matrix.csv
reports/st_awfd_d2_baseline_model_report.md
reports/st_awfd_d2_baseline_predictions.csv
data/processed/st_awfd_d2/baseline_feature_metadata.json
models/st_awfd_d2/logistic_regression.joblib
models/st_awfd_d2/random_forest.joblib

reports/st_awfd_baseline_comparison.md
```
