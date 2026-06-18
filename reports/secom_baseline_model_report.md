# SECOM Baseline Defect Prediction Report

Milestone 4 trains reproducible baseline classifiers from the canonical SECOM modeling master table.

M4 uses `data/processed/modeling_master_table.csv` as the canonical dataset. It trains on rows where `split == train`, evaluates on rows where `split == test`, and uses `label_binary` as the target.

The modeling master table preserves original SECOM label lineage: `label_original == -1` maps to `label_binary == 0` normal and `label_original == 1` maps to `label_binary == 1` defect.

## Scope

This milestone does not perform heavy hyperparameter tuning and does not add RAG, agents, APIs, dashboards, or Docker.

## Canonical Dataset

- Modeling master row count: 1567
- Selected sensor feature count: 466
- Train row count: 1253
- Test row count: 314
- Imputation strategy: median_train_only
- Feature set version: m3_5_feature_set_v1

## Metrics

| model_name | accuracy | precision | recall | f1 | average_precision | roc_auc | false_positive | false_negative | true_positive | true_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.821656 | 0.073171 | 0.142857 | 0.096774 | 0.098092 | 0.616447 | 38 | 18 | 3 | 255 |
| random_forest | 0.933121 | 0.000000 | 0.000000 | 0.000000 | 0.224757 | 0.735901 | 0 | 21 | 0 | 293 |

## Confusion Matrix

| model_name | true_negative | false_positive | false_negative | true_positive |
| --- | --- | --- | --- | --- |
| logistic_regression | 255 | 38 | 18 | 3 |
| random_forest | 293 | 0 | 21 | 0 |

## Evaluation Notes

- Positive class is defect label `1`.
- PR-AUC and ROC-AUC use predicted probabilities for defect label `1`.
- Accuracy is reported, but manufacturing defect prediction also needs recall, precision, F1, and PR-AUC because defects are rare.
- A false negative means a defective sample is predicted as normal, which can allow a quality issue to pass downstream inspection.
