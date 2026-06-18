# SECOM Baseline Defect Prediction Report

Milestone 4 trains reproducible baseline classifiers on the processed SECOM feature set.

Original SECOM labels are converted explicitly: `-1 -> 0` for pass/normal and `1 -> 1` for fail/defect.

## Scope

This milestone does not perform heavy hyperparameter tuning and does not add RAG, agents, APIs, dashboards, or Docker.
Random Forest uses `max_depth=5` as a conservative baseline regularization setting while keeping `n_estimators=300`, `class_weight="balanced"`, `random_state=42`, and `n_jobs=-1`.

## Processed Dataset

- Final feature count: 466
- Train row count: 1253
- Test row count: 314
- Imputation strategy: median
- Imputation fit on: X_train

## Metrics

| model_name | accuracy | precision | recall | f1 | average_precision | roc_auc | false_positive | false_negative | true_positive | true_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.821656 | 0.073171 | 0.142857 | 0.096774 | 0.098092 | 0.616447 | 38 | 18 | 3 | 255 |
| random_forest | 0.929936 | 0.000000 | 0.000000 | 0.000000 | 0.174064 | 0.716073 | 1 | 21 | 0 | 292 |

## Confusion Matrix

| model_name | true_negative | false_positive | false_negative | true_positive |
| --- | --- | --- | --- | --- |
| logistic_regression | 255 | 38 | 18 | 3 |
| random_forest | 292 | 1 | 21 | 0 |

## Evaluation Notes

- Positive class is defect label `1`.
- PR-AUC and ROC-AUC use predicted probabilities for defect label `1`.
- Accuracy is reported, but manufacturing defect prediction also needs recall, precision, F1, and PR-AUC because defects are rare.
- A false negative means a defective sample is predicted as normal, which can allow a quality issue to pass downstream inspection.
