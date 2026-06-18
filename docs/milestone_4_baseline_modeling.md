# Milestone 4: Baseline Defect Prediction Model

Milestone 4 trains reproducible baseline classifiers using the canonical modeling master table from Milestone 3.5:

```text
data/processed/modeling_master_table.csv
```

It uses rows where `split == "train"` for model fitting and rows where `split == "test"` for evaluation. The target is `label_binary`.

## Why Baselines First

Baseline models create a grounded reference point before more complex modeling work. They help verify that the processed data, label handling, evaluation code, and reports are working end to end before investing in feature selection, tuning, or advanced models.

This milestone uses:

- Logistic Regression with class balancing
- Random Forest with class balancing

It does not perform heavy hyperparameter tuning.

## Label Conversion

The original SECOM labels are:

- `-1`: pass / normal
- `1`: fail / defect

For binary classification they are converted to:

- `0`: pass / normal
- `1`: fail / defect

The positive class is defect label `1`.

In M4, this conversion is read from the master table as `label_binary`; `label_original` is retained in the master table for lineage.

## Why Accuracy Is Not Enough

Manufacturing defect prediction is typically imbalanced: normal samples are much more common than defect samples. A model can show high accuracy by predicting most samples as normal while still missing important defects.

Accuracy is useful, but it is not sufficient by itself.

## Metrics That Matter

Recall measures how many true defects are detected. Low recall means the model misses defects.

Precision measures how often predicted defects are actually defects. Low precision means more false alarms.

F1 balances precision and recall into a single score.

PR-AUC summarizes precision-recall performance across thresholds and is especially useful for imbalanced defect detection.

ROC-AUC is also reported using predicted defect probabilities, but PR-AUC is often more informative when the positive class is rare.

## False Negatives

A false negative means a defective sample is predicted as normal. In a manufacturing quality context, this is important because a missed defect can move downstream, increase scrap or rework, and create customer quality risk.

## Scope Boundary

This milestone does not implement RAG, agents, FastAPI, Docker, dashboards, or heavy model tuning. It does not invent meanings for anonymized SECOM sensor features.
