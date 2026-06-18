# Milestone 6 - Explainability / Feature Importance

Milestone 6 analyzes feature importance for baseline SECOM defect prediction models and compares selected feature profiles for top-risk and false-negative test samples.

M6 uses this canonical dataset:

```text
data/processed/modeling_master_table.csv
```

The analysis fits models only on rows where `split == "train"`. It explains and evaluates only on rows where `split == "test"`. The positive class is `label_binary == 1`, representing defect.

## Anonymized SECOM Sensors

SECOM sensor names are anonymized and must not be interpreted as physical process meanings. Feature names such as `sensor_000` are treated only as anonymized sensor variables.

## Feature Importance Methods

Logistic Regression importance uses absolute standardized coefficient magnitude. The model is fit with a `StandardScaler` trained only on train rows, so coefficient magnitudes are comparable across scaled features.

Random Forest importance uses built-in impurity-based `feature_importances_`. This reflects how much features contribute to impurity reduction across trees.

Permutation importance, when generated, measures the performance drop after shuffling one feature. In this project it uses average precision on the test split, so larger values indicate features whose disruption hurts defect-ranking performance more.

## Top-Risk and False-Negative Profiles

Top-risk samples show what the model considers high-risk by sorting test samples by predicted defect probability.

False-negative profiles help inspect actual defect samples missed by the model. These are rows where `label_binary == 1` but the model predicted normal at the analyzed threshold.

The error feature profile compares:

- top-risk samples
- false-negative samples
- all test samples

This gives a structured way to inspect whether missed defects differ from high-risk or overall test samples across important anonymized sensor features.

## Limitations

Feature importance is not causal proof. It can show model influence or association, but it does not prove a physical manufacturing cause.

The small defect count limits confidence in feature rankings and group-level profile comparisons. Results should be treated as baseline analysis signals rather than final manufacturing conclusions.

Anonymized sensors cannot be interpreted as physical process variables without authoritative process documentation.

## Outputs

```text
reports/secom_feature_importance.csv
reports/secom_top_feature_summary.csv
reports/secom_error_feature_profile.csv
reports/secom_explainability_report.md
```
