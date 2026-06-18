# SECOM Threshold & Error Analysis Report

M5 analyzes how classification thresholds change defect detection performance on the SECOM test split.

Positive class is `label_binary == 1` defect. Models are fit on `split == train` and evaluated on `split == test` from `data/processed/modeling_master_table.csv`.

## Why Threshold Analysis Matters

Manufacturing defect detection is imbalanced, so the default threshold of 0.5 may not match operational risk. Lowering a threshold can increase recall and reduce missed defects, while usually increasing false positives.

Accuracy can be misleading because normal samples dominate the dataset. A model can look accurate while still missing many defect samples.

SECOM sensor names are anonymized and must not be interpreted as physical process meanings.

## Default Threshold 0.5

| model_name | threshold | accuracy | precision | recall | f1 | f2 | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.5 | 0.821656050955414 | 0.07317073170731707 | 0.14285714285714285 | 0.0967741935483871 | 0.12 | 18 |
| random_forest | 0.5 | 0.9331210191082803 | 0.0 | 0.0 | 0.0 | 0.0 | 21 |

## Best F2 Threshold Per Model

| model_name | threshold | accuracy | precision | recall | f1 | f2 | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.05 | 0.732484076433121 | 0.10126582278481013 | 0.38095238095238093 | 0.16 | 0.24539877300613497 | 13 |
| random_forest | 0.1 | 0.8248407643312102 | 0.17307692307692307 | 0.42857142857142855 | 0.24657534246575344 | 0.33088235294117646 | 12 |

## False Negative Reduction vs. Threshold 0.5

| model_name | default_threshold | default_false_negative | best_f2_threshold | best_f2_false_negative | false_negative_reduction |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.5 | 18 | 0.05 | 13 | 5 |
| random_forest | 0.5 | 21 | 0.1 | 12 | 9 |

## False Negative Rows

- False negative analysis rows written: 64
- See `reports/secom_false_negative_analysis.csv` for row-level details.
