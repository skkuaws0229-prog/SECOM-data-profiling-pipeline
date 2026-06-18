# ST-AWFD st_awfd_d1 Retrospective Baseline Model Report

This milestone trains full-process retrospective baseline models from the MaterialID-level master table. It does not use raw event CSV files, merge D1/D2, merge with SECOM, or modify master tables.

## Dataset

- Dataset ID: st_awfd_d1
- Material rows: 5104
- Selected numeric model features: 185
- Train rows: 3297
- Test rows: 1807
- Train label distribution: {'0': 3297}
- Test label distribution: {'0': 1805, '1': 2}

## Preprocessing

- Median imputation is fit only on official train rows and then applied to train/test features.
- Logistic Regression uses StandardScaler fit only on official train rows.
- Random Forest does not use scaling.

## Metrics at Default Threshold 0.50

| dataset_id | model_name | threshold | accuracy | precision | recall | f1 | f2 | average_precision | roc_auc | true_positive | false_positive | true_negative | false_negative | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| st_awfd_d1 | logistic_regression | 0.500000 |  |  |  |  |  |  |  |  |  |  |  | official train split contains one class only; supervised baseline training skipped |
| st_awfd_d1 | random_forest | 0.500000 |  |  |  |  |  |  |  |  |  |  |  | official train split contains one class only; supervised baseline training skipped |

## Confusion Matrix

| dataset_id | model_name | true_positive | false_positive | true_negative | false_negative |
| --- | --- | --- | --- | --- | --- |
| st_awfd_d1 | logistic_regression |  |  |  |  |
| st_awfd_d1 | random_forest |  |  |  |  |

## Limitations

- This is a full-process retrospective baseline, not an operational early-warning model.
- If the official train split contains only one label class, supervised Logistic Regression and Random Forest training is skipped and metrics are left null with a warning.
- Default threshold 0.50 is not an operationally approved threshold.
- Class imbalance can make accuracy misleading.
- False negatives require separate threshold and cost analysis in the next milestone.
- D1/D2 results must not be directly treated as the same production environment.
