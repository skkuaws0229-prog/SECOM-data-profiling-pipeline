# ST-AWFD Retrospective Baseline Comparison

D1 and D2 were trained and evaluated as separate pipelines. This report compares outputs side by side, but it does not merge datasets or treat them as the same production environment.

| dataset_id | model_name | accuracy | precision | recall | f1 | f2 | average_precision | roc_auc | false_negative | false_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| st_awfd_d1 | logistic_regression |  |  |  |  |  |  |  |  |  |
| st_awfd_d1 | random_forest |  |  |  |  |  |  |  |  |  |
| st_awfd_d2 | logistic_regression |  |  |  |  |  |  |  |  |  |
| st_awfd_d2 | random_forest |  |  |  |  |  |  |  |  |  |

Accuracy can be misleading when defects are imbalanced. False negatives require threshold and cost analysis before any operational decision.
