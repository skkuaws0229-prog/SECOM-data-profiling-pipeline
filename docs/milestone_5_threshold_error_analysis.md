# Milestone 5 - Threshold & Error Analysis

Milestone 5 analyzes how classification thresholds affect defect detection performance on the SECOM test split.

M5 uses this canonical dataset:

```text
data/processed/modeling_master_table.csv
```

It fits models only on rows where `split == "train"` and evaluates only on rows where `split == "test"`. The positive class is `label_binary == 1`, representing defect.

## Why Accuracy Is Misleading

SECOM defect prediction is imbalanced: normal samples are much more common than defect samples. A model can achieve high accuracy by predicting most samples as normal, even if it misses many actual defects.

For manufacturing quality use cases, accuracy alone can hide the operational risk of missed defects.

## Why Recall and False Negatives Matter

Recall measures how many true defects are detected. A low-recall model misses more defects.

A false negative means `label_binary == 1` defect was predicted as normal. In manufacturing quality, this matters because a missed defect may move downstream, causing rework, scrap, customer quality risk, or delayed root-cause investigation.

## Threshold 0.5 Is Only a Default

The default classification threshold of `0.5` is only one operating point. It is not automatically the best choice for imbalanced defect detection.

Milestone 5 evaluates thresholds from `0.05` to `0.95` in steps of `0.05` so the trade-off can be inspected rather than assumed.

## Precision / Recall Trade-Off

Lowering the threshold usually predicts more samples as defects. This can improve recall and reduce false negatives, but it may also increase false positives and lower precision.

Raising the threshold usually predicts fewer samples as defects. This can improve precision, but it may miss more actual defects.

The right operating threshold depends on the cost of false negatives versus false positives.

## Why F2 Is Used

F2 is a recall-weighted score. It gives recall more importance than precision, which is useful when missed defects are more costly than false alarms.

Milestone 5 reports the best F2 threshold per model to highlight operating points that prioritize defect detection.

## How the Modeling Master Table Is Used

The analysis uses `modeling_master_table.csv` because it keeps features and row lineage together:

- `sample_id`
- `timestamp`
- `label_original`
- `label_binary`
- `split`
- feature/preprocessing version columns
- selected imputed `sensor_*` features

This avoids relying on older split files and makes threshold/error outputs traceable back to sample-level records.

## Top-Risk Samples and False Negatives

Top-risk samples are test rows sorted by predicted defect probability in descending order. They help prioritize inspection candidates by showing which samples the model considers most defect-like.

False negative analysis compares:

- default threshold `0.5`
- best F2 threshold per model

It lists which defect samples were missed because the model predicted them as normal. These rows are a practical starting point for error analysis and threshold selection.

## SECOM Feature Interpretation

SECOM sensor names are anonymized. They must not be interpreted as physical process meanings unless validated by authoritative process documentation.

## Outputs

```text
reports/secom_threshold_metrics.csv
reports/secom_top_risk_samples.csv
reports/secom_false_negative_analysis.csv
reports/secom_threshold_error_analysis_report.md
```
