# SECOM Modeling Master Table Report

M3.5 builds a canonical modeling master table with lineage columns and selected imputed anonymized sensor features.

This stage does not train models and does not assign process meanings to SECOM features.

## Summary

- Row count: 1567
- Original feature count: 590
- Drop candidate feature count excluded: 124
- Keep feature count included: 442
- Review feature count included: 24
- Selected feature count: 466

## Lineage

- Feature set version: m3_5_feature_set_v1
- Quality rule version: m2_quality_rules_v1
- Preprocessing version: m3_5_preprocessing_v1
- Imputation strategy: median_train_only
- Median imputation values are fit on train rows only and applied to train/test rows.

## Split

- Method: stratified_train_test_split
- Test size: 0.2
- Random state: 42
- Train row count: 1253
- Test row count: 314

## Train Label Distribution

| label_binary | count |
| --- | --- |
| 0 | 1170 |
| 1 | 83 |

## Test Label Distribution

| label_binary | count |
| --- | --- |
| 0 | 293 |
| 1 | 21 |

## Label Mapping

- Original `-1` -> binary `0` normal
- Original `1` -> binary `1` defect
