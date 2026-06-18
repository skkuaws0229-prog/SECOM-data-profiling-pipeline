# Milestone 2: Manufacturing Data Quality Rules

Milestone 2 adds a practical feature-level data quality rule layer before any machine learning work.

This stage uses the actual loaded SECOM dataset structure:

- `sample_id` is generated locally and is excluded from feature quality decisions.
- `label` and `timestamp` are metadata columns, not sensor features.
- Columns named `sensor_000`, `sensor_001`, and so on are anonymized SECOM sensor features.
- No process meanings are inferred or assigned to anonymized features.

## Missingness Rules

Each sensor feature is assigned one missingness bucket:

| Bucket | Rule |
| --- | --- |
| `high_missing_drop_candidate` | `missing_ratio >= 0.80` |
| `medium_missing_review` | `0.30 <= missing_ratio < 0.80` |
| `low_missing_keep` | `missing_ratio < 0.30` |

## Constant Feature Rule

Each sensor feature also receives a constant-feature flag:

| Rule | Definition |
| --- | --- |
| `constant_drop_candidate` | `unique_non_null_count <= 1` |

The count ignores missing values. A feature with all missing values has zero non-null unique values and is therefore a constant drop candidate.

## Recommended Actions

The final `recommended_action` values are:

- `drop_candidate`
- `review`
- `keep`

Action priority:

1. If `is_constant` is `True`, `recommended_action = "drop_candidate"`.
2. Else if `missing_ratio >= 0.80`, `recommended_action = "drop_candidate"`.
3. Else if `missing_ratio >= 0.30`, `recommended_action = "review"`.
4. Else `recommended_action = "keep"`.

## Output Purpose

Milestone 2 does not physically drop columns. It creates a feature quality decision report for the next feature engineering milestone.
