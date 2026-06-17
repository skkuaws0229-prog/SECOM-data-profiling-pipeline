# Milestone 1 Design Decisions

## Scope

Milestone 1 is intentionally limited to SECOM raw file loading and data profiling. It does not include ML, RAG, agents, APIs, dashboards, Docker, MLflow, or database implementation.

## SECOM Feature Names

The UCI SECOM feature columns are anonymized. The loader assigns neutral column names using the pattern `sensor_000`, `sensor_001`, and so on based on the actual feature count in the raw file.

No process meanings are inferred or fabricated.

## Data Loading

The loader reads:

- `data/raw/secom.data`
- `data/raw/secom_labels.data`

It returns one pandas DataFrame containing `sample_id`, sensor columns, `label`, and `timestamp`.

## Profiling

The profiler reports:

- Row count
- Feature count
- Label distribution
- Missing value count and ratio by feature
- Duplicate row count
- Total missing values
- Overall missing value ratio

Missing-value metrics are calculated over sensor columns only. Duplicate detection excludes the generated `sample_id`.
