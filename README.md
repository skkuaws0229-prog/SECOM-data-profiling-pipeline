# manufacturing-ai-agent-platform

Professional portfolio project for a Manufacturing AI/Data role using the UCI SECOM semiconductor manufacturing dataset.

## Project Goal

This project is intended to grow into an end-to-end manufacturing AI/Data platform demonstrating data engineering, data quality validation, ML defect prediction readiness, PostgreSQL schema readiness, RAG readiness, agent workflow readiness, and LLMOps/observability readiness.

## Milestone 1: SECOM Data Loading and Profiling

Milestone 1 implements only the local SECOM data loading and profiling pipeline.

Included now:

- Load raw SECOM feature data from `data/raw/secom.data`
- Load SECOM labels and timestamps from `data/raw/secom_labels.data`
- Assign anonymized sensor feature names such as `sensor_000`, `sensor_001`, etc.
- Produce basic data quality/profile outputs
- Provide focused pytest tests using temporary sample files

Not included yet:

- Machine learning
- RAG
- Agent workflows
- API
- Dashboard
- Docker
- MLflow
- PostgreSQL implementation

## Raw Data Placement

Download the UCI SECOM files separately and place them here:

```text
data/raw/secom.data
data/raw/secom_labels.data
```

The repository does not include raw SECOM data and does not create fake replacement data.

## Run Profiling

Install the Milestone 1 dependencies:

```bash
pip install -r requirements.txt
```

Run the profiler from the project root:

```bash
python scripts/run_profile_secom.py
```

Expected outputs:

```text
reports/secom_data_profile_report.md
reports/secom_missing_value_report.csv
reports/secom_label_distribution.csv
```

## Milestone 2: Manufacturing Data Quality Rules

Milestone 2 adds a feature-level quality rule layer before ML modeling. It evaluates anonymized SECOM sensor columns for missingness and constant values, then creates a decision report.

This milestone is quality-rule design only. It does not train models, drop columns, build APIs, add RAG, add agents, or create dashboards.

Run the quality rules from the project root:

```bash
python scripts/run_quality_rules_secom.py
```

Expected outputs:

```text
reports/secom_feature_quality_summary.csv
reports/secom_quality_decision_report.md
```

## Design Notes

SECOM feature columns are anonymized. This project names them as generic sensor features, for example `sensor_000`, and does not assign fabricated manufacturing process meanings.

The generated `sample_id` is a local row identifier used for traceability. Profiling treats duplicate rows as duplicates across the measured data, label, and timestamp, excluding the generated `sample_id`.

Missing-value totals and ratios are calculated across anonymized sensor feature columns.
