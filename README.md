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

## Milestone 3: Feature Engineering Pipeline

Milestone 3 creates reproducible train/test feature sets using the Milestone 2 quality decisions. It excludes generated metadata columns, removes `drop_candidate` features, keeps `keep` and `review` features, stratifies the split by `label`, and applies median imputation fit on `X_train` only.

This milestone prepares data for later modeling. It does not train ML models, build RAG, add agents, expose APIs, use Docker, or assign fabricated process meanings to anonymized SECOM features.

Run the feature engineering pipeline from the project root:

```bash
python scripts/run_build_features_secom.py
```

Expected outputs:

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
data/processed/feature_set_metadata.json
reports/secom_feature_engineering_report.md
```

## Milestone 3.5: Modeling Master Table & Dataset Lineage

Milestone 3.5 creates a canonical modeling master table that keeps each SECOM sample, lineage columns, split assignment, binary labels, preprocessing versions, and selected imputed sensor features together in one row-level artifact.

The split files from Milestone 3 are optimized for model training. The modeling master table is optimized for auditability, threshold analysis, and future false positive / false negative error analysis.

Run the modeling master table builder from the project root:

```bash
python scripts/run_build_modeling_master_secom.py
```

Expected outputs:

```text
data/processed/modeling_master_table.csv
data/processed/modeling_master_metadata.json
reports/secom_modeling_master_table_report.md
```

`modeling_master_table.csv` is generated locally and ignored by Git.

## Milestone 4: Baseline Defect Prediction Model

Milestone 4 trains reproducible baseline defect prediction models using the processed train/test files from Milestone 3. Original SECOM labels are converted from `-1/1` into binary labels where `0` means pass/normal and `1` means fail/defect.

This milestone is baseline modeling only. It does not add RAG, agents, FastAPI, Docker, dashboards, or heavy hyperparameter tuning.

Run baseline training from the project root:

```bash
python scripts/run_train_baseline_secom.py
```

Expected outputs:

```text
reports/secom_baseline_model_metrics.csv
reports/secom_confusion_matrix.csv
reports/secom_prediction_sample.csv
reports/secom_baseline_model_report.md
```

Main evaluation metrics:

- Accuracy
- Precision
- Recall
- F1
- Average precision / PR-AUC
- ROC-AUC
- False positives and false negatives

## Milestone 5 - Threshold & Error Analysis

Milestone 5 analyzes how different classification thresholds affect defect detection on the SECOM test split. It uses `data/processed/modeling_master_table.csv` as the canonical dataset, fits models on `split == train`, evaluates on `split == test`, and treats `label_binary == 1` as the positive defect class.

Accuracy can be misleading in imbalanced SECOM defect prediction, so the analysis emphasizes recall, false negatives, F2, and the precision/recall trade-off across thresholds. Threshold `0.5` is only a default operating point; lowering the threshold can reduce false negatives but may increase false positives. Top-risk samples help prioritize inspection candidates, and false negative analysis shows which defect samples were missed. SECOM sensor names are anonymized and must not be interpreted as physical process meanings.

Run threshold and error analysis from the project root:

```bash
python scripts/run_threshold_analysis_secom.py
```

Expected outputs:

```text
reports/secom_threshold_metrics.csv
reports/secom_top_risk_samples.csv
reports/secom_false_negative_analysis.csv
reports/secom_threshold_error_analysis_report.md
```

## Design Notes

SECOM feature columns are anonymized. This project names them as generic sensor features, for example `sensor_000`, and does not assign fabricated manufacturing process meanings.

The generated `sample_id` is a local row identifier used for traceability. Profiling treats duplicate rows as duplicates across the measured data, label, and timestamp, excluding the generated `sample_id`.

Missing-value totals and ratios are calculated across anonymized sensor feature columns.
