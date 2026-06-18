# Milestone 7 - Manufacturing Quality Report Automation

Milestone 7 generates a single integrated manufacturing quality summary report from the existing M1-M6 artifacts.

It reads the previously generated outputs for:

- data profiling
- data quality rules
- feature engineering
- modeling master table lineage
- baseline modeling
- threshold and error analysis
- explainability / feature importance

## Scope

M7 does not retrain models. It does not rerun threshold analysis or explainability analysis. It does not fabricate new metrics.

The report is an automation layer that summarizes existing structured outputs into one review-ready artifact.

## Purpose

The integrated report is intended for manufacturing quality review and portfolio presentation. It gives reviewers one place to inspect the end-to-end data and modeling readiness story without opening every milestone report individually.

## SECOM Sensor Warning

SECOM sensor features are anonymized. Sensor names such as `sensor_000` must not be interpreted as physical process meanings.

## Explainability Limitation

Feature importance is not causal proof. It describes model behavior and feature influence within the trained baseline models, but it does not prove physical manufacturing causes.

## Recommended Next Steps

The M7 report lists next steps including:

- stronger tabular models such as LightGBM or XGBoost
- cross-validation
- probability calibration
- cost-sensitive threshold selection
- dashboard/API development after structured reports are stable
- RAG/Agent workflows only after structured reporting is stable

## Outputs

```text
reports/manufacturing_quality_summary_report.md
reports/manufacturing_quality_summary.json
```
