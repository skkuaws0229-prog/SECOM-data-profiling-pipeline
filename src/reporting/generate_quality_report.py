"""Generate an integrated manufacturing quality summary report from M1-M6 artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_NAME = "manufacturing-ai-agent-platform"
DATASET_NAME = "UCI SECOM semiconductor manufacturing dataset"
REPORT_MARKDOWN_NAME = "manufacturing_quality_summary_report.md"
REPORT_JSON_NAME = "manufacturing_quality_summary.json"


def require_file(path: Path) -> None:
    """Fail clearly when a required artifact is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}. Run the relevant milestone script first."
        )


def load_csv(path: Path) -> pd.DataFrame:
    """Load a required CSV artifact."""
    require_file(path)
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    """Load a required JSON artifact."""
    require_file(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_markdown(path: Path) -> None:
    require_file(path)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def summarize_data_profile(reports_dir: Path) -> dict[str, Any]:
    """Summarize M1 data profiling artifacts."""
    _require_markdown(reports_dir / "secom_data_profile_report.md")
    missing = load_csv(reports_dir / "secom_missing_value_report.csv")
    label_distribution = load_csv(reports_dir / "secom_label_distribution.csv")

    return {
        "label_distribution": _records(label_distribution),
        "feature_count": int(len(missing)),
        "total_missing_values": int(missing["missing_count"].sum()),
        "top_missing_features": _records(
            missing.sort_values(
                ["missing_ratio", "missing_count", "feature"],
                ascending=[False, False, True],
            ).head(10)
        ),
    }


def summarize_quality_rules(reports_dir: Path) -> dict[str, Any]:
    """Summarize M2 quality-rule artifacts."""
    quality = load_csv(reports_dir / "secom_feature_quality_summary.csv")
    _require_markdown(reports_dir / "secom_quality_decision_report.md")

    action_counts = (
        quality["recommended_action"].value_counts().sort_index().to_dict()
    )
    bucket_counts = quality["missing_bucket"].value_counts().sort_index().to_dict()
    return {
        "feature_count": int(len(quality)),
        "recommended_action_counts": {
            str(key): int(value) for key, value in action_counts.items()
        },
        "missing_bucket_counts": {
            str(key): int(value) for key, value in bucket_counts.items()
        },
        "constant_drop_candidates": int(
            quality.loc[
                (quality["is_constant"].astype(str).str.lower() == "true")
                & (quality["recommended_action"] == "drop_candidate")
            ].shape[0]
        ),
    }


def summarize_feature_engineering(
    reports_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    """Summarize M3 feature engineering artifacts."""
    _require_markdown(reports_dir / "secom_feature_engineering_report.md")
    metadata = load_json(processed_dir / "feature_set_metadata.json")
    split = metadata.get("split", {})
    return {
        "original_feature_count": metadata.get("original_feature_count"),
        "drop_candidate_feature_count": metadata.get("drop_candidate_feature_count"),
        "kept_feature_count": metadata.get("kept_feature_count"),
        "review_feature_count_included": metadata.get(
            "review_feature_count_included"
        ),
        "final_feature_count": metadata.get("final_feature_count"),
        "train_row_count": split.get("train_row_count"),
        "test_row_count": split.get("test_row_count"),
        "imputation_strategy": metadata.get("imputation", {}).get("strategy"),
        "leakage_prevention_note": metadata.get("leakage_prevention_note"),
    }


def summarize_master_table(
    reports_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    """Summarize M3.5 modeling master table artifacts."""
    _require_markdown(reports_dir / "secom_modeling_master_table_report.md")
    metadata = load_json(processed_dir / "modeling_master_metadata.json")
    split = metadata.get("split", {})
    return {
        "row_count": metadata.get("row_count"),
        "selected_feature_count": metadata.get("selected_feature_count"),
        "lineage_columns": metadata.get("lineage_columns"),
        "feature_set_version": metadata.get("feature_set_version"),
        "quality_rule_version": metadata.get("quality_rule_version"),
        "preprocessing_version": metadata.get("preprocessing_version"),
        "imputation_strategy": metadata.get("imputation_strategy"),
        "train_row_count": split.get("train_row_count"),
        "test_row_count": split.get("test_row_count"),
        "train_label_distribution": split.get("train_label_distribution"),
        "test_label_distribution": split.get("test_label_distribution"),
    }


def summarize_baseline_modeling(reports_dir: Path) -> dict[str, Any]:
    """Summarize M4 baseline modeling artifacts."""
    metrics = load_csv(reports_dir / "secom_baseline_model_metrics.csv")
    confusion = load_csv(reports_dir / "secom_confusion_matrix.csv")
    _require_markdown(reports_dir / "secom_baseline_model_report.md")
    return {
        "metrics": _records(metrics),
        "confusion_matrix": _records(confusion),
        "best_by_average_precision": _records(
            metrics.sort_values("average_precision", ascending=False).head(1)
        ),
    }


def summarize_threshold_analysis(reports_dir: Path) -> dict[str, Any]:
    """Summarize M5 threshold/error analysis artifacts."""
    threshold_metrics = load_csv(reports_dir / "secom_threshold_metrics.csv")
    top_risk = load_csv(reports_dir / "secom_top_risk_samples.csv")
    false_negatives = load_csv(reports_dir / "secom_false_negative_analysis.csv")
    _require_markdown(reports_dir / "secom_threshold_error_analysis_report.md")

    default = threshold_metrics.loc[threshold_metrics["threshold"] == 0.5].copy()
    best_f2 = (
        threshold_metrics.sort_values(
            ["model_name", "f2", "threshold"],
            ascending=[True, False, True],
        )
        .groupby("model_name", as_index=False)
        .head(1)
    )
    return {
        "threshold_count": int(threshold_metrics["threshold"].nunique()),
        "default_threshold_metrics": _records(default),
        "best_f2_thresholds": _records(best_f2),
        "top_risk_sample_count": int(len(top_risk)),
        "false_negative_rows": int(len(false_negatives)),
        "false_negative_counts": _records(
            false_negatives.groupby(["model_name", "threshold_type"])
            .size()
            .rename("count")
            .reset_index()
        ),
    }


def summarize_explainability(reports_dir: Path) -> dict[str, Any]:
    """Summarize M6 explainability artifacts."""
    importance = load_csv(reports_dir / "secom_feature_importance.csv")
    top_features = load_csv(reports_dir / "secom_top_feature_summary.csv")
    error_profile = load_csv(reports_dir / "secom_error_feature_profile.csv")
    _require_markdown(reports_dir / "secom_explainability_report.md")

    return {
        "importance_row_count": int(len(importance)),
        "importance_types": sorted(importance["importance_type"].unique().tolist()),
        "top_feature_summary": _records(top_features.head(40)),
        "error_profile_groups": sorted(error_profile["group_name"].unique().tolist()),
        "profile_row_count": int(len(error_profile)),
        "consistent_top_features": _records(
            top_features.groupby("feature")
            .size()
            .rename("top_list_count")
            .reset_index()
            .sort_values(["top_list_count", "feature"], ascending=[False, True])
            .head(20)
        ),
    }


def _limitations() -> list[str]:
    return [
        "SECOM sensor features are anonymized.",
        "Sensor feature names must not be interpreted as physical process meanings.",
        "Feature importance is not causal proof.",
        "Baseline models are not production-ready.",
        "Accuracy is misleading in imbalanced defect prediction.",
        "False negatives are important manufacturing quality risks.",
        "Small defect count limits confidence in feature importance and error-profile comparisons.",
    ]


def _recommended_next_steps() -> list[str]:
    return [
        "Evaluate stronger tabular models such as LightGBM or XGBoost.",
        "Add cross-validation for more stable model and threshold estimates.",
        "Calibrate predicted probabilities before operational threshold selection.",
        "Perform cost-sensitive threshold selection with manufacturing quality stakeholders.",
        "Build a dashboard/API after structured reports and data contracts are stable.",
        "Add RAG/Agent workflows only after structured reports are stable.",
    ]


def build_summary_payload(project_root: Path) -> dict[str, Any]:
    """Build the integrated JSON-serializable report payload."""
    project_root = Path(project_root)
    reports_dir = project_root / "reports"
    processed_dir = project_root / "data" / "processed"

    return {
        "project_name": PROJECT_NAME,
        "dataset_name": DATASET_NAME,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "data_profile": summarize_data_profile(reports_dir),
        "quality_rules": summarize_quality_rules(reports_dir),
        "feature_engineering": summarize_feature_engineering(
            reports_dir,
            processed_dir,
        ),
        "modeling_master_table": summarize_master_table(reports_dir, processed_dir),
        "baseline_modeling": summarize_baseline_modeling(reports_dir),
        "threshold_error_analysis": summarize_threshold_analysis(reports_dir),
        "explainability": summarize_explainability(reports_dir),
        "limitations": _limitations(),
        "recommended_next_steps": _recommended_next_steps(),
    }


def _markdown_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "_No rows available._"
    df = pd.DataFrame.from_records(records)
    columns = [str(column) for column in df.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def render_markdown_report(payload: dict) -> str:
    """Render the integrated manufacturing quality summary report."""
    data_profile = payload["data_profile"]
    quality = payload["quality_rules"]
    features = payload["feature_engineering"]
    master = payload["modeling_master_table"]
    baseline = payload["baseline_modeling"]
    threshold = payload["threshold_error_analysis"]
    explainability = payload["explainability"]

    lines = [
        "# Manufacturing Quality Summary Report",
        "",
        f"- Project: {payload['project_name']}",
        f"- Dataset: {payload['dataset_name']}",
        f"- Generated at UTC: {payload['generated_at_utc']}",
        "",
        "## Executive Summary",
        "",
        "This report integrates SECOM data profiling, quality rules, feature engineering, baseline modeling, threshold analysis, and explainability outputs into one manufacturing quality summary.",
        "",
        "SECOM sensor features are anonymized. Sensor feature names must not be interpreted as physical process meanings.",
        "",
        "Baseline models are not production-ready. Accuracy is misleading in imbalanced defect prediction, and false negatives are important manufacturing quality risks.",
        "",
        "## Dataset Overview",
        "",
        f"- Sensor feature count profiled: {data_profile['feature_count']}",
        f"- Total missing values: {data_profile['total_missing_values']}",
        "",
        "Label distribution:",
        "",
        _markdown_table(data_profile["label_distribution"]),
        "",
        "## Data Quality Findings",
        "",
        f"- Quality-rule feature count: {quality['feature_count']}",
        f"- Recommended action counts: {quality['recommended_action_counts']}",
        f"- Missing bucket counts: {quality['missing_bucket_counts']}",
        f"- Constant drop candidates: {quality['constant_drop_candidates']}",
        "",
        "## Feature Engineering Summary",
        "",
        f"- Original feature count: {features['original_feature_count']}",
        f"- Drop candidate feature count: {features['drop_candidate_feature_count']}",
        f"- Final feature count: {features['final_feature_count']}",
        f"- Train rows: {features['train_row_count']}",
        f"- Test rows: {features['test_row_count']}",
        f"- Imputation strategy: {features['imputation_strategy']}",
        f"- Leakage prevention: {features['leakage_prevention_note']}",
        "",
        "## Modeling Master Table & Lineage",
        "",
        f"- Row count: {master['row_count']}",
        f"- Selected feature count: {master['selected_feature_count']}",
        f"- Feature set version: {master['feature_set_version']}",
        f"- Quality rule version: {master['quality_rule_version']}",
        f"- Preprocessing version: {master['preprocessing_version']}",
        f"- Imputation strategy: {master['imputation_strategy']}",
        f"- Lineage columns: {master['lineage_columns']}",
        "",
        "## Baseline Model Results",
        "",
        _markdown_table(baseline["metrics"]),
        "",
        "Feature importance is not causal proof; model results describe baseline behavior only.",
        "",
        "## Threshold & Error Analysis",
        "",
        "Default threshold metrics:",
        "",
        _markdown_table(threshold["default_threshold_metrics"]),
        "",
        "Best F2 thresholds:",
        "",
        _markdown_table(threshold["best_f2_thresholds"]),
        "",
        f"- False negative analysis rows: {threshold['false_negative_rows']}",
        "",
        "## Explainability / Feature Importance",
        "",
        f"- Feature importance rows: {explainability['importance_row_count']}",
        f"- Importance types: {explainability['importance_types']}",
        f"- Error profile groups: {explainability['error_profile_groups']}",
        "",
        "Consistent top features:",
        "",
        _markdown_table(explainability["consistent_top_features"]),
        "",
        "## Key Risks and Limitations",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["limitations"]])
    lines.extend(
        [
            "",
            "## Recommended Next Steps",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in payload["recommended_next_steps"]])
    lines.extend(
        [
            "",
            "## Reproducibility Checklist",
            "",
            "- M1 profiling artifacts were read from `reports/`.",
            "- M2 quality-rule artifacts were read from `reports/`.",
            "- M3 feature metadata was read from `data/processed/feature_set_metadata.json`.",
            "- M3.5 modeling master metadata was read from `data/processed/modeling_master_metadata.json`.",
            "- M4 baseline model artifacts were read without retraining.",
            "- M5 threshold/error artifacts were read without rerunning threshold analysis.",
            "- M6 explainability artifacts were read without rerunning explainability analysis.",
            "- No SECOM physical sensor meanings were invented.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_outputs(payload: dict, markdown: str, output_dir: Path) -> None:
    """Write M7 Markdown and JSON outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_MARKDOWN_NAME).write_text(markdown, encoding="utf-8")
    (output_dir / REPORT_JSON_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
