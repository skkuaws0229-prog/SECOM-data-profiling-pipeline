import json
from pathlib import Path

import pandas as pd
import pytest

from src.reporting.generate_quality_report import (
    build_summary_payload,
    render_markdown_report,
    require_file,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, content: str = "report") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_artifacts(root: Path) -> None:
    reports = root / "reports"
    processed = root / "data" / "processed"

    _write_text(reports / "secom_data_profile_report.md")
    _write_csv(
        reports / "secom_missing_value_report.csv",
        [{"feature": "sensor_000", "missing_count": 1, "missing_ratio": 0.1}],
    )
    _write_csv(
        reports / "secom_label_distribution.csv",
        [{"label": -1, "count": 8}, {"label": 1, "count": 2}],
    )

    _write_csv(
        reports / "secom_feature_quality_summary.csv",
        [
            {
                "feature": "sensor_000",
                "missing_bucket": "low_missing_keep",
                "recommended_action": "keep",
                "is_constant": False,
            },
            {
                "feature": "sensor_001",
                "missing_bucket": "high_missing_drop_candidate",
                "recommended_action": "drop_candidate",
                "is_constant": True,
            },
        ],
    )
    _write_text(reports / "secom_quality_decision_report.md")

    _write_text(reports / "secom_feature_engineering_report.md")
    _write_json(
        processed / "feature_set_metadata.json",
        {
            "original_feature_count": 2,
            "drop_candidate_feature_count": 1,
            "kept_feature_count": 1,
            "review_feature_count_included": 0,
            "final_feature_count": 1,
            "split": {"train_row_count": 8, "test_row_count": 2},
            "imputation": {"strategy": "median"},
            "leakage_prevention_note": "fit on train only",
        },
    )

    _write_text(reports / "secom_modeling_master_table_report.md")
    _write_json(
        processed / "modeling_master_metadata.json",
        {
            "row_count": 10,
            "selected_feature_count": 1,
            "lineage_columns": ["sample_id", "timestamp"],
            "feature_set_version": "v1",
            "quality_rule_version": "v1",
            "preprocessing_version": "v1",
            "imputation_strategy": "median_train_only",
            "split": {
                "train_row_count": 8,
                "test_row_count": 2,
                "train_label_distribution": {"0": 7, "1": 1},
                "test_label_distribution": {"0": 1, "1": 1},
            },
        },
    )

    _write_csv(
        reports / "secom_baseline_model_metrics.csv",
        [
            {
                "model_name": "model",
                "accuracy": 0.8,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "average_precision": 0.4,
                "roc_auc": 0.6,
            }
        ],
    )
    _write_csv(
        reports / "secom_confusion_matrix.csv",
        [
            {
                "model_name": "model",
                "true_negative": 1,
                "false_positive": 0,
                "false_negative": 1,
                "true_positive": 0,
            }
        ],
    )
    _write_text(reports / "secom_baseline_model_report.md")

    _write_csv(
        reports / "secom_threshold_metrics.csv",
        [
            {
                "model_name": "model",
                "threshold": 0.5,
                "accuracy": 0.8,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "f2": 0.5,
                "fn": 1,
            }
        ],
    )
    _write_csv(reports / "secom_top_risk_samples.csv", [{"sample_id": 1}])
    _write_csv(
        reports / "secom_false_negative_analysis.csv",
        [{"model_name": "model", "threshold_type": "default_0.5"}],
    )
    _write_text(reports / "secom_threshold_error_analysis_report.md")

    _write_csv(
        reports / "secom_feature_importance.csv",
        [
            {
                "model_name": "model",
                "importance_type": "coefficient_abs",
                "feature": "sensor_000",
                "importance": 1.0,
                "rank": 1,
            }
        ],
    )
    _write_csv(
        reports / "secom_top_feature_summary.csv",
        [
            {
                "model_name": "model",
                "importance_type": "coefficient_abs",
                "feature": "sensor_000",
                "importance": 1.0,
                "rank": 1,
            }
        ],
    )
    _write_csv(
        reports / "secom_error_feature_profile.csv",
        [
            {
                "group_name": "all_test_samples",
                "feature": "sensor_000",
                "mean": 1.0,
            }
        ],
    )
    _write_text(reports / "secom_explainability_report.md")


def test_required_output_sections_exist_in_markdown(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    markdown = render_markdown_report(build_summary_payload(tmp_path))

    for section in [
        "Executive Summary",
        "Dataset Overview",
        "Data Quality Findings",
        "Feature Engineering Summary",
        "Modeling Master Table & Lineage",
        "Baseline Model Results",
        "Threshold & Error Analysis",
        "Explainability / Feature Importance",
        "Key Risks and Limitations",
        "Recommended Next Steps",
        "Reproducibility Checklist",
    ]:
        assert section in markdown


def test_json_payload_contains_required_top_level_keys(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    payload = build_summary_payload(tmp_path)

    for key in [
        "project_name",
        "dataset_name",
        "generated_at_utc",
        "data_profile",
        "quality_rules",
        "feature_engineering",
        "modeling_master_table",
        "baseline_modeling",
        "threshold_error_analysis",
        "explainability",
        "limitations",
        "recommended_next_steps",
    ]:
        assert key in payload


def test_missing_required_input_file_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "reports" / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Required artifact not found"):
        require_file(missing)


def test_limitations_include_anonymized_sensor_warning(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    payload = build_summary_payload(tmp_path)

    assert any("anonymized" in item for item in payload["limitations"])
    assert any("physical process meanings" in item for item in payload["limitations"])


def test_recommended_next_steps_are_present(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    payload = build_summary_payload(tmp_path)

    assert any("LightGBM" in item or "XGBoost" in item for item in payload["recommended_next_steps"])
    assert any("cross-validation" in item for item in payload["recommended_next_steps"])
    assert any("RAG/Agent" in item for item in payload["recommended_next_steps"])


def test_markdown_contains_reproducibility_checklist(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    markdown = render_markdown_report(build_summary_payload(tmp_path))

    assert "## Reproducibility Checklist" in markdown


def test_summary_payload_does_not_contain_fabricated_physical_sensor_meanings(
    tmp_path: Path,
) -> None:
    _make_artifacts(tmp_path)
    payload_text = json.dumps(build_summary_payload(tmp_path)).lower()

    for forbidden in ["temperature", "pressure", "etch", "deposition"]:
        assert forbidden not in payload_text
