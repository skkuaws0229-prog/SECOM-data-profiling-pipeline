from pathlib import Path

import pandas as pd
import pytest

from src.quality.profile_st_awfd import (
    REFERENCE_COLUMNS,
    SUMMARY_KEYS,
    build_master_table_design_decision,
    get_feature_columns,
    profile_st_awfd_csv,
    require_columns,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_required_column_validation() -> None:
    with pytest.raises(ValueError, match="Missing required"):
        require_columns(["MaterialID", "StepID", "target"])

    require_columns(REFERENCE_COLUMNS + ["feature_1"])


def test_feature_column_selection_excludes_reference_columns() -> None:
    columns = REFERENCE_COLUMNS + ["feature_1", "feature_2"]

    assert get_feature_columns(columns) == ["feature_1", "feature_2"]


def test_material_id_target_consistency_detection(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_csv(
        csv_path,
        [
            {"MaterialID": "A", "StepID": 1, "duration_ms": 10, "feature_1": 1, "target": 0, "is_test": 0},
            {"MaterialID": "A", "StepID": 2, "duration_ms": 11, "feature_1": 2, "target": 1, "is_test": 0},
        ],
    )

    summary = profile_st_awfd_csv(csv_path, "test", chunksize=1)

    assert summary["target_consistent_by_material_id"] is False
    assert summary["mixed_target_material_id_count"] == 1


def test_material_id_is_test_consistency_detection(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_csv(
        csv_path,
        [
            {"MaterialID": "A", "StepID": 1, "duration_ms": 10, "feature_1": 1, "target": 0, "is_test": 0},
            {"MaterialID": "A", "StepID": 2, "duration_ms": 11, "feature_1": 2, "target": 0, "is_test": 1},
        ],
    )

    summary = profile_st_awfd_csv(csv_path, "test", chunksize=1)

    assert summary["is_test_consistent_by_material_id"] is False
    assert summary["mixed_is_test_material_id_count"] == 1


def test_profile_summary_required_keys(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_csv(
        csv_path,
        [
            {"MaterialID": "A", "StepID": 1, "duration_ms": 10, "feature_1": 1, "target": 0, "is_test": 0},
            {"MaterialID": "B", "StepID": 1, "duration_ms": None, "feature_1": None, "target": 1, "is_test": 1},
        ],
    )

    summary = profile_st_awfd_csv(csv_path, "test", chunksize=1)

    for key in SUMMARY_KEYS:
        assert key in summary
    assert summary["row_count"] == 2
    assert summary["duration_ms"]["missing_count"] == 1


def test_master_table_design_report_contains_all_six_decisions(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    _write_csv(
        csv_path,
        [
            {"MaterialID": "A", "StepID": 1, "duration_ms": 10, "feature_1": 1, "target": 0, "is_test": 0},
            {"MaterialID": "A", "StepID": 2, "duration_ms": 11, "feature_1": 2, "target": 0, "is_test": 0},
        ],
    )
    summary = profile_st_awfd_csv(csv_path, "test", chunksize=1)

    report = build_master_table_design_decision([summary])

    for decision_number in ["1.", "2.", "3.", "4.", "5.", "6."]:
        assert f"## {decision_number}" in report
