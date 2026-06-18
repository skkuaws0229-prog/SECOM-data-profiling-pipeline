import pandas as pd

from src.quality.apply_quality_rules import (
    assign_recommended_action,
    build_feature_quality_summary,
    get_sensor_columns,
)


def test_get_sensor_columns_excludes_generated_and_metadata_columns() -> None:
    data = pd.DataFrame(
        {
            "sample_id": [0],
            "sensor_000": [1.0],
            "sensor_001": [2.0],
            "label": [-1],
            "timestamp": ["19/07/2008 11:55:00"],
        }
    )

    assert get_sensor_columns(data) == ["sensor_000", "sensor_001"]


def test_build_feature_quality_summary_assigns_missing_buckets() -> None:
    data = pd.DataFrame(
        {
            "sensor_000": [1.0, 2.0, 3.0, 4.0, None],
            "sensor_001": [1.0, 2.0, None, None, None],
            "sensor_002": [1.0, None, None, None, None],
        }
    )

    summary = build_feature_quality_summary(data).set_index("feature")

    assert summary.loc["sensor_000", "missing_bucket"] == "low_missing_keep"
    assert summary.loc["sensor_001", "missing_bucket"] == "medium_missing_review"
    assert (
        summary.loc["sensor_002", "missing_bucket"]
        == "high_missing_drop_candidate"
    )


def test_build_feature_quality_summary_detects_constant_features() -> None:
    data = pd.DataFrame(
        {
            "sensor_000": [7.0, 7.0, None],
            "sensor_001": [1.0, 2.0, 3.0],
        }
    )

    summary = build_feature_quality_summary(data).set_index("feature")

    assert bool(summary.loc["sensor_000", "is_constant"]) is True
    assert summary.loc["sensor_000", "unique_non_null_count"] == 1
    assert bool(summary.loc["sensor_001", "is_constant"]) is False


def test_assign_recommended_action_prioritizes_constant_features() -> None:
    row = pd.Series({"is_constant": True, "missing_ratio": 0.10})

    assert assign_recommended_action(row) == "drop_candidate"


def test_assign_recommended_action_drops_high_missing_features() -> None:
    row = pd.Series({"is_constant": False, "missing_ratio": 0.80})

    assert assign_recommended_action(row) == "drop_candidate"


def test_assign_recommended_action_reviews_medium_missing_features() -> None:
    row = pd.Series({"is_constant": False, "missing_ratio": 0.30})

    assert assign_recommended_action(row) == "review"


def test_assign_recommended_action_keeps_low_missing_features() -> None:
    row = pd.Series({"is_constant": False, "missing_ratio": 0.29})

    assert assign_recommended_action(row) == "keep"
