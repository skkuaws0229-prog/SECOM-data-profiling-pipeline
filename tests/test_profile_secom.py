import pandas as pd

from src.quality.profile_secom import build_markdown_report, profile_secom_data


def test_profile_secom_data_calculates_expected_metrics() -> None:
    data = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "sensor_000": [1.0, None, 1.0],
            "sensor_001": [2.0, 2.0, 2.0],
            "sensor_002": [None, None, None],
            "label": [-1, 1, -1],
            "timestamp": [
                "19/07/2008 11:55:00",
                "19/07/2008 12:32:00",
                "19/07/2008 13:10:00",
            ],
        }
    )

    profile = profile_secom_data(data)

    assert profile["row_count"] == 3
    assert profile["feature_count"] == 3
    assert profile["total_missing_values"] == 4
    assert profile["overall_missing_value_ratio"] == 4 / 9
    assert profile["duplicate_row_count"] == 0

    missing_report = profile["missing_value_report"].set_index("feature")
    assert missing_report.loc["sensor_000", "missing_count"] == 1
    assert missing_report.loc["sensor_002", "missing_ratio"] == 1.0

    label_distribution = profile["label_distribution"].set_index("label")
    assert label_distribution.loc[-1, "count"] == 2
    assert label_distribution.loc[1, "count"] == 1


def test_profile_secom_duplicate_count_excludes_sample_id() -> None:
    data = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "sensor_000": [1.0, 1.0],
            "label": [-1, -1],
            "timestamp": ["19/07/2008 11:55:00", "19/07/2008 11:55:00"],
        }
    )

    profile = profile_secom_data(data)

    assert profile["duplicate_row_count"] == 1


def test_build_markdown_report_contains_core_sections() -> None:
    data = pd.DataFrame(
        {
            "sample_id": [0],
            "sensor_000": [None],
            "label": [-1],
            "timestamp": ["19/07/2008 11:55:00"],
        }
    )
    profile = profile_secom_data(data)

    report = build_markdown_report(profile)

    assert "# SECOM Data Profile Report" in report
    assert "Label Distribution" in report
    assert "Top Missing Features" in report

