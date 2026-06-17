"""Basic profiling utilities for the SECOM dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd


ProfileResult = dict[str, Any]


def _dataframe_to_markdown_table(data: pd.DataFrame) -> str:
    columns = [str(column) for column in data.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in data.itertuples(index=False, name=None)
    ]
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def get_sensor_columns(data: pd.DataFrame) -> list[str]:
    """Return anonymized SECOM sensor columns."""
    return [column for column in data.columns if column.startswith("sensor_")]


def profile_secom_data(data: pd.DataFrame) -> ProfileResult:
    """Calculate basic profile metrics for a loaded SECOM DataFrame."""
    sensor_columns = get_sensor_columns(data)
    sensor_data = data[sensor_columns]
    row_count = len(data)
    feature_count = len(sensor_columns)

    missing_counts = sensor_data.isna().sum()
    missing_ratios = missing_counts / row_count if row_count else missing_counts * 0
    missing_value_report = pd.DataFrame(
        {
            "feature": missing_counts.index,
            "missing_count": missing_counts.astype(int).values,
            "missing_ratio": missing_ratios.astype(float).values,
        }
    )

    label_distribution = (
        data["label"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )

    duplicate_subset = data.drop(columns=["sample_id"], errors="ignore")
    total_missing_values = int(missing_counts.sum())
    total_sensor_cells = row_count * feature_count
    overall_missing_value_ratio = (
        total_missing_values / total_sensor_cells if total_sensor_cells else 0.0
    )

    return {
        "row_count": int(row_count),
        "feature_count": int(feature_count),
        "label_distribution": label_distribution,
        "missing_value_report": missing_value_report,
        "duplicate_row_count": int(duplicate_subset.duplicated().sum()),
        "total_missing_values": total_missing_values,
        "overall_missing_value_ratio": float(overall_missing_value_ratio),
    }


def build_markdown_report(profile: ProfileResult) -> str:
    """Create a concise Markdown report from SECOM profile results."""
    label_distribution = profile["label_distribution"]
    missing_value_report = profile["missing_value_report"]
    top_missing = missing_value_report.sort_values(
        ["missing_ratio", "missing_count", "feature"],
        ascending=[False, False, True],
    ).head(20)

    lines = [
        "# SECOM Data Profile Report",
        "",
        "Milestone 1 profiling report for the UCI SECOM dataset.",
        "",
        "Feature columns are anonymized sensor features. No process meanings are inferred.",
        "",
        "## Summary",
        "",
        f"- Row count: {profile['row_count']}",
        f"- Feature count: {profile['feature_count']}",
        f"- Duplicate row count: {profile['duplicate_row_count']}",
        f"- Total missing values: {profile['total_missing_values']}",
        "- Overall missing value ratio: "
        f"{profile['overall_missing_value_ratio']:.6f}",
        "",
        "## Label Distribution",
        "",
        _dataframe_to_markdown_table(label_distribution),
        "",
        "## Top Missing Features",
        "",
        _dataframe_to_markdown_table(top_missing),
        "",
    ]
    return "\n".join(lines)
