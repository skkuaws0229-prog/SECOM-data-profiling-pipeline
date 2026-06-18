"""Feature-level quality rules for anonymized SECOM sensor columns."""

from __future__ import annotations

from typing import Any

import pandas as pd


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return anonymized SECOM sensor columns only."""
    return [column for column in df.columns if column.startswith("sensor_")]


def _missing_bucket(missing_ratio: float) -> str:
    if missing_ratio >= 0.80:
        return "high_missing_drop_candidate"
    if missing_ratio >= 0.30:
        return "medium_missing_review"
    return "low_missing_keep"


def assign_recommended_action(row: pd.Series) -> str:
    """Assign the final feature action using the Milestone 2 rule priority."""
    if bool(row["is_constant"]):
        return "drop_candidate"
    if float(row["missing_ratio"]) >= 0.80:
        return "drop_candidate"
    if float(row["missing_ratio"]) >= 0.30:
        return "review"
    return "keep"


def build_feature_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate feature quality metrics and decisions for each sensor column."""
    sensor_columns = get_sensor_columns(df)
    row_count = len(df)
    records: list[dict[str, Any]] = []

    for feature in sensor_columns:
        series = df[feature]
        missing_count = int(series.isna().sum())
        missing_ratio = missing_count / row_count if row_count else 0.0
        non_null_count = int(series.notna().sum())
        unique_non_null_count = int(series.dropna().nunique())
        is_constant = unique_non_null_count <= 1

        records.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "missing_ratio": float(missing_ratio),
                "non_null_count": non_null_count,
                "unique_non_null_count": unique_non_null_count,
                "is_constant": bool(is_constant),
                "missing_bucket": _missing_bucket(missing_ratio),
            }
        )

    summary = pd.DataFrame.from_records(
        records,
        columns=[
            "feature",
            "missing_count",
            "missing_ratio",
            "non_null_count",
            "unique_non_null_count",
            "is_constant",
            "missing_bucket",
        ],
    )
    summary["recommended_action"] = summary.apply(assign_recommended_action, axis=1)
    return summary


def _action_counts(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df["recommended_action"]
        .value_counts()
        .reindex(["drop_candidate", "review", "keep"], fill_value=0)
        .rename_axis("recommended_action")
        .reset_index(name="feature_count")
    )


def _bucket_counts(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df["missing_bucket"]
        .value_counts()
        .reindex(
            [
                "high_missing_drop_candidate",
                "medium_missing_review",
                "low_missing_keep",
            ],
            fill_value=0,
        )
        .rename_axis("missing_bucket")
        .reset_index(name="feature_count")
    )


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
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


def build_quality_decision_report(summary_df: pd.DataFrame) -> str:
    """Build a Markdown decision report from a feature quality summary."""
    drop_candidates = summary_df[
        summary_df["recommended_action"] == "drop_candidate"
    ].sort_values(
        ["is_constant", "missing_ratio", "feature"],
        ascending=[False, False, True],
    )
    review_candidates = summary_df[
        summary_df["recommended_action"] == "review"
    ].sort_values(["missing_ratio", "feature"], ascending=[False, True])

    lines = [
        "# SECOM Feature Quality Decision Report",
        "",
        "Milestone 2 feature-level quality rules for anonymized SECOM sensor features.",
        "",
        "This report does not physically drop columns. It creates decisions for the next feature engineering milestone.",
        "",
        "## Summary",
        "",
        f"- Sensor feature count: {len(summary_df)}",
        f"- Drop candidates: {len(drop_candidates)}",
        f"- Review candidates: {len(review_candidates)}",
        f"- Keep candidates: {int((summary_df['recommended_action'] == 'keep').sum())}",
        "",
        "## Recommended Action Counts",
        "",
        _dataframe_to_markdown_table(_action_counts(summary_df)),
        "",
        "## Missing Bucket Counts",
        "",
        _dataframe_to_markdown_table(_bucket_counts(summary_df)),
        "",
        "## Top Drop Candidates",
        "",
        _dataframe_to_markdown_table(drop_candidates.head(20)),
        "",
        "## Top Review Candidates",
        "",
        _dataframe_to_markdown_table(review_candidates.head(20)),
        "",
    ]
    return "\n".join(lines)
