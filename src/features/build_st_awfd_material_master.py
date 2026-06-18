"""Build retrospective MaterialID-level ST-AWFD master tables."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["MaterialID", "StepID", "duration_ms", "target", "is_test"]
FEATURE_SET_VERSION = "st_awfd_material_retrospective_v1"
PREPROCESSING_VERSION = "m11_material_aggregation_v1"
AGGREGATION_STRATEGY = "retrospective_material_level_summary"
MASTER_METADATA_COLUMNS = [
    "dataset_id",
    "material_id",
    "label_original",
    "label_binary",
    "official_split",
    "source_record_count",
    "step_count",
    "feature_set_version",
    "preprocessing_version",
    "aggregation_strategy",
    "sequence_manifest_path",
]


def require_st_awfd_columns(df: pd.DataFrame) -> None:
    """Validate required raw ST-AWFD columns."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required ST-AWFD columns: {missing}")


def validate_binary_column(df: pd.DataFrame, column: str) -> None:
    """Validate a target/split column is non-null and binary 0/1."""
    if df[column].isna().any():
        raise ValueError(f"{column} contains missing values")
    values = set(pd.to_numeric(df[column], errors="coerce").dropna().astype(int).unique())
    raw_count = df[column].nunique(dropna=False)
    if raw_count != len(values) or not values.issubset({0, 1}):
        raise ValueError(f"{column} must contain only binary values 0/1")


def validate_material_consistency(df: pd.DataFrame, column: str) -> None:
    """Validate a column is constant within each MaterialID."""
    mixed = df.groupby("MaterialID")[column].nunique(dropna=False)
    mixed_ids = mixed.loc[mixed > 1].index.tolist()
    if mixed_ids:
        raise ValueError(
            f"{column} varies within MaterialID for {len(mixed_ids)} materials"
        )


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric candidate feature columns, excluding required reference columns."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    return [column for column in numeric_columns if column not in REQUIRED_COLUMNS]


def _json_list(values: pd.Series) -> str:
    cleaned = sorted(values.dropna().unique().tolist())
    return json.dumps(cleaned)


def _official_split(value: int) -> str:
    return "test" if int(value) == 1 else "train"


def load_and_validate_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate a raw ST-AWFD CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ST-AWFD raw CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    require_st_awfd_columns(df)
    validate_binary_column(df, "target")
    validate_binary_column(df, "is_test")
    validate_material_consistency(df, "target")
    validate_material_consistency(df, "is_test")
    return df


def build_sequence_manifest(
    df: pd.DataFrame,
    dataset_id: str,
    raw_csv_path: Path,
) -> pd.DataFrame:
    """Build one-row-per-MaterialID sequence manifest."""
    grouped = df.groupby("MaterialID", sort=True)
    manifest = pd.DataFrame(
        {
            "material_id": grouped.size().index,
            "label_original": grouped["target"].first().astype(int).values,
            "label_binary": grouped["target"].first().astype(int).values,
            "official_split": grouped["is_test"].first().astype(int).map(_official_split).values,
            "source_record_count": grouped.size().astype(int).values,
            "step_count": grouped["StepID"].nunique().astype(int).values,
            "observed_step_ids_json": grouped["StepID"].agg(_json_list).values,
            "duration_ms_min": grouped["duration_ms"].min().values,
            "duration_ms_max": grouped["duration_ms"].max().values,
            "raw_event_file_path": str(raw_csv_path),
            "sequence_available": True,
        }
    )
    manifest.insert(0, "dataset_id", dataset_id)
    return manifest


def _flatten_columns(df: pd.DataFrame, suffix_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        f"{feature}_{stat}"
        for feature, stat in df.columns.to_flat_index()
    ]
    ordered_columns = []
    for feature in sorted({column.rsplit("_", 1)[0] for column in df.columns}):
        for suffix in suffix_order:
            column = f"{feature}_{suffix}"
            if column in df.columns:
                ordered_columns.append(column)
    return df[ordered_columns]


def build_material_master(
    df: pd.DataFrame,
    dataset_id: str,
    raw_csv_path: Path,
    sequence_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build MaterialID-level master table, sequence manifest, and metadata."""
    require_st_awfd_columns(df)
    feature_columns = get_numeric_feature_columns(df)
    grouped = df.groupby("MaterialID", sort=True)

    base = pd.DataFrame(
        {
            "material_id": grouped.size().index,
            "label_original": grouped["target"].first().astype(int).values,
            "label_binary": grouped["target"].first().astype(int).values,
            "official_split": grouped["is_test"].first().astype(int).map(_official_split).values,
            "source_record_count": grouped.size().astype(int).values,
            "step_count": grouped["StepID"].nunique().astype(int).values,
        }
    )
    base.insert(0, "dataset_id", dataset_id)
    base["feature_set_version"] = FEATURE_SET_VERSION
    base["preprocessing_version"] = PREPROCESSING_VERSION
    base["aggregation_strategy"] = AGGREGATION_STRATEGY
    base["sequence_manifest_path"] = str(sequence_manifest_path)

    duration_stats = grouped["duration_ms"].agg(["min", "max", "mean", "std"])
    duration_stats.columns = [f"duration_ms_{column}" for column in duration_stats.columns]
    duration_missing = grouped["duration_ms"].apply(lambda series: float(series.isna().mean()))
    duration_missing.name = "duration_ms_missing_ratio"

    if feature_columns:
        feature_stats = grouped[feature_columns].agg(["mean", "std", "min", "max"])
        feature_stats = _flatten_columns(feature_stats, ["mean", "std", "min", "max"])
        feature_missing = grouped[feature_columns].apply(lambda data: data.isna().mean())
        feature_missing.columns = [f"{column}_missing_ratio" for column in feature_missing.columns]

        step_means = []
        for feature in feature_columns:
            pivot = df.pivot_table(
                index="MaterialID",
                columns="StepID",
                values=feature,
                aggfunc="mean",
            )
            pivot.columns = [f"{feature}_step_{step_id}_mean" for step_id in pivot.columns]
            step_means.append(pivot)
        step_mean_df = pd.concat(step_means, axis=1)
    else:
        feature_stats = pd.DataFrame(index=duration_stats.index)
        feature_missing = pd.DataFrame(index=duration_stats.index)
        step_mean_df = pd.DataFrame(index=duration_stats.index)

    features = pd.concat(
        [duration_stats, duration_missing, feature_stats, feature_missing, step_mean_df],
        axis=1,
    ).reset_index(drop=True)
    master = pd.concat([base[MASTER_METADATA_COLUMNS], features], axis=1)

    sequence_manifest = build_sequence_manifest(df, dataset_id, raw_csv_path)
    metadata = {
        "dataset_id": dataset_id,
        "raw_csv_path": str(raw_csv_path),
        "material_master_row_count": int(len(master)),
        "source_record_count": int(len(df)),
        "numeric_feature_count": int(len(feature_columns)),
        "numeric_feature_columns": feature_columns,
        "observed_step_ids": sorted(df["StepID"].dropna().unique().tolist()),
        "official_split_counts": {
            str(key): int(value)
            for key, value in master["official_split"].value_counts().sort_index().items()
        },
        "label_binary_distribution": {
            str(key): int(value)
            for key, value in master["label_binary"].value_counts().sort_index().items()
        },
        "feature_set_version": FEATURE_SET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "aggregation_strategy": AGGREGATION_STRATEGY,
        "sequence_manifest_path": str(sequence_manifest_path),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "notes": [
            "One row equals one MaterialID.",
            "D1 and D2 remain separate datasets.",
            "No model training or imputation was performed.",
            "Raw event-level files are preserved for future sequence-aware modeling.",
        ],
    }
    return master, sequence_manifest, metadata


def _markdown_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "_No rows._"
    columns = list(records[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(str(record.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_material_master_report(metadata: dict[str, Any]) -> str:
    """Build a dataset-specific MaterialID master report."""
    lines = [
        f"# ST-AWFD {metadata['dataset_id']} Material Master Report",
        "",
        "M11 creates a retrospective MaterialID-level master table. It does not train models, impute values, modify raw CSV files, or merge ST-AWFD with SECOM.",
        "",
        "## Summary",
        "",
        f"- Raw CSV path: {metadata['raw_csv_path']}",
        f"- Source record count: {metadata['source_record_count']}",
        f"- Material master row count: {metadata['material_master_row_count']}",
        f"- Numeric feature count: {metadata['numeric_feature_count']}",
        f"- Observed StepIDs: {metadata['observed_step_ids']}",
        f"- Official split counts: {metadata['official_split_counts']}",
        f"- Label distribution: {metadata['label_binary_distribution']}",
        "",
        "## Lineage",
        "",
        f"- Feature set version: {metadata['feature_set_version']}",
        f"- Preprocessing version: {metadata['preprocessing_version']}",
        f"- Aggregation strategy: {metadata['aggregation_strategy']}",
        f"- Sequence manifest path: {metadata['sequence_manifest_path']}",
        "",
        "## Sequence Preservation",
        "",
        "The sequence manifest keeps one row per MaterialID with observed StepIDs, duration range, raw event file path, and a sequence availability flag. Raw event-level files remain the source for future sequence-aware and early-warning modeling.",
        "",
    ]
    return "\n".join(lines)


def build_comparison_report(metadata_by_dataset: list[dict[str, Any]]) -> str:
    """Build D1/D2 comparison report."""
    records = [
        {
            "dataset_id": metadata["dataset_id"],
            "materials": metadata["material_master_row_count"],
            "source_records": metadata["source_record_count"],
            "numeric_features": metadata["numeric_feature_count"],
            "observed_step_count": len(metadata["observed_step_ids"]),
            "split_counts": metadata["official_split_counts"],
            "label_distribution": metadata["label_binary_distribution"],
        }
        for metadata in metadata_by_dataset
    ]
    lines = [
        "# ST-AWFD Material Master Comparison",
        "",
        "D1 and D2 remain separate datasets. This comparison report summarizes the independently built retrospective MaterialID-level master tables.",
        "",
        _markdown_table(records),
        "",
        "No model training, imputation, or SECOM merge was performed.",
        "",
    ]
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write metadata JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
