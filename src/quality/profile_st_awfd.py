"""Read-only profiling for local ST-AWFD wafer datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REFERENCE_COLUMNS = ["MaterialID", "StepID", "duration_ms", "target", "is_test"]
SUMMARY_KEYS = [
    "dataset_id",
    "file_path",
    "row_count",
    "column_count",
    "column_names",
    "dtypes",
    "required_reference_columns",
    "feature_columns",
    "unique_material_id_count",
    "target_distribution",
    "target_consistent_by_material_id",
    "is_test_consistent_by_material_id",
    "step_id_distribution",
    "material_id_row_count_distribution",
    "duration_ms",
    "missing_value_counts",
    "duplicate_full_row_count",
    "mixed_target_material_id_count",
    "mixed_is_test_material_id_count",
]


def require_columns(columns: list[str]) -> None:
    """Validate required ST-AWFD reference columns."""
    missing = [column for column in REFERENCE_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Missing required ST-AWFD columns: {missing}")


def get_feature_columns(columns: list[str]) -> list[str]:
    """Return candidate sequence feature columns."""
    return [column for column in columns if column not in REFERENCE_COLUMNS]


def _value_counts_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _series_counts_to_counter(series: pd.Series) -> Counter:
    return Counter(series.fillna("<MISSING>").astype(str).tolist())


def _percentile(counter: Counter, percentile: float) -> float:
    values = sorted(counter.values())
    if not values:
        return 0.0
    return float(pd.Series(values).quantile(percentile))


def profile_st_awfd_csv(
    csv_path: Path,
    dataset_id: str,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Profile one ST-AWFD CSV using chunked reads."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ST-AWFD raw CSV not found: {csv_path}")

    row_count = 0
    column_names: list[str] | None = None
    dtypes: dict[str, str] = {}
    missing_counts: Counter = Counter()
    target_distribution: Counter = Counter()
    step_id_distribution: Counter = Counter()
    material_row_counts: Counter = Counter()
    material_targets: dict[str, set[str]] = defaultdict(set)
    material_is_test: dict[str, set[str]] = defaultdict(set)
    row_hash_counts: Counter = Counter()
    duration_min: float | None = None
    duration_max: float | None = None
    duration_missing_count = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if column_names is None:
            column_names = chunk.columns.tolist()
            require_columns(column_names)
            dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}
        row_count += len(chunk)

        missing_counts.update(chunk.isna().sum().astype(int).to_dict())
        target_distribution.update(_series_counts_to_counter(chunk["target"]))
        step_id_distribution.update(_series_counts_to_counter(chunk["StepID"]))
        material_row_counts.update(chunk["MaterialID"].astype(str).tolist())

        grouped = chunk.groupby(chunk["MaterialID"].astype(str), dropna=False)
        for material_id, group in grouped:
            material_targets[str(material_id)].update(group["target"].fillna("<MISSING>").astype(str).unique())
            material_is_test[str(material_id)].update(group["is_test"].fillna("<MISSING>").astype(str).unique())

        duration = pd.to_numeric(chunk["duration_ms"], errors="coerce")
        chunk_min = duration.min(skipna=True)
        chunk_max = duration.max(skipna=True)
        if pd.notna(chunk_min):
            duration_min = float(chunk_min) if duration_min is None else min(duration_min, float(chunk_min))
        if pd.notna(chunk_max):
            duration_max = float(chunk_max) if duration_max is None else max(duration_max, float(chunk_max))
        duration_missing_count += int(duration.isna().sum())

        row_hashes = pd.util.hash_pandas_object(chunk, index=False)
        row_hash_counts.update(row_hashes.astype(str).tolist())

    if column_names is None:
        raise ValueError(f"ST-AWFD CSV is empty: {csv_path}")

    feature_columns = get_feature_columns(column_names)
    mixed_target_count = sum(1 for values in material_targets.values() if len(values) > 1)
    mixed_is_test_count = sum(1 for values in material_is_test.values() if len(values) > 1)
    duplicate_count = sum(count - 1 for count in row_hash_counts.values() if count > 1)

    material_counts = pd.Series(list(material_row_counts.values()), dtype="float64")
    material_distribution = {
        "min": float(material_counts.min()) if not material_counts.empty else 0.0,
        "max": float(material_counts.max()) if not material_counts.empty else 0.0,
        "mean": float(material_counts.mean()) if not material_counts.empty else 0.0,
        "median": float(material_counts.median()) if not material_counts.empty else 0.0,
        "p25": _percentile(material_row_counts, 0.25),
        "p75": _percentile(material_row_counts, 0.75),
    }

    return {
        "dataset_id": dataset_id,
        "file_path": str(csv_path),
        "row_count": int(row_count),
        "column_count": int(len(column_names)),
        "column_names": column_names,
        "dtypes": dtypes,
        "required_reference_columns": REFERENCE_COLUMNS,
        "feature_columns": feature_columns,
        "unique_material_id_count": int(len(material_row_counts)),
        "target_distribution": _value_counts_to_dict(target_distribution),
        "target_consistent_by_material_id": mixed_target_count == 0,
        "is_test_consistent_by_material_id": mixed_is_test_count == 0,
        "step_id_distribution": _value_counts_to_dict(step_id_distribution),
        "material_id_row_count_distribution": material_distribution,
        "duration_ms": {
            "min": duration_min,
            "max": duration_max,
            "missing_count": int(duration_missing_count),
        },
        "missing_value_counts": {
            column: int(missing_counts.get(column, 0)) for column in column_names
        },
        "duplicate_full_row_count": int(duplicate_count),
        "mixed_target_material_id_count": int(mixed_target_count),
        "mixed_is_test_material_id_count": int(mixed_is_test_count),
    }


def _markdown_table_from_dict(values: dict[str, Any], key_name: str, value_name: str, limit: int | None = None) -> str:
    items = list(values.items())
    if limit is not None:
        items = items[:limit]
    lines = [f"| {key_name} | {value_name} |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in items)
    return "\n".join(lines)


def build_profile_report(summary: dict[str, Any]) -> str:
    """Build a Markdown profile report for one ST-AWFD dataset."""
    material_dist = summary["material_id_row_count_distribution"]
    duration = summary["duration_ms"]
    top_missing = dict(
        sorted(
            summary["missing_value_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:20]
    )
    top_steps = dict(
        sorted(
            summary["step_id_distribution"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:20]
    )
    lines = [
        f"# ST-AWFD {summary['dataset_id']} Profile Report",
        "",
        "Read-only profile of the local raw ST-AWFD CSV. No raw files were modified, no modeling master table was created, and no models were trained.",
        "",
        "## File Overview",
        "",
        f"- File path: {summary['file_path']}",
        f"- Row count: {summary['row_count']}",
        f"- Column count: {summary['column_count']}",
        f"- Required reference columns: {summary['required_reference_columns']}",
        f"- Feature column count: {len(summary['feature_columns'])}",
        f"- Duplicate full-row count: {summary['duplicate_full_row_count']}",
        "",
        "## MaterialID and Label Consistency",
        "",
        f"- Unique MaterialID count: {summary['unique_material_id_count']}",
        f"- Target consistent within MaterialID: {summary['target_consistent_by_material_id']}",
        f"- is_test consistent within MaterialID: {summary['is_test_consistent_by_material_id']}",
        f"- MaterialIDs with mixed target: {summary['mixed_target_material_id_count']}",
        f"- MaterialIDs with mixed is_test: {summary['mixed_is_test_material_id_count']}",
        "",
        "## MaterialID Row-Count Distribution",
        "",
        f"- Min: {material_dist['min']}",
        f"- Max: {material_dist['max']}",
        f"- Mean: {material_dist['mean']}",
        f"- Median: {material_dist['median']}",
        f"- P25: {material_dist['p25']}",
        f"- P75: {material_dist['p75']}",
        "",
        "## duration_ms",
        "",
        f"- Min: {duration['min']}",
        f"- Max: {duration['max']}",
        f"- Missing count: {duration['missing_count']}",
        "",
        "## Target Distribution",
        "",
        _markdown_table_from_dict(summary["target_distribution"], "target", "row_count"),
        "",
        "## Top StepID Values",
        "",
        _markdown_table_from_dict(top_steps, "StepID", "row_count"),
        "",
        "## Top Missing Columns",
        "",
        _markdown_table_from_dict(top_missing, "column", "missing_count"),
        "",
        "## Candidate Sequence Features",
        "",
        ", ".join(summary["feature_columns"]),
        "",
    ]
    return "\n".join(lines)


def build_master_table_design_decision(summaries: list[dict[str, Any]]) -> str:
    """Build the ST-AWFD master table design decision report."""
    all_target_safe = all(summary["target_consistent_by_material_id"] for summary in summaries)
    all_split_safe = all(summary["is_test_consistent_by_material_id"] for summary in summaries)
    common_metadata = ["MaterialID", "target", "is_test"]
    sequence_metadata = ["StepID", "duration_ms"]
    feature_columns = sorted(
        {
            feature
            for summary in summaries
            for feature in summary["feature_columns"]
        }
    )
    lines = [
        "# ST-AWFD Master Table Design Decision",
        "",
        "This read-only design report uses D1/D2 raw profiling outputs only. It does not create a modeling master table and does not aggregate data for model input.",
        "",
        "## 1. Can target be safely assigned at MaterialID level?",
        "",
        f"Answer: {'Yes' if all_target_safe else 'No'} based on current raw profiling. Mixed-target MaterialID counts: "
        + ", ".join(f"{summary['dataset_id']}={summary['mixed_target_material_id_count']}" for summary in summaries)
        + ".",
        "",
        "## 2. Can is_test be safely assigned at MaterialID level?",
        "",
        f"Answer: {'Yes' if all_split_safe else 'No'} based on current raw profiling. Mixed-is_test MaterialID counts: "
        + ", ".join(f"{summary['dataset_id']}={summary['mixed_is_test_material_id_count']}" for summary in summaries)
        + ".",
        "",
        "## 3. Is MaterialID the recommended modeling entity?",
        "",
        "Answer: Yes. MaterialID is the recommended modeling entity because each material contains multiple StepID rows that should remain connected as one production/wafer sequence for downstream modeling.",
        "",
        "## 4. Should future M11 create a MaterialID-level master table?",
        "",
        "Answer: Yes. Future M11 should create a MaterialID-level master table, but it should preserve sequence-aware structures rather than flattening away StepID order without a documented design.",
        "",
        "## 5. Which raw columns are metadata versus candidate sequence features?",
        "",
        f"- Material-level metadata: {common_metadata}",
        f"- Sequence/time metadata: {sequence_metadata}",
        f"- Candidate sequence features: {feature_columns}",
        "",
        "## 6. What must be preserved for future sequence-aware modeling?",
        "",
        "Future sequence-aware modeling must preserve MaterialID grouping, StepID ordering/distribution, duration_ms timing, target and is_test lineage, original feature columns, and the distinction between D1 and D2.",
        "",
    ]
    return "\n".join(lines)


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    """Write a profile summary JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
