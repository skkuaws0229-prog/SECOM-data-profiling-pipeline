"""Build the canonical SECOM modeling master table with lineage columns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_SET_VERSION = "m3_5_feature_set_v1"
QUALITY_RULE_VERSION = "m2_quality_rules_v1"
PREPROCESSING_VERSION = "m3_5_preprocessing_v1"
IMPUTATION_STRATEGY = "median_train_only"
INCLUDED_ACTIONS = {"keep", "review"}
LINEAGE_COLUMNS = [
    "sample_id",
    "timestamp",
    "label_original",
    "label_binary",
    "split",
    "feature_set_version",
    "quality_rule_version",
    "preprocessing_version",
    "imputation_strategy",
]


def load_feature_quality_summary(path: Path) -> pd.DataFrame:
    """Load the Milestone 2 feature quality summary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature quality summary not found: {path}")

    summary = pd.read_csv(path)
    required_columns = {"feature", "recommended_action"}
    missing_columns = required_columns.difference(summary.columns)
    if missing_columns:
        raise ValueError(
            "Feature quality summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    return summary


def select_master_features(
    df: pd.DataFrame,
    quality_summary: pd.DataFrame,
) -> list[str]:
    """Select sensor features marked keep/review by Milestone 2 decisions."""
    included_features = set(
        quality_summary.loc[
            quality_summary["recommended_action"].isin(INCLUDED_ACTIONS),
            "feature",
        ]
    )
    return [
        column
        for column in df.columns
        if column.startswith("sensor_") and column in included_features
    ]


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Convert original SECOM labels from -1/1 to binary 0/1 labels."""
    invalid_labels = set(labels.dropna().unique()).difference({-1, 1})
    if invalid_labels:
        raise ValueError(f"Unexpected SECOM label values: {sorted(invalid_labels)}")
    return labels.map({-1: 0, 1: 1}).astype(int)


def assign_stratified_split(
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.Series:
    """Assign train/test splits while preserving original row index."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    labels_reset = labels.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    split = pd.Series("train", index=labels_reset.index, name="split")

    for _, label_positions in labels_reset.groupby(labels_reset).groups.items():
        positions = np.array(list(label_positions), dtype=int)
        rng.shuffle(positions)

        if len(positions) == 1:
            class_test_count = 0
        else:
            class_test_count = int(round(len(positions) * test_size))
            class_test_count = max(1, min(class_test_count, len(positions) - 1))

        split.iloc[positions[:class_test_count]] = "test"

    return split


def fit_train_medians(
    df: pd.DataFrame,
    feature_columns: list[str],
    split: pd.Series,
) -> pd.Series:
    """Fit median imputation values on train rows only."""
    train_features = df.loc[split == "train", feature_columns]
    medians = train_features.median(axis=0, skipna=True)
    missing_medians = medians[medians.isna()].index.tolist()
    if missing_medians:
        raise ValueError(
            "Cannot median-impute columns with no non-null train values: "
            f"{missing_medians}"
        )
    return medians


def build_modeling_master_table(
    df: pd.DataFrame,
    quality_summary: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the canonical modeling master table and metadata."""
    feature_columns = select_master_features(df, quality_summary)
    label_binary = normalize_labels(df["label"])
    split = assign_stratified_split(
        label_binary,
        test_size=test_size,
        random_state=random_state,
    )
    train_medians = fit_train_medians(df, feature_columns, split)
    imputed_features = df[feature_columns].fillna(train_medians)

    lineage = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "timestamp": df["timestamp"],
            "label_original": df["label"],
            "label_binary": label_binary,
            "split": split,
            "feature_set_version": FEATURE_SET_VERSION,
            "quality_rule_version": QUALITY_RULE_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "imputation_strategy": IMPUTATION_STRATEGY,
        }
    )
    master_table = pd.concat([lineage[LINEAGE_COLUMNS], imputed_features], axis=1)

    action_counts = quality_summary["recommended_action"].value_counts().to_dict()
    train_label_distribution = _label_distribution(label_binary.loc[split == "train"])
    test_label_distribution = _label_distribution(label_binary.loc[split == "test"])
    train_row_count = int((split == "train").sum())
    test_row_count = int((split == "test").sum())

    metadata = {
        "row_count": int(len(master_table)),
        "original_feature_count": int(len(quality_summary)),
        "selected_feature_count": int(len(feature_columns)),
        "drop_candidate_feature_count": int(action_counts.get("drop_candidate", 0)),
        "keep_feature_count": int(action_counts.get("keep", 0)),
        "review_feature_count": int(action_counts.get("review", 0)),
        "train_row_count": train_row_count,
        "test_row_count": test_row_count,
        "train_label_distribution": train_label_distribution,
        "test_label_distribution": test_label_distribution,
        "feature_columns": feature_columns,
        "lineage_columns": LINEAGE_COLUMNS,
        "feature_set_version": FEATURE_SET_VERSION,
        "quality_rule_version": QUALITY_RULE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "imputation_strategy": IMPUTATION_STRATEGY,
        "imputation_fit_on": "train rows only",
        "split": {
            "method": "stratified_train_test_split",
            "stratify_by": "label_binary",
            "test_size": float(test_size),
            "random_state": int(random_state),
            "train_row_count": train_row_count,
            "test_row_count": test_row_count,
            "train_label_distribution": train_label_distribution,
            "test_label_distribution": test_label_distribution,
        },
        "label_mapping": {
            "-1": "0 normal",
            "1": "1 defect",
        },
        "train_median_imputation_values": {
            feature: float(value) for feature, value in train_medians.items()
        },
    }
    return master_table, metadata


def _label_distribution(labels: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in labels.value_counts(dropna=False).sort_index().items()
    }


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


def _distribution_table(distribution: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"label_binary": label, "count": count} for label, count in distribution.items()]
    )


def build_modeling_master_report(metadata: dict[str, Any]) -> str:
    """Build a Markdown report for the modeling master table."""
    split = metadata["split"]
    lines = [
        "# SECOM Modeling Master Table Report",
        "",
        "M3.5 builds a canonical modeling master table with lineage columns and selected imputed anonymized sensor features.",
        "",
        "This stage does not train models and does not assign process meanings to SECOM features.",
        "",
        "## Summary",
        "",
        f"- Row count: {metadata['row_count']}",
        f"- Original feature count: {metadata['original_feature_count']}",
        f"- Drop candidate feature count excluded: {metadata['drop_candidate_feature_count']}",
        f"- Keep feature count included: {metadata['keep_feature_count']}",
        f"- Review feature count included: {metadata['review_feature_count']}",
        f"- Selected feature count: {metadata['selected_feature_count']}",
        "",
        "## Lineage",
        "",
        f"- Feature set version: {metadata['feature_set_version']}",
        f"- Quality rule version: {metadata['quality_rule_version']}",
        f"- Preprocessing version: {metadata['preprocessing_version']}",
        f"- Imputation strategy: {metadata['imputation_strategy']}",
        "- Median imputation values are fit on train rows only and applied to train/test rows.",
        "",
        "## Split",
        "",
        f"- Method: {split['method']}",
        f"- Test size: {split['test_size']}",
        f"- Random state: {split['random_state']}",
        f"- Train row count: {split['train_row_count']}",
        f"- Test row count: {split['test_row_count']}",
        "",
        "## Train Label Distribution",
        "",
        _dataframe_to_markdown_table(
            _distribution_table(split["train_label_distribution"])
        ),
        "",
        "## Test Label Distribution",
        "",
        _dataframe_to_markdown_table(
            _distribution_table(split["test_label_distribution"])
        ),
        "",
        "## Label Mapping",
        "",
        "- Original `-1` -> binary `0` normal",
        "- Original `1` -> binary `1` defect",
        "",
    ]
    return "\n".join(lines)
