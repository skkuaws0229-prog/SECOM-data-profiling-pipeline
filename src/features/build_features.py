"""Feature engineering utilities for the SECOM dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METADATA_COLUMNS = {"sample_id", "label", "timestamp"}
INCLUDED_ACTIONS = {"keep", "review"}
IMPUTATION_STRATEGY = "median"


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


def select_model_features(
    df: pd.DataFrame,
    quality_summary: pd.DataFrame,
) -> list[str]:
    """Select sensor features marked keep/review and exclude metadata columns."""
    included_features = set(
        quality_summary.loc[
            quality_summary["recommended_action"].isin(INCLUDED_ACTIONS),
            "feature",
        ]
    )

    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
        and column.startswith("sensor_")
        and column in included_features
    ]


def split_features_and_label(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded SECOM DataFrame into feature matrix and label vector."""
    if "label" not in df.columns:
        raise ValueError("Input DataFrame must contain a label column")

    invalid_features = [column for column in feature_columns if column in METADATA_COLUMNS]
    if invalid_features:
        raise ValueError(f"Metadata columns cannot be used as features: {invalid_features}")

    return df[feature_columns].copy(), df["label"].copy()


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a deterministic stratified train/test split using label values."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    y_reset = y.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    train_positions: list[int] = []
    test_positions: list[int] = []

    for _, label_positions in y_reset.groupby(y_reset).groups.items():
        positions = np.array(list(label_positions), dtype=int)
        rng.shuffle(positions)

        if len(positions) == 1:
            class_test_count = 0
        else:
            class_test_count = int(round(len(positions) * test_size))
            class_test_count = max(1, min(class_test_count, len(positions) - 1))

        test_positions.extend(positions[:class_test_count].tolist())
        train_positions.extend(positions[class_test_count:].tolist())

    train_positions = sorted(train_positions)
    test_positions = sorted(test_positions)

    X_reset = X.reset_index(drop=True)
    return (
        X_reset.iloc[train_positions].reset_index(drop=True),
        X_reset.iloc[test_positions].reset_index(drop=True),
        y_reset.iloc[train_positions].reset_index(drop=True),
        y_reset.iloc[test_positions].reset_index(drop=True),
    )


def fit_transform_median_imputer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fit median imputation on X_train only, then transform train and test."""
    medians = X_train.median(axis=0, skipna=True)
    columns_without_train_median = medians[medians.isna()].index.tolist()
    if columns_without_train_median:
        raise ValueError(
            "Cannot median-impute columns with no non-null values in X_train: "
            f"{columns_without_train_median}"
        )

    X_train_imputed = X_train.fillna(medians)
    X_test_imputed = X_test.fillna(medians)
    return X_train_imputed, X_test_imputed, medians


def _label_distribution(y: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in y.value_counts(dropna=False).sort_index().items()
    }


def build_feature_set_metadata(
    *,
    original_feature_count: int,
    drop_candidate_feature_count: int,
    kept_feature_count: int,
    review_feature_count_included: int,
    feature_columns: list[str],
    train_row_count: int,
    test_row_count: int,
    train_label_distribution: dict[str, int],
    test_label_distribution: dict[str, int],
    imputation_values: pd.Series,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """Build reproducibility metadata for the processed feature set."""
    return {
        "original_feature_count": int(original_feature_count),
        "drop_candidate_feature_count": int(drop_candidate_feature_count),
        "kept_feature_count": int(kept_feature_count),
        "review_feature_count_included": int(review_feature_count_included),
        "final_feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "split": {
            "method": "stratified_train_test_split",
            "stratify_by": "label",
            "test_size": float(test_size),
            "random_state": int(random_state),
            "train_row_count": int(train_row_count),
            "test_row_count": int(test_row_count),
            "train_label_distribution": train_label_distribution,
            "test_label_distribution": test_label_distribution,
        },
        "imputation": {
            "strategy": IMPUTATION_STRATEGY,
            "fit_on": "X_train",
            "transformed": ["X_train", "X_test"],
            "median_values": {
                feature: float(value) for feature, value in imputation_values.items()
            },
        },
        "leakage_prevention_note": (
            "Median imputation values are fit on X_train only and then applied to "
            "X_train and X_test."
        ),
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
        [{"label": label, "count": count} for label, count in distribution.items()]
    )


def build_feature_engineering_report(metadata: dict[str, Any]) -> str:
    """Build the Milestone 3 Markdown feature engineering report."""
    split = metadata["split"]
    imputation = metadata["imputation"]

    lines = [
        "# SECOM Feature Engineering Report",
        "",
        "Milestone 3 reproducible feature engineering pipeline for anonymized SECOM sensor features.",
        "",
        "This stage does not train a model and does not assign process meanings to SECOM features.",
        "",
        "## Feature Selection Summary",
        "",
        f"- Original feature count: {metadata['original_feature_count']}",
        f"- Drop candidate feature count: {metadata['drop_candidate_feature_count']}",
        f"- Kept feature count: {metadata['kept_feature_count']}",
        f"- Review feature count included: {metadata['review_feature_count_included']}",
        f"- Final feature count: {metadata['final_feature_count']}",
        "",
        "## Split Summary",
        "",
        f"- Train row count: {split['train_row_count']}",
        f"- Test row count: {split['test_row_count']}",
        f"- Test size: {split['test_size']}",
        f"- Random state: {split['random_state']}",
        f"- Stratify by: {split['stratify_by']}",
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
        "## Imputation",
        "",
        f"- Strategy: {imputation['strategy']}",
        "- Fit on: X_train only",
        "- Transform: X_train and X_test",
        "",
        "## Leakage Prevention Note",
        "",
        metadata["leakage_prevention_note"],
        "",
    ]
    return "\n".join(lines)


def summarize_feature_quality_counts(
    quality_summary: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, int]:
    """Summarize Milestone 2 quality decisions for selected features."""
    actions = quality_summary["recommended_action"]
    return {
        "original_feature_count": int(len(quality_summary)),
        "drop_candidate_feature_count": int((actions == "drop_candidate").sum()),
        "kept_feature_count": int((actions == "keep").sum()),
        "review_feature_count_included": int(
            quality_summary[
                (quality_summary["recommended_action"] == "review")
                & (quality_summary["feature"].isin(feature_columns))
            ].shape[0]
        ),
    }


def label_distribution(y: pd.Series) -> dict[str, int]:
    """Public wrapper for label distribution metadata."""
    return _label_distribution(y)
