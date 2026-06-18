"""Baseline defect prediction models from the SECOM modeling master table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {
    "sample_id",
    "timestamp",
    "label_original",
    "label_binary",
    "split",
    "feature_set_version",
    "quality_rule_version",
    "preprocessing_version",
    "imputation_strategy",
}
REQUIRED_METRIC_COLUMNS = [
    "model_name",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
    "false_positive",
    "false_negative",
    "true_positive",
    "true_negative",
]


def get_sensor_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return anonymized sensor feature columns from the modeling master table."""
    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and column.startswith("sensor_")
    ]


def load_modeling_master_table(path: Path) -> pd.DataFrame:
    """Load the canonical M3.5 modeling master table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modeling master table not found: {path}")
    return pd.read_csv(path)


def load_processed_datasets(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train/test data from the canonical modeling master table."""
    processed_dir = Path(processed_dir)
    master_table = load_modeling_master_table(processed_dir / "modeling_master_table.csv")
    return split_train_test_from_master(master_table)


def split_train_test_from_master(
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split modeling master rows into train/test features and binary labels."""
    required_columns = {"label_binary", "split"}
    missing_columns = required_columns.difference(master_df.columns)
    if missing_columns:
        raise ValueError(f"Modeling master table missing columns: {sorted(missing_columns)}")

    feature_columns = get_sensor_feature_columns(master_df)
    if not feature_columns:
        raise ValueError("Modeling master table contains no sensor_* feature columns")

    train_df = master_df.loc[master_df["split"] == "train"].copy()
    test_df = master_df.loc[master_df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Modeling master table must contain train and test rows")

    X_train = train_df[feature_columns].reset_index(drop=True)
    X_test = test_df[feature_columns].reset_index(drop=True)
    y_train = train_df["label_binary"].astype(int).reset_index(drop=True)
    y_test = test_df["label_binary"].astype(int).reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def normalize_labels(y: pd.Series) -> pd.Series:
    """Convert SECOM labels from -1/1 to binary 0/1 defect labels."""
    invalid_labels = set(y.dropna().unique()).difference({-1, 0, 1})
    if invalid_labels:
        raise ValueError(f"Unexpected label values: {sorted(invalid_labels)}")

    return y.map({-1: 0, 0: 0, 1: 1}).astype(int)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train a class-balanced Logistic Regression baseline."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a class-balanced Random Forest baseline."""
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _positive_class_scores(model: Any, X_test: pd.DataFrame) -> pd.Series:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for PR-AUC and ROC-AUC")

    probabilities = np.asarray(model.predict_proba(X_test))
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model probability output does not contain defect class 1")

    positive_index = classes.index(1)
    return pd.Series(probabilities[:, positive_index], index=X_test.index)


def _validate_metric_inputs(y_test: pd.Series, y_pred: pd.Series) -> None:
    observed_labels = set(y_test.unique())
    if observed_labels != {0, 1}:
        raise ValueError(
            "Evaluation requires y_test to contain both binary classes 0 and 1; "
            f"observed {sorted(observed_labels)}"
        )


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float | int | str]:
    """Evaluate a classifier using defect label 1 as the positive class."""
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    y_score = _positive_class_scores(model, X_test)
    _validate_metric_inputs(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, y_score, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "true_negative": int(tn),
    }


def build_metrics_dataframe(metrics: list[dict]) -> pd.DataFrame:
    """Build a stable metrics table from model evaluation dictionaries."""
    metrics_df = pd.DataFrame(metrics)
    missing_columns = set(REQUIRED_METRIC_COLUMNS).difference(metrics_df.columns)
    if missing_columns:
        raise ValueError(f"Metrics are missing required columns: {sorted(missing_columns)}")
    return metrics_df[REQUIRED_METRIC_COLUMNS]


def build_confusion_matrix_dataframe(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Build a one-row confusion matrix table with defect class 1 as positive."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        ]
    )


def build_prediction_sample(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n: int = 30,
) -> pd.DataFrame:
    """Build a small prediction sample with true label, prediction, and score."""
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    y_score = _positive_class_scores(model, X_test)
    sample_size = min(n, len(y_test))
    return pd.DataFrame(
        {
            "model_name": model_name,
            "row_index": X_test.index[:sample_size],
            "y_true": y_test.iloc[:sample_size].to_numpy(),
            "y_pred": y_pred.iloc[:sample_size].to_numpy(),
            "y_score": y_score.iloc[:sample_size].to_numpy(),
        }
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


def build_baseline_model_report(
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    metadata: dict,
) -> str:
    """Build a Markdown report for Milestone 4 baseline modeling."""
    split = metadata.get("split", {})

    display_metrics = metrics_df.copy()
    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
    ]:
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.6f}")

    lines = [
        "# SECOM Baseline Defect Prediction Report",
        "",
        "Milestone 4 trains reproducible baseline classifiers from the canonical SECOM modeling master table.",
        "",
        "M4 uses `data/processed/modeling_master_table.csv` as the canonical dataset. It trains on rows where `split == train`, evaluates on rows where `split == test`, and uses `label_binary` as the target.",
        "",
        "The modeling master table preserves original SECOM label lineage: `label_original == -1` maps to `label_binary == 0` normal and `label_original == 1` maps to `label_binary == 1` defect.",
        "",
        "## Scope",
        "",
        "This milestone does not perform heavy hyperparameter tuning and does not add RAG, agents, APIs, dashboards, or Docker.",
        "",
        "## Canonical Dataset",
        "",
        f"- Modeling master row count: {metadata.get('row_count')}",
        f"- Selected sensor feature count: {metadata.get('selected_feature_count')}",
        f"- Train row count: {split.get('train_row_count')}",
        f"- Test row count: {split.get('test_row_count')}",
        f"- Imputation strategy: {metadata.get('imputation_strategy')}",
        f"- Feature set version: {metadata.get('feature_set_version')}",
        "",
        "## Metrics",
        "",
        _dataframe_to_markdown_table(display_metrics),
        "",
        "## Confusion Matrix",
        "",
        _dataframe_to_markdown_table(confusion_df),
        "",
        "## Evaluation Notes",
        "",
        "- Positive class is defect label `1`.",
        "- PR-AUC and ROC-AUC use predicted probabilities for defect label `1`.",
        "- Accuracy is reported, but manufacturing defect prediction also needs recall, precision, F1, and PR-AUC because defects are rare.",
        "- A false negative means a defective sample is predicted as normal, which can allow a quality issue to pass downstream inspection.",
        "",
    ]
    return "\n".join(lines)


def load_feature_set_metadata(processed_dir: Path) -> dict:
    """Load M3.5 modeling master metadata for baseline report context."""
    metadata_path = Path(processed_dir) / "modeling_master_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Modeling master metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))
