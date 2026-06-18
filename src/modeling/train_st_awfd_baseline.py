"""Train retrospective ST-AWFD baseline models from MaterialID master tables."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = [
    "dataset_id",
    "material_id",
    "label_original",
    "label_binary",
    "official_split",
]
EXCLUDED_FEATURE_COLUMNS = {
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
    "observed_step_ids_json",
}
METRIC_COLUMNS = [
    "dataset_id",
    "model_name",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f2",
    "average_precision",
    "roc_auc",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "warnings",
]
CONFUSION_COLUMNS = [
    "dataset_id",
    "model_name",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
]
DEFAULT_THRESHOLD = 0.50
RANDOM_STATE = 42


def load_material_master(path: Path) -> pd.DataFrame:
    """Load a ST-AWFD MaterialID-level master table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ST-AWFD material master not found: {path}")
    return pd.read_csv(path)


def validate_material_master(df: pd.DataFrame) -> None:
    """Validate the minimum contract required for retrospective baseline modeling."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Material master missing required columns: {missing}")

    if df["material_id"].duplicated().any():
        raise ValueError("material_id must be unique in the MaterialID master table")

    labels = set(df["label_binary"].dropna().astype(int).unique())
    if df["label_binary"].isna().any() or not labels.issubset({0, 1}):
        raise ValueError("label_binary must contain only 0/1 values")

    split_values = set(df["official_split"].dropna().astype(str).unique())
    if df["official_split"].isna().any() or not split_values.issubset({"train", "test"}):
        raise ValueError("official_split must contain only train/test values")

    train_ids = set(df.loc[df["official_split"] == "train", "material_id"])
    test_ids = set(df.loc[df["official_split"] == "test", "material_id"])
    if not train_ids:
        raise ValueError("official_split must contain at least one train sample")
    if not test_ids:
        raise ValueError("official_split must contain at least one test sample")
    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise ValueError(f"Train/test MaterialID overlap detected: {sorted(overlap)[:5]}")


def select_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric model features while excluding metadata, label, path, and JSON columns."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in EXCLUDED_FEATURE_COLUMNS]


def split_official_train_test(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Split MaterialID master rows by the official train/test split."""
    if not feature_columns:
        raise ValueError("No numeric model features were selected")

    train_df = df.loc[df["official_split"] == "train"].copy()
    test_df = df.loc[df["official_split"] == "test"].copy()
    X_train = train_df[feature_columns].reset_index(drop=True)
    X_test = test_df[feature_columns].reset_index(drop=True)
    y_train = train_df["label_binary"].astype(int).reset_index(drop=True)
    y_test = test_df["label_binary"].astype(int).reset_index(drop=True)
    return X_train, X_test, y_train, y_test, test_df.reset_index(drop=True)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Train Logistic Regression with train-fitted median imputation and scaling."""
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
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
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Train Random Forest with train-fitted median imputation and no scaling."""
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def _positive_class_scores(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X_test))
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model probability output does not include positive defect class 1")
    return probabilities[:, classes.index(1)]


def _safe_probability_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> tuple[float | None, float | None, list[str]]:
    warnings: list[str] = []
    if y_true.nunique(dropna=False) < 2:
        warnings.append(
            "official test split contains one class only; average_precision and roc_auc are null"
        )
        return None, None, warnings
    return (
        float(average_precision_score(y_true, y_score, pos_label=1)),
        float(roc_auc_score(y_true, y_score)),
        warnings,
    )


def evaluate_model(
    dataset_id: str,
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    """Evaluate a fitted model on the official test split."""
    y_score = pd.Series(_positive_class_scores(model, X_test), index=y_test.index)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    average_precision, roc_auc, warnings = _safe_probability_metrics(y_test, y_score.to_numpy())

    metrics = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_test, y_pred, beta=2, pos_label=1, zero_division=0)),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "warnings": "; ".join(warnings),
    }
    return metrics, y_pred, y_score


def build_metrics_dataframe(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a stable metrics dataframe."""
    metrics_df = pd.DataFrame(metrics)
    missing = set(METRIC_COLUMNS).difference(metrics_df.columns)
    if missing:
        raise ValueError(f"Metrics missing required columns: {sorted(missing)}")
    return metrics_df[METRIC_COLUMNS]


def build_confusion_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build confusion matrix rows from metrics."""
    return metrics_df[CONFUSION_COLUMNS].copy()


def build_predictions_dataframe(
    dataset_id: str,
    model_name: str,
    test_df: pd.DataFrame,
    y_pred: pd.Series,
    y_score: pd.Series,
) -> pd.DataFrame:
    """Build local generated prediction outputs for the official test split."""
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "material_id": test_df["material_id"].to_numpy(),
            "label_original": test_df["label_original"].to_numpy(),
            "label_binary": test_df["label_binary"].to_numpy(),
            "official_split": test_df["official_split"].to_numpy(),
            "defect_probability": y_score.to_numpy(),
            "predicted_label": y_pred.to_numpy(),
            "threshold": DEFAULT_THRESHOLD,
        }
    )


def build_skipped_predictions_dataframe(
    dataset_id: str,
    model_name: str,
    test_df: pd.DataFrame,
    warning: str,
) -> pd.DataFrame:
    """Build prediction output rows when supervised training is not possible."""
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "material_id": test_df["material_id"].to_numpy(),
            "label_original": test_df["label_original"].to_numpy(),
            "label_binary": test_df["label_binary"].to_numpy(),
            "official_split": test_df["official_split"].to_numpy(),
            "defect_probability": np.nan,
            "predicted_label": np.nan,
            "threshold": DEFAULT_THRESHOLD,
            "warning": warning,
        }
    )


def build_skipped_metrics(
    dataset_id: str,
    model_name: str,
    warning: str,
) -> dict[str, Any]:
    """Build null metric row when a supervised model cannot be trained honestly."""
    return {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "threshold": DEFAULT_THRESHOLD,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "f2": None,
        "average_precision": None,
        "roc_auc": None,
        "true_positive": None,
        "false_positive": None,
        "true_negative": None,
        "false_negative": None,
        "warnings": warning,
    }


def build_feature_metadata(
    dataset_id: str,
    feature_columns: list[str],
    master_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build feature-selection and split metadata for the retrospective baseline."""
    train_df = master_df.loc[master_df["official_split"] == "train"]
    test_df = master_df.loc[master_df["official_split"] == "test"]
    return {
        "dataset_id": dataset_id,
        "row_count": int(len(master_df)),
        "selected_feature_count": int(len(feature_columns)),
        "selected_feature_columns": feature_columns,
        "train_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "train_label_distribution": {
            str(key): int(value)
            for key, value in train_df["label_binary"].value_counts().sort_index().items()
        },
        "test_label_distribution": {
            str(key): int(value)
            for key, value in test_df["label_binary"].value_counts().sort_index().items()
        },
        "imputation_strategy": "median_fit_on_official_train_only",
        "scaling_strategy": "standard_scaler_fit_on_official_train_only_for_logistic_regression",
        "random_forest_scaling": "not_applied",
        "default_threshold": DEFAULT_THRESHOLD,
        "train_class_count": int(train_df["label_binary"].nunique()),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
        else:
            display_df[column] = display_df[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    lines = [
        "| " + " | ".join(display_df.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(display_df.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in display_df.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def build_baseline_report(
    dataset_id: str,
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    """Render a dataset-specific ST-AWFD baseline report."""
    return "\n".join(
        [
            f"# ST-AWFD {dataset_id} Retrospective Baseline Model Report",
            "",
            "This milestone trains full-process retrospective baseline models from the MaterialID-level master table. It does not use raw event CSV files, merge D1/D2, merge with SECOM, or modify master tables.",
            "",
            "## Dataset",
            "",
            f"- Dataset ID: {dataset_id}",
            f"- Material rows: {metadata['row_count']}",
            f"- Selected numeric model features: {metadata['selected_feature_count']}",
            f"- Train rows: {metadata['train_row_count']}",
            f"- Test rows: {metadata['test_row_count']}",
            f"- Train label distribution: {metadata['train_label_distribution']}",
            f"- Test label distribution: {metadata['test_label_distribution']}",
            "",
            "## Preprocessing",
            "",
            "- Median imputation is fit only on official train rows and then applied to train/test features.",
            "- Logistic Regression uses StandardScaler fit only on official train rows.",
            "- Random Forest does not use scaling.",
            "",
            "## Metrics at Default Threshold 0.50",
            "",
            _markdown_table(metrics_df),
            "",
            "## Confusion Matrix",
            "",
            _markdown_table(confusion_df),
            "",
            "## Limitations",
            "",
            "- This is a full-process retrospective baseline, not an operational early-warning model.",
            "- If the official train split contains only one label class, supervised Logistic Regression and Random Forest training is skipped and metrics are left null with a warning.",
            "- Default threshold 0.50 is not an operationally approved threshold.",
            "- Class imbalance can make accuracy misleading.",
            "- False negatives require separate threshold and cost analysis in the next milestone.",
            "- D1/D2 results must not be directly treated as the same production environment.",
            "",
        ]
    )


def build_comparison_report(dataset_metrics: dict[str, pd.DataFrame]) -> str:
    """Render a D1/D2 baseline comparison report without merging datasets."""
    comparison = pd.concat(dataset_metrics.values(), ignore_index=True)
    selected = comparison[
        [
            "dataset_id",
            "model_name",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f2",
            "average_precision",
            "roc_auc",
            "false_negative",
            "false_positive",
        ]
    ]
    return "\n".join(
        [
            "# ST-AWFD Retrospective Baseline Comparison",
            "",
            "D1 and D2 were trained and evaluated as separate pipelines. This report compares outputs side by side, but it does not merge datasets or treat them as the same production environment.",
            "",
            _markdown_table(selected),
            "",
            "Accuracy can be misleading when defects are imbalanced. False negatives require threshold and cost analysis before any operational decision.",
            "",
        ]
    )


def train_and_evaluate_dataset(
    dataset_id: str,
    master_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    """Run the complete baseline training/evaluation workflow for one ST-AWFD dataset."""
    master_df = load_material_master(master_path)
    validate_material_master(master_df)
    feature_columns = select_numeric_feature_columns(master_df)
    X_train, X_test, y_train, y_test, test_df = split_official_train_test(
        master_df,
        feature_columns,
    )

    models = {
        "logistic_regression": train_logistic_regression,
        "random_forest": train_random_forest,
    }

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    model_dir.mkdir(parents=True, exist_ok=True)
    one_class_train = y_train.nunique() < 2
    warning = (
        "official train split contains one class only; supervised baseline training skipped"
        if one_class_train
        else ""
    )
    for model_name, train_model in models.items():
        model_path = model_dir / f"{model_name}.joblib"
        if one_class_train:
            joblib.dump(
                {
                    "model_name": model_name,
                    "training_completed": False,
                    "warning": warning,
                    "train_label_distribution": {
                        str(key): int(value)
                        for key, value in y_train.value_counts().sort_index().items()
                    },
                },
                model_path,
            )
            metrics.append(build_skipped_metrics(dataset_id, model_name, warning))
            predictions.append(
                build_skipped_predictions_dataframe(dataset_id, model_name, test_df, warning)
            )
            continue

        model = train_model(X_train, y_train)
        joblib.dump(model, model_path)
        model_metrics, y_pred, y_score = evaluate_model(
            dataset_id,
            model_name,
            model,
            X_test,
            y_test,
        )
        metrics.append(model_metrics)
        predictions.append(
            build_predictions_dataframe(dataset_id, model_name, test_df, y_pred, y_score)
        )

    metrics_df = build_metrics_dataframe(metrics)
    confusion_df = build_confusion_dataframe(metrics_df)
    predictions_df = pd.concat(predictions, ignore_index=True)
    metadata = build_feature_metadata(dataset_id, feature_columns, master_df)
    report = build_baseline_report(dataset_id, metrics_df, confusion_df, metadata)

    return {
        "metrics_df": metrics_df,
        "confusion_df": confusion_df,
        "predictions_df": predictions_df,
        "metadata": metadata,
        "report": report,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
