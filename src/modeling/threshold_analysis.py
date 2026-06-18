"""Threshold and error analysis for SECOM baseline defect models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

THRESHOLDS = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
THRESHOLD_METRIC_COLUMNS = [
    "model_name",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f2",
    "tp",
    "fp",
    "tn",
    "fn",
    "false_negative_rate",
    "predicted_positive_count",
]


def load_modeling_master_table(path: Path) -> pd.DataFrame:
    """Load the canonical M3.5 modeling master table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modeling master table not found: {path}")
    return pd.read_csv(path)


def get_sensor_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sensor feature columns and exclude lineage/metadata columns."""
    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and column.startswith("sensor_")
    ]


def split_train_test_from_master(
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Create train/test matrices and labels from the modeling master table."""
    feature_columns = get_sensor_feature_columns(master_df)
    train_df = master_df.loc[master_df["split"] == "train"].copy()
    test_df = master_df.loc[master_df["split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Modeling master table must contain both train and test rows")

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df["label_binary"].astype(int)
    y_test = test_df["label_binary"].astype(int)
    return X_train, X_test, y_train, y_test, test_df


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train class-balanced Logistic Regression with train-fitted scaling."""
    model = Pipeline(
        steps=[
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
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train class-balanced Random Forest without scaling."""
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_defect_probability(model: Any, X_test: pd.DataFrame) -> pd.Series:
    """Return predicted probabilities for positive defect class 1."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba")

    probabilities = np.asarray(model.predict_proba(X_test))
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model probabilities do not include defect class 1")
    positive_index = classes.index(1)
    return pd.Series(probabilities[:, positive_index], index=X_test.index)


def fbeta_score_from_counts(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    """Calculate F-beta directly from confusion counts."""
    beta_squared = beta**2
    denominator = ((1 + beta_squared) * tp) + (beta_squared * fn) + fp
    if denominator == 0:
        return 0.0
    return float(((1 + beta_squared) * tp) / denominator)


def calculate_threshold_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float,
    model_name: str,
) -> dict[str, float | int | str]:
    """Calculate threshold-dependent defect detection metrics."""
    if not 0 < threshold < 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    f2 = fbeta_score_from_counts(tp=tp, fp=fp, fn=fn, beta=2.0)

    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else 0.0,
        "predicted_positive_count": int(tp + fp),
    }


def evaluate_thresholds(
    y_true: pd.Series,
    y_score: pd.Series,
    model_name: str,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate one model across the threshold grid."""
    thresholds = thresholds or THRESHOLDS
    metrics = [
        calculate_threshold_metrics(y_true, y_score, threshold, model_name)
        for threshold in thresholds
    ]
    return pd.DataFrame(metrics, columns=THRESHOLD_METRIC_COLUMNS)


def build_top_risk_samples(
    test_df: pd.DataFrame,
    model_scores: dict[str, pd.Series],
) -> pd.DataFrame:
    """Build test-split risk rankings sorted by defect probability."""
    records: list[pd.DataFrame] = []
    base_columns = ["sample_id", "timestamp", "label_original", "label_binary"]
    for model_name, scores in model_scores.items():
        model_risk = test_df[base_columns].copy()
        model_risk["model_name"] = model_name
        model_risk["defect_probability"] = scores.to_numpy()
        model_risk = model_risk.sort_values(
            "defect_probability",
            ascending=False,
        ).reset_index(drop=True)
        model_risk["rank"] = model_risk.index + 1
        records.append(
            model_risk[
                [
                    "model_name",
                    "sample_id",
                    "timestamp",
                    "label_original",
                    "label_binary",
                    "defect_probability",
                    "rank",
                ]
            ]
        )
    return pd.concat(records, ignore_index=True)


def best_f2_thresholds(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the best F2 threshold per model, preferring lower thresholds on ties."""
    sorted_metrics = threshold_metrics.sort_values(
        ["model_name", "f2", "threshold"],
        ascending=[True, False, True],
    )
    return sorted_metrics.groupby("model_name", as_index=False).head(1)


def extract_false_negatives(
    test_df: pd.DataFrame,
    y_score: pd.Series,
    model_name: str,
    threshold: float,
    threshold_type: str,
) -> pd.DataFrame:
    """Extract false negative test rows for one model and threshold."""
    predicted_label = (y_score >= threshold).astype(int)
    rows = test_df.loc[
        (test_df["label_binary"] == 1) & (predicted_label == 0),
        ["sample_id", "timestamp", "label_original", "label_binary"],
    ].copy()
    rows["model_name"] = model_name
    rows["threshold_type"] = threshold_type
    rows["threshold"] = float(threshold)
    rows["defect_probability"] = y_score.loc[rows.index].to_numpy()
    rows["predicted_label"] = 0
    rows["error_type"] = "false_negative"
    return rows[
        [
            "model_name",
            "threshold_type",
            "threshold",
            "sample_id",
            "timestamp",
            "label_original",
            "label_binary",
            "defect_probability",
            "predicted_label",
            "error_type",
        ]
    ].reset_index(drop=True)


def build_false_negative_analysis(
    test_df: pd.DataFrame,
    model_scores: dict[str, pd.Series],
    threshold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build false negative rows for default 0.5 and best-F2 thresholds."""
    best_thresholds = best_f2_thresholds(threshold_metrics).set_index("model_name")
    records: list[pd.DataFrame] = []

    for model_name, scores in model_scores.items():
        records.append(
            extract_false_negatives(
                test_df=test_df,
                y_score=scores,
                model_name=model_name,
                threshold=0.5,
                threshold_type="default_0.5",
            )
        )
        best_threshold = float(best_thresholds.loc[model_name, "threshold"])
        records.append(
            extract_false_negatives(
                test_df=test_df,
                y_score=scores,
                model_name=model_name,
                threshold=best_threshold,
                threshold_type="best_f2",
            )
        )

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


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


def build_threshold_error_analysis_report(
    threshold_metrics: pd.DataFrame,
    false_negative_df: pd.DataFrame,
) -> str:
    """Build Markdown report for threshold and error analysis."""
    default_metrics = threshold_metrics.loc[threshold_metrics["threshold"] == 0.5].copy()
    best_metrics = best_f2_thresholds(threshold_metrics).copy()

    default_lookup = default_metrics.set_index("model_name")
    reduction_rows = []
    for row in best_metrics.itertuples(index=False):
        model_name = row.model_name
        default_fn = int(default_lookup.loc[model_name, "fn"])
        best_fn = int(row.fn)
        reduction_rows.append(
            {
                "model_name": model_name,
                "default_threshold": 0.5,
                "default_false_negative": default_fn,
                "best_f2_threshold": row.threshold,
                "best_f2_false_negative": best_fn,
                "false_negative_reduction": default_fn - best_fn,
            }
        )

    reduction_df = pd.DataFrame(reduction_rows)
    display_default = default_metrics[
        ["model_name", "threshold", "accuracy", "precision", "recall", "f1", "f2", "fn"]
    ]
    display_best = best_metrics[
        ["model_name", "threshold", "accuracy", "precision", "recall", "f1", "f2", "fn"]
    ]

    lines = [
        "# SECOM Threshold & Error Analysis Report",
        "",
        "M5 analyzes how classification thresholds change defect detection performance on the SECOM test split.",
        "",
        "Positive class is `label_binary == 1` defect. Models are fit on `split == train` and evaluated on `split == test` from `data/processed/modeling_master_table.csv`.",
        "",
        "## Why Threshold Analysis Matters",
        "",
        "Manufacturing defect detection is imbalanced, so the default threshold of 0.5 may not match operational risk. Lowering a threshold can increase recall and reduce missed defects, while usually increasing false positives.",
        "",
        "Accuracy can be misleading because normal samples dominate the dataset. A model can look accurate while still missing many defect samples.",
        "",
        "SECOM sensor names are anonymized and must not be interpreted as physical process meanings.",
        "",
        "## Default Threshold 0.5",
        "",
        _dataframe_to_markdown_table(display_default),
        "",
        "## Best F2 Threshold Per Model",
        "",
        _dataframe_to_markdown_table(display_best),
        "",
        "## False Negative Reduction vs. Threshold 0.5",
        "",
        _dataframe_to_markdown_table(reduction_df),
        "",
        "## False Negative Rows",
        "",
        f"- False negative analysis rows written: {len(false_negative_df)}",
        "- See `reports/secom_false_negative_analysis.csv` for row-level details.",
        "",
    ]
    return "\n".join(lines)
