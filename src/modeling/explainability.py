"""Explainability and feature profile analysis for SECOM baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
FEATURE_IMPORTANCE_COLUMNS = [
    "model_name",
    "importance_type",
    "feature",
    "importance",
    "rank",
]
ERROR_PROFILE_COLUMNS = [
    "group_name",
    "feature",
    "mean",
    "median",
    "std",
    "missing_count",
    "sample_count",
]


def load_modeling_master_table(path: Path) -> pd.DataFrame:
    """Load the canonical modeling master table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modeling master table not found: {path}")
    return pd.read_csv(path)


def get_sensor_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return anonymized sensor features and exclude lineage columns."""
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
        raise ValueError("Modeling master table must contain train and test rows")

    return (
        train_df[feature_columns],
        test_df[feature_columns],
        train_df["label_binary"].astype(int),
        test_df["label_binary"].astype(int),
        test_df,
    )


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train Logistic Regression with train-fitted StandardScaler."""
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
    """Train Random Forest without scaling."""
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _rank_importance(
    model_name: str,
    importance_type: str,
    features: list[str],
    values: np.ndarray,
) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "model_name": model_name,
            "importance_type": importance_type,
            "feature": features,
            "importance": values.astype(float),
        }
    ).sort_values(
        ["importance", "feature"],
        ascending=[False, True],
    )
    importance_df["rank"] = range(1, len(importance_df) + 1)
    return importance_df[FEATURE_IMPORTANCE_COLUMNS].reset_index(drop=True)


def build_logistic_coefficient_importance(
    model: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build absolute standardized coefficient feature importance."""
    classifier = model.named_steps["classifier"]
    coefficients = np.abs(classifier.coef_[0])
    return _rank_importance(
        model_name="logistic_regression",
        importance_type="coefficient_abs",
        features=feature_columns,
        values=coefficients,
    )


def build_random_forest_importance(
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build impurity-based Random Forest feature importance."""
    return _rank_importance(
        model_name="random_forest",
        importance_type="impurity_importance",
        features=feature_columns,
        values=model.feature_importances_,
    )


def build_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build test-split permutation importance using average precision."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return _rank_importance(
        model_name=model_name,
        importance_type="permutation_average_precision",
        features=list(X_test.columns),
        values=result.importances_mean,
    )


def build_feature_importance_table(
    logistic_model: Pipeline,
    random_forest_model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    include_permutation: bool = True,
) -> pd.DataFrame:
    """Combine coefficient, tree, and optional permutation importances."""
    feature_columns = list(X_test.columns)
    tables = [
        build_logistic_coefficient_importance(logistic_model, feature_columns),
        build_random_forest_importance(random_forest_model, feature_columns),
    ]
    if include_permutation:
        tables.extend(
            [
                build_permutation_importance(
                    logistic_model,
                    X_test,
                    y_test,
                    "logistic_regression",
                ),
                build_permutation_importance(
                    random_forest_model,
                    X_test,
                    y_test,
                    "random_forest",
                ),
            ]
        )
    return pd.concat(tables, ignore_index=True)


def build_top_feature_summary(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top N features per model and importance type."""
    return (
        feature_importance.sort_values(
            ["model_name", "importance_type", "rank"],
            ascending=[True, True, True],
        )
        .groupby(["model_name", "importance_type"], as_index=False)
        .head(top_n)[FEATURE_IMPORTANCE_COLUMNS]
        .reset_index(drop=True)
    )


def load_m5_error_outputs(reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load M5 top-risk and false-negative outputs or fail clearly."""
    reports_dir = Path(reports_dir)
    top_risk_path = reports_dir / "secom_top_risk_samples.csv"
    false_negative_path = reports_dir / "secom_false_negative_analysis.csv"
    missing = [
        str(path)
        for path in [top_risk_path, false_negative_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "M6 explainability requires M5 outputs. Run "
            "`python scripts/run_threshold_analysis_secom.py` first. Missing: "
            f"{missing}"
        )
    return pd.read_csv(top_risk_path), pd.read_csv(false_negative_path)


def _profile_group(
    group_name: str,
    group_df: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    records = []
    for feature in selected_features:
        series = group_df[feature]
        records.append(
            {
                "group_name": group_name,
                "feature": feature,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "missing_count": int(series.isna().sum()),
                "sample_count": int(len(series)),
            }
        )
    return pd.DataFrame(records, columns=ERROR_PROFILE_COLUMNS)


def build_error_feature_profile(
    test_df: pd.DataFrame,
    top_risk_samples: pd.DataFrame,
    false_negative_samples: pd.DataFrame,
    selected_features: list[str],
    top_risk_rank_limit: int = 30,
) -> pd.DataFrame:
    """Compare selected feature profiles for top-risk, false-negative, and all test rows."""
    top_risk_ids = (
        top_risk_samples.loc[
            top_risk_samples["rank"] <= top_risk_rank_limit,
            "sample_id",
        ]
        .drop_duplicates()
        .tolist()
    )
    false_negative_ids = false_negative_samples["sample_id"].drop_duplicates().tolist()

    groups = {
        "top_risk_samples": test_df.loc[test_df["sample_id"].isin(top_risk_ids)],
        "false_negative_samples": test_df.loc[
            test_df["sample_id"].isin(false_negative_ids)
        ],
        "all_test_samples": test_df,
    }
    return pd.concat(
        [
            _profile_group(group_name, group_df, selected_features)
            for group_name, group_df in groups.items()
        ],
        ignore_index=True,
    )


def consistently_important_features(
    top_feature_summary: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Count features appearing in multiple top-feature lists."""
    top_features = top_feature_summary.loc[top_feature_summary["rank"] <= top_n]
    counts = (
        top_features.groupby("feature")
        .size()
        .rename("top_list_count")
        .reset_index()
        .sort_values(["top_list_count", "feature"], ascending=[False, True])
    )
    return counts.loc[counts["top_list_count"] > 1].reset_index(drop=True)


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


def build_explainability_report(
    feature_importance: pd.DataFrame,
    top_feature_summary: pd.DataFrame,
    error_feature_profile: pd.DataFrame,
) -> str:
    """Build a Markdown explainability report."""
    consistent = consistently_important_features(top_feature_summary).head(20)
    top_preview = top_feature_summary.loc[top_feature_summary["rank"] <= 5]
    profile_counts = (
        error_feature_profile[["group_name", "sample_count"]]
        .drop_duplicates()
        .sort_values("group_name")
    )

    lines = [
        "# SECOM Explainability / Feature Importance Report",
        "",
        "M6 analyzes feature importance for baseline SECOM defect prediction models and compares feature profiles for top-risk and false-negative test samples.",
        "",
        "Positive class is `label_binary == 1` defect. Models are fit on `split == train` and explained/evaluated on `split == test` from `data/processed/modeling_master_table.csv`.",
        "",
        "## Why Explainability Matters",
        "",
        "Manufacturing defect prediction needs more than a score. Feature importance helps identify which anonymized sensor columns influence baseline model predictions and gives reviewers a structured starting point for error analysis.",
        "",
        "SECOM sensor features are anonymized. Names such as `sensor_000` must not be interpreted as physical process meanings.",
        "",
        "## Importance Methods",
        "",
        "- Logistic Regression uses absolute standardized coefficient magnitude.",
        "- Random Forest uses impurity-based `feature_importances_`.",
        "- Permutation importance uses test-split average precision and measures score impact after shuffling a feature.",
        "",
        "Coefficient-based importance reflects linear standardized model weights. Tree-based importance reflects split usage and impurity reduction. These methods can rank different features because they measure different model behaviors.",
        "",
        "## Consistently Important Features",
        "",
        _dataframe_to_markdown_table(consistent)
        if not consistent.empty
        else "No feature appeared in more than one top-feature list.",
        "",
        "## Top Feature Preview",
        "",
        _dataframe_to_markdown_table(top_preview),
        "",
        "## Error Feature Profile Groups",
        "",
        _dataframe_to_markdown_table(profile_counts),
        "",
        "Top-risk and false-negative feature profiles support error analysis by comparing high-risk predictions and missed defects against the full test split.",
        "",
        "## Limitations",
        "",
        "- Feature importance is not causal proof.",
        "- Anonymized sensors cannot be interpreted as physical process variables.",
        "- The small defect count limits confidence in feature rankings and error-profile comparisons.",
        "",
        f"Feature importance rows generated: {len(feature_importance)}",
        f"Feature profile rows generated: {len(error_feature_profile)}",
        "",
    ]
    return "\n".join(lines)
