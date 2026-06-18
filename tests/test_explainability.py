import numpy as np
import pandas as pd

from src.modeling.explainability import (
    FEATURE_IMPORTANCE_COLUMNS,
    build_error_feature_profile,
    build_explainability_report,
    build_logistic_coefficient_importance,
    build_random_forest_importance,
    build_top_feature_summary,
    get_sensor_feature_columns,
)


class DummyLogisticClassifier:
    coef_ = np.array([[0.2, -1.5, 0.7]])


class DummyLogisticPipeline:
    named_steps = {"classifier": DummyLogisticClassifier()}


class DummyRandomForest:
    feature_importances_ = np.array([0.1, 0.8, 0.3])


def test_sensor_feature_selection_excludes_metadata_columns() -> None:
    df = pd.DataFrame(
        {
            "sample_id": [1],
            "timestamp": ["t"],
            "label_original": [-1],
            "label_binary": [0],
            "split": ["test"],
            "feature_set_version": ["v1"],
            "quality_rule_version": ["v1"],
            "preprocessing_version": ["v1"],
            "imputation_strategy": ["median_train_only"],
            "sensor_000": [1.0],
            "sensor_001": [2.0],
        }
    )

    assert get_sensor_feature_columns(df) == ["sensor_000", "sensor_001"]


def test_logistic_coefficient_importance_has_required_columns() -> None:
    importance = build_logistic_coefficient_importance(
        DummyLogisticPipeline(),
        ["sensor_000", "sensor_001", "sensor_002"],
    )

    assert importance.columns.tolist() == FEATURE_IMPORTANCE_COLUMNS
    assert importance.loc[0, "feature"] == "sensor_001"
    assert importance.loc[0, "importance_type"] == "coefficient_abs"


def test_random_forest_importance_has_required_columns() -> None:
    importance = build_random_forest_importance(
        DummyRandomForest(),
        ["sensor_000", "sensor_001", "sensor_002"],
    )

    assert importance.columns.tolist() == FEATURE_IMPORTANCE_COLUMNS
    assert importance.loc[0, "feature"] == "sensor_001"
    assert importance.loc[0, "importance_type"] == "impurity_importance"


def test_feature_importance_ranks_start_at_one() -> None:
    importance = build_random_forest_importance(
        DummyRandomForest(),
        ["sensor_000", "sensor_001", "sensor_002"],
    )

    assert importance["rank"].min() == 1
    assert importance["rank"].tolist() == [1, 2, 3]


def test_top_feature_summary_returns_top_n_per_group() -> None:
    feature_importance = pd.DataFrame(
        {
            "model_name": ["a", "a", "a", "b", "b", "b"],
            "importance_type": ["x", "x", "x", "x", "x", "x"],
            "feature": [
                "sensor_000",
                "sensor_001",
                "sensor_002",
                "sensor_000",
                "sensor_001",
                "sensor_002",
            ],
            "importance": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "rank": [1, 2, 3, 1, 2, 3],
        }
    )

    summary = build_top_feature_summary(feature_importance, top_n=2)

    assert len(summary) == 4
    assert summary.groupby(["model_name", "importance_type"]).size().max() == 2


def test_error_feature_profile_contains_required_group_names() -> None:
    test_df = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "sensor_000": [1.0, 2.0, 3.0],
            "sensor_001": [4.0, 5.0, 6.0],
        }
    )
    top_risk = pd.DataFrame({"sample_id": [2], "rank": [1]})
    false_negative = pd.DataFrame({"sample_id": [3]})

    profile = build_error_feature_profile(
        test_df=test_df,
        top_risk_samples=top_risk,
        false_negative_samples=false_negative,
        selected_features=["sensor_000", "sensor_001"],
    )

    assert set(profile["group_name"]) == {
        "top_risk_samples",
        "false_negative_samples",
        "all_test_samples",
    }


def test_report_generation_returns_markdown_string() -> None:
    feature_importance = pd.DataFrame(
        {
            "model_name": ["a"],
            "importance_type": ["x"],
            "feature": ["sensor_000"],
            "importance": [1.0],
            "rank": [1],
        }
    )
    error_profile = pd.DataFrame(
        {
            "group_name": ["all_test_samples"],
            "feature": ["sensor_000"],
            "mean": [1.0],
            "median": [1.0],
            "std": [0.0],
            "missing_count": [0],
            "sample_count": [3],
        }
    )

    report = build_explainability_report(
        feature_importance,
        feature_importance,
        error_profile,
    )

    assert isinstance(report, str)
    assert report.startswith("# SECOM Explainability")
