from pathlib import Path

import pandas as pd
import pytest

from src.modeling.train_st_awfd_baseline import (
    build_baseline_report,
    build_confusion_dataframe,
    build_metrics_dataframe,
    evaluate_model,
    select_numeric_feature_columns,
    split_official_train_test,
    train_and_evaluate_dataset,
    train_logistic_regression,
    train_random_forest,
    validate_material_master,
)


def _master_fixture(dataset_id: str = "st_awfd_d1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id] * 6,
            "material_id": ["A", "B", "C", "D", "E", "F"],
            "label_original": [0, 1, 0, 1, 0, 1],
            "label_binary": [0, 1, 0, 1, 0, 1],
            "official_split": ["train", "train", "train", "train", "test", "test"],
            "source_record_count": [10, 12, 11, 9, 8, 7],
            "step_count": [2, 2, 2, 2, 2, 2],
            "feature_set_version": ["v1"] * 6,
            "preprocessing_version": ["v1"] * 6,
            "aggregation_strategy": ["retrospective"] * 6,
            "sequence_manifest_path": ["sequence_manifest.csv"] * 6,
            "observed_step_ids_json": ["[1, 2]"] * 6,
            "feature_a_mean": [1.0, 3.0, None, 5.0, 1000.0, None],
            "feature_b_std": [2.0, 4.0, 6.0, 8.0, 2000.0, 3000.0],
        }
    )


def test_metadata_columns_are_excluded_from_feature_selection() -> None:
    features = select_numeric_feature_columns(_master_fixture())

    assert features == ["feature_a_mean", "feature_b_std"]


def test_train_only_imputer_behavior() -> None:
    df = _master_fixture()
    features = select_numeric_feature_columns(df)
    X_train, _, y_train, _, _ = split_official_train_test(df, features)

    model = train_random_forest(X_train, y_train)

    assert model.named_steps["imputer"].statistics_.tolist() == [3.0, 5.0]


def test_train_only_scaler_behavior() -> None:
    df = _master_fixture()
    features = select_numeric_feature_columns(df)
    X_train, _, y_train, _, _ = split_official_train_test(df, features)

    model = train_logistic_regression(X_train, y_train)

    assert model.named_steps["imputer"].statistics_.tolist() == [3.0, 5.0]
    assert model.named_steps["scaler"].mean_.tolist() == [3.0, 5.0]


def test_no_train_test_material_id_overlap() -> None:
    df = _master_fixture()
    df.loc[5, "material_id"] = "A"

    with pytest.raises(ValueError, match="material_id must be unique"):
        validate_material_master(df)


def test_metrics_and_confusion_output() -> None:
    df = _master_fixture()
    features = select_numeric_feature_columns(df)
    X_train, X_test, y_train, y_test, _ = split_official_train_test(df, features)
    model = train_random_forest(X_train, y_train)

    metrics, _, _ = evaluate_model("st_awfd_d1", "random_forest", model, X_test, y_test)
    metrics_df = build_metrics_dataframe([metrics])
    confusion_df = build_confusion_dataframe(metrics_df)

    assert {"accuracy", "precision", "recall", "f1", "f2"}.issubset(metrics_df.columns)
    assert {
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    }.issubset(confusion_df.columns)


def test_d1_d2_remain_separate(tmp_path: Path) -> None:
    d1_path = tmp_path / "d1_master.csv"
    d2_path = tmp_path / "d2_master.csv"
    _master_fixture("st_awfd_d1").to_csv(d1_path, index=False)
    _master_fixture("st_awfd_d2").to_csv(d2_path, index=False)

    d1_result = train_and_evaluate_dataset("st_awfd_d1", d1_path, tmp_path / "models_d1")
    d2_result = train_and_evaluate_dataset("st_awfd_d2", d2_path, tmp_path / "models_d2")

    assert set(d1_result["metrics_df"]["dataset_id"]) == {"st_awfd_d1"}
    assert set(d2_result["metrics_df"]["dataset_id"]) == {"st_awfd_d2"}


def test_one_class_train_split_returns_null_metrics_with_warning(tmp_path: Path) -> None:
    master_path = tmp_path / "one_class_master.csv"
    df = _master_fixture()
    df.loc[df["official_split"] == "train", "label_binary"] = 0
    df.loc[df["official_split"] == "train", "label_original"] = 0
    df.to_csv(master_path, index=False)

    result = train_and_evaluate_dataset("st_awfd_d1", master_path, tmp_path / "models")

    assert result["metrics_df"]["accuracy"].isna().all()
    assert result["metrics_df"]["warnings"].str.contains("one class only").all()
    assert result["predictions_df"]["defect_probability"].isna().all()


def test_report_includes_retrospective_limitation() -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset_id": "st_awfd_d1",
                "model_name": "logistic_regression",
                "threshold": 0.5,
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "f2": 0.5,
                "average_precision": 0.5,
                "roc_auc": 0.5,
                "true_positive": 1,
                "false_positive": 1,
                "true_negative": 1,
                "false_negative": 1,
                "warnings": "",
            }
        ]
    )
    confusion_df = build_confusion_dataframe(metrics_df)
    metadata = {
        "row_count": 6,
        "selected_feature_count": 2,
        "train_row_count": 4,
        "test_row_count": 2,
        "train_label_distribution": {"0": 2, "1": 2},
        "test_label_distribution": {"0": 1, "1": 1},
    }

    report = build_baseline_report("st_awfd_d1", metrics_df, confusion_df, metadata)

    assert "full-process retrospective baseline" in report
    assert "Default threshold 0.50 is not an operationally approved threshold" in report
