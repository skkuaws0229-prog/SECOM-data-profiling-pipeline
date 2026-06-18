import pandas as pd

from src.modeling.train_baseline import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    build_prediction_sample,
    evaluate_classifier,
    get_sensor_feature_columns,
    load_processed_datasets,
    normalize_labels,
    split_train_test_from_master,
)


class DummyClassifier:
    classes_ = pd.Index([0, 1])

    def predict(self, X: pd.DataFrame) -> list[int]:
        return [0, 1, 1, 0][: len(X)]

    def predict_proba(self, X: pd.DataFrame) -> list[list[float]]:
        return [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.60, 0.40],
        ][: len(X)]


def test_normalize_labels_converts_original_secom_labels_to_binary() -> None:
    labels = pd.Series([-1, 1, -1, 1])

    normalized = normalize_labels(labels)

    assert normalized.tolist() == [0, 1, 0, 1]


def test_sensor_feature_selection_excludes_master_metadata_columns() -> None:
    data = pd.DataFrame(
        {
            "sample_id": [0],
            "timestamp": ["19/07/2008 11:55:00"],
            "label_original": [-1],
            "label_binary": [0],
            "split": ["train"],
            "feature_set_version": ["v1"],
            "quality_rule_version": ["v1"],
            "preprocessing_version": ["v1"],
            "imputation_strategy": ["median_train_only"],
            "sensor_000": [1.0],
            "sensor_001": [2.0],
        }
    )

    assert get_sensor_feature_columns(data) == ["sensor_000", "sensor_001"]


def test_split_train_test_from_master_uses_split_and_label_binary() -> None:
    master = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "timestamp": ["a", "b", "c", "d"],
            "label_original": [-1, 1, -1, 1],
            "label_binary": [0, 1, 0, 1],
            "split": ["train", "train", "test", "test"],
            "feature_set_version": ["v1"] * 4,
            "quality_rule_version": ["v1"] * 4,
            "preprocessing_version": ["v1"] * 4,
            "imputation_strategy": ["median_train_only"] * 4,
            "sensor_000": [1.0, 2.0, 3.0, 4.0],
            "sensor_001": [5.0, 6.0, 7.0, 8.0],
        }
    )

    X_train, X_test, y_train, y_test = split_train_test_from_master(master)

    assert X_train.columns.tolist() == ["sensor_000", "sensor_001"]
    assert X_train.shape == (2, 2)
    assert X_test.shape == (2, 2)
    assert y_train.tolist() == [0, 1]
    assert y_test.tolist() == [0, 1]


def test_load_processed_datasets_reads_modeling_master_table(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    master = pd.DataFrame(
        {
            "label_binary": [0, 1],
            "split": ["train", "test"],
            "sensor_000": [1.0, 2.0],
        }
    )
    master.to_csv(processed_dir / "modeling_master_table.csv", index=False)

    X_train, X_test, y_train, y_test = load_processed_datasets(processed_dir)

    assert X_train["sensor_000"].tolist() == [1.0]
    assert X_test["sensor_000"].tolist() == [2.0]
    assert y_train.tolist() == [0]
    assert y_test.tolist() == [1]


def test_metrics_dataframe_contains_required_columns() -> None:
    metrics = [
        {
            "model_name": "dummy",
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "average_precision": 1.0,
            "roc_auc": 1.0,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 2,
            "true_negative": 2,
        }
    ]

    metrics_df = build_metrics_dataframe(metrics)

    assert set(
        [
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
    ).issubset(metrics_df.columns)


def test_confusion_matrix_fields_are_present() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])

    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred, "dummy")

    assert confusion_df.loc[0, "model_name"] == "dummy"
    assert confusion_df.loc[0, "true_negative"] == 2
    assert confusion_df.loc[0, "false_positive"] == 0
    assert confusion_df.loc[0, "false_negative"] == 1
    assert confusion_df.loc[0, "true_positive"] == 1


def test_prediction_sample_contains_true_predicted_and_score_columns() -> None:
    X_test = pd.DataFrame({"sensor_000": [1.0, 2.0, 3.0, 4.0]})
    y_test = pd.Series([0, 1, 1, 0])

    sample = build_prediction_sample(DummyClassifier(), X_test, y_test, "dummy", n=2)

    assert ["y_true", "y_pred", "y_score"] == [
        column for column in ["y_true", "y_pred", "y_score"] if column in sample.columns
    ]
    assert sample["y_score"].tolist() == [0.10, 0.80]


def test_evaluation_uses_defect_label_one_as_positive_class() -> None:
    X_test = pd.DataFrame({"sensor_000": [1.0, 2.0, 3.0, 4.0]})
    y_test = pd.Series([0, 1, 0, 1])

    metrics = evaluate_classifier(DummyClassifier(), X_test, y_test, "dummy")

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
