import pandas as pd

from src.features.build_features import (
    create_train_test_split,
    fit_transform_median_imputer,
    select_model_features,
    split_features_and_label,
)


def test_select_model_features_excludes_drop_candidates() -> None:
    data = pd.DataFrame(
        {
            "sensor_000": [1.0],
            "sensor_001": [2.0],
            "sensor_002": [3.0],
            "label": [-1],
        }
    )
    quality_summary = pd.DataFrame(
        {
            "feature": ["sensor_000", "sensor_001", "sensor_002"],
            "recommended_action": ["keep", "drop_candidate", "review"],
        }
    )

    assert select_model_features(data, quality_summary) == ["sensor_000", "sensor_002"]


def test_select_model_features_includes_keep_and_review_features() -> None:
    data = pd.DataFrame(
        {
            "sensor_000": [1.0],
            "sensor_001": [2.0],
            "sensor_002": [3.0],
        }
    )
    quality_summary = pd.DataFrame(
        {
            "feature": ["sensor_000", "sensor_001", "sensor_002"],
            "recommended_action": ["keep", "review", "drop_candidate"],
        }
    )

    assert select_model_features(data, quality_summary) == ["sensor_000", "sensor_001"]


def test_metadata_columns_are_never_included_as_features() -> None:
    data = pd.DataFrame(
        {
            "sample_id": [0],
            "sensor_000": [1.0],
            "label": [-1],
            "timestamp": ["19/07/2008 11:55:00"],
        }
    )
    quality_summary = pd.DataFrame(
        {
            "feature": ["sample_id", "sensor_000", "label", "timestamp"],
            "recommended_action": ["keep", "keep", "keep", "keep"],
        }
    )

    assert select_model_features(data, quality_summary) == ["sensor_000"]


def test_train_test_split_preserves_row_counts() -> None:
    X = pd.DataFrame({"sensor_000": range(10)})
    y = pd.Series([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_test) == 2
    assert len(y_test) == 2


def test_imputer_is_fit_on_train_and_transforms_test_without_nan() -> None:
    X_train = pd.DataFrame(
        {
            "sensor_000": [1.0, 3.0, None],
            "sensor_001": [10.0, 20.0, 30.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "sensor_000": [None, 100.0],
            "sensor_001": [None, 40.0],
        }
    )

    X_train_imputed, X_test_imputed, medians = fit_transform_median_imputer(
        X_train,
        X_test,
    )

    assert medians["sensor_000"] == 2.0
    assert medians["sensor_001"] == 20.0
    assert not X_train_imputed.isna().any().any()
    assert not X_test_imputed.isna().any().any()
    assert X_test_imputed.loc[0, "sensor_000"] == 2.0
    assert X_test_imputed.loc[0, "sensor_001"] == 20.0


def test_split_features_and_label_returns_expected_columns() -> None:
    data = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "sensor_000": [1.0, 2.0],
            "label": [-1, 1],
            "timestamp": ["19/07/2008 11:55:00", "19/07/2008 12:32:00"],
        }
    )

    X, y = split_features_and_label(data, ["sensor_000"])

    assert list(X.columns) == ["sensor_000"]
    assert y.tolist() == [-1, 1]
