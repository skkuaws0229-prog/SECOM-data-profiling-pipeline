import pandas as pd

from src.modeling.threshold_analysis import (
    THRESHOLDS,
    THRESHOLD_METRIC_COLUMNS,
    build_top_risk_samples,
    calculate_threshold_metrics,
    evaluate_thresholds,
    extract_false_negatives,
    fbeta_score_from_counts,
    get_sensor_feature_columns,
)


def test_sensor_feature_selection_excludes_metadata_columns() -> None:
    df = pd.DataFrame(
        {
            "sample_id": [0],
            "timestamp": ["19/07/2008 11:55:00"],
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


def test_threshold_metric_output_has_required_columns() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.8, 0.6, 0.4])

    metrics_df = evaluate_thresholds(y_true, y_score, "dummy", thresholds=[0.5])

    assert metrics_df.columns.tolist() == THRESHOLD_METRIC_COLUMNS


def test_f2_score_calculation_works() -> None:
    f2 = fbeta_score_from_counts(tp=2, fp=1, fn=3, beta=2.0)

    assert round(f2, 6) == round(10 / 23, 6)


def test_false_negative_extraction_works() -> None:
    test_df = pd.DataFrame(
        {
            "sample_id": [10, 11, 12],
            "timestamp": ["a", "b", "c"],
            "label_original": [-1, 1, 1],
            "label_binary": [0, 1, 1],
        }
    )
    y_score = pd.Series([0.2, 0.4, 0.9])

    false_negatives = extract_false_negatives(
        test_df,
        y_score,
        model_name="dummy",
        threshold=0.5,
        threshold_type="default_0.5",
    )

    assert len(false_negatives) == 1
    assert false_negatives.loc[0, "sample_id"] == 11
    assert false_negatives.loc[0, "error_type"] == "false_negative"


def test_top_risk_samples_are_sorted_by_probability_descending() -> None:
    test_df = pd.DataFrame(
        {
            "sample_id": [10, 11, 12],
            "timestamp": ["a", "b", "c"],
            "label_original": [-1, 1, 1],
            "label_binary": [0, 1, 1],
        }
    )
    top_risk = build_top_risk_samples(
        test_df,
        {"dummy": pd.Series([0.2, 0.9, 0.7])},
    )

    assert top_risk["defect_probability"].tolist() == [0.9, 0.7, 0.2]
    assert top_risk["rank"].tolist() == [1, 2, 3]


def test_thresholds_are_between_zero_and_one() -> None:
    assert min(THRESHOLDS) == 0.05
    assert max(THRESHOLDS) == 0.95
    assert all(0 < threshold < 1 for threshold in THRESHOLDS)


def test_threshold_metrics_count_false_negatives_for_defect_positive_class() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.8, 0.6, 0.4])

    metrics = calculate_threshold_metrics(y_true, y_score, 0.5, "dummy")

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
