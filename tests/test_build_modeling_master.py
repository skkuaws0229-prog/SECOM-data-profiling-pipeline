import pandas as pd

from src.features.build_modeling_master import (
    LINEAGE_COLUMNS,
    build_modeling_master_table,
    normalize_labels,
    select_master_features,
)


def _sample_secom_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3, 4, 5],
            "timestamp": [
                "19/07/2008 11:55:00",
                "19/07/2008 12:32:00",
                "19/07/2008 13:17:00",
                "19/07/2008 14:43:00",
                "19/07/2008 15:22:00",
                "19/07/2008 16:01:00",
            ],
            "sensor_000": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "sensor_001": [10.0, 11.0, None, 13.0, 14.0, 15.0],
            "sensor_002": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "label": [-1, -1, -1, 1, 1, 1],
        }
    )


def _quality_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["sensor_000", "sensor_001", "sensor_002"],
            "recommended_action": ["keep", "review", "drop_candidate"],
        }
    )


def test_normalize_labels_converts_original_secom_labels() -> None:
    labels = pd.Series([-1, 1, -1, 1])

    normalized = normalize_labels(labels)

    assert normalized.tolist() == [0, 1, 0, 1]


def test_select_master_features_includes_keep_review_and_excludes_drop() -> None:
    selected = select_master_features(_sample_secom_df(), _quality_summary())

    assert selected == ["sensor_000", "sensor_001"]


def test_master_table_required_lineage_columns_are_first() -> None:
    master_table, _ = build_modeling_master_table(
        _sample_secom_df(),
        _quality_summary(),
        test_size=0.33,
        random_state=42,
    )

    assert master_table.columns[: len(LINEAGE_COLUMNS)].tolist() == LINEAGE_COLUMNS
    for column in [
        "sample_id",
        "timestamp",
        "label_original",
        "label_binary",
        "split",
        "feature_set_version",
        "quality_rule_version",
        "preprocessing_version",
        "imputation_strategy",
    ]:
        assert column in master_table.columns


def test_split_values_only_contain_train_and_test() -> None:
    master_table, _ = build_modeling_master_table(
        _sample_secom_df(),
        _quality_summary(),
        test_size=0.33,
        random_state=42,
    )

    assert set(master_table["split"].unique()) == {"train", "test"}


def test_train_test_row_counts_sum_to_total_rows() -> None:
    master_table, metadata = build_modeling_master_table(
        _sample_secom_df(),
        _quality_summary(),
        test_size=0.33,
        random_state=42,
    )

    assert metadata["train_row_count"] + metadata["test_row_count"] == len(master_table)
    assert metadata["split"]["train_row_count"] + metadata["split"]["test_row_count"] == len(
        master_table
    )


def test_median_imputation_removes_nan_when_train_median_exists() -> None:
    master_table, _ = build_modeling_master_table(
        _sample_secom_df(),
        _quality_summary(),
        test_size=0.33,
        random_state=42,
    )

    assert not master_table[["sensor_000", "sensor_001"]].isna().any().any()


def test_metadata_contains_required_summary_fields() -> None:
    master_table, metadata = build_modeling_master_table(
        _sample_secom_df(),
        _quality_summary(),
        test_size=0.33,
        random_state=42,
    )

    assert metadata["row_count"] == len(master_table)
    assert metadata["selected_feature_count"] == 2
    assert "train_row_count" in metadata
    assert "test_row_count" in metadata
    assert "train_label_distribution" in metadata
    assert "test_label_distribution" in metadata
    assert metadata["train_label_distribution"] == metadata["split"]["train_label_distribution"]
    assert metadata["test_label_distribution"] == metadata["split"]["test_label_distribution"]
