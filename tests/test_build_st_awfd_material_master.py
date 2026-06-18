from pathlib import Path

import pandas as pd
import pytest

from src.features.build_st_awfd_material_master import (
    MASTER_METADATA_COLUMNS,
    build_material_master,
    get_numeric_feature_columns,
    load_and_validate_raw_csv,
    validate_binary_column,
    validate_material_consistency,
)


def _sample_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MaterialID": ["A", "A", "B", "B"],
            "StepID": [1, 2, 1, 2],
            "duration_ms": [10.0, 12.0, 20.0, None],
            "feature_1": [1.0, 3.0, 5.0, None],
            "feature_2": [2.0, 4.0, 6.0, 8.0],
            "target": [0, 0, 1, 1],
            "is_test": [0, 0, 1, 1],
        }
    )


def test_binary_validation_rejects_non_binary_values() -> None:
    df = _sample_raw()
    df.loc[0, "target"] = 2

    with pytest.raises(ValueError, match="binary"):
        validate_binary_column(df, "target")


def test_material_consistency_rejects_mixed_target() -> None:
    df = _sample_raw()
    df.loc[1, "target"] = 1

    with pytest.raises(ValueError, match="varies within MaterialID"):
        validate_material_consistency(df, "target")


def test_numeric_feature_columns_exclude_reference_columns() -> None:
    assert get_numeric_feature_columns(_sample_raw()) == ["feature_1", "feature_2"]


def test_build_material_master_has_required_metadata_columns_first(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequence_manifest.csv"

    master, _, _ = build_material_master(
        _sample_raw(),
        dataset_id="st_awfd_d1",
        raw_csv_path=tmp_path / "D1.csv",
        sequence_manifest_path=sequence_path,
    )

    assert master.columns[: len(MASTER_METADATA_COLUMNS)].tolist() == MASTER_METADATA_COLUMNS
    assert len(master) == 2
    assert master["official_split"].tolist() == ["train", "test"]


def test_duration_and_feature_aggregations_are_created(tmp_path: Path) -> None:
    master, _, _ = build_material_master(
        _sample_raw(),
        dataset_id="st_awfd_d1",
        raw_csv_path=tmp_path / "D1.csv",
        sequence_manifest_path=tmp_path / "sequence_manifest.csv",
    )

    for column in [
        "duration_ms_min",
        "duration_ms_max",
        "duration_ms_mean",
        "duration_ms_std",
        "duration_ms_missing_ratio",
        "feature_1_mean",
        "feature_1_std",
        "feature_1_min",
        "feature_1_max",
        "feature_1_missing_ratio",
        "feature_1_step_1_mean",
        "feature_1_step_2_mean",
    ]:
        assert column in master.columns


def test_sequence_manifest_contains_expected_fields(tmp_path: Path) -> None:
    _, sequence_manifest, _ = build_material_master(
        _sample_raw(),
        dataset_id="st_awfd_d1",
        raw_csv_path=tmp_path / "D1.csv",
        sequence_manifest_path=tmp_path / "sequence_manifest.csv",
    )

    for column in [
        "dataset_id",
        "material_id",
        "label_original",
        "label_binary",
        "official_split",
        "source_record_count",
        "step_count",
        "observed_step_ids_json",
        "duration_ms_min",
        "duration_ms_max",
        "raw_event_file_path",
        "sequence_available",
    ]:
        assert column in sequence_manifest.columns


def test_load_and_validate_raw_csv_uses_fixture_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw.csv"
    _sample_raw().to_csv(csv_path, index=False)

    loaded = load_and_validate_raw_csv(csv_path)

    assert loaded.shape == (4, 7)
