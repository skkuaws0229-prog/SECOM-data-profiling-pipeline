from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.load_secom import load_secom_data


def test_load_secom_data_assigns_sensor_columns_and_metadata(tmp_path: Path) -> None:
    feature_path = tmp_path / "secom.data"
    label_path = tmp_path / "secom_labels.data"
    feature_path.write_text("1.0 2.0 NaN\n4.0 5.5 6.0\n", encoding="utf-8")
    label_path.write_text(
        "-1 19/07/2008 11:55:00\n1 19/07/2008 12:32:00\n",
        encoding="utf-8",
    )

    data = load_secom_data(feature_path, label_path)

    assert list(data.columns) == [
        "sample_id",
        "sensor_000",
        "sensor_001",
        "sensor_002",
        "label",
        "timestamp",
    ]
    assert data.shape == (2, 6)
    assert data["sample_id"].tolist() == [0, 1]
    assert data["label"].tolist() == [-1, 1]
    assert data["timestamp"].tolist() == [
        "19/07/2008 11:55:00",
        "19/07/2008 12:32:00",
    ]
    assert pd.isna(data.loc[0, "sensor_002"])


def test_load_secom_data_rejects_mismatched_row_counts(tmp_path: Path) -> None:
    feature_path = tmp_path / "secom.data"
    label_path = tmp_path / "secom_labels.data"
    feature_path.write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")
    label_path.write_text("-1 19/07/2008 11:55:00\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row counts do not match"):
        load_secom_data(feature_path, label_path)
