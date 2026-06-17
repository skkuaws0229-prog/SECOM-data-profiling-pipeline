"""Load the UCI SECOM raw feature and label files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required SECOM file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")


def _sensor_columns(feature_count: int) -> list[str]:
    return [f"sensor_{index:03d}" for index in range(feature_count)]


def load_secom_features(feature_path: Path) -> pd.DataFrame:
    """Load `secom.data` and assign anonymized sensor column names."""
    feature_path = Path(feature_path)
    _validate_file(feature_path)

    logger.info("Loading SECOM features from %s", feature_path)
    features = pd.read_csv(
        feature_path,
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan", "?"],
        engine="python",
    )
    features.columns = _sensor_columns(features.shape[1])
    return features


def load_secom_labels(label_path: Path) -> pd.DataFrame:
    """Load `secom_labels.data` into label and timestamp columns."""
    label_path = Path(label_path)
    _validate_file(label_path)

    logger.info("Loading SECOM labels from %s", label_path)
    records: list[dict[str, object]] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid label row at line {line_number}: expected label and timestamp"
                )

            label_text, timestamp = parts
            records.append({"label": int(label_text), "timestamp": timestamp})

    return pd.DataFrame.from_records(records, columns=["label", "timestamp"])


def load_secom_data(feature_path: Path, label_path: Path) -> pd.DataFrame:
    """Return one DataFrame with sample_id, sensor columns, label, and timestamp."""
    features = load_secom_features(Path(feature_path))
    labels = load_secom_labels(Path(label_path))

    if len(features) != len(labels):
        raise ValueError(
            "SECOM feature and label row counts do not match: "
            f"{len(features)} features rows vs {len(labels)} label rows"
        )

    sample_ids = pd.Series(range(len(features)), name="sample_id")
    data = pd.concat([sample_ids, features, labels], axis=1)

    logger.info(
        "Loaded SECOM dataset with %s rows and %s sensor features",
        len(data),
        features.shape[1],
    )
    return data


def load_secom_from_raw_dir(raw_dir: Path) -> pd.DataFrame:
    """Load SECOM data from a directory containing the standard raw file names."""
    raw_dir = Path(raw_dir)
    return load_secom_data(
        feature_path=raw_dir / "secom.data",
        label_path=raw_dir / "secom_labels.data",
    )
