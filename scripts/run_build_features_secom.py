"""Run the Milestone 3 SECOM feature engineering pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    build_feature_engineering_report,
    build_feature_set_metadata,
    create_train_test_split,
    fit_transform_median_imputer,
    label_distribution,
    load_feature_quality_summary,
    select_model_features,
    split_features_and_label,
    summarize_feature_quality_counts,
)
from src.ingestion.load_secom import load_secom_from_raw_dir


TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    raw_dir = PROJECT_ROOT / "data" / "raw"
    reports_dir = PROJECT_ROOT / "reports"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    quality_summary_path = reports_dir / "secom_feature_quality_summary.csv"

    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    data = load_secom_from_raw_dir(raw_dir)
    quality_summary = load_feature_quality_summary(quality_summary_path)
    feature_columns = select_model_features(data, quality_summary)
    X, y = split_features_and_label(data, feature_columns)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train_imputed, X_test_imputed, medians = fit_transform_median_imputer(
        X_train,
        X_test,
    )

    quality_counts = summarize_feature_quality_counts(quality_summary, feature_columns)
    metadata = build_feature_set_metadata(
        **quality_counts,
        feature_columns=feature_columns,
        train_row_count=len(X_train_imputed),
        test_row_count=len(X_test_imputed),
        train_label_distribution=label_distribution(y_train),
        test_label_distribution=label_distribution(y_test),
        imputation_values=medians,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train_path = processed_dir / "X_train.csv"
    X_test_path = processed_dir / "X_test.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"
    metadata_path = processed_dir / "feature_set_metadata.json"
    report_path = reports_dir / "secom_feature_engineering_report.md"

    X_train_imputed.to_csv(X_train_path, index=False)
    X_test_imputed.to_csv(X_test_path, index=False)
    y_train.to_frame(name="label").to_csv(y_train_path, index=False)
    y_test.to_frame(name="label").to_csv(y_test_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(build_feature_engineering_report(metadata), encoding="utf-8")

    logging.info("Wrote X_train to %s", X_train_path)
    logging.info("Wrote X_test to %s", X_test_path)
    logging.info("Wrote y_train to %s", y_train_path)
    logging.info("Wrote y_test to %s", y_test_path)
    logging.info("Wrote feature set metadata to %s", metadata_path)
    logging.info("Wrote feature engineering report to %s", report_path)


if __name__ == "__main__":
    main()
