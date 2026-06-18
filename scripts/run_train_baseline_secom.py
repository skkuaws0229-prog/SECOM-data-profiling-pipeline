"""Train and evaluate Milestone 4 baseline SECOM defect models."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.train_baseline import (
    build_baseline_model_report,
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    build_prediction_sample,
    evaluate_classifier,
    load_feature_set_metadata,
    load_processed_datasets,
    train_logistic_regression,
    train_random_forest,
)


RANDOM_STATE = 42


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed_datasets(processed_dir)
    metadata = load_feature_set_metadata(processed_dir)

    logistic_model = train_logistic_regression(
        X_train,
        y_train,
        random_state=RANDOM_STATE,
    )
    random_forest_model = train_random_forest(
        X_train,
        y_train,
        random_state=RANDOM_STATE,
    )

    models = [
        ("logistic_regression", logistic_model),
        ("random_forest", random_forest_model),
    ]

    metrics = [
        evaluate_classifier(model, X_test, y_test, model_name)
        for model_name, model in models
    ]
    metrics_df = build_metrics_dataframe(metrics)

    confusion_df = pd.concat(
        [
            build_confusion_matrix_dataframe(
                y_test,
                pd.Series(model.predict(X_test), index=y_test.index),
                model_name,
            )
            for model_name, model in models
        ],
        ignore_index=True,
    )
    prediction_sample_df = pd.concat(
        [
            build_prediction_sample(model, X_test, y_test, model_name, n=30)
            for model_name, model in models
        ],
        ignore_index=True,
    )

    metrics_path = reports_dir / "secom_baseline_model_metrics.csv"
    confusion_path = reports_dir / "secom_confusion_matrix.csv"
    prediction_sample_path = reports_dir / "secom_prediction_sample.csv"
    report_path = reports_dir / "secom_baseline_model_report.md"

    metrics_df.to_csv(metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    prediction_sample_df.to_csv(prediction_sample_path, index=False)
    report_path.write_text(
        build_baseline_model_report(metrics_df, confusion_df, metadata),
        encoding="utf-8",
    )

    logging.info("Wrote baseline model metrics to %s", metrics_path)
    logging.info("Wrote confusion matrix to %s", confusion_path)
    logging.info("Wrote prediction sample to %s", prediction_sample_path)
    logging.info("Wrote baseline model report to %s", report_path)


if __name__ == "__main__":
    main()
