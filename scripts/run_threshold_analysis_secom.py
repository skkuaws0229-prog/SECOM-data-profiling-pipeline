"""Run M5 threshold and error analysis on the SECOM modeling master table."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.threshold_analysis import (
    build_false_negative_analysis,
    build_threshold_error_analysis_report,
    build_top_risk_samples,
    evaluate_thresholds,
    load_modeling_master_table,
    predict_defect_probability,
    split_train_test_from_master,
    train_logistic_regression,
    train_random_forest,
)


RANDOM_STATE = 42


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    master_table_path = PROJECT_ROOT / "data" / "processed" / "modeling_master_table.csv"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    master_df = load_modeling_master_table(master_table_path)
    X_train, X_test, y_train, y_test, test_df = split_train_test_from_master(master_df)

    models = {
        "logistic_regression": train_logistic_regression(
            X_train,
            y_train,
            random_state=RANDOM_STATE,
        ),
        "random_forest": train_random_forest(
            X_train,
            y_train,
            random_state=RANDOM_STATE,
        ),
    }

    model_scores = {
        model_name: predict_defect_probability(model, X_test)
        for model_name, model in models.items()
    }
    threshold_metrics = pd.concat(
        [
            evaluate_thresholds(y_test, scores, model_name)
            for model_name, scores in model_scores.items()
        ],
        ignore_index=True,
    )
    top_risk_samples = build_top_risk_samples(test_df, model_scores)
    false_negative_analysis = build_false_negative_analysis(
        test_df,
        model_scores,
        threshold_metrics,
    )

    threshold_metrics_path = reports_dir / "secom_threshold_metrics.csv"
    top_risk_path = reports_dir / "secom_top_risk_samples.csv"
    false_negative_path = reports_dir / "secom_false_negative_analysis.csv"
    report_path = reports_dir / "secom_threshold_error_analysis_report.md"

    threshold_metrics.to_csv(threshold_metrics_path, index=False)
    top_risk_samples.to_csv(top_risk_path, index=False)
    false_negative_analysis.to_csv(false_negative_path, index=False)
    report_path.write_text(
        build_threshold_error_analysis_report(
            threshold_metrics,
            false_negative_analysis,
        ),
        encoding="utf-8",
    )

    logging.info("Wrote threshold metrics to %s", threshold_metrics_path)
    logging.info("Wrote top risk samples to %s", top_risk_path)
    logging.info("Wrote false negative analysis to %s", false_negative_path)
    logging.info("Wrote threshold error analysis report to %s", report_path)


if __name__ == "__main__":
    main()
