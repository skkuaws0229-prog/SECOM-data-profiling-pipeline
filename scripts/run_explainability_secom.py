"""Run M6 explainability and feature profile analysis."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.explainability import (
    build_error_feature_profile,
    build_explainability_report,
    build_feature_importance_table,
    build_top_feature_summary,
    load_m5_error_outputs,
    load_modeling_master_table,
    split_train_test_from_master,
    train_logistic_regression,
    train_random_forest,
)


RANDOM_STATE = 42
TOP_FEATURE_COUNT = 20


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
    top_risk_samples, false_negative_samples = load_m5_error_outputs(reports_dir)

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

    feature_importance = build_feature_importance_table(
        logistic_model=logistic_model,
        random_forest_model=random_forest_model,
        X_test=X_test,
        y_test=y_test,
        include_permutation=True,
    )
    top_feature_summary = build_top_feature_summary(
        feature_importance,
        top_n=TOP_FEATURE_COUNT,
    )
    selected_features = sorted(top_feature_summary["feature"].unique().tolist())
    error_feature_profile = build_error_feature_profile(
        test_df=test_df,
        top_risk_samples=top_risk_samples,
        false_negative_samples=false_negative_samples,
        selected_features=selected_features,
    )

    feature_importance_path = reports_dir / "secom_feature_importance.csv"
    top_summary_path = reports_dir / "secom_top_feature_summary.csv"
    error_profile_path = reports_dir / "secom_error_feature_profile.csv"
    report_path = reports_dir / "secom_explainability_report.md"

    feature_importance.to_csv(feature_importance_path, index=False)
    top_feature_summary.to_csv(top_summary_path, index=False)
    error_feature_profile.to_csv(error_profile_path, index=False)
    report_path.write_text(
        build_explainability_report(
            feature_importance,
            top_feature_summary,
            error_feature_profile,
        ),
        encoding="utf-8",
    )

    logging.info("Wrote feature importance to %s", feature_importance_path)
    logging.info("Wrote top feature summary to %s", top_summary_path)
    logging.info("Wrote error feature profile to %s", error_profile_path)
    logging.info("Wrote explainability report to %s", report_path)


if __name__ == "__main__":
    main()
