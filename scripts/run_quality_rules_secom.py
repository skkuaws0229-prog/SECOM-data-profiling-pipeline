"""Run SECOM feature quality rules for Milestone 2."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.load_secom import load_secom_from_raw_dir
from src.quality.apply_quality_rules import (
    build_feature_quality_summary,
    build_quality_decision_report,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    raw_dir = PROJECT_ROOT / "data" / "raw"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = load_secom_from_raw_dir(raw_dir)
    summary = build_feature_quality_summary(data)

    summary_path = reports_dir / "secom_feature_quality_summary.csv"
    decision_report_path = reports_dir / "secom_quality_decision_report.md"

    summary.to_csv(summary_path, index=False)
    decision_report_path.write_text(
        build_quality_decision_report(summary),
        encoding="utf-8",
    )

    logging.info("Wrote feature quality summary to %s", summary_path)
    logging.info("Wrote quality decision report to %s", decision_report_path)


if __name__ == "__main__":
    main()
