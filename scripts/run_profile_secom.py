"""Run SECOM loading and profiling for Milestone 1."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.load_secom import load_secom_from_raw_dir
from src.quality.profile_secom import build_markdown_report, profile_secom_data


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    raw_dir = PROJECT_ROOT / "data" / "raw"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = load_secom_from_raw_dir(raw_dir)
    profile = profile_secom_data(data)

    report_path = reports_dir / "secom_data_profile_report.md"
    missing_report_path = reports_dir / "secom_missing_value_report.csv"
    label_distribution_path = reports_dir / "secom_label_distribution.csv"

    report_path.write_text(build_markdown_report(profile), encoding="utf-8")
    profile["missing_value_report"].to_csv(missing_report_path, index=False)
    profile["label_distribution"].to_csv(label_distribution_path, index=False)

    logging.info("Wrote profile report to %s", report_path)
    logging.info("Wrote missing value report to %s", missing_report_path)
    logging.info("Wrote label distribution to %s", label_distribution_path)


if __name__ == "__main__":
    main()
