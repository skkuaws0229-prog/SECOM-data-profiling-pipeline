"""Build the M3.5 SECOM modeling master table."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_modeling_master import (
    build_modeling_master_report,
    build_modeling_master_table,
    load_feature_quality_summary,
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
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "reports"
    quality_summary_path = reports_dir / "secom_feature_quality_summary.csv"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    secom_df = load_secom_from_raw_dir(raw_dir)
    quality_summary = load_feature_quality_summary(quality_summary_path)
    master_table, metadata = build_modeling_master_table(
        secom_df,
        quality_summary,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    master_table_path = processed_dir / "modeling_master_table.csv"
    metadata_path = processed_dir / "modeling_master_metadata.json"
    report_path = reports_dir / "secom_modeling_master_table_report.md"

    master_table.to_csv(master_table_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(build_modeling_master_report(metadata), encoding="utf-8")

    logging.info("Wrote modeling master table to %s", master_table_path)
    logging.info("Wrote modeling master metadata to %s", metadata_path)
    logging.info("Wrote modeling master report to %s", report_path)


if __name__ == "__main__":
    main()
