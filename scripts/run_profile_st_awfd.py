"""Run read-only ST-AWFD D1/D2 profiling."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.profile_st_awfd import (
    build_master_table_design_decision,
    build_profile_report,
    profile_st_awfd_csv,
    write_summary_json,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        (
            "d1",
            PROJECT_ROOT / "data" / "raw" / "st_awfd" / "wafer_d1" / "D1.csv",
            reports_dir / "st_awfd_d1_profile_report.md",
            reports_dir / "st_awfd_d1_profile_summary.json",
        ),
        (
            "d2",
            PROJECT_ROOT / "data" / "raw" / "st_awfd" / "wafer_d2" / "D2.csv",
            reports_dir / "st_awfd_d2_profile_report.md",
            reports_dir / "st_awfd_d2_profile_summary.json",
        ),
    ]

    summaries = []
    for dataset_id, csv_path, report_path, summary_path in datasets:
        logging.info("Profiling %s from %s", dataset_id, csv_path)
        summary = profile_st_awfd_csv(csv_path, dataset_id=dataset_id)
        summaries.append(summary)
        report_path.write_text(build_profile_report(summary), encoding="utf-8")
        write_summary_json(summary, summary_path)
        logging.info("Wrote profile report to %s", report_path)
        logging.info("Wrote profile summary to %s", summary_path)

    design_report_path = reports_dir / "st_awfd_master_table_design_decision.md"
    design_report_path.write_text(
        build_master_table_design_decision(summaries),
        encoding="utf-8",
    )
    logging.info("Wrote master table design decision to %s", design_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
