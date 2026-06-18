"""Build M11 ST-AWFD MaterialID-level retrospective master tables."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_st_awfd_material_master import (
    build_comparison_report,
    build_material_master,
    build_material_master_report,
    load_and_validate_raw_csv,
    write_json,
)


DATASETS = {
    "d1": {
        "dataset_id": "st_awfd_d1",
        "raw_path": PROJECT_ROOT / "data" / "raw" / "st_awfd" / "wafer_d1" / "D1.csv",
        "processed_dir": PROJECT_ROOT / "data" / "processed" / "st_awfd_d1",
        "report_path": PROJECT_ROOT / "reports" / "st_awfd_d1_material_master_report.md",
    },
    "d2": {
        "dataset_id": "st_awfd_d2",
        "raw_path": PROJECT_ROOT / "data" / "raw" / "st_awfd" / "wafer_d2" / "D2.csv",
        "processed_dir": PROJECT_ROOT / "data" / "processed" / "st_awfd_d2",
        "report_path": PROJECT_ROOT / "reports" / "st_awfd_d2_material_master_report.md",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ST-AWFD MaterialID master tables.")
    parser.add_argument(
        "--dataset",
        choices=["d1", "d2", "all"],
        default="all",
        help="Dataset to build. Default: all.",
    )
    return parser.parse_args()


def selected_keys(dataset: str) -> list[str]:
    return ["d1", "d2"] if dataset == "all" else [dataset]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    metadata_records = []
    for key in selected_keys(args.dataset):
        config = DATASETS[key]
        processed_dir = config["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        material_master_path = processed_dir / "material_master.csv"
        sequence_manifest_path = processed_dir / "sequence_manifest.csv"
        metadata_path = processed_dir / "material_master_metadata.json"

        logging.info("Loading %s raw CSV from %s", config["dataset_id"], config["raw_path"])
        raw_df = load_and_validate_raw_csv(config["raw_path"])
        master, sequence_manifest, metadata = build_material_master(
            raw_df,
            dataset_id=config["dataset_id"],
            raw_csv_path=config["raw_path"],
            sequence_manifest_path=sequence_manifest_path,
        )

        master.to_csv(material_master_path, index=False)
        sequence_manifest.to_csv(sequence_manifest_path, index=False)
        write_json(metadata_path, metadata)
        config["report_path"].write_text(
            build_material_master_report(metadata),
            encoding="utf-8",
        )
        metadata_records.append(metadata)

        logging.info("Wrote material master to %s", material_master_path)
        logging.info("Wrote metadata to %s", metadata_path)
        logging.info("Wrote sequence manifest to %s", sequence_manifest_path)
        logging.info("Wrote report to %s", config["report_path"])

    comparison_path = PROJECT_ROOT / "reports" / "st_awfd_material_master_comparison.md"
    comparison_path.write_text(build_comparison_report(metadata_records), encoding="utf-8")
    logging.info("Wrote comparison report to %s", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
