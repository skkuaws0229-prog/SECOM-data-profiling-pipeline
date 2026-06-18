"""Run M12 ST-AWFD retrospective baseline models for D1 and/or D2."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.train_st_awfd_baseline import (
    build_comparison_report,
    train_and_evaluate_dataset,
    write_json,
)


DATASETS = {
    "d1": {
        "dataset_id": "st_awfd_d1",
        "master_path": PROJECT_ROOT / "data" / "processed" / "st_awfd_d1" / "material_master.csv",
        "processed_dir": PROJECT_ROOT / "data" / "processed" / "st_awfd_d1",
        "model_dir": PROJECT_ROOT / "models" / "st_awfd_d1",
    },
    "d2": {
        "dataset_id": "st_awfd_d2",
        "master_path": PROJECT_ROOT / "data" / "processed" / "st_awfd_d2" / "material_master.csv",
        "processed_dir": PROJECT_ROOT / "data" / "processed" / "st_awfd_d2",
        "model_dir": PROJECT_ROOT / "models" / "st_awfd_d2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ST-AWFD MaterialID-level retrospective baseline models."
    )
    parser.add_argument(
        "--dataset",
        choices=["d1", "d2", "all"],
        default="all",
        help="Dataset to train. Default: all.",
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
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_dataset = {}
    for key in selected_keys(args.dataset):
        config = DATASETS[key]
        dataset_id = config["dataset_id"]
        logging.info("Training %s baseline models from %s", dataset_id, config["master_path"])
        result = train_and_evaluate_dataset(
            dataset_id=dataset_id,
            master_path=config["master_path"],
            model_dir=config["model_dir"],
        )

        metrics_path = reports_dir / f"{dataset_id}_baseline_metrics.csv"
        confusion_path = reports_dir / f"{dataset_id}_baseline_confusion_matrix.csv"
        predictions_path = reports_dir / f"{dataset_id}_baseline_predictions.csv"
        report_path = reports_dir / f"{dataset_id}_baseline_model_report.md"
        metadata_path = config["processed_dir"] / "baseline_feature_metadata.json"

        result["metrics_df"].to_csv(metrics_path, index=False)
        result["confusion_df"].to_csv(confusion_path, index=False)
        result["predictions_df"].to_csv(predictions_path, index=False)
        report_path.write_text(result["report"], encoding="utf-8")
        write_json(metadata_path, result["metadata"])
        metrics_by_dataset[dataset_id] = result["metrics_df"]

        logging.info("Wrote metrics to %s", metrics_path)
        logging.info("Wrote confusion matrix to %s", confusion_path)
        logging.info("Wrote predictions to %s", predictions_path)
        logging.info("Wrote report to %s", report_path)
        logging.info("Wrote feature metadata to %s", metadata_path)
        logging.info("Wrote models to %s", config["model_dir"])

    comparison_path = reports_dir / "st_awfd_baseline_comparison.md"
    comparison_path.write_text(build_comparison_report(metrics_by_dataset), encoding="utf-8")
    logging.info("Wrote comparison report to %s", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
