"""Validate the dataset registry and write the M8 expansion plan report."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_registry import generate_dataset_expansion_plan, load_dataset_registry


def main() -> int:
    registry_path = PROJECT_ROOT / "configs" / "datasets.yml"
    report_path = PROJECT_ROOT / "reports" / "dataset_expansion_plan_report.md"

    try:
        datasets = load_dataset_registry(registry_path)
        report = generate_dataset_expansion_plan(datasets)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    dataset_ids = [dataset["dataset_id"] for dataset in datasets]
    print(f"Dataset count: {len(datasets)}")
    print("Dataset IDs: " + ", ".join(dataset_ids))
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
