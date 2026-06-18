"""Dataset registry utilities for manufacturing AI dataset expansion."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "dataset_id",
    "display_name",
    "source_url",
    "source_type",
    "domain",
    "modality",
    "task_type",
    "label_column",
    "split_column",
    "local_raw_path",
    "local_processed_path",
    "expected_pipeline_branch",
    "license_note",
    "download_status",
    "use_in_project",
    "notes",
]
ALLOWED_MODALITIES = {"tabular_sensor", "wafer_timeseries", "wafer_map_image"}
ALLOWED_USE_IN_PROJECT = {"current", "next", "phase_2", "hold"}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    return value


def _parse_dataset_yaml(text: str) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_datasets = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "datasets:":
            in_datasets = True
            continue
        if not in_datasets:
            continue
        if stripped.startswith("- "):
            if current is not None:
                datasets.append(current)
            current = {}
            item = stripped[2:].strip()
            if item:
                key, value = item.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
            continue
        if current is None:
            raise ValueError("Invalid registry YAML: field found before dataset item")
        if ":" not in stripped:
            raise ValueError(f"Invalid registry YAML line: {raw_line}")
        key, value = stripped.split(":", 1)
        current[key.strip()] = _parse_scalar(value)

    if current is not None:
        datasets.append(current)
    if not datasets:
        raise ValueError("No datasets found in registry")
    return datasets


def load_dataset_registry(path: Path) -> list[dict[str, Any]]:
    """Load dataset registry entries from configs/datasets.yml."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset registry file not found: {path}")
    datasets = _parse_dataset_yaml(path.read_text(encoding="utf-8"))
    validate_dataset_entries(datasets)
    return datasets


def validate_dataset_entries(datasets: list[dict[str, Any]]) -> None:
    """Validate required dataset registry fields and controlled values."""
    seen_ids: set[str] = set()
    for dataset in datasets:
        missing = [field for field in REQUIRED_FIELDS if field not in dataset]
        if missing:
            dataset_id = dataset.get("dataset_id", "<unknown>")
            raise ValueError(f"Dataset {dataset_id} missing required fields: {missing}")

        dataset_id = str(dataset["dataset_id"])
        if dataset_id in seen_ids:
            raise ValueError(f"Duplicate dataset_id in registry: {dataset_id}")
        seen_ids.add(dataset_id)

        if not str(dataset["source_url"]).strip():
            raise ValueError(f"Dataset {dataset_id} has empty source_url")
        if dataset["modality"] not in ALLOWED_MODALITIES:
            raise ValueError(
                f"Dataset {dataset_id} has unsupported modality: {dataset['modality']}"
            )
        if dataset["use_in_project"] not in ALLOWED_USE_IN_PROJECT:
            raise ValueError(
                "Dataset "
                f"{dataset_id} has unsupported use_in_project: {dataset['use_in_project']}"
            )


def get_dataset_entries(path: Path) -> list[dict[str, Any]]:
    """Return validated dataset entries."""
    return load_dataset_registry(path)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join("" if row.get(column) is None else str(row.get(column)) for column in columns)
            + " |"
        )
    return "\n".join(table)


def generate_dataset_expansion_plan(datasets: list[dict[str, Any]]) -> str:
    """Generate a Markdown dataset expansion plan from registry metadata."""
    validate_dataset_entries(datasets)
    summary_columns = [
        "dataset_id",
        "display_name",
        "modality",
        "task_type",
        "download_status",
        "use_in_project",
        "expected_pipeline_branch",
    ]

    lines = [
        "# Dataset Expansion Plan",
        "",
        "Milestone 8 prepares a metadata registry for expanding the manufacturing AI portfolio beyond SECOM. It does not download large datasets, train new models, or move existing SECOM outputs.",
        "",
        "## Why SECOM Alone Is Insufficient",
        "",
        "SECOM is valuable for tabular sensor defect prediction, but the fail sample count is limited. A portfolio that only uses SECOM can demonstrate the pipeline, but it gives limited evidence for broader manufacturing AI readiness.",
        "",
        "## Registry Summary",
        "",
        _markdown_table(datasets, summary_columns),
        "",
        "## Modality Warnings",
        "",
        "- SECOM is a tabular sensor classification dataset.",
        "- ST-AWFD is a wafer time-series / production lot fault detection branch.",
        "- WM-811K is a wafer map image classification branch.",
        "",
        "These modalities should not be forced into one identical pipeline. Each branch needs appropriate preprocessing and modeling assumptions.",
        "",
        "## Expansion Rationale",
        "",
        "ST-AWFD is the next best fit because it remains close to semiconductor manufacturing quality while expanding beyond static SECOM tabular rows into wafer time-series and production lot fault detection.",
        "",
        "WM-811K is useful for demonstrating wafer map defect pattern classification, but it should be a separate image branch rather than a tabular SECOM extension.",
        "",
        "Synthetic datasets are not used for primary performance claims because the portfolio should emphasize real public manufacturing datasets and avoid overstating model quality on artificial data.",
        "",
        "## Portfolio Value",
        "",
        "This registry supports a multi-dataset manufacturing AI portfolio by making dataset provenance, modality, task type, local paths, license checks, and planned pipeline branches explicit before any large downloads occur.",
        "",
        "## Dataset Details",
        "",
    ]
    for dataset in datasets:
        lines.extend(
            [
                f"### {dataset['dataset_id']} - {dataset['display_name']}",
                "",
                f"- Source: {dataset['source_url']}",
                f"- Modality: {dataset['modality']}",
                f"- Task type: {dataset['task_type']}",
                f"- Label column: {dataset['label_column']}",
                f"- Split column: {dataset['split_column']}",
                f"- Raw path: {dataset['local_raw_path']}",
                f"- Processed path: {dataset['local_processed_path']}",
                f"- Pipeline branch: {dataset['expected_pipeline_branch']}",
                f"- Download status: {dataset['download_status']}",
                f"- Project use: {dataset['use_in_project']}",
                f"- License note: {dataset['license_note']}",
                f"- Notes: {dataset['notes']}",
                "",
            ]
        )
    return "\n".join(lines)
