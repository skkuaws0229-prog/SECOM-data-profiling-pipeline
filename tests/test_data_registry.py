from pathlib import Path

from src.data_registry import (
    ALLOWED_MODALITIES,
    ALLOWED_USE_IN_PROJECT,
    REQUIRED_FIELDS,
    generate_dataset_expansion_plan,
    load_dataset_registry,
)


REGISTRY_PATH = Path("configs/datasets.yml")


def test_registry_file_exists() -> None:
    assert REGISTRY_PATH.exists()


def test_all_required_dataset_ids_exist() -> None:
    datasets = load_dataset_registry(REGISTRY_PATH)
    dataset_ids = {dataset["dataset_id"] for dataset in datasets}

    assert {"secom", "st_awfd_d1", "st_awfd_d2", "wm811k"}.issubset(dataset_ids)


def test_required_fields_exist() -> None:
    datasets = load_dataset_registry(REGISTRY_PATH)

    for dataset in datasets:
        for field in REQUIRED_FIELDS:
            assert field in dataset


def test_no_dataset_has_empty_source_url() -> None:
    datasets = load_dataset_registry(REGISTRY_PATH)

    assert all(str(dataset["source_url"]).strip() for dataset in datasets)


def test_modality_values_are_allowed() -> None:
    datasets = load_dataset_registry(REGISTRY_PATH)

    assert all(dataset["modality"] in ALLOWED_MODALITIES for dataset in datasets)


def test_use_in_project_values_are_allowed() -> None:
    datasets = load_dataset_registry(REGISTRY_PATH)

    assert all(dataset["use_in_project"] in ALLOWED_USE_IN_PROJECT for dataset in datasets)


def test_report_generation_returns_markdown() -> None:
    report = generate_dataset_expansion_plan(load_dataset_registry(REGISTRY_PATH))

    assert report.startswith("# Dataset Expansion Plan")
    assert "## Modality Warnings" in report


def test_wm811k_is_image_branch_not_tabular_branch() -> None:
    datasets = load_dataset_registry(REGISTRY_PATH)
    wm811k = next(dataset for dataset in datasets if dataset["dataset_id"] == "wm811k")

    assert wm811k["modality"] == "wafer_map_image"
    assert wm811k["expected_pipeline_branch"] == "wm811k_image_classification"
    assert "tabular" not in wm811k["expected_pipeline_branch"]
