import zipfile
from pathlib import Path

import pytest

from src.ingestion.download_st_awfd import (
    DOWNLOAD_URLS,
    download_st_awfd_datasets,
    selected_dataset_keys,
    validate_zip_file,
)


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """datasets:
  - dataset_id: st_awfd_d1
    display_name: ST-AWFD Wafer D1
    source_url: https://github.com/STMicroelectronics/ST-AWFD
    source_type: public_github_repository
    domain: semiconductor_manufacturing
    modality: wafer_timeseries
    task_type: abnormal_material_detection
    label_column: target
    split_column: is_test
    local_raw_path: data/raw/st_awfd/wafer_d1/
    local_processed_path: data/processed/st_awfd_d1/
    expected_pipeline_branch: st_awfd_timeseries_tabular
    license_note: Verify license.
    download_status: planned
    use_in_project: next
    notes: Test D1.
  - dataset_id: st_awfd_d2
    display_name: ST-AWFD Wafer D2
    source_url: https://github.com/STMicroelectronics/ST-AWFD
    source_type: public_github_repository
    domain: semiconductor_manufacturing
    modality: wafer_timeseries
    task_type: abnormal_material_detection
    label_column: target
    split_column: is_test
    local_raw_path: data/raw/st_awfd/wafer_d2/
    local_processed_path: data/processed/st_awfd_d2/
    expected_pipeline_branch: st_awfd_timeseries_tabular
    license_note: Verify license.
    download_status: planned
    use_in_project: next
    notes: Test D2.
""",
        encoding="utf-8",
    )


def _create_zip(path: Path, member_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member_name, "value\n")


def test_selected_dataset_keys_supports_all_and_specific_values() -> None:
    assert selected_dataset_keys("all") == ["d1", "d2"]
    assert selected_dataset_keys("d1") == ["d1"]
    assert selected_dataset_keys("d2") == ["d2"]


def test_selected_dataset_keys_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        selected_dataset_keys("bad")


def test_validate_zip_file_rejects_invalid_zip(tmp_path: Path) -> None:
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_text("not a zip", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid ZIP"):
        validate_zip_file(bad_zip)


def test_download_extract_and_manifest_without_network(tmp_path: Path) -> None:
    registry_path = tmp_path / "configs" / "datasets.yml"
    _write_registry(registry_path)

    def fake_downloader(url: str, destination: Path, force: bool) -> None:
        member = "d1/file.csv" if url == DOWNLOAD_URLS["d1"] else "d2/file.csv"
        _create_zip(destination, member)

    records = download_st_awfd_datasets(
        project_root=tmp_path,
        registry_path=registry_path,
        dataset="all",
        force=False,
        keep_archives=True,
        downloader=fake_downloader,
    )

    assert [record["dataset_id"] for record in records] == ["st_awfd_d1", "st_awfd_d2"]
    assert (tmp_path / "data/raw/st_awfd/archives/D1.zip").exists()
    assert (tmp_path / "data/raw/st_awfd/archives/D2.zip").exists()
    assert (tmp_path / "data/raw/st_awfd/wafer_d1/d1/file.csv").exists()
    assert (tmp_path / "data/raw/st_awfd/wafer_d2/d2/file.csv").exists()
    assert (tmp_path / "data/raw/st_awfd/download_manifest.json").exists()
    assert all(record["extraction_completed"] for record in records)


def test_archives_removed_when_keep_archives_is_false(tmp_path: Path) -> None:
    registry_path = tmp_path / "configs" / "datasets.yml"
    _write_registry(registry_path)

    def fake_downloader(url: str, destination: Path, force: bool) -> None:
        _create_zip(destination, "file.csv")

    download_st_awfd_datasets(
        project_root=tmp_path,
        registry_path=registry_path,
        dataset="d1",
        keep_archives=False,
        downloader=fake_downloader,
    )

    assert not (tmp_path / "data/raw/st_awfd/archives/D1.zip").exists()
    assert (tmp_path / "data/raw/st_awfd/wafer_d1/file.csv").exists()


def test_existing_extracted_files_are_not_overwritten_without_force(tmp_path: Path) -> None:
    registry_path = tmp_path / "configs" / "datasets.yml"
    _write_registry(registry_path)
    existing_file = tmp_path / "data/raw/st_awfd/wafer_d1/file.csv"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_text("existing\n", encoding="utf-8")

    def fake_downloader(url: str, destination: Path, force: bool) -> None:
        _create_zip(destination, "file.csv")

    download_st_awfd_datasets(
        project_root=tmp_path,
        registry_path=registry_path,
        dataset="d1",
        force=False,
        keep_archives=True,
        downloader=fake_downloader,
    )

    assert existing_file.read_text(encoding="utf-8") == "existing\n"
