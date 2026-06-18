"""Download and extract ST-AWFD wafer datasets D1 and D2."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.data_registry import load_dataset_registry


DOWNLOAD_URLS = {
    "d1": "https://raw.githubusercontent.com/STMicroelectronics/ST-AWFD/main/Datasets/D1.zip",
    "d2": "https://raw.githubusercontent.com/STMicroelectronics/ST-AWFD/main/Datasets/D2.zip",
}
DATASET_ID_BY_SHORT_NAME = {
    "d1": "st_awfd_d1",
    "d2": "st_awfd_d2",
}
DATASET_CHOICES = {"d1", "d2", "all"}


def selected_dataset_keys(dataset: str) -> list[str]:
    """Return normalized ST-AWFD dataset keys for a CLI selection."""
    dataset = dataset.lower()
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"Unsupported dataset selection: {dataset}")
    if dataset == "all":
        return ["d1", "d2"]
    return [dataset]


def _registry_by_id(registry_path: Path) -> dict[str, dict]:
    return {
        entry["dataset_id"]: entry
        for entry in load_dataset_registry(registry_path)
    }


def stream_download(url: str, destination: Path, force: bool = False) -> None:
    """Stream a URL to a .part file and rename only after success."""
    destination = Path(destination)
    part_path = destination.with_suffix(destination.suffix + ".part")

    if destination.exists() and not force:
        validate_zip_file(destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if part_path.exists():
        part_path.unlink()

    try:
        with urllib.request.urlopen(url) as response:
            with part_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        part_path.replace(destination)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise


def validate_zip_file(path: Path) -> None:
    """Validate that a path is a readable ZIP archive."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Downloaded archive not found: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Downloaded archive is not a valid ZIP file: {path}")
    with zipfile.ZipFile(path) as archive:
        bad_file = archive.testzip()
    if bad_file is not None:
        raise ValueError(f"ZIP validation failed for {path}; first bad file: {bad_file}")


def _has_extracted_files(path: Path) -> bool:
    return path.exists() and any(child.is_file() for child in path.rglob("*"))


def extract_zip(archive_path: Path, extraction_path: Path, force: bool = False) -> int:
    """Extract a ZIP archive and return extracted file count."""
    archive_path = Path(archive_path)
    extraction_path = Path(extraction_path)
    if _has_extracted_files(extraction_path) and not force:
        return sum(1 for child in extraction_path.rglob("*") if child.is_file())

    extraction_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extraction_path)
    return sum(1 for child in extraction_path.rglob("*") if child.is_file())


def build_manifest_record(
    dataset_key: str,
    registry_entry: dict,
    archive_path: Path,
    extraction_path: Path,
    extracted_file_count: int,
) -> dict:
    """Build one manifest record for a downloaded ST-AWFD dataset."""
    return {
        "dataset_id": registry_entry["dataset_id"],
        "source_url": registry_entry["source_url"],
        "download_url": DOWNLOAD_URLS[dataset_key],
        "archive_path": str(archive_path),
        "extraction_path": str(extraction_path),
        "archive_size_bytes": int(archive_path.stat().st_size) if archive_path.exists() else 0,
        "extracted_file_count": int(extracted_file_count),
        "downloaded_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "extraction_completed": True,
        "license_note": registry_entry["license_note"],
    }


def download_st_awfd_datasets(
    *,
    project_root: Path,
    registry_path: Path,
    dataset: str = "all",
    force: bool = False,
    keep_archives: bool = False,
    downloader: Callable[[str, Path, bool], None] = stream_download,
) -> list[dict]:
    """Download, validate, extract, and manifest selected ST-AWFD datasets."""
    project_root = Path(project_root)
    registry = _registry_by_id(registry_path)
    archives_dir = project_root / "data" / "raw" / "st_awfd" / "archives"
    manifest_path = project_root / "data" / "raw" / "st_awfd" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for dataset_key in selected_dataset_keys(dataset):
        dataset_id = DATASET_ID_BY_SHORT_NAME[dataset_key]
        if dataset_id not in registry:
            raise ValueError(f"Dataset registry missing required entry: {dataset_id}")

        registry_entry = registry[dataset_id]
        archive_path = archives_dir / f"{dataset_key.upper()}.zip"
        extraction_path = project_root / registry_entry["local_raw_path"]

        downloader(DOWNLOAD_URLS[dataset_key], archive_path, force)
        validate_zip_file(archive_path)
        extracted_file_count = extract_zip(archive_path, extraction_path, force=force)
        record = build_manifest_record(
            dataset_key=dataset_key,
            registry_entry=registry_entry,
            archive_path=archive_path,
            extraction_path=extraction_path,
            extracted_file_count=extracted_file_count,
        )
        records.append(record)

        if not keep_archives and archive_path.exists():
            archive_path.unlink()

    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    return records
