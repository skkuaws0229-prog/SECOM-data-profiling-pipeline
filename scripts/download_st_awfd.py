"""CLI for downloading ST-AWFD D1/D2 archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.download_st_awfd import download_st_awfd_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ST-AWFD D1/D2 datasets.")
    parser.add_argument(
        "--dataset",
        choices=["d1", "d2", "all"],
        default="all",
        help="Dataset to download. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing archives and extracted files.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep ZIP archives after successful extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = PROJECT_ROOT / "configs" / "datasets.yml"
    try:
        records = download_st_awfd_datasets(
            project_root=PROJECT_ROOT,
            registry_path=registry_path,
            dataset=args.dataset,
            force=args.force,
            keep_archives=args.keep_archives,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Downloaded/extracted dataset count: {len(records)}")
    for record in records:
        print(
            f"{record['dataset_id']}: {record['extracted_file_count']} files -> "
            f"{record['extraction_path']}"
        )
    print(f"Manifest: {PROJECT_ROOT / 'data' / 'raw' / 'st_awfd' / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
