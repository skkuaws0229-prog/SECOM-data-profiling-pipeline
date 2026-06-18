# Milestone 9 - ST-AWFD Acquisition & Inventory

Milestone 9 adds automated local acquisition for ST-AWFD Wafer D1 and D2.

The downloader uses the configured official registry URLs and downloads:

- ST-AWFD D1
- ST-AWFD D2

## Local Raw Outputs

Raw outputs are stored locally under:

```text
data/raw/st_awfd/
```

Expected local files:

```text
data/raw/st_awfd/wafer_d1/D1.csv
data/raw/st_awfd/wafer_d2/D2.csv
data/raw/st_awfd/download_manifest.json
```

Raw CSV files, ZIP archives, and `download_manifest.json` are intentionally ignored by Git.

## Download and Extraction Behavior

The downloader streams each ZIP archive to a temporary `.part` file and renames it only after download success.

Before extraction, each ZIP archive is validated with Python's `zipfile` module.

ZIP archives are removed by default after successful validation and extraction. Use `--keep-archives` to preserve downloaded ZIP files under:

```text
data/raw/st_awfd/archives/
```

Use `--force` to overwrite existing archives and extracted files.

## Scope Boundary

This milestone does not perform:

- profiling
- modeling master table creation
- feature engineering
- model training
- RAG, Agent, API, Docker, dashboard, or cloud work

The next step is read-only profiling of:

```text
data/raw/st_awfd/wafer_d1/D1.csv
data/raw/st_awfd/wafer_d2/D2.csv
```

## Run

```bash
python scripts/download_st_awfd.py --dataset all
```
