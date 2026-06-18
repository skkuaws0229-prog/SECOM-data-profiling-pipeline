# Milestone 11 - ST-AWFD MaterialID-level Retrospective Master Tables

Milestone 11 creates separate retrospective MaterialID-level master tables for ST-AWFD D1 and D2.

## Scope

M11 keeps D1 and D2 separate. It does not merge ST-AWFD with SECOM, train models, impute values, or modify raw CSV files.

One output row equals one `MaterialID`.

Raw event-level files remain preserved for future sequence-aware and early-warning modeling.

## Validation

The builder requires these raw columns:

```text
MaterialID
StepID
duration_ms
target
is_test
```

It fails if required columns are missing, if `target` or `is_test` are not binary, or if either value varies within a `MaterialID`.

## Master Table Design

Metadata columns appear first:

```text
dataset_id
material_id
label_original
label_binary
official_split
source_record_count
step_count
feature_set_version
preprocessing_version
aggregation_strategy
sequence_manifest_path
```

`is_test` is mapped as:

- `0` -> `train`
- `1` -> `test`

The table aggregates `duration_ms` and numeric raw features retrospectively. It also keeps per-observed-`StepID` feature means without hardcoding feature names or step IDs.

## Sequence Manifest

Each dataset also gets a `sequence_manifest.csv` with one row per `MaterialID`. It preserves observed steps, duration range, raw event file path, record counts, split, label, and sequence availability.

## Outputs

```text
data/processed/st_awfd_d1/material_master.csv
data/processed/st_awfd_d1/material_master_metadata.json
data/processed/st_awfd_d1/sequence_manifest.csv
reports/st_awfd_d1_material_master_report.md

data/processed/st_awfd_d2/material_master.csv
data/processed/st_awfd_d2/material_master_metadata.json
data/processed/st_awfd_d2/sequence_manifest.csv
reports/st_awfd_d2_material_master_report.md

reports/st_awfd_material_master_comparison.md
```

Generated `material_master.csv` and `sequence_manifest.csv` files are local artifacts and should not be committed.
