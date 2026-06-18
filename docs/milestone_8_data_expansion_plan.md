# Milestone 8 - Data Registry & Dataset Expansion Plan

Milestone 8 adds a metadata-only dataset registry for expanding the manufacturing AI portfolio beyond SECOM.

This milestone does not download large datasets, train new models, move existing SECOM outputs, or fabricate dataset metrics.

## Why SECOM Alone Is Insufficient

SECOM is useful for a first end-to-end manufacturing AI pipeline because it provides real tabular semiconductor sensor data and pass/fail labels. However, the fail samples are limited, so SECOM alone is not enough to support broad performance claims across manufacturing AI tasks.

The project should keep SECOM as the current tabular baseline while planning additional public datasets that represent different manufacturing modalities.

## Why ST-AWFD Is the Next Best Fit

ST-AWFD is a strong next branch because it remains close to semiconductor manufacturing quality while expanding the task beyond static SECOM tabular rows. It introduces wafer time-series / production lot fault detection, which is a natural extension for manufacturing data engineering and modeling.

The planned ST-AWFD D1 and D2 branches should use their own time-series-aware preprocessing and validation assumptions.

## Why WM-811K Should Be Separate

WM-811K is valuable for wafer map defect pattern classification, but it is an image-style wafer map dataset. It should be handled as a separate image classification branch, not forced into the SECOM tabular sensor pipeline.

## Synthetic Data Policy

Synthetic datasets are not used for primary performance claims. They may be useful for testing software behavior, but portfolio claims should be based on real public manufacturing datasets whenever possible.

## Portfolio Value

The registry supports a multi-dataset manufacturing AI portfolio by making these items explicit before downloads occur:

- source URL
- modality
- task type
- label column
- local raw and processed paths
- expected pipeline branch
- license note
- download status
- intended project phase

This creates a clean roadmap for expanding from SECOM tabular defect prediction into time-series wafer fault detection and wafer map image classification.

## Run

```bash
python scripts/validate_dataset_registry.py
```

Expected output:

```text
reports/dataset_expansion_plan_report.md
```
