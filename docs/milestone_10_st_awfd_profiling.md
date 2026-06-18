# Milestone 10 - ST-AWFD Read-only Profiling & Master Table Design

Milestone 10 performs read-only profiling of the local ST-AWFD D1 and D2 raw CSV files.

ST-AWFD raw data has event-level / time-sample rows. Each `MaterialID` appears across multiple `StepID` rows with associated timing and feature values.

## Modeling Entity Decision

`MaterialID` is the recommended future modeling entity. The read-only profile verified that both `target` and `is_test` are consistent within each `MaterialID` for the currently downloaded D1 and D2 files.

This means future supervised modeling can safely assign the material-level target and split from the raw rows, while still preserving the underlying event/time-sample sequence.

## D1 and D2 Remain Separate

D1 and D2 remain separate datasets. They should not be merged silently because they may represent different wafer/process contexts and should retain their dataset lineage in future master tables.

## Scope Boundary

No model training occurred in M10.

No aggregation for model input occurred in M10.

No ST-AWFD modeling master table was created in M10.

Raw CSV files were not modified.

## Next Step: M11

M11 will create retrospective `MaterialID`-level master tables for D1 and D2 while preserving raw sequence data for future early-warning modeling.

That design should keep:

- `MaterialID` grouping
- `StepID` sequence information
- `duration_ms` timing information
- raw feature columns
- `target` and `is_test` lineage
- D1/D2 dataset identity

## Run

```bash
python scripts/run_profile_st_awfd.py
```

Expected outputs:

```text
reports/st_awfd_d1_profile_report.md
reports/st_awfd_d2_profile_report.md
reports/st_awfd_master_table_design_decision.md
```
