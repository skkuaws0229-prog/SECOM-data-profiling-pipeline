# ST-AWFD st_awfd_d2 Material Master Report

M11 creates a retrospective MaterialID-level master table. It does not train models, impute values, modify raw CSV files, or merge ST-AWFD with SECOM.

## Summary

- Raw CSV path: C:\work\manufacturing-ai-agent-platform\data\raw\st_awfd\wafer_d2\D2.csv
- Source record count: 126794
- Material master row count: 1156
- Numeric feature count: 20
- Observed StepIDs: [1, 2]
- Official split counts: {'test': 604, 'train': 552}
- Label distribution: {'0': 789, '1': 367}

## Lineage

- Feature set version: st_awfd_material_retrospective_v1
- Preprocessing version: m11_material_aggregation_v1
- Aggregation strategy: retrospective_material_level_summary
- Sequence manifest path: C:\work\manufacturing-ai-agent-platform\data\processed\st_awfd_d2\sequence_manifest.csv

## Sequence Preservation

The sequence manifest keeps one row per MaterialID with observed StepIDs, duration range, raw event file path, and a sequence availability flag. Raw event-level files remain the source for future sequence-aware and early-warning modeling.
