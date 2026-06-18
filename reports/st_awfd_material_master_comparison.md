# ST-AWFD Material Master Comparison

D1 and D2 remain separate datasets. This comparison report summarizes the independently built retrospective MaterialID-level master tables.

| dataset_id | materials | source_records | numeric_features | observed_step_count | split_counts | label_distribution |
| --- | --- | --- | --- | --- | --- | --- |
| st_awfd_d1 | 5104 | 602108 | 15 | 7 | {'test': 1807, 'train': 3297} | {'0': 5102, '1': 2} |
| st_awfd_d2 | 1156 | 126794 | 20 | 2 | {'test': 604, 'train': 552} | {'0': 789, '1': 367} |

No model training, imputation, or SECOM merge was performed.
