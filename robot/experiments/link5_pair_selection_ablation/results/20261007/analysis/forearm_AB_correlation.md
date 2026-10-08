# Rigidity correlation with AB forearm labels

40 scored videos match the unchanged workbook. COSMOS2.5_0018 has no label.

| AB policy | Group | Method | n | Pearson r | Spearman rho | AUROC |
|---|---|---|---:|---:|---:|---:|
| exact_0_1 | all_labeled | balanced_v0 | 32 | 0.279767 | 0.349556 | 0.708333 |
| exact_0_1 | all_labeled | refine_v1 | 32 | 0.458995 | 0.468406 | 0.779167 |
| exact_0_1 | LVP_ROBOWM | balanced_v0 | 10 | 0.256175 | 0.261116 | 0.687500 |
| exact_0_1 | LVP_ROBOWM | refine_v1 | 10 | 0.075777 | 0.174078 | 0.625000 |
| exact_0_1 | COSMOS2.5 | balanced_v0 | 12 | undefined | undefined | undefined |
| exact_0_1 | COSMOS2.5 | refine_v1 | 12 | undefined | undefined | undefined |
| exact_0_1 | COSMOS3 | balanced_v0 | 10 | 0.086996 | 0.355335 | 0.708333 |
| exact_0_1 | COSMOS3 | refine_v1 | 10 | 0.319264 | 0.426401 | 0.750000 |
| half_as_1 | all_labeled | balanced_v0 | 40 | 0.285531 | 0.345000 | 0.717262 |
| half_as_1 | all_labeled | refine_v1 | 40 | 0.365831 | 0.373357 | 0.735119 |
| half_as_1 | LVP_ROBOWM | balanced_v0 | 13 | 0.440347 | 0.422577 | 0.750000 |
| half_as_1 | LVP_ROBOWM | refine_v1 | 13 | 0.049432 | 0.126773 | 0.575000 |
| half_as_1 | COSMOS2.5 | balanced_v0 | 14 | undefined | undefined | undefined |
| half_as_1 | COSMOS2.5 | refine_v1 | 14 | undefined | undefined | undefined |
| half_as_1 | COSMOS3 | balanced_v0 | 13 | 0.036491 | 0.311805 | 0.694444 |
| half_as_1 | COSMOS3 | refine_v1 | 13 | 0.163852 | 0.267261 | 0.666667 |
| half_as_0 | all_labeled | balanced_v0 | 40 | 0.180747 | 0.168928 | 0.597500 |
| half_as_0 | all_labeled | refine_v1 | 40 | 0.427757 | 0.428817 | 0.747500 |
| half_as_0 | LVP_ROBOWM | balanced_v0 | 13 | 0.125581 | 0.113961 | 0.590909 |
| half_as_0 | LVP_ROBOWM | refine_v1 | 13 | 0.073493 | 0.113961 | 0.590909 |
| half_as_0 | COSMOS2.5 | balanced_v0 | 14 | -0.115673 | -0.202548 | 0.333333 |
| half_as_0 | COSMOS2.5 | refine_v1 | 14 | -0.135391 | -0.202548 | 0.333333 |
| half_as_0 | COSMOS3 | balanced_v0 | 13 | 0.128072 | 0.288675 | 0.666667 |
| half_as_0 | COSMOS3 | refine_v1 | 13 | 0.423715 | 0.412393 | 0.738095 |
| recorded_0_half_1 | all_labeled | balanced_v0 | 40 | 0.253757 | 0.261125 | undefined |
| recorded_0_half_1 | all_labeled | refine_v1 | 40 | 0.437637 | 0.445304 | undefined |
| recorded_0_half_1 | LVP_ROBOWM | balanced_v0 | 13 | 0.348004 | 0.379284 | undefined |
| recorded_0_half_1 | LVP_ROBOWM | refine_v1 | 13 | 0.067800 | 0.135910 | undefined |
| recorded_0_half_1 | COSMOS2.5 | balanced_v0 | 14 | -0.115673 | -0.202548 | undefined |
| recorded_0_half_1 | COSMOS2.5 | refine_v1 | 14 | -0.135391 | -0.202548 | undefined |
| recorded_0_half_1 | COSMOS3 | balanced_v0 | 13 | 0.093448 | 0.330759 | undefined |
| recorded_0_half_1 | COSMOS3 | refine_v1 | 13 | 0.332214 | 0.389823 | undefined |

AB only; no labels used in pair selection or score construction.
Additional COSMOS2.5_0018 has no workbook row and is excluded from correlation.
Spearman uses average ranks for ties. Binary Pearson is point-biserial correlation.
AUROC uses higher rigidity as the positive score, average ranks for ties, and is undefined for the ordinal policy.
Descriptive correlations on selected videos; matched tasks across generators are not independent observations.

[Exact results](forearm_AB_correlation.json) · [Labels and scores](forearm_AB_labels_scores.csv)
