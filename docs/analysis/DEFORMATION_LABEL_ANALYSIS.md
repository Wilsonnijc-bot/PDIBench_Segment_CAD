# Robot-part deformation labels versus V1 rigidity scores

Workbook: [`selected_45_matched_videos_styled.xlsx`](../../selected_45_matched_videos_styled.xlsx), sheet `Selected 45`. Cases are joined using the workbook row stored in `selection.json`, then checked against generator and matched video number. Forearm uses column **AB** and **link5**; upper arm uses column **AD** and **link2**; gripper uses column **AC** and **link7**. The 0, 0.5, and 1 values are treated as the workbook's ordered deformation labels.

The last three workbook columns (AH:AJ) are used **only** to flag rows containing `3`. Their other values do not enter any score calculation. A `3` marks a case for extra attention because it has relatively large deformation. The labels and the `3` flags remain separate; no label is overwritten.

Scores come from `v1/metrics.json`, exact-group mode, field `breakdown.epsilon_rigidity`. This rigidity score measures the spread of tracked 3D point-pair distance ratios. Higher values indicate less consistent relative distances. Each table cell shows **median (number of scored videos)**. Missing labels and unscored links are excluded rather than set to zero.

## Forearm: AB label versus link5 score

### Rigidity score (epsilon_rigidity)

| Label 0 | Label 0.5 | Label 1 |
| ---: | ---: | ---: |
| 0.0183 (14) | 0.0145 (7) | 0.0152 (23) |

Workbook forearm labels absent or nonnumeric: none.

## Upper arm: AD label versus link2 score

### Rigidity score (epsilon_rigidity)

| Label 0 | Label 0.5 | Label 1 |
| ---: | ---: | ---: |
| 0.0146 (23) | 0.0152 (8) | 0.0331 (11) |

Workbook upper arm labels absent or nonnumeric: LVP_ROBOWM_0030 (row 20), LVP_ROBOWM_0065 (row 44).

## Gripper: AC label versus link7 score

### Rigidity score (epsilon_rigidity)

| Label 0 | Label 0.5 | Label 1 |
| ---: | ---: | ---: |
| 0.0296 (14) | 0.0363 (7) | 0.0723 (16) |

Workbook gripper labels absent or nonnumeric: none.

## Score association with deformation severity

The workbook defines 0 as almost no deformation, 0.5 as moderate, and 1 as severe. **AUROC** compares labels 1 and 0 only: it is the fraction of severe/none video pairs in which the severe video has the higher rigidity score, counting ties as half. **Ordered-label concordance** also includes moderate cases and compares every pair with different labels. A value of 0.5 means no ordering.

| Mapping | Videos by label: 0 / 0.5 / 1 | AUROC (severe vs none) | Ordered concordance |
| --- | ---: | ---: | ---: |
| Upper arm → link2 | 23 / 8 / 11 | 0.941 (253 pairs) | 0.810 (525 pairs) |
| Forearm → link5 | 14 / 7 / 23 | 0.467 (322 pairs) | 0.484 (581 pairs) |
| Gripper → link7 | 14 / 7 / 16 | 0.848 (224 pairs) | 0.774 (434 pairs) |
| Gripper → link7 (cap 0.20) | 14 / 7 / 16 | 0.848 (224 pairs) | 0.774 (434 pairs) |
| Gripper → link7 (cap 0.20; missing = 0.15) | 14 / 7 / 24 | 0.899 (336 pairs) | 0.837 (602 pairs) |
| Gripper → link7 (cap 0.20; missing = 0.20) | 14 / 7 / 24 | 0.899 (336 pairs) | 0.837 (602 pairs) |

The calculation uses 42 videos for upper arm/link2 (two nonnumeric labels and one failed V1 case) and 44 for forearm/link5 (one failed case). Gripper/link7 has 37 scored videos; unscored link7 results and the failed case are excluded. These values describe score ordering in the selected set.

## Link7 cap at 20%

The capped analysis uses `min(link7 rigidity, 0.20)`; it does not change any stored score. The cap changes 4 of 37 scored videos, all labeled severe. Their original scores are COSMOS3_0001 0.2621, COSMOS2.5_0010 0.3297, COSMOS3_0010 0.4989, COSMOS3_0060 0.2229.

Capping lowers the severe-label mean from 0.1250 to 0.0929. The severe-label median stays at 0.0723. AUROC and ordered-label concordance stay the same because the capped scores remain above the lower-label scores.

**Coverage limit:** 8 severe-label gripper videos lack a link7 score, including all 5 cases marked `3` in the gripper column (0 scored). They do not enter either observed-only calculation; the cap cannot recover their missing measurements.

## Hypothetical scores for unavailable link7 cases

As a sensitivity check, assign every unavailable link7 case a score of 0.15 or 0.20. Apply the 0.20 cap to measured scores in both scenarios. This includes all 45 videos and does not alter the stored experiment outputs. The two assumed values give the same AUROC (0.899) and ordered-label concordance (0.837) because both values exceed every measured label-0 and label-0.5 link7 score. The higher assumption changes the score magnitude, though: the severe-label mean is 0.1119 at 0.15 versus 0.1286 at 0.20. These are assumed scores, not measured rigidity.

## Cases marked `3` for extra attention

The `3` flags identify rows for closer review. Their ordinary labels and corresponding link scores are shown below; a failed V1 case has no score.

| Video | Workbook row | `3` in | Forearm AB | Link5 rigidity | Upper arm AD | Link2 rigidity | Gripper AC | Link7 rigidity |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [COSMOS2.5_0005](cases/COSMOS2.5_0005/v1/metrics.json) | 6 | Upper arm, Gripper | 1 | 0.0215 | 1 | 0.0331 | 1 | — |
| COSMOS2.5_0054 | 36 | Forearm, Upper arm, Gripper | 1 | — | 1 | — | 1 | — |
| [COSMOS3_0054](cases/COSMOS3_0054/v1/metrics.json) | 37 | Gripper | 1 | 0.0105 | 0 | 0.0132 | 1 | — |
| [COSMOS2.5_0056](cases/COSMOS2.5_0056/v1/metrics.json) | 39 | Forearm, Upper arm, Gripper | 1 | 0.0143 | 1 | 0.0380 | 1 | — |
| [COSMOS2.5_0060](cases/COSMOS2.5_0060/v1/metrics.json) | 42 | Forearm, Upper arm, Gripper | 1 | 0.0120 | 1 | 0.0277 | 1 | — |

## Reading the comparison

- 44 of 45 V1 cases are complete; link2 and link5 are scored in every complete case. The failed case `COSMOS2.5_0054` has workbook labels and `3` flags but no V1 score.
- Forearm/link5 rigidity does not separate the severity labels: AUROC is 0.467 and ordered-label concordance is 0.484. Its medians are 0.0183, 0.0145, and 0.0152 for labels 0, 0.5, and 1.
- Upper-arm/link2 pooled rigidity medians rise from 0.0146 to 0.0152 to 0.0331 for labels 0, 0.5, and 1, matching its higher AUROC.
- Gripper/link7 rigidity ranks severe cases above almost-no-deformation cases with AUROC 0.848. Capping values above 0.20 changes the mean but not the rank metrics.
- The `3`-flagged cases do not all have unusually high rigidity scores. For example, `COSMOS2.5_0056` and `COSMOS2.5_0060` have link5 scores 0.0143 and 0.0120. `COSMOS3_0054` has a `3` in the gripper column only, and `COSMOS2.5_0054` has no V1 score.
- These are descriptive associations for 45 selected videos. They do not establish whether the score detects the annotated deformation reliably.
