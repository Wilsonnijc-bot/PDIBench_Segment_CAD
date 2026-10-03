# Object crop anomaly results after seven-case VLM3 refresh

Binary AUROC: **0.780** across 11 deformed and 26 non-deformed videos. One moderate label is omitted from binary analysis when scored. No score threshold was fitted.

38 complete ten-pair videos enter the correlation; 7 have no score; 0 have partial selections and are omitted from the ten-pair comparison. Missing values are never set to zero.

| Statistic | Coefficient | Two-sided p |
|---|---:|---:|
| Pearson | 0.4248 | 0.00786 |
| Spearman | 0.4252 | 0.00779 |

## Refreshed cases

All other crop selections and pair scores retain their original bytes. Existing exclusions remain. The refreshed cases use exact VLM3 seed frames, accepted SAM3 masks, synchronized occlusion evidence, and available-pixel crop selection with the original 0/2/2/2/4 quotas and the selective late-area gate.

| Case | Status | Scored pairs | Anomaly sum |
|---|---|---:|---:|
| COSMOS2.5_0056 | scored | 10 | 5.378403 |
| COSMOS2.5_0005 | excluded_by_user | 0 | — |
| COSMOS2.5_0010 | excluded_by_user | 0 | — |
| LVP_ROBOWM_0010 | scored | 10 | 5.343803 |
| LVP_ROBOWM_0015 | scored | 10 | 3.940145 |
| LVP_ROBOWM_0060 | scored | 10 | 5.185664 |
| COSMOS2.5_0065 | excluded_by_user | 0 | — |

Labels: revised workbook column AF, matched by generator and video number. AnomalyDINO can respond to appearance, viewpoint, segmentation and mapping differences as well as deformation. These results describe the selected dataset; the correlations do not establish deformation-specific causation or out-of-sample accuracy.

[Exact statistics](OBJECT_ANOMALY_STATISTICS.json) · [Updated crop gallery](../objects/selection_gallery.html)
