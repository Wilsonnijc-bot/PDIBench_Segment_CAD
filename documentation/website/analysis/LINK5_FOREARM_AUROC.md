# Link5 rigidity versus forearm deformation

**AUROC: 0.538** using the **v2 workbook**, forearm column **AB**, and the updated-mask link5 scores displayed on the review page. Higher rigidity predicts label 1; label 0 is the negative class. Label 0.5 is excluded.

All **45 cases** have valid scores: **22 severe**, **13 none**, and **10 moderate.** Binary AUROC uses 286 positive–negative pairs, with ties counting as half.

| Forearm labels | AUROC | Severe / none / moderate | Pairs |
| --- | ---: | ---: | ---: |
| v2 workbook | 0.538 | 22 / 13 / 10 | 286 |
| original workbook | 0.536 | 24 / 14 / 7 | 336 |

**Result:** AUROC 0.538 is close to chance (0.5): updated-mask link5 provides little severe-versus-none separation on this selected set. This is descriptive performance, not a fitted threshold or a cross-validated estimate.

**Provenance:** Scores are from [`summary.json`](summary.json), checked against every case’s `v1_cotracker3/metrics.json`. The separate `link5-joint-prompt-20260929` folder holds SAM prompt tuning trials; it is not the score source. It does not use the older four-way run’s link5 scores or the original V1 link5 scores.

The top-level workbooks differ in nine AB labels. Embedded `selection.json` labels and copied `input/` workbooks are not used as ground truth. Workbook rows are checked against dataset and video number. Reported scores are `epsilon_rigidity`, not PDI grades.

**Precision check:** Using full-precision scores embedded in the original replay HTML gives AUROC 0.542 (v2 labels), versus 0.538 using reported metrics. Reported values follow the score field used by the existing deformation-label analyses.

**Audit files:** [case-level labels and scores](LINK5_FOREARM_AUROC.csv), [exact results and input hashes](LINK5_FOREARM_AUROC.json).

Reproduce: `python3 scripts/analyze_link5_updated_mask_auroc.py --primary v2` from the repository root.
