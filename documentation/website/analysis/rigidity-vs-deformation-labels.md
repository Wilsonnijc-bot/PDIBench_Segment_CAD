# Rigidity scores versus deformation labels

This analysis matches the 45 cases in [selection.json](selection.json) to the `Selected 45` sheet of [selected_45_matched_videos_styledv2.xlsx](../../selected_45_matched_videos_styledv2.xlsx) by dataset, video number, and source row. Forearm uses column **AB** and link5; gripper uses column **AC** and link7. The score is `epsilon_rigidity` in [summary.json](summary.json). Higher scores indicate greater variation in tracked 3D point-pair distances, hence greater deviation from rigid motion. Compared with the previous top-level workbook, v2 changes nine AB labels and leaves AC unchanged.

## Correlations

Pearson *r* measures linear association. Spearman *ρ* measures rank association and accommodates the ordered labels 0, 0.5, and 1. Both use all rows with a valid score and label. Failed scores are excluded, not set to zero.

| Score versus workbook label | Valid videos | Pearson *r* | Spearman *ρ* | Dataset-adjusted Pearson *r* |
| --- | ---: | ---: | ---: | ---: |
| Link5 versus forearm AB | 45 | -0.019 | 0.285 | -0.055 |
| Link7 v1 / CoTracker3 versus gripper AC | 38 | 0.457 | 0.555 | 0.270 |
| Link7 v1 / TAPIP3D versus gripper AC | 45 | 0.057 | 0.277 | -0.224 |
| Link7 v2 / CoTracker3 versus gripper AC | 40 | 0.390 | 0.431 | 0.366 |
| Link7 v2 / TAPIP3D versus gripper AC | 45 | 0.411 | 0.552 | 0.340 |

The dataset-adjusted Pearson value correlates scores and labels after subtracting their respective means within each of LVP, Cosmos2.5, and Cosmos3. It is a descriptive check because the label mix differs considerably across datasets.

## AUROC

Label **1** is the positive class (deformation) and label **0** is the negative class. Label-0.5 videos are excluded: 10 for forearm and seven for gripper. AUROC is the fraction of positive–negative video pairs in which the positive video has a higher score, with score ties counting as half.

| Score versus workbook label | AUROC | Label 1 videos | Label 0 videos | Valid pairs |
| --- | ---: | ---: | ---: | ---: |
| Link5 versus forearm AB | 0.670 | 22 | 13 | 286 |
| Link7 v1 / CoTracker3 versus gripper AC | 0.845 | 17 | 14 | 238 |
| Link7 v1 / TAPIP3D versus gripper AC | 0.711 | 24 | 14 | 336 |
| Link7 v2 / CoTracker3 versus gripper AC | 0.756 | 19 | 14 | 266 |
| Link7 v2 / TAPIP3D versus gripper AC | 0.842 | 24 | 14 | 336 |

## Including the 0.5 labels

Ordered-label concordance compares every pair with different labels (0 versus 0.5, 0 versus 1, and 0.5 versus 1), counting score ties as half. It includes all three label levels but is not a standard binary AUROC. Two binary AUROCs are also possible by grouping 0.5 with either class. The table shows all three measures for each score path.

| Score | Score treatment | Valid videos | Ordered concordance | AUROC: 0.5 or 1 versus 0 | AUROC: 1 versus 0 or 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Link5 | Measured | 45 | 0.642 | 0.620 | 0.675 |
| Link7 v1 / CoTracker3 | Measured | 38 | 0.777 | 0.799 | 0.801 |
| Link7 v1 / CoTracker3 | Missing highest | 45 | 0.831 | 0.844 | 0.859 |
| Link7 v1 / TAPIP3D | Measured | 45 | 0.638 | 0.726 | 0.611 |
| Link7 v2 / CoTracker3 | Measured | 40 | 0.716 | 0.684 | 0.772 |
| Link7 v2 / CoTracker3 | Missing highest | 45 | 0.766 | 0.735 | 0.819 |
| Link7 v2 / TAPIP3D | Measured | 45 | 0.781 | 0.786 | 0.817 |

“Missing highest” assigns each unavailable score a value above every measured score for that path. All seven unavailable v1 / CoTracker3 scores and all five unavailable v2 / CoTracker3 scores have gripper label 1. These rows are sensitivity scenarios, not measured results. For v1 / CoTracker3, the highest-score assumption gives a 0-versus-1 AUROC of **299/336 = 0.890** and ordered concordance of **500.5/602 = 0.831**. The measured-only ordered concordance is **353.5/455 = 0.777**.

Link5 has almost no linear association with the forearm labels (Pearson -0.019), although its rank correlation is 0.285 and its AUROC is 0.670. Its dataset-adjusted Pearson correlation is -0.055. Link7 v2 / TAPIP3D has a positive association with the gripper labels and scores all 45 videos. Link7 v1 / CoTracker3 has a marginally higher observed AUROC, but seven label-1 videos have no valid score. Five label-1 videos lack a v2 / CoTracker3 score. All missing CoTracker3 scores failed with `insufficient_cotracker_tracks`, so the observed AUROCs for those paths do not represent the full selected set.

The workbook copied into this run's `input/` directory is **not identical** to the v2 workbook: AB differs in 23 rows and AC in 26 rows. The figures here use v2. These associations describe this selected set and do not establish general detection performance.
