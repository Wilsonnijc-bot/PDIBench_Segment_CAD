# AnomalyDINO correlation with human object deformation

**The ten-frame anomaly sums show a moderate positive association with human object deformation: Pearson r = 0.452; Spearman ρ = 0.455.** Higher anomaly scores tend to accompany deformation labels, with substantial overlap between the groups.

## Main result

| Sample | Videos | Pearson r | p | Spearman ρ | p |
|---|---:|---:|---:|---:|---:|
| Revised labels, including 0.5 | 35 | 0.452 | 0.0065 | 0.455 | 0.0061 |
| Binary labels only, excluding 0.5 | 34 | 0.458 | 0.0064 | 0.471 | 0.0049 |

Binary AUROC is **0.791** across 11 deformation and 23 non-deformation videos. No score threshold was fitted.

| Human label | Videos | Mean anomaly sum | Median anomaly sum |
|---|---:|---:|---:|
| 0 | 23 | 4.181 | 4.208 |
| 0.5 | 1 | 4.097 | 4.097 |
| 1 | 11 | 5.007 | 4.851 |

## Five versus ten frames

| Selection | Videos | Pearson r | Spearman ρ |
|---|---:|---:|---:|
| Previous five frames | 35 | 0.451 | 0.451 |
| Current ten frames | 35 | 0.452 | 0.455 |

Doubling the samples barely changes the observed association: Δr = +0.001 and Δρ = +0.004. Both runs use the same 35 videos and labels. Because every included video has the same number of pairs within a run, using the mean instead of the sum gives identical correlations. The difference between runs is descriptive; it has not been tested as an improvement.

## By generator

| Generator | Videos | Spearman ρ | p |
|---|---:|---:|---:|
| COSMOS2.5 | 10 | 0.418 | 0.2295 |
| COSMOS3 | 14 | 0.494 | 0.0725 |
| LVP_ROBOWM | 11 | 0.478 | 0.1369 |

## Largest disagreements

The following are high scores among label-0 videos and low scores among label-1 videos. They are ranking disagreements, not threshold-based classification errors.

| Video | Human label | Anomaly sum |
|---|---:|---:|
| COSMOS2.5_0030 | 0 | 5.876 |
| COSMOS2.5_0015 | 0 | 5.273 |
| COSMOS3_0005 | 0 | 4.908 |
| LVP_ROBOWM_0040 | 1 | 3.575 |
| COSMOS3_0015 | 1 | 4.311 |
| COSMOS3_0046 | 1 | 4.508 |

## Scope and interpretation

- **Label source:** `selected_45_matched_videos_styledv2.xlsx`, first sheet, revised `Object deformation (0/1)2` column AF, rows 2–46. Match by generator and numeric video ID. The original column Q is not used.
- **Score source:** the completed ten-frame AnomalyDINO run. Sum the ten unrounded official pair scores per video. Quotas are 0/2/2/2/4 with mandatory usable post-occlusion successors.
- **Coverage:** 35 of 45 videos. The existing six exclusions and four unavailable references have no score and are omitted, never assigned zero.
- **Ambiguous label:** the single 0.5 value is retained as an ordinal middle value in the main correlation and omitted from the binary check.
- **Statistics:** two-sided Pearson and asymptotic Spearman p-values. These are nominal, without adjustment for the several descriptive comparisons. Videos share matched tasks across generators, so the usual independent-observation p-values may overstate precision.
- **Meaning:** this measures association with video-level deformation labels. AnomalyDINO also responds to appearance, viewpoint, segmentation, and crop-mapping differences. The result does not establish deformation-specific detection or generalization beyond these selected videos.

[Matched scores and workbook rows](matched_videos.csv) · [Exact statistics and source hashes](statistics.json) · [Crop gallery](../../object-reference-crops-20261001/selection_gallery.html)
