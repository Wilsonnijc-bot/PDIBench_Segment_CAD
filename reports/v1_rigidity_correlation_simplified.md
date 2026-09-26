# Simplified Rigidity Correlation Analysis

Generated: September 6, 2026

## Scope

This analysis uses **rigidity only** to test whether the pipeline agrees with the human label `Robot deformation (0/1)`.

It does **not** use scale, trajectory, combined PDI score, or PDI grade as deformation targets.

The evaluated label cohort contains completed videos from `LVP_ROBOWM` and `COSMOS3`.

## How the robot rigidity score is formed

| Robot part | Pipeline rigidity used |
|---|---|
| Upper arm | Mean of usable `link2` and `link3` values |
| Forearm | Mean of usable `link4` and `link5` values |
| Gripper | Usable `link7` value |
| Whole robot | Equal-weight mean of upper arm, forearm, and gripper |

`link6` is excluded because the human labels do not provide a matching component label. A video receives a whole-robot score only when all three mapped parts remain measurable.

## Main comparison

| Analysis | What is accepted | Videos | Positive / negative | Pearson r | Spearman rho | AUROC | Use |
|---|---|---:|---:|---:|---:|---:|---|
| **Historical baseline** | Completed videos with finite rigidity, including the historical `1.0` fallback | 55 | 46 / 9 | 0.301 | 0.409 | **0.819** | Audit only |
| **Primary quality-valid analysis** | Baseline rules, but remove rigidity `1.0` sentinels | 45 | 36 / 9 | 0.293 | 0.372 | **0.769** | Main reported result |
| **Strict-quality sensitivity** | Remove `1.0` sentinels and require direct depth plus full SAM coverage | 44 | 35 / 9 | 0.277 | 0.326 | **0.733** | Sensitivity check |

### What the three rows mean

**55-video baseline:** This is the original completed cohort. Its AUROC is the highest, but it treats rigidity `1.0` as a real maximum-deformation score. In the historical evaluator, `1.0` can instead mean that too few reliable tracks, anchors, or 3D pairs survived. Therefore, the baseline is useful for comparison but is not the valid primary result.

**45-video primary analysis:** Ten videos are removed because at least one required mapped link has the invalid `1.0` insufficient-evidence sentinel. This is the cleanest result that can be reconstructed from the historical CSV. The primary whole-robot result is AUROC `0.769`, Pearson `r = 0.293`, and Spearman `rho = 0.372`.

**44-video strict-quality analysis:** This applies the sentinel removal and additionally requires every used link to have direct depth and SAM tracked fraction `1.0`. It removes one additional video beyond the 45-video cohort. This is only a sensitivity analysis because full SAM coverage does not prove that CoTracker supplied enough valid 3D tracks for rigidity.

## Why the sample count changes

| Transition | Videos removed | Reason |
|---|---:|---|
| Reliable completed cohort to baseline | 0 | All 55 completed videos receive a historical whole-robot score |
| Baseline to primary | 10 | A required rigidity value is the historical `1.0` insufficient-evidence sentinel |
| Primary to strict quality | 1 | The remaining video fails the direct-depth/full-SAM requirement |

The ten sentinel-affected videos are:

`COSMOS3_0003`, `COSMOS3_0014`, `COSMOS3_0024`, `COSMOS3_0042`, `COSMOS3_0048`, `COSMOS3_0050`, `COSMOS3_0051`, `COSMOS3_0052`, `COSMOS3_0055`, and `COSMOS3_0056`.

The additional strict-quality exclusion is `LVP_ROBOWM_0042`.

## Component-level result

| Component | Primary n | Pearson r | Spearman rho | AUROC | Interpretation |
|---|---:|---:|---:|---:|---|
| Upper arm | 55 | 0.175 | 0.036 | 0.536 | Near chance |
| Forearm | 55 | -0.074 | -0.057 | 0.467 | Near chance |
| Gripper | 45 | 0.345 | 0.410 | 0.756 | Main source of useful association |

Component sample sizes can be larger than the whole-robot sample because a component can still be evaluated when another component is unavailable. The whole-robot score requires upper arm, forearm, and gripper together.

## Interpretation

1. The **55-video baseline should not be presented as the main result**, despite its AUROC of `0.819`, because invalid `1.0` fallback values can artificially help separate positive and negative labels.
2. The defensible primary result is the **45-video sentinel-excluded analysis**: AUROC `0.769`, with weak-to-moderate positive Pearson and Spearman associations.
3. The **44-video strict-quality result** remains directionally similar but is weaker, with AUROC `0.733`.
4. The whole-robot result is driven mainly by the gripper. The current data do not support claiming reliable deformation detection for the upper arm or forearm.
5. The cohorts are not identical, so the AUROC decrease from `0.819` to `0.769` to `0.733` should not be interpreted as a controlled performance loss caused by each filter.
6. Each whole-robot cohort contains only nine negative examples. The estimates should therefore be treated as preliminary rather than precise benchmark performance.

## Short reporting version

> Using rigidity alone, the historical 55-video cohort produced AUROC 0.819, but this baseline included `1.0` values that represent insufficient tracking evidence rather than measured deformation. After excluding those sentinels, the primary 45-video cohort produced AUROC 0.769, Pearson r = 0.293, and Spearman rho = 0.372. A stricter 44-video sensitivity analysis requiring direct depth and full SAM coverage produced AUROC 0.733. Most of the association came from gripper rigidity; upper-arm and forearm results were near chance.

## Source files

- Machine-readable statistics: [`data/pdi_output_analysis.json`](data/pdi_output_analysis.json)
- Full technical diagnosis: [`pdi_pipeline_diagnosis.md`](pdi_pipeline_diagnosis.md)
- Upper-arm and forearm diagnosis: [`upperarm_forearm_near_chance_diagnosis.md`](upperarm_forearm_near_chance_diagnosis.md)
- Analysis implementation: [`../scripts/analyze_pdi_outputs.py`](../scripts/analyze_pdi_outputs.py)
