# Rigidity-Only Robot-Deformation Diagnosis

Generated: 2026-09-06

## Decision

This study validates robot deformation with `rigidity_component` only.

Scale, trajectory, the combined PDI score, and PDI grades remain pipeline outputs, but they are not used as robot-deformation targets in this report. Scale and trajectory measure different geometric consistency properties; scale is also strongly contaminated by missing-mask and interpolation artifacts in this run.

Primary historical validity rule: exclude failed links and `rigidity_component == 1.0`. In this export, `1.0` is the insufficient-evidence fallback, not a measured maximum deformation. Future runs raise an insufficient-evidence error and mark the link unavailable instead.

## Inputs and label contract

- Pipeline metrics: `outputs/metrics.csv`
- Human labels: `/Users/nijiachen/Downloads/deformationhumanlabel.xlsx`
- Human target: `Robot deformation (0/1)`
- Upper arm: mean valid rigidity from links 2+3
- Forearm: mean valid rigidity from links 4+5
- Gripper: valid rigidity from link 7
- Whole robot: equal-weight mean of upper arm, forearm, and gripper
- Link 6: excluded because no supplied component label maps to it
- Blank component cells: normalize to no deformation while preserving raw blank provenance
- Primary cohort: completed labeled LVP_ROBOWM and COSMOS3 videos

Regenerate the frozen analysis with:

```bash
python3 scripts/analyze_pdi_outputs.py \
  --labels /Users/nijiachen/Downloads/deformationhumanlabel.xlsx
```

Machine-readable results are stored in `reports/data/pdi_output_analysis.json`.

## Pipeline health

1. **Whole-video SAM3 failure:** 10 of 197 videos fail before any link score is produced. Nine are LVP_ROBOWM and one is COSMOS2.5. The local bundle retains only inaccessible remote `sam3.log` paths.
2. **Per-link failure:** among 187 completed videos, 59 of 1,122 possible link measurements are unavailable. Link 4 accounts for 51 failures.
3. **Invalid successful values:** 39 scored links have rigidity `1.0` from the insufficient-evidence fallback. Link 7 accounts for 34.
4. **Missing historical CoTracker provenance:** the CSV has SAM coverage but not requested and retained CoTracker counts, so exact track-count ablations cannot be reconstructed.

| Dataset | Videos | Complete | Failed |
|---|---:|---:|---:|
| COSMOS2.5 | 61 | 60 | 1 |
| COSMOS3 | 68 | 68 | 0 |
| LVP_ROBOWM | 68 | 59 | 9 |
| **Total** | **197** | **187** | **10** |

| Link | Unavailable | Rigidity `1.0` sentinel |
|---|---:|---:|
| link2 | 0 | 0 |
| link3 | 5 | 0 |
| link4 | 51 | 2 |
| link5 | 1 | 0 |
| link6 | 1 | 3 |
| link7 | 1 | 34 |

## Rigidity results

| Condition | Role | n | Pearson r | Spearman rho | AUROC |
|---|---|---:|---:|---:|---:|
| Historical rigidity, including `1.0` | Audit only | 55 | 0.301 | 0.409 | 0.819 |
| Exclude rigidity `1.0` | **Primary** | 45 | 0.293 | 0.372 | 0.769 |
| Direct depth + full SAM | Sensitivity | 54 | 0.300 | 0.378 | 0.793 |
| Direct + full SAM + no sentinel | Sensitivity | 44 | 0.277 | 0.326 | 0.733 |

The historical baseline has the highest AUROC, but it includes invalid `1.0` sentinels and is not the reported primary result. The primary sentinel-excluded AUROC is 0.769.

Direct-depth and full-SAM filtering is not the primary rigidity rule. Later SAM mask coverage and target-depth interpolation strongly affect scale, but do not directly prove whether CoTracker supplied enough valid 3D rigidity evidence.

## Component results

| Component | Primary n | Pearson r | Spearman rho | AUROC |
|---|---:|---:|---:|---:|
| Upper arm, links 2+3 | 55 | 0.175 | 0.036 | 0.536 |
| Forearm, links 4+5 | 55 | -0.074 | -0.057 | 0.467 |
| Gripper, link 7 | 45 | 0.345 | 0.410 | 0.756 |

The whole-robot association is driven mainly by gripper rigidity. Upper-arm and forearm rigidity remain close to chance and must not be described as reliable deformation detection.

## Root cause and repair

`audit_3d_rigidity_cv()` historically returns `(1.0, all_ones)` when fewer than five reliable anchors or fewer than three non-degenerate 3D pairs remain. The caller records Strategy 1 as successful, converting missing evidence into a strong deformation signal.

The evaluator repair now:

- raises `InsufficientRigidityEvidenceError` in the active multi-object path;
- marks links unavailable when fewer than five CoTracker tracks survive;
- records insufficient anchors or 3D pairs as failed links;
- exports requested and retained CoTracker counts, retained fraction, mean visibility, and SAM mask coverage for future runs;
- preserves legacy maximum-sentinel behavior only as an explicit compatibility option.

## Representative cases

- `LVP_ROBOWM_0003`: negative control with the highest valid whole-robot rigidity among reliable labeled negative replay cases, 0.035.
- `COSMOS3_0002`: fully labeled negative control with whole-robot rigidity 0.023.
- `LVP_ROBOWM_0000`: human forearm deformation, but valid whole-robot rigidity is 0.009 and forearm rigidity is 0.014.
- `COSMOS3_0006`: human forearm and gripper deformation, but valid whole-robot rigidity is 0.027; link 4 is unavailable.
- `COSMOS3_0003`: human forearm and gripper deformation, but link-7 rigidity is the invalid `1.0` sentinel. Sentinel-excluded whole-robot rigidity is unavailable because no valid gripper measurement remains.
- `COSMOS2.5_0002`: label-provenance audit. The structured forearm label is blank while the prose states that the forearm becomes shorter. It is not used as clean validation evidence.

Replay videos and posters are diagnostic reconstructions, not exact model inputs. This run does not contain saved input hashes for pixel-exact validation.
