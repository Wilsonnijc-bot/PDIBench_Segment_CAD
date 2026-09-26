# Why Upper-Arm and Forearm Rigidity Are Near Chance

Generated: September 6, 2026

## Executive diagnosis

The upper-arm and forearm results are near chance for different reasons.

- **Upper arm:** the evaluation has only five positive examples, the positive and negative score distributions almost completely overlap, and one high-scoring positive drives most of the Pearson association. The metric is also poorly aligned with labels such as arm bending or uniform shape change.
- **Forearm:** statistical power is adequate, but the measurement is internally inconsistent. `link4` is frequently unavailable, `link4` and `link5` scores do not agree, `link5` reverses direction in `LVP_ROBOWM`, and some labels conflict with their accompanying descriptions.
- **Both components:** the current formula is mathematically insensitive to rigid articulation and uniform link scaling, suppresses localized deformation with a robust median, and dilutes brief deformation by averaging over the whole video.

The evidence does not support a single threshold or aggregation adjustment as the fix. The main problem is a mismatch between what the human labels call deformation and what per-link rigidity actually measures, compounded by link-specific data-quality failures.

## Observed results

These are the primary component results from the 55 completed `LVP_ROBOWM` and `COSMOS3` videos. Rigidity `1.0` sentinel filtering does not change upper-arm or forearm results in this cohort.

| Component | n | Positive / negative | Positive median | Negative median | Pearson r | Spearman rho | AUROC | Approx. AUROC 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Upper arm | 55 | 5 / 50 | 0.01375 | 0.01258 | 0.175 | 0.036 | 0.536 | 0.232-0.816 |
| Forearm | 55 | 28 / 27 | 0.02153 | 0.02340 | -0.074 | -0.057 | 0.467 | 0.311-0.624 |

The confidence intervals are stratified bootstrap intervals with 10,000 resamples. Neither component shows statistically reliable separation:

| Component | Pearson p | Spearman p | Mann-Whitney p |
|---|---:|---:|---:|
| Upper arm | 0.202 | 0.795 | 0.809 |
| Forearm | 0.590 | 0.678 | 0.680 |

For the forearm, the direction is slightly inverted: the median rigidity is higher in human-labeled negatives than positives.

## 1. The metric and labels do not measure the same deformation concept

### What rigidity measures

For each link, the evaluator selects up to 30 frame-0 3D anchor pairs. At each later frame it computes:

```text
ratio_ij(t) = distance_ij(t) / distance_ij(0)
rigidity(t) = MAD(ratio_ij(t)) / median(ratio_ij(t))
final rigidity = mean(rigidity(t), t = 1 ... T-1)
```

This measures **disagreement among pairwise scale ratios inside one link**. It does not directly measure absolute length change, joint-angle abnormality, or deviation from the Franka CAD model.

### Confirmed blind spots

1. **Uniform shortening or lengthening is invisible.** If every pair distance changes by the same factor, all ratios remain equal and the MAD is zero. This directly conflicts with labels such as `COSMOS3_0006` (forearm “significantly lengthen”) and `COSMOS3_0060` (forearm “deformed/shortened”).
2. **Abnormal articulation can remain perfectly rigid per link.** The pipeline never forms rigidity pairs across link identities. If the arm bends unnaturally at a joint while each link remains internally rigid, upper-arm and forearm rigidity remain low by design.
3. **Localized deformation is downweighted.** The median absolute deviation is robust to a minority of changed pairs. A distorted endpoint, joint region, or local thickness change may affect too few selected pairs to move the median.
4. **Brief or late deformation is temporally diluted.** The final score is the mean over all post-reference frames. Many labels identify deformation beginning at second 1, 2, or later, so a short event can be averaged into a small scalar.
5. **Occluded moments can be carried forward instead of measured.** When fewer than three selected pairs are visible, the evaluator copies the previous frame’s score. If deformation occurs during occlusion, the event may contribute no new rigidity evidence.

This semantic mismatch is the most important shared root cause.

## 2. Upper arm: too few positives and unstable evidence

### Severe positive-class scarcity

Only 5 of 55 videos are labeled with upper-arm deformation. AUROC is prevalence-independent in definition, but its estimate here is based on only `5 x 50 = 250` positive-negative comparisons. Each positive video has disproportionate influence.

The approximate AUROC interval, `0.232-0.816`, includes both poor and potentially useful performance. The correct conclusion is not that upper-arm rigidity is proven useless; it is that this cohort cannot estimate its performance precisely.

### Pearson is driven by one outlier

Four of the five positive scores lie between `0.00525` and `0.01410`, inside the main negative range. The remaining positive, `LVP_ROBOWM_0031`, scores `0.04880` because `link3` reaches `0.0770`.

This produces Pearson `r = 0.175` while rank-based Spearman is only `0.036`. A leave-one-positive-out analysis moves upper-arm AUROC from `0.420` to `0.670`, depending on which single positive is removed. Dropping `LVP_ROBOWM_0031` lowers AUROC to `0.420`.

### Neither upper-arm link separates the labels

| Score | n | Pearson r | Spearman rho | AUROC |
|---|---:|---:|---:|---:|
| `link2` | 55 | -0.062 | -0.034 | 0.466 |
| `link3` | 54 | 0.295 | 0.080 | 0.580 |
| Mean of `link2` and `link3` | 55 | 0.175 | 0.036 | 0.536 |

`link3` has a positive Pearson value because of score magnitude, but its rank association remains weak. Taking the maximum instead of the mean does not help: exploratory upper-arm AUROC is `0.500` with max aggregation.

### Label provenance is weak for such a small positive set

Nine upper-arm component cells are blank and normalized to negative. Removing those blanks changes AUROC only from `0.536` to `0.551`, so blank handling is not the primary cause.

The larger issue is anatomical ambiguity. For example, `LVP_ROBOWM_0031` is upper-arm positive and forearm negative, even though its note explicitly says “minor forearm deformation.” Descriptions such as “robot arm bends unnaturally” also do not specify whether the error is within a rigid link or at a joint.

With only five positives, one ambiguous anatomical assignment materially changes the result.

## 3. Forearm: inconsistent link evidence

### `link4` and `link5` do not behave like two measurements of one component

Among the 43 videos where both links are available:

| Diagnostic | Result |
|---|---:|
| Pearson correlation between `link4` and `link5` | -0.065 |
| Spearman correlation between `link4` and `link5` | -0.015 |
| Mean `link4` rigidity | 0.0340 |
| Mean `link5` rigidity | 0.0186 |
| Median absolute difference | 0.0130 |

The two link scores are essentially unrelated, and `link4` runs at almost twice the mean magnitude of `link5`. Averaging them therefore adds heterogeneous measurement noise rather than reliably combining two views of the same deformation.

Changing the aggregation is not sufficient. Exploratory AUROC is `0.467` for the mean, `0.458` for the maximum, and `0.478` for the minimum.

### `link4` is missing in 12 of 55 cohort videos

`link4` is available in only 43 cohort videos, compared with all 55 for `link5`. All 12 `link4` failures are caused by insufficient valid target-depth frames. Six occur in forearm positives and six in negatives.

This is a pipeline-coupling defect for a rigidity-only analysis:

1. Target depth is calculated for the scale metric.
2. If target depth fails its 80% valid-frame requirement, the entire link report is marked failed.
3. Rigidity is then unavailable even though its required tracks and world pointmaps may still exist.
4. The forearm score silently changes from `mean(link4, link5)` to `link5 only`.

The meaning of “forearm rigidity” is therefore not constant across videos.

The problem is broader than this label cohort. Across all 187 completed videos, `link4` accounts for 51 of the 59 unavailable link measurements.

### The strict-quality filter changes the formula, not the forearm cohort

Direct-depth/full-SAM filtering leaves forearm `n = 55`, but removes `link4` from 10 rows and computes those rows from `link5` alone. Seven of the removed `link4` values are negatives, including several high scores (`0.0821`, `0.0866`, `0.0794`, and `0.0625`).

This changes forearm AUROC from `0.467` to `0.514`. The improvement does not demonstrate that direct depth fixes rigidity. Depth strategy is not an input to the rigidity formula. Instead, the filter removes a set of noisy `link4` contributions correlated with segmentation/depth problems.

### `link5` reverses direction in one dataset

| Dataset | Forearm positives | Component AUROC | `link5` AUROC |
|---|---:|---:|---:|
| `COSMOS3` | 18 / 25 | 0.611 | 0.583 |
| `LVP_ROBOWM` | 10 / 30 | 0.395 | 0.225 |

For `LVP_ROBOWM`, negative videos have higher `link5` scores than positives:

| LVP forearm label | Mean `link5` | Median `link5` |
|---|---:|---:|
| Negative | 0.02556 | 0.01945 |
| Positive | 0.01527 | 0.01305 |

Pooling the datasets therefore combines mildly useful `COSMOS3` ranking with strongly inverted `LVP_ROBOWM` ranking. This domain reversal is a major reason the combined forearm result is near chance.

Possible mechanisms include different clip lengths, motion patterns, occlusion, segmentation quality, and reconstruction behavior. The historical CSV does not retain enough per-frame evidence to identify which mechanism dominates.

## 4. Human-label noise adds measurable error

The component labels are not fully consistent with the text fields.

- `LVP_ROBOWM_0031` is forearm negative, but its note says “minor forearm deformation at sec1.” Its forearm rigidity is `0.05015`, so it is counted as a high-scoring false positive.
- Several forearm-positive rows have descriptions centered on gripper or object deformation without clearly documenting forearm deformation, including `COSMOS3_0003`, `COSMOS3_0014`, `COSMOS3_0050`, and `COSMOS3_0055`.
- Four forearm and nine upper-arm blanks are normalized to negative. Excluding blanks raises forearm AUROC only from `0.467` to `0.491` and upper-arm AUROC from `0.536` to `0.551`, so this policy contributes noise but does not explain the full failure.

The more consequential ambiguity is whether “arm deformation” means within-link shape change, uniform length change, abnormal joint articulation, or simply an implausible pose. Those targets require different metrics.

## 5. Likely tracking and reconstruction effects

The following mechanisms are strongly plausible from the code and representative masks, but cannot be quantified from the historical CSV.

### Frame-0-only anchor selection

Anchors and pairs are selected from frame 0. CoTracker is not re-seeded from later masks. If a deforming region becomes visible later, it may have no anchors. If an original anchor drifts, it can continue sampling unrelated pointmap geometry.

### Later tracks are not constrained by the link mask

Only the frame-0 mask affects pair ranking. Later sampled 3D points are not required to remain inside the corresponding link mask. A drifting track can sample background, another link, or an occluder and create high rigidity in a human-labeled negative.

This is consistent with the forearm’s high-scoring negatives, but the historical export lacks saved retained-track counts, per-frame mask membership, and sampled XYZ validity needed to prove the cause.

### Small and occluded `link4` region

Representative mask overlays show `link4` as a narrow joint-adjacent region with overlap and occlusion from neighboring links. That observation is consistent with its 51 full-dataset failures and 21 interpolation fallbacks. It is not, by itself, a quantitative segmentation-accuracy measurement.

Representative overlays:

- [`COSMOS3_0006/mask.png`](../outputs/COSMOS3_0006/mask.png)
- [`LVP_ROBOWM_0000/mask.png`](../outputs/LVP_ROBOWM_0000/mask.png)

### Robust statistics can suppress real local changes and retain reconstruction artifacts

The MAD protects against a few erroneous pair ratios, but it also rejects a small genuine deformation region. Conversely, if reconstruction or tracking error affects many pairs coherently but unequally, the same statistic can produce a false positive.

## 6. Root-cause ranking

| Rank | Cause | Upper arm | Forearm | Evidence strength |
|---:|---|---|---|---|
| 1 | Human-label concept does not match per-link ratio-dispersion rigidity | Major | Major | Confirmed from formula and descriptions |
| 2 | Too few positive labels | Major | Minor | Confirmed: 5 vs 28 positives |
| 3 | `link4` depth-gated missingness changes the component definition | Minor | Major | Confirmed from all 12 failure messages |
| 4 | Link-level disagreement and dataset reversal | Minor | Major | Confirmed from cohort statistics |
| 5 | Whole-video mean and robust MAD dilute brief/local deformation | Major | Major | Confirmed property of formula; event impact not saved |
| 6 | Anatomical label ambiguity and contradictory provenance | Major | Moderate | Confirmed examples; full relabel audit needed |
| 7 | Track drift, mask exit, and invalid sampled geometry | Possible | Likely | Mechanism confirmed in code; historical evidence unavailable |

## 7. Recommended validation sequence

### P0: repair the evaluation contract

1. Define separate human targets for **within-link nonuniform deformation**, **uniform length/scale change**, and **abnormal articulation**.
2. Re-adjudicate component labels with explicit link IDs or joint regions. Treat blanks as unknown until reviewed rather than automatically negative.
3. Expand upper-arm positives before drawing a performance conclusion. Five positives are not enough for a stable component benchmark.

### P1: decouple rigidity availability from scale depth

Compute and export rigidity whenever CoTracker and pointmaps provide sufficient evidence, even if target depth fails. A scale failure should not remove an otherwise valid rigidity measurement.

### P1: export diagnostic evidence

For every link, save:

- requested and retained CoTracker counts;
- per-frame visible pair counts;
- anchor mask-membership fraction over time;
- finite sampled-XYZ fraction;
- selected pair baselines and boundary distances;
- per-frame rigidity history;
- reason for every held/copied frame score.

Without these fields, false positives cannot be separated into genuine deformation, tracking drift, and MegaSAM reconstruction noise.

### P2: measure the missing deformation modes

Keep the current ratio-dispersion term, but evaluate it alongside:

1. **Uniform link-scale change:** absolute median log pair-distance ratio.
2. **Localized deformation:** upper quantile or trimmed maximum of pair residuals, with minimum support requirements.
3. **Transient deformation:** peak or high-percentile per-frame score in addition to the whole-video mean.
4. **Articulation abnormality:** CAD-relative joint centers, inter-link transforms, or cross-link endpoint geometry. Do not call this per-link rigidity.

### P2: stratify before pooling

Report `COSMOS3` and `LVP_ROBOWM` separately until the `link5` reversal is explained. A pooled near-chance result currently hides materially different behavior across the two generators.

## Bottom line

Upper-arm AUROC `0.536` is primarily an **insufficient and ambiguous evaluation-set result**: five positives, almost complete score overlap, and one influential outlier.

Forearm AUROC `0.467` is primarily a **measurement-consistency failure**: depth-gated `link4` missingness, no agreement between `link4` and `link5`, and a strong `LVP_ROBOWM` reversal in `link5`.

For both components, the current per-link rigidity formula cannot represent several deformation types named in the human labels. Until the label taxonomy and metric contract are aligned, near-chance performance is expected rather than surprising.

## Sources

- Frozen cohort statistics: [`data/pdi_output_analysis.json`](data/pdi_output_analysis.json)
- Historical metric export: [`../outputs/metrics.csv`](../outputs/metrics.csv)
- Cohort construction: [`../scripts/analyze_pdi_outputs.py`](../scripts/analyze_pdi_outputs.py)
- Rigidity formula: [`../PDI-Bench-edited/src/pdi_eval/v1/rigidity.py`](../PDI-Bench-edited/src/pdi_eval/v1/rigidity.py)
- Per-link evaluation path: [`../PDI-Bench-edited/src/pdi_eval/v1/pipeline.py`](../PDI-Bench-edited/src/pdi_eval/v1/pipeline.py)
- Target-depth gate: [`../PDI-Bench-edited/src/pdi_eval/perception/mega_sam_wrapper.py`](../PDI-Bench-edited/src/pdi_eval/perception/mega_sam_wrapper.py)
