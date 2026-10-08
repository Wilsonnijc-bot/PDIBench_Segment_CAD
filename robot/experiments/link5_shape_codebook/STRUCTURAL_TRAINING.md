# Current Link5 training: structural deformations only

Every training shape starts from frame0 of one of these four normal observations:
`COSMOS3_0046`, `COSMOS2.5_0044`, `LVP_ROBOWM_0044`, `COSMOS3_0056`.
The pooled34,280-point cloud is their replay; training uses four individual clouds.
The normal codebook receives only those four originals. Its geometry, center and
scale remain fixed. No later frame becomes a normal reference.

The user-selected guides are COSMOS2.5_0018 frame48 at3s, COSMOS2.5_0030 frame80
at5s, and LVP_ROBOWM_0005's last decoded frame48. The COSMOS examples guide bending;
LVP005 guides shortening. Each has a fresh/current three-positive guarded SAM
track, full-sequence MegaSAM CVD reconstruction, and the same2-source-pixel
erosion plus ray-normalized32NN density filter. Guides influence the synthetic
shape distribution; they supply no invented per-point offsets or defect labels
and never enter optimization or the normal codebook.

## What the generator teaches

| Family | Frozen sampling range | What changes |
|---|---|---|
| Shortening | alpha0.70–0.99 | Compress the affected axial section by1–30%, preserving thickness |
| Lengthening | alpha1.01–1.30 | Extend the affected axial section by1–30%, preserving thickness |
| Mild bending | terminal angle5–15deg | Bend the centerline into an arc, rigidly rotating each cross-section |
| Strong bending | terminal angle20–60deg | The same arc construction with larger curvature |

Bending starts between0%and60%of the fixed normal axis; axial affected fractions
retain the earlier40–60%,60–90%,90–100%ranges. Each family gets one quarter of
the training draws, with fresh parameters/directions. This is the unchanged
`robot_structural.py` generator with explicitly widened configuration ranges.
The guide clouds are qualitative examples: visibility, joints, rigid motion and
reconstruction prevent treating their raw extent ratios/PCA angles as calibrated
physical strain or bending measurements.

For every synthetic cloud, point identities are preserved. The restoration
target is exactly `normal - deformed`; the label marks actual moved points.
Training also sees the unchanged original with zero offset and zero defect
targets. The [paired objective](LOCALIZATION_TRAINING.md#exact-experimental-loss)
balances deformed/background points and asks each moved point to score above its
matching clean point. All scores remain `L1(predicted_offset)*sigmoid(mask_logit)`.

## Models and why the new head exists

`structural_point_residual` retains the upstream network plus its frozen point
features at attention output. `structural_geometry_head` keeps the strict frozen
FCGF backbone and upstream normal codebook, adding a learned restoration and
probability head. It receives native features/predictions, fixed-reference XYZ,
nearest observed normal geometry, and whole-link length/cross-section profiles.
The whole-link features give it information about shortening and bending that
local surface descriptors can miss.

Its geometry bank contains only the four frozen normal clouds. During training,
the current source is excluded from auxiliary nearest-neighbor queries, preventing
an exact-self-match shortcut. Inference queries all four. Profile quantiles are
input features; they never rescale, center, crop or replace a cloud. The new head
predicts the scores; no hand-written distance score is substituted. This is an
explicit research architecture adaptation, not a paper-exact implementation.

Two independent variants share one H200. Both use Adam0.001, batch1, seed0,
1500epochs/6000updates, fixed point order, and normal-only codebook refresh.
Sources, launchers, guide manifest and deformation settings are snapshotted.
Each variant saves16 actual training inputs/GT arrays, including each of the
four normal sources with each family. These are captured optimizer examples,
not reconstructed historical minibatches.

## Evidence required before selecting a checkpoint

The three guide-video identities are excluded from independent validation.
At epochs100/300/1500, each of39 other normal frame0 clouds is tested unchanged and
with ten corresponding synthetic variants:5%/25%shortening/lengthening, plus
10/35/60deg bends in two directions. Known offsets/regions allow honest spatial
localization evaluation.

The research gate requires median point AUROC>=0.80, median clean false-positive
fraction<=0.02 and its95th percentile<=0.10 at the frozen training-clean99.5%
threshold. At least80%of strong examples in **each** family must have raw top80,
region/background and matched-region/clean score ratios>=1.20. Low loss or a
high rank metric on almost-equal scores does not pass this gate.

Independent rigid FoundationPose registration aligns each observed guide and its
frame0 companion as `T_ref @ inverse(T_t)`; the shared normal center/scale is
unchanged. Subsequent guide predictions are qualitative checks because there is
no dense real defect ground truth. Preserve weak-fit/failed-pose diagnostics.
These frames cannot establish general real-world sensitivity on their own.

Outputs live in `results/link5_shape_codebook/round_four_frame0/` locally:
`guides/` for observed inputs/provenance and `structural_localization/report_fix_v2/<variant>/`
for policy, losses, checkpoints, captured training examples and validation.
Older default-generator trials are retained comparisons and are superseded.
Original production checkpoints/full-video gates are unchanged. Checkpoint
selection is recorded only after validation; training completion is not success.

## Reconstruction-tolerant fine-tuning

The first geometry model learned the synthetic defect regions but raised too many
false alarms on normal reconstruction variation. The new stage freezes its native
network/FCGF/codebook at epoch300 and trains the added geometry head only. Both
members of a clean/deformed pair receive the same uniform reconstruction-scale,
small rigid-pose and point-noise variation. Their difference remains exactly the
deliberate structural defect. The observed inputs and four-reference codebook
remain unchanged. Uniform scaling is taught as normal; shortening/lengthening
along the link while preserving thickness remains abnormal.

Adam0.0005/batch1 continues for600 fine-tuning epochs2400updates. The versioned
`benign_finetune_v4` run resumes the full optimizer/random state at epoch175 after
correcting a float32 roundoff assertion; the old returned points/targets are
unchanged and the failed v3 history is preserved. Checkpoints are saved before
evaluation and every25epochs, allowing coherent recovery without losing training.

Threshold calibration uses only four source-excluded originals and100 independent
benign copies of each, at the fixed99.5%quantile. Test both raw-camera clouds and
native FoundationPose rigid-aligned clouds without scale correction. Keep the
original research gate and report strong defect-point recall at that threshold,
so suppressing false alarms by making detection insensitive cannot count as
effective. These39clouds now support development and model selection; the checks
are not a blind final benchmark. The three real guide frames are qualitative
checks with actual predictions and no dense defect GT. COSMOS0030 has a weak
CAD-mask pose fit (IoU0.15746), which must accompany interpretation of its score.

## Current display and ongoing comparisons

The user-facing replay now contains only the exact pooled four-frame0 normal
cloud and42 observed test predictions: three real guide frames plus39 other
frame0 observations. It excludes guide companions, synthetic examples, prompt
previews, optimizer pairs, tables and historical variants. Those actual inputs,
scores and training records remain in the experiment root; generated diagnostic
replay assets were deleted. `build_structural_replay` and `build_training_replay`
now publish this two-view layout. Metadata records the exact displayed checkpoint.

Both plots share coordinates and axis ranges. Raw scores remain unchanged; the
shared color ceiling is3×the saved normal-only threshold, with high values
saturating rather than compressing every other cloud into purple. The optional
flag highlight applies the original `raw_score > threshold` decision exactly.
The interim display remains v4epoch600 and explicitly says validation failed.
Its LVP005 last-frame prediction flags0%of points at the saved threshold.

Allocation5625608 completed46:35/exit0 with two3000-epoch warm-start head comparisons
on the same H200: the existing geometry head and an expanded intrinsic length/thickness and
centerline-profile head. Four reference clouds, native weights/codebook, paired
loss and own defect generator remain frozen. An overlapping2000-epoch freshly
initialized learned profile decoder also completed: it retains the native model
in its checkpoint but bypasses native retrieval for scoring. This control has
aligned median pointAUROC0.99992 and strong shortening/lengthening/bending recall
0.97939/0.97893/0.95446, but median normal-point false-positive fraction0.62836
and p950.78091. It fails validation and is not promoted. Strong localization of
synthetics does not establish reliable normal-vs-defect discrimination.

The final v5geometry head has aligned median pointAUROC0.99824 and strong
shortening/lengthening/bending recall0.94339/0.86188/0.93836. It fails normal
false-positive validation: p950.50863 exceeds0.10. The intrinsic variant's
normalFPRmedian0.39800/p950.70226 also fails. On the three real guides, v5geometry
flags19.6%of COSMOS0018frame48,99.9%of COSMOS0030frame80, and0%of LVP005frame48.
The COSMOS0030 pose remains weak; flags are not real pointwise detection accuracy.
No checkpoint is promoted, and the replay remains explicitly the earlier v4
candidate. All planned training has finished; next work should diagnose actual
RGB-to-cloud defect preservation rather than repeat the same recipe.
