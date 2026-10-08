# Link5 Shape-Anomaly-Codebook trial

On 2026-10-06 the owner selected **only the latest Link5 VLM guard outputs and
full-video SAM masks** for retention. The cache contains 45 selected videos plus
COSMOS2.5_0018, totaling 46 tracks and 5,058 frames. Its
`mask_cache_manifest.json` records source identities, selected points and hashes.
Local artifacts are owned by `robot/experiments/link5_shape_codebook/results/round_four_frame0`; `results/link5_shape_codebook` is a compatibility symlink; server
artifacts remain in the ERIS four-frame0 workspace's `results/link5_shape_codebook`.

Geometry, all observed clouds (including frame0), normal-reference arrays and
normalization, trained checkpoints, generated training reports/predictions,
synthetic examples, and replay exports were deleted at the owner's request.
The previous local selected45 Link5 predecessor outputs were also removed.
Source videos, shared models, pipeline code and the project GPU ledger remain.
The following sections describe the historical recipe, not currently available
training artifacts. No checkpoint passed research validation before cleanup.

On regeneration, `native_mask` validates and reuses the saved mask adjacent to
its receipt; synced absolute receipt paths do not force new VLM/SAM calls.
Guide mask selection uses the existing mask cache independently of deleted
point-cloud completion receipts. MegaSAM and subsequent cloud construction and
training must run again. No new GPU job was launched by cleanup. See
[the training recipe](STRUCTURAL_TRAINING.md) for the retained pipeline design.

## Fresh four-frame0 round, 2026-10-06

The active refinement uses **frame0 only** from `COSMOS3_0046`,
`COSMOS2.5_0044`, `LVP_ROBOWM_0044`, and `COSMOS3_0056`. Their new SAM masks
must match the three-positive VLM guard receipt, including VLM-placed P1.
`prepare_batch --reference-only` reuses validated all-frame SAM masks and rebuilds sparse
MegaSAM context for those four videos, then constructs `normal_reference/pooled_frame0.npz`
and `.ply` from only their filtered frame0 observations. Temporal reconstruction
context does not become a normal training sample. All later observations use
the same two-source-pixel erosion and ray-normalized 3D density filter.

The pooled cloud is an exact concatenation of observed points, with source-video
and pixel provenance; it contains no fabricated surface or per-cloud rescaling.
`construct_reference_examples` generates correspondence-preserving deformation
previews from that union. `build_constructed_reference_replay` updates the existing
reference viewer immediately, independently of detector training. Normal-codebook
training still receives **four separate normal frame0 samples**; synthesized
anomalies never populate the normal codebook.

The fresh ERIS workspace is `workspace-link5-shape-codebook-four-frame0-20261006`.
Its output may reuse only this round's validated current VLM/SAM mask cache;
earlier baseline masks, clouds, normalization and checkpoints are not reused.
The new split is `splits/normal_reference.json`, and training receipts use
`normal_manifest.json`. Selection, normal-only Phase1 receipts, epoch losses and
completion counts support four observations (1500 epochs = 6000 native training
batches). Historical train20 runs remain readable through explicit compatibility.

## Earlier local refinement, 2026-10-06 (historical)

The existing `results/link5_shape_codebook/full_video/replay/index.html` now
selects the new depth-filter preview by default. Choose **New depth filter**,
**Kept + removed points**, or **Previous scored cloud**. Saved anomaly scores,
pose estimates, source masks and training inputs belong to the prior GPU run.
Updated clouds have not been scored or used to retrain either codebook.
The existing `full_video/reference_replay/index.html` has both previous/refined
normal-cloud views and a VLM-run selector for frame0 prompt comparisons.

The shared `robot/preprocessing/depth/link5_depth_filter.py` replaces the imported Simple3D filter for fresh runs. It keeps
the raw source mask, erodes detector support by two original-image pixels before
nearest-neighbor depth-grid alignment, and rejects sparse depth samples using
32-neighbor mean 3D distance divided by the camera pixel footprint z/√(fx·fy).
The threshold is max(3 × median, median + 6 × 1.4826 × MAD). It does not crop to
normal geometry, fit CAD, clip global depth percentiles, smooth depth or move
retained points. Coherent wrong masks/depth can remain. Source inputs feeding
existing trained checkpoints are protected from an in-place refresh; fresh GPU
preparation/training must use a new output root.

The Link5 guard alone requests Gemini 3.8 Flash **high**, with a 65,536-token
output budget and 600-second timeout, and Luna6 fallback **xhigh**, with the same
budget and 1,200-second timeout. The negative system/user prompts match the last GPU experiment. The positive
guard now places all three points using the cleaned user-marked reference: P1
has no box-relative default, and the reply must provide P1/P2/P3 coordinates.
See [the current positive guard prompt](POSITIVE_GUARD_PROMPT.md). Other links'
role defaults stay unchanged. Local paid tests run the exact frozen original
frame0 proposals, not SAM segmentation. The first train20 batch preceded the
larger-budget request and requested 16,384 tokens; four difficult cases are
separately retested at the final 64K settings with both models. Individual API
receipts record the actual requested settings, usage and failures.

`refinement/metadata/depth_summary.json` records CPU re-filtering of all 45 frame0
observations plus every later frame in the completed 30-video cohort (3,325
observations). `refinement/metadata/vlm_costs.json` and `.csv` record token usage,
public-price estimates, failed calls and unknown charges; `COST_COMPARISON.md`
compares the same Gemini frames to avoid the previous 15-Gemini/5-Luna mixture.
These are usage-based estimates at current 302.AI rates, not invoice charges.

```bash
python -m robot.experiments.link5_shape_codebook.refine_guard_local --root results/link5_shape_codebook --compare-luna --secrets-file /path/to/existing/.env.vlm
python -m robot.experiments.link5_shape_codebook.build_refinement_replay --root results/link5_shape_codebook
python -m robot.experiments.link5_shape_codebook.build_reference_replay --root results/link5_shape_codebook
python -m robot.experiments.link5_shape_codebook.guard_costs --root results/link5_shape_codebook
```

## Prior GPU run (2026-10-05)

The current evaluation scope is **30 full videos**: the first ten matched numbers
(0001, 0005, 0010, 0015, 0021, 0025, 0030, 0035, 0040, 0044), each from all three
generators. The user reduced the original45-video evaluation while it was running.
`stop_after_cohort.py` verifies every frame's score and both complete sums before
stopping the owned Slurm allocation. Scheduler cancellation is recorded as the
requested early stop; it is not relabeled as a successful45-video job.
The original launch config, training inputs,20/25 split and checkpoints remain
unchanged. Outputs outside the selected30 are preserved as provenance and excluded
from the selected replay and human-label analysis.

The replay reuses `infrastructure/shared/replay/rigidity_replay.html`, its offline
Plotly library, video/cloud layout, history cursor and video-decoder synchronization.
`replay_framework.py` adapts the existing layout and `replay_app.js` connects saved
anomaly evidence; it does not invent rigidity pairs or rerun inference.
After the selected job completes, `label_correlation.py` reads **AB only** from
`documentation/data/labels/selected_45_matched_videos_styledv2.xlsx`, leaving the
workbook unchanged. It correlates the requested full-video sums with numeric labels
using Pearson and average-rank Spearman, for all selected videos, held-out videos,
and each generator. Missing scores are excluded explicitly and never replaced by0.

## Full-video testing contract (user clarification, 2026-10-05)

**Link5 testing scores every decoded frame of each full video, not ten sampled
frames. The ten-frame testing configuration belongs to the Simple3D experiment
only.** The primary frame score remains `raw_mean_top80`; the primary video score
is `sum_all_frame_raw_mean_top80 = sum(raw_mean_top80_t for every frame t)`.
Frame0 is included in this requested sum and uses its exact frozen camera-space
reference input without FoundationPose. The sum is not divided by frame count,
duration or FPS. Videos from different generators have different frame counts;
the replay and comparison table display those counts beside the requested sums.
If any frame is unavailable, the complete sum is unavailable rather than treating
the missing frame as zero. An explicitly labeled available-frame subtotal and
the missing frame IDs remain recorded.

`full_video.py` calls the existing MegaSAM implementation on each original full
video and reuses this trial's fresh current guarded masks, which already cover
every frame. The completed prior run imported the Simple3D local depth filter
with zero erosion and no mask mapping; new preparation uses the refinement above. Full-sequence geometry and frame inputs live under
`results/link5_shape_codebook/full_video/`. The original twenty normal training
clouds, split, fixed normalization and both final checkpoints stay unchanged.
Dense frame0 geometry is compared against its frozen reference for scale QC;
no per-video rescaling is applied to hide discrepancies.

Later frames are independently registered with official FoundationPose and the
Link5 CAD. Its returned matrix maps original CAD coordinates into camera space.
Canonicalization is `P_reference = T_ref @ inverse(T_t) @ P_camera`, followed by
the same fixed normalization. Both detector checkpoints share the same poses and
observed clouds. No tracking, ICP or nonrigid registration is used.

The user explicitly requested full-video exploratory testing after the failed
synthetic-sensitivity checks were reported. `full_video/config.json` records this
authorization and hashes the unchanged failed receipts; it never changes those
receipts to passed. One Slurm allocation uses two H200 GPUs and ten video workers
(five per GPU), evaluating all45 videos as fifteen matched-number triplets across
LVP_ROBOWM, COSMOS2.5 and COSMOS3. The initial sparse preparation below supplied
frame0 training data and temporal reconstruction context; it is not this testing
contract. The older sparse scoring entry point is retained for provenance only.

```bash
python -m robot.experiments.link5_shape_codebook.full_video \
  --config results/link5_shape_codebook/full_video/config.json --phase batch
```

This isolated experiment calls the pinned [author repository](https://github.com/alexandor91/Shape-Anomaly-Codebook) directly. Detector, FPS patchification, codebook, attention, modulation, losses and training entry point are upstream code. `paper_original` uses the original augmentation; `robot_structural` replaces only its pseudo-anomaly callable. Link5 adapters handle native infrastructure, coordinate conventions, provenance and requested scores.

The public implementation is **not paper-exact**: see [paper_audit.json](paper_audit.json). It provides a PointNet++ substitute, no pretrained MinkUNet checkpoint, a default RoPE head dimension that fails its own assertion, and three missing-region augmentations with unchanged inputs and nonzero target offsets. None of these findings should be hidden. At the user’s explicit instruction, the trial imports the official [FCGF](https://github.com/chrischoy/FCGF)32-D pretrained MinkowskiEngine ResUNetBN2C through upstream's `external_encoder` interface. Its architecture and 3DMatch checkpoint differ from paper MinkUNet34C and are recorded explicitly. Strict source/weight checks forbid PointNet++ or random sparse-backbone fallback. Native missing-region augmentation and original losses are retained unchanged and recorded as deviations. Compatible RoPE head dimension66 is an explicit configuration change.

The experiment output is one `results/link5_shape_codebook/` root, with one case per original video. Production masking is called anew via `robot.workflows.coordination.link5_masks` and its worker. Historical masks/geometry are never read. In the completed prior run, Simple3D `Config` and `depth_filter` were imported from `infrastructure.shared.experimental.simple3d.simple3d_pair_visibility`, with zero erosion and no mask mapping. Fresh runs now use the Link5-specific filter described above. The guarded source mask is retained verbatim. Nearest-neighbor resolution alignment supplies its support on the native MegaSAM depth grid. The prior outputs retain their original local5x5 median/MAD depth rejection for comparison. Full-scene sparse real frames provide MegaSAM context; camera clouds are back-projected from native CVD depth/intrinsics. No per-cloud centering, rescaling, reconstruction or point replication is performed.

Run stages with their declared isolated environments:

```bash
python -m robot.experiments.link5_shape_codebook.prepare_batch --config results/link5_shape_codebook/config.json
python -m robot.experiments.link5_shape_codebook.pose --config results/link5_shape_codebook/config.json --reference
python -m robot.experiments.link5_shape_codebook.structural_validation --config results/link5_shape_codebook/config.json
python -m robot.experiments.link5_shape_codebook.pose_sanity --config results/link5_shape_codebook/config.json
python -m robot.experiments.link5_shape_codebook.train --config results/link5_shape_codebook/config.json
python -m robot.experiments.link5_shape_codebook.sanity --config results/link5_shape_codebook/config.json
python -m robot.experiments.link5_shape_codebook.score --config results/link5_shape_codebook/config.json
python -m robot.experiments.link5_shape_codebook.train --config results/link5_shape_codebook/config.json --augmentation-mode robot_structural
python -m robot.experiments.link5_shape_codebook.sanity --config results/link5_shape_codebook/config.json --augmentation-mode robot_structural
python -m robot.experiments.link5_shape_codebook.pose --config results/link5_shape_codebook/config.json --runtime --augmentation-mode robot_structural
python -m robot.experiments.link5_shape_codebook.score --config results/link5_shape_codebook/config.json --augmentation-mode robot_structural
```

ERIS uses `eris_session.sh` within a Slurm allocation. Preparation uses the existing geometry interpreter and calls the existing SAM3 interpreter for current masks. FoundationPose and codebook environments install independently under `Wilson/dependency`; `install_eris_dependencies.sh` uses upstream build/install instructions and public FoundationPose weights. `verify_backbone` strictly loads the official FCGF3DMatch checkpoint, verifies a real GPU32-D feature probe and freezes its SHA256 before training. An author-provided pretrained MinkUNet34C may be configured explicitly if available; the linked author repository does not release it.

`alignment/` contains unmodified frame0 overlays, centroid/bounding-box/PCA/point-count/mask/scale records and pairwise distances. Geometry-only thresholds and candidate rejections are explicit. At least20 naturally aligned candidates and all45 fresh observations are required; the twenty selected examples cover normal observed visibility variation. The remaining25 are held out, including any flags; candidates are not silently canonicalized. `splits/train20_test25.json` hashes exact inputs. One FoundationPose call on the clean medoid defines `T_ref`; transformed original CAD vertices define one fixed bbox center and radius. CAD never supplies normal anomaly templates. `link5_normalization.json` is shared by every input.

`training/` calls the original1500-epoch two-phase training, with Adam1e-3, lambda_sim/lambda_BCE0.5, tau0.85, industrial8x192/32x64/64x32 patches and augmentation0.01/0.1. Observations below10000 points stay sparse. Original codebook hash sets, absent from upstream state_dict, are retained separately. Checkpoints, exact20 manifest, effective configuration, source hashes, raw epoch losses, curves and codebook statistics are retained. The final1500-epoch checkpoint is the one detector used at runtime.

The separate structural run writes `training_robot_structural/link5/final.pt`. Both modes use the same exact20 inputs, shared normalized NPY files, split, fixed normalization, frozen official FCGF backbone, initialization seed, optimizer and1500 epochs. Native Phase1 construction and continuous Phase2 codebook updates receive **normal points only** in both modes. They are preserved unchanged; synthetic points never populate the codebook. `phase1_normal_codebook.pt` records the initial normal construction in each run. Learned patch features and later codebook entries can differ as training responds to the different pseudo-anomalies.

For the structural generator, one longitudinal unit axis `u`, proximal anchor `a` and reference length `L` are fitted from the pooled20 normal references once. Define `x=(p-a)·u`, `r=p-a-xu`, deformation start `s`, and `t=max(x-s,0)`. Axial strain is `p′=p+(alpha-1)t u`: the proximal section stays unchanged and perpendicular thickness is unchanged. Alpha is uniform in[0.70,0.95] for shortening or[1.05,1.30] for lengthening. Affected fractions are sampled uniformly from one of[0.40,0.60],[0.60,0.90],[0.90,1.00], with equal region-family probability; `s=(1-f)L`.

For bending choose a uniform direction `v` perpendicular to `u`, rotation axis `w=u×v`, and curvature `kappa`. For `x>s`, use `theta=kappa*t`, circular centerline `c=a+s*u+sin(theta)/kappa*u+(1-cos(theta))/kappa*v`, then `p′=c+R_w(theta)r`. Points with `x<=s` are unchanged. This preserves each local cross-section through a rigid rotation and parametrizes the centerline by arclength. The implementation evaluates `1-cos(theta)` as `2*sin(theta/2)^2` for stability. Start is uniform in[0,0.5]L; terminal angle is uniform5–15° for mild bends or20–45° for strong bends. Mild and strong bends are equally likely. The distribution is50%axial/50%bending, with equal shortening/lengthening. Zero curvature is identity. Targets are exactly `offset_gt=P_normal-P_anomaly` and a mask of points with an actual nonzero offset; point order/count are preserved.

`augmentation/robot_structural_examples/` contains normal, shortened, lengthened, mild-bend and strong-bend PLYs, exact correspondence/GT NPZs and rendered comparisons. `robot_structural_validation.json` gates training on target equality, zero offsets for unaffected points and preserved transverse geometry. No real deformed video or missing-surface reconstruction is needed for training. `scores_robot_structural/`, `sanity_robot_structural/` and suffixed per-observation prediction files retain its independent results without overwriting the baseline.

Runtime uses independent `FoundationPose.register` per sparse frame with five native refinement steps, full aligned scene depth and a separate intact mask, never tracking or ICP. Native NumPy diameter is cast to an equal-valued Python float for torch dtype compatibility; CAD PBR appearance is converted to SimpleMaterial without changing geometry. Native top1 is retained unless its mask IoU is below0.1 and an existing native hypothesis has IoU at least0.5 with gain at least0.4; then the best mask-associated native hypothesis is selected. This explicitly recorded pose-QC adapter handles wrong-neighbor associations on repeated robot-link appearances. It fits no new transform and does not reject low-IoU observations lacking a qualifying alternative. Original/selected transforms, mask IoUs, native rank and score are saved. Its returned matrix is `T_camera<-original_link5_mesh`; the source composes its internal mesh-centering offset before returning. Observed points are aligned as `T_ref @ inverse(T_t)`. Each pose saves matrix, native rank score (not calibrated confidence), original-mesh CAD projection, CAD-mask IoU, point count and status. A weak CAD fit remains scoreable. Only explicit catastrophic fallback/off-camera/far-away failures make the score unavailable; failures remain in reports.

`sanity/pose_checks.json` gates training with real rigid-pose alignment before the1500-epoch run. `sanity/checks.json` gates full scoring: natural normal alignment, real presumed-normal LVP rigid poses, known-transform invariance, author deformation augmentation at both severities, and twenty-five held-out frame0 normals. Configuration thresholds are predeclared trial QC, not calibrated physical deformation thresholds. The LVP later-frame pose-check candidates are provisional: the user confirms all frame0 clouds are normal but has not established later-frame normality. Failed checks stop full evaluation.

Each `anomaly.npz` retains normalized coordinates, original observed-point indices, aligned coordinates, offsets, logits, the original paper-style minmax map and raw `L1(offset) * sigmoid(logit)` values. `scores/frames.csv` retains raw max/mean/p95/top20/top80/top5%; the cross-frame primary score is raw_mean_top80. `scores/videos.csv` retains sampled-frame max/median/top3 and failures. Frame0 is scored directly, without registration, and is excluded from later-frame video aggregates. Visibility correlations are descriptive Spearman checks, never removal rules or anomaly thresholds. Scoring is experimental; it does not establish paper performance reproduction or physical deformation accuracy.

## Earlier normal and deformation reference replay (removed from current UI)

The same viewer's `#guard` tab replays the exact20 frame0 SAM prompts and masks. `build_reference_replay.py` selects the unique successful masking attempt, requires reviewed guard output, checks the SAM diagnostic points against the selected VLM points, and verifies each source mask against the segmentation archive. The tab distinguishes deterministic P1 from VLM-reviewed P2/P3 and N1/N2, shows raw answers and role-specific input images, and displays original masks plus the training-grid mask over RGB. Its replay cycles through the unchanged twenty-entry training manifest.

`build_reference_replay.py` reads and verifies the exact20 stored normal arrays against their frame0 camera clouds and fixed normalization. `full_video/reference_replay/index.html` displays all176932 saved observed points, the original RGB/masks and downloadable source arrays. Its deformation tab replays the saved9865-point LVP_ROBOWM_0065 examples with exact GT offsets and masks; intermediate animation is explicitly interpolation between saved endpoints. The overlay uses the existing published Plotly library and styling. Original per-visit minibatch indices were not saved; the viewer does not claim to reconstruct historical stochastic minibatches.

The AB workbook reader supports both inline and shared strings. Its45 AB values were checked independently against read-only openpyxl; all30 selected labels were reconciled to their workbook row identities. The original workbook remains unchanged.

Replay interface guidance belongs to this experiment: [DESIGN.md](DESIGN.md).
