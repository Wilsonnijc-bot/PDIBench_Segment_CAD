# Pair-visible link5 evaluation

This rerun leaves the existing object-occlusion pipeline and official Simple3D
implementation unchanged. It reuses the exact frozen 45-video selection,
`link5-only-selected45-updated-mask-20260929` masks, original GPU video paths,
and saved `cotracker_exact-group.npz` 2D tracks. No masks, tracks, old XYZ or
cached MegaSaM geometry are inferred/reused respectively.

The current run lives inside the existing single top-level result folder:
`results/simple3d-20261003-run2/link5_pair_visible_anchor/`. The earlier
stricter-hull trial keeps masks and score summaries in `link5_pair_visible/`;
its 40 cloud files and embedded cloud replay were deleted at the user's request. The top-level `index.html`
opens the new replay; the earlier object/CAD replay remains linked.

## Reproduction

Use the existing GPU Python environment and native Simple3D CUDA extensions.
Prepare a new, empty root locally, then stage its `inputs/` and the experiment
code to the GPU. The preparer obtains videos from the existing frozen selection
through `pdi_eval.experiment.contracts.video_path`.

```bash
python experiments/prepare_simple3d_pairs.py \
  --output results/simple3d-my-run/inputs \
  --erosion-px 2 --visibility-mode smaller_view_anchor --depth-window 5 \
  --depth-mad-multiplier 8 --depth-relative-threshold 0.05

python experiments/run_simple3d_link5_pairs.py \
  --root results/simple3d-my-run --preflight

python experiments/run_simple3d_link5_pairs.py \
  --root results/simple3d-my-run --workers 2
```

`--resume` processes missing pairs without replacing terminal pair evidence.
The actual current GPU launch is `experiments/simple3d_pairs_gpu_job.sh`, under
tmux `simple3d-link5-allpairs-20261004`, with four spawned video workers after
the user-requested concurrency check. Set `S3D_PAIR_WORKERS=4` when invoking
the GPU launcher, or pass `--workers 4` to the Python entry point.
`S3D_PAIR_ROOT` selects the revised root. The initial `S3D_PAIR_LIMIT=3`
preview is complete. The user approved all45 on 2026-10-04; full continuation
tmux `simple3d-link5-anchor-all45-20261004` resumes without that limit and
retains the three videos' terminal pairs.
`metadata/execution/run.log` and `run.exit` retain the run log and exit code.
All thresholds are frozen in `inputs/manifest.json`; modify preparation
arguments/configuration before creating a new run, never during an active run.
Advanced image-support thresholds live in the preparer's `Config` dataclass
and are recorded along with the masks. They are not selected from human labels.

For the current remote run, collect/verify/update the replay locally with:

```bash
python experiments/collect_simple3d_pairs.py --watch --root results/simple3d-20261003-run2/link5_pair_visible_anchor
python experiments/verify_simple3d_pairs.py --root results/simple3d-20261003-run2/link5_pair_visible_anchor
python scripts/build_simple3d_pair_replay.py --root results/simple3d-20261003-run2/link5_pair_visible_anchor
```

## Selection and common support

The normal image is frame0, or the first frame with at least256 mask pixels
after erosion. The original video interval `[0.1 * frame_count, frame_count)`
is divided into ten disjoint equal bins. Each bin contributes the largest raw
mask among valid saved masks; equal areas choose the earliest frame. The base
frame is excluded from tests. Empty bins are invalid, never duplicated. The
current45 videos each have eleven images: one base and exactly ten tests.
All frame indices, source timestamps, raw/eroded mask areas and bin boundaries
are saved, including the full candidate-frame area series.

The object-occlusion pipeline separates actually available object pixels from
an expected full silhouette and maps availability back to the reference. Its
direct bounding-box mapping is unsafe when the reference or test is itself
camera-clipped. Here its native `occlusion.sample` function identifies the
existing track identities that are visible and inside link5 in **both** images.
An OpenCV RANSAC global2D similarity (rotation, uniform scale, translation;
6px RANSAC error) supplies the image-only mask map. On 2026-10-04 the user
explicitly removed the track-quality gate: track count, inlier percentage and
hull coverage are now diagnostics, not pair-acceptance vetoes. No
geometry/RGB/features are warped. The view with
less observed mask area is the anchor, comparing reference area times image
scale squared with test area. Map the anchor original mask into the fuller view
and clip it to that view's original mask; remove anchor pixels whose mapped
location is unavailable there. There are no repeated hull intersections. Both
outputs remain inside their own observed masks and must each cover>=10% of
their original mask with>=256 pixels. This permits extrapolation from reliable
tracks to saved-mask edges; it does not stretch a cropped bbox to a full one.
The prior method is reproducible with `--visibility-mode strict_hulls --erosion-px 3 --track-quality-gate`.

The revised input manifest accepts **450/450 pairs**. The former 239
track-gate rejections are requeued in the same result root; successful scores
and their masks remain unchanged. `metadata/track_gate_change.json` preserves
the old rejection reasons, run identity and successful-comparison hashes.
A global similarity assumes approximate
image similarity; perspective/view changes and deformations can both break
that assumption. This is a 2D visibility estimate, not proof of dense3D
surface correspondence. Extrapolation to mask edges can be inaccurate under
perspective change or self-occlusion, and saved visibility can be wrong. No
unmatched full/partial fallback is used, and no attempt is made to fix these
remaining limitations with a new registration/anomaly model.

## Geometry and cleanup order

1. Construct smaller-view anchored support on the original pixel grid.
2. Erode that mask with a radius2px elliptical structuring element, configurable
   to2/3/4px. Zero padding also erodes at the camera boundary.
3. Run a **fresh cropped bootstrap MegaSaM sequence** to obtain the depth needed
   for cleanup. The two evaluated views use their pair-specific eroded masks.
4. Filter only finite positive depth inside those masks, on MegaSaM's native
   depth grid. In a5x5 valid-only neighborhood, reject a center whose residual
   exceeds `max(0.05 * local_median, 8 * 1.4826 * local_MAD)`, provided>=8 valid
   neighbors exist and<=2 neighbors share that center's depth within the same
   threshold. This extra isolation requirement preserves supported depth steps.
   Sparse neighborhoods retain finite depths. Depth values are never replaced.
5. Map only actual rejection flags back to source pixels with nearest-neighbor
   sampling and subtract them from the unchanged eroded source mask. Resizing
   the entire valid silhouette would add an unintended erosion even when no
   depth was rejected. When a crop is upsampled, also mark the source pixel that
   supplied each flagged native mask cell so isolated flags cannot disappear in
   the reverse resize. Run a **second fresh MegaSaM
   sequence** from cleaned pair inputs, with a different unique scene/cache
   identity and `cache_dir=None`.
6. Validate/filter the new final depth before XYZ construction. A cloud needs
   >=256 final depth pixels and>=50% retained mask area. Both passes retain
   exact input videos, unannotated masked input RGB, original-grid and
   native-depth-grid masks/rejections, depths, intrinsics, poses and status.

Each sequence contains the same eleven selected **real** images jointly, to
retain the existing temporal tracker's context and initialization behavior.
The other nine images are eroded-link5-masked context only; their geometry is
never scored or used as a Simple3D reference. One integer-coordinate crop
canvas covers the sequence's foreground masks; outside-mask RGB is black.
This changes what the depth models see, rather than applying a mask only to an
old point cloud. Cropped input coordinates are retained for replay mapping.
Original saved masks and source frames remain separate from model inputs.

A depth consistency check cannot precede depth estimation. The bootstrap
pass makes that dependency explicit; cleanup precedes the final reconstruction.
Final depth validation is still necessary because the reconstructed depths can
change. There is no iterative fitting, global depth/XYZ smoothing or3D alignment.

Native CVD is attempted as in the existing MegaSaM integration. Its normal raw
DROID fallback, if used, is recorded as `raw_DROID_CVD_unavailable`, not silently
reported as CVD. Reconstruction failure/zero pointmaps invalidates the pair.

## Official scoring and evidence

Back-projection uses native MegaSaM intrinsics and camera poses. The existing
conservative Open3D20-neighbor/statistical4-sigma outlier removal, authors'
per-cloud centroid/unit-radius normalization,0.005 normalized voxel sampling,
and seeded8192-point maximum remain unchanged. Native pixels and continuous
source-image coordinates are saved with every real sampled point. Original,
finite, outlier and sampled counts, centroid and scale divisor are recorded.

Every pair calls the existing thin `simple3d_adapter` to create a **new** native
FPFH/MSND/LFSA reference and204-prototype5% coreset from its reconstructed normal
cloud. Native nearest-prototype distance, interpolation, smoothing and mean
of the largest80 point scores are unchanged. No training/fine-tuning, label
threshold, handcrafted deformation score or added anomaly model is used.

Each `cases/<video>/bin_XX/` contains stage masks, two-pass geometry evidence,
reference/test clouds, reference prototypes, continuous point scores,
top80 indices/values and `comparison.json`. Failures preserve their reason and
have null scores. `comparisons.csv/json` and `metadata/summary.json` provide
aggregate evidence. `metadata/artifact_verification.json` checks source-pixel
mapping, mask containment, native back-projection, normalization, independent
scene identities, rebuilt coresets and exact top80 aggregation.

The replay supports stage overlays in both original images, both new3D clouds,
red top80 contributors, continuous scores/RGB colors, raw-score hover and
source-pixel inspection, per-pair downloads and scalar history. Refresh loads
newly collected results; pending and invalid pairs remain explicit gaps.
