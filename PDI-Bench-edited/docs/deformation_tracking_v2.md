# Deformation Tracking V2 Decision

Implementation: `src/pdi_eval/v2/` owns the pipeline, tracking policy, and
rigidity scorer. `pdi_eval.v2.evaluate(...)` is its short Python interface;
`python -m pdi_eval.experiment score --version v2` is its CLI. Segmentation, MegaSAM,
CoTracker execution, and replay utilities are shared. See the root
[D specification](../../D_PIPELINE_SPEC.md) for the boundary between
historical D and V2.

Decision date: September 6, 2026

## Status

The historical `python -m pdi_eval.experiment score --version v1` pipeline is retained for
reproduction, but it is retired for new robot-deformation measurements. New
deformation runs use `python -m pdi_eval.experiment score --version v2` and
`configs/deformation_v2.yaml`.

V2 reports **per-link rigidity only**. Scale, trajectory, vanishing-point,
combined PDI, PDI grade, background tracking, and target-depth availability do
not participate in the deformation result.

## Failure Diagnosis

The historical link-7 failures are primarily CoTracker visibility failures, not
large coordinate jumps:

- All 34 historical cases with `link7 rigidity == 1.0` were inspected.
- 29 removed all `100/100` link-7 queries; the remaining five removed between
  `96/100` and `98/100`.
- 33/34 were visibility-only failures. One case also had one jump rejection.
- Final link-7 support was only two tracks in 32 cases, three in one case, and
  four in one case.
- In preserved `COSMOS2.5_0004` and `COSMOS2.5_0006` artifacts, the link-7 SAM
  mask exists in all 93 frames and keeps at least 69-73% of its initial area.
- The best predicted coordinates remain inside the link-7 SAM mask for
  91-100% of frames, while CoTracker marks them visible in only 7-13 of 93
  frames.

The evidence therefore does not support jump filtering or SAM disappearance as
the main cause. Blur may contribute, but the archived outputs cannot establish
blur as the sole cause. The failure mechanism is consistent with frame-0 query
appearance becoming stale after gripper pose change, articulation, or partial
occlusion. Historical tracking never re-seeds from later SAM masks, so the
offline CoTracker visibility head eventually rejects otherwise plausible
coordinates.

Switching from offline to online CoTracker without re-seeding would not address
that failure. V2 deliberately retains the local offline CoTracker3 checkpoint
`scaled_offline.pth` and changes query lifetime instead.

## V2 Algorithm

1. Divide the video into 32-frame windows with 8-frame overlap. The stride is
   24 frames and overlap is 25%. The earlier 16-frame overlap proposal was
   rejected because it duplicated too much query and transformer work.
2. For every object and window, choose a central frame with strong SAM area and
   high image sharpness.
3. Erode the selected SAM mask by three pixels when enough interior area
   remains, then sample 32 spatially balanced queries.
4. Run those query groups through the offline CoTracker3 exact-group path while
   reusing the video backbone features.
5. Filter each query only over its own window. It must have at least 30% local
   CoTracker visibility, finite coordinates, and no local jump of 120 pixels or
   more. V2 never forces two fallback tracks to survive.
6. Set visibility to false outside the query's window. At rigidity time, also
   reject samples outside the current-frame SAM mask or on invalid MegaSAM
   pointmap values.
7. A window needs at least five retained anchors and three non-degenerate 3D
   pairs. A link result needs rigidity observations on at least 50% of video
   frames.
8. If those requirements fail, the window or link is unavailable. Missing
   evidence is never encoded as rigidity `1.0`.

The scalar result remains the mean observed MAD/median dispersion of 3D
pair-distance ratios. JSON stores both the raw value and
`deformation_score_percent = 100 * deformation_score`; the percentage is a
display scale, not a probability.

Each successful link also stores two per-frame sequences. The observed
sequence is null on frames without sufficient pair evidence and is the only
sequence used to compute the scalar score. The interpolated sequence fills
those gaps linearly for plotting and replay-adjacent inspection; interpolation
does not change the scalar score. Every run exports both forms to
`per_frame_rigidity.csv`, including an `is_observed` flag and the number of
contributing tracking windows.

## Artifacts And Replay

`cotracker_exact-group.npz` stores both views:

- `tracks`, `visibility`, and `queries` are the V2-filtered, window-scoped data
  used by rigidity and replay rendering.
- `raw_tracks`, `raw_visibility`, `raw_queries`, and `raw_object_offsets` retain
  the unfiltered CoTracker output for diagnosis.
- `metadata_json` records every seed failure, requested/submitted/retained query
  count, local visibility statistics, jump rejections, and window offsets.

Thus a V2 replay renders filtered evidence. Raw tracks remain available in the
same archive but are not silently substituted into scoring or replay.

## Remote Batch Policy

The 90-row deformation-human-label batch runs entirely on the configured GPU
host. Source videos are never downloaded to the Mac. Remote batch videos are
hard links to the verified existing corpus, so they consume no duplicate video
data. The batch renders a replay for every video and publishes stable aliases
such as:

```text
/root/autodl-tmp/pdi/batches/deformation-human-label-v2/outputs/COSMOS2.5_0000/
  replay.mp4
  replay.png
  per_frame_rigidity.csv
```

The batch launcher uses one tmux session and exactly two parallel workers by
default. Results remain remote unless the fetch script is explicitly run;
replay downloads are additionally opt-in with `PDI_FETCH_REPLAYS=1`.

```bash
bash scripts/stage_remote_deformation_v2_batch.sh
PDI_BATCH_WORKERS=2 bash scripts/run_remote_deformation_v2_batch.sh
bash scripts/status_remote_deformation_v2_batch.sh
```

## Parallel Scene Namespace Fix

On September 7, 2026, one of the 90 remote jobs failed before geometry inference.
MegaSAM derived its scene name with the text before the first period, so two
parallel `COSMOS2.5_*` jobs both used `COSMOS2` and raced while clearing the
same frame directory. This was an orchestration namespace collision, not a SAM,
CoTracker, depth, or rigidity failure.

V2 now replaces periods with underscores only in the temporary video alias used
by MegaSAM, producing unique scene names such as `COSMOS2_5_0002-*`. Dataset
labels, workbook identities, hashes, stable output aliases, and metric semantics
remain unchanged. Historical pipeline code remains untouched.

## Run Command

From `PDI-Bench-edited/`:

```bash
PYTHONPATH=src python -m pdi_eval.experiment score --version v2 \
  --config configs/deformation_v2.yaml \
  --input /path/video.mp4 \
  --segmentation-npz /path/segmentation.npz \
  --output-dir /path/output-v2 \
  --geometry-cache-dir /path/megasam-cache \
  --tracker-checkpoint checkpoints/tracker/scaled_offline.pth
```

Use `--disable-replay` for metric-only batch execution. The historical runner
remains callable, but its results must be labeled historical and must not be
mixed with V2 scores in one correlation cohort.

With replay enabled, each run also writes `replay/interactive_exact-group/`.
The precise artifact and display requirements are in
[the V2 rigidity replay contract](rigidity_replay_contract.md).
Open `index.html` there to choose a scored link. Each link page pairs a
rotatable MegaSAM point cloud and the original video on one frame control.
The lines connect the exact CoTracker track pairs selected by the scorer in
each window. Endpoint and line colors show each pair's distance-ratio
departure from that frame's median. The lower chart shows observed frame
rigidity, and the page gives the final mean rigidity score. The directory
contains a local Plotly script and a copy of the source video, so it can be
opened offline. Failed links have no scored-pair replay page.

The scorer now records `selected_pairs` and `pair_frames` inside each complete
window of `metrics.json`. Older metrics files do not contain this evidence;
rerun V2 scoring to make exact-pair replays. The exporter can also be run
separately from `PDI-Bench-edited/`:

```bash
PYTHONPATH=src python -m pdi_eval.utils.rigidity_replay \
  --metrics /path/output-v2/metrics.json \
  --tracks /path/output-v2/cotracker_exact-group.npz \
  --segmentation /path/output-v2/segmentation.npz \
  --pointmaps /path/megasam-cache/scene.npz \
  --video /path/video.mp4 \
  --output-dir /path/output-v2/replay/interactive_exact-group
```

## Required GPU Validation

Before replacing published benchmark values, rerun at least
`COSMOS3_0003`, `COSMOS2.5_0004`, and `COSMOS2.5_0006` in the configured GPU
environment. Compare seedable windows, retained tracks, SAM/pointmap-supported
visibility, temporal coverage, and rigidity availability. V2 is an implemented
method change, not evidence by itself that AUROC improves.
