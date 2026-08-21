# PDI Benchmark Project

## Objective

Maintain one native video-level PDI pipeline for the seven Franka FER links:

```text
Google Drive input
  -> Mac staging
  -> GPU-local CAD-guided SAM3
  -> one shared MegaSAM reconstruction
  -> joint-query and/or exact-group CoTracker
  -> seven isolated PDI metric reports
  -> Mac-local results
```

The GPU is compute-only. Google Drive remains input storage and the Mac remains
the source of truth for code, experiment metadata, and retrieved results.

## Code Ownership

`PDI-Bench-edited/` is the active benchmark source and the only benchmark
checkout that may be edited. Implement native pipeline and metric behavior
directly there; do not create a parallel wrapper or adapter implementation.

`PDI-Bench-original/` is a pinned, pristine, read-only Git submodule of upstream
PDI-Bench. Use it only as the legacy reference for A/B/C/D verification. Never
patch it, run formatters over it, commit from it, or deploy it as the active
pipeline. If the baseline must change, update the submodule pointer to a
recorded upstream commit rather than modifying its working tree locally.

Native behavior belongs under:

```text
PDI-Bench-edited/src/pdi_eval/
PDI-Bench-edited/evaluation/
PDI-Bench-edited/configs/
PDI-Bench-edited/assets/
```

Mac/GPU transfer and environment setup belong under `scripts/`. Shell scripts
must orchestrate the native benchmark CLI; they must not contain alternate
metric implementations or model wrappers.

The old `scripts/adapters/` architecture is retired and deleted. Do not restore
manual segmentation, RobotSeg, DINO localization, SAM2 segmentation, or
single-link fan-out as fallback paths.

## Canonical Dataflow

### Segmentation

Use the seven pinned FER visual meshes `link1.dae` through `link7.dae` as CAD
references. CAD-guided SAM3 runs once per video and writes:

```text
object_masks: bool[T, N, H, W]
object_names: str[N]
object_ids:   int[N]
masks:        bool[T, H, W]  # union, for background exclusion/replay only
```

`object_masks` is the canonical metric input. Never collapse it into a label map
that loses overlaps at articulated joints.

### Shared geometry

Run Depth Anything, UniDepth, DROID, RAFT/CVD, and MegaSAM pointmap lifting once
on the original full-frame video. Never run depth on object crops, masked RGB
frames, or stitched link images.

The shared geometry cache must be versioned by video content, reconstruction
code/settings, and model/checkpoint identities. Every link is lifted through the
same world-coordinate pointmaps and camera poses.

CoTracker archives use source-video pixel coordinates. Before rigidity samples
those tracks from a lower-resolution pointmap, map coordinates into the
pointmap grid with endpoint-preserving scaling. Keep VP calculations in source
video coordinates.

### CoTracker modes

Both modes consume the same deterministic foreground and union-background query
manifest.

`joint-query`:

```text
all link foreground queries + one shared background query set
  -> one CoTracker predictor call
  -> split by link identity
```

This is the fastest mode. CoTracker spatial attention allows queries from
different links to influence one another, so it is a distinct versioned method.

`exact-group`:

```text
encode video features once
  -> update link1 queries independently
  -> ...
  -> update linkN queries independently
  -> update shared background independently
```

This mode reuses the video backbone features but preserves isolated query-group
updates. Do not implement exact-group by rerunning the video backbone for every
link. Do not disable CoTracker spatial attention to approximate isolation.
Validate that every replayed query group invokes the same ordered backbone
chunks as the first group; fail rather than reuse a feature tensor for a
different shape, dtype, or device.

When comparing the modes, prepare queries once, run both modes in one process,
and report model time, peak GPU memory, track counts, metric deltas, and grade
changes.

## Metric Semantics

The seven links are seven rigid targets in one shared scene. Compute all PDI
metrics independently by link:

```text
scale(link_i)      <- mask height_i + masked camera depth_i
trajectory(link_i) <- mask_i + shared world pointmaps
rigidity(link_i)   <- tracks_i + visibility_i + mask_i + shared pointmaps
VP(link_i)         <- foreground tracks_i + shared true-background evidence
PDI(link_i)        <- unchanged weighted metric aggregation
```

The union of links may be used only for:

* excluding every robot pixel from background query sampling;
* excluding robot pixels from LSD/ground-plane background evidence;
* combined visualization and replay.

Never compute rigidity pairs across link identities. The articulated union is
not a rigid object and must not receive a whole-arm rigidity score.

Do not silently change formulas in `evaluator/` or `metrics/`. If a mathematical
definition changes, version the method and document the expected score impact.

## Active Entry Points

From the Mac:

```bash
PDI_VIDEO_NAME=0000.mp4 PDI_TRACKING_MODE=both bash scripts/run.sh
```

`scripts/run.sh` delegates to `scripts/run_sam3_cad_video.sh`.

On a prepared compute host, the native benchmark CLI is:

```bash
PYTHONPATH=src python evaluation/run_multi_object.py \
  --config configs/default.yaml \
  --input /path/video.mp4 \
  --segmentation-npz /path/segmentation.npz \
  --output-dir /path/output \
  --geometry-cache-dir /path/cache \
  --tracker-checkpoint /path/scaled_offline.pth \
  --tracking-mode both
```

Valid tracking mode values are `joint-query`, `exact-group`, and `both`.

## System Roles

### Mac

The Mac owns:

* orchestration and benchmark source;
* pinned CAD assets and configs;
* selected Google Drive inputs;
* final metrics, timing comparisons, manifests, and replays.

### Google Drive

Google Drive is read-only input storage through the existing `gdrive` rclone
remote. Preserve its directory structure and do not place GPU credentials there.

### GPU host

Persistent GPU storage may contain environments, weights, source caches, and the
versioned MegaSAM geometry cache. Per-run directories contain staged inputs,
SAM3 segmentation, CoTracker mode archives, output JSON, and replay artifacts.

No result is considered complete until it has been retrieved and verified on
the Mac.

## Results Contract

Each completed multi-object run should contain:

```text
metrics.json
timing.json
manifest.json
run_config.yaml
cotracker_joint-query.npz       # when requested
cotracker_exact-group.npz       # when requested
replay/
  combined_joint-query.mp4      # when requested
  combined_exact-group.mp4      # when requested
console.log                     # Mac orchestration log
```

`metrics.json` must include:

* one complete PDI report per link and tracking mode;
* direct exact-minus-joint deltas for all PDI components;
* grade-change flags;
* shared geometry identity/cache state;
* per-mode tracking time and peak GPU memory;
* shared stage timings.

Track archives must use numeric arrays plus object offsets. Do not use pickled
NumPy object arrays.

## Reliability Rules

* Validate video dimensions and frame counts against segmentation tensors.
* Preserve stable CAD link names and IDs through every output.
* Use nearest-neighbor interpolation for masks.
* Sample background outside a recorded dilation of the union mask.
* Keep at least two usable foreground tracks per link and report retained counts.
* Write caches and JSON atomically.
* Reject all-zero MegaSAM fallback geometry for scored runs.
* Record exact commands, code revision, inputs, mode, cache identity, and timing.
* Keep stage caches resumable so one failure does not require rerunning SAM3 or
  MegaSAM.s

## Testing

Lightweight local tests:

```bash
PYTHONPATH=PDI-Bench-edited/src python3 -m unittest discover \
  -s PDI-Bench-edited/tests -v
```

The local shell may skip PyTorch-dependent tracker tests. In the PDI GPU
environment, the same command must run the joint/exact tracker tests without a
skip.

Before accepting performance claims, run a cold-cache and warm-cache comparison
and record:

* SAM3 time;
* MegaSAM time/cache hit;
* query preparation time;
* CoTracker model time by mode;
* metric time by mode;
* replay time;
* peak GPU memory;
* final artifact size.
