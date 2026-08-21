# Shared Multi-Object PDI Design for Seven Franka Links

> Implementation update (2026-08-20): this architecture is now implemented
> directly inside `PDI-Bench-edited/`. References below to an outer adapter
> describe the pre-implementation state and are superseded by the native modules
> listed in "Native implementation."

## Executive answer

Yes. The seven link evaluations can be reorganized into one video-level run
without combining the links into one rigidity target.

The correct design is:

1. Run CAD-guided SAM3 once and retain seven independently addressable mask
   sequences.
2. Run Depth Anything, UniDepth, DROID, RAFT/CVD, and MegaSAM pointmap lifting
   once on the original full video.
3. Seed CoTracker for all seven links, plus one shared true-background region,
   in one video-level tracking job.
4. Split the resulting tracks by link identity.
5. Run the unchanged PDI metric formulas seven times in a cheap in-memory loop,
   always giving each metric the relevant link mask and link tracks while
   reusing the common camera poses and world pointmaps.

The important rule is: **merge the execution, not the object identities**.
The seven masks may be unioned for background exclusion and visualization, but
the articulated union must not be passed to the rigidity metric as one rigid
body.

This is not only feasible; using a single MegaSAM reconstruction is the most
coherent way to put all seven links in the same 3D coordinate frame.

## What the local pipeline does now

The active code already avoids rerunning SAM3. `sam3_cad_segment.py` constructs
`object_masks` with shape `(T, N, H, W)`, constructs a union mask with
`np.any(object_masks, axis=1)`, and also writes one archive per matched link
([source](PDI-Bench-edited/src/pdi_eval/perception/sam3_cad_segment.py#L185)). The old launcher then
loops over those link archives and invokes the complete single-target PDI runner
once per link ([source](scripts/run_sam3_cad_video.sh#L61)).

For seven matched links, the current effective computation is:

| Stage | Current count per video | Mask dependence | Observation |
|---|---:|---|---|
| CAD reference rendering | 1 | No | Shared already |
| SAM3 proposal generation and propagation | 1 | Produces all masks | Shared already |
| PDI process and model initialization | 7 | Per link | Duplicated |
| CoTracker video forward | 7 | Link mask seeds | Main remaining repeated model stage |
| Background CoTracker queries | 7 sets | Complement of one link | Duplicated and semantically contaminated by other links |
| MegaSAM reconstruction | 1 cold run, then 6 cache hits if cache is valid | Full video; mask used only for target depth | Largely shared by the former cache path |
| PDI metric synthesis | 7 | Per link | Required, but inexpensive |
| Annotated video and reconstruction replay | 7 | Per link | Duplicated decoding/rendering by design |
| GPU transfer/result handling | 7 | Per run | Duplicated orchestration |

### Existing MegaSAM sharing

The upstream `MegaSamWrapper` runs the expensive scene pipeline before it uses
the target mask: frame extraction, Depth Anything, UniDepth, DROID camera
tracking, RAFT/CVD refinement, and world-pointmap construction
([source](PDI-Bench-edited/src/pdi_eval/perception/mega_sam_wrapper.py#L106)).
Only after pointmaps have been built does it mask depth pixels to obtain a
target-specific median depth sequence
([source](PDI-Bench-edited/src/pdi_eval/perception/mega_sam_wrapper.py#L193)).

The former outer implementation exploited this separation. The native implementation now keys a geometry archive
by source-video SHA-256, stores `pointmaps`, `camera_poses`, and `focal_length`,
then derives a new target depth sequence from the cached world pointmaps and the
current link mask
([source](PDI-Bench-edited/src/pdi_eval/perception/mega_sam_wrapper.py#L30)).
`run.sh` places this cache outside individual run directories.

Consequently, the current seven-process implementation should not execute Depth
Anything seven times when the shared cache is present and healthy. It still
pays seven process startups, seven CoTracker forwards, seven per-run caches,
seven visualizations/replays, and seven orchestration cycles.

There is also a cache-integrity weakness: the current shared geometry key
contains only the video hash. It does not include the MegaSAM code revision,
model/checkpoint identities, or reconstruction settings. A future shared runner
should use a versioned cache key and store those identities in metadata so a
model change cannot silently reuse stale geometry.

## Proposed dataflow

```text
seven FER CAD meshes
        |
        v
CAD-guided SAM3
        |
        +--> object_masks[T, N, H, W]  (N <= 7, identity retained)
        +--> union_mask[T, H, W]        (background exclusion only)
        |
full original video
        |
        +--> MegaSAM once
        |      depth + intrinsics + camera poses + world pointmaps[T,H,W,3]
        |
        +--> CoTracker once
               link-1 foreground queries
               ...
               link-N foreground queries
               shared background queries outside union_mask[0]
                        |
                        v
               tracks[T,Q,2] + visibility[T,Q]
               query_object_id[Q] = 0..N-1, or -1 for background
                        |
                        v
              split into N link views
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       link 1 PDI    link 2 PDI    ... link N PDI
       own mask      own mask          own mask
       own tracks    own tracks        own tracks
       shared 3D     shared 3D         shared 3D
       shared BG     shared BG         shared BG
```

Depth Anything should **not** be run on cropped links, masked RGB images, or a
stitched image containing the seven objects. It requires the full-frame scene
context, and MegaSAM needs that context for camera motion and a consistent world
frame. Run it once on the unmodified video, then select each link's 3D samples
from the resulting pointmaps.

## Recommended in-memory representations

The SAM3 archive already contains the essential tensor:

```text
object_masks:       bool[T, N, H, W]
object_names:       str[N]
object_ids:         int[N]
union_mask:         bool[T, H, W] = any(object_masks, axis=1)
```

For downstream use, add either a non-overlapping label map or explicit overlap
metadata:

```text
label_map:          uint8[T, H, W]     # 0=background, 1..N=link
overlap_map:        uint8[T, H, W]     # number of masks claiming each pixel
```

SAM masks can overlap near joints. A plain label map loses that information, so
`object_masks` must remain the canonical metric input. The label map is useful
for display and fast indexing only. Overlap resolution should be deterministic
and recorded if it is ever used for query ownership.

Store tracking output once:

```text
tracks:             float32[T, Q, 2]
visibility:         bool[T, Q]
query_object_id:    int16[Q]           # -1 background, 0..N-1 links
query_source:       int8[Q]            # SIFT, Shi-Tomasi, grid, support
query_xy_frame0:    float32[Q, 2]
```

This allows a per-link view to be created without copying the video or model
outputs:

```python
selector = query_object_id == link_index
link_tracks = tracks[:, selector]
link_visibility = visibility[:, selector]
```

## Why each PDI metric remains independent

### 1. Scale consistency

PDI evaluates the constancy of `log(pixel_height) + log(depth)`
([source](PDI-Bench-edited/src/pdi_eval/evaluator/scale_audit.py#L4)). For link
`i`, retain:

```text
h_i[t] = pixel height of object_masks[t, i]
z_i[t] = median positive camera-Z under object_masks[t, i]
```

The world pointmap and camera pose are shared, but mask selection is per link.
The native MegaSAM wrapper implements the world-to-camera conversion and
per-mask median operation
([source](PDI-Bench-edited/src/pdi_eval/perception/mega_sam_wrapper.py#L30)). Therefore sharing geometry
does not mix scale measurements between links.

### 2. 3D trajectory consistency

The current trajectory audit takes the median world-space point beneath the
foreground mask on each frame, then evaluates velocity and acceleration
smoothness ([source](PDI-Bench-edited/src/pdi_eval/evaluator/motion_audit.py#L13)).
For every link, compute:

```text
c_i[t] = median(pointmaps[t][object_masks[t, i]])
```

All `c_i` sequences then inhabit the same MegaSAM world coordinate system. This
is preferable to separate reconstruction runs, whose gauges and numerical
solutions could differ.

### 3. Rigidity / volume stability

The primary rigidity strategy samples each foreground CoTracker point from the
shared world pointmap and measures invariance of pairwise 3D distances
([source](PDI-Bench-edited/src/pdi_eval/evaluator/volume_audit.py#L7)). It also
uses the target mask's distance transform to prefer points away from boundaries.

For link `i`, call the existing audit with only:

```text
pointmaps_shared
tracks_i
visibility_i
object_masks[:, i]
```

Do not allow a pair to contain points from different links. Adjacent links move
relative to each other at joints, so cross-link pairs would measure articulation
as non-rigidity and invalidate the score. The fallback point-cloud extent metric
must likewise index the individual link mask, not the union.

### 4. Vanishing-point coupling

The PDI pipeline estimates a foreground VP from target tracks and a background
VP from background tracks plus line segments outside the target mask
([source](PDI-Bench-edited/src/pdi_eval/pipeline.py#L118)). The proposed split
is:

```text
foreground VP for link i: tracks_i
background VP:             one shared track set seeded outside union_mask[0]
LSD background exclusion:  union_mask for the first three frames
```

The foreground VP and `epsilon_vp` remain per link. The background VP can be
computed once and reused, although `global_vp` should still be synthesized per
link because it merges that link's foreground lines with background lines.

This changes background semantics relative to the present seven-run pipeline.
Today, the complement of `link1` includes links 2-7, so moving robot pixels may
be sampled as "background." Excluding the union of every robot-link mask is more
correct, but it can cause small metric differences. Treat that as an intentional
method revision and record it in the output manifest.

### 5. PDI aggregation

The final score is only a weighted combination of four per-target values:
scale, trajectory, rigidity, and VP coupling
([source](PDI-Bench-edited/src/pdi_eval/metrics/pdi_index.py#L42)). Once the
shared inference products have been sliced into per-link inputs, the existing
calculator can be called unchanged for every link. One run can therefore emit
seven complete reports with the same schema and weighting.

### 6. Reconstruction audit and replay

The reconstruction itself is scene-level and should be cached once. The
reconstruction audit is mixed:

- ground flatness is scene-level and should exclude the union of all robot masks;
- scale jump is link-specific because it measures foreground median depth;
- a whole-scene 3D render can be generated once;
- colored link overlays or link anchor replays may be rendered per link only
  when they are needed.

Replays should reference the same pointmap archive instead of embedding seven
copies. A default combined replay can color the seven link masks/tracks
differently. Optional per-link replays can be generated lazily from that shared
bundle.

## CoTracker: the main design caveat

The current wrapper uses 100 requested foreground points and 225 requested
background points per link, and sends both groups through one CoTracker call
([source](PDI-Bench-edited/src/pdi_eval/perception/track_wrapper.py#L39)). Across
seven runs, that is up to 700 foreground queries and 1,575 repeated background
queries, or 2,275 explicit queries total. A joint run needs at most 700
foreground queries plus 225 shared background queries, or 925 explicit queries.
It also performs the video feature extraction once instead of seven times.

CoTracker's public predictor accepts arbitrary queries with shape `(B, N, 3)`,
so concatenating the seven foreground query groups and the background group is
supported. However, joint tracking is not guaranteed to be numerically identical
to concatenating seven independent results. CoTracker3's update transformer uses
spatial attention between queried points; changing the query set can change an
individual point's predicted track.

There are therefore two implementation modes:

| Mode | Behavior | Compute benefit | Comparability |
|---|---|---|---|
| Joint-query mode | One CoTracker call with all links and one background set | Highest; simple and public-API compatible | Must be A/B validated because query groups interact |
| Exact-group mode | Encode video features once, then update each link query group independently | Retains independent-query semantics | Harder; stock predictor does not expose a reusable feature-cache API |

Joint-query mode is the practical first implementation. It is also conceptually
reasonable for a model named CoTracker, but it should be treated as a new
versioned evaluation method until validation shows acceptable agreement.

If 925 queries exceed memory limits, do not return to seven complete video
passes immediately. First reduce or allocate points adaptively:

```text
link budget_i = clamp(round(total_fg_budget * area_i / total_link_area), min_i, max_i)
```

Keep a minimum number of well-spaced interior points per link so the 3D rigidity
audit still has enough valid pairs. Query chunking is another option, but simple
chunking through the stock predictor recomputes video features and loses much of
the benefit. True chunking requires a reusable CoTracker feature cache.

## Expected computational effect

No trustworthy wall-clock speedup can be claimed without timing the actual GPU
and video length. The structural savings are nevertheless clear:

| Resource | Seven-run path | Shared multi-object path |
|---|---:|---:|
| SAM3 video inference | 1 | 1 |
| Depth Anything / MegaSAM cold reconstruction | 1 with a healthy existing cache | 1 |
| CoTracker model loads | 7 | 1 |
| CoTracker video feature extraction | 7 | 1 |
| Explicit background queries | up to 1,575 | up to 225 |
| PDI metric calls | 7 | 7, inexpensive and in memory |
| Stored MegaSAM pointmap archives | potentially referenced/copied per run | 1 canonical archive |
| Video decode for reporting/replay | repeated up to 7 times | 1 combined pass by default |
| Remote run setup/transfer/retrieval | 7 | 1 |

Because MegaSAM is already cached, the largest new model-level gain should come
from CoTracker rather than Depth Anything. On a cold or invalidated cache, both
designs still need exactly one MegaSAM reconstruction. The end-to-end gain will
depend on how much time is currently spent in CoTracker versus replay, transfer,
compression, and process initialization.

## Native implementation

The implementation now lives directly in the benchmark:

```text
PDI-Bench-edited/evaluation/run_multi_object.py
PDI-Bench-edited/src/pdi_eval/multi_object_pipeline.py
PDI-Bench-edited/src/pdi_eval/perception/track_wrapper.py
```

Implemented responsibilities:

### `evaluation/run_multi_object.py`

- validate the full SAM3 archive and video identity;
- load all link masks in canonical `object_names` order;
- acquire or build one versioned MegaSAM geometry cache;
- run the multi-object tracker once;
- call the native metric pipeline for each link;
- write one manifest and one result bundle;
- create combined mode-specific reconstruction replays when enabled.

### `perception/track_wrapper.py`

- sample SIFT/Shi-Tomasi/grid foreground queries independently inside each
  link's frame-0 mask;
- sample background queries outside a dilated union mask;
- concatenate queries for joint-query mode or isolate them for exact-group mode;
- replay one cached backbone feature pass in exact-group mode;
- filter tracks per group, not globally;
- persist all tracks and group metadata in one archive.

Dilating the union mask before background sampling is advisable because depth
and segmentation errors concentrate near robot boundaries. The dilation radius
must be a recorded setting.

### `multi_object_pipeline.py`

- derive `h_pixel_i`, `x_center_i`, truncation, and normalized camera depth per
  link;
- reuse upstream audit functions without changing formulas;
- compute foreground VP and global VP per link using the shared background
  evidence;
- call `PDIIndexCalculator` once per link;
- return a dictionary keyed by stable CAD link name.

Implemented bundle layout:

```text
results/sam3-cad-multi/<video>/<manifest-hash>/
|-- manifest.json
|-- metrics.json                 # {"objects": {"link1": ..., ...}}
|-- segmentation.npz            # object_masks + identities
|-- timing.json
|-- cotracker_joint-query.npz
|-- cotracker_exact-group.npz
`-- replay/
    |-- combined_joint-query.mp4
    `-- combined_exact-group.mp4
```

To reduce storage duplication further, `geometry.npz` should live in the
versioned content-addressed cache and the result manifest should reference its
hash. Copy it into a result directory only when a self-contained export is
explicitly required.

## Validation plan

Do not replace the seven-run path based only on shape checks. Validate metric
behavior and compute behavior separately.

### Phase 1: deterministic fixture tests

1. Verify object ordering, link identity, overlap accounting, and union masks.
2. Verify every foreground query lies inside its assigned link mask.
3. Verify every shared-background query lies outside the dilated union mask.
4. Verify splitting a synthetic combined track tensor exactly reconstructs all
   per-link tensors.
5. Run each metric on synthetic shared pointmaps and compare it with the same
   metric called separately for each link.
6. Verify no rigidity pair crosses a `query_object_id` boundary.
7. Verify cache keys change when the video, MegaSAM revision, checkpoint, or
   relevant settings change.

### Phase 2: seven-run versus shared-run A/B test

Use at least one static-camera clip, one moving-camera clip, one occlusion-heavy
clip, and one clip with substantial joint motion. Record:

- SAM3 mask identity and area per link;
- foreground query coordinates;
- CoTracker trajectory endpoint error and visibility agreement per link;
- `epsilon_scale`, `epsilon_trajectory`, `epsilon_rigidity`, and `epsilon_vp`;
- final PDI score and grade;
- wall time for each stage;
- peak GPU memory;
- result artifact size.

Scale and 3D trajectory values should be identical up to floating-point and
serialization effects because they use the same masks and geometry. Rigidity
and foreground VP may differ because joint-query CoTracker can alter the tracks.
Background VP may also differ intentionally because the new background excludes
all robot links.

Define acceptance thresholds before examining the result. For example, report
absolute metric deltas and grade changes rather than claiming equivalence from
a visually similar replay. If exact benchmark comparability is mandatory and
joint queries exceed the accepted delta, pursue exact-group feature reuse rather
than silently accepting changed scores.

### Phase 3: performance acceptance

Instrument stage-level timers around:

```text
SAM3
MegaSAM cold/cache load
CoTracker decode/query sampling/model/filtering
per-link metrics
visualization/replay
transfer/compression
```

Measure both a cold cache and a warm cache. A warm-only test would hide the real
MegaSAM cost, while a cold-only test would understate the geometry cache's
existing cache benefit.

## Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Joint CoTracker spatial attention changes tracks | Rigidity/VP deltas versus old runs | Version method; A/B test; exact-group mode if required |
| 925-query peak memory is too high | OOM or slower attention | Adaptive per-link budgets; profile; then feature-cached chunks |
| Link masks overlap at joints | Ambiguous query ownership | Keep boolean object masks canonical; deterministic ownership for sampled points |
| Small/thin links lack enough features | Rigidity fallback or unstable score | Minimum interior grid budget; mask erosion/distance-transform sampling; report point counts |
| Other links treated as background | Biased background VP in current path | Seed outside union mask and use union for LSD exclusion |
| Mask boundary depth bleeding | Noisy centroids and rigidity | Nearest-neighbor resize; optional erosion for depth sampling; retain original mask for reported size |
| Stale SHA-only MegaSAM cache | Wrong geometry reused after model changes | Versioned cache key and metadata validation |
| One combined failure loses all reports | Larger failure domain | Atomic stage caches; resume from segmentation, geometry, or tracking archive |
| Whole-arm rigidity accidentally reported | Articulation scored as deformation | Enforce group IDs and reject union rigidity in code/tests |

## Recommended decision

Use the implemented single video-level, multi-object pipeline. Run one full-frame MegaSAM
reconstruction and one shared background definition. Keep seven per-link masks,
track groups, and PDI reports. Start with joint-query CoTracker because the public
API supports it and it removes the largest remaining repeated inference, but
label the result as a versioned multi-object method until the A/B validation is
complete.

In short:

```text
one video reconstruction
+ one multi-object tracking job
+ seven cheap, isolated metric evaluations
= seven valid link-level PDI reports in one shared 3D space
```
