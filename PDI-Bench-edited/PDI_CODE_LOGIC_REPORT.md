# PDI-Bench-edited: CoTracker, Rigidity, and Missing-Mask Logic

## Scope

This report describes the current native multi-object path:

```text
SAM3 segmentation archive
  -> MultiObjectPDIEvaluationPipeline.run()
  -> TrackWrapper.prepare_multi() / track_prepared()
  -> evaluate_object_metrics()
  -> audit_3d_volume_stability()
  -> PDIIndexCalculator.compute_pdi()
```

The entry point is [`evaluation/run_multi_object.py`](evaluation/run_multi_object.py#L86), and the orchestration is in [`multi_object_pipeline.py`](src/pdi_eval/multi_object_pipeline.py#L402). The older single-object path in `pipeline.py` is not the active Franka workflow.

The current automated batch evaluates `link2` through `link7`; `link1` is retired from the DINOv2-to-SAM3 frontend ([`sam3_dinov2_segment.py`](src/pdi_eval/perception/sam3_dinov2_segment.py#L28), [`export_batch_metrics_csv.py`](evaluation/export_batch_metrics_csv.py#L13)).

## Executive summary

1. **CoTracker points are initialized only once, from frame 0.** Each link's frame-0 SAM mask is resized to the CoTracker image. SIFT, Shi-Tomasi, and deterministic grid candidates inside that mask are pooled, deduplicated, and reduced with deterministic farthest-point sampling. Explicit queries have the form `[0, x, y]`; CoTracker's own grid generator is disabled with `grid_size=0`.
2. **The active rigidity score fuses CoTracker correspondences with MegaSAM 3D pointmaps.** At every tracked 2D coordinate, the code samples one world-space 3D point. It chooses up to 30 reliable, wide-baseline point pairs on frame 0. At each late
r frame it computes every pair's distance ratio relative to frame 0, then scores the dispersion of those ratios as `MAD(ratios) / median(ratios)`. The final rigidity scalar is the mean of the per-frame values, excluding frame 0.
3. **A failed SAM mask is not treated uniformly across PDI.** A missing link mask is stored as all false. Target depth is interpolated only if at least 80% of frames still have valid masked depth; otherwise that link gets `status: failed` and no PDI score. Scale receives a zero pixel height and is therefore strongly penalized. Trajectory holds the previous 3D centroid. Active 3D rigidity does not use later SAM masks at all: it follows CoTracker points and uses CoTracker visibility, carrying the previous rigidity value when fewer than three pairs are visible.

## 1. Exact CoTracker point initialization

### 1.1 Which frame and masks are used

The multi-object pipeline passes exactly this mask slice to CoTracker preparation:

```python
segmentation.object_masks[0]  # shape (number_of_links, H, W)
```

See [`multi_object_pipeline.py`](src/pdi_eval/multi_object_pipeline.py#L447). No later SAM mask is used to initialize, add, remove, or re-seed CoTracker points.

The video is decoded and each frame is downscaled only when its largest dimension is greater than `max_dimension` (default `880`). The aspect ratio is preserved. Every frame-0 link mask is resized from the original video/mask resolution to the CoTracker resolution with nearest-neighbor interpolation ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L226)).

The first resized RGB frame is converted to grayscale. All point detection is performed on that grayscale frame and constrained by each resized binary link mask.

### 1.2 Requested point counts

The default count is `grid_size ** 2 = 10 ** 2 = 100` points per link. The current configuration overrides three links ([`configs/default.yaml`](configs/default.yaml#L24)):

| Link | Requested queries |
|---|---:|
| `link2` | 256 |
| `link3` | 192 |
| `link4` | 128 |
| `link5` | 100 |
| `link6` | 100 |
| `link7` | 100 |

These are budgets, not guaranteed output counts. A small or feature-poor mask can yield fewer unique points, but each link must yield at least two or preparation raises `ValueError` and the entire run stops. Unknown link names in the override map also raise an error ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L152)).

### 1.3 Candidate generation inside each link mask

For a requested count `C`, `_sample_region_queries()` creates three candidate groups ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L124)):

1. **SIFT:** ask for up to `4C` accepted points inside the mask. Internally OpenCV SIFT is created with `nfeatures=16C`, detections are sorted by descending response, and only keypoints whose rounded pixel lies on mask value `1` are retained ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L629)).
2. **Shi-Tomasi:** ask for up to `4C` corners inside the mask, using `qualityLevel=0.01` and `minDistance=5` pixels ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L657)).
3. **Deterministic spatial grid:** ask for `C` points. The mask bounding box is divided into `ceil(sqrt(C))` rows and columns. Each occupied cell contributes the foreground pixel nearest its cell center. If fewer than `C` cells contribute, remaining mask pixels are filled at evenly spaced indices ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L682)).

Despite the older docstring calling this a fallback hierarchy, the current multi-object method **always runs and pools all three groups** in this order: SIFT, Shi-Tomasi, grid.

### 1.4 Deduplication and spatial balancing

The pooled candidates are deduplicated by `(x, y)` rounded to three decimal places, preserving first occurrence. Because SIFT candidates are stacked first, an exact duplicate keeps the SIFT version ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L167)).

If at most `C` unique candidates remain, all are used. Otherwise selection is deterministic farthest-point sampling:

```text
selected = [first pooled candidate]
repeat until C points are selected:
    for every unselected point q:
        score(q) = minimum squared 2D distance from q to any selected point
    select argmax(score)
```

Thus the first seed is normally the strongest retained SIFT feature, and subsequent points maximize spatial coverage. Ties resolve by NumPy's first `argmax`, so this is deterministic for fixed OpenCV outputs.

Every explicit query is stored in CoTracker format:

```text
[query_frame, x, y] = [0.0, x, y]
```

Coordinates at this stage are in the downscaled CoTracker image.

### 1.5 Background initialization

The first-frame masks of all links are unioned. With the default `background_dilation=5`, the union is dilated using an `11 x 11` elliptical kernel. The background region is the complement of that dilated union ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L269)).

The background uses the same SIFT + Shi-Tomasi + grid pooling and spatial-balancing logic, with a requested budget of `background_grid_size ** 2 = 15 ** 2 = 225` points.

### 1.6 CoTracker call and the two tracking modes

Queries are passed explicitly as a tensor of shape `(1, N, 3)`. The predictor call is ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L302)):

```python
tracks, visibility = model(
    video_tensor.float(),
    queries=queries,
    grid_size=0,
    grid_query_frame=0,
)
```

`grid_size=0` is important: CoTracker does **not** initialize an internal regular grid. It uses only the explicit points described above.

Both modes use the same prepared query arrays:

- `joint-query` concatenates all link groups and the background, runs one predictor call, then splits the output by the original counts.
- `exact-group` runs each link and background as a separate query group. It caches and replays the video backbone (`model.fnet`) so the image features are computed once, while the query-update computation remains isolated by link ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L356)).

### 1.7 Post-tracking filtering

Tracks and query coordinates are first scaled back to the original video/mask resolution. A track is normally retained only if all of the following hold across the whole clip ([`track_wrapper.py`](src/pdi_eval/perception/track_wrapper.py#L772)):

- all track coordinates and visibility values are finite;
- mean CoTracker visibility is at least `0.3`;
- its maximum one-frame displacement is strictly below `120` pixels in original-video coordinates.

If fewer than two tracks pass, the code keeps the two best finite tracks, ranked by higher mean visibility and then lower maximum jump. If fewer than two finite tracks exist, it raises an error. The saved query manifest is filtered with the same selector, so saved queries remain aligned with saved tracks.

## 2. Exact rigidity calculation

### 2.1 Active call path

For each link, `evaluate_object_metrics()` maps that link's CoTracker tracks from original-video coordinates to the MegaSAM pointmap grid, preserving image endpoints ([`multi_object_pipeline.py`](src/pdi_eval/multi_object_pipeline.py#L103)):

```text
u_pointmap = u_video * (W_pointmap - 1) / (W_video - 1)
v_pointmap = v_video * (H_pointmap - 1) / (H_video - 1)
```

It then calls `audit_3d_volume_stability(pointmaps, link_masks, tracks, h_seq, visibility)` ([`multi_object_pipeline.py`](src/pdi_eval/multi_object_pipeline.py#L180)). In a normal multi-object run, pointmaps, tracks, and visibility are all supplied and frame-0 geometry is valid, so **Strategy 1: 3D rigid pairwise ratios** is used ([`volume_audit.py`](src/pdi_eval/evaluator/volume_audit.py#L209)).

### 2.2 Turning CoTracker points into 3D trajectories

MegaSAM provides `pointmaps[t, v, u]`, a world-coordinate XYZ point for every pointmap pixel. For every frame `t` and every CoTracker anchor `n`, the code rounds and clips the mapped 2D track coordinate and samples one XYZ value ([`volume_audit.py`](src/pdi_eval/evaluator/volume_audit.py#L37)):

```text
u_tn = clip(round(track_x[t,n]), 0, W-1)
v_tn = clip(round(track_y[t,n]), 0, H-1)
P_tn = pointmaps[t, v_tn, u_tn]
```

So CoTracker supplies temporal correspondence; MegaSAM supplies the 3D location at each correspondence. The metric is not computed from the whole masked cloud, and it is not computed from 2D tracks alone in the normal path.

### 2.3 Frame-0 anchor filtering

Anchor reliability is decided on frame 0 only:

1. Start with anchors whose CoTracker `visibility[0] > 0.5`.
2. Compute the Sobel gradient magnitude of the frame-0 pointmap Z channel.
3. If more than four anchors are visible, set the gradient threshold to the 75th percentile of their sampled gradients and prefer visible anchors strictly below that threshold.
4. If this leaves fewer than five anchors, relax back to all frame-0-visible anchors.
5. If fewer than five remain even after relaxation, return the maximum failure value `1.0` with a history full of ones.

The frame-0 SAM mask is used to compute an L2 distance transform: an anchor deep inside the mask gets a large boundary distance, while an anchor outside or on the edge gets a small value. There is no additional hard `mask[anchor] == true` test at this stage; mask interior distance affects pair ranking instead.

### 2.4 Selecting up to 30 anchor pairs

For every candidate pair `(i, j)`, compute its frame-0 3D baseline and ranking score:

```text
d_ij(0) = ||P_0i - P_0j||_2
pair_quality_ij = d_ij(0) * min(edge_distance_i, edge_distance_j)
```

All unique pairs are sorted by descending `pair_quality`, and the best 30 are retained. This favors long 3D baselines, which improve deformation signal-to-noise, and points far from the SAM boundary, which are less likely to contain depth bleeding ([`volume_audit.py`](src/pdi_eval/evaluator/volume_audit.py#L85)).

Pairs with `d_ij(0) <= 1e-3` are dropped. If fewer than three pairs remain, the function again returns `1.0` for every frame.

### 2.5 Per-frame rigidity formula

For each later frame `t`, retain only pairs for which both CoTracker endpoints have visibility greater than `0.5`:

```text
d_ij(t) = ||P_ti - P_tj||_2
r_ij(t) = d_ij(t) / d_ij(0)
m_t     = median over visible pairs of r_ij(t)
MAD_t   = median over visible pairs of |r_ij(t) - m_t|
epsilon_rigidity(t) = MAD_t / (m_t + 1e-6)
```

If fewer than three pairs are visible, the frame is not independently evaluated; `epsilon_rigidity(t)` is copied from the previous frame. Frame 0 is assigned `0.0` as the perfect reference.

The final scalar is:

```text
epsilon_rigidity = mean(epsilon_rigidity(t) for t = 1 ... T-1)
```

Frame 0 is deliberately excluded from this mean ([`volume_audit.py`](src/pdi_eval/evaluator/volume_audit.py#L122)).

### 2.6 What the formula does and does not measure

The implemented quantity measures **coherence across pairwise scale ratios at each frame**, not absolute preservation of every distance. If every pair doubles by the same factor, all ratios are `2`, their MAD is zero, and the rigidity penalty is zero. This makes the rigidity term tolerant of a global reconstruction-scale change; such a change is expected to be caught by the separate scale term. Nonuniform stretching produces a spread of ratios and a positive rigidity penalty.

There is no Kabsch alignment, rigid transform fitting, Procrustes residual, or direct point-cloud registration in this calculation.

### 2.7 Fallback strategies

`audit_3d_volume_stability()` defines two fallbacks, though they are normally unreachable in the active multi-object path when valid pointmaps and CoTracker output exist:

- **Strategy 2, point-cloud extent:** when pointmaps exist but tracks/visibility do not, take the masked world-Y extent `percentile_95(Y) - percentile_5(Y)` per frame, forward-fill an empty-mask frame, and return `std(extent) / mean(extent)`.
- **Strategy 3, 2D CoTracker:** sample up to 30 unique random anchor pairs with NumPy seed `42`; calculate `d_ij(t)/d_ij(0)` in 2D; score each frame as `std(ratios)/(mean(ratios)+1e-6)`. Its history starts with `1.0`, and the returned mean includes that frame-0 value ([`volume_audit.py`](src/pdi_eval/evaluator/volume_audit.py#L149)).

The strategy selector returns immediately from Strategy 1 even when Strategy 1 returns the hard failure score `1.0`; it does not retry Strategy 2 or 3 in that case.

## 3. Exact handling of SAM failures on individual links

### 3.1 What counts as a failed mask frame

The archive has shape `(T, links, H, W)`. If SAM3 returns a frame but no acceptable mask is associated with a link, that link's preallocated mask remains all false.

In the DINOv2/SAM3 frontend, an output can be rejected when it is empty or when its association to the last good mask is below `0.10`. Association is `0.75 * IoU + 0.25 * area_consistency`; matching the preferred SAM object ID adds `0.10` only for ranking and does not lower the `0.10` acceptance threshold ([`sam3_dinov2_segment.py`](src/pdi_eval/perception/sam3_dinov2_segment.py#L127)). On rejection, the previous good mask is retained only as the reference for a possible later reassociation; it is **not copied into the failed frame** ([`sam3_dinov2_segment.py`](src/pdi_eval/perception/sam3_dinov2_segment.py#L398)).

There are two distinct failure cases:

- If SAM3 omits the frame response entirely, segmentation raises and no PDI run occurs.
- If the frame response exists but that link has no accepted mask, the archive contains an all-false mask for that link and frame.

The DINOv2/SAM3 CLI normally requires masks on at least 80% of frames. However, the current remote exact-group batch explicitly passes `--minimum-tracked-fraction 0.0`, disabling this segmentation-stage rejection ([`run_remote_exact_group_batch.py`](evaluation/run_remote_exact_group_batch.py#L195)). Therefore the downstream PDI depth gate described below is the effective per-link validity gate in that batch.

The older CAD/SAM3 frontend behaves similarly for per-link absence: it aborts if a whole frame response is missing, but writes zeros for a selected object ID absent from a returned frame ([`sam3_cad_segment.py`](src/pdi_eval/perception/sam3_cad_segment.py#L160)). It has no tracked-fraction check.

### 3.2 Measurements derived from an empty link mask

When the archive is loaded, stored aggregate `h_pixel` values are ignored and measurements are recomputed independently for each link ([`segmentation_archive.py`](src/pdi_eval/perception/segmentation_archive.py#L92)).

For a mask with at most 10 pixels, including an empty mask:

- `h_pixel[t] = 0`;
- `x_center[t] = x_center[t-1]`, or `0` at frame 0;
- `is_truncated[t] = True` because its initialized value is never changed.

There is no interpolation of the SAM mask itself and no interpolation of `h_pixel`.

### 3.3 Per-link target-depth gate

For each frame, the shared world pointmap is transformed back into that frame's camera coordinates. A target depth is valid only where all three conditions hold ([`mega_sam_wrapper.py`](src/pdi_eval/perception/mega_sam_wrapper.py#L39)):

```text
link mask is true
AND camera-space Z is finite
AND camera-space Z > 0
```

The frame's raw target depth is the median of valid Z pixels. An empty link mask therefore produces a missing depth (`NaN`).

For `T` common frames, the link must have at least:

```text
required_valid = min(max(2, ceil(0.80 * T)), T)
```

valid target-depth frames. Equivalently for ordinary clips, at most `floor(0.20 * T)` frames may be filled.

If the link passes this gate:

- missing internal depths are linearly interpolated by frame index;
- leading and trailing gaps use the nearest valid endpoint (`numpy.interp` behavior);
- the completed sequence is normalized by its frame-0 value;
- metadata records the exact interpolated frame indices and fraction.

If the link fails this gate, `object_depth_z` remains `NaN` for that link. The metric loop writes `status: failed`, `error_type: insufficient_target_depth`, and **does not call `evaluate_object_metrics()`**, so that link has no PDI score. Other links continue and receive their own reports ([`multi_object_pipeline.py`](src/pdi_eval/multi_object_pipeline.py#L464)).

### 3.4 What happens in each PDI component when the link passes the depth gate

| Component | Empty SAM mask behavior | Is the frame skipped? |
|---|---|---|
| Scale | `h_pixel=0`; interpolated `depth_z` is used. The formula clamps height to `1e-6` before `log(h)`, usually creating a very large scale residual. | No |
| 3D trajectory | With no masked points, the previous valid world centroid is copied. A 3-frame median filter is then applied to the trajectory. | No; centroid is held |
| 3D rigidity | Later SAM masks are not read. CoTracker tracks sample the pointmap; pair validity comes from CoTracker visibility. With fewer than three visible pairs, the previous rigidity score is copied. | No mask-based skip |
| VP coupling | Foreground/background VPs come primarily from CoTracker. Masks only affect the early-frame object-bbox degeneracy test and union-mask exclusion for LSD background lines. Default PDI weight is `0.0`. | No general skip |
| Scale-jump audit | Empty mask gives foreground median depth `0.0`; this can create a jump. This audit is reported but is not part of the PDI weighted sum. | No |

The exact scale formula is ([`scale_audit.py`](src/pdi_eval/evaluator/scale_audit.py#L4)):

```text
s(t) = log(max(h(t), 1e-6)) + log(max(Z(t), 1e-6))
baseline = median(s(t)) over the first min(5,T) frames
epsilon_scale(t) = |s(t) - baseline|, t = 1 ... T-1
```

The exact missing-centroid branch is in [`motion_audit.py`](src/pdi_eval/evaluator/motion_audit.py#L53): more than 10 masked pointmap samples are required to compute a median centroid; otherwise the previous centroid is used.

### 3.5 Final PDI synthesis

For a link that remains valid, array-valued scale and trajectory errors are converted to RMSE. Rigidity is already a scalar. The current default configuration is ([`configs/default.yaml`](configs/default.yaml#L8)):

```text
PDI = 0.4 * RMSE(scale)
    + 0.4 * RMSE(trajectory)
    + 0.2 * rigidity
    + 0.0 * VP
```

There is no dynamic weight renormalization, no global dropping of SAM-failed frames, and no aggregate imputation of a failed link's PDI. A link either passes the target-depth gate and all common frames flow through the component-specific rules above, or it is marked failed without a PDI score.

## 4. Important edge cases and implications

1. **Frame 0 is mandatory for CoTracker initialization.** If a link's frame-0 mask cannot produce two unique points, `prepare_multi()` raises and stops the whole multi-link evaluation. This is not isolated as a single-link failure.
2. **CoTracker is not re-seeded from later SAM masks.** A later SAM failure does not remove the link's CoTracker anchors, and a later SAM recovery does not add anchors.
3. **Rigidity is largely insulated from later SAM failures.** Only the frame-0 mask affects active pair ranking; later link masks do not constrain sampled XYZ points. This preserves a rigidity result through segmentation dropout, but it can also let a drifted CoTracker point sample background geometry.
4. **Missing masks can affect scale much more than rigidity.** Depth is interpolated, trajectory is held, and rigidity follows CoTracker, but pixel height becomes zero and is log-clamped. A small number of missing masks can therefore dominate `epsilon_scale` even though the link passes the 80% depth-validity gate.
5. **The active remote batch has two different thresholds.** SAM segmentation acceptance is effectively 0% because the runner passes `--minimum-tracked-fraction 0.0`; PDI target depth still requires approximately 80% valid frames. These thresholds should not be confused.
6. **The rigidity implementation checks visibility but not finite sampled XYZ values per frame.** NaN/invalid pointmap samples at a visible track can propagate into ratios and the final score. The normal MegaSAM cache validation checks that some valid geometry exists globally, not every sampled anchor in every frame.

## 5. Compact pseudocode of the active path

```text
for each link:
    mask0 = SAM_masks[frame=0, link]
    candidates = SIFT(mask0, 4C) + ShiTomasi(mask0, 4C) + Grid(mask0, C)
    queries = deterministic_farthest_point_select(deduplicate(candidates), C)
    require len(queries) >= 2

tracks, visibility = CoTracker(video, explicit_queries=[0,x,y], internal_grid=off)
tracks = whole-clip_quality_filter(tracks, visibility)

for each link:
    raw_depth[t] = median(camera_Z[t][SAM_mask[t,link] AND valid_Z])
    if valid_depth_frames < min(max(2, ceil(0.8T)), T):
        link.status = failed
        continue
    depth = interpolate_missing(raw_depth)

    points3d[t,n] = MegaSAM_pointmap[t, round(CoTracker_track[t,n])]
    pairs = best_30_frame0_pairs(points3d, visibility, depth_gradient, mask_interior)
    for t > 0:
        visible_pairs = pairs with both CoTracker endpoints visible
        if len(visible_pairs) < 3:
            rigidity[t] = rigidity[t-1]
        else:
            ratios = distance_t(visible_pairs) / distance_frame0(visible_pairs)
            rigidity[t] = MAD(ratios) / (median(ratios) + 1e-6)

    epsilon_rigidity = mean(rigidity[1:])
    PDI = 0.4*RMSE(scale) + 0.4*RMSE(trajectory) + 0.2*epsilon_rigidity
```
