# CAD-Based Robot Link Canonicalization Design

Status: implementation in progress. The local foundation slice now includes
strict CAD loading, aligned MegaSAM RGB-D/K caching, external FoundationPose
pose-archive validation, fixed CAD anchor binding, per-frame canonicalization,
proportional-shape scoring, SE(3) pose-discontinuity diagnostics, and opt-in
pipeline integration. The isolated FoundationPose worker, video-global scale
calibration, pinned threshold loader, and GPU/real-video validation remain.

Scope: Franka FER `link2` through `link7`. `link1` is explicitly excluded.

## 1. Goal

Replace the current purely temporal rigidity test with a CAD-referenced test:

```text
observed reconstructed link surface, after removal of rigid pose
                            ==
fixed official CAD link surface
```

Each robot link is an independent rigid object. At frame `t`, FoundationPose
estimates only the rigid transform from the official CAD link frame to the
camera frame. The observed MegaSAM points are inverse-transformed into the
unchanged CAD frame. Rigidity is then measured from the link's internal
distance relationships relative to the corresponding visible CAD shape, not
from absolute point-to-surface distance.

This design adds a versioned `cad-canonical-v1` rigidity method. It does not
silently alter the existing point-pair rigidity formula. During validation,
both methods must be reported. Once calibrated, `cad-canonical-v1` may supply
the existing `epsilon_rigidity` input to PDI while the legacy value remains in
the report for comparison.

## 2. Decisions

1. Use camera coordinates for pose estimation and canonicalization.
   FoundationPose consumes camera-frame RGB-D, a camera intrinsic matrix, a
   link mask, and a CAD mesh. Converting a world pointmap to camera coordinates
   is mathematically valid but unnecessary when the aligned camera depth is
   available.
2. Preserve the CAD files in meters. Do not normalize either mesh to a unit
   box, unit sphere, common diameter, or zero-centered public coordinate frame.
3. Bake every DAE scene-graph node transform into the loaded geometry. The
   transforms in these files are part of the official visual mesh definition.
4. Permit at most one scale factor for an entire video. Per-frame and per-link
   scale fitting are forbidden because they can explain away deformation.
5. Keep the CAD fixed. Only observed points are transformed into the CAD link
   frame.
6. Compare visible surfaces, not a partial observed cloud against the entire
   closed mesh. Rear and externally occluded CAD surfaces must not be counted as
   missing deformation.
7. Fail closed. Missing depth, a bad pose, inadequate visible surface, or an
   uncalibrated decision threshold produces `unscorable`, not a zero-error or
   fallback score.
8. Define CAD rigidity over shape modulo similarity transforms. Translation,
   rotation, and one uniform size factor do not change proportional shape;
   anisotropic stretch, bending, collapse, or local deformation do.
9. Report FoundationPose pose discontinuity as a separate secondary metric.
   It can corroborate an abnormal event or expose tracker failure, but pose
   discontinuity alone is not evidence that the link mesh deformed.

## 3. Current Repository Facts

The active pipeline already provides:

- independently named masks in `object_masks[T,N,H,W]`;
- one shared MegaSAM reconstruction;
- `pointmaps[T,Hg,Wg,3]` in a shared world frame;
- `camera_poses[T,4,4]`, currently documented and used as camera-to-world;
- per-link metric evaluation without cross-link rigidity pairs.

The current geometry cache is insufficient for FoundationPose. It stores a
world pointmap, camera poses, and one focal-length scalar, but FoundationPose
requires the aligned RGB image, camera-Z depth image, and full intrinsic matrix
`K = [fx, fy, cx, cy]`. The cache schema must therefore be extended rather than
reconstructing approximate intrinsics later.

The active automatic segmentation path is SAM3, even though some earlier notes
call the mask provider SAM2. The canonicalization interface is mask-provider
agnostic, but implementation and manifests must record the actual SAM3 source.

## 4. CAD Asset Inspection

The pinned assets are under
`PDI-Bench-edited/assets/cad/franka_fer/`. Every inspected DAE declares:

```xml
<unit name="meter" meter="1"/>
<up_axis>Z_UP</up_axis>
```

The following values were obtained by parsing all geometry instances, applying
their DAE node matrices, and then computing the axis-aligned bounds. They are
also required loader assertions, with a `1e-5 m` tolerance.

| Link | Geometry instances | Scene transform shared by its instances | Baked AABB extents in meters | AABB diagonal in meters |
| --- | ---: | --- | --- | ---: |
| `link2` | 1 | identity | `(0.110033, 0.249024, 0.184393)` | `0.328818` |
| `link3` | 4 | translate Z by `-0.121` | `(0.192511, 0.166063, 0.176002)` | `0.309216` |
| `link4` | 4 | translate Y by `+0.003` | `(0.192507, 0.179000, 0.166053)` | `0.310924` |
| `link5` | 3 | translate Z by `-0.259` | `(0.109996, 0.184930, 0.311199)` | `0.378342` |
| `link6` | 17 | translate Z by `-0.0148` | `(0.179926, 0.132863, 0.100243)` | `0.245101` |
| `link7` | 8 | rotate Z by `-45 deg`, translate Z by `+0.052` | `(0.125333, 0.125297, 0.054800)` | `0.185502` |

The production loader must:

1. verify the SHA-256 values already pinned in
   `configs/sam3-cad-franka.yaml`;
2. require `meter="1"` and `Z_UP` for this asset set;
3. load the file as a scene with `process=False`;
4. apply each instance's complete scene-graph transform;
5. concatenate all triangle geometry for the link while retaining material or
   vertex color where FoundationPose can use it;
6. calculate vertex normals after transforms;
7. retain the asset frame origin exactly as loaded;
8. reject empty, non-finite, non-triangular, or unexpectedly sized meshes.

FoundationPose internally centers a mesh for optimization. Its public pose is
for the original input mesh. The integration must use that returned public pose
and must never expose the internal centered-mesh pose as the CAD-frame pose.

## 5. Coordinate Contract

All implementation code uses homogeneous 4 by 4 matrices, column vectors, and
names transforms as `T_destination_from_source`. No unqualified variable named
`pose`, `extrinsic`, `R`, or `t` may cross a module boundary.

### Frames

- `V`: source-video pixel grid, with `u` right and `v` down.
- `G`: MegaSAM geometry image grid after resize and bottom/right crop.
- `C_t`: OpenCV camera frame at frame `t`: X right, Y down, Z forward.
- `W`: MegaSAM world frame.
- `L_i`: original, scene-transform-baked CAD frame of link `i`.

### Stored transforms

- `T_W_from_C[t]`: MegaSAM `cam_c2w[t]`.
- `T_C_from_L[t,i]`: FoundationPose result for the original link mesh.

Every rigid transform must satisfy:

```text
last row                == [0, 0, 0, 1]
R transpose times R    ~= identity
det(R)                 ~= +1
```

### Preferred camera-point construction

For a valid geometry-grid pixel `(u,v)` with MegaSAM camera-Z depth `D_t[v,u]`,
let `s_video` convert MegaSAM depth units to CAD meters:

```text
p_C = s_video * D_t[v,u] * inverse(K_G) * [u, v, 1]^T
```

Then, for `T_C_from_L = [R_C_from_L, t_C_from_L]`, canonicalize the observation:

```text
p_L = R_C_from_L^T * (p_C - t_C_from_L)
```

This is the only production canonicalization equation.

### World-pointmap equivalence check

The existing MegaSAM pointmap is created as:

```text
p_W = R_W_from_C * p_C + t_W_from_C
```

Therefore a diagnostic reconstruction must recover:

```text
p_C = R_W_from_C^T * (p_W - t_W_from_C)
```

Direct depth back-projection and world-pointmap inversion must agree to
`1e-5 * max(1, ||p_C||)` on finite samples. A scored run fails if this check
does not pass. The world path is a validation path, not the primary data path.

## 6. Exact RGB, Depth, Mask, and Intrinsic Alignment

The current MegaSAM camera-tracking input performs these operations:

1. resize the source frame to an area near `384 * 512` while preserving aspect
   ratio;
2. crop the resized height and width down to multiples of eight at the bottom
   and right;
3. resize depth to the pre-crop size and apply the same crop;
4. scale `fx,cx` by the width ratio and `fy,cy` by the height ratio.

A direct resize from the source mask to the final cropped geometry size is not
equivalent. It slightly compresses the image instead of discarding the cropped
bottom/right pixels.

The geometry cache must store the already aligned arrays and transform record:

```text
rgb_camera:             uint8[T,Hg,Wg,3], RGB channel order
depth_camera:           float32[T,Hg,Wg], camera-Z, invalid value 0
intrinsics_camera:      float64[T,3,3] or float64[3,3]
pointmaps_world:        float32[T,Hg,Wg,3]
T_W_from_C:             float64[T,4,4]
source_hw:              int32[2]
resized_hw_before_crop: int32[2]
crop_xywh:              int32[4]
```

Masks must be transformed by the recorded source-to-geometry operation: nearest
neighbor resize to `resized_hw_before_crop`, then the exact crop. RGB, depth,
mask, and `K_G` must have the same `Hg,Wg` before FoundationPose is called.

The cache schema must be bumped. Old caches lacking these arrays are not valid
for CAD canonicalization and must be regenerated; values must not be inferred
from the single stored focal length.

## 7. Absolute Scale Policy

### 7.1 What is observable

A monocular video cannot distinguish a uniformly larger object farther from the
camera from a smaller object closer to the camera without a metric prior. CAD
does not remove this global similarity gauge by itself. MegaSAM uses UniDepth as
a metric prior, but its result is still an estimate.

Consequently, two explicitly named policies are supported:

- `metric-prior`: set `s_video = 1`. This tests absolute size under UniDepth's
  metric prediction.
- `video-global-cad`: fit one scalar for the whole video, then test shape. This
  intentionally treats uniform whole-scene scale as unobservable and is the
  default for the deformation benchmark.

Reports must contain results under the selected policy and the raw
`metric-prior` diagnostics. They must never imply that `video-global-cad`
detects uniform global resizing.

### 7.2 Forbidden normalization

These geometry-changing operations are prohibited:

- normalizing X, Y, and Z extents independently;
- rewriting each observed cloud or CAD mesh to a unit box or unit sphere;
- fitting a scale into the FoundationPose transform for every link or frame;
- applying a per-link or per-frame scale to depth or canonical point artifacts;
- allowing similarity ICP to modify geometry after `s_video` is fixed.

These would contaminate the shared reconstruction or could remove anisotropic
proportion changes before they reach the metric.

The rigidity formula in Section 10 is nevertheless analytically invariant to a
uniform scale applied to all distances in one observation. It does not rescale
or rewrite the geometry. This separation is intentional: the existing PDI
scale component detects temporal or perspective scale inconsistency, while CAD
rigidity detects changes in proportion and shape. Uniform isotropic enlargement
is therefore a scale condition, not a CAD-shape deformation.

### 7.3 Video-global CAD calibration

Calibration is deterministic and occurs before scored frame poses are cached.

1. Select at most one calibration frame per link. A candidate frame must be
   non-truncated, contain at least 256 geometry-grid mask pixels, have at least
   70 percent valid depth under the eroded mask, and be selected by maximum
   eroded-mask area. At least three distinct links are required.
2. Evaluate the coarse scale grid
   `s = 2^k`, where `k = -2.00, -1.75, ..., +2.00`.
3. For every scale and calibration observation, multiply the complete camera
   depth map by `s`, run FoundationPose registration, and render the CAD at the
   returned `T_C_from_L` with the exact `K_G`.
4. For a registration, compute a tolerant silhouette IoU after two-pixel
   dilation of both masks and a depth residual on their eroded intersection:

   ```text
   e_sil   = 1 - IoU(dilate(render_mask,2), dilate(link_mask,2))
   e_depth = min(median(abs(s*D - rendered_depth)) / (0.05*D_link), 1)
   q       = 0.5*e_sil + 0.5*e_depth
   ```

   `D_link` is the baked CAD AABB diagonal. A failed registration contributes
   `q = 1`.
5. For each scale, aggregate `q` by taking the median across distinct links.
   Choose the scale with the lowest aggregate, breaking ties toward the value
   closest to `1` and then toward the smaller value.
6. Evaluate a fine grid in log2 space from one coarse step below to one coarse
   step above the winner, inclusive, with step `0.05`. Choose by the same rule.
7. Reject calibration when the winner is at `0.25` or `4.0`, fewer than three
   links register successfully, or the winning aggregate `q` exceeds `0.60`.
8. Freeze and record `s_video`. It cannot change during pose estimation or
   metric computation.

The scale-calibration fit score is a nuisance-parameter objective, not the
deformation score. Both the full scale curve and the contributing link/frame
identities must be retained for audit.

## 8. FoundationPose Integration

FoundationPose is a third-party pose backend, not a replacement for MegaSAM.
It receives only arrays in the geometry camera frame:

```text
RGB_G[t]
s_video * depth_G[t]
mask_G[t,i]
K_G[t]
CAD_mesh[i]
```

It returns `T_C_from_L[t,i]`. The wrapper must pin the upstream revision,
checkpoint identities, refinement counts, random seed, renderer settings, and
CUDA/PyTorch environment in the run manifest.

FoundationPose has compiled renderer and CUDA dependencies that are not part of
the pinned MegaSAM environment. It should run in a separate pinned environment
through a numeric artifact protocol, as SAM3 already does. This prevents its
dependencies from changing the PyTorch 2.1/CUDA 11.8 MegaSAM environment and
releases FoundationPose GPU memory before CoTracker starts.

### Verified upstream behavior

This design is based on the official
[project page](https://nvlabs.github.io/FoundationPose/),
[paper](https://arxiv.org/abs/2312.08344), and
[repository](https://github.com/NVlabs/FoundationPose) at revision
`a1b694b83e633c2cb6115b9063d940a687759392`.

The official demo calls `register` for the initial frame and `track_one` for
later frames. `track_one` starts pose refinement from the preceding estimate
stored in `pose_last`; it does not generate the registration stage's multiple
pose hypotheses on each tracking frame. The learned refiner predicts
camera-frame translation and SO(3) rotation updates from rendered CAD and
observed RGB-D. The public implementation does not apply a constant-velocity
model, temporal smoother, motion-plausibility check, or fixed jump threshold.

Therefore, temporal pose smoothness is information that the integration can
derive from FoundationPose output, but it is not a confidence value supplied by
FoundationPose. A pose jump can be caused by real robot motion, camera motion,
occlusion, segmentation/depth error, a symmetric-pose switch, tracker failure,
or deformation changing the best rigid fit. Section 11 keeps these meanings
separate.

### Per-link temporal procedure

For each link independently:

1. Find the first frame satisfying the mask/depth input gates and call
   `register` with five refinement iterations.
2. Call `track_one` with two refinement iterations on the next valid frame.
3. Re-register every ten processed frames and whenever catastrophic pose
   validation fails. Compare tracked and registered candidates with the `q`
   objective above and keep the lower-`q` candidate.
4. A frame is pose-valid only if its rotation is a proper rotation, translation
   is finite with positive camera Z, at least 128 canonical points remain, and
   tolerant silhouette IoU is at least `0.20`.
5. A high CAD shape residual alone is not a pose-invalid condition; otherwise a
   truly deformed link would be discarded rather than detected.
6. If neither tracking nor registration passes the catastrophic gates, mark the
   frame `pose_failed`. Never copy the preceding pose into a scored frame.

No unverified symmetry transforms are supplied initially. If exact link
symmetries are later audited, they must be explicit per-link configuration.
Equivalent symmetric poses are resolved to the representative closest to the
previous valid rotation for stable visualizations; unsigned CAD surface
distance itself is symmetry-safe.

## 9. Canonical Point Extraction

For every pose-valid `(t,i)`:

1. transform the link mask exactly into grid `G`;
2. erode it by two pixels to reduce joint-boundary and mixed-depth leakage;
3. retain finite positive depths;
4. remove isolated depth components smaller than 32 pixels, keeping all larger
   connected components because a link can be split by occlusion;
5. back-project retained pixels into `C_t` using the full `K_G`;
6. apply the inverse rigid pose to obtain `P_L[t,i]`;
7. voxel-downsample in the CAD frame with voxel size `0.005 * D_link`;
8. retain the source pixel coordinate and depth-validity flags for every
   downsampled point used by the metric.

No centering, PCA alignment, axis sorting, bounding-box alignment, ICP scale,
or non-rigid registration is allowed. Optional point-to-plane rigid ICP may be
evaluated only as an ablation and may refine `T_C_from_L` in SE(3); it is not in
`cad-canonical-v1` because FoundationPose is the specified pose estimator.

## 10. Relative Shape and Proportion Comparison

The primary score does not use absolute point-to-CAD distance. Absolute distance
would mix actual deformation with a global depth-scale error, a small coordinate
bias, and FoundationPose translation error. Instead, the observed link and CAD
are treated as two finite metric spaces and compared through their internal
pairwise distance relationships.

Translation and rotation disappear because Euclidean distances within each
shape are unchanged by rigid transforms. Uniform scale disappears by removing
the median log distance ratio. What remains is non-uniform strain: a change in
the proportions or general shape of the link.

### 10.1 Use one CoTracker/CAD anchor manifest

There must not be one set of CoTracker anchors and another independently sampled
set of CAD anchors. The existing deterministic CoTracker foreground queries are
the sole anchor initialization. The same query IDs are bound to CAD once and
preserved through every frame and tracking mode.

The existing `TrackWrapper.prepare_multi` constructs each link's query manifest
on frame 0 using SIFT, then Shi-Tomasi, then grid fallback, followed by spatial
balancing. Every query is `[query_frame=0, x_tracker, y_tracker]`. This exact
ordered manifest is the source of truth.

For each link and query `j`:

1. Assign a stable `query_id` equal to its position in the prepared per-link
   query manifest. Neither FoundationPose nor the CAD metric may reorder or
   resample it.
2. Map `(x_tracker,y_tracker)` from the CoTracker grid to source-video pixel
   coordinates using the exact pixel-center resize transform, then map source
   pixels to MegaSAM grid `G` using the recorded resize and crop from Section 6.
   Do not use `_map_tracks_between_grids` and do not use bare width/height
   multiplication.

   For X coordinates, the mapping is:

   ```text
   x_V = (x_tracker + 0.5) * W_V / W_tracker - 0.5
   x_G = (x_V       + 0.5) * W_resized / W_V - 0.5 - crop_x
   ```

   Y uses the identical formula with heights and `crop_y`. The same composed
   transforms are applied to the frame-0 query and every later CoTracker track;
   they are stored in the manifest and unit-tested by round trip.
3. Require frame-0 CoTracker query membership in the eroded link mask and valid
   MegaSAM depth at the mapped geometry pixel.
4. Use frame-0 `T_C_from_L[0,i]` and `K_G[0]` to transform that camera pixel ray
   into the CAD frame and intersect it with the baked CAD triangle mesh. The
   nearest positive ray intersection is the fixed CAD anchor `q_j`.

   With `T_C_from_L = [R,t]` and `d_C = inverse(K_G)[u_G,v_G,1]^T`, the ray in
   the CAD frame is:

   ```text
   ray_origin_L    = -R^T * t
   ray_direction_L =  R^T * d_C
   q_j             = ray_origin_L + lambda_nearest * ray_direction_L
   ```

   where `lambda_nearest > 0` is the first triangle intersection.
5. The frame-0 observed depth at the same pixel produces `p_j(0)`. If the ray
   misses the CAD, depth is invalid, or the query is outside the eroded mask,
   mark that `query_id` CAD-invalid; do not replace it with a newly sampled
   point.
6. Retain the ordered subset of CAD-valid query IDs. Require at least 16 anchors
   for that link; otherwise the link is `insufficient_anchor_initialization`.

For each later frame `t`, CoTracker supplies the pixel for the same `query_id`.
The pixel is accepted only when CoTracker marks it visible, it remains inside
that frame's eroded link mask, and mapped MegaSAM depth is valid. Back-projecting
that depth gives `p_j(t)`. The CAD point `q_j` never changes.

Thus the correspondence is explicit:

```text
query_id j
  -> frame-0 CoTracker pixel
  -> one frame-0 CAD ray intersection q_j
  -> CoTracker pixel at frame t
  -> observed MegaSAM point p_j(t)
```

FoundationPose poses from later frames are used for pose diagnostics and
replays, but they do not rebind `q_j`. Rebinding every frame would change point
identity and could hide deformation.

The canonical points from Section 9 remain useful for replay and debugging, but
the score can use `p_j(t)` in camera coordinates because internal distances are
already invariant to the rigid camera/CAD transform.

### 10.2 Pair selection

For every unordered anchor pair `(j,k)`, compute:

```text
a_jk(t) = ||p_j(t) - p_k(t)||_2   # observed internal distance at frame t
b_jk    = ||q_j - q_k||_2         # fixed CAD internal distance
```

At frame `t`, a pair exists only when both query IDs pass that frame's
visibility, mask, and depth gates. Discard a pair when
`b_jk < 0.05 * D_link`; very short baselines are dominated by depth noise. Sort
the remaining pairs by CAD baseline, split that ordering into eight equal-count
bins, and take up to 64 evenly spaced pairs from each bin, using `(j,k)` as the
deterministic tie-break. This retains at most 512 pairs while representing both
medium and long-range proportions instead of letting the longest baselines
dominate. At least 30 pairs are required for a scored frame.

### 10.3 Remove the uniform scale gauge

For every retained pair:

```text
l_jk(t) = log((a_jk(t) + delta) / (b_jk + delta))
delta = 1e-6 * D_link
```

Estimate the observation's uniform relative size in log space:

```text
mu(t) = median(l_jk(t))
relative_uniform_scale(t) = exp(mu(t))
```

`relative_uniform_scale` is diagnostic only. It is never applied to the point
cloud, pose, depth map, or another frame. Subtract it from every pair relation:

```text
r_jk(t) = l_jk(t) - mu(t)
```

If the observed shape is a translated, rotated, and uniformly scaled copy of
the CAD shape, every `l_jk(t)` is the same and all `r_jk(t)` are zero.
Anisotropic or local deformation makes different pairs change by different
ratios.

### 10.4 Frame score

Define the frame's proportional deformation as a robust capped RMS of log
strain:

```text
c = log(1.5)

epsilon_cad_frame(t) = sqrt(mean(min(r_jk(t)^2, c^2)))
```

The cap prevents a small number of bad depth or mask correspondences from
dominating the score. The score is dimensionless and directly interpretable
for small errors: `0.02` is approximately 2 percent inconsistent proportional
strain among the link's internal distances. It is not “2 percent of the CAD's
meter diameter.”

The report must also retain:

- `relative_uniform_scale`;
- median, RMS, 90th percentile, and maximum of `abs(r_jk)`;
- anchor and retained-pair counts;
- evaluable visible CAD coverage;
- the spatial distribution of per-anchor median `abs(r_jk)` for deformation
  heatmaps.

Absolute point-to-triangle residual, rendered depth residual, and silhouette IoU
remain pose/mask diagnostics only. They do not enter `epsilon_cad_frame`.

### Link score and decision

For accepted frames of one link:

```text
epsilon_cad_mean = mean(epsilon_cad_frame)
epsilon_cad_p90  = percentile(epsilon_cad_frame, 90)
```

`epsilon_cad_mean` is the `cad-canonical-v1` rigidity component supplied to the
PDI calculator. `epsilon_cad_p90` is retained to detect intermittent
deformation. A link is scorable only with at least five accepted frames and at
least 60 percent accepted frames among frames where its segmentation mask is
present.

Binary deformation thresholds must not be guessed from the generated videos.
They are stored in a versioned `cad_rigidity_thresholds_v1.yaml`, calibrated
per link from held-out, non-deformed Franka control videos after the complete
RGB-D, mask, scale, and pose pipeline:

```text
theta_mean[i] = 99th percentile of control epsilon_cad_mean for link i
theta_p90[i]  = 99th percentile of control epsilon_cad_p90  for link i

deformed = epsilon_cad_mean > theta_mean[i]
        OR epsilon_cad_p90  > theta_p90[i]
```

A scored binary decision fails closed if the pinned threshold file is absent,
has different CAD hashes or method settings, or was calibrated on the video
being evaluated. Continuous residuals may still be emitted with status
`uncalibrated`.

## 11. FoundationPose Pose-Discontinuity Metric

The secondary method is named `foundationpose-pose-discontinuity-v1`. It asks:

```text
Did the accepted rigid pose depart abruptly from the motion predicted by the
two preceding accepted poses?
```

It does not ask whether the CAD surface deformed. It is reported beside
`cad-canonical-v1`, not inserted into `epsilon_cad_frame` and not assigned a PDI
weight in version 1.

### 11.1 Remove camera motion and match metric scale

Raw `T_C_from_L` cannot be differenced because a moving camera changes it even
when the robot link is stationary. Convert it to the shared MegaSAM world
frame. The camera translation must use the same frozen video-global scale as
FoundationPose depth:

```text
T_W_from_C_metric[t] = [R_W_from_C[t], s_video * t_W_from_C[t]]
T_W_from_L[t,i]      = T_W_from_C_metric[t] * T_C_from_L[t,i]
```

The resulting link translation is in CAD meters. Scaling depth without also
scaling `t_W_from_C` is invalid and creates a false discontinuity whenever the
camera moves. The arbitrary world origin and orientation do not affect the
metric because only relative SE(3) motion is used.

Use source-video presentation timestamps `tau[t]` in seconds. When exact PTS is
not available, `tau[t] = t / fps` is permitted only for a constant-frame-rate
video with a finite positive recorded FPS. Timestamp provenance and FPS are
part of the threshold/cache identity.

### 11.2 Constant-velocity SE(3) innovation

Raw frame-to-frame movement is the wrong signal: a robot can move quickly but
smoothly. For three temporally consecutive pose-valid samples of one link,
define:

```text
h_previous = tau[t-1] - tau[t-2]
h_current  = tau[t]   - tau[t-1]

Delta_previous = inverse(T_W_from_L[t-2]) * T_W_from_L[t-1]
T_predicted     = T_W_from_L[t-1]
                  * Exp((h_current / h_previous) * Log(Delta_previous))
E_innovation    = inverse(T_predicted) * T_W_from_L[t]
```

`Log` and `Exp` are the standard SE(3) logarithm and exponential. They must be
implemented with a tested geometry library or tested closed-form routines, not
element-wise matrix logarithms. `h_previous` and `h_current` must be positive;
a sample is unavailable across a missing-pose gap longer than two nominal frame
intervals. The first two poses of a link have no innovation value.

Extract the unexpected spatial correction:

```text
u_translation(t) = ||translation(E_innovation)|| / D_link

u_rotation(t) = acos(clip((trace(rotation(E_innovation)) - 1) / 2, -1, 1))
```

`u_translation` is measured in link diameters and `u_rotation` in radians. For
frame-rate-normalized thresholding, also compute:

```text
v_translation(t) = u_translation(t) / h_current       # link diameters / second
v_rotation(t)    = degrees(u_rotation(t)) / h_current # degrees / second
```

Smooth constant body-frame velocity gives near-zero innovation even when the
absolute per-frame translation or rotation is large. Small pose jitter remains
below the dead zone. A sudden tracking reset or physical motion change produces
a large innovation.

For diagnostics only, retain the raw consecutive-pose delta
`inverse(T_W_from_L[t-1]) * T_W_from_L[t]`. It may trigger re-registration when
its translation exceeds `0.50 * D_link` or its rotation exceeds `45 degrees`,
but these raw guardrails must not produce a deformation decision.

### 11.3 Threshold and continuous severity

The initial conservative diagnostic thresholds are:

```text
theta_translation_rate = 3.0 link diameters / second
theta_rotation_rate    = 450 degrees / second

pose_discontinuity(t) =
    v_translation(t) > theta_translation_rate[i]
    OR v_rotation(t)  > theta_rotation_rate[i]

pose_discontinuity_severity(t) = max(
    v_translation(t) / theta_translation_rate[i],
    v_rotation(t)    / theta_rotation_rate[i],
)
```

The event threshold is exactly `severity > 1`; equality is accepted. At 30 fps
the two thresholds correspond to an unexpected correction of `0.10 * D_link`
or `15 degrees` in one frame. The translation equivalents for the inspected
CAD links are:

| Link | `0.10 * D_link` |
| --- | ---: |
| `link2` | `0.032882 m` |
| `link3` | `0.030922 m` |
| `link4` | `0.031092 m` |
| `link5` | `0.037834 m` |
| `link6` | `0.024510 m` |
| `link7` | `0.018550 m` |

These are startup values for diagnostics, not benchmark-derived truths. The
production threshold file stores separate translation and rotation thresholds
per link. For each primitive, use held-out non-deformed control videos and set:

```text
scaled_MAD = 1.4826 * median(abs(control_value - median(control_value)))

calibrated_threshold = max(
    provisional_threshold,
    control_99th_percentile,
    control_median + 6 * scaled_MAD,
)
```

For calibration, first take the maximum quality-gated `v_translation` and
`v_rotation` separately within each non-deformed control video. Apply the rule
above across those per-video maxima. Calibrating pooled frames would let long
videos dominate and would not control the false-positive rate of the
video-level "any event" decision. Calibration must be split by video, never by
frames from the same video, and the held-out video-level false-positive rate is
reported per link. The threshold file pins CAD hashes, scale policy,
FoundationPose revision/checkpoints, FPS and timestamp policy, pose-quality
settings, and calibration-video hashes.

For each link, report discontinuity count/rate and the median, p95, and maximum
severity over valid innovations. One valid over-threshold sample is retained as
an event; it is not silently averaged away.

### 11.4 Quality gates and interpretation

An innovation is computable only when all three input poses pass Section 8's
rigid-transform, depth, mask, and rendering gates and the symmetry
representative is temporally consistent. Also retain the pose objective `q`
from Section 7.3 for every candidate. It is valid for physical-motion
aggregation only when all three poses have `q <= 0.40`. This is the provisional
high-quality gate; calibrate and pin it with the pose thresholds. Computable
innovations that fail this stricter gate remain tracker diagnostics.

Classify, do not conflate, an over-threshold event:

| Observation | Interpretation | Scoring action |
| --- | --- | --- |
| Pose jump and current/adjacent `q` fails the high-quality gate | FoundationPose/mask/depth failure likely | `estimator_discontinuity`; exclude from physical-motion aggregation and re-register |
| Pose jump on a registration reset, without three consistent tracked poses | Estimator correction is ambiguous | `estimator_reset`; report separately and do not call deformation |
| High-quality pose jump, low CAD proportional strain | Real abrupt rigid motion or a residual pose ambiguity | `motion_discontinuity`; not deformation |
| High-quality pose jump and high CAD proportional strain | Two independent abnormal signals agree | `motion_discontinuity_with_shape_deformation`; deformation decision still comes from the calibrated CAD-shape rule |
| High CAD proportional strain without a pose jump | Shape changed without moving its best-fit rigid pose abruptly | valid deformation evidence |

For cross-link diagnosis, express the translation residual in the world frame
as `d_W = translation(T_W_from_L[t]) - translation(T_predicted)` and the
rotation residual as the SO(3) rotation vector of
`R_W_from_L[t] * R_predicted^T`. A set of residual vectors is aligned when the
norm of the sum of its unit vectors divided by its count is at least `0.80`.
Zero vectors and primitives below their own threshold are omitted. If at least
four of the six links have an event in the same frame and their translation or
rotation residuals are aligned by this rule, emit
`shared_camera_or_reconstruction_discontinuity`. This diagnostic does not erase
each link's raw values, but it blocks pose discontinuity from being used as
deformation corroboration because a shared MegaSAM camera-pose error is more
likely.

This separation is mandatory. Directly adding pose severity to rigidity would
penalize correct fast robot motion and convert FoundationPose tracking failures
into false deformation detections. A future fused decision rule requires its
own version and calibration; it must not silently replace either primitive.

## 12. Pipeline Dataflow

```text
source video
  |
  +--> SAM3 once
  |      object_masks[T,N,Hv,Wv], stable names link2...link7
  |
  +--> MegaSAM once on full frames
         aligned RGB_G + depth_G + K_G
         T_W_from_C + world pointmaps
                 |
                 +--> video-global scale calibration once
                 |      fixed s_video
                 |
                 +--> FoundationPose per link
                 |      T_C_from_L[T,N]
                 |          |
                 |          +--> compose T_W_from_L[T,N]
                 |                 constant-velocity SE(3) innovation
                 |                 pose discontinuity diagnostics
                 |
                 +--> prepare one CoTracker query manifest
                 |      stable query_id per link
                 |
                 +--> bind each frame-0 query ray to fixed CAD q_j
                 |
                 +--> joint-query and/or exact-group tracks
                        tracked MegaSAM points p_j(t)
                            |
                            +--> relative CAD shape relations
                                   epsilon_cad_frame[T,N,mode]
                                   epsilon_cad_mean[N,mode]
                                   deformation decision[N,mode]
```

Links are never merged for pose estimation, canonicalization, shape relations,
or scoring. The union mask remains valid only for existing background exclusion
and scene-level reconstruction audits.

## 13. Reuse and Minimal-Change Policy

This feature extends the existing native multi-object pipeline. It must not
create a second video runner, a second segmentation representation, a second
MegaSAM invocation, or a parallel PDI aggregation path.

### Reuse unchanged

The implementation must reuse these existing contracts and behaviors:

| Existing code | Reuse |
| --- | --- |
| `perception.segmentation_archive.load_multi_object_segmentation` | Load and validate the named multi-link masks. Do not add a CAD-specific mask archive. |
| `MultiObjectSegmentation` | Preserve link names, IDs, per-link masks, truncation flags, and union-mask semantics. |
| `MegaSamWrapper.infer_shared` | Remain the only reconstruction entry point and the owner of the one video-global geometry cache. |
| MegaSAM subprocess stages | Continue producing depth, camera motion, and pointmaps once from the full unmodified video. FoundationPose must consume their outputs, not run another depth model. |
| `SharedGeometryResult` | Extend it with aligned RGB, camera depth, full intrinsics, and image-transform metadata instead of introducing a competing geometry result. |
| `MultiObjectPDIEvaluationPipeline.run` | Remain the single orchestration path for segmentation, reconstruction, per-link metrics, comparison, timing, and reporting. |
| `evaluate_object_metrics` | Continue computing scale, trajectory, VP, and PDI. Add a narrow rigidity-method input rather than duplicating this function. |
| `PDIIndexCalculator` | Continue aggregating the selected rigidity component with the existing configured weights. |
| `evaluator.motion_audit` | Add the pure SE(3) pose-discontinuity audit beside the existing temporal motion audit; do not create another metric framework. |
| `TrackWrapper` query sampling | Reuse the existing SIFT, Shi-Tomasi, grid fallback, spatial balancing, and single shared query manifest without creating CAD-specific samples. |
| `MultiObjectTrackResult` and track archives | Extend them only with stable numeric query IDs needed to join tracks to fixed CAD anchors. Preserve joint-query/exact-group behavior. |
| `write_report`, `_jsonable`, and atomic `.tmp` replacement patterns | Continue to own JSON and NPZ serialization behavior. |
| `configs/default.yaml` and `evaluation/run_multi_object.py` | Extend the existing configuration, CLI, manifest, timing, and artifact sections. Do not add a separate production command. |
| `cad_reference.py` mesh loading and pinned CAD manifest | Promote the existing DAE scene-loading behavior into one shared loader used by both CAD silhouette segmentation and FoundationPose. |

The current private `_load_triangle_mesh` function should become a shared
strict loader that can return both the `trimesh.Trimesh` object needed by
FoundationPose and its numeric vertices/faces view needed by CAD reference
rendering. `cad_reference.py` then becomes a consumer of that loader; it is not
rewritten.

### One shared CAD initialization, cheap per-mode scoring

FoundationPose and CAD anchor binding run once. The inexpensive relational
score uses each mode's CoTracker trajectories inside the existing mode loop:

```text
load segmentation once
  -> infer_shared once
  -> FoundationPose once for link2...link7
  -> prepare CoTracker once
  -> bind prepared frame-0 query IDs to fixed CAD q_j once
  -> joint-query and/or exact-group
       -> tracked p_j(t)
       -> existing per-link metric loop computes CAD relational rigidity
```

When both tracking modes run, they use identical initial query IDs and identical
fixed CAD anchors `q_j`, but their tracked `p_j(t)` may differ. Their CAD
rigidity values are therefore allowed to differ and the existing comparison
report must record that delta. Neither mode may cause another FoundationPose
call or rebind a CAD anchor.

`evaluate_object_metrics` should accept one compact rigidity selection object,
conceptually:

```text
rigidity_method
rigidity_value
rigidity_history
rigidity_metadata
```

For `legacy-point-pair`, the function calls the current
`audit_3d_volume_stability` exactly as it does now. For `cad-canonical-v1`, the
CAD audit receives that mode's tracked query IDs and coordinates plus the one
shared CAD anchor table. Scale, trajectory, VP, PDI aggregation, and report
structure otherwise follow the same code path.

### Deliberately new code

Only four concerns require new implementation:

1. strict reusable DAE loading and mesh-surface queries;
2. a thin FoundationPose process adapter and its isolated-environment worker;
3. CoTracker-to-CAD anchor binding, canonicalization, and proportional-strain
   calculation;
4. versioned CAD threshold loading and validation.

Do not introduce a generic workflow framework, model registry, event bus,
database, new cache service, new top-level runner, or duplicate JSON report.
Small typed dataclasses and pure NumPy geometry functions are sufficient.

### Necessary modifications, not rewrites

- Bump the existing MegaSAM cache schema and retain arrays already present in
  MegaSAM's final NPZ (`images`, `depths`, `intrinsic`, `cam_c2w`). Keep the
  current pointmap lifting and target-depth derivation.
- Extend `SharedGeometryResult`; do not replace it.
- Add one CAD result dataclass beside the existing perception result dataclasses
  in `perception/base.py`.
- Add stable query IDs to the existing prepared/result track dataclasses and
  numeric track archive. Do not change how query pixels are selected.
- Add one CAD stage call and one rigidity selection branch to
  `MultiObjectPDIEvaluationPipeline`; preserve its tracking-mode loop.
- Add fields to the existing `metrics.json`, `timing.json`, and `manifest.json`.
  A separate `cad_rigidity.json` is optional debug output and is not emitted by
  default because it would duplicate `metrics.json`.
- Keep the existing combined replay path. CAD overlays may be added to it later
  as an optional view; a second replay system is out of scope.

The only existing helper that must not be reused for this feature is
`_map_tracks_between_grids`: its endpoint-preserving track scaling does not
represent MegaSAM's resize-then-crop image transform. CAD masks, initialization
queries, and later track pixels use the exact recorded pixel-center transforms
described in Sections 6 and 10.1.

## 14. Native Module Boundaries

Implementation belongs inside `PDI-Bench-edited/`:

```text
src/pdi_eval/geometry/cad_mesh.py
    DAE validation, scene-transform baking, mesh metadata, surface queries

src/pdi_eval/perception/foundation_pose_wrapper.py
    artifact contract, scale calibration, subprocess invocation, pose validation

src/pdi_eval/perception/foundation_pose_worker.py
    FoundationPose-only environment entry point

src/pdi_eval/evaluator/cad_rigidity_audit.py
    pure canonicalization, pairwise proportional strain, aggregation

src/pdi_eval/evaluator/motion_audit.py
    existing motion audit plus pure FoundationPose pose-discontinuity audit

src/pdi_eval/perception/mega_sam_wrapper.py
    geometry-cache schema bump; retain aligned RGB, depth, K, and resize/crop data

src/pdi_eval/multi_object_pipeline.py
    stage orchestration and versioned rigidity-method selection

evaluation/run_multi_object.py
    CLI/config/manifest integration

configs/default.yaml
    cad_canonicalization settings, method version, scale policy

scripts/install_foundationpose_gpu.sh
    pinned isolated environment and compiled dependency verification
```

The four new Python modules are `cad_mesh.py`, `foundation_pose_wrapper.py`,
`foundation_pose_worker.py`, and `cad_rigidity_audit.py`. All other work is a
targeted extension to an existing module.

## 15. Artifact Contracts

### Pose archive

`foundationpose_poses.npz` contains numeric arrays only:

```text
link_names:                 str[N]
frame_indices:              int32[T]
frame_times_seconds:        float64[T]
T_C_from_L:                 float64[T,N,4,4]
T_W_from_L:                 float64[T,N,4,4]
pose_valid:                 bool[T,N]
pose_source:                uint8[T,N]
silhouette_iou:             float32[T,N]
pose_depth_residual:        float32[T,N]
pose_objective:             float32[T,N]
video_depth_scale:          float64 scalar
scale_candidates:           float64[S]
scale_objective:            float64[S]
metadata_json:              scalar UTF-8 JSON
```

`pose_source` uses a manifest-defined enum: `0=invalid`, `1=register`,
`2=track`, `3=reregister`.

The existing CoTracker archive is extended with:

```text
query_ids:                 int32[Q]
query_object_ids:          int16[Q]
cad_anchor_valid:          bool[Q]
cad_anchor_points:         float64[Q,3]   # fixed q_j in original CAD frames
cad_anchor_triangle_ids:   int32[Q]
```

These arrays use the same flat query ordering and existing per-object offsets as
the track archive. Joint-query and exact-group archives must contain identical
initial query IDs and identical CAD anchor values for common queries.

### CAD rigidity report

The existing `metrics.json` contains a `cad_rigidity` section for each link:

- CAD hash, loader metadata, unit, baked bounds, and diameter;
- pose-valid, mask-present, and scored-frame counts and indices;
- scale policy, fitted scalar, calibration support, and objective curve;
- per-frame log-strain statistics, relative scale diagnostics, anchor coverage,
  and rejection reasons;
- initial query count, CAD-bound query IDs, and per-frame usable query IDs;
- mean and p90 link residuals;
- threshold identity, calibrated/uncalibrated state, and deformation decision;
- FoundationPose revision, weights, environment identity, and timings.

The same `metrics.json` contains a separate `pose_discontinuity` section per
link with:

- method version, link diameter, timestamps, and timestamp provenance;
- per-frame translation/rotation innovations and normalized rates;
- active per-link thresholds, continuous severity, event, and classification;
- computable, high-quality, reset, gap, and excluded frame indices;
- discontinuity count/rate and median, p95, and maximum severity;
- calibrated/uncalibrated state and the independent video-level discontinuity
  decision;
- shared-camera/reconstruction diagnostics across links.

An optional standalone `cad_rigidity.json` may be written only in diagnostic
mode. It is not a second authoritative report.

Canonical point clouds are derivable from the geometry cache, segmentation,
and pose archive and are not duplicated by default. Optional debug clouds use
flat numeric arrays plus `int64` offsets, never pickled NumPy object arrays.

### Cache identity

The pose/metric cache key includes:

- source-video and segmentation SHA-256;
- geometry-cache identity and schema;
- CoTracker query-manifest hash, query-sampling settings, and checkpoint hash;
- all six CAD SHA-256 values;
- FoundationPose source revision and checkpoint hashes;
- mesh-loader and canonicalization code fingerprints;
- complete scale, pose, filtering, rendering, and metric configuration.

Writes are atomic. A partial or mismatched cache is rejected.

## 16. Failure Semantics

Per-frame rejection reasons are explicit enums, including:

```text
mask_absent
mask_too_small
mask_truncated
insufficient_depth
foundationpose_failed
invalid_rigid_transform
catastrophic_pose_mismatch
invalid_or_missing_timestamp
pose_innovation_gap
estimator_discontinuity
estimator_reset
insufficient_anchor_initialization
insufficient_canonical_points
insufficient_shape_support
```

Per-link states are:

- `complete`: continuous score and calibrated binary decision are valid;
- `uncalibrated`: continuous score is valid but no compatible threshold exists;
- `unscorable`: pose/surface coverage requirements were not met;
- `missing`: the required link mask is absent.

For the six-link benchmark, missing any of `link2` through `link7` makes the
video incomplete unless an explicit diagnostic-only partial mode is requested.
No missing or failed link contributes a numeric zero.

## 17. Validation Plan

### CPU unit tests

1. DAE hashes, units, `Z_UP`, instance counts, baked transforms, bounds, and
   diameters match Section 4.
2. Random SE(3) transforms and scales recover synthetic CAD-frame points with
   canonicalization error below `1e-10 m` in float64.
3. Depth back-projection agrees with world-pointmap inversion within the stated
   tolerance.
4. Source resize plus crop maps masks and intrinsics consistently; direct
   source-to-final resizing is included as a negative test.
5. The CoTracker query manifest binds deterministically to the same ordered CAD
   anchors, and joint-query/exact-group start from identical query IDs.
6. Rigidly transformed partial CAD observations have unchanged near-zero
   residuals; translation and rotation do not affect the result.
7. Known anisotropic stretches, local bulges, bending, and collapse yield
   monotonic proportional-strain residuals at 1, 2, 5, and 10 percent.
8. Missing rear surfaces and synthetically occluded surfaces are not penalized.
9. Synthetic translation, rotation, and uniform scale changes leave the CAD
   shape score unchanged; uniform temporal scale changes remain visible to the
   existing PDI scale component.
10. A moving-camera/static-link sequence has near-zero world-frame pose
    innovation after camera-motion removal; omitting or mis-scaling
    `T_W_from_C` translation is a required negative test.
11. Constant linear and angular velocity at multiple speeds and frame rates has
    near-zero pose innovation. Injected corrections just below, equal to, and
    above each threshold exercise the exact `>` boundary.
12. Pose-quality collapse and re-registration reset are classified as estimator
    events, while a high-quality synthetic rigid jump is classified as motion
    discontinuity and never as shape deformation.

### GPU synthetic integration tests

Render every link at known camera poses and known global scale, then run the
real FoundationPose backend.

Acceptance gates:

- at least 95 percent pose-valid frames;
- median ADD-S below `0.02 * D_link` for each link;
- recovered synthetic global scale within 2 percent;
- dimensionless proportional-strain residual below `0.01` after the full RGB-D
  path;
- pose and score arrays are deterministic within recorded numeric tolerances
  over three runs with the same seed.

### Real-video gates

1. Verify exact link masks first. In particular, `link7` must exclude the
   gripper because `link7.dae` contains only the official link7 visual geometry.
2. Calibrate thresholds only on non-deformed control videos using held-out
   video splits.
3. Require at most a 1 percent false-positive rate per link on held-out control
   videos by construction, then report the achieved confidence interval.
4. Evaluate synthetic deformations with known type, location, magnitude, and
   duration. Report detection rate per link and deformation magnitude.
5. Compare `cad-canonical-v1` against the existing point-pair rigidity metric;
   retain both histories and explain every PDI/grade change.
6. Run ablations for `metric-prior` versus `video-global-cad`, pose tracking
   versus per-frame registration, and relational strain versus absolute surface
   residual diagnostics.
7. Measure pose-discontinuity false positives separately for smooth robot
   motion, deliberate abrupt rigid motion, moving cameras, occlusions, and
   FoundationPose re-registration frames.

The method remains experimental until these gates pass. A visually plausible
overlay alone is not validation.

## 18. Implementation Order

1. Implement and test the shared, strict DAE loader.
2. Extend MegaSAM cache schema with aligned RGB, camera depth, full intrinsics,
   and exact resize/crop metadata;
3. Implement pure canonicalization functions inside `cad_rigidity_audit.py` and
   add synthetic tests;
4. Pin and install FoundationPose in an isolated GPU environment;
5. Implement the pose artifact worker, scale calibration, and pose validation;
6. Extend `motion_audit.py` with world-pose composition and the pure,
   timestamp-aware SE(3) discontinuity metric;
7. Implement visible CAD sampling and `cad-canonical-v1` residuals;
8. Integrate both versioned metrics into the multi-object pipeline while
   keeping pose discontinuity outside PDI aggregation;
9. Run synthetic GPU gates, then real non-deformed controls;
10. Generate and pin threshold configuration before enabling binary deformation
   decisions or using the new component in reported PDI grades.

No step should enable a scored result before its preceding coordinate, scale,
and cache invariants are tested.
