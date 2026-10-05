# D: isolated multi-link tracking specification

## Scope

**D** names the historical seven-link, `exact-group` execution of the edited
PDI pipeline. It uses one source video, one named SAM3 mask archive, one shared
MegaSAM reconstruction, and a separate PDI report for each rigid link. This is
the method represented by `pdi_eval.v1` with `tracking_modes=("exact-group",)`.
The archived [ABCD verification plan](archive/verification/ABCD_PIPELINE_VERIFICATION_PLAN.md)
is a proposed comparison protocol; it is not evidence that D has passed its
GPU acceptance gates.

## Inputs and identity

- `object_masks` has shape `(T, N, H, W)`. For the D experiment, `N = 7` and
  object names and IDs identify `link1` through `link7`. Preserve overlapping
  masks at joints; do not collapse them into one articulated foreground.
- Every link uses the same unmodified RGB video and the same world-coordinate
  MegaSAM pointmaps, camera poses, and intrinsics. Geometry is inferred once.
- The same checkpoint, configuration, mask archive, and deterministic query
  manifest must be used when comparing D with B or C.

## Tracking contract

1. Resize the first frame and each first-frame link mask to the CoTracker grid,
   with maximum image dimension 880 by default.
2. Sample each link's frame-zero queries inside its own mask. Default request
   is 100 per link, overridden to 256, 192, and 128 for links 2, 3, and 4 in
   `configs/default.yaml`. SIFT, Shi–Tomasi, and grid candidates are balanced
   deterministically. Counts can be lower if unique mask points are scarce.
3. Dilate the union of all link masks by five tracker-grid pixels by default.
   Sample one shared background group outside this dilated union (up to 225
   queries by default). Background means explicit CoTracker query points in
   that region; it does not remove the rest of the scene from the RGB video.
4. In `exact-group`, send each link group to its own CoTracker predictor
   forward, then send the background group to another forward. The implementation
   replays cached video backbone features across groups. For seven links this
   means eight query-group forwards and one video backbone computation, subject
   to the model's internal feature chunks.
5. CoTracker query tokens for one link never appear in another link's predictor
   forward. Background query tokens never appear in a link forward. The shared
   video features still encode the full RGB scene.
6. Scale tracks back to source-video coordinates and apply the historical
   track-quality filter. Save each named link group and the background group
   separately in the track archive.

## Per-link scoring contract

For each link, pass only that link's mask, foreground tracks, and visibility to
rigidity. Map track coordinates from the source-video grid to the MegaSAM
pointmap grid before sampling 3D anchors. Rigidity pairs must contain two
anchors from the same link. There is no score for the seven-link union and no
cross-link rigidity pair.

The historical PDI report also calculates scale and trajectory from the link's
mask and shared geometry. Background tracks participate in the perspective/VP
calculation; they are not an input to the rigidity audit. The current default
configuration assigns VP weight `0.0`. A link may fail its own evidence gates
without preventing other link reports.

The C versus D isolation expectation is narrower than full report equality:
with identical selected-link queries, checkpoint, video and geometry, the
selected link's foreground tracks and rigidity should agree within numerical
tolerance. C samples background outside one link; D samples it outside all
seven, so background and VP may differ. This remains an expectation until the
archived GPU comparison protocol is run.

## Code and callable interfaces

| Pipeline | Tracking | Rigidity | Python call | CLI |
| --- | --- | --- | --- | --- |
| `src/pdi_eval/v1/pipeline.py` | `src/pdi_eval/v1/tracking.py` | `src/pdi_eval/v1/rigidity.py` | `pdi_eval.v1.evaluate(...)` | `python -m pdi_eval.experiment score --tracking-mode exact-group` |

`src/pdi_eval/perception/cotracker_core.py` owns shared model execution, query
sampling, and feature replay. Segmentation loading, MegaSAM geometry, data
types, logging, and replay remain separate utilities. The unversioned pipeline,
tracker, and rigidity module paths are compatibility imports for V1.

## Status

This document pins the implemented D behavior. It does not claim the historical
A/B/C/D verification was completed. Keep the `V1 D` method label with every
metric artifact.
