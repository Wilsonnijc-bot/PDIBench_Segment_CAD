# A/B/C/D Pipeline Verification Plan

> Status: proposed validation protocol. The shared multi-object pipeline is
> unverified until the required GPU experiments in this document pass.

## Objective

Validate the native shared multi-object PDI implementation without confusing
intentional CoTracker method differences with regressions.

The experiment uses one source video and one frozen SAM3 segmentation archive.
Every run must use the same video bytes, selected link mask, CoTracker
checkpoint, MegaSAM geometry, PDI configuration, software revision, and GPU.

## Experiment Matrix

| Run | Segmentation input | CoTracker behavior | Purpose |
| --- | --- | --- | --- |
| A | One selected link | Original PDI-Bench | Legacy reference |
| B | The same one-link mask | New `joint-query` | Closest regression/parity comparison with A |
| C | The same one-link mask | New `exact-group` | Measure the intentional foreground/background isolation effect |
| D | All seven named link masks | New `exact-group` | Verify that other link groups cannot influence an isolated foreground group |

Run A from the pristine `PDI-Bench-original/` checkout. Run B, C, and D from
`PDI-Bench-edited/`.

## Why A Versus B Is The Legacy Parity Gate

The original `TrackWrapper.infer` concatenates one foreground query group and
one background query group into the same CoTracker predictor call. Therefore,
new one-object `joint-query` is the closest equivalent execution.

One-object `exact-group` is not expected to reproduce the original tracker
exactly because it evaluates foreground and background as isolated update
groups. A versus C is useful as a behavioral comparison, but it is not a strict
regression test.

## Frozen Inputs

Before any run, record and verify:

- source-video SHA-256;
- segmentation archive SHA-256;
- selected link name and object ID;
- exact per-frame mask tensor used by A, B, and C;
- seven-link object ordering used by D;
- CoTracker checkpoint SHA-256;
- MegaSAM, Depth Anything, and RAFT checkpoint identities;
- edited and original Git revisions;
- CUDA, PyTorch, CoTracker, and GPU versions;
- complete YAML configuration;
- deterministic query manifest.

Create the one-link archive by selecting one object axis from the canonical
seven-link archive. Do not rerun segmentation independently for A, B, or C.

## Control The Shared Geometry

The strict comparison should read the same validated MegaSAM arrays:

```text
pointmaps[T,Hm,Wm,3]
camera_poses[T,4,4]
focal_length
```

Do not compare a cold reconstruction from one run with a separately generated
reconstruction from another run. First prove that both metric paths consume the
same cached arrays. Then run separate cold-cache and warm-cache timing trials.

## Required Comparisons

### A Versus B: Original Regression

Compare in this order:

1. Per-frame masks and mask measurements.
2. Foreground and background query coordinates.
3. Pointmaps, camera poses, focal length, and normalized object depth.
4. Foreground and background tracks and visibility.
5. Scale, trajectory, rigidity, and VP histories.
6. Final PDI component values, score, and grade.
7. Tracking runtime and peak allocated GPU memory.

Any difference must first be attributed to a concrete input or method change.
Do not use final PDI agreement alone as proof of correctness.

### B Versus C: CoTracker Isolation Effect

This comparison measures the effect of removing foreground/background spatial
attention. Differences in foreground tracks, visibility, rigidity, and VP are
allowed and must be reported. Scale and trajectory should remain unchanged when
their segmentation and geometry inputs are identical.

### C Versus D: Exact-Group Isolation Invariant

For the selected link, C and D must use identical foreground query coordinates.
The link's foreground tracks and visibility should then agree within numerical
tolerance because exact-group does not mix update tokens across link groups.

The selected link's scale, trajectory, and rigidity results should also agree.
VP may differ because C samples background outside one link while D samples
background outside the dilated union of all seven links. With the current
default `w_vp: 0.0`, that VP difference must not change the final weighted PDI.

### A Versus D: End-To-End Reference

Report this comparison, but do not treat it as an equality test. It combines
several deliberate changes:

- isolated rather than foreground/background-joint CoTracker updates;
- one shared background region outside the seven-link union;
- one shared reconstruction;
- corrected mapping from video-pixel tracks into the pointmap grid for rigidity.

## Initial Numerical Tolerances

| Quantity | Initial acceptance threshold |
| --- | --- |
| Masks | Exact equality |
| Object names and IDs | Exact equality |
| Query manifest | Exact equality |
| Reused geometry arrays | Exact equality |
| Mean matched-track error | At most `0.1` pixel |
| Maximum matched-track error | At most `0.5` pixel |
| Visibility agreement | At least `99.9%` |
| Metric component delta with identical inputs | At most `1e-4` |
| Grade | Exact equality for the A/B parity run |

If deterministic CUDA execution still produces larger track differences, retain
the raw archives and establish a justified tolerance from repeated runs. Do not
silently loosen thresholds based on a single failure.

## Known Expected Difference: Rigidity Coordinate Mapping

The edited pipeline maps CoTracker coordinates from the source-video grid into
the MegaSAM pointmap grid before sampling 3D anchors. The original implementation
assumes those grids have the same resolution.

When their resolutions differ, compare both:

- the unmodified legacy rigidity result, to document historical behavior;
- a corrected legacy calculation using the edited coordinate mapping, to test
  metric parity after aligning inputs.

This known correction must not be hidden by forcing the edited result to match
the old indexing behavior.

## Timing Protocol

Run timing separately from correctness checks:

1. One cold-cache trial for A, B, C, and D.
2. At least three warm-cache trials per run.
3. Synchronize CUDA before and after each timed model region.
4. Record median model time and total tracking time.
5. Reset and record peak allocated GPU memory for every trial.
6. Record SAM3 and MegaSAM time separately from CoTracker and metric time.

Do not include model installation, checkpoint download, file transfer, or replay
rendering in CoTracker speed comparisons.

## Acceptance Gates

The pipeline remains unverified until all of these gates pass:

- A/B inputs are proven equivalent and unexplained metric regressions are zero;
- C/D foreground query manifests match for the selected link;
- C/D foreground tracks satisfy the stated tolerance;
- no rigidity pair crosses link identity boundaries;
- both modes produce complete per-link reports and numeric track archives;
- cold- and warm-cache timing and peak-memory results are recorded;
- all failures and expected differences are retained in the final report.

## Required Artifacts

```text
verification/
|-- input_manifest.json
|-- run_A_original_single/
|-- run_B_edited_joint_single/
|-- run_C_edited_exact_single/
|-- run_D_edited_exact_seven/
|-- comparisons/
|   |-- A_vs_B.json
|   |-- B_vs_C.json
|   |-- C_vs_D.json
|   `-- A_vs_D.json
`-- VERIFICATION_REPORT.md
```

Each run directory must retain its exact command, configuration, metrics,
timing, query manifest, track archive, geometry identity, and console log.
