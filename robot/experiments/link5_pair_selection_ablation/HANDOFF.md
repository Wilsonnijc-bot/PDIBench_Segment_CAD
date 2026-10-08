# Handoff prompt — run two Link5 rigidity pair-selection ablations

## Current retained outcome

The experiment completed: **40 primary videos + 1 additional video, 82/82 scores**.
The original planning instructions below predate later user steering. The final
run applies the shared refined depth filter to **every original frame**, using
`filter_video_depth.py`, `score_depth_v2.py` and `run_scores_depth_v2.py`; both pair
selectors share identical masks, queries, tracks, geometry and depth support.
The measured eight-worker run shared **one H200**. The five-worker baseline and
earlier gradient-gated scores remain historical capacity/validation evidence.

Results: [retained report](results/20261007/REPORT.md),
[interactive replay](results/20261007/report/index.html), and
[AB-label analysis](results/20261007/analysis/forearm_AB_correlation.md).
The root `results/link5_pair_selection_ablation_v1` is a compatibility link.
See [README.md](README.md) for the final implementation and [AGENTS.md](AGENTS.md)
for the user's explicit preservation instruction. Restoration does not launch a new run.

## Original planning instructions

Work in `/Users/nijiachen/Downloads/deformationdetection`. Read `AGENTS.md` and `agent.md` before scheduling GPU work. Complete the full-video experiment below, not just previews or a plan. The user authorizes ERIS, **one H200, five videos in parallel**. Five concurrent video workers total share the one GPU; do not launch five per method or allocate five GPUs. Preserve original masks/results. All new trial code goes under `robot/experiments/link5_pair_selection_ablation/`.

## Scope and exclusions

Use the current validated mask cache only, locally:
`results/link5_shape_codebook/round_four_frame0/mask_cache_manifest.json`.
The experiment manifest is `results/link5_pair_selection_ablation_v1/manifest.json`.
Exclude exactly `LVP_ROBOWM_0010`, `LVP_ROBOWM_0030`, `COSMOS3_0021`, `COSMOS3_0035`
COSMOS2.5_0010
: user rejected those masks. Preserve their existing artifacts. There are **41 remaining selected-45 videos**, plus **COSMOS2.5_0018 as a separate additional cohort**. Evaluate all 42 available included videos, but report the main cohort of 41 separately. This yields 82 primary method/video scores and 2 additional scores, with explicit failure rows if any fail.

## Implemented methods and exact locations

- `robot/experiments/link5_pair_selection_ablation/balanced_v0.py`: the earlier proposed graph. Target 10 longitudinal + 10 transverse + 10 local diagonal; two transverse and two diagonal midpoints per each of five sections; maximum point degree 3. Preserves the original thresholds, greedy objective and tie breaks. Finite world-position input enables the original 2%-of-3D-diameter baseline floor. Quota deficits are reported, not silently filled. Port parity against the old prototype passes on all five audited historical examples.
- `robot/experiments/link5_pair_selection_ablation/refine_v1.py`: current transition-focused initialization with objective rounded to 12 decimals before ID tie-break (prevents insignificant BLAS rounding from changing tied choices), named **pair selection refine v1**. Mask PCA/width profile detects narrow-to-broad transition; 6 upper-flank bridges, 6 lower-flank bridges, 4 transverse in that band, 6 global long, 4 other transverse, 4 other diagonal. Maximum degree 3. Mask geometry is a prior, not a reliable anatomical label in every case. Flag ambiguous/clipped geometry, retain failures.
- `selectors.py`: `select_pairs(method, xy, xyz, mask0)` returns pairs whose i/j index the supplied eligible point array, plus coverage statistics. Methods: `balanced_v0`, `refine_v1`. Uses frame zero only. xy and mask must share the same pixel grid. Use the common CoTracker/pointmap grid in the actual experiment. Fixed pixel floors (6 balanced / 8 refine) are measured on that grid. The source-resolution previews may differ after resizing and reliable-anchor gating; save the actual selected graph and disclose this distinction.
- `score.py`: isolated copy of baseline `audit_3d_rigidity_cv`, replacing only ranking with the chosen selector. Default frame-zero anchor gate is baseline **v1 world-Z gradient + visibility with visible-only fallback**. Preserve this common gate in both methods. Do not confuse point_filter_version='v1' with pair_method='refine_v1'. Uses existing tiny-baseline >1e-3 rule, visibility threshold >0.5, MAD(ratio)/(median(ratio)+1e-6), carry previous score with fewer than 3 available pairs, and final temporal mean excluding frame zero. No production scorer was changed. `provenance.json` records source hashes.
- `run_scores.py`: runnable CPU scoring CLI for BOTH methods from shared raw preparation arrays; defaults to five processes. Saves per-method evidence, history, score, coverage/carry statistics, summary JSON and `rigidity_scores.csv`. Failure rows are explicit, and the command exits nonzero if any failed.
- `prepare_manifest.py`: rebuilds the inclusion manifest while retaining mask/video identities and the four exclusions.
- `preview.py`: generates mask/points and both graphs without GPU inference. Existing comparison viewer: `results/link5_pair_selection_ablation_v1/previews/index.html`.
- Tests: `/tmp/link5-analysis-env/bin/python -m unittest discover -s robot/experiments/link5_pair_selection_ablation/tests -v` (local environment). Use the corresponding ERIS Python remotely.

## What is done versus still needed

Both selector implementations, scoring adapter, manifest, CPU scoring batch runner and frame-zero previews exist. **No new full-video rigidity scores have been generated.** Implement and validate the missing GPU preparation adapter/launcher under this same experiment folder. It must generate shared MegaSAM world pointmaps and raw CoTracker arrays for the current masks/queries. Do not assume the shape-codebook preparation script is a complete rigidity pipeline: inspect its interfaces; it may subsample temporal inputs. Do not run shape-codebook training. Use the full original frame sequence and preserve original zero-based indices.

Original historical raw arrays and prior geometry were deleted. Regenerate geometry using MegaSAM and tracking using CoTracker; reuse the exact selected cached masks without any new VLM/SAM calls. Verify source-video SHA-256 and selected mask NPZ SHA-256 against the manifest (some masks came from later attempts; never assume attempt-0001). Both variants must share the same source, mask, initialized queries, tracks, visibility and pointmaps. Run expensive inference once per video and score both graphs from it; do not recompute stochastic tracking separately for each graph.

For reproducibility, reuse the new initialized source-pixel queries and IDs saved at:
`results/link5_shape_codebook/round_four_frame0/pair_initialization_frame0/<case>/pairs.json`
(keys `query_points_xy`, `point_ids`, `mask_sha256`, `source_video_sha256`). Those queries came from the existing SIFT/Shi–Tomasi/grid sampler on the latest frame-zero mask's largest component eroded 2 source pixels, nominal 100 queries at max image dimension 880. Scale x/y explicitly to the tracker grid, query time=0, and preserve IDs through every resize/filter. Avoid silently invoking a tracker wrapper that replaces these queries or drops entire tracks based on future visibility. Inspect `infrastructure/shared/inference/cotracker_core.py` `_run_queries` and the caller. Initialization and pair selection must not depend on later frames or deformation labels.

Do not change score aggregation in this ablation. Do not accidentally apply the new shape-codebook depth filter to only one method. If improved depth gating or aggregation is desired, that is a separate ablation. Save real tracker visibility separately from finite 3D/exported point availability. The old availability/export flags are not raw tracker visibility.

## Input contract and scoring command

For every included video, save `<case>.npz` with:
- `pointmaps`: float [T,H,W,3], MegaSAM world-space positions;
- `tracks_2d`: float [T,N,2], CoTracker coordinates on that exact H,W grid;
- `visibility`: [T,N], original CoTracker visibility/confidence;
- `masks`: bool [T,H,W], nearest-neighbor resizing of exact cached masks.

Companion `<case>.json` must include `video_id`, `source_video_sha256`, `mask_sha256` matching the manifest. Also record query IDs, source/grid coordinate transforms, frame indices, raw file checksums, model revisions/checkpoints, environment, seeds and actual preprocessing settings. Preserve raw track arrays before any filtering. The scorer rejects nonfinite arrays: investigate them rather than silently substituting scores. Avoid destructive cleanup of raw evidence.

Run from repository root:
```
python -m robot.experiments.link5_pair_selection_ablation.run_scores \
  --manifest results/link5_pair_selection_ablation_v1/manifest.json \
  --inputs results/link5_pair_selection_ablation_v1/shared_inputs \
  --output results/link5_pair_selection_ablation_v1/rigidity \
  --workers 5
```
The five-worker CPU command does not itself launch GPU inference; build the corresponding five-video shared-H200 preparation launcher first. Pointmap RAM is large: measure host memory as well as GPU memory, do not mistake successful CPU previews for GPU capacity validation.

## ERIS locations and scheduling

SSH: `zy992@eris2n7.research.partners.org`. Do not store credentials in code/docs/logs. The earlier session used SSH ControlPath `/tmp/codex-link5-eris-%C`; reuse it if alive, otherwise obtain authentication through the user's configured SSH/session.

Remote workspace:
`/PHShome/zy992/Wilson/deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006`
Remote latest cache is `<workspace>/results/link5_shape_codebook/` (unlike the local path, no `round_four_frame0` suffix). Verify its manifest and NPZ hashes before staging an experiment-specific manifest. Do not copy local absolute paths into a remote launcher unchanged.

Source videos:
`/PHShome/zy992/Wilson/deformationdetection/inputs/videos/{LVP,COSMOS2.5,COSMOS3}/{NNNN}.mp4`.
`LVP_ROBOWM` maps to directory `LVP`.

Previous CPU-only initialization artifacts:
`/PHShome/zy992/Wilson/deformationdetection/pair-init-frame0-20261007/output/`.
The previous allocation 5660284 used node erishpc-gpu-001, one H200 143771 MiB. **Do not assume that allocation is still live or idle.** Inspect scheduler and current GPU processes, use an appropriate authorized allocation, and never cancel someone else's work. New ablation code currently exists locally; stage it to a new isolated remote experiment directory and freeze launcher/code snapshots before launch. Read available experiment instructions/skills and record substantive GPU work in the project ledger without overwriting pre-existing entries.

Run five concurrent video workers sharing one H200 as explicitly requested. Record device UUID, allocation ID, GPU count=1, workers=5, peak memory, utilization, CPU/RAM/I/O pressure and throughput. Start with a measured capacity check and diagnose OOMs; don't claim five safe lanes from CPU timings. Keep GPU allocation separate from worker count. Use isolated per-video scratch and logs, completion checks, failures and stop propagation to active and queued dependent jobs. A stop request cancels this experiment's active/dependent work only.

## Required final outputs

1. Rigidity score for each included video under each method: paired CSV/JSON and comparison table, primary 41 versus additional 0018 explicitly distinguished; excluded cases listed with reasons, failures never silently dropped.
2. Per-frame score, selected/available pair counts, raw visible point counts, finite sampled 3D point counts and carried-frame flags. Include actual selected point IDs, endpoint trajectories, 3D baselines and distance ratios; `score.py` evidence is a starting point, augment the raw-array diagnostics as needed.
3. Actual frame-zero pair overlays plus synchronized video/score evidence for the originally reported successes/misses (COSMOS2.5_0010 around zero-based25; COSMOS3_0010 around101; LVP_ROBOWM_0015; COSMOS2.5_0015; COSMOS3_0015 around73). Verify replay one-based display versus zero-based indices and inspect neighboring frames.
4. Explain whether differences are graph coverage, depth/tracking error, availability/carry, or aggregation. More available pairs do not by themselves prove a more meaningful score. Mark quota deficits; no forced 30-pair claims.
5. A separate reviewable experiment report/viewer under `results/link5_pair_selection_ablation_v1/`, leaving the canonical shape-codebook replay unchanged. Open the local report, provide its path, and say explicitly whether it was published. Do not report detection improvement until supported by the full-video evidence.

## Negative guard review, 2026-10-08

`link5_point_guard.py` now requires only N1 inside the SAM green box or its
5-pixel margin. N2 may lie outside the box. Negative parsing failures (including
bare `REJECT`) and failed N1 checks retry identical prompts/images, with three
total attempts. Exhausted retries stop that case; API transport failures retain
their existing handling. `negative_only=True` preserves supplied cached positives
and makes no positive-guard call.

The ten-case API-only review is local at
`results/link5_pair_selection_ablation_v1/guard_n1_retry_20261008/index.html`.
It reuses nine cases' prior test calls and resumes COSMOS2.5_0010: initial bare
`REJECT`, then valid negatives on attempt 2, with identical retry inputs verified.
LVP_ROBOWM_0030 selects saved attempt 2 under the N1-only rule (three calls were
actually made under the earlier two-point rule). LVP_ROBOWM_0010 retains its
previous negatives per user review. All displayed positives remain cached;
no SAM, mask propagation, rigidity scoring, or GPU work was run for this review.
Nine new negative proposals pass the N1 rule; one case retains previous points.
This is point-placement review, not evidence of improved masks or scores.
