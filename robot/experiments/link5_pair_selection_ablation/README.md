# Link5 pair-selection ablations

Implementation and executable handoff: [HANDOFF.md](HANDOFF.md).

- `balanced_v0`: earlier 10 long / 10 cross-width / 10 local diagonal proposal.
- `refine_v1`: current mask-transition-focused 16 local / 14 remaining pairs.
- `score.py`: unchanged baseline MAD/median, anchor gating and carry policy; new pair selector only.
- `run_scores.py`: both scores from shared raw tracking/geometry inputs.
- `preview.py`: current masks, identical query positions, two graph previews.

The current handoff additionally lists `COSMOS2.5_0010` as excluded, although its
older counts still say 41 primary videos. `prepare_manifest.py` follows all five
explicit exclusions, yielding 40 primary + 1 additional video. Existing excluded
artifacts are preserved. No production defaults changed.

`prepare_gpu.py` generates full-sequence MegaSAM CVD world pointmaps and raw
CoTracker outputs with the saved explicit frame-zero queries. It rejects source,
mask and query hash mismatches, preserves query IDs, scales queries directly to
the native pointmap grid, and saves unfiltered tracker visibility. Both scorers
read the same input NPZ. No SAM/VLM calls or shape-codebook training are used.

`run_gpu_batch.py` requires one visible H200 and a Slurm allocation. It starts with
one longest-video capacity measurement, then a five-video wave, then runs the
remaining videos with five workers total. Failed capacity waves prevent the
remaining GPU work from launching. GPU memory/utilization, host memory, CPU/I/O
pressure and throughput are saved as JSONL. SIGINT/SIGTERM cancels active process
groups and prevents queued videos and dependent CPU scoring from starting.
`eris_job.sh` runs it in an allocation-local tmux session with an immutable shell
snapshot, durable logs and an exit receipt. Stage all source into a new isolated,
frozen remote workspace before submission; never overwrite a running snapshot.

The real Eris pilot and five-video capacity wave passed on one H200. Six early
baseline method/video scores were checked, and 19 full-sequence raw inputs were
complete before the original allocation expired. The revised full batch has since completed: 82/82 scores, no failures.

## User update: shared depth filtering and eight-worker capacity

The user superseded the handoff’s fixed baseline gate: reuse the cached masks,
completed native MegaSAM depth/world pointmaps and raw CoTracker tracks, and apply
the existing Link5 refined depth policy to every original frame.
`filter_video_depth.py` reads exact receipted native CVD depth, erodes the intact
source mask by two source pixels, and applies the shared ray-normalized 32-neighbor
density filter. It saves full-grid support/rejection reasons, query support, cloud
quality flags and per-frame statistics. It never smooths depth or moves XYZ.

`score_depth_v2.py` selects both graphs using frame-zero raw visibility AND depth
support, with no gradient or visible-only fallback. Later depth support gates
pair availability without reselection. `run_scores_depth_v2.py` preserves raw
visibility separately and retains the baseline distance-ratio MAD/median,
fewer-than-three-pair carry and frame-zero-excluded temporal mean. Cloud-quality
failures disable depth availability; insufficient frame-zero anchors fail
explicitly. Earlier baseline scores remain preliminary, in separate validation
folders. Nine local tests and four allocated Eris depth-support tests passed.

`run_gpu_batch_v2.py`, `eris_job_v2.sh` and `continue_eris_v2.sh` preserve original
running code and test eight concurrent remaining videos on one H200. The measured
trial is retained only if memory has headroom and source-frame throughput exceeds
the real five-worker baseline; otherwise subsequent work uses five. Remaining
clips differ from baseline identities/lengths, so this is a scheduling measurement
and not a controlled scientific speedup claim. GPU preparation is separate from
CPU all-frame filtering/scoring and `build_report_v2.py`.

## Completed results and restored replay

41 videos / 82 scores completed with identical all-frame refined depth support.
GPU5677575 and streamed CPU5678539 exited0; eight shared-H200 lanes retained,
peak104529/143771MiB. CPU-only replay export5681398 also exited0.

label_correlation.py reads physical AB only and saves Pearson, Spearman and
AUROC with explicit0.5 policies.40 videos match labels; additional0018 is unlabeled.
Exact binary32: AUROC balanced0.708333, refine0.779167.

build_previous_replay.py reuses the maintained
infrastructure/shared/replay/rigidity_replay.html template and Plotly interaction
code. The rejected canvas HTML was removed from build_report_v2.py.
export_replay_context.py reuses source-colored scene sampling, with saved filtered
support on every original frame. Display samples never modify scoring arrays.
prepare_playback.py makes browser display copies with verified source frame
counts, dimensions, FPS and PTS. Original videos remain.

Open results/link5_pair_selection_ablation_v1/report/index.html for video/method
selection, synchronized source/point-cloud views, 3D orbit, pair inspection and
score-history seeking. Tables and label analysis are under rigidity/ and analysis/.
Scientific hashes and replay QA: metadata/final_review.json. Local, not published.

## Retention and ownership

This is a retained research experiment, reinstated by the user on October 8, 2026.
Preserve its code, scores, evidence and interactive replay during workspace cleanup.
Local results belong to `results/20261007/` within this experiment; the repository-root
`results/link5_pair_selection_ablation_v1` is a compatibility link. The full original
geometry, tracker arrays and scratch provenance remain in the unchanged ERIS v2
workspace. Guard-only reviews now belong to `../link5_guard_review/`; compatibility
links keep their historical experiment URLs working. See [AGENTS.md](AGENTS.md) and [RESTORATION.md](RESTORATION.md).
