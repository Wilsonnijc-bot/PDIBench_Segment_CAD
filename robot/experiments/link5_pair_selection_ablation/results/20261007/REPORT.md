# Link5 pair-selection rigidity comparison

82/82 method/video scores complete; 40 primary videos and 1 additional video.

Metric: Mean per-frame MAD(distance ratio)/(median(distance ratio)+1e-6), excluding frame0. Both methods use identical cached masks, saved frame-zero query IDs, raw CoTracker visibility/tracks and full-sequence CVD world pointmaps. Both methods use the same refined ray-normalized 32-neighbor density filter on every frame, including two source-pixel mask erosion. Frame-zero eligibility is depth support plus raw visibility, without the old gradient/fallback gate. Tiny-baseline rejection, fewer-than-three-pair carry and temporal mean are unchanged.

The five explicit handoff exclusions include COSMOS2.5_0010, so the older 41-primary count is superseded by 40. Its old artifacts remain preserved; no new diagnostic score is invented for the excluded video.

Actual graphs may differ from CPU source-resolution previews because native-grid resizing and the common frame-zero depth-support/visibility gate affect eligibility. Quota deficits are retained.

Interpretation: graph choice changes which spatial deformations and reconstruction/tracking errors enter MAD. More pairs alone do not establish detection improvement. A coherent scale change produces nearly equal distance ratios and can be suppressed by MAD; minority pair changes can also be suppressed. Occlusion can carry a previous score. The unchanged temporal mean can dilute brief events.

Inspect neighboring_frame_diagnostics.json and the synchronized viewer around zero-based101/display102 for COSMOS3_0010 and zero-based73/display74 for COSMOS3_0015. For the other two guide videos, automatically centered neighborhoods are score peaks, not manually labeled deformation onset.

Per-method evidence includes coverage/geometry flags, actual point IDs/baselines, frame diagnostics, distance-ratio trajectories and selected graph PNGs. Finite 3D samples are numerical availability, separate from raw tracker visibility and filtered depth support; finite depth can still be geometrically wrong.

Sources, model/checkpoint hashes, preprocessing, query coordinate transforms, seeds and environment are in shared_inputs/*.json. Original high-precision arrays and trajectories are preserved; viewer coordinate rounding is display-only.

Viewer: report/index.html. Tables: rigidity/paired_scores.csv and rigidity/rigidity_scores.csv. This report is local and has not been published.

## Restoration and preservation — October 8, 2026

The user reinstated this experiment after workspace cleanup. Code, scores,
trajectories, depth support, videos, replay context and run records were recovered.
860 files were verified against ERIS hashes; all 541 scientific files remain
unchanged. The previous Plotly replay has 82 method pages across 41 videos.
Browser copies preserve all 4,489 source frames and original timestamps.

[AB analysis](analysis/forearm_AB_correlation.md) ·
[Restoration receipt](metadata/restoration.json) ·
[Preservation report](../../RESTORATION.md). No inference was rerun.
