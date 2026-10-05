# V1 rigidity replay contract

The interactive replay audits one source video, one CoTracker mode, and one scored link per HTML page. V1 selects 3D anchor pairs at frame 0. The exporter repeats that deterministic selection from the saved tracks, masks, and MegaSAM pointmaps, then requires the complete reconstructed rigidity history to match `metrics.json` before writing a page. It never substitutes a new score.

## Inputs and identity

- `metrics.json` supplies `modes[mode].objects[link]`, including the saved rigidity strategy and `breakdown.volume_history`.
- `cotracker_<mode>.npz` supplies source-video-pixel tracks, visibility, object names, and offsets.
- `segmentation.npz` supplies boolean `object_masks[T,L,H,W]` and names.
- The scored MegaSAM cache supplies world-coordinate pointmaps, camera poses, and focal length. The original video supplies the same `T` frames and a positive FPS.

Object names and frame counts must agree across inputs. The link's tracks are `object_offsets[i]:object_offsets[i+1]`; map them to the pointmap grid with endpoint-preserving scaling and resize masks with nearest-neighbor sampling. Export a pair page only for a complete link scored with V1's 3D pairwise strategy.

## Scoring evidence

Each pair has `track_i`, `track_j`, and its frame-0 `baseline_distance`. A scored frame records the contributing `pair_indices`, `distance_ratios`, median ratio, median absolute deviation, and rigidity score. At least three visible pairs are needed for a fresh score. Otherwise V1 carries the prior frame's score and records the frame in `carried_frames`. The final rigidity score is the mean of frames 1 onward, including carried scores. Frame 0 is the reference and is excluded.

The pair JSON records the selected pairs, scored frames, carried frames, full history, final rigidity score, overall PDI score, and SHA-256 of the source `metrics.json`. The HTML displays the same pair IDs and frame ratios. A missing fresh score must not be presented as a fresh observation.

## Display and output

The page shows the reconstructed whole robot in the source camera view beside the synchronized original video. It offers 3D orbit, seeking, playback, pair selection, a history chart, and downloadable evidence JSON. Pair endpoints and lines use the same color. Orange and red show increasing deviation from the frame median; grey means the pair did not contribute a fresh measurement. Display point-cloud sampling may change rendering density but never pair positions or metrics.

With replay enabled, the scorer writes `replay/combined_<mode>.mp4` plus `replay/interactive_<mode>/index.html`, one `<link>_<mode>.html` and `<link>_<mode>_pairs.json` per eligible link, `source.mp4`, and local `plotly.min.js`. The interactive bundle works offline; the run manifest records its paths. Fail export when identities, dimensions, FPS, or reconstructed rigidity history disagree with saved scoring artifacts.
