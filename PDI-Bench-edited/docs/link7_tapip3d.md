# Selectable Link 7 tracker

The V1 scorer keeps its existing masks, selected Link 7 query points, MegaSAM
geometry cache, track quality filter, and downstream deformation metrics. The
default is `--link7-tracker cotracker3`. Select `tapip3d` to send only Link 7's
selected points to the official TAPIP3D model. Other links and shared background
remain on CoTracker3.

```bash
PYTHONPATH=src python -m pdi_eval.experiment score \
  --input /path/video.mp4 \
  --segmentation-npz /path/segmentation.npz \
  --geometry-cache-dir /path/existing-geometry-cache \
  --output-dir /path/tapip3d-output \
  --tracker-checkpoint /path/scaled_offline.pth \
  --tracking-mode exact-group \
  --link7-tracker tapip3d \
  --tapip3d-python /path/tapip3d-env/bin/python \
  --tapip3d-repository /path/TAPIP3D \
  --tapip3d-checkpoint /path/tapip3d_final.pth
```

Use the comparison runner to score both methods on the same selected points and
save labeled videos plus `comparison.json`:

```bash
PYTHONPATH=src python -m pdi_eval.v1.link7_compare \
  --video /path/video.mp4 \
  --segmentation-npz /path/segmentation.npz \
  --geometry-cache-dir /path/existing-geometry-cache \
  --tracker-checkpoint /path/scaled_offline.pth \
  --tapip3d-python /path/tapip3d-env/bin/python \
  --tapip3d-repository /path/TAPIP3D \
  --tapip3d-checkpoint /path/tapip3d_final.pth \
  --output-dir /path/comparison
```

The comparison videos label 16 spatially spread IDs for readability. The
interactive initial-point map identifies every selected query, and the raw
archives plus full-point videos under `provenance/` retain all trajectories.
`--render-only` refreshes visualizations from existing score archives without
running either tracker again.

`link7_initial_queries.npz` assigns each original query a stable zero-based
`point_id`. The TAPIP3D raw archive keeps `point_ids`, per-frame `tracks_uv`,
`trajectories_xyz` in MegaSAM world coordinates, boolean `visibility`, and
continuous `confidence`. The scored track archive keeps only points retained by
the existing downstream quality filter and saves their original `point_ids`.
TAPIP3D's additional 16 by 16 support grid remains internal to official
inference and is removed before returning the selected trajectories.

The cached MegaSAM pointmaps and camera poses provide metric depth. The adapter
recovers the original camera intrinsic matrix algebraically from the cached
pixel-to-world pointmap, since older geometry caches store only focal length.
It validates reprojection before inference and does not rerun MegaSAM.

Official implementation: <https://github.com/zbw001/TAPIP3D>, revision
`4cb7e69a1687f67d56ec3e506768f51f2c581b46` used for the initial
integration. Keep the official repository and model in a separate environment.
