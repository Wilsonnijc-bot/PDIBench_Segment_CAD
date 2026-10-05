"""Score one SAM3 task-object mask with the active V1 3D rigidity formula."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from infrastructure.shared.inference.mega_sam_wrapper import MegaSamWrapper
from infrastructure.shared.contracts.segmentation_archive import load_multi_object_segmentation
from robot.workflows.pipeline import _map_tracks_between_grids, save_track_result
from infrastructure.shared.scoring.rigidity.rigidity import InsufficientRigidityEvidenceError, audit_3d_rigidity_cv
from infrastructure.shared.inference.tracking import TrackWrapper

from object.experiments.rigidity.workflows import METHOD
from object.preprocessing.selection.selection import sha256, write_json


def score(video: Path, segmentation_path: Path, output: Path, geometry_cache: Path,
          tracker_checkpoint: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    segmentation = load_multi_object_segmentation(segmentation_path, video)
    if segmentation.object_names != ("task_object",):
        raise ValueError("expected one task_object mask channel")
    masks = segmentation.object_masks[:, 0]
    if not masks[0].any() or len(masks) < 2:
        raise ValueError("first-frame mask or video length is insufficient")
    geometry = MegaSamWrapper(device="cuda").infer_shared(
        str(video), segmentation.object_masks, cache_dir=geometry_cache)
    if geometry.frames_count != len(masks):
        raise ValueError("MegaSAM and source video frame counts differ")
    pointmaps = geometry.pointmaps
    mask0 = cv2.resize(masks[0].astype(np.uint8),
                       (pointmaps.shape[2], pointmaps.shape[1]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    initial_geometry = pointmaps[0][mask0]
    if not len(initial_geometry) or np.mean(np.any(initial_geometry != 0, axis=-1)) <= 0.5:
        raise InsufficientRigidityEvidenceError("invalid frame-zero object pointmap",
                                                valid_anchor_count=0, valid_pair_count=0)
    tracker = TrackWrapper(checkpoint=str(tracker_checkpoint), device="cuda")
    prepared = tracker.prepare_multi(str(video), segmentation.object_masks[0],
                                     segmentation.object_names, grid_size=10,
                                     bg_grid_size=15, background_dilation=5,
                                     max_dim=880, object_query_counts={"task_object": 100})
    tracks = tracker.track_prepared(prepared, "exact-group")
    foreground = tracks.object_tracks[0]
    visibility = tracks.object_visibility[0]
    if foreground.shape[1] < 5:
        raise InsufficientRigidityEvidenceError("fewer than five retained object tracks",
                                                valid_anchor_count=foreground.shape[1],
                                                valid_pair_count=0)
    mapped = _map_tracks_between_grids(
        foreground, masks.shape[1:], pointmaps.shape[1:3])
    evidence = {}
    value, history = audit_3d_rigidity_cv(
        pointmaps, mapped, visibility, masks,
        insufficient_policy="raise", point_filter_version="v1", evidence=evidence)
    if not math.isfinite(value) or not np.isfinite(history).all():
        raise ValueError("rigidity result contains nonfinite values")
    save_track_result(output / "cotracker_exact-group.npz", tracks)
    result = {
        "schema_version": 1, "status": "complete", "method": METHOD,
        "rigidity_score": value, "rigidity_history": history.tolist(),
        "rigidity_strategy": "V1 Strategy 1 (3D rigid pairwise ratios)",
        "evidence": evidence,
        "tracking": {"requested_foreground_query_count":
                     int(tracks.metadata["requested_foreground_query_counts"][0]),
                     "retained_foreground_track_count": int(foreground.shape[1]),
                     "mean_visibility": float(visibility.mean()),
                     "video_coordinate_hw": list(masks.shape[1:]),
                     "pointmap_coordinate_hw": list(pointmaps.shape[1:3]),
                     "exact_group_metadata": tracks.metadata},
        "geometry": {"cache_path": geometry.cache_path,
                     "cache_hit": geometry.metadata["cache_hit"],
                     "pointmap_shape": list(pointmaps.shape),
                     "target_depth_status": geometry.metadata["object_depth"][0]["status"]},
        "input": {"video": str(video), "video_sha256": sha256(video),
                  "segmentation": str(segmentation_path),
                  "segmentation_sha256": sha256(segmentation_path)},
    }
    write_json(output / "rigidity.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.video, args.segmentation, args.output,
                   args.geometry_cache, args.tracker_checkpoint)
    print(json.dumps({"status": result["status"], "rigidity_score": result["rigidity_score"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
