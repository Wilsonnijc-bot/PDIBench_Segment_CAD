"""Optional Link 7 TAPIP3D handoff after the existing query selection."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from infrastructure.shared.contracts.base import MultiObjectTrackResult, SharedGeometryResult
from infrastructure.shared.inference.tracking import PreparedMultiObjectTracking, TrackWrapper


def selected_link7_queries(prepared: PreparedMultiObjectTracking) -> np.ndarray:
    index = prepared.object_names.index("link7")
    queries = prepared.object_queries[index].astype(np.float32).copy()
    queries[:, 1] *= prepared.scale_xy[0]
    queries[:, 2] *= prepared.scale_xy[1]
    return queries


def save_initial_queries(path: Path, prepared: PreparedMultiObjectTracking) -> None:
    """Keep the unfiltered selected IDs for a tracker-only A/B comparison."""
    queries = selected_link7_queries(prepared)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        point_ids=np.arange(len(queries), dtype=np.int64),
        queries=queries,
        source_hw=np.asarray(prepared.original_hw, dtype=np.int64),
    )


def _intrinsics_from_cached_pointmap(
    pointmap: np.ndarray, pose: np.ndarray, focal_length: float,
) -> np.ndarray:
    """Recover MegaSAM's original K from its cached pixel-to-world mapping."""
    h, w = pointmap.shape[:2]
    camera = (pointmap - pose[:3, 3]) @ pose[:3, :3]
    valid = (np.isfinite(camera).all(axis=-1) & (camera[..., 2] > 0)
             & np.any(pointmap != 0, axis=-1))
    yy, xx = np.indices((h, w))
    rows, cols = np.where(valid)
    if len(rows) < 16:
        raise ValueError("cached MegaSAM pointmap cannot recover camera intrinsics")
    subset = np.linspace(0, len(rows) - 1, min(len(rows), 10000), dtype=int)
    rows, cols = rows[subset], cols[subset]
    ratios = camera[rows, cols, :2] / camera[rows, cols, 2][:, None]
    fx, cx = np.linalg.lstsq(
        np.column_stack((ratios[:, 0], np.ones(len(rows)))), cols,
        rcond=None,
    )[0]
    fy, cy = np.linalg.lstsq(
        np.column_stack((ratios[:, 1], np.ones(len(rows)))), rows,
        rcond=None,
    )[0]
    residual = np.max(np.abs(np.column_stack((fx * ratios[:, 0] + cx - cols,
                                              fy * ratios[:, 1] + cy - rows))))
    if (not np.isfinite([fx, fy, cx, cy, residual]).all()
            or min(fx, fy) <= 0 or residual > 0.25
            or abs(fx - focal_length) / focal_length > 0.02):
        raise ValueError("cached MegaSAM geometry and focal length disagree on intrinsics")
    return np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _input_geometry(
    prepared: PreparedMultiObjectTracking,
    geometry: SharedGeometryResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lift selected source-pixel queries with cached MegaSAM world pointmaps."""
    pointmaps = np.asarray(geometry.pointmaps, dtype=np.float32)
    poses = np.asarray(geometry.camera_poses, dtype=np.float32)
    frames = prepared.frames_count
    if len(pointmaps) != frames or len(poses) != frames:
        raise ValueError("TAPIP3D needs one cached pointmap and pose per video frame")
    h, w = pointmaps.shape[1:3]
    source_h, source_w = prepared.original_hw
    k = _intrinsics_from_cached_pointmap(
        pointmaps[0], poses[0], geometry.focal_length,
    )
    camera_points = np.einsum(
        "thwc,tcd->thwd", pointmaps - poses[:, None, None, :3, 3],
        poses[:, :3, :3],
    )
    depths = camera_points[..., 2].astype(np.float32)
    valid = np.isfinite(pointmaps).all(axis=-1) & np.any(pointmaps != 0, axis=-1)
    depths[~valid | ~np.isfinite(depths) | (depths <= 0)] = 0

    selected = selected_link7_queries(prepared)
    if len(selected) < 2:
        raise ValueError("Link 7 needs at least two original selected points")
    query_world = np.empty((len(selected), 4), dtype=np.float32)
    query_world[:, 0] = selected[:, 0]
    for point_id, (frame_float, u, v) in enumerate(selected):
        frame = int(frame_float)
        if frame != frame_float or not 0 <= frame < frames:
            raise ValueError(f"invalid Link 7 initialization frame for point {point_id}")
        x = float(u * (w - 1) / max(source_w - 1, 1))
        y = float(v * (h - 1) / max(source_h - 1, 1))
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        patch = depths[frame, [y0, y0, y1, y1], [x0, x1, x0, x1]]
        if np.all(patch > 0):
            dx, dy = x - x0, y - y0
            depth = float(np.dot(patch, [(1 - dx) * (1 - dy),
                                         dx * (1 - dy), (1 - dx) * dy, dx * dy]))
        else:
            depth = float(depths[frame, int(round(y)), int(round(x))])
        if depth <= 0:
            raise ValueError(f"selected Link 7 point {point_id} has no MegaSAM depth")
        camera_xyz = np.asarray([(x - k[0, 2]) * depth / k[0, 0],
                                 (y - k[1, 2]) * depth / k[1, 1], depth])
        query_world[point_id, 1:] = camera_xyz @ poses[frame, :3, :3].T + poses[frame, :3, 3]
    return depths, k, poses, query_world


def replace_link7_with_tapip3d(
    *,
    baseline: MultiObjectTrackResult,
    prepared: PreparedMultiObjectTracking,
    geometry: SharedGeometryResult,
    tracker: TrackWrapper,
    output_dir: Path,
    python: str,
    repository: str,
    checkpoint: str,
    support_grid_size: int = 16,
    num_iters: int = 6,
    resolution_factor: int = 2,
    visibility_threshold: float = 0.9,
) -> MultiObjectTrackResult:
    """Track only the original Link 7 queries; retain all other CoTracker groups."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_initial_queries(output_dir / "link7_initial_queries.npz", prepared)
    depths, intrinsics, poses, query_world = _input_geometry(prepared, geometry)
    selected = selected_link7_queries(prepared)
    bridge_input = output_dir / "link7_tapip3d_input.npz"
    bridge_output = output_dir / "link7_tapip3d_raw.npz"
    np.savez_compressed(
        bridge_input,
        depths=depths,
        intrinsics=intrinsics,
        camera_poses=poses,
        query_world=query_world,
        selected_queries=selected,
        point_ids=np.arange(len(selected), dtype=np.int64),
        source_hw=np.asarray(prepared.original_hw, dtype=np.int64),
    )
    for path, label in ((Path(python), "TAPIP3D Python"),
                        (Path(repository) / "utils/inference_utils.py", "official TAPIP3D source"),
                        (Path(checkpoint), "TAPIP3D checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(Path(repository).resolve()), str((_workspace_root())))
    )
    command = [python, "-m", "robot.experiments.link7_tapip3d.tapip3d_worker",
               "--video", prepared.video_path, "--input", str(bridge_input),
               "--output", str(bridge_output), "--checkpoint", checkpoint,
               "--support-grid-size", str(support_grid_size),
               "--num-iters", str(num_iters),
               "--resolution-factor", str(resolution_factor),
               "--visibility-threshold", str(visibility_threshold)]
    worker_started = time.perf_counter()
    subprocess.run(command, cwd=repository, env=env, check=True)
    worker_seconds = time.perf_counter() - worker_started
    with np.load(bridge_output, allow_pickle=False) as data:
        tracks = np.asarray(data["tracks_uv"], dtype=np.float32)
        visibility = np.asarray(data["visibility"], dtype=np.float32)
        point_ids = np.asarray(data["point_ids"], dtype=np.int64)
        tapip_peak_memory = int(np.asarray(data["peak_gpu_memory_bytes"]).item())
    if tracks.shape != (prepared.frames_count, len(selected), 2):
        raise ValueError(f"TAPIP3D returned unexpected tracks: {tracks.shape}")
    if visibility.shape != tracks.shape[:2] or not np.array_equal(
        point_ids, np.arange(len(selected))
    ):
        raise ValueError("TAPIP3D visibility or selected point IDs changed")
    keep = tracker._track_keep_mask(tracks, visibility)
    index = prepared.object_names.index("link7")
    object_tracks = list(baseline.object_tracks)
    object_visibility = list(baseline.object_visibility)
    object_queries = list(baseline.object_queries)
    object_point_ids = list(baseline.object_point_ids)
    object_tracks[index] = tracks[:, keep]
    object_visibility[index] = visibility[:, keep]
    object_queries[index] = selected[keep]
    object_point_ids[index] = point_ids[keep]
    metadata = dict(baseline.metadata)
    metadata["link7_tracker"] = "tapip3d"
    metadata["link7_initialized_count"] = len(selected)
    metadata["link7_downstream_rejected_count"] = int((~keep).sum())
    metadata["cotracker_model_seconds"] = metadata["model_seconds"]
    metadata["tapip3d_worker_seconds"] = worker_seconds
    metadata["model_seconds"] += worker_seconds
    metadata["total_tracking_seconds"] += worker_seconds
    metadata["tapip3d_peak_gpu_memory_bytes"] = tapip_peak_memory
    metadata["peak_gpu_memory_scope"] = "CoTracker process only; TAPIP3D process reported separately"
    metadata["foreground_query_counts"] = [len(value) for value in object_queries]
    return replace(
        baseline,
        object_tracks=tuple(object_tracks),
        object_visibility=tuple(object_visibility),
        object_queries=tuple(object_queries),
        object_point_ids=tuple(object_point_ids),
        metadata=metadata,
    )
