"""Robot-link scoring over prepared arrays; no model inference or replay rendering."""
from __future__ import annotations
from typing import Any
import numpy as np
from infrastructure.shared.scoring.pdi.motion_audit import audit_3d_trajectory_consistency
from infrastructure.shared.scoring.pdi.reconstruction_audit import audit_scale_jump
from infrastructure.shared.scoring.pdi.scale_audit import audit_scale_consistency
from infrastructure.shared.scoring.rigidity.rigidity import audit_3d_volume_stability
from infrastructure.shared.geometry.camera import CameraModel
from infrastructure.shared.geometry.projection import ProjectionJudge
from infrastructure.shared.scoring.pdi.pdi_index import PDIIndexCalculator


def _vp_in_object_bbox(
    vp_xy: tuple[float, float],
    masks: np.ndarray,
    margin_ratio: float = 0.1,
) -> bool:
    if len(masks) == 0:
        return False
    combined = np.any(masks[:min(5, len(masks))], axis=0)
    ys, xs = np.where(combined)
    if len(xs) == 0:
        return False
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    margin_x = (x_max - x_min) * margin_ratio
    margin_y = (y_max - y_min) * margin_ratio
    return bool(
        x_min - margin_x <= vp_xy[0] <= x_max + margin_x
        and y_min - margin_y <= vp_xy[1] <= y_max + margin_y
    )


def _vp_epsilon(
    foreground_vp: tuple[float, float],
    background_vp: tuple[float, float],
    camera: CameraModel,
    image_hw: tuple[int, int],
) -> float:
    foreground_direction = np.asarray(
        [foreground_vp[0] - camera.cx, foreground_vp[1] - camera.cy],
        dtype=np.float64,
    )
    background_direction = np.asarray(
        [background_vp[0] - camera.cx, background_vp[1] - camera.cy],
        dtype=np.float64,
    )
    foreground_norm = float(np.linalg.norm(foreground_direction))
    background_norm = float(np.linalg.norm(background_direction))
    height, width = image_hw
    foreground_offscreen = not (
        0 <= foreground_vp[0] <= width and 0 <= foreground_vp[1] <= height
    )
    if foreground_norm < 5.0 or background_norm < 5.0 or foreground_offscreen:
        return 0.0
    cosine = float(np.dot(foreground_direction, background_direction)) / (
        foreground_norm * background_norm
    )
    return (1.0 - float(np.clip(cosine, -1.0, 1.0))) / 2.0


def _map_tracks_between_grids(
    tracks: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Map video-pixel tracks to a pointmap grid without changing endpoints."""
    source_height, source_width = source_hw
    target_height, target_width = target_hw
    if min(source_height, source_width, target_height, target_width) < 1:
        raise ValueError("track and pointmap dimensions must be positive")
    mapped = np.asarray(tracks, dtype=np.float64).copy()
    mapped[..., 0] *= (
        (target_width - 1) / (source_width - 1) if source_width > 1 else 0.0
    )
    mapped[..., 1] *= (
        (target_height - 1) / (source_height - 1) if source_height > 1 else 0.0
    )
    return mapped


def evaluate_object_metrics(
    *,
    object_name: str,
    masks: np.ndarray,
    h_pixel: np.ndarray,
    depth_z: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    background_tracks: np.ndarray,
    pointmaps: np.ndarray,
    focal_length: float,
    fps: float,
    lsd_frames: np.ndarray | None,
    lsd_exclusion_masks: np.ndarray,
    weights: dict[str, float],
    requested_track_count: int | None = None,
    point_filter_version: str = "v1",
) -> dict[str, Any]:
    """Apply unchanged PDI formulas to one object view of shared inference."""
    frame_count = min(
        len(masks), len(h_pixel), len(depth_z), len(tracks), len(visibility), len(pointmaps)
    )
    if frame_count < 2:
        raise ValueError(f"{object_name} has fewer than two common metric frames")
    masks = masks[:frame_count]
    h_pixel = h_pixel[:frame_count]
    depth_z = depth_z[:frame_count]
    tracks = tracks[:frame_count]
    visibility = visibility[:frame_count]
    pointmaps = pointmaps[:frame_count]

    camera = CameraModel(focal_length=focal_length, image_size=masks.shape[1:])
    foreground_ntd = tracks.transpose(1, 0, 2)
    background_ntd = (
        background_tracks[:frame_count].transpose(1, 0, 2)
        if background_tracks.ndim == 3 and background_tracks.shape[1] >= 2
        else None
    )
    projection = ProjectionJudge(cx=camera.cx, cy=camera.cy)
    global_vp, foreground_vp, background_vp = projection.estimate_vanishing_point_v2(
        fg_tracks=foreground_ntd,
        bg_tracks=background_ntd,
        frames=lsd_frames,
        masks=(
            lsd_exclusion_masks[:len(lsd_frames)]
            if lsd_frames is not None
            else None
        ),
    )
    trajectory_vp = foreground_vp
    if foreground_vp == (camera.cx, camera.cy) or _vp_in_object_bbox(
        foreground_vp, masks
    ):
        trajectory_vp = background_vp
    epsilon_vp = _vp_epsilon(
        foreground_vp, background_vp, camera, masks.shape[1:]
    )
    effective_vp = epsilon_vp if background_vp != (camera.cx, camera.cy) else 0.0

    scale_history = audit_scale_consistency(h_pixel, depth_z)
    trajectory_history = audit_3d_trajectory_consistency(pointmaps, masks, fps=fps)
    rigidity_tracks = _map_tracks_between_grids(
        tracks,
        source_hw=masks.shape[1:],
        target_hw=pointmaps.shape[1:3],
    )
    rigidity, rigidity_history, rigidity_strategy = audit_3d_volume_stability(
        pointmaps,
        masks,
        tracks=rigidity_tracks,
        h_seq=h_pixel,
        visibility=visibility,
        insufficient_policy="raise",
        point_filter_version=point_filter_version if object_name == "link7" else "v1",
    )
    calculator = PDIIndexCalculator(
        w_scale=weights.get("w_scale", 0.3),
        w_traj=weights.get("w_trajectory", 0.3),
        w_rigidity=weights.get("w_rigidity", 0.2),
        w_vp=weights.get("w_vp", 0.2),
    )
    report = calculator.compute_pdi(
        scale_history, trajectory_history, rigidity, effective_vp
    )
    scale_jump, scale_jump_pass = audit_scale_jump(pointmaps[..., 2], masks)
    report["object_name"] = object_name
    report["breakdown"].update(
        {
            "scale_history": scale_history,
            "traj_history": trajectory_history,
            "volume_history": rigidity_history,
            "rigidity_strategy": rigidity_strategy,
        }
    )
    report.update(
        {
            "vanishing_point": global_vp,
            "foreground_vanishing_point": foreground_vp,
            "background_vanishing_point": background_vp,
            "trajectory_vanishing_point": trajectory_vp,
            "tracking": {
                "requested_foreground_query_count": (
                    int(requested_track_count)
                    if requested_track_count is not None
                    else int(tracks.shape[1])
                ),
                "foreground_track_count": int(tracks.shape[1]),
                "retained_track_fraction": (
                    float(tracks.shape[1] / requested_track_count)
                    if requested_track_count
                    else 1.0
                ),
                "background_track_count": int(background_tracks.shape[1]),
                "mean_visibility": float(visibility.mean()) if visibility.size else 0.0,
                "video_coordinate_hw": list(masks.shape[1:]),
                "rigidity_coordinate_hw": list(pointmaps.shape[1:3]),
            },
            "reconstruction_audit": {
                "scale_jump": scale_jump,
                "scale_jump_pass": scale_jump_pass,
            },
        }
    )
    return report

