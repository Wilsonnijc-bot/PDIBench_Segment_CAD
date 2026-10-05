#!/usr/bin/env python3
"""Render a 3D target reconstruction replay from cached PDI outputs.

The replay intentionally separates the two point sources:

* small grey points: Mega-SAM world points selected by the per-frame SAM mask
* large colored points: visible CoTracker anchors lifted through the pointmap

This native utility reads artifacts produced by the shared multi-object
PDI-Bench pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


class _FfmpegPipeWriter:
    """Small imageio-compatible RGB writer backed by a system FFmpeg."""

    def __init__(self, output_path: Path, fps: float, width: int, height: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "MP4 rendering requires either imageio with FFmpeg support or an ffmpeg executable"
            )
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append_data(self, frame_rgb: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is closed")
        self._process.stdin.write(np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        stderr = self._process.stderr.read().decode("utf-8", errors="replace")
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr.strip()}")


def _open_video_writer(output_path: Path, fps: float, width: int, height: int):
    """Use system FFmpeg when available, otherwise try imageio's FFmpeg plugin."""
    if shutil.which("ffmpeg"):
        return _FfmpegPipeWriter(output_path, fps, width, height)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "MP4 rendering requires either an ffmpeg executable or imageio with FFmpeg support"
        ) from exc

    return imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def resize_mask_nearest(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D mask with deterministic nearest-neighbour sampling."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target size must be positive, got {target_hw}")
    if mask.shape == (target_h, target_w):
        return mask.copy()

    source_h, source_w = mask.shape
    y_index = np.minimum(
        (np.arange(target_h, dtype=np.float64) * source_h / target_h).astype(int),
        source_h - 1,
    )
    x_index = np.minimum(
        (np.arange(target_w, dtype=np.float64) * source_w / target_w).astype(int),
        source_w - 1,
    )
    return mask[y_index[:, None], x_index[None, :]]


def map_xy_between_grids(
    xy: np.ndarray,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Map image coordinates between grids while preserving both endpoints.

    CoTracker caches use coordinates in the SAM/video grid. Mega-SAM pointmaps
    may have a different spatial resolution. Mapping [0, size - 1] to the same
    closed interval in the target prevents the last source pixel from falling
    outside the pointmap.
    """
    points = np.asarray(xy, dtype=np.float64)
    if points.shape[-1] != 2:
        raise ValueError(f"xy must end in dimension 2, got shape {points.shape}")

    source_h, source_w = (int(source_hw[0]), int(source_hw[1]))
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("source and target dimensions must be positive")

    scale_x = (target_w - 1) / (source_w - 1) if source_w > 1 else 0.0
    scale_y = (target_h - 1) / (source_h - 1) if source_h > 1 else 0.0
    mapped = points.copy()
    mapped[..., 0] *= scale_x
    mapped[..., 1] *= scale_y
    return mapped


def lift_tracks_to_3d(
    pointmaps: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    track_hw: Tuple[int, int],
    visibility_threshold: float = 0.5,
) -> np.ndarray:
    """Lift CoTracker coordinates into world space with nearest point sampling.

    Invalid, invisible, non-finite, or out-of-image tracks become NaN. Tracks
    are checked against the source image bounds before remapping, so clipping
    cannot turn a drifted track into a plausible boundary point.
    """
    pointmaps = np.asarray(pointmaps)
    tracks = np.asarray(tracks)
    visibility = np.asarray(visibility)

    if pointmaps.ndim != 4 or pointmaps.shape[-1] != 3:
        raise ValueError(f"pointmaps must have shape (T,H,W,3), got {pointmaps.shape}")
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must have shape (T,N,2), got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"visibility must have shape {tracks.shape[:2]}, got {visibility.shape}"
        )
    if pointmaps.shape[0] != tracks.shape[0]:
        raise ValueError("pointmaps and tracks must have the same frame count")

    source_h, source_w = (int(track_hw[0]), int(track_hw[1]))
    target_h, target_w = pointmaps.shape[1:3]
    mapped = map_xy_between_grids(tracks, track_hw, (target_h, target_w))

    finite_xy = np.isfinite(tracks).all(axis=-1)
    in_source = (
        (tracks[..., 0] >= 0.0)
        & (tracks[..., 0] <= source_w - 1)
        & (tracks[..., 1] >= 0.0)
        & (tracks[..., 1] <= source_h - 1)
    )
    visible = np.isfinite(visibility) & (visibility > visibility_threshold)
    valid = finite_xy & in_source & visible

    # Replace invalid coordinates before integer conversion to avoid NaN casts.
    safe_xy = np.where(valid[..., None], mapped, 0.0)
    x_index = np.clip(np.rint(safe_xy[..., 0]).astype(int), 0, target_w - 1)
    y_index = np.clip(np.rint(safe_xy[..., 1]).astype(int), 0, target_h - 1)
    frame_index = np.arange(pointmaps.shape[0])[:, None]
    lifted = pointmaps[frame_index, y_index, x_index].astype(np.float64, copy=True)

    valid_point = np.isfinite(lifted).all(axis=-1) & np.any(lifted != 0.0, axis=-1)
    lifted[~(valid & valid_point)] = np.nan
    return lifted


def _load_array(path: Path, keys: Sequence[str], label: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        for key in keys:
            if key in archive.files:
                return np.asarray(archive[key])
        available = ", ".join(archive.files) or "<empty>"
    raise KeyError(f"{label} not found in {path}; available keys: {available}")


def load_segmentation_union(path: Path) -> np.ndarray:
    """Load a legacy mask sequence or union canonical (T,N,H,W) object masks."""
    with np.load(path, allow_pickle=False) as archive:
        for key in ("masks", "mask"):
            if key in archive.files:
                return _normalise_masks(np.asarray(archive[key]))
        if "object_masks" in archive.files:
            object_masks = np.asarray(archive["object_masks"])
            if object_masks.ndim != 4:
                raise ValueError(
                    "object_masks must have shape (T,N,H,W), "
                    f"got {object_masks.shape}"
                )
            return np.any(object_masks > 0, axis=1)
        available = ", ".join(archive.files) or "<empty>"
    raise KeyError(
        f"segmentation masks not found in {path}; available keys: {available}"
    )


def _normalise_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (T,H,W), got {masks.shape}")
    return masks > 0


def _normalise_tracks(tracks: np.ndarray) -> np.ndarray:
    tracks = np.asarray(tracks)
    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    if tracks.ndim == 2 and tracks.shape[-1] == 2:
        tracks = tracks[None]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must have shape (T,N,2), got {tracks.shape}")
    return tracks.astype(np.float64, copy=False)


def _normalise_visibility(visibility: np.ndarray, track_shape: Tuple[int, int]) -> np.ndarray:
    visibility = np.asarray(visibility)
    if visibility.ndim == 3 and visibility.shape[0] == 1:
        visibility = visibility[0]
    if visibility.ndim == 1:
        if track_shape[0] == 1 and visibility.shape[0] == track_shape[1]:
            visibility = visibility[None]
        elif track_shape[1] == 1 and visibility.shape[0] == track_shape[0]:
            visibility = visibility[:, None]
    if visibility.shape == track_shape[::-1] and visibility.shape != track_shape:
        visibility = visibility.T
    if visibility.shape != track_shape:
        raise ValueError(f"visibility must have shape {track_shape}, got {visibility.shape}")
    return visibility.astype(np.float64, copy=False)


def _normalise_pointmaps(pointmaps: np.ndarray) -> np.ndarray:
    pointmaps = np.asarray(pointmaps)
    if pointmaps.ndim == 3 and pointmaps.shape[-1] == 3:
        pointmaps = pointmaps[None]
    if pointmaps.ndim == 4 and pointmaps.shape[1] == 3 and pointmaps.shape[-1] != 3:
        pointmaps = np.moveaxis(pointmaps, 1, -1)
    if pointmaps.ndim != 4 or pointmaps.shape[-1] != 3:
        raise ValueError(f"pointmaps must have shape (T,H,W,3), got {pointmaps.shape}")
    return pointmaps.astype(np.float64, copy=False)


def transform_world_to_camera(points: np.ndarray, camera_c2w: np.ndarray) -> np.ndarray:
    """Express world points in a reference camera frame while preserving sentinels."""
    points = np.asarray(points, dtype=np.float64)
    camera_c2w = np.asarray(camera_c2w, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"points must end in dimension 3, got shape {points.shape}")
    if camera_c2w.shape != (4, 4):
        raise ValueError(f"camera_c2w must have shape (4,4), got {camera_c2w.shape}")

    rotation = camera_c2w[:3, :3]
    translation = camera_c2w[:3, 3]
    zero_sentinel = np.all(points == 0.0, axis=-1)
    camera_points = (points - translation) @ rotation
    camera_points[zero_sentinel] = 0.0
    return camera_points


def project_camera_points(
    points: np.ndarray,
    focal_length: float,
    image_hw: Tuple[int, int],
    minimum_depth: float = 1e-6,
) -> np.ndarray:
    """Perspective-project camera-coordinate points into an image plane."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"points must end in dimension 3, got shape {points.shape}")
    focal_length = float(focal_length)
    if not np.isfinite(focal_length) or focal_length <= 0:
        raise ValueError(f"focal_length must be positive and finite, got {focal_length}")
    height, width = (int(image_hw[0]), int(image_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image size must be positive, got {image_hw}")

    projected = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > minimum_depth)
    if np.any(valid):
        camera = points[valid]
        projected[valid, 0] = focal_length * camera[:, 0] / camera[:, 2] + width / 2.0
        projected[valid, 1] = focal_length * camera[:, 1] / camera[:, 2] + height / 2.0
    return projected


def masked_world_points(
    pointmap: np.ndarray,
    mask: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select valid SAM-masked world points, optionally subsampling them."""
    target_hw = pointmap.shape[:2]
    resized_mask = resize_mask_nearest(mask, target_hw) > 0
    valid = (
        resized_mask
        & np.isfinite(pointmap).all(axis=-1)
        & np.any(pointmap != 0.0, axis=-1)
    )
    points = pointmap[valid]
    if max_points > 0 and len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
    return points


def _iter_bounds_points(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    anchors_3d: np.ndarray,
    max_points: int,
    seed: int,
) -> Iterable[np.ndarray]:
    bounds_sample = max_points if max_points > 0 else 50_000
    bounds_sample = min(bounds_sample, 50_000)
    for frame_idx in range(pointmaps.shape[0]):
        yield masked_world_points(
            pointmaps[frame_idx],
            masks[frame_idx],
            bounds_sample,
            np.random.default_rng(seed + frame_idx),
        )
        visible_anchors = anchors_3d[frame_idx]
        yield visible_anchors[np.isfinite(visible_anchors).all(axis=-1)]


def compute_camera_view(
    clouds: Iterable[np.ndarray],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[np.ndarray, float]:
    """Compute robust camera-plane limits shared by all replay frames."""
    nonempty = [np.asarray(points) for points in clouds if len(points)]
    if not nonempty:
        raise ValueError("no valid masked Mega-SAM or lifted CoTracker points to render")
    points = np.concatenate(nonempty, axis=0)
    points = points[np.isfinite(points).all(axis=-1)]
    if not len(points):
        raise ValueError("all reconstruction points are non-finite")

    low = np.percentile(points[:, :2], lower_percentile, axis=0)
    high = np.percentile(points[:, :2], upper_percentile, axis=0)
    center = (low + high) / 2.0
    half_range = float(np.max(high - low) / 2.0)
    if not np.isfinite(half_range) or half_range <= 1e-9:
        half_range = 1.0
    return center, half_range * 1.08


def _render_orthographic_replay(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    anchors_3d: np.ndarray,
    output_mp4: Path,
    first_frame_png: Path,
    fps: float,
    max_grey_points: int,
    grey_size: float,
    anchor_size: float,
    width: int,
    height: int,
    dpi: int,
    seed: int,
) -> Tuple[np.ndarray, float, list[int], list[int]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    center, half_range = compute_camera_view(
        _iter_bounds_points(pointmaps, masks, anchors_3d, max_grey_points, seed)
    )
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    first_frame_png.parent.mkdir(parents=True, exist_ok=True)

    anchor_count = anchors_3d.shape[1]
    cmap = plt.get_cmap("turbo")
    anchor_colors = cmap(np.linspace(0.02, 0.98, max(anchor_count, 1)))
    figsize = (width / dpi, height / dpi)
    writer = _open_video_writer(output_mp4, fps, width, height)

    grey_counts: list[int] = []
    visible_counts: list[int] = []
    try:
        for frame_idx in range(pointmaps.shape[0]):
            grey = masked_world_points(
                pointmaps[frame_idx],
                masks[frame_idx],
                max_grey_points,
                np.random.default_rng(seed + frame_idx),
            )
            anchors = anchors_3d[frame_idx]
            anchor_valid = np.isfinite(anchors).all(axis=-1)
            grey_counts.append(int(len(grey)))
            visible_counts.append(int(anchor_valid.sum()))

            fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
            ax = fig.add_subplot(111)
            if len(grey):
                ax.scatter(
                    grey[:, 0], -grey[:, 1],
                    s=grey_size, c="#8b9198", alpha=0.52,
                    linewidths=0, rasterized=True,
                )
            if np.any(anchor_valid):
                ax.scatter(
                    anchors[anchor_valid, 0],
                    -anchors[anchor_valid, 1],
                    s=anchor_size,
                    c=anchor_colors[np.flatnonzero(anchor_valid)],
                    edgecolors="white", linewidths=0.65,
                )

            ax.set_xlim(center[0] - half_range, center[0] + half_range)
            ax.set_ylim(-center[1] - half_range, -center[1] + half_range)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Camera X")
            ax.set_ylabel("Camera Y (up)")
            ax.set_title(
                f"3D reconstruction - original camera view  |  "
                f"frame {frame_idx + 1}/{pointmaps.shape[0]}",
                fontsize=11,
            )
            ax.grid(True, alpha=0.22)
            legend = [
                Line2D([0], [0], marker="o", color="none", label="SAM-mask Mega-SAM points",
                       markerfacecolor="#8b9198", markeredgewidth=0, markersize=4),
                Line2D([0], [0], marker="o", color="none", label="Lifted CoTracker anchors",
                       markerfacecolor=cmap(0.15), markeredgecolor="white", markersize=8),
            ]
            ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=8)
            fig.subplots_adjust(left=0.01, right=0.96, bottom=0.11, top=0.94)
            if frame_idx == 0:
                fig.savefig(first_frame_png, dpi=dpi, facecolor=fig.get_facecolor())
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame_rgb = np.ascontiguousarray(rgba[..., :3])
            writer.append_data(frame_rgb)
            plt.close(fig)
    finally:
        writer.close()
        plt.close("all")

    return center, half_range, grey_counts, visible_counts


def _render_camera_pov_replay(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    anchors_3d: np.ndarray,
    camera_poses: np.ndarray,
    focal_length: float,
    source_video: Optional[Path],
    overlay_source_video: bool,
    output_mp4: Path,
    first_frame_png: Path,
    fps: float,
    max_grey_points: int,
    grey_size: float,
    anchor_size: float,
    width: int,
    height: int,
    dpi: int,
    seed: int,
) -> Tuple[list[int], list[int], list[float], list[float]]:
    """Render reconstructed points through each original perspective camera."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    source_capture = None
    if overlay_source_video:
        if source_video is None:
            raise ValueError("--overlay-source-video requires --source-video")
        source_capture = cv2.VideoCapture(str(source_video))
        if not source_capture.isOpened():
            raise RuntimeError(f"Cannot open replay source video: {source_video}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    first_frame_png.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_video_writer(output_mp4, fps, width, height)
    cmap = plt.get_cmap("turbo")
    anchor_count = anchors_3d.shape[1]
    anchor_colors = cmap(np.linspace(0.02, 0.98, max(anchor_count, 1)))
    pointmap_hw = pointmaps.shape[1:3]
    figsize = (width / dpi, height / dpi)

    projected_samples: list[np.ndarray] = []
    depth_samples: list[np.ndarray] = []
    frame_centers: list[np.ndarray] = []
    frame_spans: list[np.ndarray] = []
    for frame_idx in range(pointmaps.shape[0]):
        sample_world = masked_world_points(
            pointmaps[frame_idx],
            masks[frame_idx],
            min(max_grey_points, 3_000) if max_grey_points > 0 else 3_000,
            np.random.default_rng(seed + frame_idx),
        )
        sample_camera = transform_world_to_camera(sample_world, camera_poses[frame_idx])
        sample_xy = project_camera_points(sample_camera, focal_length, pointmap_hw)
        sample_valid = np.isfinite(sample_xy).all(axis=-1)
        if np.any(sample_valid):
            valid_xy = sample_xy[sample_valid]
            projected_samples.append(valid_xy)
            depth_samples.append(sample_camera[sample_valid, 2])
            frame_low = np.percentile(valid_xy, 1.0, axis=0)
            frame_high = np.percentile(valid_xy, 99.0, axis=0)
            frame_centers.append((frame_low + frame_high) / 2.0)
            frame_spans.append(np.maximum(frame_high - frame_low, 1.0))
        else:
            raise ValueError(f"frame {frame_idx} has no camera-projected reconstruction points")
    if not projected_samples:
        raise ValueError("no valid camera-projected reconstruction points to render")
    all_depths = np.concatenate(depth_samples, axis=0)
    center_path = np.stack(frame_centers)
    padded_centers = np.pad(center_path, ((2, 2), (0, 0)), mode="edge")
    center_path = np.stack(
        [padded_centers[index:index + 5].mean(axis=0) for index in range(len(center_path))]
    )
    span_xy = np.percentile(np.stack(frame_spans), 95.0, axis=0) * 1.25
    output_aspect = width / height
    if span_xy[0] / span_xy[1] < output_aspect:
        span_xy[0] = span_xy[1] * output_aspect
    else:
        span_xy[1] = span_xy[0] / output_aspect
    view_bounds_by_frame = np.column_stack(
        (
            center_path[:, 0] - span_xy[0] / 2.0,
            center_path[:, 1] - span_xy[1] / 2.0,
            center_path[:, 0] + span_xy[0] / 2.0,
            center_path[:, 1] + span_xy[1] / 2.0,
        )
    )
    depth_bounds = [float(np.percentile(all_depths, 1.0)), float(np.percentile(all_depths, 99.0))]
    if depth_bounds[1] - depth_bounds[0] <= 1e-9:
        depth_bounds[1] = depth_bounds[0] + 1.0

    grey_counts: list[int] = []
    visible_counts: list[int] = []
    try:
        for frame_idx in range(pointmaps.shape[0]):
            view_bounds = view_bounds_by_frame[frame_idx]
            if source_capture is None:
                background = np.full((height, width, 3), (10, 14, 18), dtype=np.uint8)
            else:
                ok, frame_bgr = source_capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Replay source ended at frame {frame_idx}; "
                        f"need {pointmaps.shape[0]} frames"
                    )
                background = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if background.shape[:2] != (height, width):
                    background = cv2.resize(
                        background, (width, height), interpolation=cv2.INTER_AREA
                    )

            grey_world = masked_world_points(
                pointmaps[frame_idx],
                masks[frame_idx],
                max_grey_points,
                np.random.default_rng(seed + frame_idx),
            )
            anchors_world = anchors_3d[frame_idx]
            grey_camera = transform_world_to_camera(grey_world, camera_poses[frame_idx])
            anchors_camera = transform_world_to_camera(anchors_world, camera_poses[frame_idx])
            grey_xy = project_camera_points(grey_camera, focal_length, pointmap_hw)
            anchor_xy = project_camera_points(anchors_camera, focal_length, pointmap_hw)
            grey_valid = (
                np.isfinite(grey_xy).all(axis=-1)
                & (grey_xy[:, 0] >= view_bounds[0])
                & (grey_xy[:, 0] <= view_bounds[2])
                & (grey_xy[:, 1] >= view_bounds[1])
                & (grey_xy[:, 1] <= view_bounds[3])
            )
            anchor_valid = (
                np.isfinite(anchor_xy).all(axis=-1)
                & (anchor_xy[:, 0] >= view_bounds[0])
                & (anchor_xy[:, 0] <= view_bounds[2])
                & (anchor_xy[:, 1] >= view_bounds[1])
                & (anchor_xy[:, 1] <= view_bounds[3])
            )
            grey_counts.append(int(grey_valid.sum()))
            visible_counts.append(int(anchor_valid.sum()))

            fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="black")
            ax = fig.add_axes([0, 0, 1, 1])
            if source_capture is None:
                ax.imshow(background, extent=(view_bounds[0], view_bounds[2], view_bounds[3], view_bounds[1]))
            else:
                ax.imshow(background, extent=(0, pointmap_hw[1], pointmap_hw[0], 0))
            if np.any(grey_valid):
                ax.scatter(
                    grey_xy[grey_valid, 0], grey_xy[grey_valid, 1],
                    s=grey_size * 2.5,
                    c=grey_camera[grey_valid, 2],
                    cmap="bone_r",
                    vmin=depth_bounds[0],
                    vmax=depth_bounds[1],
                    alpha=0.82,
                    linewidths=0, rasterized=True,
                )
            if np.any(anchor_valid):
                ax.scatter(
                    anchor_xy[anchor_valid, 0], anchor_xy[anchor_valid, 1],
                    s=anchor_size,
                    c=anchor_colors[np.flatnonzero(anchor_valid)],
                    edgecolors="white", linewidths=0.65,
                )
            ax.text(
                view_bounds[0] + span_xy[0] * 0.015,
                view_bounds[1] + span_xy[1] * 0.04,
                f"3D reconstruction - camera POV  |  frame {frame_idx + 1}/{pointmaps.shape[0]}",
                color="white",
                fontsize=10,
                ha="left",
                va="top",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 4},
            )
            ax.set_xlim(view_bounds[0], view_bounds[2])
            ax.set_ylim(view_bounds[3], view_bounds[1])
            ax.axis("off")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame_rgb = np.ascontiguousarray(rgba[..., :3])
            if frame_idx == 0:
                Image.fromarray(frame_rgb).save(first_frame_png)
            writer.append_data(frame_rgb)
            plt.close(fig)
    finally:
        if source_capture is not None:
            source_capture.release()
        writer.close()
        plt.close("all")

    return (
        grey_counts,
        visible_counts,
        view_bounds_by_frame[0].astype(float).tolist(),
        depth_bounds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render segmentation-masked Mega-SAM points and lifted CoTracker anchors in 3D."
    )
    parser.add_argument(
        "--segmentation-npz",
        "--sam-npz",
        dest="segmentation_npz",
        type=Path,
        required=True,
        help="PDI-compatible segmentation cache NPZ",
    )
    parser.add_argument("--cotracker-npz", type=Path, required=True, help="PDI CoTracker cache NPZ")
    parser.add_argument("--megasam-npz", type=Path, required=True, help="PDI Mega-SAM cache NPZ")
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--first-frame-png", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument(
        "--view-mode",
        choices=("camera-pov", "frame-0-orthographic"),
        default="camera-pov",
        help="Perspective source-camera replay or legacy frame-0 orthographic plot",
    )
    parser.add_argument(
        "--source-video",
        type=Path,
        help="Source video used only with --overlay-source-video",
    )
    parser.add_argument(
        "--overlay-source-video",
        action="store_true",
        help="Debug mode: draw camera-projected points over the RGB source video",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument(
        "--track-height", type=int,
        help="CoTracker coordinate-grid height; defaults to the SAM mask height",
    )
    parser.add_argument(
        "--track-width", type=int,
        help="CoTracker coordinate-grid width; defaults to the SAM mask width",
    )
    parser.add_argument(
        "--max-grey-points", type=int, default=25_000,
        help="Maximum grey surface points per frame; 0 renders every valid point",
    )
    parser.add_argument("--grey-size", type=float, default=1.2)
    parser.add_argument("--anchor-size", type=float, default=42.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if min(args.width, args.height, args.dpi) <= 0:
        raise ValueError("render dimensions and DPI must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be even for yuv420p MP4 output")
    if args.max_grey_points < 0:
        raise ValueError("--max-grey-points cannot be negative")
    if (args.track_height is None) != (args.track_width is None):
        raise ValueError("provide both --track-height and --track-width, or neither")

    masks = load_segmentation_union(args.segmentation_npz)
    tracks = _normalise_tracks(
        _load_array(args.cotracker_npz, ("tracks", "tracks_2d"), "CoTracker tracks")
    )
    visibility = _normalise_visibility(
        _load_array(
            args.cotracker_npz,
            ("visibility", "confidence", "visibilities"),
            "CoTracker visibility",
        ),
        tracks.shape[:2],
    )
    pointmaps = _normalise_pointmaps(
        _load_array(args.megasam_npz, ("pointmaps", "pointmap"), "Mega-SAM pointmaps")
    )
    with np.load(args.megasam_npz, allow_pickle=False) as archive:
        camera_poses = np.asarray(archive["camera_poses"]) if "camera_poses" in archive.files else None
        focal_length = (
            float(np.asarray(archive["focal_length"]).reshape(-1)[0])
            if "focal_length" in archive.files
            else None
        )

    original_frames = {
        "masks": int(masks.shape[0]),
        "tracks": int(tracks.shape[0]),
        "pointmaps": int(pointmaps.shape[0]),
    }
    frame_count = min(original_frames.values())
    if frame_count < 1:
        raise ValueError(f"cache inputs contain no common frames: {original_frames}")
    masks = masks[:frame_count]
    tracks = tracks[:frame_count]
    visibility = visibility[:frame_count]
    pointmaps = pointmaps[:frame_count]

    track_hw = (
        (args.track_height, args.track_width)
        if args.track_height is not None
        else masks.shape[1:3]
    )
    anchors_3d = lift_tracks_to_3d(
        pointmaps,
        tracks,
        visibility,
        track_hw=track_hw,
        visibility_threshold=args.visibility_threshold,
    )
    if camera_poses is not None and (
        camera_poses.ndim != 3
        or camera_poses.shape[1:] != (4, 4)
        or len(camera_poses) < frame_count
    ):
        raise ValueError(
            f"camera_poses must have shape (T,4,4) with T >= {frame_count}, "
            f"got {camera_poses.shape}"
        )
    if args.view_mode == "camera-pov":
        if camera_poses is None:
            raise ValueError("camera-pov replay requires camera_poses in the MegaSAM cache")
        if focal_length is None:
            raise ValueError("camera-pov replay requires focal_length in the MegaSAM cache")
        camera_poses = camera_poses[:frame_count]
        reference_camera_c2w = None
    else:
        reference_camera_c2w = (
            np.eye(4, dtype=np.float64) if camera_poses is None else camera_poses[0]
        )
        pointmaps = transform_world_to_camera(pointmaps, reference_camera_c2w)
        anchors_3d = transform_world_to_camera(anchors_3d, reference_camera_c2w)

    output_mp4 = args.output_mp4.expanduser().resolve()
    first_frame_png = (
        args.first_frame_png.expanduser().resolve()
        if args.first_frame_png
        else output_mp4.with_name(f"{output_mp4.stem}_first_frame.png")
    )
    metadata_json = (
        args.metadata_json.expanduser().resolve()
        if args.metadata_json
        else output_mp4.with_suffix(".json")
    )
    source_video = args.source_video.expanduser().resolve() if args.source_video else None
    if source_video is not None and not source_video.is_file():
        raise FileNotFoundError(f"Replay source video does not exist: {source_video}")
    if args.view_mode == "camera-pov":
        grey_counts, visible_counts, view_bounds, depth_bounds = _render_camera_pov_replay(
            pointmaps=pointmaps,
            masks=masks,
            anchors_3d=anchors_3d,
            camera_poses=camera_poses,
            focal_length=focal_length,
            source_video=source_video,
            overlay_source_video=args.overlay_source_video,
            output_mp4=output_mp4,
            first_frame_png=first_frame_png,
            fps=args.fps,
            max_grey_points=args.max_grey_points,
            grey_size=args.grey_size,
            anchor_size=args.anchor_size,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
            seed=args.seed,
        )
        view_metadata = {
            "view_mode": "camera-pov",
            "projection": "per-frame world-to-camera perspective projection",
            "focal_length_pointmap_pixels": focal_length,
            "principal_point": "pointmap image center",
            "initial_target_crop_xyxy": view_bounds,
            "target_follow_crop": True,
            "camera_depth_percentile_bounds": depth_bounds,
            "depth_shading": "camera Z mapped through bone_r",
            "source_video_overlay": args.overlay_source_video,
            "source_video": str(source_video) if args.overlay_source_video else None,
        }
    else:
        center, half_range, grey_counts, visible_counts = _render_orthographic_replay(
            pointmaps=pointmaps,
            masks=masks,
            anchors_3d=anchors_3d,
            output_mp4=output_mp4,
            first_frame_png=first_frame_png,
            fps=args.fps,
            max_grey_points=args.max_grey_points,
            grey_size=args.grey_size,
            anchor_size=args.anchor_size,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
            seed=args.seed,
        )
        view_metadata = {
            "view_mode": "frame-0-orthographic",
            "orientation": "+X right, -Y up, optical axis +Z",
            "reference_camera_c2w": reference_camera_c2w.tolist(),
            "view_center": center.tolist(),
            "view_half_range": half_range,
        }

    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 3,
        "inputs": {
            "segmentation_npz": str(args.segmentation_npz.expanduser().resolve()),
            "cotracker_npz": str(args.cotracker_npz.expanduser().resolve()),
            "megasam_npz": str(args.megasam_npz.expanduser().resolve()),
        },
        "outputs": {
            "mp4": str(output_mp4),
            "first_frame_png": str(first_frame_png),
            "metadata_json": str(metadata_json),
        },
        "frames": {"input": original_frames, "rendered": frame_count, "fps": args.fps},
        "arrays": {
            "masks": list(masks.shape),
            "tracks": list(tracks.shape),
            "visibility": list(visibility.shape),
            "pointmaps": list(pointmaps.shape),
            "track_coordinate_hw": list(track_hw),
        },
        "render": {
            "grey_points": "SAM mask applied to each Mega-SAM world pointmap",
            "colored_points": "visible CoTracker anchors lifted by nearest pointmap sample",
            "max_grey_points_per_frame": args.max_grey_points,
            "grey_point_counts": grey_counts,
            "visible_anchor_counts": visible_counts,
            "visibility_threshold": args.visibility_threshold,
            "coordinate_mapping": "endpoint-preserving scale from track grid to pointmap grid",
            "image_size": [args.width, args.height],
            "random_seed": args.seed,
            **view_metadata,
        },
    }
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Rendered {frame_count} frames to {output_mp4}")
    print(f"First frame: {first_frame_png}")
    print(f"Metadata: {metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
