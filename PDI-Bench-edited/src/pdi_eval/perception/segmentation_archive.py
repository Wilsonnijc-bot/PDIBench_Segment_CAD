"""Native loading and validation for multi-object segmentation archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import MultiObjectSegmentation


def frame_measurements(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure pixel height, x centroid, and frame-edge truncation."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (T,H,W), got {masks.shape}")
    heights = np.zeros(len(masks), dtype=np.float64)
    centers = np.zeros(len(masks), dtype=np.float64)
    truncated = np.ones(len(masks), dtype=bool)
    for frame_index, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if len(xs) <= 10:
            centers[frame_index] = centers[frame_index - 1] if frame_index else 0.0
            continue
        heights[frame_index] = float(ys.max() - ys.min())
        centers[frame_index] = float(xs.mean())
        truncated[frame_index] = bool(
            ys.min() < 5
            or ys.max() >= mask.shape[0] - 5
            or xs.min() < 5
            or xs.max() >= mask.shape[1] - 5
        )
    return heights, centers, truncated


def _validate_video_shape(video_path: Path, masks: np.ndarray) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    capture.release()
    if masks.shape[2:] != (metadata["height"], metadata["width"]):
        raise ValueError(
            f"mask size {masks.shape[3]}x{masks.shape[2]} does not match "
            f"video {metadata['width']}x{metadata['height']}"
        )
    if metadata["frames"] > 0 and len(masks) != metadata["frames"]:
        raise ValueError(
            f"segmentation has {len(masks)} frames but video has {metadata['frames']}"
        )
    return metadata


def load_multi_object_segmentation(
    archive_path: str | Path,
    video_path: str | Path | None = None,
) -> MultiObjectSegmentation:
    """Load the canonical SAM3 archive without collapsing object identity."""
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Segmentation archive is missing: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"object_masks", "object_names", "object_ids"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"segmentation archive is missing arrays: {sorted(missing)}")
        object_masks = np.asarray(archive["object_masks"], dtype=bool)
        object_names = tuple(str(value) for value in np.asarray(archive["object_names"]).tolist())
        object_ids = np.asarray(archive["object_ids"], dtype=np.int64)

    if object_masks.ndim != 4:
        raise ValueError(
            f"object_masks must have shape (T,N,H,W), got {object_masks.shape}"
        )
    frame_count, object_count, height, width = object_masks.shape
    if min(frame_count, object_count, height, width) < 1:
        raise ValueError("object_masks cannot contain an empty dimension")
    if len(object_names) != object_count or object_ids.shape != (object_count,):
        raise ValueError("object_names and object_ids must match object_masks axis 1")
    if len(set(object_names)) != object_count:
        raise ValueError("object_names must be unique")
    if len(set(object_ids.tolist())) != object_count:
        raise ValueError("object_ids must be unique")

    h_pixel = np.zeros((frame_count, object_count), dtype=np.float64)
    x_center = np.zeros((frame_count, object_count), dtype=np.float64)
    is_truncated = np.zeros((frame_count, object_count), dtype=bool)
    for object_index in range(object_count):
        values = frame_measurements(object_masks[:, object_index])
        h_pixel[:, object_index] = values[0]
        x_center[:, object_index] = values[1]
        is_truncated[:, object_index] = values[2]

    metadata: dict[str, Any] = {
        "archive": str(archive_path),
        "overlap_pixel_count": int(np.count_nonzero(object_masks.sum(axis=1) > 1)),
    }
    if video_path is not None:
        video_path = Path(video_path).resolve()
        metadata["video"] = _validate_video_shape(video_path, object_masks)
        video_id = video_path.stem
    else:
        video_id = archive_path.stem

    return MultiObjectSegmentation(
        video_id=video_id,
        object_names=object_names,
        object_ids=object_ids,
        object_masks=object_masks,
        h_pixel=h_pixel,
        x_center=x_center,
        is_truncated=is_truncated,
        union_masks=np.any(object_masks, axis=1),
        metadata=metadata,
    )
