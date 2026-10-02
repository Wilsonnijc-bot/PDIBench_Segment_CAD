"""V2 Link 7 foreground depth gate, in the source camera's coordinates."""

from __future__ import annotations

import cv2
import numpy as np


def camera_depth(pointmap: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """Convert MegaSAM world points to positive camera Z, rejecting empty pixels."""
    rotation = camera_pose[:3, :3]
    z = (pointmap - camera_pose[:3, 3]) @ rotation[:, 2]
    valid = np.isfinite(pointmap).all(axis=-1) & np.any(pointmap != 0, axis=-1)
    return np.where(valid & np.isfinite(z) & (z > 0), z, np.nan)


def link7_depth_keep(
    pointmap: np.ndarray,
    camera_pose: np.ndarray,
    mask: np.ndarray,
    xy: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Keep queries on the near, coherent Link 7 surface in frame zero.

    The near half of the masked depth distribution estimates the gripper
    surface. A far background sheet can be smooth (and pass V1's Sobel gate),
    but cannot set an arbitrarily deep cutoff here. A small local check also
    rejects isolated lifted pixels inside an otherwise plausible mask.
    """
    h, w = pointmap.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    depth = camera_depth(pointmap, camera_pose)
    foreground = depth[mask & np.isfinite(depth)]
    if len(foreground) < 25:
        raise ValueError("Link 7 V2 needs at least 25 valid masked depth pixels")
    near = foreground[foreground <= np.percentile(foreground, 60)]
    center = float(np.median(near))
    spread = float(1.4826 * np.median(np.abs(near - center)))
    tolerance = max(0.04, 3.0 * spread, 0.12 * center)
    far_limit = min(center + tolerance, 1.35 * center)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
    query_depth = depth[y, x]
    keep = mask[y, x] & np.isfinite(query_depth) & (query_depth <= far_limit)
    local_rejected = 0
    for index in np.flatnonzero(keep):
        x0, x1 = max(0, x[index] - 3), min(w, x[index] + 4)
        y0, y1 = max(0, y[index] - 3), min(h, y[index] + 4)
        patch = depth[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
        patch = patch[np.isfinite(patch)]
        if len(patch) >= 5 and abs(query_depth[index] - np.median(patch)) > tolerance:
            keep[index] = False
            local_rejected += 1
    return keep, {
        "version": "v2", "camera_depth_center": center,
        "camera_depth_spread": spread, "far_limit": far_limit,
        "candidate_count": len(x), "retained_count": int(keep.sum()),
        "local_rejected_count": local_rejected,
    }


def link7_depth_region(
    pointmap: np.ndarray,
    camera_pose: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Apply the V2 gate to mask pixels before tracker queries are sampled."""
    h, w = pointmap.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    y, x = np.where(mask)
    xy = np.column_stack((x, y))
    keep, stats = link7_depth_keep(pointmap, camera_pose, mask, xy)
    region = np.zeros((h, w), dtype=bool)
    region[y[keep], x[keep]] = True
    return region, {**stats, "stage": "before_query_sampling", "unit": "pixels"}


def link7_query_mask(
    pointmap: np.ndarray,
    camera_pose: np.ndarray,
    source_mask: np.ndarray,
    tracker_hw: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Build tracker-grid eligibility before SIFT/corner/grid query sampling.

    A query must pass the existing nearest-pixel gate and have an acceptable
    depth under TAPIP3D's bilinear query lift. This avoids dropping queries
    after sampling or allowing interpolation across a far-depth boundary.
    """
    region, stats = link7_depth_region(pointmap, camera_pose, source_mask)
    tracker_h, tracker_w = tracker_hw
    source_h, source_w = source_mask.shape
    point_h, point_w = pointmap.shape[:2]
    mask = cv2.resize(source_mask.astype(np.uint8), (tracker_w, tracker_h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    yy, xx = np.indices(tracker_hw, dtype=np.float64)
    nearest_x = np.rint(xx * (point_w - 1) / max(tracker_w - 1, 1)).astype(int)
    nearest_y = np.rint(yy * (point_h - 1) / max(tracker_h - 1, 1)).astype(int)
    mask &= region[nearest_y, nearest_x]

    x = (xx * source_w / tracker_w) * (point_w - 1) / max(source_w - 1, 1)
    y = (yy * source_h / tracker_h) * (point_h - 1) / max(source_h - 1, 1)
    x = np.clip(x, 0, point_w - 1)
    y = np.clip(y, 0, point_h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, point_w - 1), np.minimum(y0 + 1, point_h - 1)
    depth = camera_depth(pointmap, camera_pose)
    d00, d10 = depth[y0, x0], depth[y0, x1]
    d01, d11 = depth[y1, x0], depth[y1, x1]
    dx, dy = x - x0, y - y0
    bilinear = ((1 - dx) * (1 - dy) * d00 + dx * (1 - dy) * d10
                + (1 - dx) * dy * d01 + dx * dy * d11)
    valid_patch = (np.isfinite(d00) & np.isfinite(d10)
                   & np.isfinite(d01) & np.isfinite(d11))
    lifted = np.where(valid_patch, bilinear,
                      depth[np.rint(y).astype(int), np.rint(x).astype(int)])
    before_lift = int(mask.sum())
    mask &= np.isfinite(lifted) & (lifted > 0) & (lifted <= stats["far_limit"])
    return mask, {
        **stats,
        "eligible_tracker_pixel_count": int(mask.sum()),
        "bilinear_lift_rejected_pixel_count": before_lift - int(mask.sum()),
    }
