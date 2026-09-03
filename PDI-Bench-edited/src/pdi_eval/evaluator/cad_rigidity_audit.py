"""CAD anchor binding and scale-invariant proportional-shape scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..geometry.se3 import require_rigid_transform


CAD_RIGIDITY_METHOD = "cad-canonical-v1"
CAD_LINK_NAMES = tuple(f"link{index}" for index in range(2, 8))


@dataclass(frozen=True)
class ImageGridTransform:
    """Exact source resize followed by a pixel crop into MegaSAM geometry."""

    source_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    crop_xywh: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        source_height, source_width = self.source_hw
        resized_height, resized_width = self.resized_hw
        crop_x, crop_y, crop_width, crop_height = self.crop_xywh
        if min(source_height, source_width, resized_height, resized_width) < 1:
            raise ValueError("image grid dimensions must be positive")
        if min(crop_x, crop_y) < 0 or min(crop_width, crop_height) < 1:
            raise ValueError("crop must have non-negative origin and positive size")
        if crop_x + crop_width > resized_width or crop_y + crop_height > resized_height:
            raise ValueError("crop exceeds the resized image")

    @property
    def geometry_hw(self) -> tuple[int, int]:
        return self.crop_xywh[3], self.crop_xywh[2]

    def map_source_pixels(self, pixels_xy: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels_xy, dtype=np.float64)
        if pixels.shape[-1:] != (2,) or not np.isfinite(pixels).all():
            raise ValueError("pixels_xy must be a finite array ending in (x,y)")
        source_height, source_width = self.source_hw
        resized_height, resized_width = self.resized_hw
        crop_x, crop_y, _, _ = self.crop_xywh
        mapped = pixels.copy()
        mapped[..., 0] = (
            (pixels[..., 0] + 0.5) * resized_width / source_width - 0.5 - crop_x
        )
        mapped[..., 1] = (
            (pixels[..., 1] + 0.5) * resized_height / source_height - 0.5 - crop_y
        )
        return mapped

    def transform_masks(self, masks: np.ndarray) -> np.ndarray:
        values = np.asarray(masks, dtype=bool)
        if values.shape[-2:] != self.source_hw:
            raise ValueError(
                f"mask source shape {values.shape[-2:]} does not match {self.source_hw}"
            )
        flat = values.reshape((-1, *self.source_hw))
        resized_height, resized_width = self.resized_hw
        crop_x, crop_y, crop_width, crop_height = self.crop_xywh
        transformed = []
        for mask in flat:
            resized = cv2.resize(
                mask.astype(np.uint8),
                (resized_width, resized_height),
                interpolation=cv2.INTER_NEAREST,
            )
            transformed.append(
                resized[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
                > 0
            )
        return np.stack(transformed).reshape((*values.shape[:-2], *self.geometry_hw))


@dataclass(frozen=True)
class CadAnchorSet:
    query_ids: np.ndarray
    query_pixels_source: np.ndarray
    points_cad: np.ndarray
    triangle_ids: np.ndarray
    valid: np.ndarray

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))


def erode_masks(masks: np.ndarray, radius: int = 2) -> np.ndarray:
    values = np.asarray(masks, dtype=bool)
    if radius < 0:
        raise ValueError("erosion radius cannot be negative")
    if radius == 0:
        return values.copy()
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    flat = values.reshape((-1, *values.shape[-2:]))
    eroded = [cv2.erode(mask.astype(np.uint8), kernel) > 0 for mask in flat]
    return np.stack(eroded).reshape(values.shape)


def backproject_pixels(
    pixels_xy: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    depths = np.asarray(depth, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_xy must have shape (N,2)")
    if depths.shape != (len(pixels),):
        raise ValueError("depth must have one value per pixel")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if not np.isfinite(pixels).all() or not np.isfinite(depths).all():
        raise ValueError("pixels and depth must be finite")
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    rays = homogeneous @ np.linalg.inv(matrix).T
    return rays * depths[:, None]


def canonicalize_camera_points(
    points_camera: np.ndarray,
    T_C_from_L: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points_camera must be a finite array ending in xyz")
    transform = require_rigid_transform(T_C_from_L, name="T_C_from_L")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points - translation) @ rotation


def canonicalize_observed_sequence(
    points_camera: np.ndarray,
    point_valid: np.ndarray,
    T_C_from_L: np.ndarray,
    pose_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-transform valid camera points into the fixed CAD link frame."""
    points = np.asarray(points_camera, dtype=np.float64)
    valid = np.asarray(point_valid, dtype=bool)
    transforms = np.asarray(T_C_from_L, dtype=np.float64)
    valid_poses = np.asarray(pose_valid, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3 or valid.shape != points.shape[:2]:
        raise ValueError("camera points/valid must have shapes (T,Q,3) and (T,Q)")
    if transforms.shape != (len(points), 4, 4):
        raise ValueError("T_C_from_L must have shape (T,4,4)")
    if valid_poses.shape != (len(points),):
        raise ValueError("pose_valid must have shape (T,)")

    points_cad = np.full_like(points, np.nan)
    output_valid = valid & valid_poses[:, None] & np.isfinite(points).all(axis=2)
    for frame_index in np.flatnonzero(valid_poses):
        transform = require_rigid_transform(
            transforms[frame_index], name=f"T_C_from_L[{frame_index}]"
        )
        selector = output_valid[frame_index]
        if np.any(selector):
            points_cad[frame_index, selector] = canonicalize_camera_points(
                points[frame_index, selector], transform
            )
    return points_cad, output_valid


def audit_depth_world_consistency(
    *,
    depths_camera: np.ndarray,
    intrinsics: np.ndarray,
    pointmaps_world: np.ndarray,
    T_W_from_C: np.ndarray,
    samples_per_frame: int = 64,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Verify that depth and world pointmaps encode the same camera points."""
    depths = np.asarray(depths_camera, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    pointmaps = np.asarray(pointmaps_world, dtype=np.float64)
    camera_poses = np.asarray(T_W_from_C, dtype=np.float64)
    if depths.ndim != 3:
        raise ValueError("depths_camera must have shape (T,H,W)")
    if pointmaps.shape != (*depths.shape, 3):
        raise ValueError("pointmaps_world must have shape (T,H,W,3)")
    if camera_poses.shape != (len(depths), 4, 4):
        raise ValueError("T_W_from_C must have shape (T,4,4)")
    if matrices.shape == (3, 3):
        matrices = np.repeat(matrices[None], len(depths), axis=0)
    if matrices.shape != (len(depths), 3, 3):
        raise ValueError("intrinsics must have shape (3,3) or (T,3,3)")
    if samples_per_frame < 1 or relative_tolerance <= 0.0:
        raise ValueError("coordinate audit settings must be positive")

    errors = []
    tolerance_ratios = []
    for frame_index in range(len(depths)):
        transform = require_rigid_transform(
            camera_poses[frame_index], name=f"T_W_from_C[{frame_index}]"
        )
        depth = depths[frame_index]
        world = pointmaps[frame_index]
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(world).all(axis=2)
            & np.any(world != 0.0, axis=2)
        )
        flat_indices = np.flatnonzero(valid)
        if not len(flat_indices):
            continue
        sample_count = min(samples_per_frame, len(flat_indices))
        positions = np.linspace(0, len(flat_indices) - 1, sample_count)
        selected = flat_indices[np.unique(positions.round().astype(np.int64))]
        rows, columns = np.unravel_index(selected, depth.shape)
        pixels = np.column_stack((columns, rows)).astype(np.float64)
        points_from_depth = backproject_pixels(
            pixels, depth[rows, columns], matrices[frame_index]
        )
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_from_world = (world[rows, columns] - translation) @ rotation
        frame_errors = np.linalg.norm(points_from_depth - points_from_world, axis=1)
        tolerances = relative_tolerance * np.maximum(
            1.0, np.linalg.norm(points_from_depth, axis=1)
        )
        errors.extend(frame_errors.tolist())
        tolerance_ratios.extend((frame_errors / tolerances).tolist())

    if not errors:
        return {
            "status": "unscorable",
            "sample_count": 0,
            "maximum_error": None,
            "maximum_tolerance_ratio": None,
        }
    ratios = np.asarray(tolerance_ratios, dtype=np.float64)
    return {
        "status": "complete" if np.all(ratios <= 1.0) else "failed",
        "sample_count": len(errors),
        "maximum_error": float(np.max(errors)),
        "maximum_tolerance_ratio": float(np.max(ratios)),
        "relative_tolerance": float(relative_tolerance),
    }


def intersect_rays_with_triangles(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nearest positive Moller-Trumbore hit for each ray."""
    origins = np.asarray(ray_origins, dtype=np.float64)
    directions = np.asarray(ray_directions, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
        raise ValueError("ray origins and directions must have shape (N,3)")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (V,3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F,3)")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces contain out-of-range vertex indices")

    triangle0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - triangle0
    edge2 = vertices[faces[:, 2]] - triangle0
    hit_points = np.full((len(origins), 3), np.nan, dtype=np.float64)
    hit_triangles = np.full(len(origins), -1, dtype=np.int64)
    hit_distance = np.full(len(origins), np.nan, dtype=np.float64)
    for ray_index, (origin, direction) in enumerate(zip(origins, directions)):
        if not np.isfinite(origin).all() or not np.isfinite(direction).all():
            continue
        if np.linalg.norm(direction) <= epsilon:
            continue
        cross = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, cross)
        usable = np.abs(determinant) > epsilon
        inverse_determinant = np.zeros_like(determinant)
        inverse_determinant[usable] = 1.0 / determinant[usable]
        offset = origin - triangle0
        barycentric_u = np.einsum("ij,ij->i", offset, cross) * inverse_determinant
        usable &= (barycentric_u >= -epsilon) & (barycentric_u <= 1.0 + epsilon)
        offset_cross = np.cross(offset, edge1)
        barycentric_v = (
            np.einsum(
                "ij,ij->i", np.broadcast_to(direction, offset_cross.shape), offset_cross
            )
            * inverse_determinant
        )
        usable &= (barycentric_v >= -epsilon) & (
            barycentric_u + barycentric_v <= 1.0 + epsilon
        )
        distance = np.einsum("ij,ij->i", edge2, offset_cross) * inverse_determinant
        usable &= distance > epsilon
        candidates = np.flatnonzero(usable)
        if not len(candidates):
            continue
        triangle_index = int(candidates[np.argmin(distance[candidates])])
        nearest = float(distance[triangle_index])
        hit_points[ray_index] = origin + nearest * direction
        hit_triangles[ray_index] = triangle_index
        hit_distance[ray_index] = nearest
    return hit_points, hit_triangles, hit_distance


def bind_cad_anchors(
    *,
    query_ids: np.ndarray,
    query_pixels_source: np.ndarray,
    grid_transform: ImageGridTransform,
    frame_mask_geometry: np.ndarray,
    frame_depth_geometry: np.ndarray,
    intrinsics: np.ndarray,
    T_C_from_L: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> CadAnchorSet:
    ids = np.asarray(query_ids, dtype=np.int32)
    source_pixels = np.asarray(query_pixels_source, dtype=np.float64)
    mask = np.asarray(frame_mask_geometry, dtype=bool)
    depth = np.asarray(frame_depth_geometry, dtype=np.float64)
    if ids.ndim != 1 or source_pixels.shape != (len(ids), 2):
        raise ValueError("query IDs and source pixels must have shapes (Q,) and (Q,2)")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("query IDs must be unique")
    if mask.shape != grid_transform.geometry_hw or depth.shape != mask.shape:
        raise ValueError("frame mask/depth do not match the geometry grid")
    transform = require_rigid_transform(T_C_from_L, name="T_C_from_L")
    pixels_geometry = grid_transform.map_source_pixels(source_pixels)
    indices = np.floor(pixels_geometry + 0.5).astype(np.int64)
    inside = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < mask.shape[1])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < mask.shape[0])
    )
    valid_input = inside.copy()
    safe_x = np.clip(indices[:, 0], 0, mask.shape[1] - 1)
    safe_y = np.clip(indices[:, 1], 0, mask.shape[0] - 1)
    sampled_depth = depth[safe_y, safe_x]
    valid_input &= mask[safe_y, safe_x]
    valid_input &= np.isfinite(sampled_depth) & (sampled_depth > 0.0)

    homogeneous = np.column_stack((pixels_geometry, np.ones(len(ids))))
    directions_camera = homogeneous @ np.linalg.inv(intrinsics).T
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    ray_origin = -rotation.T @ translation
    origins_cad = np.repeat(ray_origin[None], len(ids), axis=0)
    directions_cad = directions_camera @ rotation
    hit_points, hit_triangles, _ = intersect_rays_with_triangles(
        origins_cad, directions_cad, vertices, faces
    )
    valid = valid_input & (hit_triangles >= 0)
    hit_points[~valid] = np.nan
    hit_triangles[~valid] = -1
    return CadAnchorSet(
        query_ids=ids,
        query_pixels_source=source_pixels,
        points_cad=hit_points,
        triangle_ids=hit_triangles,
        valid=valid,
    )


def sample_observed_track_points(
    *,
    tracks_source: np.ndarray,
    visibility: np.ndarray,
    grid_transform: ImageGridTransform,
    masks_geometry: np.ndarray,
    depths_geometry: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks_source, dtype=np.float64)
    visible = np.asarray(visibility) > 0.5
    masks = np.asarray(masks_geometry, dtype=bool)
    depths = np.asarray(depths_geometry, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[2] != 2 or visible.shape != tracks.shape[:2]:
        raise ValueError("tracks/visibility must have shapes (T,Q,2) and (T,Q)")
    if masks.shape != (len(tracks), *grid_transform.geometry_hw):
        raise ValueError("masks_geometry must have shape (T,Hg,Wg)")
    if depths.shape != masks.shape:
        raise ValueError("depths_geometry must match masks_geometry")
    if matrices.shape == (3, 3):
        matrices = np.repeat(matrices[None], len(tracks), axis=0)
    if matrices.shape != (len(tracks), 3, 3):
        raise ValueError("intrinsics must have shape (3,3) or (T,3,3)")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")

    mapped = grid_transform.map_source_pixels(tracks.reshape(-1, 2)).reshape(tracks.shape)
    indices = np.floor(mapped + 0.5).astype(np.int64)
    points = np.full((len(tracks), tracks.shape[1], 3), np.nan, dtype=np.float64)
    valid = np.zeros(tracks.shape[:2], dtype=bool)
    for frame_index in range(len(tracks)):
        x = indices[frame_index, :, 0]
        y = indices[frame_index, :, 1]
        inside = (
            (x >= 0)
            & (x < masks.shape[2])
            & (y >= 0)
            & (y < masks.shape[1])
        )
        safe_x = np.clip(x, 0, masks.shape[2] - 1)
        safe_y = np.clip(y, 0, masks.shape[1] - 1)
        sampled_depth = depths[frame_index, safe_y, safe_x] * depth_scale
        frame_valid = (
            visible[frame_index]
            & inside
            & masks[frame_index, safe_y, safe_x]
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
        )
        valid[frame_index] = frame_valid
        if np.any(frame_valid):
            points[frame_index, frame_valid] = backproject_pixels(
                mapped[frame_index, frame_valid],
                sampled_depth[frame_valid],
                matrices[frame_index],
            )
    return points, valid


def _select_pairs(
    cad_points: np.ndarray,
    valid: np.ndarray,
    link_diameter: float,
    *,
    maximum_pairs: int,
    baseline_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    local_j, local_k = np.triu_indices(len(indices), 1)
    pair_j = indices[local_j]
    pair_k = indices[local_k]
    baselines = np.linalg.norm(cad_points[pair_j] - cad_points[pair_k], axis=1)
    keep = np.isfinite(baselines) & (baselines >= 0.05 * link_diameter)
    pair_j, pair_k, baselines = pair_j[keep], pair_k[keep], baselines[keep]
    if not len(baselines):
        return pair_j, pair_k, baselines
    ordering = np.lexsort((pair_k, pair_j, baselines))
    ordering_bins = np.array_split(ordering, baseline_bins)
    per_bin = max(1, maximum_pairs // baseline_bins)
    selected = []
    for values in ordering_bins:
        if not len(values):
            continue
        count = min(per_bin, len(values))
        positions = np.linspace(0, len(values) - 1, count).round().astype(int)
        selected.extend(values[np.unique(positions)].tolist())
    selected_array = np.asarray(selected[:maximum_pairs], dtype=np.int64)
    return pair_j[selected_array], pair_k[selected_array], baselines[selected_array]


def audit_cad_proportional_shape(
    *,
    observed_points: np.ndarray,
    observed_valid: np.ndarray,
    cad_anchor_points: np.ndarray,
    cad_anchor_valid: np.ndarray,
    link_diameter: float,
    mask_present: np.ndarray | None = None,
    minimum_pairs: int = 30,
    maximum_pairs: int = 512,
    baseline_bins: int = 8,
    minimum_scored_frames: int = 5,
    minimum_scored_fraction: float = 0.60,
    mean_threshold: float | None = None,
    p90_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare observed and CAD internal relations modulo one uniform scale."""
    points = np.asarray(observed_points, dtype=np.float64)
    valid = np.asarray(observed_valid, dtype=bool)
    anchors = np.asarray(cad_anchor_points, dtype=np.float64)
    anchor_valid = np.asarray(cad_anchor_valid, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3 or valid.shape != points.shape[:2]:
        raise ValueError("observed points/valid must have shapes (T,Q,3) and (T,Q)")
    if anchors.shape != points.shape[1:] or anchor_valid.shape != (points.shape[1],):
        raise ValueError("CAD anchors must align with the observed query axis")
    if not np.isfinite(link_diameter) or link_diameter <= 0.0:
        raise ValueError("link_diameter must be finite and positive")
    if minimum_pairs < 1 or maximum_pairs < minimum_pairs or baseline_bins < 1:
        raise ValueError("pair selection settings are inconsistent")
    if mask_present is None:
        present = np.any(valid, axis=1)
    else:
        present = np.asarray(mask_present, dtype=bool)
        if present.shape != (len(points),):
            raise ValueError("mask_present must have shape (T,)")

    frame_scores = np.full(len(points), np.nan, dtype=np.float64)
    relative_scales = np.full(len(points), np.nan, dtype=np.float64)
    residual_p90 = np.full(len(points), np.nan, dtype=np.float64)
    residual_max = np.full(len(points), np.nan, dtype=np.float64)
    usable_anchor_counts = np.zeros(len(points), dtype=np.int32)
    pair_counts = np.zeros(len(points), dtype=np.int32)
    rejection_reasons = ["insufficient_shape_support"] * len(points)
    cap = float(np.log(1.5))
    delta = 1e-6 * link_diameter
    for frame_index in range(len(points)):
        if not present[frame_index]:
            rejection_reasons[frame_index] = "mask_absent"
            continue
        frame_valid = (
            valid[frame_index]
            & anchor_valid
            & np.isfinite(points[frame_index]).all(axis=1)
            & np.isfinite(anchors).all(axis=1)
        )
        usable_anchor_counts[frame_index] = int(np.count_nonzero(frame_valid))
        pair_j, pair_k, cad_distances = _select_pairs(
            anchors,
            frame_valid,
            link_diameter,
            maximum_pairs=maximum_pairs,
            baseline_bins=baseline_bins,
        )
        pair_counts[frame_index] = len(pair_j)
        if len(pair_j) < minimum_pairs:
            continue
        observed_distances = np.linalg.norm(
            points[frame_index, pair_j] - points[frame_index, pair_k], axis=1
        )
        log_ratio = np.log(
            (observed_distances + delta) / (cad_distances + delta)
        )
        median_log_ratio = float(np.median(log_ratio))
        residual = log_ratio - median_log_ratio
        absolute_residual = np.abs(residual)
        frame_scores[frame_index] = float(
            np.sqrt(np.mean(np.minimum(residual * residual, cap * cap)))
        )
        relative_scales[frame_index] = float(np.exp(median_log_ratio))
        residual_p90[frame_index] = float(np.percentile(absolute_residual, 90))
        residual_max[frame_index] = float(np.max(absolute_residual))
        rejection_reasons[frame_index] = "accepted"

    scored = np.isfinite(frame_scores)
    scored_count = int(np.count_nonzero(scored))
    present_count = int(np.count_nonzero(present))
    scored_fraction = scored_count / max(present_count, 1)
    scorable = (
        scored_count >= minimum_scored_frames
        and scored_fraction >= minimum_scored_fraction
    )
    epsilon_mean = float(np.mean(frame_scores[scored])) if scorable else None
    epsilon_p90 = float(np.percentile(frame_scores[scored], 90)) if scorable else None
    calibrated = mean_threshold is not None and p90_threshold is not None
    decision = None
    if scorable and calibrated:
        decision = bool(
            epsilon_mean > float(mean_threshold)
            or epsilon_p90 > float(p90_threshold)
        )
    return {
        "method": CAD_RIGIDITY_METHOD,
        "status": (
            "unscorable" if not scorable else "complete" if calibrated else "uncalibrated"
        ),
        "epsilon_cad_frame": frame_scores,
        "relative_uniform_scale": relative_scales,
        "residual_p90": residual_p90,
        "residual_max": residual_max,
        "usable_anchor_counts": usable_anchor_counts,
        "pair_counts": pair_counts,
        "rejection_reasons": rejection_reasons,
        "scored_frame_indices": np.flatnonzero(scored),
        "scored_frame_count": scored_count,
        "mask_present_frame_count": present_count,
        "scored_frame_fraction": scored_fraction,
        "epsilon_cad_mean": epsilon_mean,
        "epsilon_cad_p90": epsilon_p90,
        "mean_threshold": mean_threshold,
        "p90_threshold": p90_threshold,
        "deformed": decision,
    }


@dataclass
class CadCanonicalizationRuntime:
    """CAD state initialized once and reused by every CoTracker mode."""

    config: dict[str, Any]
    link_names: tuple[str, ...]
    grid_transform: ImageGridTransform
    meshes: dict[str, Any]
    masks_geometry: dict[str, np.ndarray]
    anchors: dict[str, CadAnchorSet]
    anchor_errors: dict[str, str]
    poses: Any
    pose_discontinuity: dict[str, dict[str, Any]]
    coordinate_audit: dict[str, Any] | None = None

    @staticmethod
    def _project_path(project_root: Path, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    @classmethod
    def prepare(
        cls,
        *,
        config: dict[str, Any],
        project_root: str | Path,
        segmentation,
        geometry,
        prepared_tracking,
    ) -> "CadCanonicalizationRuntime | None":
        if not config.get("enabled", False):
            return None
        from ..geometry.cad_mesh import load_cad_manifest
        from ..perception.foundation_pose_wrapper import load_foundation_pose_archive
        from ..perception.track_wrapper import map_tracker_pixels_to_source
        from .motion_audit import (
            audit_foundation_pose_discontinuity,
            compose_metric_world_link_poses,
        )

        root = Path(project_root).resolve()
        link_names = tuple(config.get("link_names", CAD_LINK_NAMES))
        if link_names != CAD_LINK_NAMES:
            raise ValueError(
                "CAD canonicalization scope must be exactly link2 through link7"
            )
        segmentation_indices = {
            name: index for index, name in enumerate(segmentation.object_names)
        }
        missing = sorted(set(link_names).difference(segmentation_indices))
        if missing:
            raise ValueError(f"segmentation is missing CAD links: {missing}")
        geometry_fields = (
            geometry.rgb_camera,
            geometry.depth_camera,
            geometry.intrinsics_camera,
            geometry.frame_times_seconds,
            geometry.source_hw,
            geometry.resized_hw_before_crop,
            geometry.crop_xywh,
        )
        if any(value is None for value in geometry_fields):
            raise ValueError(
                "CAD canonicalization requires MegaSAM cache schema 4 RGB-D/K artifacts"
            )
        timing = geometry.metadata.get("timing") or {}
        if timing.get("timestamp_provenance") != "constant_fps_metadata":
            raise ValueError(
                "CAD pose discontinuity requires a finite positive video FPS"
            )
        coordinate_audit = audit_depth_world_consistency(
            depths_camera=geometry.depth_camera,
            intrinsics=geometry.intrinsics_camera,
            pointmaps_world=geometry.pointmaps,
            T_W_from_C=geometry.camera_poses,
            samples_per_frame=int(
                config.get("coordinate_validation_samples_per_frame", 64)
            ),
        )
        if coordinate_audit["status"] != "complete":
            raise ValueError(
                "MegaSAM camera depth and world pointmaps fail the coordinate audit: "
                f"{coordinate_audit}"
            )
        pose_archive = config.get("foundation_pose_archive")
        if not pose_archive:
            raise ValueError(
                "cad_canonicalization.foundation_pose_archive is required when enabled"
            )
        poses = load_foundation_pose_archive(
            cls._project_path(root, pose_archive),
            expected_link_names=link_names,
            expected_frame_count=geometry.frames_count,
        )
        if not np.allclose(
            poses.frame_times_seconds,
            geometry.frame_times_seconds,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("FoundationPose and MegaSAM frame timestamps differ")
        grid_transform = ImageGridTransform(
            source_hw=geometry.source_hw,
            resized_hw=geometry.resized_hw_before_crop,
            crop_xywh=geometry.crop_xywh,
        )
        manifest = cls._project_path(
            root, config.get("cad_manifest", "configs/sam3-cad-franka.yaml")
        )
        meshes = load_cad_manifest(manifest, link_names=link_names)
        tracker_indices = {
            name: index
            for index, name in enumerate(prepared_tracking.object_names)
        }
        if set(link_names).difference(tracker_indices):
            raise ValueError("CoTracker query manifest does not contain all CAD links")
        prepared_ids = (
            prepared_tracking.object_query_ids
            if prepared_tracking.object_query_ids
            else tuple(
                np.arange(len(queries), dtype=np.int32)
                for queries in prepared_tracking.object_queries
            )
        )
        masks_geometry: dict[str, np.ndarray] = {}
        anchors: dict[str, CadAnchorSet] = {}
        anchor_errors: dict[str, str] = {}
        discontinuity = {}
        intrinsics = np.asarray(geometry.intrinsics_camera, dtype=np.float64)
        for pose_link_index, name in enumerate(link_names):
            segmentation_index = segmentation_indices[name]
            tracker_index = tracker_indices[name]
            link_masks = grid_transform.transform_masks(
                segmentation.object_masks[:geometry.frames_count, segmentation_index]
            )
            link_masks = erode_masks(
                link_masks, int(config.get("mask_erosion_radius", 2))
            )
            masks_geometry[name] = link_masks
            world_link_poses = compose_metric_world_link_poses(
                geometry.camera_poses,
                poses.T_C_from_L[:, pose_link_index],
                poses.video_depth_scale,
            )
            pose_thresholds = config.get("pose_discontinuity", {}).get(name, {})
            discontinuity[name] = audit_foundation_pose_discontinuity(
                world_link_poses,
                poses.frame_times_seconds,
                poses.pose_valid[:, pose_link_index],
                meshes[name].diameter,
                pose_objective=poses.pose_objective[:, pose_link_index],
                pose_source=poses.pose_source[:, pose_link_index],
                translation_rate_threshold=float(
                    pose_thresholds.get("translation_rate", 3.0)
                ),
                rotation_rate_threshold_degrees=float(
                    pose_thresholds.get("rotation_rate_degrees", 450.0)
                ),
                quality_threshold=float(config.get("pose_quality_threshold", 0.40)),
            )
            if not poses.pose_valid[0, pose_link_index]:
                anchor_errors[name] = "frame_0_foundation_pose_invalid"
                continue
            queries = prepared_tracking.object_queries[tracker_index]
            source_pixels = map_tracker_pixels_to_source(
                queries[:, 1:3],
                prepared_tracking.tracker_hw,
                prepared_tracking.original_hw,
            )
            frame_intrinsics = intrinsics[0] if intrinsics.ndim == 3 else intrinsics
            anchor_set = bind_cad_anchors(
                query_ids=prepared_ids[tracker_index],
                query_pixels_source=source_pixels,
                grid_transform=grid_transform,
                frame_mask_geometry=link_masks[0],
                frame_depth_geometry=(
                    geometry.depth_camera[0] * poses.video_depth_scale
                ),
                intrinsics=frame_intrinsics,
                T_C_from_L=poses.T_C_from_L[0, pose_link_index],
                vertices=meshes[name].vertices,
                faces=meshes[name].faces,
            )
            anchors[name] = anchor_set
            minimum_anchors = int(config.get("minimum_anchors", 16))
            if anchor_set.valid_count < minimum_anchors:
                anchor_errors[name] = (
                    f"only_{anchor_set.valid_count}_of_{minimum_anchors}_anchors_valid"
                )
        return cls(
            config=config,
            link_names=link_names,
            grid_transform=grid_transform,
            meshes=meshes,
            masks_geometry=masks_geometry,
            anchors=anchors,
            anchor_errors=anchor_errors,
            poses=poses,
            pose_discontinuity=discontinuity,
            coordinate_audit=coordinate_audit,
        )

    def audit_track_result(
        self,
        *,
        geometry,
        track_result,
        object_index: int,
        object_name: str,
    ) -> dict[str, Any]:
        if object_name in self.anchor_errors:
            return {
                "method": CAD_RIGIDITY_METHOD,
                "status": "unscorable",
                "error": self.anchor_errors[object_name],
            }
        anchor_set = self.anchors[object_name]
        query_ids = (
            track_result.object_query_ids[object_index]
            if track_result.object_query_ids
            else np.arange(
                len(track_result.object_queries[object_index]), dtype=np.int32
            )
        )
        anchor_index = {
            int(query_id): index
            for index, query_id in enumerate(anchor_set.query_ids)
        }
        unknown = sorted(set(int(value) for value in query_ids).difference(anchor_index))
        if unknown:
            raise ValueError(
                f"{object_name} tracks contain query IDs absent from CAD anchors: {unknown}"
            )
        selector = np.asarray(
            [anchor_index[int(query_id)] for query_id in query_ids], dtype=np.int64
        )
        frame_count = min(
            geometry.frames_count,
            len(track_result.object_tracks[object_index]),
            len(self.masks_geometry[object_name]),
        )
        masks = self.masks_geometry[object_name][:frame_count]
        observed_points, observed_valid = sample_observed_track_points(
            tracks_source=track_result.object_tracks[object_index][:frame_count],
            visibility=track_result.object_visibility[object_index][:frame_count],
            grid_transform=self.grid_transform,
            masks_geometry=masks,
            depths_geometry=geometry.depth_camera[:frame_count],
            intrinsics=(
                geometry.intrinsics_camera[:frame_count]
                if np.asarray(geometry.intrinsics_camera).ndim == 3
                else geometry.intrinsics_camera
            ),
            depth_scale=self.poses.video_depth_scale,
        )
        pose_link_index = self.link_names.index(object_name)
        pose_valid = self.poses.pose_valid[:frame_count, pose_link_index]
        observed_points, observed_valid = canonicalize_observed_sequence(
            observed_points,
            observed_valid,
            self.poses.T_C_from_L[:frame_count, pose_link_index],
            pose_valid,
        )
        thresholds = self.config.get("shape_thresholds", {}).get(object_name, {})
        report = audit_cad_proportional_shape(
            observed_points=observed_points,
            observed_valid=observed_valid,
            cad_anchor_points=anchor_set.points_cad[selector],
            cad_anchor_valid=anchor_set.valid[selector],
            link_diameter=self.meshes[object_name].diameter,
            mask_present=np.any(masks, axis=(1, 2)),
            minimum_pairs=int(self.config.get("minimum_pairs", 30)),
            maximum_pairs=int(self.config.get("maximum_pairs", 512)),
            minimum_scored_frames=int(self.config.get("minimum_scored_frames", 5)),
            minimum_scored_fraction=float(
                self.config.get("minimum_scored_fraction", 0.60)
            ),
            mean_threshold=thresholds.get("mean"),
            p90_threshold=thresholds.get("p90"),
        )
        report.update(
            {
                "cad_sha256": self.meshes[object_name].sha256,
                "link_diameter": self.meshes[object_name].diameter,
                "initial_query_count": len(anchor_set.query_ids),
                "cad_bound_query_count": anchor_set.valid_count,
                "retained_query_ids": query_ids,
                "observed_coordinate_frame": "original-cad-link",
                "pose_valid_frame_count": int(np.count_nonzero(pose_valid)),
            }
        )
        return report

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "method": CAD_RIGIDITY_METHOD,
            "link_names": self.link_names,
            "foundation_pose": self.poses.metadata,
            "video_depth_scale": self.poses.video_depth_scale,
            "anchor_errors": self.anchor_errors,
            "coordinate_audit": self.coordinate_audit,
            "meshes": {
                name: {
                    "sha256": mesh.sha256,
                    "path": mesh.path,
                    "diameter": mesh.diameter,
                    "extents": mesh.extents,
                    "instance_count": mesh.instance_count,
                }
                for name, mesh in self.meshes.items()
            },
        }
