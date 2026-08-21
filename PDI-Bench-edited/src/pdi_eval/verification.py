"""Frozen-input preparation and comparisons for A/B/C/D verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .perception.mega_sam_wrapper import target_depth_from_world_pointmaps
from .perception.segmentation_archive import (
    frame_measurements,
    load_multi_object_segmentation,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_frozen_inputs(
    video_path: str | Path,
    canonical_archive: str | Path,
    output_dir: str | Path,
    selected_object: str = "link1",
) -> dict[str, Any]:
    """Create the one-link B/C archive from the canonical seven-link archive."""
    video_path = Path(video_path).resolve()
    canonical_archive = Path(canonical_archive).resolve()
    output_dir = Path(output_dir).resolve()
    segmentation = load_multi_object_segmentation(canonical_archive, video_path)
    if segmentation.object_count != 7:
        raise ValueError(
            f"A/B/C/D verification requires seven SAM3 objects, got "
            f"{segmentation.object_count}: {segmentation.object_names}"
        )
    try:
        selected_index = segmentation.object_names.index(selected_object)
    except ValueError as exc:
        raise ValueError(
            f"selected object {selected_object!r} is not in {segmentation.object_names}"
        ) from exc

    object_masks = segmentation.object_masks[:, selected_index:selected_index + 1]
    masks = object_masks[:, 0]
    heights, centers, truncated = frame_measurements(masks)
    single_archive = output_dir / "single_link_segmentation.npz"
    _write_npz(
        single_archive,
        masks=masks,
        object_masks=object_masks,
        object_names=np.asarray([selected_object]),
        object_ids=np.asarray([segmentation.object_ids[selected_index]], dtype=np.int64),
        h_pixel=heights,
        x_center=centers,
        is_truncated=truncated,
    )
    with np.load(single_archive, allow_pickle=False) as frozen:
        if not np.array_equal(frozen["object_masks"][:, 0], masks):
            raise RuntimeError("one-link archive changed the selected SAM3 mask")

    manifest = {
        "schema_version": 1,
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "canonical_segmentation": str(canonical_archive),
        "canonical_segmentation_sha256": sha256_file(canonical_archive),
        "single_link_segmentation": str(single_archive),
        "single_link_segmentation_sha256": sha256_file(single_archive),
        "selected_object": selected_object,
        "selected_object_index": selected_index,
        "selected_object_id": int(segmentation.object_ids[selected_index]),
        "object_names": list(segmentation.object_names),
        "object_ids": segmentation.object_ids.tolist(),
        "mask_exactly_frozen": True,
        "mask_shape": list(masks.shape),
        "overlap_pixel_count": segmentation.metadata["overlap_pixel_count"],
    }
    _write_json(output_dir / "input_manifest.json", manifest)
    return manifest


def seed_original_cache(
    video_path: str | Path,
    single_archive: str | Path,
    geometry_archive: str | Path,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Populate only the original pipeline's documented perception cache inputs."""
    video_path = Path(video_path).resolve()
    single_archive = Path(single_archive).resolve()
    geometry_archive = Path(geometry_archive).resolve()
    cache_dir = Path(cache_dir).resolve()
    segmentation = load_multi_object_segmentation(single_archive, video_path)
    masks = segmentation.object_masks[:, 0]

    with np.load(geometry_archive, allow_pickle=False) as geometry:
        pointmaps = np.asarray(geometry["pointmaps"])
        camera_poses = np.asarray(geometry["camera_poses"])
        focal_length = float(np.asarray(geometry["focal_length"]).reshape(-1)[0])
    frame_count = min(len(masks), len(pointmaps), len(camera_poses))
    if frame_count < 1 or not np.any(np.isfinite(pointmaps) & (pointmaps != 0)):
        raise ValueError("shared MegaSAM geometry is empty or invalid")
    depth_z = target_depth_from_world_pointmaps(
        pointmaps[:frame_count],
        camera_poses[:frame_count],
        masks[:frame_count],
    )

    # Preserve the original benchmark's inclusive height convention.
    original_heights = np.zeros(len(masks), dtype=np.float64)
    centers = np.zeros(len(masks), dtype=np.float64)
    for index, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if len(xs):
            original_heights[index] = float(np.ptp(ys) + 1)
            centers[index] = float(xs.mean())
        elif index:
            original_heights[index] = original_heights[index - 1]
            centers[index] = centers[index - 1]
    truncated = frame_measurements(masks)[2]
    video_id = video_path.stem
    sam_cache = cache_dir / f"{video_id}_sam2.npz"
    geometry_cache = cache_dir / f"{video_id}_mega_sam.npz"
    _write_npz(
        sam_cache,
        masks=masks,
        h_pixel=original_heights,
        x_center=centers,
        is_truncated=truncated,
    )
    _write_npz(
        geometry_cache,
        depth_z=depth_z,
        focal_length=np.asarray(focal_length),
        camera_poses=camera_poses[:frame_count],
        pointmaps=pointmaps[:frame_count],
    )
    result = {
        "sam3_mask_cache": str(sam_cache),
        "sam3_mask_cache_sha256": sha256_file(sam_cache),
        "shared_geometry_cache": str(geometry_cache),
        "shared_geometry_cache_sha256": sha256_file(geometry_cache),
        "source_geometry_archive": str(geometry_archive),
        "source_geometry_archive_sha256": sha256_file(geometry_archive),
        "frames": frame_count,
    }
    _write_json(cache_dir / "seed_manifest.json", result)
    return result


def load_track_group(path: str | Path, object_name: str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        names = [str(value) for value in archive["object_names"].tolist()]
        index = names.index(object_name)
        offsets = np.asarray(archive["object_offsets"], dtype=np.int64)
        start, end = int(offsets[index]), int(offsets[index + 1])
        return {
            "tracks": np.asarray(archive["tracks"])[:, start:end],
            "visibility": np.asarray(archive["visibility"])[:, start:end],
            "queries": np.asarray(archive["queries"])[start:end],
        }


def compare_track_groups(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, Any]:
    def key(query: np.ndarray) -> tuple[float, float, float]:
        return tuple(round(float(value), 3) for value in query)

    left_index = {key(query): index for index, query in enumerate(left["queries"])}
    right_index = {key(query): index for index, query in enumerate(right["queries"])}
    common = sorted(set(left_index).intersection(right_index))
    result: dict[str, Any] = {
        "left_query_count": len(left["queries"]),
        "right_query_count": len(right["queries"]),
        "common_query_count": len(common),
        "queries_exact": list(left_index) == list(right_index),
    }
    if not common:
        result.update(
            mean_track_l2_pixels=None,
            maximum_track_l2_pixels=None,
            endpoint_l2_pixels=None,
            visibility_agreement=None,
        )
        return result
    left_selector = [left_index[item] for item in common]
    right_selector = [right_index[item] for item in common]
    frames = min(len(left["tracks"]), len(right["tracks"]))
    deltas = np.linalg.norm(
        left["tracks"][:frames, left_selector]
        - right["tracks"][:frames, right_selector],
        axis=-1,
    )
    left_visibility = left["visibility"][:frames, left_selector] > 0.5
    right_visibility = right["visibility"][:frames, right_selector] > 0.5
    result.update(
        mean_track_l2_pixels=float(deltas.mean()),
        maximum_track_l2_pixels=float(deltas.max()),
        endpoint_l2_pixels=float(deltas[-1].mean()),
        visibility_agreement=float(np.mean(left_visibility == right_visibility)),
    )
    return result


def metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    breakdown = report["breakdown"]
    return {
        "pdi_score": float(report["pdi_score"]),
        "grade": str(report["grade"]),
        "scale": float(breakdown["scale_component"]),
        "trajectory": float(breakdown["traj_component"]),
        "rigidity": float(breakdown["epsilon_rigidity"]),
        "vp": float(breakdown["vp_component"]),
    }


def compare_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_values = metric_summary(left)
    right_values = metric_summary(right)
    return {
        "left": left_values,
        "right": right_values,
        "right_minus_left": {
            key: right_values[key] - left_values[key]
            for key in ("pdi_score", "scale", "trajectory", "rigidity", "vp")
        },
        "grade_changed": left_values["grade"] != right_values["grade"],
    }
