#!/usr/bin/env python3
"""Render CAD silhouettes and rank segmentation proposals against them."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("CAD rendering requires trimesh and pycollada") from exc

    loaded = trimesh.load(path, force="scene", process=False)
    meshes = []
    if isinstance(loaded, trimesh.Trimesh):
        meshes.append(loaded)
    else:
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            meshes.append(geometry)
    if not meshes:
        raise ValueError(f"CAD mesh contains no triangle geometry: {path}")
    combined = trimesh.util.concatenate(meshes)
    vertices = np.asarray(combined.vertices, dtype=np.float64)
    faces = np.asarray(combined.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"CAD mesh is not triangulated: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"CAD mesh has non-finite vertices: {path}")
    return vertices, faces


def _view_basis(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    view = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(view, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return np.stack((right, up, view), axis=1)


def render_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    azimuth_degrees: float,
    elevation_degrees: float,
    image_size: int,
    margin_fraction: float,
) -> np.ndarray:
    if image_size < 32:
        raise ValueError("render image_size must be at least 32")
    if not 0.0 <= margin_fraction < 0.45:
        raise ValueError("render margin_fraction must be in [0, 0.45)")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("CAD rendering requires opencv-python-headless") from exc

    centered = vertices - np.mean(vertices, axis=0, keepdims=True)
    projected = centered @ _view_basis(azimuth_degrees, elevation_degrees)
    xy = projected[:, :2]
    extent = np.ptp(xy, axis=0)
    max_extent = float(np.max(extent))
    if not np.isfinite(max_extent) or max_extent <= 1e-12:
        raise ValueError("CAD projection has zero extent")
    usable = image_size * (1.0 - 2.0 * margin_fraction)
    xy = xy * (usable / max_extent)
    xy[:, 1] *= -1.0
    xy += image_size / 2.0
    polygons = np.rint(xy[faces]).astype(np.int32)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(mask, polygons, color=255)
    return mask


def silhouette_descriptor(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("shape descriptors require opencv-python-headless") from exc

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.where(binary)
    if len(xs) < 10:
        raise ValueError("silhouette must contain at least ten pixels")
    hu = cv2.HuMoments(cv2.moments(binary)).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    hu = np.clip(hu, -30.0, 30.0) / 30.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    aspect = math.log(max(width, height) / max(min(width, height), 1.0))
    fill = float(len(xs)) / (width * height)
    return np.concatenate((hu, [aspect / 4.0, fill]))


def build_reference_bank(
    manifest: dict[str, Any],
    project_root: Path,
    output_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cad_config = manifest["cad"]
    render_config = manifest["rendering"]
    mesh_root = project_root / cad_config["mesh_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors = []
    records = []

    for mesh_spec in cad_config["meshes"]:
        mesh_path = mesh_root / mesh_spec["file"]
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing CAD mesh: {mesh_path}")
        actual_sha = sha256_file(mesh_path)
        if actual_sha != mesh_spec["sha256"]:
            raise ValueError(f"CAD checksum mismatch for {mesh_path.name}: {actual_sha}")
        vertices, faces = _load_triangle_mesh(mesh_path)
        mesh_output = output_dir / mesh_spec["name"]
        mesh_output.mkdir(parents=True, exist_ok=True)
        for elevation in render_config["elevation_degrees"]:
            for azimuth in render_config["azimuth_degrees"]:
                mask = render_silhouette(
                    vertices,
                    faces,
                    float(azimuth),
                    float(elevation),
                    int(render_config["image_size"]),
                    float(render_config["margin_fraction"]),
                )
                filename = f"az{int(azimuth):03d}_el{int(elevation):+03d}.png"
                from PIL import Image

                Image.fromarray(mask).save(mesh_output / filename)
                descriptors.append(silhouette_descriptor(mask))
                records.append(
                    {
                        "mesh": mesh_spec["name"],
                        "mesh_sha256": actual_sha,
                        "azimuth_degrees": float(azimuth),
                        "elevation_degrees": float(elevation),
                        "render": str((mesh_output / filename).resolve()),
                    }
                )
    descriptor_array = np.stack(descriptors)
    descriptor_path = output_dir / "reference_descriptors.npz"
    descriptor_temporary = descriptor_path.with_suffix(".tmp.npz")
    np.savez_compressed(descriptor_temporary, descriptors=descriptor_array)
    descriptor_temporary.replace(descriptor_path)
    manifest_path = output_dir / "reference_manifest.json"
    manifest_temporary = manifest_path.with_suffix(".tmp.json")
    manifest_temporary.write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)
    return descriptor_array, records


def select_candidates_from_descriptors(
    candidate_scores: Iterable[float],
    candidate_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    *,
    minimum_sam_score: float,
    maximum_objects: int,
    cad_similarity_weight: float,
    cad_similarity_temperature: float,
    minimum_combined_score: float,
) -> list[dict[str, float | int]]:
    scores = np.asarray(list(candidate_scores), dtype=np.float64)
    candidates = np.asarray(candidate_descriptors, dtype=np.float64)
    references = np.asarray(reference_descriptors, dtype=np.float64)
    if candidates.ndim != 2 or references.ndim != 2 or candidates.shape[1] != references.shape[1]:
        raise ValueError("candidate and reference descriptors must be compatible 2D arrays")
    if len(scores) != len(candidates):
        raise ValueError("candidate score and descriptor counts differ")
    if maximum_objects < 1:
        raise ValueError("maximum_objects must be positive")
    if not 0.0 <= cad_similarity_weight <= 1.0:
        raise ValueError("cad_similarity_weight must be in [0, 1]")
    if cad_similarity_temperature <= 0:
        raise ValueError("cad_similarity_temperature must be positive")

    distances = np.linalg.norm(candidates[:, None, :] - references[None, :, :], axis=2)
    best_reference = np.argmin(distances, axis=1)
    best_distance = distances[np.arange(len(candidates)), best_reference]
    cad_similarity = np.exp(-best_distance / cad_similarity_temperature)
    combined = (1.0 - cad_similarity_weight) * scores + cad_similarity_weight * cad_similarity
    ranked = []
    for index in range(len(candidates)):
        if scores[index] < minimum_sam_score or combined[index] < minimum_combined_score:
            continue
        ranked.append(
            {
                "candidate_index": index,
                "sam_score": float(scores[index]),
                "cad_distance": float(best_distance[index]),
                "cad_similarity": float(cad_similarity[index]),
                "combined_score": float(combined[index]),
                "reference_index": int(best_reference[index]),
            }
        )
    ranked.sort(key=lambda item: (-float(item["combined_score"]), int(item["candidate_index"])))
    return ranked[:maximum_objects]


def select_grouped_candidates_from_descriptors(
    candidate_scores: Iterable[float],
    candidate_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    reference_groups: Iterable[str],
    **selection_config: Any,
) -> list[dict[str, float | int | str]]:
    scores = np.asarray(list(candidate_scores), dtype=np.float64)
    candidates = np.asarray(candidate_descriptors, dtype=np.float64)
    references = np.asarray(reference_descriptors, dtype=np.float64)
    groups = np.asarray(list(reference_groups), dtype=str)
    if candidates.ndim != 2 or references.ndim != 2 or candidates.shape[1] != references.shape[1]:
        raise ValueError("candidate and reference descriptors must be compatible 2D arrays")
    if len(scores) != len(candidates) or len(groups) != len(references):
        raise ValueError("score, candidate, reference, or group counts differ")

    minimum_sam_score = float(selection_config["minimum_sam_score"])
    maximum_objects = int(selection_config["maximum_objects"])
    weight = float(selection_config["cad_similarity_weight"])
    temperature = float(selection_config["cad_similarity_temperature"])
    minimum_combined = float(selection_config["minimum_combined_score"])
    if maximum_objects < 1 or not 0.0 <= weight <= 1.0 or temperature <= 0:
        raise ValueError("invalid grouped candidate selection configuration")

    pairs = []
    for group_name in sorted(set(groups.tolist())):
        group_indices = np.flatnonzero(groups == group_name)
        group_distances = np.linalg.norm(
            candidates[:, None, :] - references[group_indices][None, :, :], axis=2
        )
        local_best = np.argmin(group_distances, axis=1)
        best_distances = group_distances[np.arange(len(candidates)), local_best]
        similarities = np.exp(-best_distances / temperature)
        combined = (1.0 - weight) * scores + weight * similarities
        for candidate_index in range(len(candidates)):
            if scores[candidate_index] < minimum_sam_score:
                continue
            if combined[candidate_index] < minimum_combined:
                continue
            reference_index = int(group_indices[int(local_best[candidate_index])])
            pairs.append(
                {
                    "candidate_index": candidate_index,
                    "sam_score": float(scores[candidate_index]),
                    "cad_distance": float(best_distances[candidate_index]),
                    "cad_similarity": float(similarities[candidate_index]),
                    "combined_score": float(combined[candidate_index]),
                    "reference_index": reference_index,
                    "reference_group": group_name,
                }
            )
    group_names = sorted(set(groups.tolist()))
    group_bit = {name: 1 << index for index, name in enumerate(group_names)}
    options_by_candidate: dict[int, list[dict[str, float | int | str]]] = {}
    for pair in pairs:
        options_by_candidate.setdefault(int(pair["candidate_index"]), []).append(pair)

    # Exact assignment is cheap for seven groups: O(candidates * 2^groups * groups).
    # Each state stores the highest total score for one used-group bitmask.
    states: dict[int, tuple[float, tuple[dict[str, float | int | str], ...]]] = {
        0: (0.0, ())
    }
    for candidate_index in sorted(options_by_candidate):
        next_states = dict(states)
        for mask, (score, choices) in states.items():
            if mask.bit_count() >= maximum_objects:
                continue
            for pair in sorted(
                options_by_candidate[candidate_index],
                key=lambda item: str(item["reference_group"]),
            ):
                bit = group_bit[str(pair["reference_group"])]
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_score = score + float(pair["combined_score"])
                new_choices = choices + (pair,)
                incumbent = next_states.get(new_mask)
                choice_key = tuple(
                    (int(item["candidate_index"]), str(item["reference_group"]))
                    for item in new_choices
                )
                incumbent_key = (
                    tuple(
                        (int(item["candidate_index"]), str(item["reference_group"]))
                        for item in incumbent[1]
                    )
                    if incumbent is not None
                    else ()
                )
                if (
                    incumbent is None
                    or new_score > incumbent[0] + 1e-12
                    or (
                        abs(new_score - incumbent[0]) <= 1e-12
                        and choice_key < incumbent_key
                    )
                ):
                    next_states[new_mask] = (new_score, new_choices)
        states = next_states

    best_mask = min(
        states,
        key=lambda mask: (
            -mask.bit_count(),
            -states[mask][0],
            tuple(
                (int(item["candidate_index"]), str(item["reference_group"]))
                for item in states[mask][1]
            ),
        ),
    )
    selected = list(states[best_mask][1])
    selected.sort(key=lambda item: str(item["reference_group"]))
    return selected


def select_cad_supported_masks(
    candidate_masks: np.ndarray,
    candidate_scores: Iterable[float],
    reference_descriptors: np.ndarray,
    reference_groups: Iterable[str] | None = None,
    **selection_config: Any,
) -> list[dict[str, float | int | str]]:
    masks = np.asarray(candidate_masks)
    if masks.ndim != 3:
        raise ValueError(f"candidate masks must have shape (N,H,W), got {masks.shape}")
    descriptors = np.stack([silhouette_descriptor(mask) for mask in masks])
    if reference_groups is None:
        return select_candidates_from_descriptors(
            candidate_scores,
            descriptors,
            reference_descriptors,
            **selection_config,
        )
    return select_grouped_candidates_from_descriptors(
        candidate_scores,
        descriptors,
        reference_descriptors,
        reference_groups,
        **selection_config,
    )
