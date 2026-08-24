#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pdi_eval.perception.dinov2_reference_boxes import (
    Dinov2DenseEncoder,
    crop_reference_foreground,
    discover_reference_groups,
    load_prompt_frame,
    patch_grid_size,
)


ROOT = Path("/root/autodl-tmp/pdi")
MODEL = ROOT / "models/dinov2/base-f9e44c814b77203eaa57a6bdbbd535f21ede1415"
VIDEO = ROOT / "runs/cosmos-2.5/videos/0000.mp4"
REFERENCES = ROOT / "runs/dinov2-box-diagnostic-0000/references"


def foreground_crop(image: Image.Image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground = np.max(rgb, axis=2) > 8
    rows, columns = np.where(foreground)
    x1, x2 = int(columns.min()), int(columns.max()) + 1
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    pad_x = int(round((x2 - x1) * 0.12))
    pad_y = int(round((y2 - y1) * 0.12))
    crop_box = (
        max(0, x1 - pad_x), max(0, y1 - pad_y),
        min(image.width, x2 + pad_x), min(image.height, y2 + pad_y),
    )
    return image.convert("RGB").crop(crop_box), Image.fromarray((foreground * 255).astype(np.uint8)).crop(crop_box)


def candidate(encoder, scene, scene_features, path):
    with Image.open(path) as image:
        crop, mask = foreground_crop(image)
    features, _ = encoder.encode(crop, 448)
    patch_size = encoder.patch_size
    resized_hw = patch_grid_size((crop.height, crop.width), 448, patch_size)
    resized_mask = np.asarray(mask.resize((resized_hw[1], resized_hw[0]), Image.Resampling.NEAREST)) > 0
    mask_grid = resized_mask.reshape(
        features.shape[0], patch_size, features.shape[1], patch_size
    ).mean(axis=(1, 3)) >= 0.25
    ref_y, ref_x = np.where(mask_grid)
    reference = features[mask_grid]
    flat_scene = scene_features.reshape(-1, scene_features.shape[-1])
    similarities = flat_scene @ reference.T
    best_ref_for_scene = similarities.argmax(axis=1)
    best_scene_for_ref = similarities.argmax(axis=0)
    ref_indices = np.arange(len(reference))
    mutual = best_ref_for_scene[best_scene_for_ref] == ref_indices
    scores = similarities[best_scene_for_ref, ref_indices]
    mutual &= scores >= 0.25
    scene_indices = best_scene_for_ref[mutual]
    if len(scene_indices) < 3:
        return None
    source = np.column_stack((ref_x[mutual] + 0.5, ref_y[mutual] + 0.5)).astype(np.float32)
    scene_y, scene_x = np.unravel_index(scene_indices, scene_features.shape[:2])
    target = np.column_stack((scene_x + 0.5, scene_y + 0.5)).astype(np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=3000,
        confidence=0.995,
    )
    if matrix is None or inliers is None:
        return None
    inliers = inliers.ravel().astype(bool)
    if inliers.sum() < 3:
        return None
    corners = np.asarray(
        [[0, 0], [features.shape[1], 0], [features.shape[1], features.shape[0]], [0, features.shape[0]]],
        dtype=np.float32,
    )
    projected = cv2.transform(corners[None], matrix)[0]
    x1 = int(np.floor(projected[:, 0].min() * scene.width / scene_features.shape[1]))
    x2 = int(np.ceil(projected[:, 0].max() * scene.width / scene_features.shape[1]))
    y1 = int(np.floor(projected[:, 1].min() * scene.height / scene_features.shape[0]))
    y2 = int(np.ceil(projected[:, 1].max() * scene.height / scene_features.shape[0]))
    box = [max(0, x1), max(0, y1), min(scene.width, x2), min(scene.height, y2)]
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
    score = float(scores[mutual][inliers].mean() * np.sqrt(inliers.sum()))
    return score, int(inliers.sum()), len(scene_indices), scale, box


def main():
    scene = load_prompt_frame(VIDEO, 0)
    encoder = Dinov2DenseEncoder(MODEL)
    scene_features, _ = encoder.encode(scene, 840)
    groups = discover_reference_groups(REFERENCES)
    for name, paths in groups.items():
        candidates = []
        for path in paths:
            result = candidate(encoder, scene, scene_features, path)
            if result is not None:
                candidates.append((*result, path.name))
        candidates.sort(reverse=True)
        print(name)
        for item in candidates[:5]:
            print(item)
        if name == "link1":
            print("ALL_LINK1")
            for item in candidates:
                print(item)
        if candidates:
            boxes = np.asarray([item[4] for item in candidates], dtype=np.float64)
            similarities = np.zeros((len(boxes), len(boxes)), dtype=np.float64)
            for i, first in enumerate(boxes):
                for j, second in enumerate(boxes):
                    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
                        0, min(first[3], second[3]) - max(first[1], second[1])
                    )
                    union = (
                        (first[2] - first[0]) * (first[3] - first[1])
                        + (second[2] - second[0]) * (second[3] - second[1])
                        - intersection
                    )
                    similarities[i, j] = intersection / max(union, 1)
            support = (similarities >= 0.35).sum(axis=1)
            medoid = max(
                range(len(boxes)),
                key=lambda index: (support[index], similarities[index].sum(), candidates[index][0]),
            )
            members = similarities[medoid] >= 0.35
            consensus = np.median(boxes[members], axis=0).round().astype(int).tolist()
            print("CONSENSUS", int(members.sum()), consensus, candidates[medoid])


if __name__ == "__main__":
    main()
