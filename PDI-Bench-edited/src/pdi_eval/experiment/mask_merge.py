#!/usr/bin/env python3
"""Merge a completed persistent gripper mask into a named PDI segmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _union_measurements(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the canonical union-mask measurements in segmentation_archive."""
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


def build_refined_segmentation(
    *,
    video: Path,
    base_segmentation: Path,
    persistent_work: Path,
    case: str,
    output_npz: Path,
    object_name: str = "link7",
) -> dict[str, Any]:
    """Use the package's validated full-video output, preserving other links."""
    video = video.resolve()
    base_segmentation = base_segmentation.resolve()
    persistent_work = persistent_work.resolve()
    output_npz = output_npz.resolve()
    if not video.is_file() or not base_segmentation.is_file():
        raise FileNotFoundError("video and base segmentation must both exist")
    if output_npz.exists():
        raise FileExistsError(f"refined segmentation already exists: {output_npz}")
    manifest_path = output_npz.with_suffix(".json")
    if manifest_path.exists():
        raise FileExistsError(f"refinement manifest already exists: {manifest_path}")
    provenance_path = persistent_work / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    result = provenance["results"][case]
    if result.get("status") != "completed_checks":
        raise ValueError(f"{case} persistent masking status is {result.get('status')!r}")
    video_hash = sha256_file(video)
    if result.get("source_sha256") != video_hash:
        raise ValueError("persistent masking source video hash does not match")

    mask_source = Path(result["mask_source"])
    if not mask_source.is_file():
        mask_source = persistent_work / "sam" / case / "seed_masks.npz"
    if not mask_source.is_file():
        raise FileNotFoundError(f"persistent full-video masks are missing: {mask_source}")
    mask_hash = sha256_file(mask_source)
    if result.get("masks_sha256") != mask_hash:
        raise ValueError("persistent mask archive hash does not match provenance")
    with np.load(mask_source, allow_pickle=False) as archive:
        refined_mask = np.asarray(archive["masks"], dtype=bool)
    if refined_mask.ndim != 3:
        raise ValueError("persistent mask must have shape (T,H,W)")
    if not np.all(refined_mask.any(axis=(1, 2))):
        raise ValueError("completed persistent mask has empty video frames")

    with np.load(base_segmentation, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"object_masks", "object_names", "object_ids"}
    if not required.issubset(payload):
        raise ValueError(f"base segmentation lacks {sorted(required - payload.keys())}")
    object_masks = np.asarray(payload["object_masks"], dtype=bool).copy()
    object_names = tuple(str(name) for name in payload["object_names"].tolist())
    if object_masks.ndim != 4 or len(object_names) != object_masks.shape[1]:
        raise ValueError("base segmentation has invalid object dimensions")
    if payload["object_ids"].shape != (len(object_names),):
        raise ValueError("base segmentation object IDs do not match names")
    if object_names.count(object_name) != 1:
        raise ValueError(f"expected exactly one {object_name!r} object")
    if refined_mask.shape != (object_masks.shape[0], *object_masks.shape[2:]):
        raise ValueError(
            f"persistent mask shape {refined_mask.shape} does not match "
            f"base segmentation {(object_masks.shape[0], *object_masks.shape[2:])}"
        )
    if result.get("source_frame_count") != len(refined_mask):
        raise ValueError("persistent source frame count does not match masks")
    if result.get("source_hw") != list(refined_mask.shape[1:]):
        raise ValueError("persistent source dimensions do not match masks")

    index = object_names.index(object_name)
    baseline = object_masks[:, index].copy()
    object_masks[:, index] = refined_mask
    union = np.any(object_masks, axis=1)
    heights, centers, truncated = _union_measurements(union)
    payload.update(
        object_masks=object_masks,
        masks=union,
        h_pixel=heights,
        x_center=centers,
        is_truncated=truncated,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_npz.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output_npz)

    manifest = {
        "schema_version": 1,
        "method": "persistent-mask-link-replacement",
        "case": case,
        "object_name": object_name,
        "video": str(video),
        "video_sha256": video_hash,
        "base_segmentation": str(base_segmentation),
        "base_segmentation_sha256": sha256_file(base_segmentation),
        "persistent_provenance": str(provenance_path),
        "persistent_mask_source": str(mask_source),
        "persistent_mask_sha256": mask_hash,
        "output_segmentation": str(output_npz),
        "output_segmentation_sha256": sha256_file(output_npz),
        "frame_count": len(refined_mask),
        "changed_target_pixels": int(np.count_nonzero(baseline ^ refined_mask)),
        "changed_target_frames": int(np.count_nonzero(np.any(baseline ^ refined_mask, axis=(1, 2)))),
        "unchanged_object_names": [name for name in object_names if name != object_name],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--base-segmentation", type=Path, required=True)
    parser.add_argument("--persistent-work", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-name", default="link7")
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_refined_segmentation(
        video=args.video,
        base_segmentation=args.base_segmentation,
        persistent_work=args.persistent_work,
        case=args.case,
        object_name=args.object_name,
        output_npz=args.output_npz,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
