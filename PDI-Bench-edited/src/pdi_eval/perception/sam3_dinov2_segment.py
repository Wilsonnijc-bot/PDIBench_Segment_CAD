#!/usr/bin/env python3
"""Segment a video with SAM3 boxes generated from new DINOv2 references."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dinov2_reference_boxes import (
    Dinov2DenseEncoder,
    discover_reference_groups,
    load_prompt_frame,
    localize_reference_groups,
    write_box_preview,
)


FRANKA_LINK_NAMES = tuple(f"link{index}" for index in range(1, 8))
MASK_COLORS = (
    (49, 130, 189),
    (57, 174, 88),
    (255, 127, 14),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
)


def _video_metadata(video_path: Path) -> dict[str, int | float]:
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
    if min(metadata["width"], metadata["height"], metadata["frames"]) <= 0:
        raise ValueError(f"Video has invalid metadata: {metadata}")
    return metadata


def _select_prompt_result(
    object_ids: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
) -> tuple[int, np.ndarray, float]:
    if len(object_ids) != len(masks) or len(object_ids) != len(scores):
        raise RuntimeError("SAM3 returned inconsistent candidate arrays")
    x1, y1, x2, y2 = box_xyxy
    ranked: list[tuple[float, int, np.ndarray, float]] = []
    box_area = max((x2 - x1) * (y2 - y1), 1)
    for index, object_id in enumerate(object_ids.tolist()):
        mask = np.asarray(masks[index], dtype=bool)
        area = int(mask.sum())
        if area == 0:
            continue
        intersection = int(mask[y1:y2, x1:x2].sum())
        inside = intersection / area
        overlap = intersection / max(area + box_area - intersection, 1)
        score = float(scores[index])
        rank_score = 0.45 * inside + 0.35 * overlap + 0.20 * score
        ranked.append((rank_score, int(object_id), mask, score))
    if not ranked:
        raise RuntimeError("SAM3 returned no non-empty object for a DINOv2 box")
    _, object_id, mask, score = max(ranked, key=lambda item: (item[0], -item[1]))
    return object_id, mask, score


def _prompt_diagnostics(
    target_name: str,
    box_xyxy: tuple[int, int, int, int],
    object_ids: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    selected_object_id: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box_xyxy
    box_area = max((x2 - x1) * (y2 - y1), 1)
    candidates = []
    for index, object_id in enumerate(object_ids.tolist()):
        mask = np.asarray(masks[index], dtype=bool)
        area = int(mask.sum())
        intersection = int(mask[y1:y2, x1:x2].sum())
        candidates.append(
            {
                "object_id": int(object_id),
                "score": float(scores[index]),
                "area_pixels": area,
                "mask_inside_box": intersection / max(area, 1),
                "box_iou": intersection / max(area + box_area - intersection, 1),
                "selected": int(object_id) == selected_object_id,
            }
        )
    return {
        "target": target_name,
        "box_xyxy": list(box_xyxy),
        "selected_object_id": selected_object_id,
        "candidates": candidates,
    }


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_franka_groups(groups: dict[str, list[Path]]) -> None:
    names = tuple(sorted(groups))
    if names != FRANKA_LINK_NAMES:
        missing = sorted(set(FRANKA_LINK_NAMES).difference(groups))
        extra = sorted(set(groups).difference(FRANKA_LINK_NAMES))
        raise ValueError(
            f"Franka references must contain exactly link1 through link7; "
            f"missing={missing}, extra={extra}"
        )


def _write_object_mask_preview(
    image: Any,
    object_masks: np.ndarray,
    object_names: list[str],
    output_path: Path,
) -> None:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for index, (name, mask) in enumerate(zip(object_names, object_masks)):
        color = np.asarray(MASK_COLORS[index % len(MASK_COLORS)], dtype=np.uint8)
        mask = np.asarray(mask, dtype=bool)
        canvas[mask] = (0.38 * canvas[mask] + 0.62 * color).astype(np.uint8)
        rows, columns = np.where(mask)
        if len(columns):
            x1, y1 = int(columns.min()), int(rows.min())
            cv2.putText(
                canvas,
                name,
                (x1 + 3, max(20, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                tuple(int(value) for value in color),
                2,
                cv2.LINE_AA,
            )
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write mask preview: {output_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from .segmentation_archive import frame_measurements

    started = time.perf_counter()
    video_path = args.input.resolve()
    output_npz = args.output_npz.resolve()
    output_dir = output_npz.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    video = _video_metadata(video_path)
    groups = discover_reference_groups(args.reference_dir)
    if args.require_franka_links:
        _validate_franka_groups(groups)
    prompt_image = load_prompt_frame(video_path, args.frame_index)

    localization_started = time.perf_counter()
    encoder = Dinov2DenseEncoder(args.dinov2_model, args.device)
    boxes, heatmaps = localize_reference_groups(
        prompt_image,
        groups,
        encoder,
        scene_side=args.scene_side,
        reference_side=args.reference_side,
        top_fraction=args.top_fraction,
        padding_fraction=args.padding_fraction,
        minimum_contrast=args.minimum_contrast,
        reference_spatial_priors=args.reference_spatial_priors,
    )
    localization_seconds = time.perf_counter() - localization_started
    del encoder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    boxes_path = output_dir / "dinov2_boxes.json"
    boxes_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": str(video_path),
                "frame_index": args.frame_index,
                "targets": [asdict(box) for box in boxes],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preview_path = output_dir / "dinov2_boxes.jpg"
    write_box_preview(prompt_image, boxes, preview_path)
    np.savez_compressed(output_dir / "dinov2_heatmaps.npz", **heatmaps)

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError("Install the pinned SAM3 environment first") from exc
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(args.sam3_checkpoint.resolve()),
        bpe_path=str(args.sam3_bpe.resolve()),
    )
    archive_object_ids = np.arange(1, len(boxes) + 1, dtype=np.int64)
    object_masks = np.zeros(
        (int(video["frames"]), len(boxes), int(video["height"]), int(video["width"])),
        dtype=bool,
    )
    sam_scores: dict[str, float] = {}
    session_object_ids: dict[str, int] = {}
    prompt_diagnostic_records: list[dict[str, Any]] = []
    prompt_diagnostics_path = output_dir / "sam3_prompt_diagnostics.json"
    sam_started = time.perf_counter()
    try:
        for target_index, target in enumerate(boxes):
            session_id = None
            seen_frames: set[int] = set()
            try:
                session_id = predictor.handle_request(
                    {
                        "type": "start_session",
                        "resource_path": str(video_path),
                        "offload_video_to_cpu": True,
                    }
                )["session_id"]
                outputs = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": args.frame_index,
                        "text": args.text_prompt,
                        "bounding_boxes": [list(target.box_xywh_normalized)],
                        "bounding_box_labels": [1],
                    }
                )["outputs"]
                candidate_ids = np.asarray(outputs["out_obj_ids"], dtype=np.int64)
                candidate_masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
                candidate_scores = np.asarray(outputs["out_probs"], dtype=np.float64)
                object_id, mask, score = _select_prompt_result(
                    candidate_ids,
                    candidate_masks,
                    candidate_scores,
                    target.box_xyxy,
                )
                diagnostic = _prompt_diagnostics(
                    target.name,
                    target.box_xyxy,
                    candidate_ids,
                    candidate_masks,
                    candidate_scores,
                    object_id,
                )
                prompt_diagnostic_records.append(diagnostic)
                _write_json(prompt_diagnostics_path, prompt_diagnostic_records)
                print(json.dumps({"sam3_prompt": diagnostic}, sort_keys=True), file=sys.stderr)

                object_masks[args.frame_index, target_index] = mask
                seen_frames.add(args.frame_index)
                sam_scores[target.name] = score
                session_object_ids[target.name] = object_id

                for response in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "both",
                        "start_frame_index": args.frame_index,
                    }
                ):
                    frame_index = int(response["frame_index"])
                    seen_frames.add(frame_index)
                    frame_outputs = response["outputs"]
                    frame_ids = np.asarray(frame_outputs["out_obj_ids"], dtype=np.int64)
                    frame_masks = np.asarray(frame_outputs["out_binary_masks"], dtype=bool)
                    matches = np.flatnonzero(frame_ids == object_id)
                    if len(matches):
                        object_masks[frame_index, target_index] = frame_masks[int(matches[0])]
            finally:
                if session_id is not None:
                    predictor.handle_request(
                        {"type": "close_session", "session_id": session_id}
                    )

            missing = sorted(set(range(int(video["frames"]))).difference(seen_frames))
            if missing:
                raise RuntimeError(
                    f"SAM3 propagation for {target.name} omitted {len(missing)} frames; "
                    f"first missing frame is {missing[0]}"
                )
    finally:
        shutdown = getattr(predictor, "shutdown", None)
        if shutdown is not None:
            shutdown()
    sam3_seconds = time.perf_counter() - sam_started
    union_masks = np.any(object_masks, axis=1)
    heights, centers, truncated = frame_measurements(union_masks)
    temporary_npz = output_npz.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_npz,
        masks=union_masks,
        object_masks=object_masks,
        object_names=np.asarray([box.name for box in boxes]),
        object_ids=archive_object_ids,
        h_pixel=heights,
        x_center=centers,
        is_truncated=truncated,
    )
    temporary_npz.replace(output_npz)
    mask_preview_path = output_dir / "first_frame_mask.png"
    _write_object_mask_preview(
        prompt_image,
        object_masks[args.frame_index],
        [box.name for box in boxes],
        mask_preview_path,
    )
    metadata: dict[str, Any] = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "timing": {
            "dinov2_localization_seconds": localization_seconds,
            "sam3_seconds": sam3_seconds,
        },
        "input": str(video_path),
        "video": video,
        "frame_index": args.frame_index,
        "reference_dir": str(args.reference_dir.resolve()),
        "targets": [
            {
                **asdict(box),
                "object_id": int(object_id),
                "sam3_session_object_id": session_object_ids[box.name],
                "sam3_score": sam_scores[box.name],
            }
            for box, object_id in zip(boxes, archive_object_ids)
        ],
        "models": {
            "dinov2": str(args.dinov2_model.resolve()),
            "sam3_checkpoint": str(args.sam3_checkpoint.resolve()),
            "sam3_bpe": str(args.sam3_bpe.resolve()),
        },
        "artifacts": {
            "segmentation_npz": str(output_npz),
            "boxes_json": str(boxes_path),
            "boxes_preview": str(preview_path),
            "first_frame_mask": str(mask_preview_path),
            "sam3_prompt_diagnostics": str(prompt_diagnostics_path),
        },
    }
    metadata_path = output_dir / "segmentation.json"
    _write_json(metadata_path, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--dinov2-model", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3-bpe", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--text-prompt", default="white robotic arm link")
    parser.add_argument("--require-franka-links", action="store_true")
    parser.add_argument("--sam3-threshold", type=float, default=0.10)
    parser.add_argument("--scene-side", type=int, default=840)
    parser.add_argument("--reference-side", type=int, default=448)
    parser.add_argument("--top-fraction", type=float, default=0.12)
    parser.add_argument("--padding-fraction", type=float, default=0.12)
    parser.add_argument("--minimum-contrast", type=float, default=0.02)
    parser.add_argument("--reference-spatial-priors", action="store_true")
    return parser


def main() -> int:
    print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
