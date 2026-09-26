"""Six-link DINOv2 + SAM3 tracker adapted from PDI-Bench-edited.

Source: src/pdi_eval/perception/sam3_dinov2_segment.py in the user-provided
PDI-Bench-edited checkout. Prompt ranking and tracking association are retained
exactly; PDI-specific reconstruction and metric stages are not included.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .dinov2_reference_boxes import (
    Dinov2DenseEncoder,
    ReferenceBox,
    discover_reference_groups,
    localize_reference_groups,
    write_box_preview,
)


FRANKA_LINK_NAMES = ("link2", "link3", "link4", "link5", "link6", "link7")
RETIRED_FRANKA_LINK_NAMES = {"link1"}
MIN_TRACKING_ASSOCIATION_SCORE = 0.10
MASK_COLORS = (
    (49, 130, 189),
    (57, 174, 88),
    (255, 127, 14),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
)


@dataclass(frozen=True)
class LinkTrackingResult:
    object_masks: np.ndarray
    object_names: tuple[str, ...]
    object_ids: np.ndarray
    metadata: dict[str, Any]


def video_metadata(video_path: Path) -> dict[str, int | float]:
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


def load_video_frame(video_path: Path, frame_index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_index} from {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def select_prompt_result(
    object_ids: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
) -> tuple[int, np.ndarray, float]:
    """Select a SAM prompt result with the exact PDI ranking formula."""
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


def select_tracking_result(
    object_ids: np.ndarray,
    masks: np.ndarray,
    preferred_object_id: int,
    previous_mask: np.ndarray,
) -> tuple[int, np.ndarray] | None:
    """Associate a propagated candidate with the exact PDI tracking formula."""
    if len(object_ids) != len(masks):
        raise RuntimeError("SAM3 returned inconsistent tracking arrays")
    if len(object_ids) == 0:
        return None
    previous_mask = np.asarray(previous_mask, dtype=bool)
    previous_area = int(previous_mask.sum())
    ranked: list[tuple[float, float, int, np.ndarray]] = []
    for index, object_id in enumerate(object_ids.tolist()):
        mask = np.asarray(masks[index], dtype=bool)
        area = int(mask.sum())
        if area == 0:
            continue
        intersection = int(np.logical_and(previous_mask, mask).sum())
        union = previous_area + area - intersection
        overlap = intersection / max(union, 1)
        area_consistency = min(previous_area, area) / max(previous_area, area, 1)
        association_score = 0.75 * overlap + 0.25 * area_consistency
        identity_bonus = 0.10 if int(object_id) == preferred_object_id else 0.0
        ranked.append(
            (
                association_score + identity_bonus,
                association_score,
                int(object_id),
                mask,
            )
        )
    if not ranked:
        return None
    _, association_score, object_id, mask = max(
        ranked, key=lambda item: (item[0], -item[2])
    )
    if association_score < MIN_TRACKING_ASSOCIATION_SCORE:
        return None
    return object_id, mask


def parse_named_text_prompts(
    values: list[str], allowed_names: set[str]
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --link-text-prompt {value!r}; expected NAME=TEXT"
            )
        name, prompt = (part.strip() for part in value.split("=", 1))
        if name not in allowed_names:
            raise ValueError(
                f"Invalid target {name!r}; expected one of {sorted(allowed_names)}"
            )
        if not prompt:
            raise ValueError(f"Invalid --link-text-prompt {value!r}; prompt is empty")
        if name in prompts:
            raise ValueError(f"Duplicate --link-text-prompt for {name}")
        prompts[name] = prompt
    return prompts


def prompt_diagnostics(
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


def frame_measurements(
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.asarray(masks, dtype=bool)
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_mask_preview(
    image: Image.Image,
    object_masks: np.ndarray,
    object_names: tuple[str, ...],
    output_path: Path,
) -> None:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for index, (name, mask) in enumerate(zip(object_names, object_masks, strict=True)):
        color = np.asarray(MASK_COLORS[index % len(MASK_COLORS)], dtype=np.uint8)
        mask = np.asarray(mask, dtype=bool)
        canvas[mask] = (0.38 * canvas[mask] + 0.62 * color).astype(np.uint8)
        rows, columns = np.where(mask)
        if len(columns):
            cv2.putText(
                canvas,
                name,
                (int(columns.min()) + 3, max(20, int(rows.min()) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                tuple(int(value) for value in color),
                2,
                cv2.LINE_AA,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write mask preview: {output_path}")


def validate_franka_groups(groups: dict[str, list[Path]]) -> dict[str, list[Path]]:
    active = {
        name: paths
        for name, paths in groups.items()
        if name not in RETIRED_FRANKA_LINK_NAMES
    }
    names = tuple(sorted(active))
    if names != FRANKA_LINK_NAMES:
        missing = sorted(set(FRANKA_LINK_NAMES).difference(active))
        extra = sorted(set(active).difference(FRANKA_LINK_NAMES))
        raise ValueError(
            "Active Franka references must contain exactly link2 through link7; "
            f"missing={missing}, extra={extra}"
        )
    return active


def load_link_archive(
    archive_path: Path, video_path: Path | None = None
) -> LinkTrackingResult:
    archive_path = archive_path.expanduser().resolve()
    with np.load(archive_path, allow_pickle=False) as archive:
        object_masks = np.asarray(archive["object_masks"], dtype=bool)
        object_names = tuple(
            str(value) for value in np.asarray(archive["object_names"]).tolist()
        )
        object_ids = np.asarray(archive["object_ids"], dtype=np.int64)
    if object_masks.ndim != 4 or object_masks.shape[1] != len(FRANKA_LINK_NAMES):
        raise ValueError(f"Invalid six-link archive shape: {object_masks.shape}")
    if object_names != FRANKA_LINK_NAMES:
        raise ValueError(f"Unexpected link names: {object_names}")
    if not np.array_equal(object_ids, np.arange(2, 8, dtype=np.int64)):
        raise ValueError(f"Unexpected link IDs: {object_ids.tolist()}")
    metadata_path = archive_path.with_name("segmentation.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {"status": "loaded", "artifacts": {"segmentation_npz": str(archive_path)}}
    )
    if video_path is not None:
        info = video_metadata(video_path.expanduser().resolve())
        expected = (int(info["frames"]), int(info["height"]), int(info["width"]))
        actual = (object_masks.shape[0], object_masks.shape[2], object_masks.shape[3])
        if actual != expected:
            raise ValueError(f"Archive/video shape mismatch: {actual} != {expected}")
    return LinkTrackingResult(object_masks, object_names, object_ids, metadata)


def track_links(
    *,
    video_path: Path,
    reference_dir: Path,
    output_dir: Path,
    dinov2_model: Path,
    sam3_checkpoint: Path,
    sam3_bpe: Path,
    device: str = "cuda",
    frame_index: int = 0,
    text_prompt: str = "visual",
    link_text_prompt: list[str] | None = None,
    sam3_temporal_disambiguation: bool = False,
    minimum_tracked_fraction: float = 0.0,
    scene_side: int = 840,
    reference_side: int = 448,
    top_fraction: float = 0.12,
    padding_fraction: float = 0.10,
    minimum_contrast: float = 0.02,
    reference_spatial_priors: bool = True,
) -> LinkTrackingResult:
    """Localize frame 0 and propagate six independent SAM3 link masks."""
    if not 0.0 <= minimum_tracked_fraction <= 1.0:
        raise ValueError("minimum_tracked_fraction must be between 0 and 1")
    if frame_index != 0:
        raise ValueError("The link-crop wrapper requires frame_index 0")
    started = time.perf_counter()
    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = video_metadata(video_path)
    groups = validate_franka_groups(discover_reference_groups(reference_dir))
    prompt_image = load_video_frame(video_path, frame_index)

    localization_started = time.perf_counter()
    encoder = Dinov2DenseEncoder(dinov2_model, device)
    boxes, heatmaps = localize_reference_groups(
        prompt_image,
        groups,
        encoder,
        scene_side=scene_side,
        reference_side=reference_side,
        top_fraction=top_fraction,
        padding_fraction=padding_fraction,
        minimum_contrast=minimum_contrast,
        reference_spatial_priors=reference_spatial_priors,
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
    write_json(
        boxes_path,
        {
            "schema_version": 1,
            "input": str(video_path),
            "frame_index": frame_index,
            "targets": [asdict(box) for box in boxes],
        },
    )
    boxes_preview_path = output_dir / "dinov2_boxes.jpg"
    write_box_preview(prompt_image, boxes, boxes_preview_path)
    np.savez_compressed(output_dir / "dinov2_heatmaps.npz", **heatmaps)

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError("Install the pinned SAM3 environment first") from exc
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(sam3_checkpoint.expanduser().resolve()),
        bpe_path=str(sam3_bpe.expanduser().resolve()),
        apply_temporal_disambiguation=sam3_temporal_disambiguation,
    )
    object_names = tuple(box.name for box in boxes)
    if object_names != FRANKA_LINK_NAMES:
        raise RuntimeError(f"DINO output link order is invalid: {object_names}")
    object_ids = np.arange(2, 8, dtype=np.int64)
    object_masks = np.zeros(
        (
            int(info["frames"]),
            len(boxes),
            int(info["height"]),
            int(info["width"]),
        ),
        dtype=bool,
    )
    named_prompts = parse_named_text_prompts(
        link_text_prompt or [], set(object_names)
    )
    sam_scores: dict[str, float] = {}
    session_object_ids: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []
    diagnostics_path = output_dir / "sam3_prompt_diagnostics.json"
    sam_started = time.perf_counter()
    try:
        for target_index, target in enumerate(boxes):
            session_id = None
            seen_frames: set[int] = set()
            prompt = named_prompts.get(target.name, text_prompt)
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
                        "frame_index": frame_index,
                        "text": prompt,
                        "bounding_boxes": [list(target.box_xywh_normalized)],
                        "bounding_box_labels": [1],
                    }
                )["outputs"]
                candidate_ids = np.asarray(outputs["out_obj_ids"], dtype=np.int64)
                candidate_masks = np.asarray(
                    outputs["out_binary_masks"], dtype=bool
                )
                candidate_scores = np.asarray(outputs["out_probs"], dtype=np.float64)
                object_id, mask, score = select_prompt_result(
                    candidate_ids,
                    candidate_masks,
                    candidate_scores,
                    target.box_xyxy,
                )
                diagnostic = prompt_diagnostics(
                    target.name,
                    target.box_xyxy,
                    candidate_ids,
                    candidate_masks,
                    candidate_scores,
                    object_id,
                )
                diagnostic["text_prompt"] = prompt
                diagnostics.append(diagnostic)
                write_json(diagnostics_path, diagnostics)
                object_masks[frame_index, target_index] = mask
                seen_frames.add(frame_index)
                sam_scores[target.name] = score
                session_object_ids[target.name] = object_id
                active_object_id = object_id
                previous_mask = mask
                reidentifications: list[dict[str, int]] = []

                for response in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "both",
                        "start_frame_index": frame_index,
                    }
                ):
                    current_frame = int(response["frame_index"])
                    seen_frames.add(current_frame)
                    frame_outputs = response["outputs"]
                    frame_ids = np.asarray(
                        frame_outputs["out_obj_ids"], dtype=np.int64
                    )
                    frame_masks = np.asarray(
                        frame_outputs["out_binary_masks"], dtype=bool
                    )
                    selected = select_tracking_result(
                        frame_ids, frame_masks, active_object_id, previous_mask
                    )
                    if selected is not None:
                        next_object_id, next_mask = selected
                        if next_object_id != active_object_id:
                            reidentifications.append(
                                {
                                    "frame_index": current_frame,
                                    "from_object_id": active_object_id,
                                    "to_object_id": next_object_id,
                                }
                            )
                        active_object_id = next_object_id
                        previous_mask = next_mask
                        object_masks[current_frame, target_index] = next_mask
            finally:
                if session_id is not None:
                    predictor.handle_request(
                        {"type": "close_session", "session_id": session_id}
                    )

            missing_responses = sorted(
                set(range(int(info["frames"]))).difference(seen_frames)
            )
            if missing_responses:
                raise RuntimeError(
                    f"SAM3 propagation for {target.name} omitted "
                    f"{len(missing_responses)} frames; first={missing_responses[0]}"
                )
            frame_areas = object_masks[:, target_index].reshape(
                int(info["frames"]), -1
            ).sum(axis=1)
            tracked_frames = int(np.count_nonzero(frame_areas))
            tracked_fraction = tracked_frames / int(info["frames"])
            diagnostic["tracking"] = {
                "tracked_frames": tracked_frames,
                "total_frames": int(info["frames"]),
                "tracked_fraction": tracked_fraction,
                "minimum_area_pixels": int(frame_areas[frame_areas > 0].min())
                if tracked_frames
                else 0,
                "maximum_area_pixels": int(frame_areas.max()),
                "reidentifications": reidentifications,
            }
            write_json(diagnostics_path, diagnostics)
            if tracked_fraction < minimum_tracked_fraction:
                raise RuntimeError(
                    f"SAM3 tracked {target.name} on only {tracked_frames}/"
                    f"{int(info['frames'])} frames ({tracked_fraction:.1%}); "
                    f"minimum is {minimum_tracked_fraction:.1%}"
                )
    finally:
        shutdown = getattr(predictor, "shutdown", None)
        if shutdown is not None:
            shutdown()
    sam_seconds = time.perf_counter() - sam_started

    union_masks = np.any(object_masks, axis=1)
    heights, centers, truncated = frame_measurements(union_masks)
    archive_path = output_dir / "links.npz"
    temporary_archive = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_archive,
        masks=union_masks,
        object_masks=object_masks,
        object_names=np.asarray(object_names),
        object_ids=object_ids,
        h_pixel=heights,
        x_center=centers,
        is_truncated=truncated,
    )
    temporary_archive.replace(archive_path)
    first_frame_mask_path = output_dir / "first_frame_mask.png"
    write_mask_preview(
        prompt_image, object_masks[frame_index], object_names, first_frame_mask_path
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "timing": {
            "dinov2_localization_seconds": localization_seconds,
            "sam3_seconds": sam_seconds,
        },
        "input": str(video_path),
        "video": info,
        "frame_index": frame_index,
        "reference_dir": str(reference_dir.expanduser().resolve()),
        "sam3_temporal_disambiguation": sam3_temporal_disambiguation,
        "minimum_tracked_fraction": minimum_tracked_fraction,
        "targets": [
            {
                **asdict(box),
                "sam3_text_prompt": named_prompts.get(box.name, text_prompt),
                "object_id": int(object_id),
                "sam3_session_object_id": session_object_ids[box.name],
                "sam3_score": sam_scores[box.name],
            }
            for box, object_id in zip(boxes, object_ids, strict=True)
        ],
        "models": {
            "dinov2": str(dinov2_model.expanduser().resolve()),
            "sam3_checkpoint": str(sam3_checkpoint.expanduser().resolve()),
            "sam3_bpe": str(sam3_bpe.expanduser().resolve()),
        },
        "artifacts": {
            "segmentation_npz": str(archive_path),
            "boxes_json": str(boxes_path),
            "boxes_preview": str(boxes_preview_path),
            "heatmaps": str(output_dir / "dinov2_heatmaps.npz"),
            "first_frame_mask": str(first_frame_mask_path),
            "sam3_prompt_diagnostics": str(diagnostics_path),
        },
    }
    write_json(output_dir / "segmentation.json", metadata)
    return LinkTrackingResult(object_masks, object_names, object_ids, metadata)
