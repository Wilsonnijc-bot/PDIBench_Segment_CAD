#!/usr/bin/env python3
"""Inspect the unguarded, box-relative Link 5 SAM3 prompt on source frame zero."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/probe_link5_first_frames.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from infrastructure.shared.inference.dinov2_reference_boxes import (
    Dinov2DenseEncoder,
    discover_reference_groups,
    load_prompt_frame,
    localize_reference_groups,
    xyxy_to_normalized_xywh,
)
from robot.preprocessing.segmentation.sam3_dinov2_segment import (
    LINK5_TEXT_PROMPT,
    _active_franka_groups,
    _link5_prompt_box,
    _link5_seed_points,
    _refine_link5_wrist_prompt,
    _select_prompt_result,
    _validate_franka_groups,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preview(image, mask: np.ndarray | None, box, points: list[list[int]],
             *, show_points: bool = True) -> np.ndarray:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if mask is not None:
        if mask.shape != canvas.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match frame {canvas.shape[:2]}")
        mask = np.asarray(mask, dtype=np.uint8)
        canvas[mask > 0] = (
            0.45 * canvas[mask > 0] + 0.55 * np.array([50, 210, 60])
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (20, 20, 235), 2)
    cv2.rectangle(canvas, box[:2], box[2:], (70, 255, 125), 2)
    for index, (x, y) in enumerate(points if show_points else []):
        positive = index < 3
        color = (70, 255, 125) if positive else (40, 55, 240)
        label = f"P{index + 1}" if positive else f"N{index - 2}"
        cv2.circle(canvas, (x, y), 5, color, -1)
        cv2.putText(canvas, label, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _stage_strip(stages: list[np.ndarray]) -> np.ndarray:
    labels = ("INITIAL POINTS", "AFTER VLM", "FINAL MASK")
    panels = []
    for label, stage in zip(labels, stages, strict=True):
        banner = np.zeros((38, stage.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, label, (16, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack((banner, stage)))
    return np.hstack(panels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--dinov2-model", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3-bpe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True,
                        help="DATASET:VIDEO_NUMBER, for example COSMOS3:0005")
    parser.add_argument("--vlm-guard", action="store_true",
                        help="Review N1, N2, and P3 together before SAM3 point refinement")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    groups = _active_franka_groups(discover_reference_groups(args.reference_dir))
    _validate_franka_groups(groups)
    encoder = Dinov2DenseEncoder(args.dinov2_model, "cuda")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(args.sam3_checkpoint), bpe_path=str(args.sam3_bpe),
        apply_temporal_disambiguation=False,
    )
    records: list[dict] = []
    panels: list[np.ndarray] = []
    try:
        for case in args.case:
            dataset, number = case.split(":", 1)
            source = args.video_root / dataset / f"{number}.mp4"
            case_dir = args.output_root / f"{dataset}_{number}"
            case_dir.mkdir(parents=True, exist_ok=True)
            record: dict = {"case": case, "source": str(source), "frame_index": 0}
            session_id = None
            try:
                if not source.is_file():
                    raise FileNotFoundError(source)
                image = load_prompt_frame(source, 0)
                boxes, _ = localize_reference_groups(
                    image, groups, encoder, scene_side=840, reference_side=448,
                    top_fraction=0.12, padding_fraction=0.12,
                    minimum_contrast=0.02, reference_spatial_priors=True,
                )
                target = next(box for box in boxes if box.name == "link5")
                prompt_box = _link5_prompt_box(target.box_xyxy, image.width)
                normalized_box = xyxy_to_normalized_xywh(
                    prompt_box, (image.height, image.width)
                )
                session_id = predictor.handle_request({
                    "type": "start_session", "resource_path": str(source),
                    "offload_video_to_cpu": True,
                })["session_id"]
                output = predictor.handle_request({
                    "type": "add_prompt", "session_id": session_id,
                    "frame_index": 0, "text": LINK5_TEXT_PROMPT,
                    "bounding_boxes": [list(normalized_box)],
                    "bounding_box_labels": [1],
                })["outputs"]
                object_id, initial_mask, score = _select_prompt_result(
                    np.asarray(output["out_obj_ids"], dtype=np.int64),
                    np.asarray(output["out_binary_masks"], dtype=bool),
                    np.asarray(output["out_probs"], dtype=np.float64),
                    prompt_box,
                )
                seed_points, _, _ = _link5_seed_points(prompt_box, initial_mask.shape)
                guarded_points = None
                if args.vlm_guard:
                    from robot.preprocessing.link5_refinement.link5_point_guard import review_link5_points

                    guarded_points, guard_record = review_link5_points(
                        image, prompt_box, seed_points, case_dir
                    )
                    record["guard_decision"] = guard_record["decision"]
                selected_points = seed_points if guarded_points is None else guarded_points
                initial_stage = _preview(image, None, prompt_box,
                                         seed_points.tolist())
                selected_stage = _preview(image, None, prompt_box,
                                          selected_points.tolist())
                refined_mask, refinement = _refine_link5_wrist_prompt(
                    predictor, session_id, 0, object_id, initial_mask,
                    prompt_box, points_override=guarded_points,
                    diagnostic_allow_point_mismatch=True,
                )
                preview = _preview(image, refined_mask, prompt_box,
                                   refinement["points_xy"])
                final_stage = _preview(image, refined_mask, prompt_box,
                                       refinement["points_xy"], show_points=False)
                for name, stage in (
                    ("initial_points.png", initial_stage),
                    ("selected_points.png", selected_stage),
                    ("final_mask.png", final_stage),
                    ("three_stage_replay.png", _stage_strip(
                        [initial_stage, selected_stage, final_stage])),
                ):
                    if not cv2.imwrite(str(case_dir / name), stage):
                        raise RuntimeError(f"Could not save {name}")
                if not cv2.imwrite(str(case_dir / "first_frame_mask.png"), preview):
                    raise RuntimeError("Could not save preview")
                if not cv2.imwrite(str(case_dir / "mask.png"),
                                   refined_mask.astype(np.uint8) * 255):
                    raise RuntimeError("Could not save binary mask")
                record.update({
                    "status": "complete", "dinov2_box_xyxy": list(target.box_xyxy),
                    "sam3_prompt_box_xyxy": list(prompt_box),
                    "initial_mask_pixels": int(initial_mask.sum()),
                    "refined_mask_pixels": int(refined_mask.sum()),
                    "sam3_score": score, "point_refinement": refinement,
                })
                panel = cv2.resize(preview, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.putText(panel, f"{dataset}_{number}", (12, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255),
                            2, cv2.LINE_AA)
                panels.append(panel)
                print(f"CASE_COMPLETE {case} area={int(refined_mask.sum())} "
                      f"labels={refinement['point_labels_accepted']}", flush=True)
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__,
                              error=str(exc))
                print(f"CASE_FAILED {case} {type(exc).__name__}: {exc}", flush=True)
            finally:
                if session_id is not None:
                    predictor.handle_request({"type": "close_session",
                                              "session_id": session_id})
                _write_json(case_dir / "diagnostic.json", record)
                records.append(record)
    finally:
        shutdown = getattr(predictor, "shutdown", None)
        if shutdown is not None:
            shutdown()

    if panels:
        if len(panels) % 2:
            panels.append(np.zeros_like(panels[0]))
        rows = [np.hstack(panels[index:index + 2])
                for index in range(0, len(panels), 2)]
        if not cv2.imwrite(str(args.output_root / "contact_sheet.png"), np.vstack(rows)):
            raise RuntimeError("Could not save contact sheet")
    _write_json(args.output_root / "summary.json", records)
    completed = sum(record["status"] == "complete" for record in records)
    print(f"COMPLETE {completed}/{len(records)}", flush=True)
    return 0 if completed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
