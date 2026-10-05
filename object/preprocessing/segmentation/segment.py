"""Ground one task object and propagate its VLM-guided SAM3 mask."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from object.preprocessing.grounding.prompting import Grounding, call_302, parse_grounding
from object.preprocessing.selection.selection import sha256, write_json


def _first_frame(video: Path) -> tuple[Image.Image, int]:
    capture = cv2.VideoCapture(str(video))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, bgr = capture.read()
    capture.release()
    if not ok or frames < 2:
        raise ValueError(f"video has fewer than two readable frames: {video}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), frames


def _prompt_preview(image: Image.Image, grounding: Grounding, mask: np.ndarray | None,
                    path: Path) -> None:
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if mask is not None:
        overlay = bgr.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([220, 180, 20])).astype(np.uint8)
        bgr = overlay
    x1, y1, x2, y2 = grounding.box_xyxy
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (50, 230, 80), 2)
    cv2.circle(bgr, grounding.point_xy, 5, (50, 230, 80), -1)
    cv2.imwrite(str(path), bgr)


def _initialize_and_track(video: Path, grounding: Grounding, frame_count: int,
                          checkpoint: Path, bpe: Path) -> tuple[np.ndarray, dict]:
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint), bpe_path=str(bpe),
        apply_temporal_disambiguation=False)
    session = None
    try:
        session = predictor.handle_request({"type": "start_session", "resource_path": str(video),
                                            "offload_video_to_cpu": True})["session_id"]
        image, _ = _first_frame(video)
        width, height = image.size
        x1, y1, x2, y2 = grounding.box_xyxy
        px, py = grounding.point_xy
        box = [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]
        initial = predictor.handle_request({
            "type": "add_prompt", "session_id": session, "frame_index": 0,
            "text": "visual", "bounding_boxes": [box], "bounding_box_labels": [1],
        })["outputs"]
        ids = [int(value) for value in initial["out_obj_ids"]]
        candidates = np.asarray(initial["out_binary_masks"], dtype=bool)
        probs = np.asarray(initial["out_probs"], dtype=float).reshape(-1)
        if len(ids) != len(candidates) or not ids:
            raise ValueError("SAM3 box prompt returned no object")
        eligible = [index for index, mask in enumerate(candidates)
                    if mask.shape == (height, width) and mask[py, px] and mask.any()]
        if not eligible:
            raise ValueError("SAM3 box mask does not contain the positive point")
        selected = max(eligible, key=lambda index: float(probs[index]))
        object_id = ids[selected]
        # SAM3's point path needs its video cache primed by the box propagation.
        primed = {int(item["frame_index"]) for item in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": session,
            "propagation_direction": "forward", "start_frame_index": 0,
            "max_frame_num_to_track": frame_count - 1,
        })}
        if len(primed) != frame_count:
            raise ValueError(f"SAM3 cache priming covered {len(primed)}/{frame_count} frames")
        refined = predictor.handle_request({
            "type": "add_prompt", "session_id": session, "frame_index": 0,
            "obj_id": object_id, "points": [[px / width, py / height]], "point_labels": [1],
        })["outputs"]
        refined_ids = [int(value) for value in refined["out_obj_ids"]]
        if object_id not in refined_ids:
            raise ValueError("SAM3 lost the prompted object after point refinement")
        frame_zero = np.asarray(refined["out_binary_masks"][refined_ids.index(object_id)], bool)
        if frame_zero.shape != (height, width) or not frame_zero[py, px] or not frame_zero.any():
            raise ValueError("SAM3 point-refined mask excludes the positive point")
        masks = np.zeros((frame_count, height, width), dtype=bool)
        masks[0] = frame_zero
        seen = {0}
        for response in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": session,
            "propagation_direction": "forward", "start_frame_index": 0,
            "max_frame_num_to_track": frame_count - 1,
        }):
            frame = int(response["frame_index"])
            if not 0 <= frame < frame_count:
                raise ValueError("SAM3 returned an out-of-range frame")
            seen.add(frame)
            output = response["outputs"]
            tracked_ids = [int(value) for value in output["out_obj_ids"]]
            if object_id in tracked_ids:
                masks[frame] = np.asarray(
                    output["out_binary_masks"][tracked_ids.index(object_id)], bool)
        if len(seen) != frame_count:
            raise ValueError(f"SAM3 returned {len(seen)}/{frame_count} frames")
        if not masks[0, py, px]:
            raise ValueError("final frame-zero mask excludes the positive point")
        return masks, {"sam3_object_id": object_id, "sam3_box_probability": float(probs[selected]),
                       "cache_primed_frames": len(primed), "propagated_frames": len(seen),
                       "empty_mask_frames": np.flatnonzero(~masks.any(axis=(1, 2))).tolist(),
                       "mask_areas": masks.sum(axis=(1, 2)).tolist()}
    finally:
        if session is not None:
            predictor.handle_request({"type": "close_session", "session_id": session})
        predictor.shutdown()


def _replay(video: Path, masks: np.ndarray, target: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 16.0
    height, width = masks.shape[1:]
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("cannot open mask replay writer")
    try:
        for mask in masks:
            ok, frame = capture.read()
            if not ok or frame.shape[:2] != (height, width):
                raise ValueError("video/replay frame mismatch")
            overlay = frame.copy()
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([220, 180, 20])).astype(np.uint8)
            writer.write(np.concatenate([frame, overlay], axis=1))
    finally:
        writer.release()
        capture.release()


def segment(video: Path, prompt: str, output: Path, checkpoint: Path, bpe: Path, *, models=("gpt-6-luna", "gemini-3.8-flash")) -> dict:
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
    benchmark = (_workspace_root())
    load_env_file(benchmark / ".env.vlm")
    output.mkdir(parents=True, exist_ok=True)
    image, frame_count = _first_frame(video)
    image.save(output / "first_frame.png")
    record = {"video": str(video), "video_sha256": sha256(video), "generation_prompt": prompt,
              "source_hw": [image.height, image.width], "source_frame_count": frame_count,
              "first_frame_sha256": sha256(output / "first_frame.png"), "vlm_attempts": []}
    for model in models:
        attempt = {"model_requested": model}
        record["vlm_attempts"].append(attempt)
        try:
            response = call_302(image, prompt, model)
            attempt.update(response)
            grounding = parse_grounding(response["answer"], image.width, image.height)
            attempt["grounding"] = {"object_name": grounding.object_name,
                                    "box_xyxy": grounding.box_xyxy, "point_xy": grounding.point_xy}
            _prompt_preview(image, grounding, None, output / f"prompt_{model}.png")
            masks, sam = _initialize_and_track(video, grounding, frame_count, checkpoint, bpe)
            attempt["sam3"] = sam
            attempt["status"] = "complete"
            np.savez_compressed(output / "segmentation.npz", object_masks=masks[:, None],
                                object_names=np.asarray(["task_object"]),
                                object_ids=np.asarray([sam["sam3_object_id"]], dtype=np.int64),
                                masks=masks)
            _prompt_preview(image, grounding, masks[0], output / "first_frame_mask.png")
            _replay(video, masks, output / "masking.mp4")
            record.update(status="complete", target_object=grounding.object_name,
                          grounding=attempt["grounding"], sam3=sam,
                          segmentation_sha256=sha256(output / "segmentation.npz"))
            break
        except Exception as exc:
            attempt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            record["status"] = "unscorable"
        finally:
            write_json(output / "grounding.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3-bpe", type=Path, required=True)
    args = parser.parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    result = segment(args.video, prompt, args.output, args.sam3_checkpoint, args.sam3_bpe)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
