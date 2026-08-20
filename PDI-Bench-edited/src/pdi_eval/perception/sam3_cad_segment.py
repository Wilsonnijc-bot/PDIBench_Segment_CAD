#!/usr/bin/env python3
"""Segment and track Franka links with SAM3, using CAD renders to select proposals."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .cad_reference import build_reference_bank, select_cad_supported_masks, sha256_file
from .segmentation_archive import frame_measurements


def _video_metadata(video_path: Path) -> dict[str, int | float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("SAM3 CAD video segmentation requires opencv-python-headless") from exc
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


def _write_preview(video_path: Path, mask: np.ndarray, output_path: Path) -> None:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot read first video frame: {video_path}")
    overlay = frame.copy()
    tint = np.zeros_like(frame)
    tint[:, :, 1] = 255
    foreground = mask.astype(bool)
    overlay[foreground] = cv2.addWeighted(
        frame[foreground], 0.35, tint[foreground], 0.65, 0.0
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Cannot write preview: {output_path}")


def _selection_config(sam3_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "minimum_sam_score": float(sam3_config["minimum_sam_score"]),
        "maximum_objects": int(sam3_config["maximum_objects"]),
        "cad_similarity_weight": float(sam3_config["cad_similarity_weight"]),
        "cad_similarity_temperature": float(sam3_config["cad_similarity_temperature"]),
        "minimum_combined_score": float(sam3_config["minimum_combined_score"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    project_root = args.project_root.resolve()
    video_path = args.input.resolve()
    manifest_path = args.manifest.resolve()
    output_npz = args.output_npz.resolve()
    output_dir = output_npz.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    video = _video_metadata(video_path)

    reference_descriptors, reference_records = build_reference_bank(
        manifest,
        project_root,
        output_dir / "cad-renders",
    )

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError("Install the pinned facebookresearch/sam3 package first") from exc

    predictor_kwargs = {}
    if args.checkpoint is not None:
        predictor_kwargs["checkpoint_path"] = str(args.checkpoint.resolve())
    predictor = build_sam3_video_predictor(**predictor_kwargs)
    session_id = None
    sam3_config = manifest["sam3"]
    prompt_frame = int(sam3_config["prompt_frame"])
    try:
        session = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video_path),
                "offload_video_to_cpu": True,
            }
        )
        session_id = session["session_id"]
        prompted = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame,
                "text": str(sam3_config["text_prompt"]),
                "output_prob_thresh": float(sam3_config["minimum_sam_score"]),
            }
        )["outputs"]
        candidate_ids = np.asarray(prompted["out_obj_ids"], dtype=np.int64)
        candidate_masks = np.asarray(prompted["out_binary_masks"], dtype=bool)
        candidate_scores = np.asarray(prompted["out_probs"], dtype=np.float64)
        if len(candidate_ids) == 0:
            raise RuntimeError("SAM3 returned no first-frame candidates for the configured prompt")
        if len(candidate_ids) != len(candidate_masks) or len(candidate_ids) != len(candidate_scores):
            raise RuntimeError("SAM3 returned inconsistent candidate arrays")

        ranked = select_cad_supported_masks(
            candidate_masks,
            candidate_scores,
            reference_descriptors,
            reference_groups=[record["mesh"] for record in reference_records],
            **_selection_config(sam3_config),
        )
        if not ranked:
            raise RuntimeError("No SAM3 proposal passed the CAD similarity thresholds")
        selected_indices = [int(item["candidate_index"]) for item in ranked]
        selected_ids = {int(candidate_ids[index]) for index in selected_indices}
        for item in ranked:
            reference = reference_records[int(item["reference_index"])]
            item["matched_mesh"] = reference["mesh"]
            item["matched_azimuth_degrees"] = reference["azimuth_degrees"]
            item["matched_elevation_degrees"] = reference["elevation_degrees"]
            item["sam3_object_id"] = int(candidate_ids[int(item["candidate_index"])])

        for object_id in candidate_ids.tolist():
            if int(object_id) not in selected_ids:
                predictor.handle_request(
                    {
                        "type": "remove_object",
                        "session_id": session_id,
                        "frame_index": prompt_frame,
                        "obj_id": int(object_id),
                    }
                )

        frame_object_masks: dict[int, dict[int, np.ndarray]] = {
            prompt_frame: {
                int(candidate_ids[index]): candidate_masks[index]
                for index in selected_indices
            }
        }
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "both",
                "start_frame_index": prompt_frame,
                "output_prob_thresh": float(sam3_config["minimum_sam_score"]),
            }
        ):
            frame_index = int(response["frame_index"])
            outputs = response["outputs"]
            object_ids = np.asarray(outputs["out_obj_ids"], dtype=np.int64)
            object_masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
            frame_object_masks[frame_index] = {
                int(object_id): object_masks[index]
                for index, object_id in enumerate(object_ids)
                if int(object_id) in selected_ids
            }

        missing = [
            index for index in range(int(video["frames"])) if index not in frame_object_masks
        ]
        if missing:
            raise RuntimeError(
                f"SAM3 propagation did not return {len(missing)} frames; first missing frame is {missing[0]}"
            )
        selected_entries = sorted(ranked, key=lambda item: str(item["reference_group"]))
        object_names = [str(item["reference_group"]) for item in selected_entries]
        object_ids = [int(item["sam3_object_id"]) for item in selected_entries]
        object_masks = np.zeros(
            (
                int(video["frames"]),
                len(object_ids),
                int(video["height"]),
                int(video["width"]),
            ),
            dtype=bool,
        )
        for frame_index, masks_by_id in frame_object_masks.items():
            for object_index, object_id in enumerate(object_ids):
                if object_id in masks_by_id:
                    object_masks[frame_index, object_index] = masks_by_id[object_id]
        masks = np.any(object_masks, axis=1)
    finally:
        if session_id is not None:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        shutdown = getattr(predictor, "shutdown", None)
        if shutdown is not None:
            shutdown()

    heights, centers, truncated = frame_measurements(masks)
    temporary_npz = output_npz.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_npz,
        masks=masks,
        object_masks=object_masks,
        object_names=np.asarray(object_names),
        object_ids=np.asarray(object_ids, dtype=np.int64),
        h_pixel=heights,
        x_center=centers,
        is_truncated=truncated,
        selected_obj_ids=np.asarray(sorted(selected_ids), dtype=np.int64),
    )
    temporary_npz.replace(output_npz)
    preview_path = output_dir / "first_frame_mask.png"
    _write_preview(video_path, masks[0], preview_path)

    metadata = {
        "status": "complete",
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_video": str(video_path),
        "input_sha256": sha256_file(video_path),
        "video": video,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_config": manifest,
        "sam3_source_commit": sam3_config["source_commit"],
        "checkpoint": (
            {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint.resolve()),
            }
            if args.checkpoint is not None
            else {"source": "facebook/sam3 Hugging Face default"}
        ),
        "text_prompt": sam3_config["text_prompt"],
        "cad_meshes": manifest["cad"]["meshes"],
        "reference_render_count": len(reference_records),
        "candidate_count": len(candidate_ids),
        "selected": ranked,
        "artifacts": {
            "segmentation_npz": str(output_npz),
            "first_frame_mask": str(preview_path),
            "cad_renders": str((output_dir / "cad-renders").resolve()),
        },
    }
    metadata_path = output_dir / "segmentation.json"
    metadata_temporary = metadata_path.with_suffix(".tmp.json")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main() -> int:
    metadata = run(build_parser().parse_args())
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
