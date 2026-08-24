#!/usr/bin/env python3
"""Compare seven SAM3 link masks with their corresponding Franka CAD silhouettes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from pdi_eval.perception.cad_reference import (
    build_reference_bank,
    silhouette_descriptor,
)
from sam3.model_builder import build_sam3_video_predictor


ROOT = Path("/root/autodl-tmp/pdi")
CODE = ROOT / "code/PDI-Bench-edited"
VIDEO = ROOT / "runs/cosmos-2.5/videos/0000.mp4"
OUTPUT = ROOT / "seven-link-masking-performance/0000"
MANIFEST = CODE / "configs/sam3-cad-franka.yaml"

# Pixel XYWH regions follow the visible kinematic chain from base to hand on frame 0.
# They are normalized before being passed to SAM3. These are localization prompts;
# CAD silhouettes remain shape priors used for link-specific validation/ranking.
LINK_BOXES = {
    "link1": [875, 590, 210, 114],
    "link2": [915, 455, 205, 210],
    "link3": [980, 225, 225, 330],
    "link4": [1080, 55, 200, 260],
    "link5": [880, 0, 325, 135],
    "link6": [680, 15, 310, 175],
    "link7": [625, 70, 135, 225],
}

COLORS = [
    (49, 130, 189),
    (57, 174, 88),
    (255, 127, 14),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
]


def normalized_box(box: list[int], width: int, height: int) -> list[float]:
    x, y, w, h = box
    return [x / width, y / height, w / width, h / height]


def mask_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def box_coverage(mask: np.ndarray, box: list[int]) -> tuple[float, float]:
    x, y, w, h = box
    inside = int(mask[y : y + h, x : x + w].sum())
    area = int(mask.sum())
    box_area = w * h
    return inside / max(area, 1), inside / max(box_area, 1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(union, 1))


def panel(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    canvas = np.full((image.shape[0] + 66, image.shape[1], 3), 248, np.uint8)
    canvas[66:] = image
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (20, 20, 20), 2)
    cv2.putText(canvas, subtitle, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1)
    return canvas


def main() -> None:
    import yaml

    OUTPUT.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read {VIDEO}")
    height, width = frame.shape[:2]

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    descriptors, records = build_reference_bank(
        manifest, CODE, OUTPUT / "cad-renders"
    )

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(ROOT / "models/sam3/sam3.pt"),
        bpe_path=str(ROOT / "models/sam3/bpe_simple_vocab_16e6.txt.gz"),
    )
    session_id = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(VIDEO),
            "offload_video_to_cpu": True,
        }
    )["session_id"]

    masks: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    try:
        for link_name, box in LINK_BOXES.items():
            output = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": "white robotic arm link",
                    "bounding_boxes": [normalized_box(box, width, height)],
                    "bounding_box_labels": [1],
                    "output_prob_thresh": 0.10,
                }
            )["outputs"]
            candidates = np.asarray(output["out_binary_masks"], dtype=bool)
            scores = np.asarray(output["out_probs"], dtype=float)
            if len(candidates) == 0:
                masks[link_name] = np.zeros((height, width), dtype=bool)
                metrics[link_name] = {
                    "sam3_score": None,
                    "area_pixels": 0,
                    "status": "no-mask",
                }
                continue

            reference_indices = [
                index for index, record in enumerate(records) if record["mesh"] == link_name
            ]
            ranked: list[tuple[float, float, int, int]] = []
            for candidate_index, candidate in enumerate(candidates):
                descriptor = silhouette_descriptor(candidate)
                distances = np.linalg.norm(
                    descriptors[reference_indices] - descriptor[None, :], axis=1
                )
                local_reference = int(np.argmin(distances))
                cad_distance = float(distances[local_reference])
                combined = 0.55 * float(scores[candidate_index]) + 0.45 * np.exp(
                    -cad_distance / 0.20
                )
                ranked.append(
                    (combined, cad_distance, candidate_index, reference_indices[local_reference])
                )
            combined, cad_distance, candidate_index, reference_index = max(
                ranked, key=lambda item: (item[0], -item[1], -item[2])
            )
            selected = candidates[candidate_index]
            masks[link_name] = selected
            contained, prompt_fill = box_coverage(selected, box)
            metrics[link_name] = {
                "sam3_score": float(scores[candidate_index]),
                "area_pixels": int(selected.sum()),
                "mask_box_xywh": mask_box(selected),
                "prompt_box_xywh": box,
                "mask_inside_prompt_box": contained,
                "prompt_box_fill": prompt_fill,
                "cad_distance": cad_distance,
                "cad_similarity": float(np.exp(-cad_distance / 0.20)),
                "combined_score": float(combined),
                "matched_cad_render": records[reference_index],
                "candidate_count": int(len(candidates)),
                "status": "mask",
            }
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        predictor.shutdown()

    names = list(LINK_BOXES)
    pairwise = np.eye(len(names), dtype=float)
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i != j:
                pairwise[i, j] = iou(masks[name_i], masks[name_j])
    union = np.any(np.stack([masks[name] for name in names]), axis=0)
    sum_area = sum(int(masks[name].sum()) for name in names)
    overlap_pixels = int(sum_area - union.sum())

    overlay = frame.copy()
    for index, name in enumerate(names):
        mask = masks[name]
        color = np.asarray(COLORS[index], dtype=np.uint8)
        overlay[mask] = (0.42 * overlay[mask] + 0.58 * color).astype(np.uint8)
        x, y, w, h = LINK_BOXES[name]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLORS[index], 2)
        cv2.putText(
            overlay, name, (x + 3, min(y + 20, height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[index], 2,
        )
    cv2.imwrite(str(OUTPUT / "seven-mask-overlay.jpg"), overlay)

    row_panels = []
    for index, name in enumerate(names):
        metric = metrics[name]
        mask_preview = frame.copy()
        mask = masks[name]
        color = np.asarray(COLORS[index], dtype=np.uint8)
        mask_preview[mask] = (0.35 * mask_preview[mask] + 0.65 * color).astype(np.uint8)
        x, y, w, h = LINK_BOXES[name]
        cv2.rectangle(mask_preview, (x, y), (x + w, y + h), COLORS[index], 3)
        cad_path = Path(str(metric["matched_cad_render"]["render"])) if metric["status"] == "mask" else None
        cad = cv2.imread(str(cad_path), cv2.IMREAD_GRAYSCALE) if cad_path else np.zeros((512, 512), np.uint8)
        cad = cv2.cvtColor(cad, cv2.COLOR_GRAY2BGR)
        cad = cv2.resize(cad, (390, 260), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(mask_preview, (473, 260), interpolation=cv2.INTER_AREA)
        score = metric.get("sam3_score")
        subtitle = (
            f"SAM={score:.3f}  area={metric['area_pixels']}  "
            f"CAD={metric.get('cad_similarity', 0):.3f}  "
            f"inside={metric.get('mask_inside_prompt_box', 0):.1%}"
            if score is not None else "SAM3 returned no mask"
        )
        left = panel(crop, f"{name}: SAM3 text + link box", subtitle)
        render = metric.get("matched_cad_render", {})
        cad_subtitle = (
            f"az={render.get('azimuth_degrees', 0):.0f}  "
            f"el={render.get('elevation_degrees', 0):.0f}"
        )
        right = panel(cad, f"{name}: closest CAD silhouette", cad_subtitle)
        row_panels.append(np.concatenate((left, right), axis=1))
        cv2.imwrite(str(OUTPUT / f"{name}.jpg"), row_panels[-1])
    montage = np.concatenate(row_panels, axis=0)
    cv2.imwrite(str(OUTPUT / "seven-link-cad-comparison.jpg"), montage)

    summary = {
        "status": "diagnostic-only",
        "input_video": str(VIDEO),
        "frame_index": 0,
        "text_prompt": "white robotic arm link",
        "prompt_strategy": "one SAM3 model; seven independent text+positive-box queries",
        "cad_role": "per-link silhouette validation and candidate ranking",
        "duration_seconds": time.perf_counter() - started,
        "link_count": len(names),
        "nonempty_mask_count": sum(bool(masks[name].any()) for name in names),
        "union_area_pixels": int(union.sum()),
        "summed_area_pixels": sum_area,
        "overlap_pixels": overlap_pixels,
        "overlap_fraction_of_union": overlap_pixels / max(int(union.sum()), 1),
        "maximum_pairwise_iou": float(np.max(pairwise - np.eye(len(names)))),
        "mean_offdiagonal_iou": float(
            (pairwise.sum() - len(names)) / (len(names) * (len(names) - 1))
        ),
        "links": metrics,
        "pairwise_iou": {
            name: {other: float(pairwise[i, j]) for j, other in enumerate(names)}
            for i, name in enumerate(names)
        },
        "artifacts": {
            "overlay": str(OUTPUT / "seven-mask-overlay.jpg"),
            "cad_comparison": str(OUTPUT / "seven-link-cad-comparison.jpg"),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
