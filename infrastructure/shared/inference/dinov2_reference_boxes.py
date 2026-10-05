#!/usr/bin/env python3
"""Localize reference-defined targets with dense DINOv2 patch features."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
REFERENCE_ARTIFACT_PREFIXES = ("contact_sheet", "qa_", "overview")


@dataclass(frozen=True)
class ReferenceBox:
    name: str
    box_xyxy: tuple[int, int, int, int]
    box_xywh_normalized: tuple[float, float, float, float]
    peak_similarity: float
    mean_similarity: float
    similarity_contrast: float
    threshold: float
    component_patches: int
    references: tuple[str, ...]
    localization_strategy: str
    reference_prior_box_xyxy: tuple[int, int, int, int] | None


def discover_reference_groups(reference_dir: Path) -> dict[str, list[Path]]:
    """Return image files grouped by each immediate child directory."""
    reference_dir = reference_dir.resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {reference_dir}")
    group_root = reference_dir / "by_link"
    if not group_root.is_dir():
        group_root = reference_dir
    groups: dict[str, list[Path]] = {}
    for child in sorted(group_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        match = re.fullmatch(r"link[_-]?([1-7])", child.name, flags=re.IGNORECASE)
        name = f"link{match.group(1)}" if match else child.name
        images = sorted(
            path.resolve()
            for path in child.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and not path.stem.lower().startswith(REFERENCE_ARTIFACT_PREFIXES)
        )
        if images:
            if name in groups:
                raise ValueError(f"Reference groups resolve to duplicate target name: {name}")
            groups[name] = images
    if not groups:
        raise ValueError(
            f"No reference groups found under {reference_dir}; expected "
            "<reference-dir>/<target-name>/<image>"
        )
    return groups


def reference_foreground_box(
    image: Image.Image,
    *,
    black_threshold: int = 8,
) -> tuple[int, int, int, int]:
    """Return non-black support as an xyxy box."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground = np.max(rgb, axis=2) > black_threshold
    rows, columns = np.where(foreground)
    if len(columns) == 0:
        raise ValueError("Reference image has no non-black foreground pixels")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def crop_reference_foreground(
    image: Image.Image,
    *,
    black_threshold: int = 8,
    padding_fraction: float = 0.12,
) -> Image.Image:
    """Crop references whose target is isolated on an approximately black canvas."""
    x1, y1, x2, y2 = reference_foreground_box(
        image, black_threshold=black_threshold
    )
    pad_x = int(round((x2 - x1) * padding_fraction))
    pad_y = int(round((y2 - y1) * padding_fraction))
    return image.convert("RGB").crop(
        (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image.width, x2 + pad_x),
            min(image.height, y2 + pad_y),
        )
    )


def guided_box_from_similarity(
    similarity: np.ndarray,
    image_hw: tuple[int, int],
    prior_box: tuple[int, int, int, int],
    *,
    padding_fraction: float = 0.10,
    center_refinement_weight: float = 0.25,
) -> tuple[tuple[int, int, int, int], dict[str, float | int]]:
    """Refine a full-frame reference box with a bounded DINO similarity search."""
    similarity = np.asarray(similarity, dtype=np.float32)
    if similarity.ndim != 2 or not np.isfinite(similarity).all():
        raise ValueError("Similarity must be a finite two-dimensional array")
    image_height, image_width = image_hw
    x1, y1, x2, y2 = prior_box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid reference prior box: {prior_box}")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    smoothed = cv2.GaussianBlur(similarity, (3, 3), 0)
    grid_y, grid_x = np.mgrid[: similarity.shape[0], : similarity.shape[1]]
    pixel_x = (grid_x + 0.5) * image_width / similarity.shape[1]
    pixel_y = (grid_y + 0.5) * image_height / similarity.shape[0]
    distance = ((pixel_x - center_x) / (0.75 * width)) ** 2 + (
        (pixel_y - center_y) / (0.75 * height)
    ) ** 2
    score = (smoothed - np.median(smoothed)) / max(float(np.std(smoothed)), 1e-6)
    score -= distance
    outside = (np.abs(pixel_x - center_x) > 0.75 * width) | (
        np.abs(pixel_y - center_y) > 0.75 * height
    )
    score[outside] = -np.inf
    peak_y, peak_x = np.unravel_index(int(np.argmax(score)), score.shape)
    matched_x = float(pixel_x[peak_y, peak_x])
    matched_y = float(pixel_y[peak_y, peak_x])
    refined_x = (1.0 - center_refinement_weight) * center_x + center_refinement_weight * matched_x
    refined_y = (1.0 - center_refinement_weight) * center_y + center_refinement_weight * matched_y
    padded_width = width * (1.0 + 2.0 * padding_fraction)
    padded_height = height * (1.0 + 2.0 * padding_fraction)
    box = (
        max(0, int(round(refined_x - padded_width / 2.0))),
        max(0, int(round(refined_y - padded_height / 2.0))),
        min(image_width, int(round(refined_x + padded_width / 2.0))),
        min(image_height, int(round(refined_y + padded_height / 2.0))),
    )
    grid_x1 = max(0, int(math.floor(box[0] * similarity.shape[1] / image_width)))
    grid_x2 = min(
        similarity.shape[1], int(math.ceil(box[2] * similarity.shape[1] / image_width))
    )
    grid_y1 = max(0, int(math.floor(box[1] * similarity.shape[0] / image_height)))
    grid_y2 = min(
        similarity.shape[0], int(math.ceil(box[3] * similarity.shape[0] / image_height))
    )
    selected = similarity[grid_y1:grid_y2, grid_x1:grid_x2]
    metrics: dict[str, float | int] = {
        "threshold": float(similarity[peak_y, peak_x]),
        "peak_similarity": float(similarity[peak_y, peak_x]),
        "mean_similarity": float(selected.mean()),
        "similarity_contrast": float(
            similarity[peak_y, peak_x] - np.median(similarity)
        ),
        "component_patches": int(selected.size),
    }
    return box, metrics


def parse_reference_arguments(values: Iterable[str]) -> dict[str, list[Path]]:
    """Parse repeatable NAME=IMAGE reference arguments."""
    groups: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --reference {value!r}; expected NAME=IMAGE")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name:
            raise ValueError(f"Invalid --reference {value!r}; target name is empty")
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"Reference image does not exist or is unsupported: {path}")
        groups.setdefault(name, []).append(path)
    if not groups:
        raise ValueError("At least one --reference or --reference-dir is required")
    return groups


def load_prompt_frame(path: Path, frame_index: int = 0) -> Image.Image:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.suffix.lower() in IMAGE_SUFFIXES:
        if frame_index != 0:
            raise ValueError("Image inputs only support --frame-index 0")
        with Image.open(path) as image:
            return image.convert("RGB")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open input video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_index} from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def patch_grid_size(image_hw: tuple[int, int], maximum_side: int, patch_size: int) -> tuple[int, int]:
    height, width = image_hw
    if min(height, width, maximum_side, patch_size) <= 0:
        raise ValueError("Image and patch dimensions must be positive")
    scale = maximum_side / max(height, width)
    resized_height = max(patch_size, int(round(height * scale / patch_size)) * patch_size)
    resized_width = max(patch_size, int(round(width * scale / patch_size)) * patch_size)
    return resized_height, resized_width


def xyxy_to_normalized_xywh(
    box: tuple[int, int, int, int], image_hw: tuple[int, int]
) -> tuple[float, float, float, float]:
    height, width = image_hw
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid xyxy box {box} for image {(width, height)}")
    return x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height


def box_from_similarity(
    similarity: np.ndarray,
    image_hw: tuple[int, int],
    *,
    top_fraction: float = 0.12,
    padding_fraction: float = 0.12,
    minimum_component_patches: int = 2,
) -> tuple[tuple[int, int, int, int], dict[str, float | int]]:
    """Convert a patch similarity grid into a padded source-image xyxy box."""
    similarity = np.asarray(similarity, dtype=np.float32)
    if similarity.ndim != 2 or min(similarity.shape) == 0:
        raise ValueError("Similarity must be a non-empty two-dimensional array")
    if not np.isfinite(similarity).all():
        raise ValueError("Similarity contains non-finite values")
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be between zero and one")
    if padding_fraction < 0.0:
        raise ValueError("padding_fraction cannot be negative")
    median = float(np.median(similarity))
    threshold = max(
        float(np.quantile(similarity, 1.0 - top_fraction)),
        median + 0.35 * (float(similarity.max()) - median),
    )
    active = (similarity >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    components: list[tuple[float, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_component_patches:
            continue
        values = similarity[labels == label]
        strength = float(values.mean()) + 0.02 * math.log1p(area)
        components.append((strength, area, label))
    if not components:
        peak_y, peak_x = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
        labels = np.zeros_like(active, dtype=np.int32)
        labels[peak_y, peak_x] = 1
        components = [(float(similarity[peak_y, peak_x]), 1, 1)]
    _, area, selected_label = max(components, key=lambda item: (item[0], item[1], -item[2]))
    rows, columns = np.where(labels == selected_label)
    grid_height, grid_width = similarity.shape
    image_height, image_width = image_hw
    x1 = int(math.floor(columns.min() * image_width / grid_width))
    y1 = int(math.floor(rows.min() * image_height / grid_height))
    x2 = int(math.ceil((columns.max() + 1) * image_width / grid_width))
    y2 = int(math.ceil((rows.max() + 1) * image_height / grid_height))
    pad_x = int(round((x2 - x1) * padding_fraction))
    pad_y = int(round((y2 - y1) * padding_fraction))
    box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )
    selected_values = similarity[labels == selected_label]
    metrics: dict[str, float | int] = {
        "threshold": threshold,
        "peak_similarity": float(similarity.max()),
        "mean_similarity": float(selected_values.mean()),
        "similarity_contrast": float(similarity.max() - np.median(similarity)),
        "component_patches": area,
    }
    return box, metrics


class Dinov2DenseEncoder:
    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("DINOv2 localization requires torch and transformers") from exc
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"DINOv2 model directory is missing: {self.model_path}")
        self.model = AutoModel.from_pretrained(
            str(self.model_path), local_files_only=True, use_safetensors=True
        ).eval().to(self.device)
        self.patch_size = int(self.model.config.patch_size)
        self.register_tokens = int(getattr(self.model.config, "num_register_tokens", 0))

    def encode(self, image: Image.Image, maximum_side: int) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        target_hw = patch_grid_size((image.height, image.width), maximum_side, self.patch_size)
        resized = image.convert("RGB").resize((target_hw[1], target_hw[0]), Image.Resampling.LANCZOS)
        pixels = np.array(resized, dtype=np.float32, copy=True) / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(pixel_values=tensor)
        tokens = outputs.last_hidden_state[0]
        cls = tokens[0].float()
        patches = tokens[1 + self.register_tokens :].float()
        grid_hw = (target_hw[0] // self.patch_size, target_hw[1] // self.patch_size)
        if patches.shape[0] != grid_hw[0] * grid_hw[1]:
            raise RuntimeError(
                f"DINOv2 returned {patches.shape[0]} patch tokens for grid {grid_hw}"
            )
        patches = torch.nn.functional.normalize(patches, dim=-1)
        cls = torch.nn.functional.normalize(cls, dim=-1)
        return patches.reshape(*grid_hw, -1).cpu().numpy(), cls.cpu().numpy()


def _reference_prototype(
    encoder: Dinov2DenseEncoder, paths: list[Path], reference_side: int
) -> tuple[np.ndarray, np.ndarray]:
    descriptors = []
    normalized_support_boxes = []
    for path in paths:
        with Image.open(path) as image:
            support = reference_foreground_box(image)
            normalized_support_boxes.append(
                np.asarray(
                    [
                        support[0] / image.width,
                        support[1] / image.height,
                        support[2] / image.width,
                        support[3] / image.height,
                    ],
                    dtype=np.float64,
                )
            )
            reference = crop_reference_foreground(image)
            patches, cls = encoder.encode(reference, reference_side)
        height, width = patches.shape[:2]
        y_margin = int(height * 0.15)
        x_margin = int(width * 0.15)
        center = patches[
            y_margin : max(y_margin + 1, height - y_margin),
            x_margin : max(x_margin + 1, width - x_margin),
        ]
        descriptor = 0.75 * center.mean(axis=(0, 1)) + 0.25 * cls
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
        descriptors.append(descriptor)
    prototype = np.mean(np.stack(descriptors), axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
    return prototype, np.median(np.stack(normalized_support_boxes), axis=0)


def localize_reference_groups(
    image: Image.Image,
    groups: dict[str, list[Path]],
    encoder: Dinov2DenseEncoder,
    *,
    scene_side: int = 840,
    reference_side: int = 448,
    top_fraction: float = 0.12,
    padding_fraction: float = 0.12,
    minimum_contrast: float = 0.02,
    reference_spatial_priors: bool = False,
) -> tuple[list[ReferenceBox], dict[str, np.ndarray]]:
    scene_patches, _ = encoder.encode(image, scene_side)
    boxes: list[ReferenceBox] = []
    heatmaps: dict[str, np.ndarray] = {}
    for name, paths in sorted(groups.items()):
        prototype, normalized_prior = _reference_prototype(encoder, paths, reference_side)
        similarity = np.einsum("hwd,d->hw", scene_patches, prototype)
        reference_prior: tuple[int, int, int, int] | None = None
        if reference_spatial_priors:
            reference_prior = (
                int(round(normalized_prior[0] * image.width)),
                int(round(normalized_prior[1] * image.height)),
                int(round(normalized_prior[2] * image.width)),
                int(round(normalized_prior[3] * image.height)),
            )
            box, metrics = guided_box_from_similarity(
                similarity,
                (image.height, image.width),
                reference_prior,
                padding_fraction=padding_fraction,
            )
            strategy = "full-frame-reference-prior+dense-dinov2"
        else:
            box, metrics = box_from_similarity(
                similarity,
                (image.height, image.width),
                top_fraction=top_fraction,
                padding_fraction=padding_fraction,
            )
            strategy = "dense-dinov2-component"
        if float(metrics["similarity_contrast"]) < minimum_contrast:
            raise RuntimeError(
                f"DINOv2 match for {name!r} is ambiguous: contrast "
                f"{float(metrics['similarity_contrast']):.4f} < {minimum_contrast:.4f}"
            )
        boxes.append(
            ReferenceBox(
                name=name,
                box_xyxy=box,
                box_xywh_normalized=xyxy_to_normalized_xywh(box, (image.height, image.width)),
                peak_similarity=float(metrics["peak_similarity"]),
                mean_similarity=float(metrics["mean_similarity"]),
                similarity_contrast=float(metrics["similarity_contrast"]),
                threshold=float(metrics["threshold"]),
                component_patches=int(metrics["component_patches"]),
                references=tuple(str(path) for path in paths),
                localization_strategy=strategy,
                reference_prior_box_xyxy=reference_prior,
            )
        )
        heatmaps[name] = similarity
    return boxes, heatmaps


def write_box_preview(image: Image.Image, boxes: list[ReferenceBox], output_path: Path) -> None:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    colors = [(49, 130, 189), (57, 174, 88), (255, 127, 14), (214, 39, 40)]
    for index, result in enumerate(boxes):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = result.box_xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            canvas,
            f"{result.name} {result.peak_similarity:.3f}",
            (x1 + 4, max(22, y1 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write preview: {output_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.reference_dir is not None and args.reference:
        raise ValueError("Use either --reference-dir or --reference, not both")
    groups = (
        discover_reference_groups(args.reference_dir)
        if args.reference_dir is not None
        else parse_reference_arguments(args.reference)
    )
    image = load_prompt_frame(args.input, args.frame_index)
    encoder = Dinov2DenseEncoder(args.model, args.device)
    boxes, heatmaps = localize_reference_groups(
        image,
        groups,
        encoder,
        scene_side=args.scene_side,
        reference_side=args.reference_side,
        top_fraction=args.top_fraction,
        padding_fraction=args.padding_fraction,
        minimum_contrast=args.minimum_contrast,
        reference_spatial_priors=args.reference_spatial_priors,
    )
    output_path = args.output.resolve()
    preview_path = args.preview.resolve() if args.preview else output_path.with_name("dinov2_boxes.jpg")
    write_box_preview(image, boxes, preview_path)
    heatmap_path = output_path.with_name("dinov2_heatmaps.npz")
    np.savez_compressed(heatmap_path, **heatmaps)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "input": str(args.input.resolve()),
        "frame_index": args.frame_index,
        "image_size": {"width": image.width, "height": image.height},
        "model": str(args.model.resolve()),
        "targets": [asdict(box) for box in boxes],
        "artifacts": {"preview": str(preview_path), "heatmaps": str(heatmap_path)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image or video to localize")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=IMAGE")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--device", default="cuda")
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
