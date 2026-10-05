"""Group tracked link masks and build stable centroid-centered RGB crops."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SEGMENT_LINKS: dict[str, tuple[str, ...]] = {
    "upperarm": ("link2", "link3"),
    "forearm": ("link4", "link5"),
    "gripper": ("link7",),
}
SEGMENT_COLORS = {
    "upperarm": (49, 130, 189),
    "forearm": (57, 174, 88),
    "gripper": (214, 39, 40),
}


@dataclass(frozen=True)
class CropGeometry:
    frame_index: int
    segment: str
    source_links: tuple[str, ...]
    status: str
    nonempty_link_count: int
    mask_area: int
    mask_x1: int | None
    mask_y1: int | None
    mask_x2: int | None
    mask_y2: int | None
    centroid_x: float | None
    centroid_y: float | None
    crop_x1: int | None
    crop_y1: int | None
    crop_x2: int | None
    crop_y2: int | None
    crop_width: int | None
    crop_height: int | None


@dataclass(frozen=True)
class CropPadding:
    left: int
    top: int
    right: int
    bottom: int


def mask_geometry(
    mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[float, float]] | None:
    """Return a tight half-open box and foreground-pixel centroid."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}")
    rows, columns = np.where(mask)
    if len(columns) == 0:
        return None
    return (
        (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ),
        (float(columns.mean()), float(rows.mean())),
    )


def group_masks(
    object_masks: np.ndarray,
    object_names: tuple[str, ...],
    segment_links: dict[str, tuple[str, ...]] = SEGMENT_LINKS,
) -> dict[str, np.ndarray]:
    """Build exact logical unions while preserving individual masks upstream."""
    object_masks = np.asarray(object_masks, dtype=bool)
    if object_masks.ndim != 4:
        raise ValueError(f"object_masks must have shape (T,N,H,W), got {object_masks.shape}")
    if object_masks.shape[1] != len(object_names):
        raise ValueError("object_names must match object_masks axis 1")
    index_by_name = {name: index for index, name in enumerate(object_names)}
    if len(index_by_name) != len(object_names):
        raise ValueError("object_names must be unique")
    grouped: dict[str, np.ndarray] = {}
    for segment, links in segment_links.items():
        missing = sorted(set(links).difference(index_by_name))
        if missing:
            raise ValueError(f"Segment {segment} is missing link masks: {missing}")
        indices = [index_by_name[link] for link in links]
        grouped[segment] = np.any(object_masks[:, indices], axis=1)
    return grouped


def _interpolate(values: dict[int, float], target: int) -> float:
    frames = np.asarray(sorted(values), dtype=np.float64)
    samples = np.asarray([values[int(frame)] for frame in frames], dtype=np.float64)
    return float(np.interp(float(target), frames, samples))


def _centered_box(
    centroid: tuple[float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = int(math.floor(centroid[0] - width / 2.0))
    y1 = int(math.floor(centroid[1] - height / 2.0))
    return x1, y1, x1 + width, y1 + height


def fit_crop_box_to_image(
    box: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Keep a crop centered as closely as possible while fitting it in-frame."""
    if image_width < 1 or image_height < 1:
        raise ValueError("Image dimensions must be positive")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Crop box must have positive area: {box}")
    crop_width = min(x2 - x1, image_width)
    crop_height = min(y2 - y1, image_height)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    fitted_x1 = int(math.floor(center_x - crop_width / 2.0))
    fitted_y1 = int(math.floor(center_y - crop_height / 2.0))
    fitted_x1 = min(max(fitted_x1, 0), image_width - crop_width)
    fitted_y1 = min(max(fitted_y1, 0), image_height - crop_height)
    return (
        fitted_x1,
        fitted_y1,
        fitted_x1 + crop_width,
        fitted_y1 + crop_height,
    )


def build_crop_geometries(
    object_masks: np.ndarray,
    object_names: tuple[str, ...],
    frame_indices: list[int] | tuple[int, ...],
    *,
    context_factor: float = 1.35,
    minimum_crop_width: int = 96,
    minimum_crop_height: int = 96,
    segment_links: dict[str, tuple[str, ...]] = SEGMENT_LINKS,
    crop_size_by_segment: dict[str, tuple[int, int]] | None = None,
) -> tuple[dict[str, np.ndarray], list[CropGeometry]]:
    """Build fixed-size per-segment crop rectangles for comparison frames."""
    if context_factor < 1.0:
        raise ValueError("context_factor must be at least 1.0")
    if min(minimum_crop_width, minimum_crop_height) < 1:
        raise ValueError("minimum crop dimensions must be positive")
    object_masks = np.asarray(object_masks, dtype=bool)
    frame_indices = [int(frame) for frame in frame_indices]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("frame_indices must be unique")
    if not frame_indices:
        raise ValueError("At least one frame is required")
    if min(frame_indices) < 0 or max(frame_indices) >= object_masks.shape[0]:
        raise ValueError("A selected frame is outside the mask archive")

    grouped = group_masks(object_masks, object_names, segment_links)
    index_by_name = {name: index for index, name in enumerate(object_names)}
    records: list[CropGeometry] = []
    for segment, links in segment_links.items():
        raw: dict[int, dict] = {}
        widths: list[int] = []
        heights: list[int] = []
        valid_centers_x: dict[int, float] = {}
        valid_centers_y: dict[int, float] = {}
        for frame in frame_indices:
            per_link_nonempty = [
                bool(object_masks[frame, index_by_name[link]].any()) for link in links
            ]
            combined = grouped[segment][frame]
            measured = mask_geometry(combined)
            if measured is None:
                raw[frame] = {
                    "status": "missing",
                    "nonempty_link_count": 0,
                    "mask_area": 0,
                    "box": None,
                    "centroid": None,
                }
                continue
            box, centroid = measured
            width, height = box[2] - box[0], box[3] - box[1]
            widths.append(width)
            heights.append(height)
            valid_centers_x[frame] = centroid[0]
            valid_centers_y[frame] = centroid[1]
            nonempty_count = sum(per_link_nonempty)
            raw[frame] = {
                "status": "valid" if nonempty_count == len(links) else "partial",
                "nonempty_link_count": nonempty_count,
                "mask_area": int(combined.sum()),
                "box": box,
                "centroid": centroid,
            }

        if not widths:
            for frame in frame_indices:
                records.append(
                    CropGeometry(
                        frame_index=frame,
                        segment=segment,
                        source_links=links,
                        status="missing",
                        nonempty_link_count=0,
                        mask_area=0,
                        mask_x1=None,
                        mask_y1=None,
                        mask_x2=None,
                        mask_y2=None,
                        centroid_x=None,
                        centroid_y=None,
                        crop_x1=None,
                        crop_y1=None,
                        crop_x2=None,
                        crop_y2=None,
                        crop_width=None,
                        crop_height=None,
                    )
                )
            continue

        if crop_size_by_segment is None:
            crop_width = max(
                minimum_crop_width, int(math.ceil(max(widths) * context_factor))
            )
            crop_height = max(
                minimum_crop_height, int(math.ceil(max(heights) * context_factor))
            )
        else:
            try:
                crop_width, crop_height = crop_size_by_segment[segment]
            except KeyError as exc:
                raise ValueError(f"No fixed crop size supplied for {segment}") from exc
            if min(crop_width, crop_height) < 1:
                raise ValueError(f"Invalid fixed crop size for {segment}")
        for frame in frame_indices:
            item = raw[frame]
            status = item["status"]
            centroid = item["centroid"]
            if centroid is None:
                centroid = (
                    _interpolate(valid_centers_x, frame),
                    _interpolate(valid_centers_y, frame),
                )
                status = "fallback"
            crop = _centered_box(centroid, crop_width, crop_height)
            box = item["box"]
            records.append(
                CropGeometry(
                    frame_index=frame,
                    segment=segment,
                    source_links=links,
                    status=status,
                    nonempty_link_count=int(item["nonempty_link_count"]),
                    mask_area=int(item["mask_area"]),
                    mask_x1=None if box is None else box[0],
                    mask_y1=None if box is None else box[1],
                    mask_x2=None if box is None else box[2],
                    mask_y2=None if box is None else box[3],
                    centroid_x=centroid[0],
                    centroid_y=centroid[1],
                    crop_x1=crop[0],
                    crop_y1=crop[1],
                    crop_x2=crop[2],
                    crop_y2=crop[3],
                    crop_width=crop_width,
                    crop_height=crop_height,
                )
            )
    return grouped, records


def extract_centered_crop(
    image: np.ndarray, geometry: CropGeometry
) -> tuple[np.ndarray, CropPadding]:
    """Extract a centered rectangle with explicit solid-black edge padding."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 image, got {image.shape}")
    if geometry.crop_x1 is None or geometry.crop_width is None:
        raise ValueError("Cannot crop missing geometry")
    x1, y1 = geometry.crop_x1, geometry.crop_y1
    x2, y2 = geometry.crop_x2, geometry.crop_y2
    assert x2 is not None and y2 is not None
    height, width = image.shape[:2]
    clipped_x1, clipped_y1 = max(0, x1), max(0, y1)
    clipped_x2, clipped_y2 = min(width, x2), min(height, y2)
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        raise ValueError("Crop rectangle does not intersect the image")
    crop = image[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
    padding = CropPadding(
        left=max(0, -x1),
        top=max(0, -y1),
        right=max(0, x2 - width),
        bottom=max(0, y2 - height),
    )
    crop = cv2.copyMakeBorder(
        crop,
        padding.top,
        padding.bottom,
        padding.left,
        padding.right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    expected = (geometry.crop_height, geometry.crop_width)
    if crop.shape[:2] != expected:
        raise RuntimeError(f"Crop shape {crop.shape[:2]} does not match {expected}")
    return crop, padding


def black_padding_fraction(
    geometry: CropGeometry, image_width: int, image_height: int
) -> float:
    """Return the fraction of a centered crop that lies outside the image."""
    if image_width < 1 or image_height < 1:
        raise ValueError("Image dimensions must be positive")
    if geometry.crop_x1 is None or geometry.crop_width is None:
        raise ValueError("Cannot measure padding for missing geometry")
    x1, y1 = geometry.crop_x1, geometry.crop_y1
    x2, y2 = geometry.crop_x2, geometry.crop_y2
    assert x2 is not None and y1 is not None and y2 is not None
    visible_width = max(0, min(image_width, x2) - max(0, x1))
    visible_height = max(0, min(image_height, y2) - max(0, y1))
    visible_area = visible_width * visible_height
    crop_area = geometry.crop_width * geometry.crop_height
    return 1.0 - (visible_area / crop_area)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_rgb(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != expected_hw:
        raise ValueError(
            f"Frame {path} has shape {rgb.shape[:2]}, expected {expected_hw}"
        )
    return rgb


def write_qa_overlay(
    image: np.ndarray,
    frame_index: int,
    geometries: list[CropGeometry],
    grouped_masks: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for geometry in geometries:
        color = SEGMENT_COLORS[geometry.segment]
        mask = grouped_masks[geometry.segment][frame_index]
        if mask.any():
            overlay = np.asarray(color, dtype=np.uint8)
            canvas[mask] = (0.55 * canvas[mask] + 0.45 * overlay).astype(np.uint8)
        if geometry.crop_x1 is not None:
            height, width = image.shape[:2]
            x1 = min(max(geometry.crop_x1, 0), width - 1)
            y1 = min(max(geometry.crop_y1, 0), height - 1)
            x2 = min(max(geometry.crop_x2 - 1, 0), width - 1)
            y2 = min(max(geometry.crop_y2 - 1, 0), height - 1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"{geometry.segment}:{geometry.status}",
                (x1 + 3, max(18, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write QA overlay: {output_path}")


def geometry_to_row(geometry: CropGeometry) -> dict:
    row = asdict(geometry)
    row["source_links"] = "+".join(geometry.source_links)
    return row
