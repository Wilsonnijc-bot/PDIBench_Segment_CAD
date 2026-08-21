#!/usr/bin/env python3
"""Render per-object SAM3 masks from a multi-object segmentation archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


COLORS = (
    (49, 130, 189),
    (57, 174, 88),
    (255, 127, 14),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=480)
    return parser.parse_args()


def load_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_index} from {video_path}")
    return frame


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = (frame.astype(np.float32) * 0.58).astype(np.uint8)
    color_array = np.asarray(color, dtype=np.float32)
    canvas[mask] = (0.25 * frame[mask] + 0.75 * color_array).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
    return canvas


def combined_overlay(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    canvas = (frame.astype(np.float32) * 0.72).astype(np.uint8)
    for index, mask in enumerate(masks):
        color = COLORS[index % len(COLORS)]
        color_array = np.asarray(color, dtype=np.float32)
        canvas[mask] = (0.28 * frame[mask] + 0.72 * color_array).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
    return canvas


def make_tile(image: np.ndarray, label: str, color: tuple[int, int, int], width: int) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    title_height = 42
    tile = np.full((height + title_height, width, 3), 245, dtype=np.uint8)
    tile[title_height:] = resized
    cv2.rectangle(tile, (0, 0), (8, title_height - 1), color, -1)
    cv2.putText(tile, label, (22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (28, 28, 28), 2, cv2.LINE_AA)
    return tile


def main() -> None:
    args = parse_args()
    with np.load(args.segmentation, allow_pickle=False) as archive:
        masks = np.asarray(archive["object_masks"], dtype=bool)
        names = [str(value) for value in archive["object_names"].tolist()]
    if not 0 <= args.frame_index < len(masks):
        raise ValueError(f"frame index must be in [0, {len(masks) - 1}]")

    frame = load_frame(args.video, args.frame_index)
    if masks.shape[2:] != frame.shape[:2]:
        raise ValueError(f"mask shape {masks.shape[2:]} does not match frame shape {frame.shape[:2]}")

    frame_masks = masks[args.frame_index]
    tiles = []
    for index, (name, mask) in enumerate(zip(names, frame_masks)):
        color = COLORS[index % len(COLORS)]
        pixels = int(mask.sum())
        label = f"{name} | {pixels:,} px"
        tiles.append(make_tile(overlay_mask(frame, mask, color), label, color, args.tile_width))
    tiles.append(
        make_tile(
            combined_overlay(frame, frame_masks),
            f"all 7 links | frame {args.frame_index}",
            (70, 70, 70),
            args.tile_width,
        )
    )

    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    while len(tiles) < rows * columns:
        tiles.append(np.full_like(tiles[0], 245))
    sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Could not write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
