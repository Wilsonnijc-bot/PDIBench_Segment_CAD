#!/usr/bin/env python3
"""Render SAM3 masks and grouped CoTracker points over a source video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


# OpenCV uses BGR colors.
COLORS = (
    (49, 130, 189),
    (57, 174, 88),
    (255, 127, 14),
    (214, 39, 40),
    (148, 103, 189),
    (227, 119, 194),
)
BACKGROUND_COLOR = (175, 175, 175)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Original source MP4")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Fetched job directory, or its output directory",
    )
    parser.add_argument("--output", type=Path, required=True, help="Rendered MP4")
    parser.add_argument(
        "--first-frame",
        type=Path,
        help="Optional path for a PNG of the first rendered frame",
    )
    parser.add_argument("--fps", type=float, help="Override the source frame rate")
    parser.add_argument("--mask-alpha", type=float, default=0.30)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--show-background", action="store_true")
    return parser.parse_args()


def artifact_dir(run_dir: Path) -> Path:
    direct = run_dir / "segmentation.npz"
    nested = run_dir / "output" / "segmentation.npz"
    if direct.is_file():
        return run_dir
    if nested.is_file():
        return run_dir / "output"
    raise FileNotFoundError(f"Cannot find segmentation.npz under {run_dir}")


def load_artifacts(run_dir: Path) -> dict[str, np.ndarray | list[str]]:
    root = artifact_dir(run_dir)
    with np.load(root / "segmentation.npz", allow_pickle=False) as archive:
        masks = np.asarray(archive["object_masks"], dtype=bool)
        mask_names = [str(value) for value in archive["object_names"].tolist()]
    with np.load(root / "cotracker_exact-group.npz", allow_pickle=False) as archive:
        track_names = [str(value) for value in archive["object_names"].tolist()]
        data: dict[str, np.ndarray | list[str]] = {
            "masks": masks,
            "names": mask_names,
            "offsets": np.asarray(archive["object_offsets"], dtype=np.int64),
            "tracks": np.asarray(archive["tracks"], dtype=np.float32),
            "visibility": np.asarray(archive["visibility"], dtype=bool),
            "background_tracks": np.asarray(archive["background_tracks"], dtype=np.float32),
            "background_visibility": np.asarray(archive["background_visibility"], dtype=bool),
        }
    if mask_names != track_names:
        raise ValueError(f"Segmentation objects {mask_names} do not match tracks {track_names}")
    if masks.ndim != 4 or masks.shape[1] != len(mask_names):
        raise ValueError(f"Unexpected object_masks shape: {masks.shape}")
    offsets = data["offsets"]
    tracks = data["tracks"]
    visibility = data["visibility"]
    if not isinstance(offsets, np.ndarray) or len(offsets) != len(mask_names) + 1:
        raise ValueError("object_offsets does not match the object count")
    if not isinstance(tracks, np.ndarray) or not isinstance(visibility, np.ndarray):
        raise TypeError("Invalid track archive")
    if tracks.shape[:2] != visibility.shape or tracks.shape[2:] != (2,):
        raise ValueError("tracks and visibility have incompatible shapes")
    if int(offsets[-1]) != tracks.shape[1]:
        raise ValueError("object_offsets does not span all foreground tracks")
    return data


def overlay_masks(
    frame: np.ndarray,
    masks: np.ndarray,
    alpha: float,
) -> np.ndarray:
    canvas = frame.copy()
    for index, mask in enumerate(masks):
        color = COLORS[index % len(COLORS)]
        color_array = np.asarray(color, dtype=np.float32)
        canvas[mask] = (
            (1.0 - alpha) * canvas[mask].astype(np.float32) + alpha * color_array
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 1, cv2.LINE_AA)
    return canvas


def draw_points(
    canvas: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> int:
    height, width = canvas.shape[:2]
    count = 0
    for x_float, y_float in points[visible]:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), radius + 1, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
            count += 1
    return count


def add_legend(
    frame: np.ndarray,
    names: list[str],
    visible_counts: list[int],
    background_count: int | None,
    frame_index: int,
    frame_count: int,
) -> np.ndarray:
    panel_width = 190
    panel = np.full((frame.shape[0], panel_width, 3), 24, dtype=np.uint8)
    cv2.putText(
        panel,
        "SAM3 + CoTracker",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"frame {frame_index + 1}/{frame_count}",
        (14, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    for index, (name, count) in enumerate(zip(names, visible_counts)):
        y = 88 + index * 39
        color = COLORS[index % len(COLORS)]
        cv2.rectangle(panel, (14, y - 10), (27, y + 3), color, -1)
        cv2.putText(
            panel,
            f"{name}  {count} pts",
            (38, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    if background_count is not None:
        y = 88 + len(names) * 39
        cv2.circle(panel, (21, y - 3), 4, BACKGROUND_COLOR, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"background  {background_count}",
            (38, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
    return np.hstack((frame, panel))


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Cannot open video writer for {path}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1")
    if args.point_radius < 1:
        raise ValueError("--point-radius must be at least 1")

    data = load_artifacts(args.run_dir)
    masks = data["masks"]
    names = data["names"]
    offsets = data["offsets"]
    tracks = data["tracks"]
    visibility = data["visibility"]
    background_tracks = data["background_tracks"]
    background_visibility = data["background_visibility"]
    assert isinstance(masks, np.ndarray)
    assert isinstance(names, list)
    assert isinstance(offsets, np.ndarray)
    assert isinstance(tracks, np.ndarray)
    assert isinstance(visibility, np.ndarray)
    assert isinstance(background_tracks, np.ndarray)
    assert isinstance(background_visibility, np.ndarray)

    frame_count = min(masks.shape[0], tracks.shape[0], background_tracks.shape[0])
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = args.fps if args.fps is not None else source_fps
    if fps <= 0:
        fps = 16.0
    if masks.shape[2:] != (source_height, source_width):
        capture.release()
        raise ValueError(
            f"Mask shape {masks.shape[2:]} does not match video "
            f"shape {(source_height, source_width)}"
        )

    writer = open_writer(args.output, fps, (source_width + 190, source_height))
    rendered = 0
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            canvas = overlay_masks(frame, masks[frame_index], args.mask_alpha)
            visible_counts = []
            for object_index in range(len(names)):
                start, end = offsets[object_index : object_index + 2]
                count = draw_points(
                    canvas,
                    tracks[frame_index, start:end],
                    visibility[frame_index, start:end],
                    COLORS[object_index % len(COLORS)],
                    args.point_radius,
                )
                visible_counts.append(count)
            background_count = None
            if args.show_background:
                background_count = draw_points(
                    canvas,
                    background_tracks[frame_index],
                    background_visibility[frame_index],
                    BACKGROUND_COLOR,
                    1,
                )
            rendered_frame = add_legend(
                canvas,
                names,
                visible_counts,
                background_count,
                frame_index,
                frame_count,
            )
            writer.write(rendered_frame)
            if frame_index == 0 and args.first_frame is not None:
                args.first_frame.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.first_frame), rendered_frame):
                    raise RuntimeError(f"Cannot write {args.first_frame}")
            rendered += 1
    finally:
        capture.release()
        writer.release()

    if rendered != frame_count:
        raise RuntimeError(f"Rendered {rendered} of {frame_count} expected frames")
    print(f"Rendered {rendered} frames at {fps:g} FPS: {args.output}")


if __name__ == "__main__":
    main()
