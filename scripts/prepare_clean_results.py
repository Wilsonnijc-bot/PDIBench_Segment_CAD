#!/usr/bin/env python3
"""Build a concise, visual-only delivery tree from fetched batch artifacts."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from pathlib import Path


LINKS = tuple(f"link{index}" for index in range(2, 8))
GRADE_PATTERN = re.compile(r"^([A-Z])(?:\s|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--raw-outputs", type=Path, required=True)
    parser.add_argument("--flat-replays", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def video_name(row: dict[str, str]) -> str:
    dataset = row["dataset"].replace("/", "_").replace(" ", "_")
    stem = Path(row["relative_path"]).stem
    return f"{dataset}_{stem}"


def grade_letter(value: str, status: str) -> str:
    if status != "complete":
        return "FAILED"
    match = GRADE_PATTERN.match(value.strip())
    return match.group(1) if match else value.strip()


def concise_row(row: dict[str, str]) -> dict[str, str]:
    result = {
        "video": video_name(row),
        "dataset": row["dataset"],
        "status": row["status"],
        "failed_links": ";".join(
            link for link in LINKS if row.get(f"{link}_status") != "complete"
        ),
    }
    link_errors = []
    for link in LINKS:
        status = row.get(f"{link}_status", "")
        result[f"{link}_pdi"] = row.get(f"{link}_pdi_score", "")
        result[f"{link}_grade"] = grade_letter(row.get(f"{link}_grade", ""), status)
        result[f"{link}_interpolated_frames"] = row.get(
            f"{link}_depth_interpolated_frame_count", ""
        )
        error = row.get(f"{link}_error", "").strip()
        if error:
            link_errors.append(f"{link}: {error}")
    total_seconds = row.get("total_seconds", "").strip()
    result["total_seconds"] = f"{float(total_seconds):.1f}" if total_seconds else ""
    errors = [row.get("error", "").strip(), *link_errors]
    result["failure_reason"] = " | ".join(error for error in errors if error).replace(
        "\n", " "
    )
    return result


def find_raw_job(raw_outputs: Path, video_sha256: str) -> Path | None:
    matches = sorted(raw_outputs.glob(f"*-{video_sha256[:12]}"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Multiple raw jobs match hash {video_sha256[:12]}: {matches}")
    return matches[0]


def link_or_copy(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def copy_visuals(
    row: dict[str, str],
    raw_outputs: Path,
    flat_replays: Path,
    destination: Path,
) -> dict[str, int]:
    name = video_name(row)
    video_dir = destination / name
    video_dir.mkdir(parents=True, exist_ok=True)
    counts = {"png": 0, "mp4": 0}
    raw_job = find_raw_job(raw_outputs, row["video_sha256"])
    if raw_job is not None:
        raw_output = raw_job / "output"
        png_sources = (
            (raw_output / "first_frame_mask.png", video_dir / "mask.png"),
            (
                raw_output / "replay" / "combined_exact-group_first_frame.png",
                video_dir / "replay.png",
            ),
            (raw_output / "replay_2d_first_frame.png", video_dir / "tracking_2d.png"),
        )
        for source, target in png_sources:
            counts["png"] += int(link_or_copy(source, target))

    replay_sources = (
        (flat_replays / f"{name}.mp4", video_dir / "replay.mp4"),
        (flat_replays / f"{name}_2d.mp4", video_dir / "tracking_2d.mp4"),
    )
    for source, target in replay_sources:
        counts["mp4"] += int(link_or_copy(source, target))
    return counts


def main() -> None:
    args = parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise RuntimeError(f"Destination must be empty: {args.destination}")
    args.destination.mkdir(parents=True, exist_ok=True)

    with args.source_metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metric rows found in {args.source_metrics}")

    fieldnames = ["video", "dataset", "status", "failed_links"]
    for link in LINKS:
        fieldnames.extend(
            [f"{link}_pdi", f"{link}_grade", f"{link}_interpolated_frames"]
        )
    fieldnames.extend(["total_seconds", "failure_reason"])

    visual_counts = {"png": 0, "mp4": 0}
    concise_rows = []
    seen_names = set()
    for row in rows:
        name = video_name(row)
        if name in seen_names:
            raise RuntimeError(f"Duplicate clean video name: {name}")
        seen_names.add(name)
        concise_rows.append(concise_row(row))
        counts = copy_visuals(
            row,
            args.raw_outputs,
            args.flat_replays,
            args.destination,
        )
        for extension, count in counts.items():
            visual_counts[extension] += count

    metrics_path = args.destination / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(concise_rows)

    print(
        f"Prepared {len(rows)} video folders, {visual_counts['png']} PNGs, "
        f"{visual_counts['mp4']} MP4s, and {metrics_path}"
    )


if __name__ == "__main__":
    main()
