#!/usr/bin/env python3
"""Export one row per video with exact-group PDI metrics for links 2-7."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LINK_NAMES = tuple(f"link{index}" for index in range(2, 8))
METRIC_FIELDS = (
    "status",
    "error",
    "depth_strategy",
    "depth_valid_frame_count",
    "depth_interpolated_frame_count",
    "depth_interpolated_frame_fraction",
    "depth_total_frame_count",
    "pdi_score",
    "grade",
    "scale_component",
    "trajectory_component",
    "rigidity_component",
    "vanishing_point_component",
    "sam3_tracked_fraction",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_columns() -> list[str]:
    return [f"{link}_{field}" for link in LINK_NAMES for field in METRIC_FIELDS]


def export_batch_csv(manifest_path: Path, batch_root: Path, output: Path) -> None:
    manifest = read_json(manifest_path)
    fields = [
        "dataset",
        "relative_path",
        "video_sha256",
        "status",
        "error",
        "replay_selected",
        "replay_status",
        "replay_video",
        *metric_columns(),
        "geometry_seconds",
        "query_preparation_seconds",
        "tracker_load_seconds",
        "exact_group_model_seconds",
        "exact_group_total_tracking_seconds",
        "peak_gpu_memory_bytes",
        "total_seconds",
    ]
    rows = []
    for entry in manifest["videos"]:
        job_root = batch_root / "jobs" / entry["job_id"]
        status_path = job_root / "status.json"
        metrics_path = job_root / "output" / "metrics.json"
        diagnostics_path = job_root / "intermediate" / "sam3_prompt_diagnostics.json"
        status = read_json(status_path) if status_path.is_file() else {"state": "pending"}
        replay_selected = bool(entry.get("replay_selected", False))
        replay_path = job_root / "output/replay/combined_exact-group.mp4"
        row: dict[str, Any] = {
            "dataset": entry["dataset"],
            "relative_path": entry["relative_path"],
            "video_sha256": entry["sha256"],
            "status": status.get("state", "pending"),
            "error": status.get("error", ""),
            "replay_selected": replay_selected,
            "replay_status": (
                "complete"
                if replay_selected and replay_path.is_file() and replay_path.stat().st_size > 0
                else "pending"
                if replay_selected
                else "not_selected"
            ),
            "replay_video": (
                replay_path.relative_to(batch_root).as_posix()
                if replay_selected and replay_path.is_file()
                else ""
            ),
        }
        tracked = {}
        if diagnostics_path.is_file():
            for diagnostic in read_json(diagnostics_path):
                tracked[diagnostic["target"]] = diagnostic.get("tracking", {}).get(
                    "tracked_fraction"
                )
        if metrics_path.is_file():
            metrics = read_json(metrics_path)
            mode = metrics.get("modes", {}).get("exact-group", {})
            objects = mode.get("objects", {})
            for link in LINK_NAMES:
                report = objects.get(link)
                if report is None:
                    continue
                row[f"{link}_status"] = report.get("status", "complete")
                row[f"{link}_error"] = report.get("error", "")
                depth = report.get("depth", {})
                row.update(
                    {
                        f"{link}_depth_strategy": depth.get("strategy", ""),
                        f"{link}_depth_valid_frame_count": depth.get(
                            "valid_frame_count"
                        ),
                        f"{link}_depth_interpolated_frame_count": depth.get(
                            "interpolated_frame_count"
                        ),
                        f"{link}_depth_interpolated_frame_fraction": depth.get(
                            "interpolated_frame_fraction"
                        ),
                        f"{link}_depth_total_frame_count": depth.get(
                            "total_frame_count"
                        ),
                    }
                )
                if report.get("status") == "failed":
                    continue
                breakdown = report["breakdown"]
                row.update(
                    {
                        f"{link}_pdi_score": report["pdi_score"],
                        f"{link}_grade": report["grade"],
                        f"{link}_scale_component": breakdown["scale_component"],
                        f"{link}_trajectory_component": breakdown["traj_component"],
                        f"{link}_rigidity_component": breakdown["epsilon_rigidity"],
                        f"{link}_vanishing_point_component": breakdown["vp_component"],
                        f"{link}_sam3_tracked_fraction": tracked.get(link),
                    }
                )
            shared = metrics.get("timing", {})
            tracking = mode.get("timing", {}).get("tracking", {})
            row.update(
                {
                    "geometry_seconds": shared.get("geometry_seconds"),
                    "query_preparation_seconds": shared.get("query_preparation_seconds"),
                    "tracker_load_seconds": shared.get("tracker_load_seconds"),
                    "exact_group_model_seconds": tracking.get("model_seconds"),
                    "exact_group_total_tracking_seconds": tracking.get(
                        "total_tracking_seconds"
                    ),
                    "peak_gpu_memory_bytes": tracking.get("peak_gpu_memory_bytes"),
                    "total_seconds": shared.get("total_seconds"),
                }
            )
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_batch_csv(args.manifest.resolve(), args.batch_root.resolve(), args.output.resolve())
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
