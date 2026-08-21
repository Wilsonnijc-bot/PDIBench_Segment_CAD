#!/usr/bin/env python3
"""Compare edited joint/exact results with a frozen-input original baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdi_eval.verification import (  # noqa: E402
    compare_metrics,
    compare_track_groups,
    load_track_group,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def original_tracks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "tracks": np.asarray(archive["foreground_tracks"]),
            "visibility": np.asarray(archive["foreground_visibility"]),
            "queries": np.asarray(archive["foreground_queries"]),
        }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_metrics_csv(
    path: Path,
    video: str,
    object_name: str,
    original_prompt: str,
    edited: dict,
    original: dict,
    edited_timing: dict,
    original_timing: dict,
    original_manifest: dict,
) -> None:
    fields = [
        "video",
        "method",
        "pipeline",
        "object_name",
        "sam3_prompt",
        "segmentation_source",
        "pdi_score",
        "grade",
        "scale_component",
        "trajectory_component",
        "rigidity_component",
        "vanishing_point_component",
        "foreground_tracks",
        "background_tracks",
        "model_forward_count",
        "model_seconds",
        "total_tracking_seconds",
        "wall_seconds",
        "peak_gpu_memory_bytes",
    ]
    rows = []
    for mode in ("joint-query", "exact-group"):
        report = edited["modes"][mode]["objects"][object_name]
        timing = edited_timing["modes"][mode]["tracking"]
        breakdown = report["breakdown"]
        rows.append(
            {
                "video": video,
                "method": mode,
                "pipeline": "new-cotracker",
                "object_name": object_name,
                "sam3_prompt": "entire white quadrangular robot gripper",
                "segmentation_source": "SAM3 link7 adapter (DINOv2 box)",
                "pdi_score": report["pdi_score"],
                "grade": report["grade"],
                "scale_component": breakdown["scale_component"],
                "trajectory_component": breakdown["traj_component"],
                "rigidity_component": breakdown["epsilon_rigidity"],
                "vanishing_point_component": breakdown["vp_component"],
                "foreground_tracks": timing["foreground_query_counts"][0],
                "background_tracks": timing["background_query_count"],
                "model_forward_count": timing["model_forward_count"],
                "model_seconds": timing["model_seconds"],
                "total_tracking_seconds": timing["total_tracking_seconds"],
                "wall_seconds": "",
                "peak_gpu_memory_bytes": timing["peak_gpu_memory_bytes"],
            }
        )
    original_breakdown = original["breakdown"]
    rows.append(
        {
            "video": video,
            "method": "original",
            "pipeline": "PDI-Bench-original",
            "object_name": object_name,
            "sam3_prompt": original_prompt,
            "segmentation_source": original_manifest["segmentation_source"],
            "pdi_score": original["pdi_score"],
            "grade": original["grade"],
            "scale_component": original_breakdown["scale_component"],
            "trajectory_component": original_breakdown["traj_component"],
            "rigidity_component": original_breakdown["epsilon_rigidity"],
            "vanishing_point_component": original_breakdown["vp_component"],
            "foreground_tracks": original_manifest["foreground_query_count"],
            "background_tracks": original_manifest["background_query_count"],
            "model_forward_count": 1,
            "model_seconds": original_timing["model_seconds"],
            "total_tracking_seconds": "",
            "wall_seconds": original_timing["wall_seconds"],
            "peak_gpu_memory_bytes": original_timing["peak_gpu_memory_bytes"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited-run", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--object-name", default="link7")
    parser.add_argument("--original-prompt", default="gripper")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    edited_run = args.edited_run.resolve()
    original_run = args.original_run.resolve()
    edited = read_json(edited_run / "metrics.json")
    original_metrics = read_json(original_run / "metrics.json")
    edited_timing = read_json(edited_run / "timing.json")
    original_timing = read_json(original_run / "timing.json")
    original_manifest = read_json(original_run / "manifest.json")
    original_track_group = original_tracks(
        original_run / "cotracker_original.npz"
    )

    mode_metrics = {
        mode: edited["modes"][mode]["objects"][args.object_name]
        for mode in ("joint-query", "exact-group")
    }
    mode_tracks = {
        mode: load_track_group(
            edited_run / f"cotracker_{mode}.npz", args.object_name
        )
        for mode in ("joint-query", "exact-group")
    }
    comparison = {
        "schema_version": 1,
        "object_name": args.object_name,
        "methods": {
            "joint-query": (
                "One CoTracker update containing link7 foreground and shared "
                "background queries."
            ),
            "exact-group": (
                "Separate link7 and background CoTracker updates with one "
                "replayed video-backbone feature pass."
            ),
            "original": (
                "Pristine PDI-Bench-original tracking and metrics using the "
                "SAM3 link7 adapter mask from the requested original prompt "
                "and shared MegaSAM geometry."
            ),
        },
        "original_vs_joint": {
            "metrics": compare_metrics(original_metrics, mode_metrics["joint-query"]),
            "tracking": compare_track_groups(
                original_track_group, mode_tracks["joint-query"]
            ),
        },
        "original_vs_exact": {
            "metrics": compare_metrics(original_metrics, mode_metrics["exact-group"]),
            "tracking": compare_track_groups(
                original_track_group, mode_tracks["exact-group"]
            ),
        },
        "joint_vs_exact": {
            "metrics": compare_metrics(
                mode_metrics["joint-query"], mode_metrics["exact-group"]
            ),
            "tracking": compare_track_groups(
                mode_tracks["joint-query"], mode_tracks["exact-group"]
            ),
        },
        "timing": {
            "edited": edited_timing,
            "original": original_timing,
        },
        "edited_internal_comparison": edited.get("comparison"),
    }
    write_json(args.output, comparison)
    csv_output = (args.csv_output or args.output.with_name("metrics.csv")).resolve()
    write_metrics_csv(
        csv_output,
        Path(original_manifest["video"]).name,
        args.object_name,
        args.original_prompt,
        edited,
        original_metrics,
        edited_timing,
        original_timing,
        original_manifest,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "csv_output": str(csv_output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
