#!/usr/bin/env python3
"""Package the two requested replays and one concise batch metrics CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any


DELIVERABLE_FILES = {
    "point_cloud_replay.mp4": "output/replay/combined_exact-group.mp4",
    "cotracker_cad_replay.mp4": "output/replay/cad/cotracker_cad_replay.mp4",
}

LINK_NAMES = tuple(f"link{index}" for index in range(2, 8))
CONCISE_METRIC_FIELDS = [
    "dataset",
    "video",
    "pipeline_status",
    *(f"{link}_pdi_score" for link in LINK_NAMES),
    *(f"{link}_cad_epsilon_mean" for link in LINK_NAMES),
    *(f"{link}_cad_epsilon_p90" for link in LINK_NAMES),
    *(f"{link}_pose_discontinuity" for link in LINK_NAMES),
    "cad_statuses",
    "foundation_pose_video_depth_scale",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_hard_link(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"deliverable source is missing or empty: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and source.stat().st_ino == target.stat().st_ino:
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.link(source, temporary)
    temporary.replace(target)


def deliverable_dir(output_root: Path, entry: dict[str, Any]) -> Path:
    relative = Path(entry["relative_path"])
    parts = (entry["dataset"], *relative.with_suffix("").parts)
    folder_name = "_".join(str(part).replace(" ", "_") for part in parts)
    return output_root / folder_name


def concise_metric_row(batch_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    job_root = batch_root / "jobs" / entry["job_id"]
    status_path = job_root / "status.json"
    metrics_path = job_root / "output/metrics.json"
    status = read_json(status_path) if status_path.is_file() else {}
    metrics = read_json(metrics_path)
    objects = metrics.get("modes", {}).get("exact-group", {}).get("objects", {})
    row: dict[str, Any] = {
        "dataset": entry["dataset"],
        "video": entry["relative_path"],
        "pipeline_status": status.get("state", ""),
        "foundation_pose_video_depth_scale": metrics.get(
            "cad_canonicalization", {}
        ).get("video_depth_scale"),
    }
    cad_statuses = []
    for link in LINK_NAMES:
        report = objects.get(link, {})
        cad = report.get("cad_rigidity") or {}
        pose = report.get("pose_discontinuity") or {}
        row.update(
            {
                f"{link}_pdi_score": report.get("pdi_score"),
                f"{link}_cad_epsilon_mean": cad.get("epsilon_cad_mean"),
                f"{link}_cad_epsilon_p90": cad.get("epsilon_cad_p90"),
                f"{link}_pose_discontinuity": pose.get("pose_discontinuity"),
            }
        )
        cad_statuses.append(f"{link}:{cad.get('status', 'missing')}")
    row["cad_statuses"] = ";".join(cad_statuses)
    return row


def write_concise_metrics_csv(
    batch_root: Path,
    output: Path,
    entries: list[dict[str, Any]],
) -> None:
    rows = [concise_metric_row(batch_root, entry) for entry in entries]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CONCISE_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def package_entry(batch_root: Path, output_root: Path, entry: dict[str, Any]) -> bool:
    if not entry.get("replay_selected", False):
        return False
    job_root = batch_root / "jobs" / entry["job_id"]
    sources = {
        output_name: job_root / relative_source
        for output_name, relative_source in DELIVERABLE_FILES.items()
    }
    if not all(source.is_file() and source.stat().st_size > 0 for source in sources.values()):
        return False

    target_root = deliverable_dir(output_root, entry)
    target_root.mkdir(parents=True, exist_ok=True)
    for path in target_root.iterdir():
        if path.name not in sources and (path.is_file() or path.is_symlink()):
            path.unlink()
    for output_name, source in sources.items():
        replace_hard_link(source, target_root / output_name)
    return True


def package_available(manifest_path: Path, batch_root: Path, output_root: Path) -> dict[str, int]:
    entries = read_json(manifest_path)["videos"]
    selected = [entry for entry in entries if entry.get("replay_selected", False)]
    packaged_entries = [
        entry for entry in selected if package_entry(batch_root, output_root, entry)
    ]
    write_concise_metrics_csv(
        batch_root,
        output_root / "metrics.csv",
        packaged_entries,
    )
    packaged = len(packaged_entries)
    return {
        "selected": len(selected),
        "packaged": packaged,
        "pending": len(selected) - packaged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")

    while True:
        result = package_available(
            args.manifest.resolve(), args.batch_root.resolve(), args.output_root.resolve()
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if not args.watch or result["pending"] == 0:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
