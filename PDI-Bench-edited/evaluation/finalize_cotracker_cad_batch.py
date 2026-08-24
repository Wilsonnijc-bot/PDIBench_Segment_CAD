#!/usr/bin/env python3
"""Wait for CoTracker-CAD rerendering, encode H.264, and republish outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from evaluation.package_batch_deliverables import (
    package_entry,
    write_concise_metrics_csv,
)
from pdi_eval.utils.cad_replay import _encode_h264


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def video_codec(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def transcode_in_place(path: Path) -> bool:
    if video_codec(path) == "h264":
        return False
    temporary = path.with_name(f".{path.stem}.h264.tmp.mp4")
    if temporary.exists():
        temporary.unlink()
    _encode_h264(path, temporary)
    temporary.replace(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rerender-status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")

    manifest = args.manifest.resolve()
    batch_root = args.batch_root.resolve()
    output_root = args.output_root.resolve()
    rerender_status = args.rerender_status.resolve()
    status_path = batch_root / "cotracker-cad-finalize-status.json"
    while True:
        status = read_json(rerender_status) if rerender_status.is_file() else {}
        if status.get("state") != "running":
            break
        write_json_atomic(
            status_path,
            {
                "state": "waiting",
                "rerender_complete": status.get("complete", 0),
                "rerender_total": status.get("total", 0),
                "rerender_failed": status.get("failed", 0),
            },
        )
        time.sleep(args.poll_seconds)
    if status.get("state") != "complete" or status.get("failed") != 0:
        raise RuntimeError(f"rerender did not complete cleanly: {status}")

    completed_job_ids = {
        result["job_id"]
        for result in status.get("results", [])
        if result.get("status") == "complete"
    }
    entries = [
        entry
        for entry in read_json(manifest)["videos"]
        if entry["job_id"] in completed_job_ids
    ]
    if len(entries) != len(completed_job_ids):
        raise RuntimeError("rerender status contains jobs absent from the manifest")
    transcoded = []
    for entry in entries:
        replay = (
            batch_root
            / "jobs"
            / entry["job_id"]
            / "output/replay/cad/cotracker_cad_replay.mp4"
        )
        if not replay.is_file() or replay.stat().st_size <= 0:
            raise FileNotFoundError(f"CoTracker-CAD replay is missing: {replay}")
        if transcode_in_place(replay):
            transcoded.append(entry["job_id"])

    packaged_entries = [
        entry for entry in entries if package_entry(batch_root, output_root, entry)
    ]
    write_concise_metrics_csv(
        batch_root,
        output_root / "metrics.csv",
        packaged_entries,
    )
    packaged = {
        "selected": len(entries),
        "packaged": len(packaged_entries),
        "pending": len(entries) - len(packaged_entries),
    }
    final = {
        "state": "complete",
        "selected": len(entries),
        "transcoded": len(transcoded),
        "transcoded_jobs": transcoded,
        "packaging": packaged,
    }
    write_json_atomic(status_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
