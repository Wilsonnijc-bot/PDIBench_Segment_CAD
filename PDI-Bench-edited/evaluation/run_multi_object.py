#!/usr/bin/env python3
"""Run native shared-geometry PDI for every object in a SAM3 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT / "src"))

from pdi_eval.multi_object_pipeline import (  # noqa: E402
    MultiObjectPDIEvaluationPipeline,
    write_report,
)
from pdi_eval.perception.track_wrapper import TRACKING_MODES  # noqa: E402
from pdi_eval.utils.reconstruct_replay import main as render_replay  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted((root / "src/pdi_eval").rglob("*.py"))
    sources.extend(
        [root / "evaluation/run_multi_object.py", root / "configs/default.yaml"]
    )
    for source in sources:
        relative = source.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(_sha256(source).encode())
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BENCHMARK_ROOT / "configs/default.yaml")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--segmentation-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path)
    parser.add_argument(
        "--tracking-mode",
        choices=(*TRACKING_MODES, "both"),
        default="both",
        help="Run one CoTracker mode or produce a direct two-mode comparison",
    )
    parser.add_argument(
        "--disable-replay",
        action="store_true",
        help="Skip replay rendering for metric-only batch runs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    video = args.input.resolve()
    segmentation = args.segmentation_npz.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.geometry_cache_dir.resolve()
    config_path = args.config.resolve()
    for path, label in (
        (video, "input video"),
        (segmentation, "segmentation archive"),
        (config_path, "config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.disable_replay:
        config.setdefault("multi_object_replay", {})["enabled"] = False
    if args.tracker_checkpoint is not None:
        checkpoint = args.tracker_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"tracker checkpoint is missing: {checkpoint}")
        config["tracker_ckpt"] = str(checkpoint)
    modes = TRACKING_MODES if args.tracking_mode == "both" else (args.tracking_mode,)
    report = MultiObjectPDIEvaluationPipeline(config).run(
        video_path=str(video),
        segmentation_npz=str(segmentation),
        tracking_modes=modes,
        output_dir=output_dir,
        geometry_cache_dir=cache_dir,
    )
    segmentation_metadata_path = segmentation.with_suffix(".json")
    if segmentation_metadata_path.is_file():
        segmentation_metadata = json.loads(
            segmentation_metadata_path.read_text(encoding="utf-8")
        )
        report["timing"]["sam3_seconds"] = segmentation_metadata.get(
            "duration_seconds"
        )
        metadata_output = output_dir / "segmentation.json"
        if segmentation_metadata_path != metadata_output:
            shutil.copy2(segmentation_metadata_path, metadata_output)
    segmentation_output = output_dir / "segmentation.npz"
    if segmentation != segmentation_output:
        shutil.copy2(segmentation, segmentation_output)
    segmentation_preview = segmentation.with_name("first_frame_mask.png")
    if segmentation_preview.is_file():
        shutil.copy2(segmentation_preview, output_dir / "first_frame_mask.png")
    metrics_path = output_dir / "metrics.json"
    write_report(metrics_path, report)
    write_report(
        output_dir / "timing.json",
        {
            "shared": report["timing"],
            "modes": {
                mode: values["timing"] for mode, values in report["modes"].items()
            },
            "speed_comparison": (
                report["comparison"]["speed"] if report["comparison"] else None
            ),
            "status": "metrics_complete_replay_pending",
        },
    )
    shutil.copy2(config_path, output_dir / "run_config.yaml")
    replay_artifacts = {}
    replay_config = config.get("multi_object_replay", {})
    if replay_config.get("enabled", True):
        replay_dir = output_dir / "replay"
        for mode in modes:
            replay_started = time.perf_counter()
            replay_mp4 = replay_dir / f"combined_{mode}.mp4"
            replay_png = replay_dir / f"combined_{mode}_first_frame.png"
            replay_json = replay_dir / f"combined_{mode}.json"
            render_replay(
                [
                    "--segmentation-npz",
                    str(segmentation),
                    "--cotracker-npz",
                    str(output_dir / f"cotracker_{mode}.npz"),
                    "--megasam-npz",
                    str(report["geometry"]["cache_path"]),
                    "--source-video",
                    str(video),
                    "--view-mode",
                    str(replay_config.get("view_mode", "camera-pov")),
                    "--output-mp4",
                    str(replay_mp4),
                    "--first-frame-png",
                    str(replay_png),
                    "--metadata-json",
                    str(replay_json),
                    "--fps",
                    str(float(replay_config.get("fps", 16))),
                    "--max-grey-points",
                    str(int(replay_config.get("max_mask_points", 12000))),
                    "--grey-size",
                    str(float(replay_config.get("mask_point_size", 2))),
                    "--anchor-size",
                    str(float(replay_config.get("anchor_point_size", 28))),
                ]
            )
            replay_artifacts[mode] = {
                "video": str(replay_mp4),
                "first_frame": str(replay_png),
                "metadata": str(replay_json),
            }
            report["modes"][mode]["timing"]["replay_seconds"] = (
                time.perf_counter() - replay_started
            )
    report["timing"]["total_with_replay_seconds"] = time.perf_counter() - wall_started
    write_report(metrics_path, report)
    write_report(
        output_dir / "timing.json",
        {
            "shared": report["timing"],
            "modes": {
                mode: values["timing"] for mode, values in report["modes"].items()
            },
            "speed_comparison": (
                report["comparison"]["speed"] if report["comparison"] else None
            ),
            "status": "complete",
        },
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_revision": _git_revision(BENCHMARK_ROOT),
        "benchmark_source_sha256": _source_fingerprint(BENCHMARK_ROOT),
        "input": {"path": str(video), "sha256": _sha256(video)},
        "segmentation": {
            "path": str(segmentation),
            "sha256": _sha256(segmentation),
            "backend": "sam3-cad",
        },
        "tracking_modes": list(modes),
        "shared_geometry_cache": report["geometry"],
        "rigidity_scope": "per-object only; articulated union is never scored",
        "exact_command": shlex.join(sys.argv),
        "artifacts": {
            "metrics": str(metrics_path),
            "timing": str(output_dir / "timing.json"),
            "segmentation": str(output_dir / "segmentation.npz"),
            "track_archives": {
                mode: str(output_dir / f"cotracker_{mode}.npz") for mode in modes
            },
            "combined_replays": replay_artifacts,
        },
    }
    write_report(output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "metrics": str(metrics_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
