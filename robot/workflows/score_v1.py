#!/usr/bin/env python3
"""Run native shared-geometry PDI for every object in a SAM3 archive."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_ROOT = (_workspace_root())
sys.path.insert(0, str((_workspace_root())))

TRACKING_MODES = ("joint-query", "exact-group")


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
    sources = sorted(
        path for owner in ("robot", "object", "infrastructure/shared", "infrastructure/pdibench")
        for path in (root / owner).rglob("*.py")
        if not {"archive", "results", "tests", "__pycache__"}.intersection(path.relative_to(root).parts)
    )
    sources.append(root / "robot/configs/default.yaml")
    for source in sources:
        relative = source.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(_sha256(source).encode())
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=(_workspace_root() / 'robot/configs/default.yaml'))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--segmentation-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path)
    parser.add_argument("--link7-tracker", choices=("cotracker3", "tapip3d"), default="cotracker3")
    parser.add_argument("--link7-point-filter", choices=("v1", "v2"), default="v1")
    parser.add_argument("--score-link", action="append", default=[],
                        choices=tuple(f"link{i}" for i in range(2, 8)),
                        help="Score only selected links; other links remain recorded as skipped")
    parser.add_argument("--mask-label", default=None,
                        help="Human-readable selected-mask policy for interactive replays")
    parser.add_argument("--tapip3d-python", type=Path)
    parser.add_argument("--tapip3d-repository", type=Path)
    parser.add_argument("--tapip3d-checkpoint", type=Path)
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
    parser.add_argument(
        "--skip-low-coverage-link", action="append", default=[],
        choices=tuple(f"link{i}" for i in range(2, 8)),
        help="Leave a link unscored when its SAM3 mask covers too few frames",
    )
    parser.add_argument("--minimum-tracked-fraction", type=float, default=0.80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import yaml
    from robot.workflows.pipeline import MultiObjectPDIEvaluationPipeline, write_report
    from infrastructure.shared.replay.reconstruct_replay import main as render_replay
    from infrastructure.shared.replay.rigidity_replay import export_v1_replay
    started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    # Keep the dataset-qualified symlink name as MegaSAM's scene identifier.
    video = Path(os.path.abspath(args.input))
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
    if not 0.0 <= args.minimum_tracked_fraction <= 1.0:
        raise ValueError("minimum tracked fraction must be between 0 and 1")
    if args.skip_low_coverage_link:
        tracking_config = config.setdefault("multi_object_tracking", {})
        tracking_config["skip_low_coverage_links"] = list(dict.fromkeys(args.skip_low_coverage_link))
        tracking_config["minimum_tracked_fraction"] = args.minimum_tracked_fraction
    if args.disable_replay:
        config.setdefault("multi_object_replay", {})["enabled"] = False
    if args.tracker_checkpoint is not None:
        checkpoint = args.tracker_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"tracker checkpoint is missing: {checkpoint}")
        config["tracker_ckpt"] = str(checkpoint)
    if args.link7_tracker == "tapip3d":
        tapip = config.setdefault("tapip3d", {})
        for key, value in (("python", args.tapip3d_python),
                           ("repository", args.tapip3d_repository),
                           ("checkpoint", args.tapip3d_checkpoint)):
            if value is not None:
                # Keep the virtualenv interpreter symlink: resolve() points to
                # its base Python and silently switches environments.
                tapip[key] = os.path.abspath(value) if key == "python" else str(value.resolve())
        missing = [key for key in ("python", "repository", "checkpoint") if not tapip.get(key)]
        if missing:
            raise ValueError(f"TAPIP3D requires paths for: {missing}")
    modes = TRACKING_MODES if args.tracking_mode == "both" else (args.tracking_mode,)
    report = MultiObjectPDIEvaluationPipeline(config).run(
        video_path=str(video),
        segmentation_npz=str(segmentation),
        tracking_modes=modes,
        link7_tracker=args.link7_tracker,
        link7_point_filter=args.link7_point_filter,
        score_links=tuple(dict.fromkeys(args.score_link)) or None,
        output_dir=output_dir,
        geometry_cache_dir=cache_dir,
    )
    segmentation_metadata_path = segmentation.with_suffix(".json")
    segmentation_metadata = {}
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
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    replay_artifacts = {}
    track_prefix = "cotracker" if args.link7_tracker == "cotracker3" else "tapip3d_link7"
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
                    str(output_dir / f"{track_prefix}_{mode}.npz"),
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
    if replay_config.get("enabled", True):
        for mode in modes:
            interactive_dir = output_dir / "replay" / f"interactive_{mode}"
            pages = export_v1_replay(
                metrics_path,
                output_dir / f"{track_prefix}_{mode}.npz",
                segmentation,
                Path(report["geometry"]["cache_path"]),
                video,
                interactive_dir,
                mode=mode,
                max_cloud_points=int(replay_config.get("interactive_cloud_points", 900)),
                plotly_js=(_workspace_root() / 'infrastructure/shared/replay/assets/plotly.min.js'),
                tracker_label="TAPIP3D" if args.link7_tracker == "tapip3d" else "CoTracker3",
                mask_label=args.mask_label,
            )
            replay_artifacts[mode]["interactive_index"] = str(interactive_dir / "index.html")
            replay_artifacts[mode]["interactive_pages"] = [str(page) for page in pages]
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
            "backend": (
                segmentation_metadata.get("backend")
                or segmentation_metadata.get("method")
                or ("dinov2-sam3" if "models" in segmentation_metadata else "provided-segmentation")
            ),
        },
        "tracking_modes": list(modes),
        "link7_tracker": args.link7_tracker,
        "link7_point_filter": args.link7_point_filter,
        "score_links": list(dict.fromkeys(args.score_link)) or list(report["segmentation"]["object_names"]),
        "mask_label": args.mask_label,
        "shared_geometry_cache": report["geometry"],
        "rigidity_scope": "per-object only; articulated union is never scored",
        "exact_command": shlex.join(sys.argv),
        "artifacts": {
            "metrics": str(metrics_path),
            "timing": str(output_dir / "timing.json"),
            "segmentation": str(output_dir / "segmentation.npz"),
            "track_archives": {
                mode: str(output_dir / f"{track_prefix}_{mode}.npz") for mode in modes
            },
            "combined_replays": replay_artifacts,
        },
    }
    write_report(output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "metrics": str(metrics_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
