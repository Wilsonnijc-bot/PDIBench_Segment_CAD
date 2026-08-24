#!/usr/bin/env python3
"""Regenerate CoTracker-CAD replays for a frozen snapshot with two workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def frozen_entries(
    manifest_path: Path,
    source_replay_root: Path,
) -> list[dict[str, Any]]:
    entries_by_job = {
        entry["job_id"]: entry for entry in read_json(manifest_path)["videos"]
    }
    selected = []
    for bundle_path in sorted(source_replay_root.glob("*/bundle.json")):
        job_id = read_json(bundle_path)["job_id"]
        if job_id not in entries_by_job:
            raise KeyError(f"frozen replay job is absent from manifest: {job_id}")
        selected.append(entries_by_job[job_id])
    if not selected:
        raise ValueError(f"no frozen replay bundles found under {source_replay_root}")
    return selected


def cleanup_generated_geometry(
    *,
    batch_root: Path,
    code_root: Path,
    result: dict[str, Any],
) -> int:
    """Remove rerender-only geometry after its clear replay has been published."""
    if result.get("geometry_cache_hit", False):
        return 0
    job_id = str(result["job_id"])
    job_root = (batch_root / "jobs" / job_id).resolve()
    cad_root = job_root / "output/replay/cad"
    required = [
        cad_root / "cotracker_cad_replay.mp4",
        cad_root / "cad_replay.json",
        cad_root / "initial_sam_masks.png",
        cad_root / "point_cloud_frame_0000.png",
        *sorted(cad_root.glob("foundationpose_frame_*.png")),
        *sorted(cad_root.glob("cotracker_cad_frame_*.png")),
    ]
    if len(required) < 12 or not all(
        path.is_file() and path.stat().st_size > 0 for path in required
    ):
        raise FileNotFoundError(f"refusing cleanup before replay is complete: {job_id}")

    removed_bytes = 0
    cache_path = Path(str(result["geometry_cache"])).resolve()
    cache_root = (job_root / "geometry-cache").resolve()
    if cache_path.parent != cache_root:
        raise ValueError(f"geometry cache is outside its job root: {cache_path}")
    if cache_path.is_file():
        removed_bytes += cache_path.stat().st_size
        cache_path.unlink()

    scene_name = job_id.replace(".", "_")
    mega_sam_root = code_root / "third_party/mega_sam"
    workspace = mega_sam_root / "work_space" / scene_name
    if workspace.is_dir():
        removed_bytes += sum(
            path.stat().st_size for path in workspace.rglob("*") if path.is_file()
        )
        shutil.rmtree(workspace)
    for artifact in (
        mega_sam_root / "outputs" / f"{scene_name}_droid.npz",
        mega_sam_root / "outputs_cvd" / f"{scene_name}_sgd_cvd_hr.npz",
    ):
        if artifact.is_file():
            removed_bytes += artifact.stat().st_size
            artifact.unlink()
    return removed_bytes


def rerender_one(task: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    os.environ.update(
        {
            "HF_HOME": "/root/autodl-tmp/pdi/cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    batch_root = Path(task["batch_root"])
    code_root = Path(task["code_root"])
    entry = json.loads(task["entry_json"])
    job_root = batch_root / "jobs" / entry["job_id"]
    input_root = job_root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    source_video = batch_root / "videos" / entry["staged_relative_path"]
    if not source_video.is_file():
        raise FileNotFoundError(f"staged video is missing: {source_video}")
    alias_name = f"{entry['job_id'].replace('.', '_')}.mp4"
    video = input_root / alias_name
    if video.is_symlink() and not video.exists():
        video.unlink()
    if not video.exists():
        video.symlink_to(source_video)
    if not video.is_file():
        raise FileNotFoundError(f"dot-free input alias is missing: {video}")
    segmentation_path = job_root / "intermediate/segmentation.npz"
    foundation_pose_path = job_root / "output/foundationpose_poses.npz"
    cotracker_path = job_root / "output/cotracker_exact-group.npz"
    cache_dir = job_root / "geometry-cache"
    cad_manifest = code_root / "configs/sam3-cad-franka.yaml"

    from pdi_eval.perception.mega_sam_wrapper import MegaSamWrapper
    from pdi_eval.perception.segmentation_archive import (
        load_multi_object_segmentation,
    )
    from pdi_eval.utils.cad_replay import main as render_cad_replay

    segmentation = load_multi_object_segmentation(segmentation_path, video)
    geometry = MegaSamWrapper(device="cuda").infer_shared(
        str(video),
        segmentation.object_masks,
        cache_dir=cache_dir,
    )
    if geometry.cache_path is None:
        raise RuntimeError(f"{entry['job_id']} did not produce a geometry cache")

    replay_root = job_root / "output/replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cad-clear-",
        dir=replay_root,
    ) as temporary:
        temporary_path = Path(temporary)
        status = render_cad_replay(
            [
                "--video",
                str(video),
                "--segmentation-npz",
                str(segmentation_path),
                "--geometry-npz",
                geometry.cache_path,
                "--cotracker-npz",
                str(cotracker_path),
                "--foundation-pose-npz",
                str(foundation_pose_path),
                "--cad-manifest",
                str(cad_manifest),
                "--output-dir",
                str(temporary_path),
                "--fps",
                "16",
            ]
        )
        if status != 0:
            raise RuntimeError(f"CAD replay renderer returned {status}")
        generated = sorted(path for path in temporary_path.iterdir() if path.is_file())
        required = {
            "cotracker_cad_replay.mp4",
            "cad_replay.json",
            "initial_sam_masks.png",
            "point_cloud_frame_0000.png",
        }
        missing = required.difference(path.name for path in generated)
        if missing:
            raise FileNotFoundError(f"CAD replay lacks artifacts: {sorted(missing)}")
        final_root = replay_root / "cad"
        final_root.mkdir(parents=True, exist_ok=True)
        obsolete = [
            final_root / "cad_pipeline_replay.mp4",
            *final_root.glob("cad_frame_*.png"),
        ]
        for path in obsolete:
            if path.is_file() or path.is_symlink():
                path.unlink()
        for source in generated:
            source.replace(final_root / source.name)

    result = {
        "dataset": entry["dataset"],
        "relative_path": entry["relative_path"],
        "job_id": entry["job_id"],
        "geometry_cache": geometry.cache_path,
        "geometry_cache_hit": geometry.metadata["cache_hit"],
        "seconds": time.perf_counter() - started,
        "status": "complete",
    }
    result["cleanup_bytes"] = cleanup_generated_geometry(
        batch_root=batch_root,
        code_root=code_root,
        result=result,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--source-replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers != 2:
        parser.error("clear CAD batch requires exactly two workers")

    manifest = args.manifest.resolve()
    batch_root = args.batch_root.resolve()
    code_root = args.code_root.resolve()
    source_replay_root = args.source_replay_root.resolve()
    output_root = args.output_root.resolve()
    entries = frozen_entries(manifest, source_replay_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = batch_root / "cotracker-cad-rerender-status.json"
    previous = read_json(status_path) if status_path.is_file() else {}
    entry_jobs = {entry["job_id"] for entry in entries}
    results: list[dict[str, Any]] = [
        result
        for result in previous.get("results", [])
        if result.get("status") == "complete" and result.get("job_id") in entry_jobs
    ]
    for result in results:
        result["cleanup_bytes"] = cleanup_generated_geometry(
            batch_root=batch_root,
            code_root=code_root,
            result=result,
        )
    failures: list[dict[str, Any]] = []
    write_json_atomic(
        status_path,
        {
            "state": "running",
            "total": len(entries),
            "complete": len(results),
            "failed": 0,
            "results": sorted(results, key=lambda value: value["job_id"]),
        },
    )
    completed_jobs = {result["job_id"] for result in results}
    tasks = [
        {
            "batch_root": str(batch_root),
            "code_root": str(code_root),
            "entry_json": json.dumps(entry),
        }
        for entry in entries
        if entry["job_id"] not in completed_jobs
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        futures = {executor.submit(rerender_one, task): task for task in tasks}
        for future in as_completed(futures):
            task_entry = json.loads(futures[future]["entry_json"])
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as exc:
                failure = {
                    "dataset": task_entry["dataset"],
                    "relative_path": task_entry["relative_path"],
                    "job_id": task_entry["job_id"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(json.dumps(failure, sort_keys=True), flush=True)
            write_json_atomic(
                status_path,
                {
                    "state": "running",
                    "total": len(entries),
                    "complete": len(results),
                    "failed": len(failures),
                    "results": sorted(results, key=lambda value: value["job_id"]),
                    "failures": sorted(failures, key=lambda value: value["job_id"]),
                },
            )

    from evaluation.package_batch_deliverables import (
        package_entry,
        write_concise_metrics_csv,
    )

    completed_jobs = {result["job_id"] for result in results}
    completed_entries = [
        entry for entry in entries if entry["job_id"] in completed_jobs
    ]
    packaged_entries = [
        entry
        for entry in completed_entries
        if package_entry(batch_root, output_root, entry)
    ]
    write_concise_metrics_csv(
        batch_root,
        output_root / "metrics.csv",
        packaged_entries,
    )
    final = {
        "state": "complete" if len(packaged_entries) == len(entries) else "failed",
        "total": len(entries),
        "complete": len(results),
        "packaged": len(packaged_entries),
        "failed": len(failures),
        "results": sorted(results, key=lambda value: value["job_id"]),
        "failures": sorted(failures, key=lambda value: value["job_id"]),
    }
    write_json_atomic(status_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if final["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
