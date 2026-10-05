"""Two-path Link 7 experiment using the existing case runner stages."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from infrastructure.shared.contracts.contracts import MASK_VIDEO_FOLDERS, case_name, checked_file, read_selection, sha256_file, video_path, write_json
from robot.preprocessing.segmentation.mask_merge import build_refined_segmentation
from robot.workflows.runner import LINK_NAMES, clean_megasam_intermediates, command, ensure_base_segmentation, ensure_video_alias, persistent_mask, score_argv, validate_mask_coverage
from robot.workflows.spec import load_spec


def _complete(destination: Path, video_hash: str, mask_hash: str,
              *, tapip: bool) -> bool:
    archive = "tapip3d_link7_exact-group.npz" if tapip else "cotracker_exact-group.npz"
    required = [destination / name for name in (
        "manifest.json", "metrics.json", "timing.json", archive,
    )]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    manifest = json.loads((destination / "manifest.json").read_text())
    metrics = json.loads((destination / "metrics.json").read_text())
    link7 = metrics["modes"]["exact-group"]["objects"]["link7"]
    if link7["status"] == "complete":
        replay = destination / "replay/interactive_exact-group"
        if not all((replay / name).is_file() and (replay / name).stat().st_size > 0
                   for name in ("link7_exact-group.html",
                                "link7_exact-group_pairs.json", "source.mp4",
                                "plotly.min.js")):
            return False
    return (manifest.get("status") == "complete"
            and manifest.get("input", {}).get("sha256") == video_hash
            and manifest.get("segmentation", {}).get("sha256") == mask_hash
            and manifest.get("link7_point_filter", "v1") == ("v2" if tapip else "v1")
            and manifest.get("link7_tracker") == ("tapip3d" if tapip else "cotracker3")
            and link7["status"] in {"complete", "failed", "skipped"})


def _export_link7(destination: Path, video: Path, mask: Path, cache: Path,
                  *, tapip: bool, mask_policy: str = "base") -> None:
    from infrastructure.shared.replay.rigidity_replay import export_v1_replay

    archive = "tapip3d_link7_exact-group.npz" if tapip else "cotracker_exact-group.npz"
    pages = export_v1_replay(
        destination / "metrics.json", destination / archive, mask, cache, video,
        destination / "replay/interactive_exact-group", mode="exact-group",
        plotly_js=(_workspace_root() / 'infrastructure/shared/replay/assets/plotly.min.js'),
        only_objects=("link7",), tracker_label="TAPIP3D" if tapip else "CoTracker3",
        mask_label=("persistent refined mask" if mask_policy == "persistent_refined"
                    else "base mask fallback (persistent refinement failed)"
                    if mask_policy.startswith("base_fallback") else "base mask"),
    )
    if len(pages) != 1:
        raise RuntimeError("Link 7 interactive pair replay was not exported")


def run_case(spec, entry: dict) -> bool:
    sample = f"{entry['dataset']}_{entry['video_number']}"
    case = spec.output_root / "cases" / sample
    case.mkdir(parents=True, exist_ok=True)
    log = case / "end_to_end.log"
    status_path = case / "status.json"
    status = {"sample_id": sample, "state": "running",
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    try:
        source = video_path(spec.video_root, entry)
        checked_file(source, "source video")
        if source.stat().st_size != entry["size_bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError("source video differs from frozen selection")
        video = case / f"{sample.replace('.', '_')}.mp4"
        ensure_video_alias(video, source, entry["sha256"])
        base = ensure_base_segmentation(spec, entry, case, source, log)
        base_hash = sha256_file(base)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str((_workspace_root()))
        baseline = case / "path1_v1_cotracker"
        if not _complete(baseline, entry["sha256"], base_hash, tapip=False):
            command(score_argv(spec, video, base, baseline, case),
                    log, environment, spec.gpu_lock)
            if not _complete(baseline, entry["sha256"], base_hash, tapip=False):
                raise RuntimeError("Path 1 scorer or required replay is incomplete")
        status["path1"] = "complete"
        write_json(status_path, status)

        source_link = (spec.output_root / "source_video_links"
                       / MASK_VIDEO_FOLDERS[entry["dataset"]]
                       / f"{entry['video_number']}.mp4")
        ensure_video_alias(source_link, source, entry["sha256"])
        work, mask_record = persistent_mask(spec, entry, log)
        status["persistent_mask_status"] = mask_record["status"]
        refined = case / "refined_segmentation.npz"
        if mask_record["status"] == "completed_checks":
            if not refined.is_file():
                build_refined_segmentation(video=video, base_segmentation=base,
                                           persistent_work=work, case=case_name(entry),
                                           output_npz=refined)
            selected_mask = refined
            mask_policy = "persistent_refined"
        else:
            selected_mask = base
            mask_policy = f"base_fallback_after_{mask_record['status']}"
        validate_mask_coverage(selected_mask)
        selected_hash = sha256_file(selected_mask)
        alternate = case / "path2_v2_tapip3d"
        if not _complete(alternate, entry["sha256"], selected_hash, tapip=True):
            argv = score_argv(spec, video, selected_mask, alternate, case)
            argv += ["--link7-tracker", "tapip3d", "--link7-point-filter", "v2",
                     "--tapip3d-python", "/root/autodl-tmp/pdi/env/tapip3d/bin/python",
                     "--tapip3d-repository", "/root/autodl-tmp/pdi/code/TAPIP3D",
                     "--tapip3d-checkpoint", "/root/autodl-tmp/pdi/models/tapip3d/tapip3d_final.pth",
                     "--disable-replay"]
            command(argv, log, environment, spec.gpu_lock)
            metrics = json.loads((alternate / "metrics.json").read_text())
            if metrics["modes"]["exact-group"]["objects"]["link7"]["status"] == "complete":
                cache = Path(metrics["geometry"]["cache_path"])
                _export_link7(alternate, video, selected_mask, cache, tapip=True,
                              mask_policy=mask_policy)
            if not _complete(alternate, entry["sha256"], selected_hash, tapip=True):
                raise RuntimeError("Path 2 scorer or required replay is incomplete")
        first = json.loads((baseline / "metrics.json").read_text())
        second = json.loads((alternate / "metrics.json").read_text())
        write_json(case / "comparison.json", {
            "sample_id": sample,
            "paths": {
                "path1": {"mask": "base", "tracker": "cotracker3", "filter": "v1"},
                "path2": {"mask": mask_policy, "tracker": "tapip3d", "filter": "v2"},
            },
            "links": {name: {
                path: report["modes"]["exact-group"]["objects"][name].get("status")
                for path, report in (("path1", first), ("path2", second))
            } for name in LINK_NAMES},
            "link7_rigidity": {
                path: report["modes"]["exact-group"]["objects"]["link7"]
                .get("breakdown", {}).get("epsilon_rigidity")
                for path, report in (("path1", first), ("path2", second))
            },
        })
        status.update(state="complete", path2="complete",
                      completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status.update(state="failed", error=str(exc),
                      completed_at=datetime.now(timezone.utc).isoformat())
        with log.open("a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
    try:
        clean_megasam_intermediates(sample)
    except Exception:
        with log.open("a", encoding="utf-8") as stream:
            stream.write("MegaSAM scratch cleanup failed:\n" + traceback.format_exc())
    write_json(status_path, status)
    print(f"{sample}: {status['state']}", flush=True)
    return status["state"] == "complete"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[],
                        help="Run only this sample ID (repeatable)")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    selection = read_selection(spec.output_root)
    entries = selection["videos"]
    if args.sample:
        requested = set(args.sample)
        entries = [entry for entry in entries
                   if f"{entry['dataset']}_{entry['video_number']}" in requested]
        if len(entries) != len(requested):
            found = {f"{entry['dataset']}_{entry['video_number']}" for entry in entries}
            raise ValueError(f"samples absent from selection: {sorted(requested - found)}")
    with ThreadPoolExecutor(max_workers=spec.concurrency) as pool:
        outcomes = [pool.submit(run_case, spec, entry) for entry in entries]
        results = [future.result() for future in as_completed(outcomes)]
        success = all(results)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
