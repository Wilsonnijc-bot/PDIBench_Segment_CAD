"""Selected-45 Link 5/7 matrix using the native masking, scorer, and replay interfaces."""

from __future__ import annotations

from infrastructure.deformation_detect.layout import root as _workspace_root

import html
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from infrastructure.shared.contracts.contracts import MASK_VIDEO_FOLDERS, ROOT, case_name, checked_file, mask_is_valid, read_selection, sha256_file, status_summary, video_path, write_json
from robot.preprocessing.segmentation.mask_merge import build_refined_segmentation
from robot.workflows.runner import MINIMUM_TRACKED_FRACTION, clean_megasam_intermediates, command, ensure_base_segmentation, ensure_video_alias, mask_coverage, persistent_mask, score_argv
from robot.workflows.score_v1 import _source_fingerprint

PATHS = (("v1_cotracker3", "v1", "cotracker3"),
         ("v2_cotracker3", "v2", "cotracker3"),
         ("v1_tapip3d", "v1", "tapip3d"),
         ("v2_tapip3d", "v2", "tapip3d"))
# The guard/client change only handles empty VLM answers during mask creation.
# Already scored masks under this fingerprint remain valid scorer artifacts.
COMPATIBLE_SCORER_FINGERPRINTS = frozenset({
    "465e424e2b4de2e124a104f19db86c917a8a339034f3b03bd6f3ef12cda0e686",
    "81a6319132638cf34f2b674f58cdf4d7bceb1b88a48fc1582dab24ff1fb62b03",
    "24aa66a575ad1312def5b08ce7bfd52f4891534d7e9cb08c653a17ed1d4b6e1a",
})
COMPATIBLE_VLM_CLIENT_SHA256 = frozenset({
    "ec961dc4ad73c06dfd01c42d364f0104d76d324f56ecaaf967b87a682a2f4238",
})


def _complete(folder: Path, source_hash: str, mask_hash: str, filter_name: str,
              tracker_name: str, score_links: tuple[str, ...]) -> bool:
    archive = ("cotracker" if tracker_name == "cotracker3" else "tapip3d_link7") + "_exact-group.npz"
    required = [folder / path for path in (
        "manifest.json", "metrics.json", "timing.json", "segmentation.npz", archive,
        "replay/combined_exact-group.mp4",
        "replay/interactive_exact-group/index.html",
        "replay/interactive_exact-group/source.mp4",
        "replay/interactive_exact-group/plotly.min.js",
    )]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
        metrics = json.loads((folder / "metrics.json").read_text())
        timing = json.loads((folder / "timing.json").read_text())
        objects = metrics["modes"]["exact-group"]["objects"]
        if any(objects[name]["status"] not in {"complete", "failed"} for name in score_links):
            return False
        for name in score_links:
            if objects[name]["status"] == "complete":
                pair = folder / "replay/interactive_exact-group" / f"{name}_exact-group_pairs.json"
                if not pair.is_file() or pair.stat().st_size == 0:
                    return False
        return (manifest["status"] == "complete" and timing["status"] == "complete"
                and manifest["input"]["sha256"] == source_hash
                and manifest["segmentation"]["sha256"] == mask_hash
                and manifest["benchmark_source_sha256"] in
                    (COMPATIBLE_SCORER_FINGERPRINTS | {_source_fingerprint(ROOT)})
                and manifest["link7_point_filter"] == filter_name
                and manifest["link7_tracker"] == tracker_name
                and manifest["score_links"] == list(score_links)
                and manifest["tracking_modes"] == ["exact-group"])
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _persistent_implementation_matches(work: Path) -> bool:
    provenance = work / "provenance.json"
    if not provenance.is_file():
        return True
    record = json.loads(provenance.read_text())
    sources = record.get("environment", {}).get("implementation", {})
    return bool(sources) and all(
        (ROOT / name).is_file()
        and (sha256_file(ROOT / name) == digest
             or (name == "persistent_masking/vlm_client.py"
                 and digest in COMPATIBLE_VLM_CLIENT_SHA256))
        for name, digest in sources.items()
    )


def _score(spec, video: Path, mask: Path, case: Path, folder: Path,
           source_hash: str, mask_hash: str, filter_name: str,
           tracker_name: str, score_links: tuple[str, ...], log: Path,
           mask_label: str) -> dict:
    if not _complete(folder, source_hash, mask_hash, filter_name, tracker_name, score_links):
        argv = score_argv(spec, video, mask, folder, case)
        argv += [item for name in score_links for item in ("--score-link", name)]
        argv += ["--link7-point-filter", filter_name, "--link7-tracker", tracker_name]
        argv += ["--mask-label", mask_label]
        if tracker_name == "tapip3d":
            argv += ["--tapip3d-python", str(spec.tapip3d_python),
                     "--tapip3d-repository", str(spec.tapip3d_repository),
                     "--tapip3d-checkpoint", str(spec.tapip3d_checkpoint)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str((_workspace_root()))
        command(argv, log, environment, spec.gpu_lock)
        if not _complete(folder, source_hash, mask_hash, filter_name, tracker_name, score_links):
            raise RuntimeError(f"scorer did not produce complete matching artifacts: {folder}")
    metrics = json.loads((folder / "metrics.json").read_text())
    objects = metrics["modes"]["exact-group"]["objects"]
    return {name: {"status": objects[name]["status"],
                   "error_type": objects[name].get("error_type"),
                   "epsilon_rigidity": objects[name].get("breakdown", {}).get("epsilon_rigidity")}
            for name in score_links}


def run_case(spec, entry: dict) -> bool:
    sample = f"{entry['dataset']}_{entry['video_number']}"
    case = spec.output_root / "cases" / sample
    case.mkdir(parents=True, exist_ok=True)
    log = case / "end_to_end.log"
    status_path = case / "status.json"
    status = {"sample_id": sample, "workbook_row": entry["workbook_row"],
              "source_sha256": entry["sha256"], "state": "running",
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    try:
        source = video_path(spec.video_root, entry)
        checked_file(source, "frozen source video")
        if source.stat().st_size != entry["size_bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError(f"source differs from frozen selection: {sample}")
        video = case / f"{sample.replace('.', '_')}.mp4"
        ensure_video_alias(video, source, entry["sha256"])
        base = ensure_base_segmentation(spec, entry, case, source, log)
        source_link = (spec.output_root / "source_video_links"
                       / MASK_VIDEO_FOLDERS[entry["dataset"]] / f"{entry['video_number']}.mp4")
        ensure_video_alias(source_link, source, entry["sha256"])
        from robot.workflows.runner import mask_work
        work = mask_work(spec.output_root, case_name(entry))
        if not _persistent_implementation_matches(work):
            raise ValueError(f"persistent masking implementation differs on resume: {sample}")
        work, mask_record = persistent_mask(spec, entry, log)
        status["persistent_mask_status"] = mask_record["status"]
        refined = case / "refined_segmentation.npz"
        if mask_record["status"] == "completed_checks":
            if not mask_is_valid(refined):
                build_refined_segmentation(video=video, base_segmentation=base,
                                           persistent_work=work, case=case_name(entry),
                                           output_npz=refined)
            if not mask_is_valid(refined):
                raise ValueError(f"invalid refined segmentation: {sample}")
            mask = refined
            policy = "persistent_refined_link7"
        else:
            mask = base
            policy = f"base_link7_after_{mask_record['status']}"
        coverage = mask_coverage(mask)
        if coverage["link5"] < MINIMUM_TRACKED_FRACTION:
            raise ValueError(f"Link 5 mask below 80% coverage: {coverage['link5']:.3f}")
        mask_hash = sha256_file(mask)
        status.update(mask_policy=policy, selected_mask_sha256=mask_hash,
                      tracked_fraction=coverage, paths={})
        low_link7 = coverage["link7"] < MINIMUM_TRACKED_FRACTION
        for path_name, filter_name, tracker_name in PATHS:
            folder = case / path_name
            if low_link7:
                status["paths"][path_name] = {
                    "link7": {"status": "failed", "error_type": "insufficient_sam3_coverage",
                              "tracked_fraction": coverage["link7"]},
                    "filter": filter_name, "tracker": tracker_name,
                    "selected_mask_sha256": mask_hash,
                }
                if path_name == "v1_cotracker3":
                    first = _score(spec, video, mask, case, folder, entry["sha256"],
                                   mask_hash, filter_name, tracker_name, ("link5",), log,
                                   policy)
                    status["link5"] = first["link5"]
                write_json(status_path, status)
                continue
            targets = ("link5", "link7") if path_name == "v1_cotracker3" else ("link7",)
            result = _score(spec, video, mask, case, folder, entry["sha256"],
                            mask_hash, filter_name, tracker_name, targets, log, policy)
            if "link5" in result:
                status["link5"] = result["link5"]
            status["paths"][path_name] = {
                "link7": result["link7"], "filter": filter_name,
                "tracker": tracker_name, "selected_mask_sha256": mask_hash,
            }
            write_json(status_path, status)
        references = json.loads((spec.output_root / "historical_references.json").read_text())
        status["historical_links"] = references["cases"].get(sample, {})
        status.update(state="complete", completed_at=datetime.now(timezone.utc).isoformat())
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


def export_index(output_root: Path, selection: dict) -> dict:
    summary = status_summary(output_root, selection)

    def result_cell(case: Path, sample: str, path_name: str, link_name: str) -> str:
        folder = case / path_name
        metrics_path = folder / "metrics.json"
        if not metrics_path.is_file():
            return "<td class='empty'>—</td>"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            result = metrics["modes"]["exact-group"]["objects"][link_name]
        except (OSError, ValueError, KeyError, TypeError):
            return "<td class='empty'>Metrics unavailable</td>"
        score = result.get("breakdown", {}).get("epsilon_rigidity")
        status = result.get("status", "pending")
        if isinstance(score, (int, float)):
            metric = f"<span class='score'>{score:.4f}</span>"
        else:
            metric = f"<span class='empty'>{html.escape(status)}</span>"
        replay = folder / "replay" / "interactive_exact-group" / f"{link_name}_exact-group.html"
        if replay.is_file():
            href = f"cases/{html.escape(sample)}/{path_name}/replay/interactive_exact-group/{link_name}_exact-group.html"
            metric += f'<a class="replay" href="{href}">Open replay</a>'
        return f"<td>{metric}</td>"

    rows = []
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        case = output_root / "cases" / sample
        state = summary["cases"][sample].get("state", "pending")
        cells = [result_cell(case, sample, "v1_cotracker3", "link5")]
        cells.extend(result_cell(case, sample, path_name, "link7")
                     for path_name, _, _ in PATHS)
        rows.append(f"<tr><th scope='row'>{html.escape(sample)}</th>"
                    f"<td class='status'>{html.escape(state)}</td>{''.join(cells)}</tr>")
    headings = "".join(f"<th scope='col'>Link 7 <small>{html.escape(path_name)}</small></th>"
                       for path_name, _, _ in PATHS)
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Selected 45 Link 5/7</title><style>"
            "body{font:15px system-ui,sans-serif;margin:32px;color:#18232d;background:#f7f9fb}"
            "h1{margin:0 0 8px}p{margin:0 0 24px;color:#52616e}"
            ".table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:980px;background:white}"
            "th,td{padding:12px 14px;border-bottom:1px solid #dde4e9;text-align:left;vertical-align:top}"
            "thead th{background:#e9f0f5;white-space:nowrap}tbody th{white-space:nowrap;font-weight:600}"
            "small,.replay{display:block}"
            ".score{font-variant-numeric:tabular-nums;font-weight:650}.replay{margin-top:6px}"
            ".status,.empty{color:#52616e}a{color:#075d9b}a:focus-visible{outline:2px solid #075d9b;outline-offset:2px}"
            "</style><body><h1>Selected 45 · Link 5/7</h1>"
            "<p>Rigidity scores (epsilon_rigidity) and direct replays for Link 5 and all four Link 7 paths.</p>"
            "<div class='table-wrap'><table><thead><tr><th scope='col'>Case</th>"
            "<th scope='col'>Status</th><th scope='col'>Link 5 <small>v1_cotracker3</small></th>"
            + headings + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></body></html>")
    (output_root / "index.html").write_text(page, encoding="utf-8")
    write_json(output_root / "summary.json", summary)
    return summary


def run_experiment(spec, *, samples: list[str] | None = None) -> int:
    selection = read_selection(spec.output_root)
    entries = selection["videos"]
    if samples:
        requested = set(samples)
        entries = [entry for entry in entries
                   if f"{entry['dataset']}_{entry['video_number']}" in requested]
        if len(entries) != len(requested):
            found = {f"{entry['dataset']}_{entry['video_number']}" for entry in entries}
            raise ValueError(f"samples absent from frozen selection: {sorted(requested - found)}")
    with ThreadPoolExecutor(max_workers=spec.concurrency) as workers:
        futures = [workers.submit(run_case, spec, entry) for entry in entries]
        success = True
        for future in as_completed(futures):
            success = future.result() and success
            export_index(spec.output_root, selection)
    return 0 if success else 1
