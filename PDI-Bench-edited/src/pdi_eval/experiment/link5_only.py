"""Selected 45, updated SAM3 Link 5 mask, unchanged V1 Link 5 scorer."""

from __future__ import annotations

import html
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .contracts import checked_file, read_selection, sha256_file, status_summary, video_path, write_json
from .runner import (clean_megasam_intermediates, command,
                     ensure_base_segmentation, ensure_video_alias,
                     mask_coverage, score_argv)

MASK_LABEL = "Updated Link 5 SAM3 guard: 2 VLM calls, 3 positive, 3 negative"


def _complete(folder: Path, video_hash: str, mask_hash: str) -> bool:
    required = ("metrics.json", "manifest.json", "timing.json", "segmentation.npz",
                "cotracker_exact-group.npz", "replay/combined_exact-group.mp4",
                "replay/interactive_exact-group/index.html",
                "replay/interactive_exact-group/source.mp4",
                "replay/interactive_exact-group/plotly.min.js")
    if any(not (folder / name).is_file() or not (folder / name).stat().st_size
           for name in required):
        return False
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
        metrics = json.loads((folder / "metrics.json").read_text())
        timing = json.loads((folder / "timing.json").read_text())
        objects = metrics["modes"]["exact-group"]["objects"]
        return (manifest["status"] == timing["status"] == "complete"
                and manifest["input"]["sha256"] == video_hash
                and manifest["segmentation"]["sha256"] == mask_hash
                and manifest["score_links"] == ["link5"]
                and manifest["tracking_modes"] == ["exact-group"]
                and manifest["link7_point_filter"] == "v1"
                and set(objects) == {"link5"}
                and objects["link5"]["status"] in {"complete", "failed"}
                and (objects["link5"]["status"] != "complete" or
                     (folder / "replay/interactive_exact-group/link5_exact-group_pairs.json").is_file()))
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_case(spec, entry: dict) -> bool:
    sample = f"{entry['dataset']}_{entry['video_number']}"
    case = spec.output_root / "cases" / sample
    case.mkdir(parents=True, exist_ok=True)
    status_path = case / "status.json"
    status = {"sample_id": sample, "state": "running", "source_sha256": entry["sha256"],
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    log = case / "end_to_end.log"
    try:
        source = video_path(spec.video_root, entry)
        checked_file(source, "frozen source video")
        if source.stat().st_size != entry["size_bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError(f"source differs from frozen selection: {sample}")
        alias = case / f"{sample.replace('.', '_')}.mp4"
        ensure_video_alias(alias, source, entry["sha256"])
        base = ensure_base_segmentation(spec, entry, case, source, log,
                                        required_links=(),
                                        selected_target="link5")
        guard_path = case / "base_generation/link5_guard.json"
        checked_file(guard_path, "updated Link 5 VLM guard record")
        guard = json.loads(guard_path.read_text())
        if (len(guard.get("attempts", [])) != 2
                or len(guard.get("selected_points_xy", [])) != 6
                or set(guard.get("reviews", {})) != {"positive", "negative"}):
            raise ValueError("Link 5 guard did not record two reviews and six points")
        status.update(mask_sha256=sha256_file(base),
                      link5_coverage=mask_coverage(base)["link5"],
                      guard_decision=guard["decision"],
                      guard_attempt_count=2)
        # Local controller stages one verified historical geometry cache per case.
        ready = case / "geometry-cache/ready"
        deadline = time.monotonic() + 24 * 3600
        while not ready.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"geometry cache was not staged: {sample}")
            time.sleep(10)
        cache_files = list((case / "geometry-cache").glob("*.npz"))
        if len(cache_files) != 1 or cache_files[0].stat().st_size == 0:
            raise ValueError(f"expected one staged geometry cache: {sample}")
        destination = case / "v1_cotracker3"
        if not _complete(destination, entry["sha256"], status["mask_sha256"]):
            argv = score_argv(spec, alias, base, destination, case)
            argv += ["--score-link", "link5", "--link7-point-filter", "v1",
                     "--mask-label", MASK_LABEL]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
            command(argv, log, environment, spec.gpu_lock)
        if not _complete(destination, entry["sha256"], status["mask_sha256"]):
            raise RuntimeError("V1 link5 scorer did not produce matching metrics and replays")
        metrics = json.loads((destination / "metrics.json").read_text())
        link5 = metrics["modes"]["exact-group"]["objects"]["link5"]
        if link5["status"] != "complete":
            raise RuntimeError(f"V1 Link 5 score failed: {link5.get('error_type', 'unknown')}")
        status.update(state="complete", link5_status=link5["status"],
                      epsilon_rigidity=link5.get("breakdown", {}).get("epsilon_rigidity"),
                      completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status.update(state="failed", error=str(exc),
                      completed_at=datetime.now(timezone.utc).isoformat())
        with log.open("a") as stream:
            stream.write("\n" + traceback.format_exc() + "\n")
    if status["state"] == "complete":
        try:
            clean_megasam_intermediates(sample)
        except Exception:
            with log.open("a") as stream:
                stream.write("MegaSAM scratch cleanup failed:\n" + traceback.format_exc())
    write_json(status_path, status)
    print(f"{sample}: {status['state']}", flush=True)
    return status["state"] == "complete"


def export_index(output_root: Path, selection: dict) -> dict:
    summary = status_summary(output_root, selection)
    rows = []
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        state = summary["cases"][sample]
        case = output_root / "cases" / sample
        score = state.get("epsilon_rigidity")
        value = f"{score:.4f}" if isinstance(score, (int, float)) else html.escape(state.get("link5_status", "—"))
        replay = case / "v1_cotracker3/replay/interactive_exact-group/link5_exact-group.html"
        link = (f'<a href="cases/{html.escape(sample)}/v1_cotracker3/replay/'
                f'interactive_exact-group/link5_exact-group.html">Interactive replay</a>') if replay.is_file() else "—"
        rows.append(f"<tr><th>{html.escape(sample)}</th><td>{html.escape(state['state'])}</td>"
                    f"<td>{value}</td><td>{link}</td></tr>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Selected 45 · Link 5</title><style>"
            "body{font:15px system-ui,sans-serif;margin:32px;background:#f7f9fb;color:#18232d}"
            "table{border-collapse:collapse;width:100%;background:white}"
            "th,td{padding:10px 14px;border-bottom:1px solid #dde4e9;text-align:left}"
            "a{color:#075d9b}</style><body><h1>Selected 45 · Link 5</h1>"
            "<p>Updated SAM3 guard; V1 exact-group CoTracker rigidity and interactive replays.</p>"
            "<table><thead><tr><th>Case</th><th>Status</th><th>ε rigidity</th>"
            "<th>Replay</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></body></html>")
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
    with ThreadPoolExecutor(max_workers=spec.concurrency) as pool:
        futures = [pool.submit(run_case, spec, entry) for entry in entries]
        success = True
        for future in as_completed(futures):
            success = future.result() and success
            export_index(spec.output_root, selection)
    return 0 if success else 1
