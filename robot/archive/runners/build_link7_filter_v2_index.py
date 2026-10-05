#!/usr/bin/env python3
"""Build a local comparison index from verified two-path case artifacts."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/build_link7_filter_v2_index.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import html
import json
from pathlib import Path

ROOT = _SOURCE_PATH.parents[1] / "results/link7-point-filter-v2-20260926"
selection = json.loads((ROOT / "selection.json").read_text())
rows = []
summary = {}
for entry in selection["videos"]:
    sample = f"{entry['dataset']}_{entry['video_number']}"
    case = ROOT / "cases" / sample
    status_path = case / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {"state": "pending"}
    summary[sample] = status
    comparison_path = case / "comparison.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.is_file() else {}
    cells = []
    for label, folder in (("Path 1 · V1 + CoTracker3", "path1_v1_cotracker"),
                          ("Path 2 · V2 + TAPIP3D", "path2_v2_tapip3d")):
        metrics_path = case / folder / "metrics.json"
        if metrics_path.is_file():
            report = json.loads(metrics_path.read_text())["modes"]["exact-group"]["objects"]["link7"]
            state = report["status"]
            epsilon = report.get("breakdown", {}).get("epsilon_rigidity")
            details = f"{html.escape(state)}"
            if epsilon is not None:
                details += f" · rigidity ε {epsilon:.4f}"
            gate = report.get("tracking", {}).get("initial_depth_gate")
            if gate:
                if gate.get("stage") == "before_query_sampling":
                    details += (f" · depth eligible {gate['retained_count']}/"
                                f"{gate['candidate_count']} pixels"
                                f" · queries {gate['selected_query_count']}/"
                                f"{gate['requested_query_count']}")
                else:
                    details += (f" · depth gate {gate['retained_count']}/"
                                f"{gate['candidate_count']} queries")
            if report.get("error_type"):
                details += f" · {html.escape(report['error_type'])}"
            links = [f'<a href="cases/{html.escape(sample)}/{folder}/metrics.json">metrics</a>']
            replay = case / folder / "replay/interactive_exact-group/link7_exact-group.html"
            if replay.is_file():
                links.append(f'<a href="cases/{html.escape(sample)}/{folder}/replay/interactive_exact-group/link7_exact-group.html">interactive Link 7</a>')
            if folder == "path1_v1_cotracker":
                for link in ("link2", "link5"):
                    other = case / folder / f"replay/interactive_exact-group/{link}_exact-group.html"
                    if other.is_file():
                        links.append(
                            f'<a href="cases/{html.escape(sample)}/{folder}/replay/'
                            f'interactive_exact-group/{link}_exact-group.html">'
                            f'interactive Link {link[-1]}</a>'
                        )
            cells.append(f"<strong>{label}</strong><br>{details}<br>{' · '.join(links)}")
        else:
            cells.append(f"<strong>{label}</strong><br>"
                         + ("not scored" if status["state"] == "failed" else "pending"))
    note = comparison.get("paths", {}).get("path2", {}).get("mask", "")
    if not note and status.get("persistent_mask_status"):
        note = "persistent masking: " + status["persistent_mask_status"]
    if status.get("error"):
        note += (" · " if note else "") + status["error"]
    rows.append(f"<tr><td>{html.escape(sample)}</td><td>{html.escape(status['state'])}"
                f"<br><small>{html.escape(note)}</small></td><td>{cells[0]}</td><td>{cells[1]}</td></tr>")
page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Link 7 point filtering V1/V2 comparison</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1300px;margin:40px auto;padding:0 20px;color:#182321}"
        "table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #ccd5d0;text-align:left;vertical-align:top}"
        "th{background:#eaf0ed}a{color:#145c9b}small{color:#5a6e68}</style>"
        "<h1>Link 7 point filtering V1/V2</h1>"
        "<p>Path 1 uses the fresh base mask, V1 depth-gradient pair filter, and CoTracker3. "
        "Path 2 attempts persistent Link 7 refinement, then runs V2 camera-depth gating before query sampling and TAPIP3D; "
        "the case record identifies any base-mask fallback. "
        "Failed links retain their recorded failure and have no scored-pair replay.</p>"
        "<p><a href='link5-prompt-review/index.html'>Inspect the original Link 5 prompt boxes "
        "and points before the preflight correction</a></p>"
        "<p><a href='vlm2-review/COSMOS2.5_0054/points.png'>Inspect the accepted "
        "COSMOS2.5_0054 VLM2 points</a> · "
        "<a href='vlm2-review/COSMOS2.5_0054/after_provenance.json'>VLM2 provenance</a></p>"
        "<table><thead><tr><th>Video</th><th>Case</th><th>Path 1</th><th>Path 2</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></html>")
(ROOT / "index.html").write_text(page, encoding="utf-8")
(ROOT / "summary.json").write_text(json.dumps({"cases": summary}, indent=2) + "\n")
print(ROOT / "index.html")
