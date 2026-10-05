"""Create a rigidity-only entry page for the first completed four-way case."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/build_link5_link7_first_review.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import html
import json
from pathlib import Path

ROOT = _SOURCE_PATH.parents[1]
RESULTS = ROOT / "results/link5-link7-four-way-selected45-20260927"
SAMPLE = "LVP_ROBOWM_0001"
PATHS = (("v1_cotracker3", "V1 filter + CoTracker3"),
         ("v2_cotracker3", "V2 depth filter + CoTracker3"),
         ("v1_tapip3d", "V1 filter + TAPIP3D"),
         ("v2_tapip3d", "V2 depth filter + TAPIP3D"))


def main() -> None:
    case = RESULTS / "cases" / SAMPLE
    status = json.loads((case / "status.json").read_text())
    if status.get("state") != "complete" or len(status.get("paths", {})) != 4:
        raise ValueError("first case is not complete")
    rows = []
    for folder, label in (("v1_cotracker3", "Link 5 · guarded base mask"), *PATHS):
        link = "link5" if label.startswith("Link 5") else "link7"
        report = status["link5"] if link == "link5" else status["paths"][folder]["link7"]
        replay = case / folder / "replay/interactive_exact-group" / f"{link}_exact-group.html"
        if report["status"] == "complete" and not replay.is_file():
            raise FileNotFoundError(f"missing scored pair replay: {replay}")
        value = report.get("epsilon_rigidity")
        rigidity = f"{value:.4f}" if isinstance(value, (int, float)) else report["status"]
        href = replay.relative_to(RESULTS).as_posix()
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{rigidity}</td>"
                    f"<td><a href='{html.escape(href)}'>Open interactive replay</a></td></tr>")
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>First case · Link 5/7 rigidity</title>"
            "<style>body{font:16px system-ui;max-width:1000px;margin:48px auto;padding:0 20px;color:#202529}"
            "table{width:100%;border-collapse:collapse}th,td{padding:14px;border-bottom:1px solid #ccc;text-align:left}"
            "th{font-size:13px;text-transform:uppercase;letter-spacing:.06em}a{color:#075d9a}</style>"
            "</head><body><h1>LVP_ROBOWM_0001 · Link 5/7 rigidity</h1>"
            "<p>All five replays use the same selected six-link mask. Link 5 is scored once; "
            "Link 7 uses four filter/tracker paths. Values are measured rigidity epsilon.</p>"
            "<table><thead><tr><th>Path</th><th>Rigidity</th><th>Replay</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></body></html>")
    destination = RESULTS / "first-case-review.html"
    destination.write_text(page)
    print(destination)


if __name__ == "__main__":
    main()
