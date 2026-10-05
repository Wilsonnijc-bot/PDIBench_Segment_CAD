#!/usr/bin/env python3
"""Build a local navigation page for selected-45 Link 7 exact pair replays."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import html
import json
from pathlib import Path


ROOT = (_workspace_root() / 'results/selected-45-v1')
OUTPUT = (_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison')


def main() -> int:
    summary_file = (_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/summary.json')
    summary = json.loads(summary_file.read_text()) if summary_file.is_file() else {"cases": {}}
    rows = []
    done = 0
    for case in sorted(((_workspace_root() / 'results/selected-45-v1/cases')).iterdir()):
        status_file = case / "status.json"
        if not status_file.is_file():
            continue
        original = json.loads(status_file.read_text())
        sample = case.name
        run = summary["cases"].get(sample, {})
        manifest_file = (_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/cases') / sample / "replay/manifest.json"
        if manifest_file.is_file() and run.get("state") == "complete":
            replay = json.loads(manifest_file.read_text())
            if not replay["identical_initial_queries"]:
                raise ValueError(f"mismatched initial Link 7 queries: {sample}")
            done += 1
            links = [f'<a href="cases/{sample}/replay/index.html">Both replays</a>']
            scores = []
            for key, label in (("cotracker3", "CoTracker3"), ("tapip3d", "TAPIP3D")):
                item = replay["methods"][key]
                if item["page"]:
                    links.append(f'<a href="cases/{sample}/replay/{item["page"]}">{label}</a>')
                    scores.append(f'{label} ε={item["rigidity_epsilon"]:.4f}')
                else:
                    scores.append(f'{label}: unscored')
            state = f'{replay["selected_point_count"]} identical queries · ' + "; ".join(scores)
            action = " · ".join(links)
        elif original["state"] != "complete":
            state = "Original V1 case failed before tracking"
            action = "—"
        elif run.get("state") == "failed":
            state = "Comparison failed: " + html.escape(str(run.get("error", "unknown"))[:180])
            action = "—"
        else:
            state = "Comparison pending"
            action = "—"
        rows.append(f'<tr><th>{html.escape(sample)}</th><td>{state}</td><td>{action}</td></tr>')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ((_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/index.html')).write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selected 45 · Link 7 pair replays</title>
<style>body{font:15px/1.5 system-ui;margin:0;background:#f6f7f8;color:#17212b}
main{max-width:1200px;margin:auto;padding:32px 22px 80px}h1{margin:0 0 6px}
p{color:#566370}table{border-collapse:collapse;width:100%;background:white}
th,td{padding:10px;border-bottom:1px solid #dce2e8;text-align:left;vertical-align:top}
thead th{background:#edf2f5}tbody th{white-space:nowrap}a{color:#07579c;margin-right:10px}</style>
<main><h1>Selected 45 · Link 7 tracker pair replays</h1>
<p>''' + str(done) + ''' completed comparisons. Both trackers use the same existing Link 7 point selection,
saved MegaSAM geometry, and V1 replay logic. Each replay shows the actual selected pairs and
frame-by-frame rigidity score. The original selected-45 V1 results are <a href="../index.html">here</a>.</p>
<table><thead><tr><th>Video</th><th>Comparison</th><th>Exact pair replays</th></tr></thead>
<tbody>''' + "\n".join(rows) + '''</tbody></table></main></html>''', encoding="utf-8")
    print((_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/index.html'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
