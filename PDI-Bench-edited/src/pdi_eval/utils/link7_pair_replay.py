"""Wrap the existing V1 interactive replay for matched Link 7 trackers."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from pdi_eval.utils.rigidity_replay import export_v1_replay


def export_link7_pair_replay(
    *, video_path: Path, segmentation_path: Path, pointmaps_path: Path,
    cotracker_dir: Path, tapip3d_dir: Path, output_dir: Path,
    plotly_js: Path, max_cloud_points: int = 900,
) -> Path:
    with np.load(cotracker_dir / "link7_initial_queries.npz", allow_pickle=False) as a, \
         np.load(tapip3d_dir / "link7_initial_queries.npz", allow_pickle=False) as b:
        if (not np.array_equal(a["queries"], b["queries"])
                or not np.array_equal(a["point_ids"], b["point_ids"])):
            raise ValueError("Link 7 replay trackers have different initial queries or IDs")
        selected_count = int(len(a["point_ids"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = {}
    for backend, folder, archive, label in (
        ("cotracker3", cotracker_dir, "cotracker_exact-group.npz", "CoTracker3"),
        ("tapip3d", tapip3d_dir, "tapip3d_link7_exact-group.npz", "TAPIP3D"),
    ):
        pages = export_v1_replay(
            folder / "metrics.json", folder / archive,
            segmentation_path, pointmaps_path, video_path,
            output_dir / backend, mode="exact-group",
            max_cloud_points=max_cloud_points, plotly_js=plotly_js,
            only_objects=("link7",), tracker_label=label,
        )
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        link7 = metrics["modes"]["exact-group"]["objects"]["link7"]
        if link7["status"] == "complete" and len(pages) != 1:
            raise ValueError(f"{label} scored Link 7 but produced no exact replay")
        methods[backend] = {"status": link7["status"],
                            "rigidity_epsilon": link7.get("breakdown", {}).get("epsilon_rigidity"),
                            "selected_pairs": None,
                            "page": f"{backend}/link7_exact-group.html" if pages else None,
                            "reason": link7.get("error", link7.get("error_type"))}
        if pages:
            evidence = json.loads((output_dir / backend / "link7_exact-group_pairs.json").read_text())
            methods[backend]["selected_pairs"] = len(evidence["selected_pairs"])
            if not np.allclose(evidence["rigidity_history"],
                               link7["breakdown"]["volume_history"],
                               rtol=1e-8, atol=1e-10):
                raise ValueError(f"{label} replay history differs from saved score")
    cards = []
    for backend, label in (("cotracker3", "CoTracker3"), ("tapip3d", "TAPIP3D")):
        item = methods[backend]
        if item["page"]:
            link = f'<a href="{item["page"]}">Open exact Link 7 replay</a>'
            detail = (f'{item["selected_pairs"]} actual scored point pairs · '
                      f'rigidity ε {item["rigidity_epsilon"]:.4f}')
        else:
            link = '<span class="unscored">No scored pair replay</span>'
            detail = html.escape(item["reason"] or "Link 7 was unscored")
        cards.append(f'<section><h2>{label}</h2><p>{detail}</p>{link}</section>')
    page = output_dir / "index.html"
    page.write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Link 7 · exact V1 pair replays</title>
<style>body{font:16px/1.5 system-ui;margin:0;background:#12181a;color:#edf2ed}
main{max-width:980px;margin:auto;padding:36px 24px}h1{margin:0 0 8px;font-size:28px}
p{color:#b9c8bf} .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:24px 0}
section{border:1px solid #3b4845;background:#1c2525;padding:20px}h2{margin:0;font-size:20px}
a{color:#f3c77c}.unscored{color:#ef9a8f}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style>
<main><h1>Link 7 · exact V1 pair replays</h1>
<p>The two trackers received the same ''' + str(selected_count) + ''' selected Link 7 queries and IDs.
Each link opens the original V1 interactive replay: source video, reconstructed point cloud,
actual selected scoring pairs, frame-by-frame rigidity history, and per-frame pair ratios.</p>
<div class="grid">''' + "".join(cards) + '''</div></main></html>''', encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "selected_point_count": selected_count,
        "identical_initial_queries": True, "methods": methods,
    }, indent=2) + "\n", encoding="utf-8")
    return page


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "segmentation", "pointmaps", "cotracker-dir",
                 "tapip3d-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--plotly-js", type=Path,
                        default=Path(__file__).resolve().parents[3] / "assets/replay/plotly.min.js")
    parser.add_argument("--max-cloud-points", type=int, default=900)
    args = parser.parse_args(argv)
    if args.max_cloud_points < 0:
        parser.error("--max-cloud-points must be nonnegative")
    print(export_link7_pair_replay(
        video_path=args.video, segmentation_path=args.segmentation,
        pointmaps_path=args.pointmaps, cotracker_dir=args.cotracker_dir,
        tapip3d_dir=args.tapip3d_dir, output_dir=args.output_dir,
        plotly_js=args.plotly_js, max_cloud_points=args.max_cloud_points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
