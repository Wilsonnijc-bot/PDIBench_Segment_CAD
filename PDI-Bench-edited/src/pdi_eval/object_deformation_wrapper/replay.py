"""Adapt a rigidity-only score to the existing V1 interactive replay exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdi_eval.utils.rigidity_replay import export_v1_replay

from .selection import sha256, write_json


def export_replay(score_path: Path, tracks_path: Path, segmentation_path: Path,
                  pointmaps_path: Path, video_path: Path, output: Path,
                  target_name: str) -> Path:
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if score["status"] != "complete" or score["rigidity_strategy"] != "V1 Strategy 1 (3D rigid pairwise ratios)":
        raise ValueError("interactive replay requires a completed 3D pairwise score")
    if score["input"]["video_sha256"] != sha256(video_path):
        raise ValueError("replay video differs from scored source")
    if score["input"]["segmentation_sha256"] != sha256(segmentation_path):
        raise ValueError("replay mask differs from scored mask")
    output.mkdir(parents=True, exist_ok=True)
    # The shared exporter reads this established V1 shape and independently
    # reconstructs the pair selection/history. PDI is never calculated here.
    adapter = output / "replay_input.json"
    write_json(adapter, {
        "schema_version": 1,
        "segmentation": {"object_names": ["task_object"]},
        "modes": {"exact-group": {"objects": {"task_object": {
            "status": "complete", "pdi_score": None,
            "breakdown": {
                "rigidity_strategy": "Strategy 1 (3D rigid pairwise ratios)",
                "volume_history": score["rigidity_history"],
            },
        }}}},
    })
    plotly = Path(__file__).resolve().parents[3] / "assets/replay/plotly.min.js"
    pages = export_v1_replay(
        adapter, tracks_path, segmentation_path, pointmaps_path, video_path,
        output, mode="exact-group", max_cloud_points=900,
        plotly_js=plotly, only_objects=("task_object",),
        tracker_label="CoTracker3", mask_label=target_name)
    if len(pages) != 1:
        raise ValueError("V1 exporter did not create one task-object replay")
    page = pages[0]
    html = page.read_text(encoding="utf-8")
    html = (html.replace("Whole-robot cloud", "Task-object cloud")
            .replace("whole robot", "task object")
            .replace("selected link", "task object")
            .replace("V1 Link 7 filter", "V1 task object"))
    page.write_text(html, encoding="utf-8")
    index = output / "index.html"
    index.write_text(index.read_text(encoding="utf-8").replace(
        "V1 rigidity pair replays", "Task-object rigidity replay"), encoding="utf-8")
    pair_path = output / "task_object_exact-group_pairs.json"
    pairs = json.loads(pair_path.read_text(encoding="utf-8"))
    if abs(float(pairs["final_rigidity_score"]) - float(score["rigidity_score"])) > 1e-10:
        raise ValueError("interactive replay score differs from rigidity.json")
    write_json(output / "replay.json", {
        "status": "complete", "target_object": target_name,
        "rigidity_score": score["rigidity_score"],
        "score_sha256": sha256(score_path),
        "segmentation_sha256": sha256(segmentation_path),
        "track_sha256": sha256(tracks_path),
        "geometry_sha256": sha256(pointmaps_path),
        "html": page.name, "evidence": pair_path.name,
    })
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("score", "tracks", "segmentation", "pointmaps", "video", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    args = parser.parse_args()
    page = export_replay(args.score, args.tracks, args.segmentation,
                         args.pointmaps, args.video, args.output, args.target_name)
    print(json.dumps({"status": "complete", "interactive_replay": str(page)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
