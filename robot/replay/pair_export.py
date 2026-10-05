"""Refresh V1 scoring-pair replay bundles from saved experiment artifacts."""

from __future__ import annotations

from infrastructure.pdibench.layout import root as _workspace_root

import json
from pathlib import Path

from infrastructure.shared.contracts.contracts import ROOT, read_selection
from robot.workflows.spec import ExperimentSpec


def refresh_pair_replays(spec: ExperimentSpec) -> dict[str, object]:
    from infrastructure.shared.replay.rigidity_replay import export_v1_replay

    selection = read_selection(spec.output_root)
    completed: dict[str, dict[str, int]] = {}
    skipped: list[str] = []
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        case = spec.output_root / "cases" / sample
        status_path = case / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if status.get("state") != "complete":
            skipped.append(sample)
            continue
        mode = "exact-group"
        v1_metrics = case / "v1" / "metrics.json"
        v1_report = json.loads(v1_metrics.read_text(encoding="utf-8"))
        v1_pages = export_v1_replay(
            v1_metrics,
            case / "v1" / f"cotracker_{mode}.npz",
            case / "v1" / "segmentation.npz",
            Path(v1_report["geometry"]["cache_path"]),
            case / f"{sample.replace('.', '_')}.mp4",
            case / "v1" / "replay" / f"interactive_{mode}",
            mode=mode,
            plotly_js=(_workspace_root() / 'infrastructure/shared/replay/assets/plotly.min.js'),
        )
        completed[sample] = {"v1": len(v1_pages)}
    return {"pair_replay_pages": completed, "skipped_incomplete": skipped}
