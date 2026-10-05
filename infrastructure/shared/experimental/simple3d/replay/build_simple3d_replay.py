#!/usr/bin/env python3
"""Export saved Simple3D evidence into an offline, interactive point-cloud replay.

No inference is run. Point coordinates, order and scores are preserved as float32
binary arrays. Per-case scripts permit file:// viewing without fetch/CORS issues.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = (_workspace_root())
sys.path.insert(0, str(ROOT))
from infrastructure.shared.experimental.simple3d.simple3d_layout import result_path, resolve_recorded_path


def read(path):
    return json.loads(path.read_text())


def packed(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def local(path, run_id):
    p = resolve_recorded_path(path, run_id)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261003-run2")
    parser.add_argument("--output", type=Path, help="defaults to this run's replay/; existing outputs are never overwritten")
    args = parser.parse_args()
    output = (args.output or result_path(args.run_id, "replay")).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = {"object": result_path(args.run_id, "object", existing=True),
             "cad": result_path(args.run_id, "cad", existing=True),
             "first": result_path(args.run_id, "first_frame", existing=True)}
    sources, counts, point_values = {}, defaultdict(int), {"link5": [], "object": []}

    def provenance(p):
        p = p.resolve()
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

    def href(p):
        return os.path.relpath(p.resolve(), output)

    def cloud(path):
        if not path.exists():
            return None
        provenance(path)
        info = path.with_suffix(".json")
        provenance(info)
        with np.load(path, allow_pickle=False) as data:
            xyz = data["xyz"]
            assert xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all()
            result = {"count": len(xyz), "xyz": packed(xyz, "<f4"), "metadata": read(info), "source": href(path)}
            if "pixels_yx" in data:
                pixels = data["pixels_yx"]
                assert pixels.shape == (len(xyz), 2) and pixels.min() >= 0 and pixels.max() < 65536
                result["pixels"] = packed(pixels, "<u2")
            if "rgb" in data:
                rgb = np.round(np.clip(data["rgb"], 0, 1) * 255)
                result["rgb"] = packed(rgb, "u1")
        counts["exported_clouds"] += 1
        counts["exported_points"] += len(xyz)
        return result

    comparisons = {}
    expected_successes, expected_failures = 0, 0
    for kind, directory in paths.items():
        source = directory / "comparisons.json"
        provenance(source)
        records = read(source)
        comparisons[kind] = {(r["video_id"], r["test_frame_id"]): r for r in records}
        if len(records) != len(comparisons[kind]):
            raise ValueError(f"Duplicate comparisons: {source}")
        expected_successes += sum(r["simple3d_status"] == "complete" for r in records)
        expected_failures += sum(r["simple3d_status"] != "complete" for r in records)
    manifests = {}
    for kind in ("object", "link5"):
        source = result_path(args.run_id, "inputs", existing=True) / f"{kind}_manifest.json"
        provenance(source)
        manifests[kind] = {r["video_id"]: r for r in read(source)["rows"]}
    selection_path = (_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/selection.json')
    provenance(selection_path)
    selection = read(selection_path)["videos"]
    label_path = result_path(args.run_id, "correlation") / "video_scores_and_labels.csv"
    labels = {}
    if label_path.exists():
        provenance(label_path)
        with label_path.open() as stream:
            labels = {(r["experiment"], r["video_id"]): r for r in csv.DictReader(stream)}
    case_data, index = [], []
    for v in selection:
        video_id = f'{v["dataset"]}_{int(v["video_number"]):04d}'
        payload = {"video_id": video_id, "targets": {}}
        item = {"video_id": video_id, "dataset": v["dataset"], "task": v["task"], "labels": {}}
        for target in ("link5", "object"):
            inp = manifests[target][video_id]
            frames, modes = [], (["cad", "first"] if target == "link5" else ["object"])
            for frame_id in inp.get("frames", []):
                geometry_root = paths["cad" if target == "link5" else "object"]
                frame_dir = geometry_root / f"observed/{video_id}/frames/{frame_id:06d}"
                entry = {"frame_id": frame_id, "cloud": cloud(frame_dir / "cloud.npz"), "comparisons": {}, "images": {}}
                info_path = frame_dir / "cloud.json"
                entry["cloud_metadata"] = read(info_path) if info_path.exists() else None
                for role, filename in (("frame", "frame.jpg"), ("crop", "crop.png"), ("mask", "mask.png")):
                    image = frame_dir / filename
                    if image.exists():
                        entry["images"][role] = href(image)
                for mode in modes:
                    r = comparisons[mode].get((video_id, frame_id))
                    if r is None:
                        entry["comparisons"][mode] = {"status": "reference", "scalar": None, "reason": None}
                        continue
                    comparison = {"status": r["simple3d_status"], "scalar": r["scalar_anomaly_score"],
                                  "reason": r["failure_reason"], "reference_frame_id": r["reference_frame_id"],
                                  "reference_type": r["reference_type"], "megasam_status": r["megasam_status"]}
                    if r["simple3d_status"] == "complete":
                        score_path = local(r["point_anomaly_scores_path"], args.run_id)
                        provenance(score_path)
                        scores = np.load(score_path, allow_pickle=False).reshape(-1)
                        c = entry["cloud"]
                        assert c and len(scores) == c["count"] and np.isfinite(scores).all()
                        assert np.isclose(np.sort(scores)[-80:].mean(), r["scalar_anomaly_score"], rtol=2e-6, atol=1e-6)
                        # Round-trip validation also verifies every score is assigned to its saved point index.
                        encoded = packed(scores, "<f4")
                        assert np.array_equal(np.frombuffer(base64.b64decode(encoded), dtype="<f4"), scores)
                        comparison.update(scores=encoded, score_source=href(score_path))
                        point_values[target].append(scores)
                        counts["successful_comparisons"] += 1
                    else:
                        counts["failed_comparisons"] += 1
                    evidence = paths[mode] / f"cases/{video_id}/frame_{frame_id:06d}_comparison.json"
                    if evidence.exists():
                        comparison["evidence"] = href(evidence)
                    entry["comparisons"][mode] = comparison
                frames.append(entry)
            payload["targets"][target] = {"status": inp["status"], "frames": frames,
                                          "input_reason": inp.get("reason", inp.get("failure_reason")),
                                          "input_details": {k: inp[k] for k in ("status", "error") if k in inp}}
            key = "object" if target == "object" else "link5_first_frame"
            if (key, video_id) in labels:
                lab = labels[key, video_id]
                item["labels"][target] = {"value": float(lab["human_label"]), "cell": f'{lab["label_column"]}{lab["workbook_row"]}'}
        case_data.append(payload)
        index.append(item)
    assert len(index) == 45 and counts["successful_comparisons"] == expected_successes and counts["failed_comparisons"] == expected_failures
    cad = cloud(paths["cad"] / "cad/cad_cloud.npz")
    manifest = {"run_id": args.run_id, "cases": index, "cad": cad,
                "heatmap_max": {k: float(np.percentile(np.concatenate(v), 99)) for k, v in point_values.items()},
                "heatmap_policy": "fixed 0–99th percentile range of all saved point scores per target; link5 CAD and first-frame share the same range; values above range saturate",
                "counts": dict(counts)}
    output.mkdir(parents=True)
    (output / "cases").mkdir()
    for payload in case_data:
        folder = output / "cases" / payload["video_id"]
        folder.mkdir()
        (folder / "replay.js").write_text(f'window.Simple3DReplayCases[{json.dumps(payload["video_id"])}]={json.dumps(payload, separators=(",", ":"), allow_nan=False)};\n')
    plotly = (_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/cases/LVP_ROBOWM_0001/v1_cotracker3/replay/interactive_exact-group/plotly.min.js')
    shutil.copy2(plotly, output / "plotly.min.js")
    template = (_workspace_root() / 'infrastructure/shared/experimental/simple3d/replay/simple3d_replay.html')
    provenance(template)
    provenance(_SOURCE_PATH)
    html = template.read_text().replace("__REPLAY_MANIFEST__", json.dumps(manifest, separators=(",", ":"), allow_nan=False))
    (output / "index.html").write_text(html)
    (output / "metadata").mkdir()
    verification = {"run_id": args.run_id, "counts": dict(counts), "case_count": len(index),
                    "point_export": "all sampled model-input points in original saved order; float32 XYZ and point scores encoded without rounding or visualization resampling",
                    "checks": ["finite cloud and scores", "point/score count parity", "all successful and failed source comparisons exported",
                               "official top80 scalar matches saved scores", "lossless point-score encoding round trip"],
                    "heatmap_policy": manifest["heatmap_policy"], "heatmap_max": manifest["heatmap_max"],
                    "source_sha256": sources}
    (output / "metadata/export.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"output": str(output / "index.html"), **dict(counts), "heatmap_max": manifest["heatmap_max"]}, indent=2))


if __name__ == "__main__":
    main()
