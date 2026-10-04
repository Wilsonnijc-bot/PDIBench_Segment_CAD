#!/usr/bin/env python3
"""Correlate existing Simple3D video scores with the frozen selected-45 labels.

Requires numpy, scipy and openpyxl. Reads the workbook without modifying it.
Example:
    python scripts/analyze_simple3d_correlation.py --run-id 20261003-run2 \
        --output results/simple3d-20261003-run2/correlation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.simple3d_layout import result_path
DATASETS = {"LVP": "LVP_ROBOWM", "Cosmos2.5": "COSMOS2.5", "Cosmos3 seed101": "COSMOS3"}
EXPERIMENTS = {
    "object": ("object-deformation", "AF", "Object deformation (0/1)2", 9),
    "link5_cad": ("link5-cad-reference", "AB", "Forearm deformation", 10),
    "link5_first_frame": ("link5-firstframe-reference", "AB", "Forearm deformation", 9),
}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_labels(workbook_path, selection_path):
    """Match dataset/number and validate row identity against the existing selection."""
    selection = read_json(selection_path)["videos"]
    selected = {f'{v["dataset"]}_{int(v["video_number"]):04d}': v for v in selection}
    if len(selection) != 45 or len(selected) != 45:
        raise ValueError("The existing frozen selection must have 45 unique videos")
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        sheet = workbook["Selected 45"]
        if sheet["AB1"].value != "Forearm deformation" or sheet["AF1"].value != "Object deformation (0/1)2":
            raise ValueError("The requested AB/AF label columns have different headers")
        if sheet["A1"].value != "Matched number" or sheet["C1"].value != "Dataset":
            raise ValueError("Workbook video identification headers have changed")
        labels = {}
        for row_number, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
            if all(v is None for v in values):
                continue
            dataset = DATASETS[values[2]]
            number = int(values[0])
            video_id = f"{dataset}_{number:04d}"
            if video_id in labels or video_id not in selected:
                raise ValueError(f"Duplicate or unmatched workbook video: {video_id}")
            frozen = selected[video_id]
            if frozen["workbook_row"] != row_number or frozen["video_id"] != values[4]:
                raise ValueError(f"Workbook identity/row differs from frozen selection: {video_id}")
            label_values = {"AB": float(values[27]), "AF": float(values[31])}
            if any(v not in (0.0, 0.5, 1.0) for v in label_values.values()):
                raise ValueError(f"Unsupported human label: {video_id}")
            labels[video_id] = {
                "dataset": dataset, "matched_number": number, "workbook_row": row_number,
                "workbook_video_id": values[4], "task": frozen["task"], **label_values,
            }
        if set(labels) != set(selected):
            raise ValueError("Workbook and frozen selection are not the same 45 videos")
        return labels
    finally:
        workbook.close()


def distribution(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "std": float(np.std(values)), "min": min(values), "max": max(values)}


def correlation(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    # Independently verify Pearson's centered-dot formula and Spearman's tied ranks.
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    a, b = xx - xx.mean(), yy - yy.mean()
    manual = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
    rank_r = float(stats.pearsonr(stats.rankdata(xx), stats.rankdata(yy)).statistic)
    if not math.isclose(manual, float(pearson.statistic), abs_tol=1e-12):
        raise ValueError("Pearson independent verification failed")
    if not math.isclose(rank_r, float(spearman.statistic), abs_tol=1e-12):
        raise ValueError("Spearman tied-rank verification failed")
    return {"pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue)}


def auroc(x, y):
    pos = [s for s, label in zip(x, y) if label == 1]
    neg = [s for s, label in zip(x, y) if label == 0]
    if not pos or not neg:
        return None
    pairwise = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    ranks = stats.rankdata(pos + neg)
    ranked = (sum(ranks[:len(pos)]) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    if not math.isclose(pairwise, float(ranked), abs_tol=1e-12):
        raise ValueError("AUROC independent verification failed")
    return pairwise


def metric(experiment, rows, *, subset="all_scored", aggregation="mean_score", labels="ordinal", dataset="all"):
    included = [r for r in rows if r[aggregation] is not None
                and (subset != "complete_videos" or r["coverage_status"] == "complete")
                and (labels != "binary" or r["human_label"] in (0, 1))
                and (dataset == "all" or r["dataset"] == dataset)]
    x = [r[aggregation] for r in included]
    y = [r["human_label"] for r in included]
    return {"experiment": experiment, "subset": subset, "aggregation": aggregation,
            "label_policy": labels, "dataset": dataset, "n_videos": len(x),
            "label_0_count": y.count(0), "label_0_5_count": y.count(0.5), "label_1_count": y.count(1),
            **correlation(x, y), "binary_auroc": auroc(x, y) if labels == "binary" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261003-run2")
    parser.add_argument("--workbook", type=Path, default=ROOT / "selected_45_matched_videos_styledv2.xlsx")
    parser.add_argument("--selection", type=Path, default=ROOT / "results/link5-only-selected45-updated-mask-20260929/selection.json")
    parser.add_argument("--output", type=Path, help="defaults to this run's correlation/; must not exist")
    args = parser.parse_args()
    args.output = args.output or result_path(args.run_id, "correlation")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    labels = read_labels(args.workbook, args.selection)
    provenance = [args.workbook.resolve(), args.selection.resolve(), Path(__file__).resolve()]
    video_rows, comparison_rows, metrics, coverage = [], [], [], {}
    all_comparisons = {}
    for name, (directory, column, header, expected) in EXPERIMENTS.items():
        role = {"object": "object", "link5_cad": "cad", "link5_first_frame": "first_frame"}[name]
        result_dir = result_path(args.run_id, role, existing=True)
        comparisons_path, summary_path = result_dir / "comparisons.json", result_dir / "summary.json"
        manifest_path = result_path(args.run_id, "inputs", existing=True) / ("object_manifest.json" if name == "object" else "link5_manifest.json")
        provenance.extend([comparisons_path, summary_path, manifest_path])
        comparisons, summary = read_json(comparisons_path), read_json(summary_path)
        manifest = {r["video_id"]: r for r in read_json(manifest_path)["rows"]}
        if set(manifest) != set(labels):
            raise ValueError(f"{name}: input manifest differs from the frozen 45 videos")
        if len(comparisons) != summary["attempted_comparisons"]:
            raise ValueError(f"{name}: attempted comparison count differs from summary")
        grouped = defaultdict(list)
        seen = set()
        for row in comparisons:
            video_id = row["video_id"]
            key = (video_id, row["test_frame_id"])
            if key in seen or video_id not in labels:
                raise ValueError(f"{name}: duplicate/unmatched comparison {key}")
            seen.add(key)
            scalar = row["scalar_anomaly_score"]
            complete = row["simple3d_status"] == "complete"
            if complete != (scalar is not None) or (complete and not math.isfinite(scalar)):
                raise ValueError(f"{name}: invalid status or scalar at {key}")
            if complete and row["megasam_status"] != "complete":
                raise ValueError(f"{name}: scored comparison without completed reconstruction: {key}")
            grouped[video_id].append(row)
            comparison_rows.append({"experiment": name, "human_label": labels[video_id][column],
                                    "workbook_cell": f'{column}{labels[video_id]["workbook_row"]}', **row})
        if sum(r["simple3d_status"] == "complete" for r in comparisons) != summary["successful_simple3d_scores"]:
            raise ValueError(f"{name}: successful score count differs from summary")
        all_comparisons[name] = grouped
        experiment_rows = []
        for video_id, label in labels.items():
            attempted = grouped[video_id]
            inputs = manifest[video_id]
            if inputs["status"] == "ready":
                frames = inputs["frames"] if name == "link5_cad" else inputs["frames"][1:]
                if len(frames) != expected or set(frames) != {r["test_frame_id"] for r in attempted}:
                    raise ValueError(f"{name}: comparison frames differ from original selection: {video_id}")
            elif attempted:
                raise ValueError(f"{name}: unavailable input unexpectedly has comparisons: {video_id}")
            scores = [r["scalar_anomaly_score"] for r in attempted if r["simple3d_status"] == "complete"]
            status = ("complete" if len(scores) == expected else "partial" if scores else
                      "all_scores_failed" if attempted else "input_unavailable")
            reasons = sorted({r["failure_reason"] for r in attempted if r["failure_reason"]})
            if not attempted:
                reasons.append(json.dumps({k: v for k, v in inputs.items() if k in ("status", "reason", "failure_reason")}, sort_keys=True))
            r = {"experiment": name, "video_id": video_id, "dataset": label["dataset"],
                 "matched_number": label["matched_number"], "task": label["task"],
                 "workbook_sheet": "Selected 45", "workbook_row": label["workbook_row"],
                 "label_column": column, "label_header": header, "human_label": label[column],
                 "expected_comparisons": expected, "attempted_comparisons": len(attempted),
                 "successful_scores": len(scores), "failed_comparisons": len(attempted) - len(scores),
                 "mean_score": math.fsum(scores) / len(scores) if scores else None,
                 "max_score": max(scores) if scores else None,
                 "coverage_status": status, "failure_reasons": " | ".join(reasons)}
            experiment_rows.append(r)
        video_rows.extend(experiment_rows)
        coverage[name] = {
            "total_selected_videos": len(labels), "label_column": column, "label_header": header,
            "status_counts": dict(Counter(r["coverage_status"] for r in experiment_rows)),
            "missing_score_video_ids": [r["video_id"] for r in experiment_rows if r["mean_score"] is None],
            "score_distribution_by_label": {str(v): distribution([r["mean_score"] for r in experiment_rows
                if r["human_label"] == v and r["mean_score"] is not None]) for v in (0.0, 0.5, 1.0)},
        }
        for label_policy in ("ordinal", "binary"):
            metrics.append(metric(name, experiment_rows, labels=label_policy))
            metrics.append(metric(name, experiment_rows, labels=label_policy, aggregation="max_score"))
            metrics.append(metric(name, experiment_rows, labels=label_policy, subset="complete_videos"))
            for dataset in DATASETS.values():
                metrics.append(metric(name, experiment_rows, labels=label_policy, dataset=dataset))

    # CAD sensitivity uses exactly the same nine observed test clouds as first-frame mode.
    matched_rows = []
    for video_id, label in labels.items():
        cad = {r["test_frame_id"]: r for r in all_comparisons["link5_cad"][video_id]}
        first = all_comparisons["link5_first_frame"][video_id]
        if len(first) != 9 or len({r["reference_frame_id"] for r in first}) != 1:
            raise ValueError(f"Invalid first-frame comparisons: {video_id}")
        reference_id = first[0]["reference_frame_id"]
        if set(cad) != {r["test_frame_id"] for r in first} | {reference_id}:
            raise ValueError(f"CAD/first-frame test selections differ: {video_id}")
        for r in first:
            if r["test_point_cloud_path"] != cad[r["test_frame_id"]]["test_point_cloud_path"]:
                raise ValueError(f"CAD/first-frame observed geometry differs: {video_id}")
            if r["reference_point_cloud_path"] != cad[reference_id]["test_point_cloud_path"]:
                raise ValueError(f"First-frame reference geometry differs: {video_id}")
        scores = [cad[r["test_frame_id"]]["scalar_anomaly_score"] for r in first]
        matched_rows.append({"experiment": "link5_cad_matched_later_frames", "video_id": video_id,
                             "dataset": label["dataset"], "human_label": label["AB"],
                             "mean_score": math.fsum(scores) / len(scores), "max_score": max(scores),
                             "coverage_status": "complete"})
    for policy in ("ordinal", "binary"):
        metrics.append(metric("link5_cad_matched_later_frames", matched_rows, labels=policy))
        metrics.append(metric("link5_cad_matched_later_frames", matched_rows, labels=policy, aggregation="max_score"))

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
        "method": {
            "statistical_unit": "video", "primary_score": "mean of successful Simple3D frame comparison scalar scores",
            "primary_labels": "ordinal labels 0, 0.5, 1; 0.5 retained as provided",
            "binary_sensitivity": "exclude 0.5 labels; higher Simple3D score predicts label 1",
            "missing_scores": "never imputed; all-scored analysis includes partial videos; complete-video sensitivity is separate",
            "other_sensitivity": "maximum scalar score and CAD restricted to the identical nine later observed clouds",
            "p_values": "two-sided nominal SciPy p-values; no correction for multiple analyses or dependence among matched tasks/generators",
            "interpretation": "association only; reference modes use the same observed link5 geometry; no detector or score direction tuning",
        },
        "coverage": coverage, "metrics": metrics,
        "source_files": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in dict.fromkeys(provenance)],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "openpyxl": openpyxl.__version__},
        "verification": {"workbook_matches_frozen_selection": True, "all_45_workbook_rows_validated": True,
                         "comparison_counts_and_frame_selections_verified": True,
                         "cad_and_first_frame_share_identical_test_geometry": True,
                         "pearson_spearman_and_auroc_independent_formula_checks": True},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "video_scores_and_labels.csv", video_rows)
    write_csv(args.output / "comparisons_with_labels.csv", comparison_rows)
    write_csv(args.output / "matched_later_frame_cad_scores.csv", matched_rows)
    write_csv(args.output / "metrics.csv", metrics)
    write_json(args.output / "summary.json", summary)
    print(json.dumps([m for m in metrics if m["subset"] == "all_scored" and m["aggregation"] == "mean_score"
                      and m["dataset"] == "all"], indent=2))
    print(f"Saved analysis to {args.output.resolve()}")


if __name__ == "__main__":
    main()
