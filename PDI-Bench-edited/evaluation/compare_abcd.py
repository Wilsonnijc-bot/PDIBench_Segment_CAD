#!/usr/bin/env python3
"""Compare completed A/B/C/D runs and write verification artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdi_eval.verification import (  # noqa: E402
    compare_metrics,
    compare_track_groups,
    load_track_group,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def original_tracks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "tracks": np.asarray(archive["foreground_tracks"]),
            "visibility": np.asarray(archive["foreground_visibility"]),
            "queries": np.asarray(archive["foreground_queries"]),
        }


def object_report(run_dir: Path, mode: str, name: str) -> dict:
    return read_json(run_dir / "metrics.json")["modes"][mode]["objects"][name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--selected-object", default="link1")
    args = parser.parse_args()
    root = args.verification_dir.resolve()
    name = args.selected_object
    run_a = root / "run_A_original_single"
    run_b = root / "run_B_edited_joint_single"
    run_c = root / "run_C_edited_exact_single"
    run_d = root / "run_D_edited_exact_seven"
    a_metrics = read_json(run_a / "metrics.json")
    b_metrics = object_report(run_b, "joint-query", name)
    c_metrics = object_report(run_c, "exact-group", name)
    d_metrics = object_report(run_d, "exact-group", name)
    a_tracks = original_tracks(run_a / "cotracker_original.npz")
    b_tracks = load_track_group(run_b / "cotracker_joint-query.npz", name)
    c_tracks = load_track_group(run_c / "cotracker_exact-group.npz", name)
    d_tracks = load_track_group(run_d / "cotracker_exact-group.npz", name)
    comparisons = {
        "A_vs_B": {
            "metrics": compare_metrics(a_metrics, b_metrics),
            "tracking": compare_track_groups(a_tracks, b_tracks),
        },
        "B_vs_C": {
            "metrics": compare_metrics(b_metrics, c_metrics),
            "tracking": compare_track_groups(b_tracks, c_tracks),
        },
        "C_vs_D": {
            "metrics": compare_metrics(c_metrics, d_metrics),
            "tracking": compare_track_groups(c_tracks, d_tracks),
        },
        "A_vs_D": {
            "metrics": compare_metrics(a_metrics, d_metrics),
            "tracking": compare_track_groups(a_tracks, d_tracks),
        },
    }
    comparison_dir = root / "comparisons"
    for label, value in comparisons.items():
        write_json(comparison_dir / f"{label}.json", value)
    c_d = comparisons["C_vs_D"]
    a_b = comparisons["A_vs_B"]
    gates = {
        "a_b_grade_equal": not a_b["metrics"]["grade_changed"],
        "c_d_queries_exact": c_d["tracking"]["queries_exact"],
        "c_d_mean_track_error_le_0_1px": (
            c_d["tracking"]["mean_track_l2_pixels"] is not None
            and c_d["tracking"]["mean_track_l2_pixels"] <= 0.1
        ),
        "c_d_max_track_error_le_0_5px": (
            c_d["tracking"]["maximum_track_l2_pixels"] is not None
            and c_d["tracking"]["maximum_track_l2_pixels"] <= 0.5
        ),
        "c_d_visibility_at_least_0_999": (
            c_d["tracking"]["visibility_agreement"] is not None
            and c_d["tracking"]["visibility_agreement"] >= 0.999
        ),
    }
    report = [
        "# A/B/C/D Verification Report",
        "",
        f"Selected object: `{name}`",
        "",
        "## Acceptance Gates",
        "",
        "| Gate | Result |",
        "| --- | --- |",
    ]
    report.extend(
        f"| `{gate}` | {'PASS' if passed else 'FAIL'} |"
        for gate, passed in gates.items()
    )
    report.extend(["", "## Comparisons", ""])
    for label, value in comparisons.items():
        metrics = value["metrics"]
        tracking = value["tracking"]
        report.extend(
            [
                f"### {label.replace('_', ' ')}",
                "",
                f"- PDI delta: `{metrics['right_minus_left']['pdi_score']:.8f}`",
                f"- Grade changed: `{metrics['grade_changed']}`",
                f"- Common retained queries: `{tracking['common_query_count']}`",
                f"- Mean track error (px): `{tracking['mean_track_l2_pixels']}`",
                f"- Maximum track error (px): `{tracking['maximum_track_l2_pixels']}`",
                f"- Visibility agreement: `{tracking['visibility_agreement']}`",
                "",
            ]
        )
    report.extend(
        [
            "## Status",
            "",
            "`VERIFIED`" if all(gates.values()) else "`UNVERIFIED: one or more gates failed`",
            "",
        ]
    )
    (root / "VERIFICATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(root / "verification_summary.json", {"gates": gates, "comparisons": comparisons})
    print(json.dumps({"gates": gates, "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
