"""Audit completed native clouds and write a combined actual-results snapshot."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = (_workspace_root())
sys.path.insert(0, str(ROOT))
from infrastructure.shared.experimental.simple3d.simple3d_pipeline import distribution, write_json, write_csv
from infrastructure.shared.experimental.simple3d.simple3d_layout import result_path


def cloud_audit(observed):
    rows = []
    for path in observed.glob("*/geometry.json"):
        geometry = json.loads(path.read_text())
        if geometry["status"] == "running":
            continue
        case = path.parent
        depths = poses = intrinsic = None
        native = case / "reconstruction.npz"
        if native.is_file():
            with np.load(native, allow_pickle=False) as z:
                depths, poses, intrinsic = z["depths"], z["cam_c2w"], z["intrinsic"]
        for i, t in enumerate(geometry["source"]["frames"]):
            info = geometry.get("clouds", {}).get(str(t), {})
            record = {"video_id": geometry["source"]["video_id"], "frame_id": t,
                      "megasam_status": geometry["status"], "refinement": geometry.get("refinement"),
                      "cloud_status": info.get("status", "unavailable"),
                      "failure_reason": info.get("failure_reason", geometry.get("failure_reason")),
                      "original_point_count": info.get("original_point_count"),
                      "valid_point_count": info.get("valid_point_count"),
                      "final_sampled_point_count": info.get("final_sampled_point_count", 0),
                      "centroid": info.get("centroid"), "scale_divisor": info.get("scale_divisor"),
                      "normalization_applied": info.get("status") == "complete",
                      "rigid_rotation_applied": False, "registration": None}
            # Failed small clouds still have a measurable centroid in native units.
            # Audit them from this run's saved depths; no geometry rerun or scoring.
            if depths is not None:
                frame = case / "frames" / f"{t:06d}"
                mask = cv2.imread(str(frame / "mask.png"), cv2.IMREAD_GRAYSCALE)
                record["original_mask_pixel_count"] = int((mask > 0).sum())
                h, w = depths[i].shape
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
                yy, xx = np.where(mask)
                d = depths[i, yy, xx]
                record["original_point_count"] = len(d)
                valid = np.isfinite(d) & (d > 0)
                record["valid_depth_point_count"] = int(valid.sum())
                if info.get("status") != "complete" and valid.any():
                    k = intrinsic
                    fx, fy, cx, cy = k[:4] if k.ndim == 1 else (k[0, 0], k[1, 1], k[0, 2], k[1, 2])
                    yy, xx, d = yy[valid], xx[valid], d[valid]
                    camera = np.column_stack(((xx - cx) * d / fx, (yy - cy) * d / fy, d))
                    xyz = camera @ poses[i, :3, :3].T + poses[i, :3, 3]
                    finite = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
                    record["valid_point_count"] = int(finite.sum())
                    record["centroid"] = xyz[finite].mean(axis=0).tolist() if finite.any() else None
            rows.append(record)
        write_json(case / "cloud_audit.json", [r for r in rows if r["video_id"] == case.name])
    return rows


def finalize(run_id, output_root):
    output = result_path(run_id, "metadata", output_root)
    output.mkdir(parents=True, exist_ok=True)
    roots = {role: result_path(run_id, role, output_root, existing=True)
             for role in ("object", "cad", "first_frame")}
    summaries = {k: json.loads((r / "summary.json").read_text()) for k, r in roots.items() if (r / "summary.json").is_file()}
    audit = {kind: cloud_audit(roots[kind] / "observed") for kind in ("object", "cad")}
    for kind, records in audit.items():
        write_json(output / f"{kind}_cloud_audit.json", records)
        if records:
            fields = sorted(set().union(*(r.keys() for r in records)))
            write_csv(output / f"{kind}_cloud_audit.csv", records, fields)
    table_path = roots["cad"] / "combined_comparison.json"
    table = json.loads(table_path.read_text()) if table_path.is_file() else []
    matched = [r for r in table if r["cad_reference_score"] is not None and r["first_frame_reference_score"] is not None]
    cad = [r["cad_reference_score"] for r in matched]
    first = [r["first_frame_reference_score"] for r in matched]
    paired = {"matched_later_frame_count": len(matched), "cad_distribution": distribution(cad),
              "first_frame_distribution": distribution(first),
              "cad_minus_first_frame_distribution": distribution([c - f for c, f in zip(cad, first)])}
    diagnostics = {}
    for kind, records in audit.items():
        good = [r for r in records if r["cloud_status"] == "complete"]
        diagnostics[kind] = {"audited_cloud_count": len(records),
                             "successful_cloud_count": len(good),
                             "valid_point_count_distribution": distribution([r["valid_point_count"] for r in records if r["valid_point_count"] is not None]),
                             "sampled_point_count_distribution": distribution([r["final_sampled_point_count"] for r in good]),
                             "clouds_below_native_4096_feature_centers": sum(r["final_sampled_point_count"] < 4096 for r in good),
                             "cloud_failure_reasons": dict(Counter(r["failure_reason"] for r in records if r["cloud_status"] != "complete")),
                             "normalization_scale_divisor_distribution": distribution([r["scale_divisor"] for r in good])}
    result = {"status": "running" if any(s["status"] == "running" for s in summaries.values()) or len(summaries) < 3 else "terminal",
              "run_id": run_id, "experiments": summaries, "paired_reference_comparison": paired,
              "geometry_diagnostics": diagnostics,
              "interpretation_limits": ["Raw descriptor distances are not calibrated deformation probabilities.",
                                        "Complete CAD and partial, noisy reconstructed surfaces have different visible geometry and neighborhood density.",
                                        "Mask/occlusion/viewpoint changes can alter local geometry without true deformation.",
                                        "Per-cloud unit-radius normalization removes global size changes and magnifies partial-mask differences.",
                                        "Native FPS repeats centers in clouds below its fixed group sizes; no real points were upsampled.",
                                        "Lower CAD or first-frame distance alone does not establish which reference detects deformation better."]}
    write_json(output / "summary.json", result)
    print("RESULTS_SNAPSHOT", json.dumps({"status": result["status"], "matched_pairs": len(matched),
                                        "scores": {k: s["successful_simple3d_scores"] for k, s in summaries.items()}}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-root", type=Path, default=(_workspace_root() / 'results'))
    p.add_argument("--watch", action="store_true", help="wait for the tmux batch exit marker before final audit")
    a = p.parse_args()
    if a.watch:
        marker = result_path(a.run_id, "metadata", a.output_root) / "execution" / f"simple3d-{a.run_id}.exit"
        legacy_marker = a.output_root / f"simple3d-{a.run_id}.exit"
        if not marker.is_file() and legacy_marker.is_file():
            marker = legacy_marker
        while not marker.is_file():
            time.sleep(30)
    result = finalize(a.run_id, a.output_root)
    if a.watch:
        from infrastructure.shared.experimental.simple3d.verify_simple3d_outputs import verify
        obj = result_path(a.run_id, "object", a.output_root, existing=True)
        cad = result_path(a.run_id, "cad", a.output_root, existing=True)
        first = result_path(a.run_id, "first_frame", a.output_root, existing=True)
        checks = {"object": verify(obj), "cad": verify(cad, first), "first_frame": verify(first, cad)}
        write_json(result_path(a.run_id, "metadata", a.output_root) / "finalization.json",
                   {"status": "passed", "run_exit": marker.read_text().strip(),
                    "artifact_verification": {k: v["verified_successful_comparisons"] for k, v in checks.items()}})
        print("FINALIZATION_PASSED", flush=True)
