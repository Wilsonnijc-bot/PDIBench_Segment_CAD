"""Check completed score/cloud artifacts, original pixel correspondence and mode parity."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = (_workspace_root())
sys.path.insert(0, str(ROOT))
from infrastructure.shared.experimental.simple3d.simple3d_pipeline import write_json, sha
from infrastructure.shared.experimental.simple3d.simple3d_layout import resolve_recorded_path


def verify(root, other=None):
    root = Path(root)
    run_id = json.loads((root / "run.json").read_text())["run_id"]
    rows = json.loads((root / "comparisons.json").read_text())
    checked, verified_clouds = 0, set()
    other_rows = {(r["video_id"], r["test_frame_id"]): r for r in
                  json.loads((other / "comparisons.json").read_text())} if other else {}
    hashes = {}
    for row in rows:
        if row["simple3d_status"] != "complete":
            assert row["scalar_anomaly_score"] is None and row["failure_reason"]
            continue
        score_path = resolve_recorded_path(row["point_anomaly_scores_path"], run_id)
        point_path = resolve_recorded_path(row["test_point_cloud_path"], run_id)
        scores = np.load(score_path, allow_pickle=False)
        with np.load(point_path, allow_pickle=False) as z:
            xyz, raw, pixels = z["xyz"], z["raw_xyz"], z["pixels_yx"]
        assert len(scores) == len(xyz) == row["sampled_test_point_count"]
        assert np.isfinite(scores).all() and np.isfinite(xyz).all()
        assert np.isclose(np.sort(scores)[-80:].mean(), row["scalar_anomaly_score"], rtol=2e-6, atol=1e-6)
        if str(point_path) not in verified_clouds:
            info = json.loads(point_path.with_suffix(".json").read_text())
            centroid, radius = np.array(info["centroid"]), info["scale_divisor"]
            assert np.allclose((raw - centroid) / radius, xyz, rtol=5e-4, atol=1e-4)
            mask = cv2.imread(str(resolve_recorded_path(row["mask_path"], run_id)), cv2.IMREAD_GRAYSCALE)
            h, w = info["pixel_grid_hw"]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            assert mask[pixels[:, 0], pixels[:, 1]].all()
            observed = point_path.parents[2]
            with np.load(observed / "reconstruction.npz", allow_pickle=False) as z:
                frame = z["frame_ids"].tolist().index(row["test_frame_id"])
                depth, pose, k = z["depths"][frame], z["cam_c2w"][frame], z["intrinsic"]
            if k.ndim == 1:
                fx, fy, cx, cy = k[:4]
            else:
                fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            yy, xx = pixels.T
            d = depth[yy, xx]
            camera = np.column_stack(((xx - cx) * d / fx, (yy - cy) * d / fy, d))
            world = camera @ pose[:3, :3].T + pose[:3, 3]
            assert (d > 0).all() and np.allclose(world, raw, rtol=1e-5, atol=1e-5)
            verified_clouds.add(str(point_path))
            hashes[str(point_path)] = sha(point_path)
        match = other_rows.get((row["video_id"], row["test_frame_id"]))
        if match:
            assert match["test_point_cloud_path"] == row["test_point_cloud_path"]
        checked += 1
    assert checked, "no successful score artifacts available to verify"
    result = {"status": "passed", "verified_successful_comparisons": checked,
              "verified_test_cloud_count": len(verified_clouds), "test_cloud_sha256": hashes,
              "checks": ["finite Nx3/scores/counts", "official top80 scalar", "cloud centering/scale",
                         "saved mask pixel correspondence", "native depth/intrinsic/pose back-projection",
                         "identical observed cloud paths across reference modes" if other else "single reference mode"]}
    write_json(root / "artifact_verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--other-reference", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.other_reference), indent=2))
