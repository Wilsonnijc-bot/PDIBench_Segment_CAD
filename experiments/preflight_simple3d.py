"""Validate native method, existing infrastructure, and all prepared source paths."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PDI-Bench-edited/src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--cad", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    from pdi_eval.perception.mega_sam_wrapper import MegaSamWrapper
    from pdi_eval.object_deformation_wrapper.reference_visible_pixels import mask_bbox, rgba_crop, write_png
    from pdi_eval.experiment.runner import clean_megasam_intermediates
    from experiments.simple3d_adapter import build_simple3d_reference, score_simple3d, provenance
    from experiments.simple3d_pipeline import sha, write_json
    assert torch.cuda.is_available()
    assert float(torch.ones(2, device="cuda").sum()) == 2
    wrapper = MegaSamWrapper()
    required = [Path(wrapper.da_ckpt), Path(wrapper.megasam_weights), Path(wrapper.raft_weights), a.cad]
    for file in required:
        assert file.is_file() and file.stat().st_size, str(file)
    manifests = {}
    for kind in ("object", "link5"):
        path = a.inputs / f"{kind}_manifest.json"
        manifest = json.loads(path.read_text())
        assert manifest["kind"] == kind
        count = 0
        for row in manifest["rows"]:
            if row["status"] != "ready":
                continue
            assert sha(row["video_path"]) == row["video_sha256"], row["video_path"]
            mask = a.inputs / row["mask_path"]
            assert sha(mask) == row["selected_masks_sha256"], str(mask)
            with np.load(mask, allow_pickle=False) as z:
                assert z["frame_ids"].tolist() == row["frames"]
                assert len(z["masks"]) == 10 and all(m.any() for m in z["masks"])
            count += 1
        manifests[kind] = {"manifest_sha256": sha(path), "verified_videos": count}
    # Exercises actual CUDA FPS, KNN, FPFH, MSND, LFSA, coreset and scoring, including
    # native repeated-center behavior on a finite cloud smaller than 4096 points.
    x = np.random.default_rng(1).normal(size=(256, 3)).astype(np.float32)
    reference = build_simple3d_reference(x)
    points, scalar = score_simple3d(reference, x)
    assert points.shape == (256,) and np.isfinite(points).all() and np.isfinite(scalar)
    result = {"status": "passed", "container_hostname": socket.gethostname(),
              "working_directory": str(ROOT), "python": sys.executable,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "manifests": manifests,
              "cad_sha256": sha(a.cad), "required_paths": [str(f) for f in required],
              "method": provenance(), "native_smoke_score": scalar,
              "packages": {n: importlib.metadata.version(n) for n in
                           ("numpy", "open3d", "trimesh", "pycollada", "tifffile", "pointnet2_ops", "KNN_CUDA")}}
    write_json(a.output, result)
    print("PREFLIGHT_PASSED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
