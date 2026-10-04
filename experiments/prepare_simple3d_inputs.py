"""Freeze the existing object selections and link5 mask replay into compact inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PDI-Bench-edited/src"))
sys.path.insert(0, str(ROOT))
from experiments.simple3d_layout import result_path
from pdi_eval.perception.segmentation_archive import frame_measurements
from pdi_eval.experiment.contracts import video_path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(output, video_root):
    output.mkdir(parents=True, exist_ok=False)
    object_run = ROOT / "results/anomalydino-object-crops-10frames-20261002"
    pair_manifest = json.loads((object_run / "metadata/pair_manifest.json").read_text())
    link_root = ROOT / "results/link5-only-selected45-updated-mask-20260929"
    frozen = json.loads((link_root / "selection.json").read_text())
    mappings = {f"{e['dataset']}_{e['video_number']}": e for e in frozen["videos"]}
    object_rows, link_rows = [], []
    for v in pair_manifest["videos"]:
        name = v["video_id"]
        crop = ROOT / "results" / pair_manifest["crop_root"] / name
        selection = json.loads((crop / "selection.json").read_text())
        frames = [r["frame"] for r in selection["selected_frames"]]
        if not frames:
            object_rows.append({"video_id": name, "status": "skipped", "reason": "no existing usable selected frames", "frames": [],
                                "prior_anomalydino_status": v["status"]})
            continue
        pair_frames = [r["frame"] for r in pair_manifest["pairs"] if r["video_id"] == name]
        if (v["status"] == "ready" and frames != pair_frames) or len(frames) != 10:
            raise ValueError(f"existing object selection differs from manifest: {name}")
        with np.load(crop / "masks.npz", allow_pickle=False) as z:
            masks = z["available_pure_object"][frames].astype(bool)
        if not all(m.any() for m in masks):
            raise ValueError(f"empty selected usable object mask: {name}")
        dest = output / "object" / name
        dest.mkdir(parents=True)
        np.savez_compressed(dest / "selected_masks.npz", masks=masks, frame_ids=frames)
        meta = json.loads((crop / "manifest.json").read_text())
        e = mappings[name]
        if meta["source_video_sha256"] != e["sha256"]:
            raise ValueError(f"object and frozen video identities differ: {name}")
        object_rows.append({"video_id": name, "status": "ready", "frames": frames,
                            "prior_anomalydino_status": v["status"],
                            "video_path": str(video_path(video_root, e)), "video_sha256": e["sha256"],
                            "mask_path": f"object/{name}/selected_masks.npz",
                            "source_mask_path": str(crop / "masks.npz"), "source_mask_sha256": sha(crop / "masks.npz"),
                            "selection_path": str(crop / "selection.json"), "selection_sha256": sha(crop / "selection.json"),
                            "existing_crop_paths": [str(crop / f"selected/frame_{t:05d}/current_available.png") for t in frames],
                            "labels": e["labels"]})
    for name, e in mappings.items():
        source = link_root / "cases" / name / "base_segmentation.npz"
        with np.load(source, allow_pickle=False) as z:
            names = [str(n) for n in z["object_names"]]
            masks = z["object_masks"][:, names.index("link5")].astype(bool)
        # Reuse native saved-mask validity measurements and the pipeline's uniform
        # temporal sampling convention. This selects frames within the frozen 45;
        # it never selects another set of videos or generates masks.
        heights, _, truncated = frame_measurements(masks)
        valid = np.flatnonzero(heights > 0)
        frames = valid[np.linspace(0, len(valid) - 1, min(10, len(valid)), dtype=int)].tolist() if len(valid) else []
        dest = output / "link5" / name
        dest.mkdir(parents=True)
        np.savez_compressed(dest / "selected_masks.npz", masks=masks[frames], frame_ids=frames)
        link_rows.append({"video_id": name, "status": "ready" if len(frames) >= 2 else "skipped",
                          "reason": None if len(frames) >= 2 else "insufficient valid link5 masks",
                          "frames": frames, "video_path": str(video_path(video_root, e)),
                          "video_sha256": e["sha256"], "mask_path": f"link5/{name}/selected_masks.npz",
                          "source_mask_path": str(source), "source_mask_sha256": sha(source),
                          "valid_mask_frame_count": len(valid), "truncated_selected_frames": [t for t in frames if truncated[t]],
                          "labels": e["labels"]})
    for kind, rows in (("object", object_rows), ("link5", link_rows)):
        for row in rows:
            if "mask_path" in row:
                row["selected_masks_sha256"] = sha(output / row["mask_path"])
        payload = {"kind": kind, "rows": rows,
                   "frozen_video_selection_sha256": sha(link_root / "selection.json"),
                   "object_pair_manifest_sha256": sha(object_run / "metadata/pair_manifest.json"),
                   "link5_frame_policy": "10 uniformly distributed valid saved-mask frames; native frame_measurements height > 0"}
        (output / f"{kind}_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(kind, "ready", sum(r["status"] == "ready" for r in rows), "frames", sum(len(r["frames"]) for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    p.add_argument("--output", type=Path, help="defaults to results/simple3d-RUN/inputs")
    p.add_argument("--video-root", type=Path, default=Path("/root/autodl-tmp/motionsmoothness-robot/videos"))
    a = p.parse_args()
    prepare(a.output or result_path(a.run_id, "inputs"), a.video_root)
