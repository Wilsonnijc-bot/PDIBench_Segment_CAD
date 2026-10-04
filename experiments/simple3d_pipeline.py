"""Fresh selected-sequence MegaSaM geometry and direct upstream Simple3D scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PDI-Bench-edited/src"))
from experiments.simple3d_layout import result_path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(xyz, pixels=None, rgb=None, maximum=8192):
    import open3d as o3d
    from experiments.simple3d_adapter import UPSTREAM
    from feature_extractors.pointnet2_utils import pc_normalize
    xyz = np.asarray(xyz, dtype=np.float64)
    original = len(xyz)
    valid = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    indices = np.flatnonzero(valid)
    xyz = xyz[indices]
    valid_count = len(xyz)
    if valid_count < 128:
        raise ValueError(f"only {valid_count} valid points; official neighborhoods need 128")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    _, keep = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=4.0)
    keep = np.asarray(keep)
    xyz, indices = xyz[keep], indices[keep]
    centroid = xyz.mean(axis=0)
    radius = float(np.linalg.norm(xyz - centroid, axis=1).max())
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError("degenerate point-cloud scale")
    # Use the authors' pc_normalize utility, also on CAD. No rotation or alignment.
    normalized = pc_normalize(xyz.copy())
    voxels = np.floor(normalized / 0.005).astype(np.int64)
    _, keep = np.unique(voxels, axis=0, return_index=True)
    keep.sort()
    if len(keep) > maximum:
        keep = keep[np.sort(np.random.default_rng(0).choice(len(keep), maximum, replace=False))]
    if len(keep) < 128:
        raise ValueError(f"only {len(keep)} unique voxel points; official neighborhoods need 128")
    selected = indices[keep]
    cloud = {"xyz": normalized[keep].astype(np.float32), "raw_xyz": xyz[keep].astype(np.float32)}
    if pixels is not None:
        cloud["pixels_yx"] = np.asarray(pixels)[selected]
    if rgb is not None:
        cloud["rgb"] = np.asarray(rgb)[selected]
    meta = {"original_point_count": original, "valid_point_count": valid_count,
            "outlier_removed_point_count": valid_count - len(xyz),
            "final_sampled_point_count": len(keep), "centroid": centroid.tolist(),
            "scale_divisor": radius, "scale_multiplier": 1 / radius,
            "normalization": "official pc_normalize: subtract each cloud centroid, divide by maximum radius",
            "rotation_applied": False, "registration": None,
            "sampling": "0.005 unit-radius voxel, first real point per cell; seeded uniform cap 8192",
            "outlier_filter": "Open3D 20-neighbor statistical filter, std_ratio=4",
            "unit": "dimensionless after normalization; raw MegaSaM units preserved in raw_xyz"}
    return cloud, meta


def plot_cloud(path, xyz, colors=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(6, 5))
    ax = f.add_subplot(111, projection="3d")
    sc = ax.scatter(*xyz.T, c=colors if colors is not None else xyz[:, 2], s=2, cmap="inferno")
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=title)
    if colors is None or np.ndim(colors) == 1:
        f.colorbar(sc, ax=ax, shrink=.6, label="raw Simple3D score" if colors is not None else "Z")
    f.tight_layout()
    f.savefig(path, dpi=120)
    plt.close(f)


def build_cad(mesh_path, destination):
    import trimesh
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    mesh_path = Path(mesh_path)
    scene = trimesh.load(mesh_path, force="scene", process=False)
    mesh = scene.to_mesh()
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("invalid link5 CAD triangles")
    # Sample the original mesh uniformly by triangle surface area; no mesh fitting.
    dense, _ = trimesh.sample.sample_surface(mesh, 100000, seed=0)
    cloud, meta = preprocess(dense)
    meta.update(source_mesh=str(mesh_path), source_mesh_sha256=sha(mesh_path),
                dense_surface_sample_count=100000, triangle_count=len(mesh.faces),
                mesh_bounds=mesh.bounds.tolist(), megasam_status="not_applicable_cad",
                rigid_normalization="translation only; unit-radius scale; no rotation/ICP")
    np.savez_compressed(dest / "cad_cloud.npz", **cloud)
    write_json(dest / "cad_cloud.json", meta)
    plot_cloud(dest / "cad_cloud.png", cloud["xyz"], title="Unmodified link5 CAD: sampled surface")


def prepare_sequence(row, input_root, case, scene):
    from pdi_eval.object_deformation_wrapper.reference_visible_pixels import mask_bbox, rgba_crop, write_png
    video = Path(row["video_path"])
    if sha(video) != row["video_sha256"]:
        raise ValueError("original video differs from frozen existing selection")
    mask_file = Path(input_root) / row["mask_path"]
    if sha(mask_file) != row["selected_masks_sha256"]:
        raise ValueError("selected mask archive changed")
    with np.load(mask_file, allow_pickle=False) as z:
        masks = z["masks"].astype(bool)
        ids = z["frame_ids"].tolist()
    if ids != row["frames"] or len(masks) != len(ids):
        raise ValueError("mask frame IDs differ from frozen selection")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("cannot decode original video")
    sequence = case / f"{scene}.mp4"
    writer = cv2.VideoWriter(str(sequence), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                             (masks.shape[2], masks.shape[1]))
    if not writer.isOpened():
        raise ValueError("cannot create selected-sequence MP4")
    try:
        for t, mask in zip(ids, masks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != mask.shape:
                raise ValueError(f"frame {t} is unreadable or differs from saved mask grid")
            writer.write(frame)
            dest = case / "frames" / f"{t:06d}"
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(dest / "mask.png"), mask.astype(np.uint8) * 255)
            bbox = mask_bbox(mask)
            write_png(dest / "crop.png", rgba_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mask, bbox))
    finally:
        cap.release()
        writer.release()
    return sequence, masks


def reconstruct(row, input_root, case, run_id):
    from pdi_eval.perception.mega_sam_wrapper import MegaSamWrapper
    from pdi_eval.experiment.runner import clean_megasam_intermediates
    scene = "s3d_" + run_id.replace("-", "_") + "_" + row["video_id"].replace(".", "_") + "_" + uuid.uuid4().hex[:8]
    sequence, masks = prepare_sequence(row, input_root, case, scene)
    wrapper = MegaSamWrapper(device="cuda")
    meta = {"status": "running", "scene_name": scene, "source": row,
            "selected_sequence": str(sequence), "cache_hit": False,
            "sequence_frame_ids": row["frames"], "clouds": {},
            "geometry_code_identity": wrapper._reconstruction_code_identity()}
    write_json(case / "geometry.json", meta)
    try:
        # No cache_dir: every new experiment reconstructs all selected frames together.
        geometry = wrapper.infer_shared(str(sequence), masks[:, None], cache_dir=None)
        if geometry.frames_count != len(masks) or geometry.metadata["cache_hit"]:
            raise ValueError("fresh selected sequence reconstruction is incomplete")
        root = Path(wrapper.mega_sam_root)
        cvd = root / "outputs_cvd" / f"{scene}_sgd_cvd_hr.npz"
        droid = root / "outputs" / f"{scene}_droid.npz"
        raw = cvd if cvd.is_file() else droid
        with np.load(raw, allow_pickle=False) as z:
            depths, poses, intrinsic = z["depths"], z["cam_c2w"], z["intrinsic"]
            if len(depths) != len(masks) or len(poses) != len(masks):
                raise ValueError("native output frame count differs from selected sequence")
            np.savez_compressed(case / "reconstruction.npz", depths=depths, cam_c2w=poses, intrinsic=intrinsic,
                                frame_ids=row["frames"])
        meta.update(status="complete", refinement="CVD" if raw == cvd else "raw_DROID_CVD_unavailable",
                    reconstruction_path=str(case / "reconstruction.npz"),
                    pointmap_shape=list(geometry.pointmaps.shape), focal_length=geometry.focal_length,
                    geometry_metadata=geometry.metadata)
        for i, t in enumerate(row["frames"]):
            dest = case / "frames" / f"{t:06d}"
            try:
                points = geometry.pointmaps[i]
                h, w = points.shape[:2]
                mask = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                pose = geometry.camera_poses[i]
                camera_z = ((points - pose[:3, 3]) @ pose[:3, :3])[..., 2]
                yy, xx = np.where(mask)
                xyz = points[yy, xx].copy()
                xyz[~np.isfinite(camera_z[yy, xx]) | (camera_z[yy, xx] <= 0)] = np.nan
                image = cv2.resize(cv2.imread(str(dest / "frame.jpg")), (w, h))
                cloud, info = preprocess(xyz, pixels=np.column_stack((yy, xx)), rgb=image[yy, xx, ::-1] / 255)
                info.update(status="complete", megasam_status=meta["refinement"],
                            original_mask_pixel_count=int(masks[i].sum()), geometry_mask_pixel_count=int(mask.sum()),
                            frame_id=t, pixel_grid_hw=[h, w], cloud_path=str(dest / "cloud.npz"))
                np.savez_compressed(dest / "cloud.npz", **cloud)
                write_json(dest / "cloud.json", info)
            except Exception as e:
                info = {"status": "failed", "frame_id": t, "failure_reason": str(e),
                        "megasam_status": meta["refinement"],
                        "original_point_count": int(mask.sum()), "valid_point_count": int(np.isfinite(xyz).all(axis=1).sum()),
                        "final_sampled_point_count": 0, "centroid": None, "scale_divisor": None}
                write_json(dest / "cloud.json", info)
            meta["clouds"][str(t)] = info
        write_json(case / "geometry.json", meta)
    except Exception as e:
        meta.update(status="failed", failure_reason=str(e), traceback=traceback.format_exc())
        write_json(case / "geometry.json", meta)
    finally:
        # Only unique scenes owned by this invocation; retain fresh native depths and
        # all real sampled points, but avoid exhausting the rented disk with scratch.
        clean_megasam_intermediates(scene)
    return meta


def score_mode(row, geometry, observed, output, mode, cad, visual):
    from experiments.simple3d_adapter import build_simple3d_reference, score_simple3d
    frames = row["frames"]
    output = Path(output) / "cases" / row["video_id"]
    output.mkdir(parents=True, exist_ok=True)
    ref_t = frames[0] if mode == "first_frame" else None
    ref_path = Path(cad) / "cad_cloud.npz" if mode == "cad" else observed / "frames" / f"{ref_t:06d}" / "cloud.npz"
    ref_meta_path = ref_path.with_suffix(".json")
    tests = frames if mode == "cad" else frames[1:]
    reference, ref_meta, ref_error = None, {}, None
    try:
        ref_meta = json.loads(ref_meta_path.read_text())
        if ref_meta.get("status", "complete") != "complete":
            raise ValueError(ref_meta.get("failure_reason", "reference cloud unavailable"))
        with np.load(ref_path, allow_pickle=False) as z:
            reference = build_simple3d_reference(z["xyz"])
        np.savez_compressed(output / "reference_prototypes.npz", features=reference.method.patch_lib.cpu().numpy(),
                            coreset_indices=reference.method.coreset_idx.cpu().numpy())
    except Exception as e:
        ref_error = str(e)
    comparisons = []
    for j, t in enumerate(tests):
        dest = observed / "frames" / f"{t:06d}"
        test_meta = geometry.get("clouds", {}).get(str(t), {})
        result = {"video_id": row["video_id"], "reference_type": "cad" if mode == "cad" else "first_observed_frame",
                  "reference_frame_id": ref_t, "test_frame_id": t,
                  "mask_path": str(dest / "mask.png"), "crop_path": str(dest / "crop.png"),
                  "frame_path": str(dest / "frame.jpg"), "reference_point_cloud_path": str(ref_path),
                  "test_point_cloud_path": str(dest / "cloud.npz"),
                  "point_anomaly_scores_path": None, "scalar_anomaly_score": None,
                  "valid_reference_point_count": ref_meta.get("valid_point_count", 0),
                  "valid_test_point_count": test_meta.get("valid_point_count", 0),
                  "sampled_reference_point_count": ref_meta.get("final_sampled_point_count", 0),
                  "sampled_test_point_count": test_meta.get("final_sampled_point_count", 0),
                  "megasam_status": geometry["status"], "megasam_refinement": geometry.get("refinement"),
                  "simple3d_status": "failed", "failure_reason": None}
        try:
            if geometry["status"] != "complete":
                raise ValueError(geometry.get("failure_reason", "MegaSaM failed"))
            if ref_error:
                raise ValueError("reference: " + ref_error)
            if test_meta.get("status") != "complete":
                raise ValueError("test: " + test_meta.get("failure_reason", "cloud unavailable"))
            with np.load(dest / "cloud.npz", allow_pickle=False) as z:
                xyz, rgb = z["xyz"], z["rgb"]
                scores, scalar = score_simple3d(reference, xyz)
            path = output / f"frame_{t:06d}_point_scores.npy"
            np.save(path, scores)
            result.update(simple3d_status="complete", scalar_anomaly_score=scalar, point_anomaly_scores_path=str(path))
            if visual and j in (0, len(tests) // 2, len(tests) - 1):
                plot_cloud(output / f"frame_{t:06d}_reconstruction.png", xyz, rgb, f"{row['video_id']} frame {t}")
                plot_cloud(output / f"frame_{t:06d}_anomaly.png", xyz, scores, f"{mode} reference: score {scalar:.5f}")
        except Exception as e:
            result["failure_reason"] = str(e)
        comparisons.append(result)
        write_json(output / f"frame_{t:06d}_comparison.json", result)
        print(json.dumps({"video": row["video_id"], "mode": mode, "frame": t,
                          "score": result["scalar_anomaly_score"], "status": result["simple3d_status"],
                          "error": result["failure_reason"]}), flush=True)
    write_json(output / "comparisons.json", comparisons)
    return comparisons


def run_case(row, config):
    observed = Path(config["observed_root"]) / row["video_id"]
    observed.mkdir(parents=True, exist_ok=True)
    # The shared link5 geometry root is explicitly tied to a run and input manifest.
    # A file lock prevents independent CAD/first-frame invocations from rebuilding it.
    with (observed / "geometry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta_path = observed / "geometry.json"
        if meta_path.is_file():
            geometry = json.loads(meta_path.read_text())
            if geometry.get("source") != row:
                raise ValueError("shared observed geometry belongs to different inputs")
            if geometry["status"] == "running":
                raise ValueError("interrupted geometry; use a new run directory")
        else:
            try:
                geometry = reconstruct(row, config["input_root"], observed, config["run_id"])
            except Exception as e:
                geometry = {"status": "failed", "source": row, "clouds": {},
                            "failure_reason": str(e), "traceback": traceback.format_exc()}
                write_json(meta_path, geometry)
    for mode in config["modes"]:
        comparisons = Path(config["outputs"][mode]) / "cases" / row["video_id"] / "comparisons.json"
        if comparisons.exists():
            continue
        score_mode(row, geometry, observed, config["outputs"][mode], mode, config.get("cad_root"),
                   row["video_id"] in config["visual_ids"])
    return row["video_id"]


def distribution(values):
    if not values:
        return {"count": 0}
    a = np.asarray(values)
    return {"count": len(a), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "q25": float(np.quantile(a, .25)), "median": float(np.median(a)),
            "q75": float(np.quantile(a, .75)), "max": float(a.max())}


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else ["video_id", "test_frame_id", "scalar_anomaly_score"])
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def aggregate(config, manifest):
    summaries, maps = {}, {}
    ready = [r for r in manifest["rows"] if r["status"] == "ready"]
    for mode in config["modes"]:
        root = Path(config["outputs"][mode])
        rows, done = [], []
        for row in ready:
            path = root / "cases" / row["video_id"] / "comparisons.json"
            if path.is_file():
                rows.extend(json.loads(path.read_text()))
                done.append(row["video_id"])
        success = [r for r in rows if r["simple3d_status"] == "complete"]
        complete_vids = [v for v in done if all(r["simple3d_status"] == "complete" for r in rows if r["video_id"] == v)]
        reconstructions = []
        for row in ready:
            path = Path(config["observed_root"]) / row["video_id"] / "geometry.json"
            if path.is_file():
                reconstructions.append(json.loads(path.read_text()))
        summary = {"status": ("completed_with_failures" if len(success) != len(rows) else "completed") if len(done) == len(ready) else "running", "reference": mode,
                   "video_count": len(manifest["rows"]), "eligible_video_count": len(ready),
                   "videos_processed": len(done), "videos_all_scores_successful": len(complete_vids),
                   "expected_comparisons": sum(len(r["frames"]) - (mode == "first_frame") for r in ready),
                   "attempted_comparisons": len(rows), "successful_simple3d_scores": len(success),
                   "attempted_megasam_reconstructions": len(reconstructions),
                   "successful_megasam_reconstructions": sum(g["status"] == "complete" for g in reconstructions),
                   "cvd_reconstructions": sum(g.get("refinement") == "CVD" for g in reconstructions),
                   "raw_droid_reconstructions": sum(g.get("refinement") == "raw_DROID_CVD_unavailable" for g in reconstructions),
                   "score_distribution": distribution([r["scalar_anomaly_score"] for r in success]),
                   "failures": [r for r in rows if r["simple3d_status"] != "complete"],
                   "skipped_inputs": [r for r in manifest["rows"] if r["status"] != "ready"],
                   "pending_video_ids": [r["video_id"] for r in ready if r["video_id"] not in done]}
        write_json(root / "comparisons.json", rows)
        write_csv(root / "comparisons.csv", rows)
        write_json(root / "summary.json", summary)
        summaries[mode] = summary
        maps[mode] = {(r["video_id"], r["test_frame_id"]): r for r in rows}
    if manifest["kind"] == "link5":
        # Read both modes even when this invocation only scores one of them.
        for mode in ("cad", "first_frame"):
            path = Path(config["outputs"][mode]) / "comparisons.json"
            maps[mode] = {(r["video_id"], r["test_frame_id"]): r for r in json.loads(path.read_text())} if path.is_file() else {}
        combined = [{"video_id": r["video_id"], "frame_id": t,
                     "cad_reference_score": maps["cad"].get((r["video_id"], t), {}).get("scalar_anomaly_score"),
                     "first_frame_reference_score": maps["first_frame"].get((r["video_id"], t), {}).get("scalar_anomaly_score"),
                     "first_frame_is_reference": t == r["frames"][0],
                     "cad_status": maps["cad"].get((r["video_id"], t), {}).get("simple3d_status", "pending"),
                     "first_frame_status": "reference_not_scored" if t == r["frames"][0] else maps["first_frame"].get((r["video_id"], t), {}).get("simple3d_status", "pending")}
                    for r in ready for t in r["frames"]]
        root = Path(config["outputs"]["cad"])
        write_json(root / "combined_comparison.json", combined)
        write_csv(root / "combined_comparison.csv", combined)
    return summaries


def main(kind):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True, help="prepared, hash-verified input directory")
    p.add_argument("--output-root", type=Path, default=ROOT / "results")
    p.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--resume", action="store_true", help="resume only this run's outputs, never historical geometry")
    p.add_argument("--summarize-only", action="store_true")
    if kind == "link5":
        p.add_argument("--reference", choices=["cad", "first_frame", "both"], default="both")
        p.add_argument("--cad", type=Path, default=Path("/Users/nijiachen/Downloads/6D_CAD_deformation_detection/assets/cad/franka_fer/link5.dae"))
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    manifest = json.loads((a.inputs / f"{kind}_manifest.json").read_text())
    modes = ["first_frame"] if kind == "object" else (["cad", "first_frame"] if a.reference == "both" else [a.reference])
    if kind == "object":
        outputs = {"first_frame": str(result_path(a.run_id, "object", a.output_root))}
        observed = Path(outputs["first_frame"]) / "observed"
    else:
        outputs = {"cad": str(result_path(a.run_id, "cad", a.output_root)),
                   "first_frame": str(result_path(a.run_id, "first_frame", a.output_root))}
        observed = Path(outputs["cad"]) / "observed"
    ready = [r for r in manifest["rows"] if r["status"] == "ready"]
    visual_ids = []
    for dataset in ("COSMOS2.5", "COSMOS3", "LVP_ROBOWM"):
        sample = next((r["video_id"] for r in ready if r["video_id"].startswith(dataset + "_")), None)
        if sample:
            visual_ids.append(sample)
    config = {"input_root": str(a.inputs.resolve()), "observed_root": str(observed), "outputs": outputs,
              "modes": modes, "run_id": a.run_id, "visual_ids": visual_ids, "workers": a.workers}
    if kind == "link5":
        config["cad_root"] = str(Path(outputs["cad"]) / "cad")
    if a.summarize_only:
        print(json.dumps(aggregate(config, manifest), indent=2))
        return 0
    from experiments.simple3d_adapter import provenance
    method = provenance()
    for mode in modes:
        path = Path(outputs[mode])
        if path.exists() and not a.resume:
            p.error(f"output already exists: {path}; choose a new run ID or explicit --resume")
        path.mkdir(parents=True, exist_ok=True)
        stored = path / "run.json"
        identity = {"kind": kind, "run_id": a.run_id, "manifest_sha256": sha(a.inputs / f"{kind}_manifest.json"),
                    "method": method, "preprocessing": "cloud centering/unit radius; isolated outliers; voxel/uniform sampling; no rotation/registration",
                    "config": config, "cad_mesh_sha256": sha(a.cad) if kind == "link5" and a.cad.is_file() else None}
        if stored.exists():
            old = json.loads(stored.read_text())
            if old["manifest_sha256"] != identity["manifest_sha256"] or old["method"] != method:
                raise ValueError("cannot resume changed inputs or Simple3D source/settings")
        else:
            write_json(stored, identity)
    if "cad" in modes and not (Path(config["cad_root"]) / "cad_cloud.npz").is_file():
        build_cad(a.cad, config["cad_root"])
    chosen = [r for r in ready if not a.cases or r["video_id"] in a.cases]
    if a.cases and set(a.cases) - {r["video_id"] for r in chosen}:
        p.error("unknown or unavailable requested cases")
    # Spawn avoids inheriting a CUDA context; independent video workers share GPU 0.
    import multiprocessing as mp
    failures = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(run_case, row, config): row for row in chosen}
        for future in as_completed(pending):
            row = pending[future]
            try:
                print("VIDEO_FINISHED", future.result(), flush=True)
            except Exception as e:
                failures.append({"video_id": row["video_id"], "failure_reason": str(e), "traceback": traceback.format_exc()})
                write_json(Path(outputs[modes[0]]) / "worker_failures.json", failures)
                print("WORKER_FAILED", row["video_id"], str(e), flush=True)
            aggregate(config, manifest)
    summary = aggregate(config, manifest)
    print("BATCH_FINISHED", json.dumps({m: {k: v for k, v in s.items() if k not in ("failures", "skipped_inputs", "pending_video_ids")}
                                       for m, s in summary.items()}), flush=True)
    return 1 if failures or any(s["failures"] for s in summary.values()) else 0
