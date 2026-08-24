"""Run FoundationPose in its isolated environment and write a numeric pose archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from ..evaluator.cad_rigidity_audit import ImageGridTransform, erode_masks
from ..geometry.cad_mesh import load_cad_manifest
from ..geometry.se3 import rigid_transform_valid
from .foundation_pose_wrapper import (
    POSE_SOURCE_INVALID,
    POSE_SOURCE_REGISTER,
    POSE_SOURCE_REREGISTER,
    POSE_SOURCE_TRACK,
)
from .segmentation_archive import load_multi_object_segmentation


CAD_LINK_NAMES = ("link2", "link3", "link4", "link5", "link6", "link7")


def _foundationpose_mesh(mesh):
    """Convert texture-less PBR CAD visuals to FoundationPose vertex colors."""
    import trimesh

    compatible = mesh.copy()
    visual = compatible.visual
    if not isinstance(visual, trimesh.visual.texture.TextureVisuals):
        return compatible
    material = visual.material
    if getattr(material, "image", None) is not None and visual.uv is not None:
        return compatible
    raw_color = getattr(material, "baseColorFactor", None)
    if raw_color is None:
        raw_color = [128, 128, 128, 255]
    color = np.asarray(raw_color, dtype=np.uint8).reshape(-1)
    if color.size == 3:
        color = np.concatenate((color, np.asarray([255], dtype=np.uint8)))
    if color.size != 4:
        color = np.asarray([128, 128, 128, 255], dtype=np.uint8)
    compatible.visual = trimesh.visual.ColorVisuals(
        mesh=compatible,
        vertex_colors=np.tile(color, (len(compatible.vertices), 1)),
    )
    return compatible


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_geometry(path: Path) -> dict[str, np.ndarray]:
    required = {
        "pointmaps",
        "camera_poses",
        "rgb_camera",
        "depth_camera",
        "intrinsics_camera",
        "frame_times_seconds",
        "source_hw",
        "resized_hw_before_crop",
        "crop_xywh",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"MegaSAM geometry archive lacks arrays: {missing}")
        values = {name: np.asarray(archive[name]) for name in required}
    rgb = np.asarray(values["rgb_camera"], dtype=np.uint8)
    depth = np.asarray(values["depth_camera"], dtype=np.float32)
    if rgb.shape[:3] != depth.shape or rgb.shape[-1] != 3:
        raise ValueError("MegaSAM RGB and depth arrays do not share one image grid")
    return values


def _intrinsics_at(intrinsics: np.ndarray, frame_index: int) -> np.ndarray:
    return np.asarray(
        intrinsics[frame_index] if intrinsics.ndim == 3 else intrinsics,
        dtype=np.float64,
    )


def _mask_depth_gate(mask: np.ndarray, depth: np.ndarray) -> tuple[bool, float]:
    area = int(np.count_nonzero(mask))
    if area < 256:
        return False, 0.0
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    fraction = float(np.count_nonzero(valid) / area)
    return fraction >= 0.70, fraction


def _dilate(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _render_diagnostics(estimator, public_pose, K, height, width, glctx):
    import torch
    from Utils import make_mesh_tensors, nvdiffrast_render

    if not hasattr(estimator, "_pdi_public_mesh_tensors"):
        estimator._pdi_public_mesh_tensors = make_mesh_tensors(estimator.mesh_ori)
    with torch.inference_mode():
        color, rendered_depth, _ = nvdiffrast_render(
            K=K,
            H=height,
            W=width,
            ob_in_cams=torch.as_tensor(
                public_pose[None], dtype=torch.float32, device="cuda"
            ),
            glctx=glctx,
            mesh_tensors=estimator._pdi_public_mesh_tensors,
        )
    rendered_depth = rendered_depth[0].detach().cpu().numpy()
    rendered_mask = rendered_depth > 0.001
    return rendered_mask, rendered_depth, color[0].detach().cpu().numpy()


def _pose_objective(
    estimator,
    public_pose: np.ndarray,
    *,
    K: np.ndarray,
    observed_depth: np.ndarray,
    observed_mask: np.ndarray,
    diameter: float,
    glctx,
) -> tuple[float, float, float, int]:
    if not rigid_transform_valid(public_pose) or public_pose[2, 3] <= 0.0:
        return 1.0, 0.0, np.nan, 0
    rendered_mask, rendered_depth, _ = _render_diagnostics(
        estimator,
        public_pose,
        K,
        observed_depth.shape[0],
        observed_depth.shape[1],
        glctx,
    )
    observed_dilated = _dilate(observed_mask)
    rendered_dilated = _dilate(rendered_mask)
    union = observed_dilated | rendered_dilated
    intersection = observed_dilated & rendered_dilated
    iou = float(np.count_nonzero(intersection) / max(np.count_nonzero(union), 1))
    core = erode_masks(observed_mask[None], 2)[0] & rendered_mask
    core &= np.isfinite(observed_depth) & (observed_depth > 0.0)
    if not np.any(core):
        return 1.0, iou, np.nan, 0
    raw_depth_residual = float(
        np.median(np.abs(observed_depth[core] - rendered_depth[core]))
    )
    normalized_depth = min(raw_depth_residual / max(0.05 * diameter, 1e-9), 1.0)
    objective = 0.5 * (1.0 - iou) + 0.5 * normalized_depth
    return float(objective), iou, raw_depth_residual, int(np.count_nonzero(core))


def _register(estimator, *, K, rgb, depth, mask, iterations):
    try:
        return np.asarray(
            estimator.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=iterations,
            ),
            dtype=np.float64,
        )
    except Exception as exc:  # FoundationPose failures are per-frame artifacts.
        print(f"FoundationPose registration failed: {exc}", file=sys.stderr)
        return None


def _track(estimator, *, K, rgb, depth, iterations):
    try:
        return np.asarray(
            estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=iterations),
            dtype=np.float64,
        )
    except Exception as exc:
        print(f"FoundationPose tracking failed: {exc}", file=sys.stderr)
        return None


def _restore_tracking_pose(estimator, public_pose: np.ndarray) -> None:
    import torch

    transform_to_center = estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
    centered = public_pose @ np.linalg.inv(transform_to_center)
    estimator.pose_last = torch.as_tensor(
        centered, dtype=torch.float32, device="cuda"
    ).reshape(1, 4, 4)


def _candidate_frames(
    masks: np.ndarray,
    depths: np.ndarray,
    truncated: np.ndarray,
) -> dict[int, int]:
    selected: dict[int, int] = {}
    for link_index in range(masks.shape[1]):
        choices = []
        for frame_index in range(len(masks)):
            if truncated[frame_index, link_index]:
                continue
            ok, valid_fraction = _mask_depth_gate(
                masks[frame_index, link_index], depths[frame_index]
            )
            if ok:
                choices.append(
                    (
                        int(np.count_nonzero(masks[frame_index, link_index])),
                        valid_fraction,
                        -frame_index,
                    )
                )
        if choices:
            best = max(choices)
            selected[link_index] = -best[2]
    return selected


def _evaluate_scale(
    scale: float,
    *,
    estimators,
    meshes,
    candidate_frames,
    rgb,
    depth,
    masks,
    intrinsics,
    registration_iterations,
    glctx,
) -> tuple[float, dict[str, dict]]:
    diagnostics = {}
    objectives = []
    for link_index, frame_index in candidate_frames.items():
        name = CAD_LINK_NAMES[link_index]
        K = _intrinsics_at(intrinsics, frame_index)
        scaled_depth = np.asarray(depth[frame_index] * scale, dtype=np.float32)
        pose = _register(
            estimators[name],
            K=K,
            rgb=rgb[frame_index],
            depth=scaled_depth,
            mask=masks[frame_index, link_index],
            iterations=registration_iterations,
        )
        if pose is None:
            objective, iou, residual, support = 1.0, 0.0, np.nan, 0
        else:
            objective, iou, residual, support = _pose_objective(
                estimators[name],
                pose,
                K=K,
                observed_depth=scaled_depth,
                observed_mask=masks[frame_index, link_index],
                diameter=meshes[name].diameter,
                glctx=glctx,
            )
        objectives.append(objective)
        diagnostics[name] = {
            "frame_index": frame_index,
            "objective": objective,
            "silhouette_iou": iou,
            "depth_residual": residual,
            "support_pixels": support,
        }
    return float(np.median(objectives)), diagnostics


def _calibrate_scale(
    *,
    policy,
    estimators,
    meshes,
    candidate_frames,
    rgb,
    depth,
    masks,
    intrinsics,
    registration_iterations,
    glctx,
):
    if policy == "metric-prior":
        objective, diagnostics = _evaluate_scale(
            1.0,
            estimators=estimators,
            meshes=meshes,
            candidate_frames=candidate_frames,
            rgb=rgb,
            depth=depth,
            masks=masks,
            intrinsics=intrinsics,
            registration_iterations=registration_iterations,
            glctx=glctx,
        )
        return 1.0, np.asarray([1.0]), np.asarray([objective]), {"1.0": diagnostics}
    if len(candidate_frames) < 3:
        raise RuntimeError("video-global CAD calibration requires at least three links")
    coarse = np.power(2.0, np.arange(-2.0, 2.0001, 0.25))
    candidates = []
    objectives = []
    all_diagnostics = {}
    for scale in coarse:
        value, diagnostics = _evaluate_scale(
            float(scale),
            estimators=estimators,
            meshes=meshes,
            candidate_frames=candidate_frames,
            rgb=rgb,
            depth=depth,
            masks=masks,
            intrinsics=intrinsics,
            registration_iterations=registration_iterations,
            glctx=glctx,
        )
        candidates.append(float(scale))
        objectives.append(value)
        all_diagnostics[f"{scale:.12g}"] = diagnostics
    ranking = sorted(
        range(len(candidates)),
        key=lambda index: (
            objectives[index],
            abs(np.log2(candidates[index])),
            candidates[index],
        ),
    )
    coarse_winner = candidates[ranking[0]]
    center = np.log2(coarse_winner)
    fine_logs = np.arange(center - 0.25, center + 0.2501, 0.05)
    for log_scale in fine_logs:
        scale = float(2.0**log_scale)
        if any(np.isclose(scale, previous, rtol=0.0, atol=1e-12) for previous in candidates):
            continue
        value, diagnostics = _evaluate_scale(
            scale,
            estimators=estimators,
            meshes=meshes,
            candidate_frames=candidate_frames,
            rgb=rgb,
            depth=depth,
            masks=masks,
            intrinsics=intrinsics,
            registration_iterations=registration_iterations,
            glctx=glctx,
        )
        candidates.append(scale)
        objectives.append(value)
        all_diagnostics[f"{scale:.12g}"] = diagnostics
    ranking = sorted(
        range(len(candidates)),
        key=lambda index: (
            objectives[index],
            abs(np.log2(candidates[index])),
            candidates[index],
        ),
    )
    winner = candidates[ranking[0]]
    winning_objective = objectives[ranking[0]]
    successful = sum(
        item["objective"] < 1.0
        for item in all_diagnostics[f"{winner:.12g}"].values()
    )
    if np.isclose(winner, 0.25) or np.isclose(winner, 4.0):
        raise RuntimeError("video-global scale calibration selected a boundary value")
    if successful < 3 or winning_objective > 0.60:
        raise RuntimeError(
            "video-global scale calibration failed quality gates: "
            f"successful_links={successful}, objective={winning_objective:.4f}"
        )
    return (
        winner,
        np.asarray(candidates, dtype=np.float64),
        np.asarray(objectives, dtype=np.float64),
        all_diagnostics,
    )


def _pose_valid(objective, iou, support, pose) -> bool:
    return bool(
        pose is not None
        and rigid_transform_valid(pose)
        and pose[2, 3] > 0.0
        and support >= 128
        and iou >= 0.20
        and np.isfinite(objective)
    )


def _run_temporal_poses(
    *,
    scale,
    estimators,
    meshes,
    rgb,
    depth,
    masks,
    intrinsics,
    registration_iterations,
    tracking_iterations,
    reregister_interval,
    glctx,
):
    frames = len(rgb)
    links = len(CAD_LINK_NAMES)
    transforms = np.full((frames, links, 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros((frames, links), dtype=bool)
    source = np.zeros((frames, links), dtype=np.uint8)
    objective = np.full((frames, links), np.inf, dtype=np.float32)
    iou = np.full((frames, links), np.nan, dtype=np.float32)
    residual = np.full((frames, links), np.nan, dtype=np.float32)
    for link_index, name in enumerate(CAD_LINK_NAMES):
        estimator = estimators[name]
        processed = 0
        initialized = False
        for frame_index in range(frames):
            frame_mask = masks[frame_index, link_index]
            scaled_depth = np.asarray(depth[frame_index] * scale, dtype=np.float32)
            input_ok, _ = _mask_depth_gate(frame_mask, scaled_depth)
            if not input_ok:
                source[frame_index, link_index] = POSE_SOURCE_INVALID
                continue
            K = _intrinsics_at(intrinsics, frame_index)
            candidates = []
            if initialized:
                tracked = _track(
                    estimator,
                    K=K,
                    rgb=rgb[frame_index],
                    depth=scaled_depth,
                    iterations=tracking_iterations,
                )
                if tracked is not None:
                    candidates.append((POSE_SOURCE_TRACK, tracked))
            should_register = not initialized or processed % reregister_interval == 0
            if should_register or not candidates:
                registered = _register(
                    estimator,
                    K=K,
                    rgb=rgb[frame_index],
                    depth=scaled_depth,
                    mask=frame_mask,
                    iterations=registration_iterations,
                )
                if registered is not None:
                    candidates.append(
                        (
                            POSE_SOURCE_REGISTER
                            if not initialized
                            else POSE_SOURCE_REREGISTER,
                            registered,
                        )
                    )
            scored = []
            for candidate_source, candidate_pose in candidates:
                values = _pose_objective(
                    estimator,
                    candidate_pose,
                    K=K,
                    observed_depth=scaled_depth,
                    observed_mask=frame_mask,
                    diameter=meshes[name].diameter,
                    glctx=glctx,
                )
                scored.append((values[0], candidate_source, candidate_pose, values))
            scored.sort(key=lambda item: (item[0], item[1]))
            accepted = None
            for item in scored:
                if _pose_valid(item[0], item[3][1], item[3][3], item[2]):
                    accepted = item
                    break
            if accepted is None:
                initialized = False
                estimator.pose_last = None
                continue
            q, candidate_source, candidate_pose, values = accepted
            transforms[frame_index, link_index] = candidate_pose
            valid[frame_index, link_index] = True
            source[frame_index, link_index] = candidate_source
            objective[frame_index, link_index] = q
            iou[frame_index, link_index] = values[1]
            residual[frame_index, link_index] = values[2]
            _restore_tracking_pose(estimator, candidate_pose)
            initialized = True
            processed += 1
            print(
                f"pose {name} frame={frame_index} source={candidate_source} "
                f"q={q:.4f} iou={values[1]:.4f}",
                flush=True,
            )
    return transforms, valid, source, objective, iou, residual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-npz", type=Path, required=True)
    parser.add_argument("--segmentation-npz", type=Path, required=True)
    parser.add_argument("--cad-manifest", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument(
        "--scale-policy",
        choices=("metric-prior", "video-global-cad"),
        default="video-global-cad",
    )
    parser.add_argument("--registration-iterations", type=int, default=5)
    parser.add_argument("--tracking-iterations", type=int, default=2)
    parser.add_argument("--reregister-interval", type=int, default=10)
    parser.add_argument("--debug-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    geometry_path = args.geometry_npz.resolve()
    segmentation_path = args.segmentation_npz.resolve()
    manifest_path = args.cad_manifest.resolve()
    output_path = args.output_npz.resolve()
    if min(args.registration_iterations, args.tracking_iterations, args.reregister_interval) < 1:
        raise ValueError("FoundationPose iteration settings must be positive")
    geometry = _load_geometry(geometry_path)
    segmentation = load_multi_object_segmentation(segmentation_path)
    indices = {name: segmentation.object_names.index(name) for name in CAD_LINK_NAMES}
    frame_count = min(len(geometry["rgb_camera"]), segmentation.frames_count)
    grid = ImageGridTransform(
        source_hw=tuple(int(value) for value in geometry["source_hw"]),
        resized_hw=tuple(int(value) for value in geometry["resized_hw_before_crop"]),
        crop_xywh=tuple(int(value) for value in geometry["crop_xywh"]),
    )
    source_masks = np.stack(
        [segmentation.object_masks[:frame_count, indices[name]] for name in CAD_LINK_NAMES],
        axis=1,
    )
    masks = np.stack(
        [grid.transform_masks(source_masks[:, index]) for index in range(len(CAD_LINK_NAMES))],
        axis=1,
    )
    masks = erode_masks(masks.reshape(-1, *masks.shape[2:]), 2).reshape(masks.shape)
    truncated = np.stack(
        [segmentation.is_truncated[:frame_count, indices[name]] for name in CAD_LINK_NAMES],
        axis=1,
    )
    meshes = load_cad_manifest(manifest_path, link_names=CAD_LINK_NAMES)

    foundation_source = Path(os.environ["PDI_FOUNDATIONPOSE_SOURCE"]).resolve()
    sys.path[:0] = [str(foundation_source), str(foundation_source / "mycpp/build")]
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, set_seed

    set_seed(0)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    debug_root = (args.debug_dir or output_path.parent / "foundationpose_debug").resolve()
    debug_root.mkdir(parents=True, exist_ok=True)
    foundation_meshes = {
        name: _foundationpose_mesh(meshes[name].mesh) for name in CAD_LINK_NAMES
    }
    estimators = {
        name: FoundationPose(
            model_pts=meshes[name].vertices,
            model_normals=np.asarray(foundation_meshes[name].vertex_normals),
            mesh=foundation_meshes[name],
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
            debug_dir=str(debug_root / name),
        )
        for name in CAD_LINK_NAMES
    }
    rgb = np.asarray(geometry["rgb_camera"][:frame_count], dtype=np.uint8)
    depth = np.asarray(geometry["depth_camera"][:frame_count], dtype=np.float32)
    intrinsics = np.asarray(geometry["intrinsics_camera"], dtype=np.float64)
    candidates = _candidate_frames(masks, depth, truncated)
    scale, scale_candidates, scale_objective, scale_diagnostics = _calibrate_scale(
        policy=args.scale_policy,
        estimators=estimators,
        meshes=meshes,
        candidate_frames=candidates,
        rgb=rgb,
        depth=depth,
        masks=masks,
        intrinsics=intrinsics,
        registration_iterations=args.registration_iterations,
        glctx=glctx,
    )
    print(f"video depth scale={scale:.12g}", flush=True)
    transforms, valid, source, objective, iou, residual = _run_temporal_poses(
        scale=scale,
        estimators=estimators,
        meshes=meshes,
        rgb=rgb,
        depth=depth,
        masks=masks,
        intrinsics=intrinsics,
        registration_iterations=args.registration_iterations,
        tracking_iterations=args.tracking_iterations,
        reregister_interval=args.reregister_interval,
        glctx=glctx,
    )
    camera_poses = np.asarray(geometry["camera_poses"][:frame_count], dtype=np.float64).copy()
    camera_poses[:, :3, 3] *= scale
    world_from_link = camera_poses[:, None] @ transforms
    world_from_link[~valid] = np.nan
    weights = foundation_source / "weights"
    weight_files = sorted(weights.glob("*/model_best.pth"))
    metadata = {
        "schema": 1,
        "foundationpose_revision": subprocess.run(
            ["git", "-C", str(foundation_source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "weight_sha256": {str(path.relative_to(foundation_source)): _sha256(path) for path in weight_files},
        "geometry_sha256": _sha256(geometry_path),
        "segmentation_sha256": _sha256(segmentation_path),
        "cad_sha256": {name: meshes[name].sha256 for name in CAD_LINK_NAMES},
        "scale_policy": args.scale_policy,
        "scale_calibration_frames": {CAD_LINK_NAMES[index]: frame for index, frame in candidates.items()},
        "scale_diagnostics": scale_diagnostics,
        "registration_iterations": args.registration_iterations,
        "tracking_iterations": args.tracking_iterations,
        "reregister_interval": args.reregister_interval,
        "random_seed": 0,
        "duration_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        link_names=np.asarray(CAD_LINK_NAMES),
        frame_indices=np.arange(frame_count, dtype=np.int32),
        frame_times_seconds=np.asarray(geometry["frame_times_seconds"][:frame_count], dtype=np.float64),
        T_C_from_L=transforms,
        T_W_from_L=world_from_link,
        pose_valid=valid,
        pose_source=source,
        silhouette_iou=iou,
        pose_depth_residual=residual,
        pose_objective=objective,
        video_depth_scale=np.asarray(scale, dtype=np.float64),
        scale_candidates=scale_candidates,
        scale_objective=scale_objective,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_path),
                "video_depth_scale": scale,
                "pose_valid_fraction": float(valid.mean()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
