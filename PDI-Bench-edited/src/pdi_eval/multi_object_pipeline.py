"""Video-level PDI evaluation for multiple independently rigid targets."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .evaluator.cad_rigidity_audit import (
    CAD_LINK_NAMES,
    CadCanonicalizationRuntime,
)
from .evaluator.motion_audit import audit_3d_trajectory_consistency
from .evaluator.reconstruction_audit import audit_ground_flatness, audit_scale_jump
from .evaluator.scale_audit import audit_scale_consistency
from .evaluator.volume_audit import audit_3d_volume_stability
from .geometry.camera import CameraModel
from .geometry.projection import ProjectionJudge
from .metrics.pdi_index import PDIIndexCalculator
from .perception.base import MultiObjectTrackResult
from .perception.foundation_pose_wrapper import run_foundation_pose_worker
from .perception.mega_sam_wrapper import MegaSamWrapper
from .perception.segmentation_archive import load_multi_object_segmentation
from .perception.track_wrapper import TRACKING_MODES, TrackWrapper
from .utils.logger import pdi_logger


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_lsd_frames(video_path: str, maximum: int = 3) -> tuple[np.ndarray | None, float]:
    capture = cv2.VideoCapture(video_path)
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    for _ in range(maximum):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return (np.asarray(frames) if frames else None), (raw_fps if raw_fps > 0 else 24.0)


def _vp_in_object_bbox(
    vp_xy: tuple[float, float],
    masks: np.ndarray,
    margin_ratio: float = 0.1,
) -> bool:
    if len(masks) == 0:
        return False
    combined = np.any(masks[:min(5, len(masks))], axis=0)
    ys, xs = np.where(combined)
    if len(xs) == 0:
        return False
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    margin_x = (x_max - x_min) * margin_ratio
    margin_y = (y_max - y_min) * margin_ratio
    return bool(
        x_min - margin_x <= vp_xy[0] <= x_max + margin_x
        and y_min - margin_y <= vp_xy[1] <= y_max + margin_y
    )


def _vp_epsilon(
    foreground_vp: tuple[float, float],
    background_vp: tuple[float, float],
    camera: CameraModel,
    image_hw: tuple[int, int],
) -> float:
    foreground_direction = np.asarray(
        [foreground_vp[0] - camera.cx, foreground_vp[1] - camera.cy],
        dtype=np.float64,
    )
    background_direction = np.asarray(
        [background_vp[0] - camera.cx, background_vp[1] - camera.cy],
        dtype=np.float64,
    )
    foreground_norm = float(np.linalg.norm(foreground_direction))
    background_norm = float(np.linalg.norm(background_direction))
    height, width = image_hw
    foreground_offscreen = not (
        0 <= foreground_vp[0] <= width and 0 <= foreground_vp[1] <= height
    )
    if foreground_norm < 5.0 or background_norm < 5.0 or foreground_offscreen:
        return 0.0
    cosine = float(np.dot(foreground_direction, background_direction)) / (
        foreground_norm * background_norm
    )
    return (1.0 - float(np.clip(cosine, -1.0, 1.0))) / 2.0


def _map_tracks_between_grids(
    tracks: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Map video-pixel tracks to a pointmap grid without changing endpoints."""
    source_height, source_width = source_hw
    target_height, target_width = target_hw
    if min(source_height, source_width, target_height, target_width) < 1:
        raise ValueError("track and pointmap dimensions must be positive")
    mapped = np.asarray(tracks, dtype=np.float64).copy()
    mapped[..., 0] *= (
        (target_width - 1) / (source_width - 1) if source_width > 1 else 0.0
    )
    mapped[..., 1] *= (
        (target_height - 1) / (source_height - 1) if source_height > 1 else 0.0
    )
    return mapped


def evaluate_object_metrics(
    *,
    object_name: str,
    masks: np.ndarray,
    h_pixel: np.ndarray,
    depth_z: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    background_tracks: np.ndarray,
    pointmaps: np.ndarray,
    focal_length: float,
    fps: float,
    lsd_frames: np.ndarray | None,
    lsd_exclusion_masks: np.ndarray,
    weights: dict[str, float],
    rigidity_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the existing metrics with one explicit, versioned rigidity input."""
    frame_count = min(
        len(masks), len(h_pixel), len(depth_z), len(tracks), len(visibility), len(pointmaps)
    )
    if frame_count < 2:
        raise ValueError(f"{object_name} has fewer than two common metric frames")
    masks = masks[:frame_count]
    h_pixel = h_pixel[:frame_count]
    depth_z = depth_z[:frame_count]
    tracks = tracks[:frame_count]
    visibility = visibility[:frame_count]
    pointmaps = pointmaps[:frame_count]

    camera = CameraModel(focal_length=focal_length, image_size=masks.shape[1:])
    foreground_ntd = tracks.transpose(1, 0, 2)
    background_ntd = (
        background_tracks[:frame_count].transpose(1, 0, 2)
        if background_tracks.ndim == 3 and background_tracks.shape[1] >= 2
        else None
    )
    projection = ProjectionJudge(cx=camera.cx, cy=camera.cy)
    global_vp, foreground_vp, background_vp = projection.estimate_vanishing_point_v2(
        fg_tracks=foreground_ntd,
        bg_tracks=background_ntd,
        frames=lsd_frames,
        masks=(
            lsd_exclusion_masks[:len(lsd_frames)]
            if lsd_frames is not None
            else None
        ),
    )
    trajectory_vp = foreground_vp
    if foreground_vp == (camera.cx, camera.cy) or _vp_in_object_bbox(
        foreground_vp, masks
    ):
        trajectory_vp = background_vp
    epsilon_vp = _vp_epsilon(
        foreground_vp, background_vp, camera, masks.shape[1:]
    )
    effective_vp = epsilon_vp if background_vp != (camera.cx, camera.cy) else 0.0

    scale_history = audit_scale_consistency(h_pixel, depth_z)
    trajectory_history = audit_3d_trajectory_consistency(pointmaps, masks, fps=fps)
    if rigidity_selection is None:
        rigidity_tracks = _map_tracks_between_grids(
            tracks,
            source_hw=masks.shape[1:],
            target_hw=pointmaps.shape[1:3],
        )
        rigidity, rigidity_history, rigidity_strategy = audit_3d_volume_stability(
            pointmaps,
            masks,
            tracks=rigidity_tracks,
            h_seq=h_pixel,
            visibility=visibility,
        )
        rigidity_metadata = {}
    else:
        required = {"method", "value", "history"}
        missing = sorted(required.difference(rigidity_selection))
        if missing:
            raise ValueError(f"rigidity selection lacks fields: {missing}")
        rigidity = float(rigidity_selection["value"])
        rigidity_history = np.asarray(
            rigidity_selection["history"], dtype=np.float64
        )
        if not np.isfinite(rigidity) or rigidity < 0.0:
            raise ValueError("selected rigidity value must be finite and non-negative")
        rigidity_strategy = str(rigidity_selection["method"])
        rigidity_metadata = dict(rigidity_selection.get("metadata", {}))
    calculator = PDIIndexCalculator(
        w_scale=weights.get("w_scale", 0.3),
        w_traj=weights.get("w_trajectory", 0.3),
        w_rigidity=weights.get("w_rigidity", 0.2),
        w_vp=weights.get("w_vp", 0.2),
    )
    report = calculator.compute_pdi(
        scale_history, trajectory_history, rigidity, effective_vp
    )
    scale_jump, scale_jump_pass = audit_scale_jump(pointmaps[..., 2], masks)
    report["object_name"] = object_name
    report["breakdown"].update(
        {
            "scale_history": scale_history,
            "traj_history": trajectory_history,
            "volume_history": rigidity_history,
            "rigidity_strategy": rigidity_strategy,
            "rigidity_metadata": rigidity_metadata,
        }
    )
    report.update(
        {
            "vanishing_point": global_vp,
            "foreground_vanishing_point": foreground_vp,
            "background_vanishing_point": background_vp,
            "trajectory_vanishing_point": trajectory_vp,
            "tracking": {
                "foreground_track_count": int(tracks.shape[1]),
                "background_track_count": int(background_tracks.shape[1]),
                "mean_visibility": float(visibility.mean()) if visibility.size else 0.0,
                "video_coordinate_hw": list(masks.shape[1:]),
                "rigidity_coordinate_hw": list(pointmaps.shape[1:3]),
            },
            "reconstruction_audit": {
                "scale_jump": scale_jump,
                "scale_jump_pass": scale_jump_pass,
            },
        }
    )
    return report


def save_track_result(path: str | Path, result: MultiObjectTrackResult) -> None:
    """Persist variable-size object groups without pickle/object arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    for tracks in result.object_tracks:
        offsets.append(offsets[-1] + tracks.shape[1])
    all_tracks = np.concatenate(result.object_tracks, axis=1)
    all_visibility = np.concatenate(result.object_visibility, axis=1)
    all_queries = np.concatenate(result.object_queries, axis=0)
    query_ids = (
        np.concatenate(result.object_query_ids)
        if result.object_query_ids
        else np.concatenate(
            [np.arange(len(queries), dtype=np.int32) for queries in result.object_queries]
        )
    )
    query_object_ids = np.concatenate(
        [
            np.full(len(queries), object_index, dtype=np.int16)
            for object_index, queries in enumerate(result.object_queries)
        ]
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        mode=np.asarray(result.mode),
        object_names=np.asarray(result.object_names),
        object_offsets=np.asarray(offsets, dtype=np.int64),
        tracks=all_tracks,
        visibility=all_visibility,
        queries=all_queries,
        query_ids=query_ids,
        query_object_ids=query_object_ids,
        background_tracks=result.background_tracks,
        background_visibility=result.background_visibility,
        background_queries=result.background_queries,
        metadata_json=np.asarray(json.dumps(_jsonable(result.metadata), sort_keys=True)),
    )
    temporary.replace(path)


def compare_mode_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not all(mode in reports for mode in TRACKING_MODES):
        return None
    joint = reports["joint-query"]
    exact = reports["exact-group"]
    if joint["objects"].keys() != exact["objects"].keys():
        raise ValueError("tracking mode reports contain different object identities")
    comparison: dict[str, Any] = {"objects": {}}
    for name in joint["objects"]:
        joint_object = joint["objects"][name]
        exact_object = exact["objects"][name]
        if (
            joint_object.get("status") != "complete"
            or exact_object.get("status") != "complete"
        ):
            comparison["objects"][name] = {
                "status": "unavailable",
                "error": "one or both tracking modes are not scorable",
            }
            continue
        comparison["objects"][name] = {
            "status": "complete",
            "pdi_score_exact_minus_joint": exact_object["pdi_score"] - joint_object["pdi_score"],
            "scale_exact_minus_joint": (
                exact_object["breakdown"]["scale_component"]
                - joint_object["breakdown"]["scale_component"]
            ),
            "trajectory_exact_minus_joint": (
                exact_object["breakdown"]["traj_component"]
                - joint_object["breakdown"]["traj_component"]
            ),
            "rigidity_exact_minus_joint": (
                exact_object["breakdown"]["epsilon_rigidity"]
                - joint_object["breakdown"]["epsilon_rigidity"]
            ),
            "vp_exact_minus_joint": (
                exact_object["breakdown"]["vp_component"]
                - joint_object["breakdown"]["vp_component"]
            ),
            "grade_changed": exact_object["grade"] != joint_object["grade"],
        }
    joint_seconds = joint["timing"]["tracking"]["model_seconds"]
    exact_seconds = exact["timing"]["tracking"]["model_seconds"]
    joint_total = joint["timing"]["tracking"]["total_tracking_seconds"]
    exact_total = exact["timing"]["tracking"]["total_tracking_seconds"]
    joint_memory = joint["timing"]["tracking"]["peak_gpu_memory_bytes"]
    exact_memory = exact["timing"]["tracking"]["peak_gpu_memory_bytes"]
    comparison["speed"] = {
        "joint_model_seconds": joint_seconds,
        "exact_model_seconds": exact_seconds,
        "exact_over_joint_ratio": exact_seconds / max(joint_seconds, 1e-12),
        "joint_speedup_over_exact": exact_seconds / max(joint_seconds, 1e-12),
        "joint_total_tracking_seconds": joint_total,
        "exact_total_tracking_seconds": exact_total,
        "exact_total_over_joint_ratio": exact_total / max(joint_total, 1e-12),
        "joint_peak_gpu_memory_bytes": joint_memory,
        "exact_peak_gpu_memory_bytes": exact_memory,
        "exact_minus_joint_peak_gpu_memory_bytes": exact_memory - joint_memory,
        "exact_over_joint_peak_memory_ratio": (
            exact_memory / joint_memory if joint_memory > 0 else None
        ),
    }
    return comparison


def _compare_track_group(
    joint_tracks: np.ndarray,
    joint_visibility: np.ndarray,
    joint_queries: np.ndarray,
    exact_tracks: np.ndarray,
    exact_visibility: np.ndarray,
    exact_queries: np.ndarray,
    joint_query_ids: np.ndarray | None = None,
    exact_query_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    if joint_query_ids is not None and exact_query_ids is not None:
        joint_ids = np.asarray(joint_query_ids, dtype=np.int64)
        exact_ids = np.asarray(exact_query_ids, dtype=np.int64)
        if joint_ids.shape != (len(joint_queries),) or exact_ids.shape != (
            len(exact_queries),
        ):
            raise ValueError("query IDs must align with retained query arrays")
        joint_index = {int(query_id): index for index, query_id in enumerate(joint_ids)}
        exact_index = {int(query_id): index for index, query_id in enumerate(exact_ids)}
    else:
        def query_key(query: np.ndarray) -> tuple[float, float, float]:
            return tuple(round(float(value), 3) for value in query)

        joint_index = {query_key(query): index for index, query in enumerate(joint_queries)}
        exact_index = {query_key(query): index for index, query in enumerate(exact_queries)}
    common = sorted(set(joint_index).intersection(exact_index))
    if not common:
        return {
            "common_query_count": 0,
            "joint_retained_count": len(joint_queries),
            "exact_retained_count": len(exact_queries),
            "mean_track_l2_pixels": None,
            "endpoint_l2_pixels": None,
            "visibility_agreement": None,
        }
    joint_selector = [joint_index[key] for key in common]
    exact_selector = [exact_index[key] for key in common]
    frame_count = min(len(joint_tracks), len(exact_tracks))
    deltas = np.linalg.norm(
        joint_tracks[:frame_count, joint_selector]
        - exact_tracks[:frame_count, exact_selector],
        axis=-1,
    )
    joint_vis = joint_visibility[:frame_count, joint_selector] > 0.5
    exact_vis = exact_visibility[:frame_count, exact_selector] > 0.5
    return {
        "common_query_count": len(common),
        "joint_retained_count": len(joint_queries),
        "exact_retained_count": len(exact_queries),
        "mean_track_l2_pixels": float(deltas.mean()),
        "endpoint_l2_pixels": float(deltas[-1].mean()),
        "visibility_agreement": float(np.mean(joint_vis == exact_vis)),
    }


def compare_track_results(
    results: dict[str, MultiObjectTrackResult],
) -> dict[str, Any] | None:
    if not all(mode in results for mode in TRACKING_MODES):
        return None
    joint = results["joint-query"]
    exact = results["exact-group"]
    if joint.object_names != exact.object_names:
        raise ValueError("tracking modes contain different object identities")
    objects = {}
    for index, name in enumerate(joint.object_names):
        objects[name] = _compare_track_group(
            joint.object_tracks[index],
            joint.object_visibility[index],
            joint.object_queries[index],
            exact.object_tracks[index],
            exact.object_visibility[index],
            exact.object_queries[index],
            (
                joint.object_query_ids[index]
                if joint.object_query_ids
                else None
            ),
            (
                exact.object_query_ids[index]
                if exact.object_query_ids
                else None
            ),
        )
    return {
        "objects": objects,
        "background": _compare_track_group(
            joint.background_tracks,
            joint.background_visibility,
            joint.background_queries,
            exact.background_tracks,
            exact.background_visibility,
            exact.background_queries,
        ),
    }


class MultiObjectPDIEvaluationPipeline:
    """One reconstruction and one query manifest, with isolated per-link metrics."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run(
        self,
        *,
        video_path: str,
        segmentation_npz: str,
        tracking_modes: Iterable[str] = TRACKING_MODES,
        output_dir: str | Path | None = None,
        geometry_cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        modes = tuple(dict.fromkeys(tracking_modes))
        invalid = [mode for mode in modes if mode not in TRACKING_MODES]
        if invalid or not modes:
            raise ValueError(f"invalid tracking modes: {invalid or modes}")
        output_dir = Path(output_dir).resolve() if output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        segmentation = load_multi_object_segmentation(segmentation_npz, video_path)
        segmentation_seconds = time.perf_counter() - started

        started = time.perf_counter()
        geometry_engine = MegaSamWrapper(device=self.config.get("device", "cuda"))
        geometry = geometry_engine.infer_shared(
            video_path,
            segmentation.object_masks,
            cache_dir=geometry_cache_dir,
        )
        geometry_seconds = time.perf_counter() - started
        lsd_frames, fps = _read_lsd_frames(video_path)

        cad_config = dict(self.config.get("cad_canonicalization", {}))
        foundation_pose_seconds = 0.0
        if cad_config.get("enabled", False) and cad_config.get("auto_run", False):
            if output_dir is None or geometry.cache_path is None:
                raise ValueError(
                    "automatic FoundationPose requires output_dir and geometry_cache_dir"
                )
            pose_archive = cad_config.get("foundation_pose_archive")
            pose_path = (
                Path(pose_archive).resolve()
                if pose_archive and Path(pose_archive).is_absolute()
                else output_dir / (pose_archive or "foundationpose_poses.npz")
            )
            manifest_value = Path(
                cad_config.get("cad_manifest", "configs/sam3-cad-franka.yaml")
            )
            manifest_path = (
                manifest_value.resolve()
                if manifest_value.is_absolute()
                else (BENCHMARK_ROOT / manifest_value).resolve()
            )
            pose_started = time.perf_counter()
            if cad_config.get("force_foundation_pose", False) or not pose_path.is_file():
                run_foundation_pose_worker(
                    geometry_npz=geometry.cache_path,
                    segmentation_npz=segmentation_npz,
                    cad_manifest=manifest_path,
                    output_npz=pose_path,
                    config=cad_config,
                    benchmark_root=BENCHMARK_ROOT,
                )
            foundation_pose_seconds = time.perf_counter() - pose_started
            cad_config["foundation_pose_archive"] = str(pose_path)

        started = time.perf_counter()
        tracker = TrackWrapper(
            checkpoint=self.config.get("tracker_ckpt"),
            device=self.config.get("device", "cuda"),
        )
        tracker_load_seconds = time.perf_counter() - started
        tracking_config = self.config.get("multi_object_tracking", {})
        prepared = tracker.prepare_multi(
            video_path,
            segmentation.object_masks[0],
            segmentation.object_names,
            grid_size=int(tracking_config.get("grid_size", 10)),
            bg_grid_size=int(tracking_config.get("background_grid_size", 15)),
            background_dilation=int(tracking_config.get("background_dilation", 5)),
            max_dim=int(tracking_config.get("max_dimension", 880)),
            object_query_counts=tracking_config.get("object_query_counts"),
        )
        cad_started = time.perf_counter()
        cad_runtime = CadCanonicalizationRuntime.prepare(
            config=cad_config,
            project_root=BENCHMARK_ROOT,
            segmentation=segmentation,
            geometry=geometry,
            prepared_tracking=prepared,
        )
        cad_setup_seconds = time.perf_counter() - cad_started

        mode_reports: dict[str, dict[str, Any]] = {}
        track_results: dict[str, MultiObjectTrackResult] = {}
        try:
            for mode in modes:
                track_result = tracker.track_prepared(prepared, mode)
                track_results[mode] = track_result
                metric_started = time.perf_counter()
                object_reports = {}
                for object_index, object_name in enumerate(segmentation.object_names):
                    cad_report = (
                        cad_runtime.audit_track_result(
                            geometry=geometry,
                            track_result=track_result,
                            object_index=object_index,
                            object_name=object_name,
                        )
                        if cad_runtime is not None
                        and object_name in cad_runtime.link_names
                        else None
                    )
                    depth_metadata = geometry.metadata["object_depth"][object_index]
                    if depth_metadata["status"] == "failed":
                        object_reports[object_name] = {
                            "object_name": object_name,
                            "status": "failed",
                            "error_type": "insufficient_target_depth",
                            "error": depth_metadata["error"],
                            "depth": depth_metadata,
                            "cad_rigidity": cad_report,
                        }
                        continue
                    rigidity_selection = None
                    rigidity_method = self.config.get("cad_canonicalization", {}).get(
                        "rigidity_method", "legacy-point-pair"
                    )
                    if rigidity_method not in (
                        "legacy-point-pair",
                        "cad-canonical-v1",
                    ):
                        raise ValueError(f"unknown rigidity method: {rigidity_method}")
                    if (
                        rigidity_method == "cad-canonical-v1"
                        and object_name in CAD_LINK_NAMES
                    ):
                        if cad_report is None or cad_report.get("status") != "complete":
                            object_reports[object_name] = {
                                "object_name": object_name,
                                "status": (
                                    cad_report.get("status", "unscorable")
                                    if cad_report is not None
                                    else "unscorable"
                                ),
                                "error_type": "cad_rigidity_not_calibrated_or_scorable",
                                "error": (
                                    cad_report.get("error")
                                    if cad_report is not None
                                    else "CAD rigidity is unavailable"
                                ),
                                "depth": depth_metadata,
                                "cad_rigidity": cad_report,
                            }
                            continue
                        rigidity_selection = {
                            "method": "cad-canonical-v1",
                            "value": cad_report["epsilon_cad_mean"],
                            "history": cad_report["epsilon_cad_frame"],
                            "metadata": {
                                "threshold_status": cad_report["status"],
                                "mean_threshold": cad_report["mean_threshold"],
                                "p90_threshold": cad_report["p90_threshold"],
                            },
                        }
                    object_report = evaluate_object_metrics(
                        object_name=object_name,
                        masks=segmentation.object_masks[:, object_index],
                        h_pixel=segmentation.h_pixel[:, object_index],
                        depth_z=geometry.object_depth_z[:, object_index],
                        tracks=track_result.object_tracks[object_index],
                        visibility=track_result.object_visibility[object_index],
                        background_tracks=track_result.background_tracks,
                        pointmaps=geometry.pointmaps,
                        focal_length=geometry.focal_length,
                        fps=fps,
                        lsd_frames=lsd_frames,
                        lsd_exclusion_masks=segmentation.union_masks,
                        weights=self.config.get("weights", {}),
                        rigidity_selection=rigidity_selection,
                    )
                    object_report["status"] = "complete"
                    object_report["depth"] = depth_metadata
                    object_report["cad_rigidity"] = cad_report
                    if (
                        cad_runtime is not None
                        and object_name in cad_runtime.pose_discontinuity
                    ):
                        object_report["pose_discontinuity"] = (
                            cad_runtime.pose_discontinuity[object_name]
                        )
                    object_reports[object_name] = object_report
                metric_seconds = time.perf_counter() - metric_started
                mode_report = {
                    "tracking_mode": mode,
                    "objects": object_reports,
                    "timing": {
                        "tracking": track_result.metadata,
                        "metrics_seconds": metric_seconds,
                    },
                }
                mode_reports[mode] = mode_report
                if output_dir is not None:
                    save_track_result(output_dir / f"cotracker_{mode}.npz", track_result)
        finally:
            del prepared.video_tensor
            if tracker.device.type == "cuda":
                import torch

                torch.cuda.empty_cache()

        ground_rmse, ground_pass = audit_ground_flatness(
            geometry.pointmaps,
            segmentation.union_masks[:geometry.frames_count],
        )
        comparison = compare_mode_reports(mode_reports)
        if comparison is not None:
            comparison["tracking"] = compare_track_results(track_results)
        report = {
            "schema_version": 1,
            "video": str(Path(video_path).resolve()),
            "segmentation": {
                "archive": str(Path(segmentation_npz).resolve()),
                "object_names": segmentation.object_names,
                "object_ids": segmentation.object_ids,
                "object_count": segmentation.object_count,
                "frames_count": segmentation.frames_count,
                "overlap_pixel_count": segmentation.metadata["overlap_pixel_count"],
            },
            "geometry": {
                "cache_path": geometry.cache_path,
                "cache_hit": geometry.metadata["cache_hit"],
                "frames_count": geometry.frames_count,
                "pointmap_shape": geometry.pointmaps.shape,
                "rgb_camera_shape": (
                    geometry.rgb_camera.shape
                    if geometry.rgb_camera is not None
                    else None
                ),
                "depth_camera_shape": (
                    geometry.depth_camera.shape
                    if geometry.depth_camera is not None
                    else None
                ),
                "shared_world_frame": True,
            },
            "cad_canonicalization": (
                {"enabled": False}
                if cad_runtime is None
                else cad_runtime.summary()
            ),
            "shared_reconstruction_audit": {
                "ground_rmse": ground_rmse,
                "ground_pass": ground_pass,
                "foreground_exclusion": "union of all object masks",
            },
            "modes": mode_reports,
            "comparison": comparison,
            "timing": {
                "segmentation_load_seconds": segmentation_seconds,
                "geometry_seconds": geometry_seconds,
                "tracker_load_seconds": tracker_load_seconds,
                "query_preparation_seconds": prepared.timings["decode_and_query_seconds"],
                "cad_setup_seconds": cad_setup_seconds,
                "foundation_pose_seconds": foundation_pose_seconds,
                "total_seconds": time.perf_counter() - total_started,
            },
        }
        pdi_logger.info(
            f"Multi-object PDI complete: {segmentation.object_count} objects, "
            f"modes={','.join(modes)}"
        )
        return _jsonable(report)


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
