"""Video-level PDI evaluation for multiple independently rigid targets."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from infrastructure.shared.scoring.pdi.motion_audit import audit_3d_trajectory_consistency
from infrastructure.shared.scoring.pdi.reconstruction_audit import audit_ground_flatness, audit_scale_jump
from infrastructure.shared.scoring.pdi.scale_audit import audit_scale_consistency
from infrastructure.shared.scoring.rigidity.rigidity import InsufficientRigidityEvidenceError, audit_3d_volume_stability
from infrastructure.shared.geometry.camera import CameraModel
from infrastructure.shared.geometry.projection import ProjectionJudge
from infrastructure.shared.scoring.pdi.pdi_index import PDIIndexCalculator
from infrastructure.shared.contracts.base import MultiObjectTrackResult
from infrastructure.shared.inference.mega_sam_wrapper import MegaSamWrapper
from infrastructure.shared.contracts.segmentation_archive import load_multi_object_segmentation
from infrastructure.shared.inference.tracking import TRACKING_MODES, TrackWrapper
from infrastructure.shared.io.logger import pdi_logger
from robot.scoring.metrics import _vp_in_object_bbox, _vp_epsilon, _map_tracks_between_grids, evaluate_object_metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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
    all_point_ids = np.concatenate(result.object_point_ids, axis=0) if result.object_point_ids else np.concatenate(
        [np.arange(len(queries), dtype=np.int64) for queries in result.object_queries]
    )
    temporary = path.with_suffix(".tmp.npz")
    extra = {}
    if result.link7_raw_tracks is not None:
        extra["link7_raw_tracks"] = result.link7_raw_tracks
        extra["link7_raw_visibility"] = result.link7_raw_visibility
    np.savez_compressed(
        temporary,
        mode=np.asarray(result.mode),
        object_names=np.asarray(result.object_names),
        object_offsets=np.asarray(offsets, dtype=np.int64),
        tracks=all_tracks,
        visibility=all_visibility,
        queries=all_queries,
        point_ids=all_point_ids,
        background_tracks=result.background_tracks,
        background_visibility=result.background_visibility,
        background_queries=result.background_queries,
        metadata_json=np.asarray(json.dumps(_jsonable(result.metadata), sort_keys=True)),
        **extra,
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
            joint_object.get("status") == "failed"
            or exact_object.get("status") == "failed"
        ):
            comparison["objects"][name] = {
                "status": "unavailable",
                "error": "one or both tracking modes lack sufficient target depth",
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
) -> dict[str, Any]:
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
        link7_tracker: str = "cotracker3",
        link7_point_filter: str = "v1",
        score_links: tuple[str, ...] | None = None,
        output_dir: str | Path | None = None,
        geometry_cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        modes = tuple(dict.fromkeys(tracking_modes))
        invalid = [mode for mode in modes if mode not in TRACKING_MODES]
        if invalid or not modes:
            raise ValueError(f"invalid tracking modes: {invalid or modes}")
        if link7_tracker not in ("cotracker3", "tapip3d"):
            raise ValueError(f"invalid Link 7 tracker: {link7_tracker}")
        if link7_point_filter not in ("v1", "v2"):
            raise ValueError(f"invalid Link 7 point filter: {link7_point_filter}")
        output_dir = Path(output_dir).resolve() if output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        segmentation = load_multi_object_segmentation(segmentation_npz, video_path)
        segmentation_seconds = time.perf_counter() - started
        requested_scores = set(score_links or segmentation.object_names)
        if not requested_scores or not requested_scores.issubset(segmentation.object_names):
            raise ValueError(f"invalid score links: {sorted(requested_scores)}")

        started = time.perf_counter()
        geometry_engine = MegaSamWrapper(device=self.config.get("device", "cuda"))
        geometry = geometry_engine.infer_shared(
            video_path,
            segmentation.object_masks,
            cache_dir=geometry_cache_dir,
        )
        geometry_seconds = time.perf_counter() - started
        lsd_frames, fps = _read_lsd_frames(video_path)

        started = time.perf_counter()
        tracker = TrackWrapper(
            checkpoint=self.config.get("tracker_ckpt"),
            device=self.config.get("device", "cuda"),
        )
        tracker_load_seconds = time.perf_counter() - started
        tracking_config = self.config.get("multi_object_tracking", {})
        configured_query_counts = tracking_config.get("object_query_counts") or {}
        object_query_counts = {
            name: count for name, count in configured_query_counts.items()
            if name in segmentation.object_names
        }
        skip_low_coverage_links = set(
            tracking_config.get("skip_low_coverage_links", [])
        )
        minimum_tracked_fraction = float(
            tracking_config.get("minimum_tracked_fraction", 0.80)
        )
        max_dimension = int(tracking_config.get("max_dimension", 880))
        depth_gate = None
        query_support_masks = None
        if link7_point_filter == "v2":
            from robot.experiments.link7_depth_filter_v2.link7_depth_gate import link7_query_mask
            link7_index = segmentation.object_names.index("link7")
            source_h, source_w = segmentation.object_masks.shape[-2:]
            scale = min(1.0, max_dimension / max(source_h, source_w))
            expected_tracker_hw = (int(source_h * scale), int(source_w * scale))
            support, depth_gate = link7_query_mask(
                geometry.pointmaps[0], geometry.camera_poses[0],
                segmentation.object_masks[0, link7_index],
                expected_tracker_hw,
            )
            query_support_masks = {"link7": support}
        prepared = tracker.prepare_multi(
            video_path,
            segmentation.object_masks[0],
            segmentation.object_names,
            grid_size=int(tracking_config.get("grid_size", 10)),
            bg_grid_size=int(tracking_config.get("background_grid_size", 15)),
            background_dilation=int(tracking_config.get("background_dilation", 5)),
            max_dim=max_dimension,
            object_query_counts=object_query_counts,
            allow_empty_names=skip_low_coverage_links,
            query_support_masks=query_support_masks,
        )
        if link7_point_filter == "v2":
            if prepared.tracker_hw != expected_tracker_hw:
                raise ValueError("video and segmentation dimensions disagree for Link 7 queries")
            index = prepared.object_names.index("link7")
            depth_gate["requested_query_count"] = prepared.requested_object_query_counts[index]
            depth_gate["selected_query_count"] = len(prepared.object_queries[index])
        if output_dir is not None and "link7" in prepared.object_names:
            from robot.experiments.link7_tapip3d.tapip3d_link7 import save_initial_queries
            save_initial_queries(output_dir / "link7_initial_queries.npz", prepared)
        cotracker_prepared = prepared
        if link7_tracker == "tapip3d":
            index = prepared.object_names.index("link7")
            groups = list(prepared.object_queries)
            groups[index] = np.empty((0, 3), dtype=np.float32)
            cotracker_prepared = replace(prepared, object_queries=tuple(groups))

        mode_reports: dict[str, dict[str, Any]] = {}
        track_results: dict[str, MultiObjectTrackResult] = {}
        try:
            for mode in modes:
                track_result = tracker.track_prepared(cotracker_prepared, mode)
                if link7_tracker == "tapip3d":
                    if output_dir is None:
                        raise ValueError("TAPIP3D requires an output directory for its 3D track archive")
                    from robot.experiments.link7_tapip3d.tapip3d_link7 import replace_link7_with_tapip3d
                    tapip = self.config.get("tapip3d", {})
                    track_result = replace_link7_with_tapip3d(
                        baseline=track_result, prepared=prepared, geometry=geometry,
                        tracker=tracker, output_dir=output_dir, python=tapip["python"],
                        repository=tapip["repository"], checkpoint=tapip["checkpoint"],
                        support_grid_size=int(tapip.get("support_grid_size", 16)),
                        num_iters=int(tapip.get("num_iters", 6)),
                        resolution_factor=int(tapip.get("resolution_factor", 2)),
                        visibility_threshold=float(tapip.get("visibility_threshold", 0.9)),
                    )
                track_results[mode] = track_result
                metric_started = time.perf_counter()
                object_reports = {}
                for object_index, object_name in enumerate(segmentation.object_names):
                    depth_metadata = geometry.metadata["object_depth"][object_index]
                    object_masks = segmentation.object_masks[:, object_index]
                    mask_areas = object_masks.reshape(len(object_masks), -1).sum(axis=1)
                    sam_tracked_fraction = float(np.mean(mask_areas > 0))
                    retained_track_count = int(
                        track_result.object_tracks[object_index].shape[1]
                    )
                    requested_track_count = int(
                        track_result.metadata["requested_foreground_query_counts"]
                        [object_index]
                    )
                    quality = {
                        "sam_tracked_fraction": sam_tracked_fraction,
                        "requested_foreground_query_count": requested_track_count,
                        "foreground_track_count": retained_track_count,
                        "retained_track_fraction": (
                            retained_track_count / requested_track_count
                            if requested_track_count
                            else 0.0
                        ),
                    }
                    if object_name not in requested_scores:
                        object_reports[object_name] = {
                            "object_name": object_name,
                            "status": "skipped",
                            "error_type": "outside_score_scope",
                            "depth": depth_metadata,
                            "tracking": quality,
                        }
                        continue
                    if (object_name in skip_low_coverage_links
                            and sam_tracked_fraction < minimum_tracked_fraction):
                        object_reports[object_name] = {
                            "object_name": object_name,
                            "status": "skipped",
                            "error_type": "insufficient_sam3_coverage",
                            "error": (
                                f"SAM3 mask covers {sam_tracked_fraction:.1%} of frames; "
                                f"minimum for scoring is {minimum_tracked_fraction:.1%}"
                            ),
                            "depth": depth_metadata,
                            "tracking": quality,
                        }
                        continue
                    if retained_track_count < 5:
                        object_reports[object_name] = {
                            "object_name": object_name,
                            "status": "failed",
                            "error_type": (
                                "insufficient_tapip3d_tracks"
                                if link7_tracker == "tapip3d" and object_name == "link7"
                                else "insufficient_cotracker_tracks"
                            ),
                            "error": (
                                f"only {retained_track_count}/{requested_track_count} "
                                f"{link7_tracker if object_name == 'link7' else 'CoTracker3'} points survived; need at least 5"
                            ),
                            "depth": depth_metadata,
                            "tracking": quality,
                        }
                        continue
                    if depth_metadata["status"] == "failed":
                        object_reports[object_name] = {
                            "object_name": object_name,
                            "status": "failed",
                            "error_type": "insufficient_target_depth",
                            "error": depth_metadata["error"],
                            "depth": depth_metadata,
                            "tracking": quality,
                        }
                        continue
                    try:
                        object_report = evaluate_object_metrics(
                            object_name=object_name,
                            masks=object_masks,
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
                            requested_track_count=requested_track_count,
                            point_filter_version=link7_point_filter,
                        )
                    except InsufficientRigidityEvidenceError as exc:
                        object_reports[object_name] = {
                            "object_name": object_name,
                            "status": "failed",
                            "error_type": "insufficient_rigidity_evidence",
                            "error": str(exc),
                            "depth": depth_metadata,
                            "tracking": {
                                **quality,
                                "valid_rigidity_anchor_count": exc.valid_anchor_count,
                                "valid_rigidity_pair_count": exc.valid_pair_count,
                            },
                        }
                        continue
                    object_report["status"] = "complete"
                    object_report["depth"] = depth_metadata
                    if object_name == "link7":
                        object_report["tracking"]["point_filter_version"] = link7_point_filter
                        object_report["tracking"]["initial_depth_gate"] = depth_gate
                    object_report["tracking"]["sam_tracked_fraction"] = (
                        sam_tracked_fraction
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
                    archive_prefix = "cotracker" if link7_tracker == "cotracker3" else "tapip3d_link7"
                    save_track_result(output_dir / f"{archive_prefix}_{mode}.npz", track_result)
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
            "link7_tracker": link7_tracker,
            "link7_point_filter": link7_point_filter,
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
                "shared_world_frame": True,
            },
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
