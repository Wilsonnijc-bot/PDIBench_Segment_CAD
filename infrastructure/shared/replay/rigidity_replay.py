"""Export interactive V1 rigidity replays from verified scoring evidence."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from infrastructure.shared.replay.reconstruct_replay import map_xy_between_grids, transform_world_to_camera


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def build_scene_context(
    masks: np.ndarray, pointmaps: np.ndarray, camera_poses: np.ndarray,
    video_path: Path, max_cloud_points: int,
) -> tuple[list, list, list, list, list]:
    """Sample the source-colored scene and robot in each source camera frame."""
    height, width = pointmaps.shape[1:3]
    rng = np.random.default_rng(17)
    robot_clouds, robot_colors, scene_clouds, scene_colors = [], [], [], []
    lows, highs = [], []
    capture = cv2.VideoCapture(str(video_path))
    try:
        for frame in range(len(pointmaps)):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"source video ended before replay frame {frame}")
            rgb = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
            robot = np.any(masks[frame], axis=0)
            robot = cv2.resize(robot.astype(np.uint8), (width, height),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            valid = np.isfinite(pointmaps[frame]).all(axis=2) & np.any(
                np.abs(pointmaps[frame]) > 1e-12, axis=2)
            sampled = []
            for region, limit, clouds, colors in (
                (valid & ~robot, max(900, max_cloud_points), scene_clouds, scene_colors),
                (valid & robot, max(2600, max_cloud_points * 3), robot_clouds, robot_colors),
            ):
                pixels = np.flatnonzero(region)
                if len(pixels) > limit:
                    pixels = rng.choice(pixels, limit, replace=False)
                yy, xx = np.unravel_index(pixels, (height, width))
                world = pointmaps[frame, yy, xx]
                camera = transform_world_to_camera(world, camera_poses[frame])
                clouds.append(np.round(camera, 4).tolist())
                colors.append([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb[yy, xx]])
                if len(camera):
                    sampled.append(camera)
            if sampled:
                all_points = np.concatenate(sampled)
                lows.append(np.percentile(all_points, 0.5, axis=0))
                highs.append(np.percentile(all_points, 99.5, axis=0))
    finally:
        capture.release()
    if not lows:
        raise ValueError("no valid scene point cloud for interactive replay")
    low = np.min(lows, axis=0)
    high = np.max(highs, axis=0)
    padding = max(float(np.max(high - low)) * 0.08, 0.02)
    bounds = np.stack([low - padding, high + padding], axis=1)
    return robot_clouds, robot_colors, scene_clouds, scene_colors, np.round(bounds, 5).tolist()


def export_v1_replay(
    metrics_path: Path, tracks_path: Path, segmentation_path: Path,
    pointmaps_path: Path, video_path: Path, output_dir: Path,
    *, mode: str = "exact-group", max_cloud_points: int = 900,
    plotly_js: Path, only_objects: tuple[str, ...] | None = None,
    tracker_label: str = "CoTracker", mask_label: str | None = None,
) -> list[Path]:
    """Recover V1's actual frame-0 pairs and verify their saved score history."""
    from infrastructure.shared.scoring.rigidity.rigidity import audit_3d_rigidity_cv

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(tracks_path, allow_pickle=False) as archive:
        names = [str(name) for name in archive["object_names"]]
        offsets = archive["object_offsets"]
        tracks = archive["tracks"]
        visibility = archive["visibility"]
        point_ids = archive["point_ids"] if "point_ids" in archive else np.concatenate(
            [np.arange(int(offsets[i + 1] - offsets[i]), dtype=np.int64)
             for i in range(len(names))])
    with np.load(segmentation_path, allow_pickle=False) as archive:
        masks = archive["object_masks"]
        mask_names = [str(name) for name in archive["object_names"]]
    with np.load(pointmaps_path, allow_pickle=False) as archive:
        pointmaps = archive["pointmaps"]
        camera_poses = archive["camera_poses"]
        focal_length = float(archive["focal_length"])
    if names != mask_names or names != report["segmentation"]["object_names"]:
        raise ValueError("V1 object order differs across metrics, masks, and tracks")
    if only_objects is not None and not set(only_objects).issubset(names):
        raise ValueError("requested replay object is absent from track archive")
    if len(pointmaps) != len(masks) or len(tracks) != len(masks):
        raise ValueError("V1 frame counts differ across pointmaps, masks, and tracks")
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or video_frames != len(masks):
        raise ValueError("V1 source video FPS or frame count disagrees with artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if only_objects is not None and name not in only_objects:
            continue
        for suffix in (".html", "_pairs.json"):
            (output_dir / f"{name}_{mode}{suffix}").unlink(missing_ok=True)
    for source, name in ((plotly_js, "plotly.min.js"), (video_path, "source.mp4")):
        target = output_dir / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    template = (_workspace_root() / 'infrastructure/shared/replay/rigidity_replay.html').read_text(encoding="utf-8")
    template = template.replace("CoTracker", tracker_label)
    robot_clouds, robot_colors, scene_clouds, scene_colors, scene_bounds = build_scene_context(
        masks, pointmaps, camera_poses, video_path, max_cloud_points)
    metrics_sha256 = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    created = []
    height, width = pointmaps.shape[1:3]
    rng = np.random.default_rng(17)
    for index, name in enumerate(names):
        if only_objects is not None and name not in only_objects:
            continue
        object_report = report["modes"][mode]["objects"][name]
        if object_report["status"] != "complete":
            continue
        if object_report["breakdown"]["rigidity_strategy"] != "Strategy 1 (3D rigid pairwise ratios)":
            continue
        object_tracks = tracks[:, offsets[index]:offsets[index + 1]]
        object_visibility = visibility[:, offsets[index]:offsets[index + 1]]
        object_point_ids = point_ids[offsets[index]:offsets[index + 1]]
        mapped = map_xy_between_grids(
            object_tracks, source_hw=masks.shape[2:4], target_hw=(height, width),
        )
        evidence: dict = {}
        score, history = audit_3d_rigidity_cv(
            pointmaps, mapped, object_visibility, masks[:, index],
            insufficient_policy="raise", evidence=evidence,
            point_filter_version=(report.get("link7_point_filter", "v1")
                                  if name == "link7" else "v1"),
        )
        saved = np.asarray(object_report["breakdown"]["volume_history"])
        if not np.allclose(history, saved, rtol=1e-8, atol=1e-10):
            raise ValueError(f"V1 pair reconstruction differs from scored history: {name}")
        frame_points = []
        clouds = []
        for frame in range(len(pointmaps)):
            xy = mapped[frame]
            if not np.isfinite(xy).all():
                raise ValueError(f"V1 track coordinates are nonfinite: {name}, frame {frame}")
            u = np.clip(np.round(xy[:, 0]).astype(int), 0, width - 1)
            v = np.clip(np.round(xy[:, 1]).astype(int), 0, height - 1)
            frame_points.append([
                np.round(transform_world_to_camera(
                    pointmaps[frame, v[i], u[i]], camera_poses[frame]), 5).tolist()
                if object_visibility[frame, i] > 0.5 else None
                for i in range(len(u))
            ])
            mask = cv2.resize(masks[frame, index].astype(np.uint8),
                              (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            cloud = pointmaps[frame][mask]
            cloud = cloud[np.isfinite(cloud).all(axis=1) & np.any(np.abs(cloud) > 1e-12, axis=1)]
            if len(cloud) > max_cloud_points:
                cloud = cloud[rng.choice(len(cloud), max_cloud_points, replace=False)]
            clouds.append(np.round(transform_world_to_camera(cloud, camera_poses[frame]), 4).tolist())
        evidence_name = f"{name}_{mode}_pairs.json"
        (output_dir / evidence_name).write_text(json.dumps({
            "schema_version": 1, "source_metrics_sha256": metrics_sha256,
            "object": name, "mode": mode,
            "version": (report.get("link7_point_filter", "v1") if name == "link7" else "v1"),
            "original_point_ids": object_point_ids.astype(int).tolist(),
            "pdi_score": object_report["pdi_score"], **evidence,
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        window = {"start": 0, "end": len(pointmaps), "query_frame": 0,
                  "selected_pairs": evidence["selected_pairs"],
                  "pair_frames": evidence["pair_frames"]}
        carried = set(evidence["carried_frames"])
        data = {"object": name, "mode": mode,
                "version": (report.get("link7_point_filter", "v1").upper()
                            if name == "link7" else "V1"),
                "tracker_label": tracker_label, "mask_label": mask_label, "fps": fps,
                "focal_length": focal_length, "image_hw": [height, width],
                "score": score, "pdi_score": object_report["pdi_score"],
                "history": history.tolist(),
                "observed": [None if frame == 0 or frame in carried else float(history[frame])
                             for frame in range(len(history))],
                "carried_frames": evidence["carried_frames"],
                "coverage": len(evidence["pair_frames"]) / max(1, len(history) - 1),
                "point_ids": object_point_ids.astype(int).tolist(),
                "windows": [window], "points": frame_points, "clouds": clouds,
                "robot_clouds": robot_clouds, "robot_colors": robot_colors,
                "scene_clouds": scene_clouds, "scene_colors": scene_colors,
                "scene_bounds": scene_bounds,
                "evidence_file": evidence_name}
        target = output_dir / f"{name}_{mode}.html"
        target.write_text(template.replace("__REPLAY_DATA__", _json(data)), encoding="utf-8")
        created.append(target)
    links = "\n".join(
        f'<li><a href="{page.name}">{page.stem}</a> · '
        f'<a href="{page.stem}_pairs.json">scored pairs JSON</a></li>'
        for page in created
    )
    (output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>V1 rigidity pair replays</title>'
        f'<h1>V1 rigidity pair replays</h1><ul>{links}</ul>', encoding="utf-8",
    )
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metrics", "tracks", "segmentation", "pointmaps", "video", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--plotly-js", type=Path,
        default=(_workspace_root() / 'infrastructure/shared/replay/assets/plotly.min.js'),
    )
    parser.add_argument("--mode", default="exact-group")
    parser.add_argument("--max-cloud-points", type=int, default=900)
    args = parser.parse_args(argv)
    if args.max_cloud_points < 0:
        parser.error("--max-cloud-points must be nonnegative")
    created = export_v1_replay(
        args.metrics, args.tracks, args.segmentation, args.pointmaps,
        args.video, args.output_dir, mode=args.mode,
        max_cloud_points=args.max_cloud_points, plotly_js=args.plotly_js,
    )
    print("\n".join(map(str, created)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
