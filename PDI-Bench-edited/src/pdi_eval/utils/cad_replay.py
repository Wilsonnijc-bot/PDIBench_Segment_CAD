"""Render SAM, MegaSAM, FoundationPose, and CoTracker-to-CAD diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from ..evaluator.cad_rigidity_audit import (
    CadAnchorSet,
    ImageGridTransform,
    bind_cad_anchors,
    canonicalize_observed_sequence,
    erode_masks,
    sample_observed_track_points,
)
from ..geometry.cad_mesh import load_cad_manifest
from ..perception.foundation_pose_wrapper import load_foundation_pose_archive
from ..perception.segmentation_archive import load_multi_object_segmentation


CAD_LINK_NAMES = ("link2", "link3", "link4", "link5", "link6", "link7")
COLORS_BGR = (
    (70, 210, 255),
    (90, 220, 120),
    (240, 160, 70),
    (200, 100, 230),
    (80, 180, 245),
    (230, 210, 80),
)


def _encode_h264(source: Path, target: Path) -> None:
    """Finalize an OpenCV frame stream as a Mac-compatible H.264 MP4."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        "-colorspace",
        "bt709",
        "-y",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to finalize CAD replays") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg H.264 encoding failed: {exc.stderr.strip()}") from exc
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg did not create a valid replay: {target}")


def _label(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (16, 18, 22), -1)
    cv2.putText(
        image,
        text,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _fit_panel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _mask_overlay(frame_bgr: np.ndarray, masks: np.ndarray) -> np.ndarray:
    result = frame_bgr.copy()
    for index, mask in enumerate(masks):
        color = np.asarray(COLORS_BGR[index % len(COLORS_BGR)], dtype=np.float32)
        selector = np.asarray(mask, dtype=bool)
        result[selector] = np.clip(
            0.45 * result[selector] + 0.55 * color, 0, 255
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            selector.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, tuple(int(v) for v in color), 2)
    return result


def _project_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    panel_hw: tuple[int, int],
    bounds: tuple[np.ndarray, np.ndarray],
    point_radius: int = 1,
) -> np.ndarray:
    height, width = panel_hw
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    if not len(points):
        return canvas
    low, high = bounds
    span = np.maximum(high - low, 1e-9)
    x = np.clip(((points[:, 0] - low[0]) / span[0] * (width - 1)).round(), 0, width - 1)
    y = np.clip(((1.0 - (points[:, 1] - low[1]) / span[1]) * (height - 1)).round(), 0, height - 1)
    order = np.argsort(points[:, 2])[::-1]
    for index in order:
        cv2.circle(
            canvas,
            (int(x[index]), int(y[index])),
            point_radius,
            tuple(int(value) for value in colors[index]),
            -1,
            cv2.LINE_AA,
        )
    return canvas


def _rasterize_mesh(
    uv: np.ndarray,
    faces: np.ndarray,
    *,
    panel_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize projected triangles and return their silhouette and polygons."""
    height, width = panel_hw
    triangles = np.asarray(uv, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    finite = np.isfinite(triangles).all(axis=(1, 2))
    triangles = triangles[finite]
    if not len(triangles):
        return np.zeros((height, width), dtype=np.uint8), np.empty((0, 3, 2), dtype=np.int32)

    low = triangles.min(axis=1)
    high = triangles.max(axis=1)
    intersects = (
        (high[:, 0] >= 0)
        & (high[:, 1] >= 0)
        & (low[:, 0] < width)
        & (low[:, 1] < height)
    )
    triangles = triangles[intersects]
    if not len(triangles):
        return np.zeros((height, width), dtype=np.uint8), np.empty((0, 3, 2), dtype=np.int32)

    twice_area = np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    triangles = triangles[twice_area >= 0.5]
    if not len(triangles):
        return np.zeros((height, width), dtype=np.uint8), np.empty((0, 3, 2), dtype=np.int32)

    limit = 4 * max(height, width)
    polygons = np.rint(np.clip(triangles, -limit, limit)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255, lineType=cv2.LINE_8)
    return mask, polygons


def _rasterize_silhouette(
    uv: np.ndarray,
    *,
    panel_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a fast convex silhouette for a projected moving link."""
    height, width = panel_hw
    points = np.asarray(uv, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return np.zeros((height, width), dtype=np.uint8), np.empty((0, 1, 2), dtype=np.int32)
    limit = 4 * max(height, width)
    points = np.rint(np.clip(points, -limit, limit)).astype(np.int32)
    hull = cv2.convexHull(points)
    low = hull[:, 0].min(axis=0)
    high = hull[:, 0].max(axis=0)
    if high[0] < 0 or high[1] < 0 or low[0] >= width or low[1] >= height:
        return np.zeros((height, width), dtype=np.uint8), np.empty((0, 1, 2), dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_8)
    return mask, hull[None, :, 0, :]


def _composite_mesh(
    image: np.ndarray,
    mask: np.ndarray,
    _polygons: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
    outline_color: tuple[int, int, int],
) -> None:
    selector = mask > 0
    if not np.any(selector):
        return
    fill = np.asarray(color, dtype=np.float32)
    image[selector] = np.clip(
        (1.0 - alpha) * image[selector].astype(np.float32) + alpha * fill,
        0,
        255,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, outline_color, 2, cv2.LINE_AA)


def _scene_view_basis(pointmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = []
    for frame_index in np.linspace(0, len(pointmaps) - 1, min(7, len(pointmaps))).astype(int):
        points = pointmaps[frame_index].reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0)]
        if len(points):
            samples.append(points[:: max(1, len(points) // 6000)])
    merged = np.concatenate(samples, axis=0)
    center = np.median(merged, axis=0)
    centered = merged - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    low = np.percentile(projected[:, :2], 1, axis=0)
    high = np.percentile(projected[:, :2], 99, axis=0)
    margin = 0.08 * np.maximum(high - low, 1e-6)
    return center, axes, np.stack((low - margin, high + margin))


def _scene_panel(
    pointmap: np.ndarray,
    rgb: np.ndarray,
    *,
    center: np.ndarray,
    axes: np.ndarray,
    bounds_2d: np.ndarray,
    panel_hw: tuple[int, int],
) -> np.ndarray:
    points = pointmap.reshape(-1, 3)
    colors = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0)
    points = points[valid]
    colors = colors[valid]
    step = max(1, len(points) // 35000)
    projected = (points[::step] - center) @ axes.T
    low = np.asarray([bounds_2d[0, 0], bounds_2d[0, 1], np.nanmin(projected[:, 2])])
    high = np.asarray([bounds_2d[1, 0], bounds_2d[1, 1], np.nanmax(projected[:, 2])])
    return _project_points(
        projected,
        colors[::step],
        panel_hw=panel_hw,
        bounds=(low, high),
    )


def _pose_overlay(
    rgb: np.ndarray,
    K: np.ndarray,
    poses: np.ndarray,
    valid: np.ndarray,
    meshes,
) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    render_items = []
    for link_index, name in enumerate(CAD_LINK_NAMES):
        if not valid[link_index]:
            continue
        mesh = meshes[name]
        camera = mesh.vertices @ poses[link_index, :3, :3].T + poses[link_index, :3, 3]
        vertex_valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.001)
        if np.count_nonzero(vertex_valid) < 3:
            continue
        uv = np.full((len(camera), 2), np.nan, dtype=np.float64)
        projected = camera[vertex_valid] @ K.T
        uv[vertex_valid] = projected[:, :2] / projected[:, 2:3]
        mask, polygons = _rasterize_silhouette(
            uv[vertex_valid],
            panel_hw=image.shape[:2],
        )
        render_items.append(
            (
                float(np.median(camera[vertex_valid, 2])),
                link_index,
                name,
                mask,
                polygons,
                poses[link_index, :3, 3],
            )
        )

    for _, link_index, name, mask, polygons, origin in sorted(
        render_items, reverse=True, key=lambda item: item[0]
    ):
        color = COLORS_BGR[link_index]
        _composite_mesh(
            image,
            mask,
            polygons,
            color=color,
            alpha=0.48,
            outline_color=tuple(min(255, int(value) + 35) for value in color),
        )
        p = K @ origin
        if p[2] > 0:
            uv_origin = tuple(np.round(p[:2] / p[2]).astype(int))
            cv2.putText(
                image,
                name,
                uv_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (8, 8, 8),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                name,
                uv_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )
    return image


def _cad_views(meshes):
    result = {}
    rng = np.random.default_rng(0)
    for name in CAD_LINK_NAMES:
        mesh = meshes[name].mesh
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        try:
            points, _ = __import__("trimesh").sample.sample_surface(mesh, 8000)
        finally:
            np.random.set_state(state)
        center = np.mean(points, axis=0)
        _, _, axes = np.linalg.svd(points - center, full_matrices=False)
        projected = (points - center) @ axes.T
        low = np.percentile(projected[:, :2], 0.5, axis=0)
        high = np.percentile(projected[:, :2], 99.5, axis=0)
        margin = 0.08 * np.maximum(high - low, 1e-6)
        result[name] = {
            "points": points,
            "vertices": meshes[name].vertices,
            "faces": meshes[name].faces,
            "center": center,
            "axes": axes,
            "bounds": np.stack((low - margin, high + margin)),
        }
    return result


def _cotracker_cad_panel(
    observed_points: list[np.ndarray],
    observed_valid: list[np.ndarray],
    cad_anchors: list[CadAnchorSet | None],
    cad_views,
    panel_hw: tuple[int, int],
) -> np.ndarray:
    panel_h, panel_w = panel_hw
    tile_h, tile_w = panel_h // 2, panel_w // 3
    panel = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)
    for index, name in enumerate(CAD_LINK_NAMES):
        view = cad_views[name]
        row, column = divmod(index, 3)
        observed = observed_points[index]
        anchor_set = cad_anchors[index]
        low2d, high2d = view["bounds"]
        span = np.maximum(high2d - low2d, 1e-9)
        cache_key = (tile_h, tile_w, row)
        if view.get("tile_cache_key") != cache_key:
            cad_projected = (
                (view["vertices"] - view["center"]) @ view["axes"].T
            )
            cad_uv = np.stack(
                (
                    (cad_projected[:, 0] - low2d[0]) / span[0] * (tile_w - 1),
                    (1.0 - (cad_projected[:, 1] - low2d[1]) / span[1])
                    * (tile_h - 1),
                ),
                axis=1,
            )
            mesh_mask, mesh_polygons = _rasterize_mesh(
                cad_uv,
                view["faces"],
                panel_hw=(tile_h, tile_w),
            )
            tile = np.full((tile_h, tile_w, 3), 18, dtype=np.uint8)
            _composite_mesh(
                tile,
                mesh_mask,
                mesh_polygons,
                color=(82, 82, 82),
                alpha=1.0,
                outline_color=(190, 190, 190),
            )
            label_top = 38 if row == 0 else 3
            cv2.rectangle(
                tile, (3, label_top), (54, label_top + 20), (12, 14, 18), -1
            )
            cv2.putText(
                tile,
                name,
                (7, label_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            view["tile_cache_key"] = cache_key
            view["tile_cache"] = tile
        tile = view["tile_cache"].copy()
        correspondence_valid = np.zeros(len(observed), dtype=bool)
        if anchor_set is not None:
            correspondence_valid = (
                observed_valid[index]
                & anchor_set.valid
                & np.isfinite(observed).all(axis=1)
                & np.isfinite(anchor_set.points_cad).all(axis=1)
            )
        if np.any(correspondence_valid):
            observed_projected = (
                observed[correspondence_valid] - view["center"]
            ) @ view["axes"].T
            anchor_projected = (
                anchor_set.points_cad[correspondence_valid] - view["center"]
            ) @ view["axes"].T
            observed_uv = np.stack(
                (
                    (observed_projected[:, 0] - low2d[0]) / span[0] * (tile_w - 1),
                    (1.0 - (observed_projected[:, 1] - low2d[1]) / span[1]) * (tile_h - 1),
                ),
                axis=1,
            )
            anchor_uv = np.stack(
                (
                    (anchor_projected[:, 0] - low2d[0]) / span[0] * (tile_w - 1),
                    (1.0 - (anchor_projected[:, 1] - low2d[1]) / span[1]) * (tile_h - 1),
                ),
                axis=1,
            )
            finite = np.isfinite(observed_uv).all(axis=1) & np.isfinite(anchor_uv).all(axis=1)
            for observed_point, anchor_point in zip(observed_uv[finite], anchor_uv[finite]):
                observed_xy = tuple(np.rint(observed_point).astype(int))
                anchor_xy = tuple(np.rint(anchor_point).astype(int))
                if not (
                    0 <= observed_xy[0] < tile_w
                    and 0 <= observed_xy[1] < tile_h
                    and 0 <= anchor_xy[0] < tile_w
                    and 0 <= anchor_xy[1] < tile_h
                ):
                    continue
                cv2.line(tile, anchor_xy, observed_xy, (170, 170, 170), 1, cv2.LINE_AA)
                cv2.circle(tile, anchor_xy, 4, (248, 248, 248), 1, cv2.LINE_AA)
                cv2.drawMarker(
                    tile,
                    anchor_xy,
                    (248, 248, 248),
                    cv2.MARKER_CROSS,
                    7,
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(tile, observed_xy, 4, COLORS_BGR[index], -1, cv2.LINE_AA)
                cv2.circle(tile, observed_xy, 4, (20, 20, 20), 1, cv2.LINE_AA)
        count_text = f"pairs {int(np.count_nonzero(correspondence_valid))}"
        cv2.putText(
            tile,
            count_text,
            (tile_w - 76, 17 if row else 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        panel[row * tile_h : (row + 1) * tile_h, column * tile_w : (column + 1) * tile_w] = tile
    return panel


def _load_cotracker_groups(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(str(value) for value in archive["object_names"].tolist())
        offsets = np.asarray(archive["object_offsets"], dtype=np.int64)
        tracks = np.asarray(archive["tracks"], dtype=np.float64)
        visibility = np.asarray(archive["visibility"], dtype=np.float64)
        queries = np.asarray(archive["queries"], dtype=np.float64)
        query_ids = np.asarray(archive["query_ids"], dtype=np.int32)
    if offsets.shape != (len(names) + 1,) or offsets[0] != 0:
        raise ValueError("CoTracker object offsets do not match object names")
    if offsets[-1] != tracks.shape[1] or visibility.shape != tracks.shape[:2]:
        raise ValueError("CoTracker tracks/visibility do not match object offsets")
    if queries.shape != (tracks.shape[1], 3) or query_ids.shape != (tracks.shape[1],):
        raise ValueError("CoTracker queries/query IDs do not align with tracks")
    groups = {}
    for index, name in enumerate(names):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        groups[name] = {
            "tracks": tracks[:, start:stop],
            "visibility": visibility[:, start:stop],
            "queries": queries[start:stop],
            "query_ids": query_ids[start:stop],
        }
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segmentation-npz", type=Path, required=True)
    parser.add_argument("--geometry-npz", type=Path, required=True)
    parser.add_argument("--cotracker-npz", type=Path, required=True)
    parser.add_argument("--foundation-pose-npz", type=Path, required=True)
    parser.add_argument("--cad-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=16.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    segmentation = load_multi_object_segmentation(args.segmentation_npz, args.video)
    poses = load_foundation_pose_archive(
        args.foundation_pose_npz,
        expected_link_names=CAD_LINK_NAMES,
    )
    meshes = load_cad_manifest(args.cad_manifest, link_names=CAD_LINK_NAMES)
    cotracker = _load_cotracker_groups(args.cotracker_npz)
    missing_track_groups = sorted(set(CAD_LINK_NAMES).difference(cotracker))
    if missing_track_groups:
        raise ValueError(f"CoTracker archive is missing CAD links: {missing_track_groups}")
    with np.load(args.geometry_npz, allow_pickle=False) as archive:
        pointmaps = np.asarray(archive["pointmaps"], dtype=np.float32)
        rgb = np.asarray(archive["rgb_camera"], dtype=np.uint8)
        depth = np.asarray(archive["depth_camera"], dtype=np.float32)
        intrinsics = np.asarray(archive["intrinsics_camera"], dtype=np.float64)
        grid = ImageGridTransform(
            source_hw=tuple(int(v) for v in archive["source_hw"]),
            resized_hw=tuple(int(v) for v in archive["resized_hw_before_crop"]),
            crop_xywh=tuple(int(v) for v in archive["crop_xywh"]),
        )
    indices = [segmentation.object_names.index(name) for name in CAD_LINK_NAMES]
    masks_geometry = np.stack(
        [grid.transform_masks(segmentation.object_masks[:, index]) for index in indices],
        axis=1,
    )
    masks_geometry = erode_masks(
        masks_geometry.reshape(-1, *masks_geometry.shape[2:]), 2
    ).reshape(masks_geometry.shape)
    frame_count = min(
        len(pointmaps),
        len(rgb),
        len(depth),
        len(poses.frame_indices),
        segmentation.frames_count,
        *(len(cotracker[name]["tracks"]) for name in CAD_LINK_NAMES),
    )
    cad_anchors: list[CadAnchorSet | None] = []
    canonical_tracks: list[np.ndarray] = []
    canonical_valid: list[np.ndarray] = []
    for link_index, name in enumerate(CAD_LINK_NAMES):
        group = cotracker[name]
        frame_intrinsics = intrinsics[0] if intrinsics.ndim == 3 else intrinsics
        anchor_set = None
        if poses.pose_valid[0, link_index]:
            anchor_set = bind_cad_anchors(
                query_ids=group["query_ids"],
                query_pixels_source=group["queries"][:, 1:3],
                grid_transform=grid,
                frame_mask_geometry=masks_geometry[0, link_index],
                frame_depth_geometry=depth[0] * poses.video_depth_scale,
                intrinsics=frame_intrinsics,
                T_C_from_L=poses.T_C_from_L[0, link_index],
                vertices=meshes[name].vertices,
                faces=meshes[name].faces,
            )
        observed_camera, observed_is_valid = sample_observed_track_points(
            tracks_source=group["tracks"][:frame_count],
            visibility=group["visibility"][:frame_count],
            grid_transform=grid,
            masks_geometry=masks_geometry[:frame_count, link_index],
            depths_geometry=depth[:frame_count],
            intrinsics=intrinsics[:frame_count] if intrinsics.ndim == 3 else intrinsics,
            depth_scale=poses.video_depth_scale,
        )
        observed_cad, observed_is_valid = canonicalize_observed_sequence(
            observed_camera,
            observed_is_valid,
            poses.T_C_from_L[:frame_count, link_index],
            poses.pose_valid[:frame_count, link_index],
        )
        cad_anchors.append(anchor_set)
        canonical_tracks.append(observed_cad)
        canonical_valid.append(observed_is_valid)
    capture = cv2.VideoCapture(str(args.video))
    source_frames = []
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        source_frames.append(frame)
    capture.release()
    frame_count = min(frame_count, len(source_frames))
    panel_size = (640, 360)
    video_path = output / "cotracker_cad_replay.mp4"
    intermediate_video_path = output / ".cotracker_cad_replay.mp4v.mp4"
    writer = cv2.VideoWriter(
        str(intermediate_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_size[0] * 2, panel_size[1] * 2),
    )
    scene_center, scene_axes, scene_bounds = _scene_view_basis(pointmaps[:frame_count])
    cad_views = _cad_views(meshes)
    selected_frames = sorted(
        {
            int(value)
            for value in np.linspace(
                0, frame_count - 1, min(4, frame_count)
            ).astype(int)
        }
    )
    first_mask = None
    first_point_cloud = None
    for frame_index in range(frame_count):
        sam = _mask_overlay(
            source_frames[frame_index],
            segmentation.object_masks[frame_index, indices],
        )
        sam = _fit_panel(sam, panel_size)
        _label(sam, "SAM3 masks")
        scene = _scene_panel(
            pointmaps[frame_index],
            rgb[frame_index],
            center=scene_center,
            axes=scene_axes,
            bounds_2d=scene_bounds,
            panel_hw=(panel_size[1], panel_size[0]),
        )
        _label(scene, "MegaSAM point cloud")
        K = intrinsics[frame_index] if intrinsics.ndim == 3 else intrinsics
        pose_image = _pose_overlay(
            rgb[frame_index],
            K,
            poses.T_C_from_L[frame_index],
            poses.pose_valid[frame_index],
            meshes,
        )
        pose_image = _fit_panel(pose_image, panel_size)
        _label(pose_image, "FoundationPose: filled CAD silhouette + outline")
        canonical_image = _cotracker_cad_panel(
            [values[frame_index] for values in canonical_tracks],
            [values[frame_index] for values in canonical_valid],
            cad_anchors,
            cad_views,
            (panel_size[1] * 2, panel_size[0] * 2),
        )
        _label(
            canonical_image,
            "CoTracker -> RGB-D -> FoundationPose CAD frame  |  color: observed  white +: CAD anchor",
        )
        writer.write(canonical_image)
        if frame_index == 0:
            first_mask = sam.copy()
            first_point_cloud = scene.copy()
        if frame_index in selected_frames:
            cv2.imwrite(str(output / f"foundationpose_frame_{frame_index:04d}.png"), pose_image)
            cv2.imwrite(
                str(output / f"cotracker_cad_frame_{frame_index:04d}.png"),
                canonical_image,
            )
    writer.release()
    _encode_h264(intermediate_video_path, video_path)
    intermediate_video_path.unlink()
    if first_mask is not None:
        cv2.imwrite(str(output / "initial_sam_masks.png"), first_mask)
    if first_point_cloud is not None:
        cv2.imwrite(str(output / "point_cloud_frame_0000.png"), first_point_cloud)
    summary = {
        "schema_version": 1,
        "video": str(args.video.resolve()),
        "frame_count": frame_count,
        "selected_foundationpose_frames": selected_frames,
        "video_depth_scale": poses.video_depth_scale,
        "pose_valid_fraction": {
            name: float(np.mean(poses.pose_valid[:frame_count, index]))
            for index, name in enumerate(CAD_LINK_NAMES)
        },
        "cotracker_cad_correspondence": {
            "tracking_mode": "exact-group",
            "observed_points": "visible CoTracker coordinates lifted with MegaSAM RGB-D and remapped by FoundationPose",
            "cad_anchor_points": "the corresponding frame-0 query rays intersected with each CAD mesh",
            "valid_anchor_count": {
                name: (cad_anchors[index].valid_count if cad_anchors[index] is not None else 0)
                for index, name in enumerate(CAD_LINK_NAMES)
            },
        },
        "artifacts": {
            "cotracker_cad_replay": str(video_path),
            "initial_sam_masks": str(output / "initial_sam_masks.png"),
            "point_cloud": str(output / "point_cloud_frame_0000.png"),
        },
    }
    (output / "cad_replay.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "replay": str(video_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
