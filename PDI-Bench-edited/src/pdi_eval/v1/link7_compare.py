"""Run and visualize a matched Link 7 CoTracker3 versus TAPIP3D comparison."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _tracks(path: Path, backend: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if backend == "tapip3d":
            return (np.asarray(data["tracks_uv"]),
                    np.asarray(data["visibility"]),
                    np.asarray(data["point_ids"]))
        return (np.asarray(data["link7_raw_tracks"]),
                np.asarray(data["link7_raw_visibility"]),
                np.arange(data["link7_raw_tracks"].shape[1], dtype=np.int64))


def _retained_ids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        names = data["object_names"].tolist()
        index = names.index("link7")
        start, end = data["object_offsets"][index:index + 2]
        return np.asarray(data["point_ids"][start:end], dtype=np.int64)


def _draw_video(
    source: Path,
    output: Path,
    queries: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    name: str,
    color: tuple[int, int, int],
    display_ids: np.ndarray,
) -> None:
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 16.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {output}")
    try:
        for frame in range(len(tracks)):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"video ended before track frame {frame}")
            if frame == 0:
                for query in queries:
                    cv2.circle(image, tuple(np.rint(query[1:]).astype(int)),
                               2, (170, 170, 170), -1)
            for point_id in display_ids:
                query = queries[point_id]
                x0, y0 = (int(round(value)) for value in query[1:])
                if frame == int(query[0]):
                    cv2.circle(image, (x0, y0), 4, color, -1)
                    cv2.putText(image, str(point_id), (x0 + 5, y0 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
                    continue
                history = np.flatnonzero(visibility[:frame + 1, point_id] > 0.5)
                if len(history):
                    recent = history[-24:]
                    for left, right in zip(recent[:-1], recent[1:]):
                        if right != left + 1:
                            continue
                        a = tuple(np.rint(tracks[left, point_id]).astype(int))
                        b = tuple(np.rint(tracks[right, point_id]).astype(int))
                        cv2.line(image, a, b, color, 2, cv2.LINE_AA)
                    last = tuple(np.rint(tracks[history[-1], point_id]).astype(int))
                else:
                    last = (x0, y0)
                if visibility[frame, point_id] > 0.5:
                    cv2.circle(image, last, 4, color, -1)
                else:
                    cv2.drawMarker(image, last, (0, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 9, 2)
                cv2.putText(image, str(point_id), (last[0] + 5, last[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (460, 30), (0, 0, 0), -1)
            cv2.putText(image, f"Link 7 | {name} | {len(display_ids)}/{len(queries)} IDs | frame {frame}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            writer.write(image)
    finally:
        capture.release()
        writer.release()


def _display_ids(queries: np.ndarray, count: int = 16) -> np.ndarray:
    """Choose a fixed spatially spread set from the shared initial queries."""
    xy = queries[:, 1:3]
    selected = [0]
    nearest = np.sum((xy - xy[0]) ** 2, axis=1)
    while len(selected) < min(count, len(xy)):
        nearest[selected] = -1
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((xy - xy[index]) ** 2, axis=1))
    return np.asarray(sorted(selected), dtype=np.int64)


def _write_interactive_initial(path: Path, image: np.ndarray, queries: np.ndarray) -> None:
    height, width = image.shape[:2]
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cannot encode initial RGB frame")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    circles = "\n".join(
        f'<circle cx="{float(q[1]):.2f}" cy="{float(q[2]):.2f}" r="4" '
        f'data-id="{i}" tabindex="0"><title>Point {i}: '
        f'({float(q[1]):.1f}, {float(q[2]):.1f})</title></circle>'
        for i, q in enumerate(queries)
    )
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Link 7 initial query IDs</title>
<style>body{{margin:24px;background:#111827;color:#f9fafb;font:16px system-ui}}
svg{{max-width:100%;height:auto;border:1px solid #64748b}}
circle{{fill:#facc15;stroke:#111827;stroke-width:1;cursor:pointer}}
circle:hover,circle:focus{{fill:#fb7185;r:7}}
p{{max-width:60ch}}</style>
<h1>Link 7 initial query IDs</h1><p>All {len(queries)} selected points. Hover or click a point to inspect its exact ID. Both trackers use this same query set.</p>
<p id="chosen">Select a point.</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Initial video frame and Link 7 query points">
<image width="{width}" height="{height}" href="data:image/png;base64,{data}"/>{circles}</svg>
<script>document.querySelectorAll('circle').forEach(c=>c.addEventListener('click',()=>{{
document.getElementById('chosen').textContent=c.querySelector('title').textContent;
}}));</script></html>""", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("segmentation-npz", "geometry-cache-dir", "tracker-checkpoint",
                 "tapip3d-python", "tapip3d-repository", "tapip3d-checkpoint"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--tracking-mode", choices=("joint-query", "exact-group"),
                        default="exact-group")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--render-only", action="store_true",
                        help="Refresh visualizations from existing score archives")
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    provenance = root / "provenance"
    root.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        required = ("segmentation_npz", "geometry_cache_dir", "tracker_checkpoint",
                    "tapip3d_python", "tapip3d_repository", "tapip3d_checkpoint")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"missing model inputs: {missing}")
        common = ["--input", str(args.video.resolve()),
              "--segmentation-npz", str(args.segmentation_npz.resolve()),
              "--geometry-cache-dir", str(args.geometry_cache_dir.resolve()),
              "--tracker-checkpoint", str(args.tracker_checkpoint.resolve()),
              "--tracking-mode", args.tracking_mode, "--disable-replay"]
        if args.config is not None:
            common += ["--config", str(args.config.resolve())]
        for backend in ("cotracker3", "tapip3d"):
            command = [sys.executable, "-m", "pdi_eval.experiment", "score", *common,
                   "--output-dir", str(provenance / backend),
                   "--link7-tracker", backend]
            if backend == "tapip3d":
                command += ["--tapip3d-python", str(args.tapip3d_python.absolute()),
                        "--tapip3d-repository", str(args.tapip3d_repository.resolve()),
                        "--tapip3d-checkpoint", str(args.tapip3d_checkpoint.resolve())]
            subprocess.run(command, check=True)
    query_files = [provenance / backend / "link7_initial_queries.npz"
                   for backend in ("cotracker3", "tapip3d")]
    with np.load(query_files[0], allow_pickle=False) as a, np.load(query_files[1], allow_pickle=False) as b:
        if not np.array_equal(a["queries"], b["queries"]) or not np.array_equal(
            a["point_ids"], b["point_ids"]
        ):
            raise ValueError("CoTracker3 and TAPIP3D received different Link 7 points")
        queries = np.asarray(a["queries"], dtype=np.float32)
        ids = np.asarray(a["point_ids"], dtype=np.int64)
    initial = cv2.VideoCapture(str(args.video))
    ok, first_frame = initial.read()
    initial.release()
    if not ok:
        raise ValueError("cannot read initial video frame")
    _write_interactive_initial(root / "initial_queries_interactive.html", first_frame, queries)
    display_ids = _display_ids(queries)
    for query in queries:
        x, y = (int(round(value)) for value in query[1:])
        cv2.circle(first_frame, (x, y), 2, (170, 170, 170), -1)
    for point_id in display_ids:
        query = queries[point_id]
        x, y = (int(round(value)) for value in query[1:])
        cv2.circle(first_frame, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(first_frame, str(point_id), (x + 5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(root / "initial_queries.png"), first_frame)
    summary = {"initial_queries_identical": True, "initialized_points": len(queries),
               "displayed_point_ids": display_ids.tolist(),
               "tracking_mode": args.tracking_mode, "methods": {}}
    for backend, color in (("cotracker3", (0, 180, 255)),
                           ("tapip3d", (255, 180, 0))):
        folder = provenance / backend
        prefix = "cotracker" if backend == "cotracker3" else "tapip3d_link7"
        archive = folder / f"{prefix}_{args.tracking_mode}.npz"
        raw = folder / "link7_tapip3d_raw.npz" if backend == "tapip3d" else archive
        tracks, visible, point_ids = _tracks(raw, backend)
        if not np.array_equal(point_ids, ids) or tracks.shape[1] != len(queries):
            raise ValueError(f"{backend} changed Link 7 point ordering")
        retained = _retained_ids(archive)
        retained_visible_final = int(np.sum(visible[-1, retained] > 0.5))
        summary["methods"][backend] = {
            "initialized_points": len(ids),
            "surviving_final_frame": retained_visible_final,
            "surviving_final_frame_percent": 100.0 * retained_visible_final / len(ids),
            "average_visible_frame_ratio": float(np.mean(visible > 0.5)),
            "downstream_rejected_tracks": len(ids) - len(retained),
            "retained_track_count": len(retained),
            "link7_scoring": json.loads((folder / "metrics.json").read_text())["modes"][args.tracking_mode]["objects"]["link7"],
        }
        _draw_video(args.video, root / f"{backend}_trajectories.mp4", queries,
                    tracks, visible, backend, color, display_ids)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"comparison": str(root / "comparison.json"),
                      "initialized_points": len(ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
