#!/usr/bin/env python3
"""Run pristine PDI-Bench using frozen SAM3 and shared MegaSAM cache inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def keep_mask(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    visible = visibility.mean(axis=0) >= 0.3
    jumps = (
        np.linalg.norm(np.diff(tracks, axis=0), axis=2).max(axis=0) < 120.0
        if len(tracks) > 1
        else np.ones(tracks.shape[1], dtype=bool)
    )
    keep = visible & jumps
    return keep if keep.sum() >= 2 else np.ones(tracks.shape[1], dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-query", default="gripper")
    args = parser.parse_args()
    original_root = args.original_root.resolve()
    video = args.video.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sam_cache = cache_dir / f"{video.stem}_sam2.npz"
    geometry_cache = cache_dir / f"{video.stem}_mega_sam.npz"
    if not sam_cache.is_file() or not geometry_cache.is_file():
        raise FileNotFoundError("SAM3 adapter and shared geometry caches are required")
    (cache_dir / f"{video.stem}_cotracker.npz").unlink(missing_ok=True)
    sys.path.insert(0, str(original_root / "src"))

    from pdi_eval.perception.track_wrapper import TrackWrapper  # noqa: E402
    from pdi_eval.pipeline import PDIEvaluationPipeline  # noqa: E402

    config_path = original_root / "configs/default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["cache_dir"] = str(cache_dir)
    config["tracker_ckpt"] = str(args.tracker_checkpoint.resolve())
    config["reconstruction_audit"]["enabled"] = False
    captured: dict[str, Any] = {}
    original_loader = TrackWrapper._load_model

    class CaptureModel:
        def __init__(self, inner: Any):
            self.inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def __call__(self, video_tensor: torch.Tensor, *model_args: Any, **kwargs: Any):
            queries = kwargs.get("queries")
            if queries is not None:
                captured["queries_tracker"] = queries.detach().cpu().numpy()[0]
                captured["tracker_hw"] = list(video_tensor.shape[-2:])
            if video_tensor.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(video_tensor.device)
                torch.cuda.synchronize(video_tensor.device)
            started = time.perf_counter()
            result = self.inner(video_tensor, *model_args, **kwargs)
            if video_tensor.device.type == "cuda":
                torch.cuda.synchronize(video_tensor.device)
                captured["peak_gpu_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated(video_tensor.device)
                )
            captured["model_seconds"] = time.perf_counter() - started
            captured["raw_tracks"] = result[0][0].detach().cpu().numpy()
            captured["raw_visibility"] = result[1][0].detach().cpu().numpy()
            return result

    def capture_loader(wrapper: TrackWrapper, checkpoint: str) -> CaptureModel:
        started = time.perf_counter()
        model = original_loader(wrapper, checkpoint)
        captured["tracker_load_seconds"] = time.perf_counter() - started
        return CaptureModel(model)

    TrackWrapper._load_model = capture_loader
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        report = PDIEvaluationPipeline(config).run(
            video_path=str(video),
            text_query=args.text_query,
        )
    finally:
        TrackWrapper._load_model = original_loader
    wall_seconds = time.perf_counter() - started

    with np.load(sam_cache, allow_pickle=False) as archive:
        first_mask = np.asarray(archive["masks"])[0].astype(np.uint8)
    queries = np.asarray(captured["queries_tracker"])
    tracker_height, tracker_width = captured["tracker_hw"]
    small_mask = cv2.resize(
        first_mask,
        (tracker_width, tracker_height),
        interpolation=cv2.INTER_NEAREST,
    )
    membership = []
    for query in queries:
        x = int(round(float(query[1])))
        y = int(round(float(query[2])))
        membership.append(
            0 <= x < tracker_width and 0 <= y < tracker_height and small_mask[y, x] > 0
        )
    foreground_count = next(
        (index for index, inside in enumerate(membership) if not inside),
        len(queries),
    )
    source_height, source_width = first_mask.shape
    source_queries = queries.copy()
    source_queries[:, 1] *= source_width / tracker_width
    source_queries[:, 2] *= source_height / tracker_height
    raw_tracks = np.asarray(captured["raw_tracks"]).copy()
    raw_tracks[..., 0] *= source_width / tracker_width
    raw_tracks[..., 1] *= source_height / tracker_height
    raw_visibility = np.asarray(captured["raw_visibility"])
    foreground_keep = keep_mask(
        raw_tracks[:, :foreground_count], raw_visibility[:, :foreground_count]
    )
    background_keep = keep_mask(
        raw_tracks[:, foreground_count:], raw_visibility[:, foreground_count:]
    )
    track_path = output_dir / "cotracker_original.npz"
    temporary_track = track_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_track,
        foreground_queries=source_queries[:foreground_count][foreground_keep],
        background_queries=source_queries[foreground_count:][background_keep],
        foreground_tracks=raw_tracks[:, :foreground_count][:, foreground_keep],
        foreground_visibility=raw_visibility[:, :foreground_count][:, foreground_keep],
        background_tracks=raw_tracks[:, foreground_count:][:, background_keep],
        background_visibility=raw_visibility[:, foreground_count:][:, background_keep],
        tracker_hw=np.asarray(captured["tracker_hw"], dtype=np.int64),
        source_hw=np.asarray(first_mask.shape, dtype=np.int64),
    )
    temporary_track.replace(track_path)
    write_json(output_dir / "metrics.json", report)
    write_json(
        output_dir / "timing.json",
        {
            "wall_seconds": wall_seconds,
            "tracker_load_seconds": captured.get("tracker_load_seconds"),
            "model_seconds": captured.get("model_seconds"),
            "peak_gpu_memory_bytes": captured.get("peak_gpu_memory_bytes"),
        },
    )
    shutil.copy2(config_path, output_dir / "original_config.yaml")
    shutil.copy2(sam_cache, output_dir / "frozen_sam3_mask_cache.npz")
    shutil.copy2(cache_dir / f"{video.stem}_mega_sam.npz", output_dir / "shared_geometry_cache.npz")
    write_json(
        output_dir / "manifest.json",
        {
            "original_root": str(original_root),
            "video": str(video),
            "cache_dir": str(cache_dir),
            "tracker_checkpoint": str(args.tracker_checkpoint.resolve()),
            "segmentation_source": "frozen SAM3 cache",
            "sam3_text_prompt": args.text_query,
            "geometry_source": "shared MegaSAM cache",
            "foreground_query_count": int(foreground_keep.sum()),
            "background_query_count": int(background_keep.sum()),
        },
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
