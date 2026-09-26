"""Run official TAPIP3D inference in its isolated Python environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import models

from datasets.data_ops import _filter_one_depth
from utils.inference_utils import (
    _inference_with_grid, read_video, resize_depth_bilinear,
)


def _load_official_checkpoint(path: str) -> torch.nn.Module:
    """Load all official weights without an unnecessary torch.hub backbone fetch."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["cfg"]
    # The official constructor's pretrained=True downloads CoTracker before
    # immediately replacing every weight from this complete TAPIP3D checkpoint.
    config["model"]["encoder"]["pretrained"] = False
    model = models.from_config(
        config["model"], image_size=config["train_dataset"]["resolution"]
    )
    model.load_state_dict(checkpoint["weight"], strict=True)
    model.set_eval_mode("raw")
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("video", "input", "output", "checkpoint"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--support-grid-size", type=int, default=16)
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--resolution-factor", type=int, default=2)
    parser.add_argument("--visibility-threshold", type=float, default=0.9)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        depths = np.asarray(data["depths"], dtype=np.float32)
        k = np.asarray(data["intrinsics"], dtype=np.float32)
        poses = np.asarray(data["camera_poses"], dtype=np.float32)
        queries = np.asarray(data["query_world"], dtype=np.float32)
        selected = np.asarray(data["selected_queries"], dtype=np.float32)
        point_ids = np.asarray(data["point_ids"], dtype=np.int64)
        source_hw = tuple(int(value) for value in data["source_hw"])
    video = read_video(args.video)[:len(depths)]
    if len(video) != len(depths) or tuple(video.shape[1:3]) != source_hw:
        raise ValueError("RGB frames do not align with cached MegaSAM geometry")
    model = _load_official_checkpoint(args.checkpoint).to("cuda")
    model_h, model_w = model.image_size
    scale = np.sqrt(args.resolution_factor)
    inference_h, inference_w = int(model_h * scale), int(model_w * scale)
    depth_h, depth_w = depths.shape[1:3]
    k_inference = np.repeat(k[None], len(depths), axis=0)
    k_inference[:, 0, :] *= (inference_w - 1) / max(depth_w - 1, 1)
    k_inference[:, 1, :] *= (inference_h - 1) / max(depth_h - 1, 1)
    video = np.stack([cv2.resize(frame, (inference_w, inference_h)) for frame in video])
    depths = np.stack([
        resize_depth_bilinear(depth, (inference_w, inference_h)) for depth in depths
    ])
    depths = np.stack([
        _filter_one_depth(depth, 0.08, 15, intrinsic)
        for depth, intrinsic in zip(depths, k_inference)
    ])
    if not np.any(depths > 0):
        raise ValueError("TAPIP3D input has no positive cached depth")
    extrinsics = np.linalg.inv(poses).astype(np.float32)
    rgb_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().cuda() / 255.0
    depth_tensor = torch.from_numpy(depths).float().cuda()
    k_tensor = torch.from_numpy(k_inference).float().cuda()
    pose_tensor = torch.from_numpy(extrinsics).float().cuda()
    query_tensor = torch.from_numpy(queries).float().cuda()
    valid_depths = depth_tensor[depth_tensor > 0]
    # Match the official inference helper's kthvalue depth ROI; quantile has
    # a tensor-size limit on these full-video depth inputs.
    q25 = torch.kthvalue(valid_depths, int(0.25 * len(valid_depths))).values
    q75 = torch.kthvalue(valid_depths, int(0.75 * len(valid_depths))).values
    depth_roi = torch.tensor([1e-7, (q75 + 1.5 * (q75 - q25)).item()],
                             dtype=torch.float32, device="cuda")
    model.set_image_size((inference_h, inference_w))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        preds, _ = _inference_with_grid(
            model=model, video=rgb_tensor[None], depths=depth_tensor[None],
            intrinsics=k_tensor[None], extrinsics=pose_tensor[None],
            query_point=query_tensor[None], num_iters=args.num_iters,
            depth_roi=depth_roi, grid_size=args.support_grid_size,
        )
    coords = preds.coords[0].float().cpu().numpy()
    confidence = torch.sigmoid(preds.visibs[0].float()).cpu().numpy()
    visibility = confidence >= args.visibility_threshold
    if coords.shape != (len(depths), len(queries), 3):
        raise ValueError(f"official TAPIP3D changed selected query count: {coords.shape}")
    local = (coords - poses[:, None, :3, 3]) @ poses[:, :3, :3]
    k_source = k.copy()
    k_source[0, :] *= (source_hw[1] - 1) / max(depth_w - 1, 1)
    k_source[1, :] *= (source_hw[0] - 1) / max(depth_h - 1, 1)
    z = local[..., 2]
    tracks = np.empty((*z.shape, 2), dtype=np.float32)
    tracks[..., 0] = k_source[0, 0] * local[..., 0] / np.where(z > 0, z, 1) + k_source[0, 2]
    tracks[..., 1] = k_source[1, 1] * local[..., 1] / np.where(z > 0, z, 1) + k_source[1, 2]
    visibility &= np.isfinite(tracks).all(axis=-1) & (z > 0)
    for i, query in enumerate(selected):
        frame = int(query[0])
        coords[frame, i] = queries[i, 1:]
        tracks[frame, i] = query[1:]
        confidence[frame, i] = 1.0
        visibility[frame, i] = True
    torch.cuda.synchronize()
    peak_gpu_memory_bytes = torch.cuda.max_memory_allocated()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, point_ids=point_ids, queries=selected, query_world=queries,
        frame=np.arange(len(depths), dtype=np.int64), tracks_uv=tracks,
        trajectories_xyz=coords, visibility=visibility, confidence=confidence,
        peak_gpu_memory_bytes=np.asarray(peak_gpu_memory_bytes, dtype=np.int64),
    )
    print(f"TAPIP3D_COMPLETE points={len(queries)} frames={len(depths)} output={output}")


if __name__ == "__main__":
    main()
