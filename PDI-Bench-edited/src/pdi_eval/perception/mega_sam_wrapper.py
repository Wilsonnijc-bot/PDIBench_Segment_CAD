import subprocess
import os
import sys
import glob
import hashlib
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from .base import BasePerceptor, PerceptionResult, SharedGeometryResult
from ..utils.logger import pdi_logger


GEOMETRY_CACHE_SCHEMA = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_depth_from_world_pointmaps(
    pointmaps: np.ndarray,
    camera_poses: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Derive normalized target camera-Z from shared world-coordinate geometry."""
    pointmaps = np.asarray(pointmaps)
    camera_poses = np.asarray(camera_poses)
    masks = np.asarray(masks)
    frame_count = min(len(pointmaps), len(camera_poses), len(masks))
    if frame_count < 1:
        raise ValueError("pointmaps, camera poses, and masks need a common frame")
    if pointmaps.ndim != 4 or pointmaps.shape[-1] != 3:
        raise ValueError(f"pointmaps must have shape (T,H,W,3), got {pointmaps.shape}")
    if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
        raise ValueError(
            f"camera_poses must have shape (T,4,4), got {camera_poses.shape}"
        )

    depth_z = np.empty(frame_count, dtype=np.float64)
    target_height, target_width = pointmaps.shape[1:3]
    for frame_index in range(frame_count):
        mask = masks[frame_index]
        if mask.shape != (target_height, target_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        pose = camera_poses[frame_index]
        camera_points = (pointmaps[frame_index] - pose[:3, 3]) @ pose[:3, :3]
        z_map = camera_points[..., 2]
        valid = (mask > 0) & np.isfinite(z_map) & (z_map > 0)
        if not np.any(valid):
            raise ValueError(f"frame {frame_index} has no valid target depths")
        depth_z[frame_index] = float(np.median(z_map[valid]))
    if not np.isfinite(depth_z[0]) or abs(depth_z[0]) <= 1e-8:
        raise ValueError("first target depth is invalid")
    return depth_z / depth_z[0]

def _masks_to_h_pixel_x_center(masks: np.ndarray):
    """Compute height and x-center time series from (T,H,W) masks."""
    T = masks.shape[0]
    h_list, x_list = [], []
    for t in range(T):
        m = masks[t]
        if np.any(m > 0):
            ys, xs = np.where(m > 0)
            h_list.append(float(np.ptp(ys) + 1))
            x_list.append(float(np.mean(xs)))
        else:
            h_list.append(1.0)
            x_list.append(m.shape[1] / 2.0)
    return np.array(h_list, dtype=np.float64), np.array(x_list, dtype=np.float64)

def _extract_frames(video_path: str, out_dir: str) -> int:
    """Extract frames with 6-digit names; clear directory first to drop stale frames."""
    if os.path.isdir(out_dir):
        for f in glob.glob(os.path.join(out_dir, "*.jpg")) + glob.glob(os.path.join(out_dir, "*.png")):
            os.remove(f)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        path = os.path.join(out_dir, f"{count:06d}.jpg")
        cv2.imwrite(path, frame)
        count += 1
    cap.release()
    return count

class MegaSamWrapper(BasePerceptor):
    def __init__(self, checkpoint=None, device="cuda"):
        super().__init__(device)
        self.mega_sam_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../third_party/mega_sam")
        )
        self.da_ckpt = os.path.join(
            self.mega_sam_root, "Depth-Anything", "checkpoints", "depth_anything_vitl14.pth"
        )
        self.megasam_weights = os.path.join(self.mega_sam_root, "checkpoints", "megasam_final.pth")
        self.raft_weights = os.path.join(self.mega_sam_root, "cvd_opt", "raft-things.pth")

    @staticmethod
    def _file_identity(path: str, *, hash_content: bool = False) -> dict:
        source = Path(path)
        if not source.is_file():
            return {"path": str(source.resolve()), "missing": True}
        stat = source.stat()
        identity = {
            "path": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if hash_content:
            identity["sha256"] = _sha256_file(source)
        return identity

    def _reconstruction_code_identity(self) -> dict[str, dict]:
        sources = {
            "wrapper": Path(__file__),
            "depth_anything": Path(self.mega_sam_root) / "Depth-Anything/run_videos.py",
            "unidepth": Path(self.mega_sam_root) / "UniDepth/scripts/demo_mega-sam.py",
            "droid": Path(self.mega_sam_root) / "camera_tracking_scripts/test_demo.py",
            "raft": Path(self.mega_sam_root) / "cvd_opt/preprocess_flow.py",
            "cvd": Path(self.mega_sam_root) / "cvd_opt/cvd_opt.py",
        }
        return {
            name: self._file_identity(str(path), hash_content=True)
            for name, path in sources.items()
        }

    def _geometry_cache_identity(self, video_path: str) -> tuple[str, dict]:
        video = Path(video_path).resolve()
        metadata = {
            "schema": GEOMETRY_CACHE_SCHEMA,
            "video_sha256": _sha256_file(video),
            "depth_anything": self._file_identity(self.da_ckpt),
            "megasam": self._file_identity(self.megasam_weights),
            "raft": self._file_identity(self.raft_weights),
            "code": self._reconstruction_code_identity(),
            "settings": {
                "depth_encoder": "vitl",
                "droid_disable_vis": True,
                "raft_mixed_precision": True,
                "cvd_w_grad": 2.0,
                "cvd_w_normal": 5.0,
            },
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest(), metadata

    def infer_shared(
        self,
        video_path: str,
        object_masks: np.ndarray,
        cache_dir: str | Path | None = None,
    ) -> SharedGeometryResult:
        """Build/load one scene reconstruction and derive depth for every object."""
        object_masks = np.asarray(object_masks, dtype=bool)
        if object_masks.ndim != 4:
            raise ValueError(
                f"object_masks must have shape (T,N,H,W), got {object_masks.shape}"
            )
        cache_path = None
        cache_metadata = None
        pointmaps = camera_poses = None
        focal_length = None
        if cache_dir is not None:
            cache_key, cache_metadata = self._geometry_cache_identity(video_path)
            cache_path = Path(cache_dir).resolve() / f"{cache_key}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as archive:
                    pointmaps = np.asarray(archive["pointmaps"])
                    camera_poses = np.asarray(archive["camera_poses"])
                    focal_length = float(np.asarray(archive["focal_length"]).reshape(-1)[0])
                    stored_metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
                if stored_metadata != cache_metadata:
                    raise ValueError(f"MegaSAM cache metadata mismatch: {cache_path}")
                if pointmaps.ndim != 4 or pointmaps.shape[-1] != 3:
                    raise ValueError(f"Invalid cached pointmaps shape: {pointmaps.shape}")
                if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
                    raise ValueError(
                        f"Invalid cached camera_poses shape: {camera_poses.shape}"
                    )
                if len(pointmaps) != len(camera_poses):
                    raise ValueError("Cached pointmaps and camera poses have different lengths")
                if not np.isfinite(focal_length) or focal_length <= 0:
                    raise ValueError(f"Invalid cached focal length: {focal_length}")
                if not np.any(np.isfinite(pointmaps) & (pointmaps != 0)):
                    raise ValueError(f"Cached pointmaps contain no valid geometry: {cache_path}")
                pdi_logger.info(f"Reusing shared MegaSAM geometry: {cache_path}")

        cache_hit = pointmaps is not None
        if not cache_hit:
            union_masks = np.any(object_masks, axis=1)
            geometry = self.infer(video_path, masks=union_masks)
            if geometry.pointmaps is None or geometry.camera_poses is None:
                raise RuntimeError("MegaSAM did not return pointmaps and camera poses")
            if not np.any(geometry.pointmaps != 0):
                raise RuntimeError("MegaSAM returned fallback all-zero pointmaps")
            pointmaps = np.asarray(geometry.pointmaps)
            camera_poses = np.asarray(geometry.camera_poses)
            focal_length = float(geometry.focal_length)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    pointmaps=pointmaps,
                    camera_poses=camera_poses,
                    focal_length=focal_length,
                    metadata_json=np.asarray(json.dumps(cache_metadata, sort_keys=True)),
                )
                temporary.replace(cache_path)
                pdi_logger.info(f"Saved shared MegaSAM geometry: {cache_path}")

        frame_count = min(len(pointmaps), len(camera_poses), len(object_masks))
        object_depth_z = np.stack(
            [
                target_depth_from_world_pointmaps(
                    pointmaps[:frame_count],
                    camera_poses[:frame_count],
                    object_masks[:frame_count, object_index],
                )
                for object_index in range(object_masks.shape[1])
            ],
            axis=1,
        )
        return SharedGeometryResult(
            video_id=Path(video_path).stem,
            frames_count=frame_count,
            pointmaps=pointmaps[:frame_count],
            camera_poses=camera_poses[:frame_count],
            focal_length=focal_length,
            object_depth_z=object_depth_z,
            cache_path=str(cache_path) if cache_path is not None else None,
            metadata={
                "engine": "Mega-SAM-Shared-Multi-Object",
                "cache_hit": cache_hit,
                "cache_schema": GEOMETRY_CACHE_SCHEMA,
                "cache_identity": cache_metadata,
            },
        )

    @staticmethod
    def _parse_intrinsic(K) -> tuple:
        """Robustly parse intrinsics from various K layouts."""
        K = np.asarray(K)
        try:
            if K.ndim == 1 and K.size >= 4:
                return float(K[0]), float(K[1]), float(K[2]), float(K[3])
            if K.ndim == 2 and K.shape[0] >= 3:
                return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        except Exception:
            pass
        pdi_logger.warning("Intrinsics parse failed; using defaults")
        return 1000.0, 1000.0, 320.0, 240.0

    @staticmethod
    def _depth_to_pointmaps(depths, cam_c2w, fx, fy, cx, cy):
        """Build world-coordinate point maps for volume_audit."""
        T, h, w = depths.shape
        # row_idx increases along y; col_idx along x
        row_idx, col_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pointmaps = np.zeros((T, h, w, 3), dtype=np.float32)
        
        for t in range(T):
            d = depths[t].astype(np.float32)
            # 1. Back-project to camera coordinates
            x_cam = (col_idx - cx) * d / fx
            y_cam = (row_idx - cy) * d / fy
            z_cam = d
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
            
            # 2. Transform to world: P_w = R * P_c + t
            R = cam_c2w[t, :3, :3]
            t_vec = cam_c2w[t, :3, 3]
            pts_world = (pts_cam @ R.T) + t_vec
            
            pointmaps[t] = pts_world.reshape(h, w, 3)
        return pointmaps

    def _mega_sam_env(self):
        parts = [
            self.mega_sam_root,
            os.path.join(self.mega_sam_root, "Depth-Anything"),
            os.path.join(self.mega_sam_root, "UniDepth"),
            os.path.join(self.mega_sam_root, "cvd_opt"),
            os.path.join(self.mega_sam_root, "cvd_opt", "core"),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(parts) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def infer(self, video_path: str, masks: np.ndarray, **kwargs) -> PerceptionResult:
        video_id = os.path.basename(video_path).split(".")[0]
        work = os.path.join(self.mega_sam_root, "work_space", video_id)
        frames_dir = os.path.join(work, "frames")
        
        # 1. Extract frames
        n_frames = _extract_frames(video_path, frames_dir)
        if n_frames == 0: return self._fallback_result(video_path, masks)

        # Align paths; clear old depth outputs to avoid stale files when frame count changes
        mono_depth_base = os.path.join(work, "da_depth")
        da_out_dir = os.path.join(mono_depth_base, video_id)
        metric_depth_base = os.path.join(work, "unidepth")
        metric_out_dir = os.path.join(metric_depth_base, video_id)

        for d, pat in [(da_out_dir, "*.npy"), (metric_out_dir, "*.npz")]:
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, pat)):
                    os.remove(f)
        os.makedirs(da_out_dir, exist_ok=True)
        os.makedirs(metric_depth_base, exist_ok=True)
        env = self._mega_sam_env()

        pdi_logger.info(f"Mega-SAM pipeline: processing {video_id} ({n_frames} frames)...")

        # 2. Depth-Anything
        r1 = subprocess.run([
            sys.executable, os.path.join(self.mega_sam_root, "Depth-Anything", "run_videos.py"),
            "--img-path", frames_dir, "--outdir", da_out_dir, "--encoder", "vitl", "--load-from", self.da_ckpt,
        ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
        if r1.returncode != 0:
            pdi_logger.error(f"Depth-Anything failed (code {r1.returncode}):\n{r1.stderr[-2000:]}")
            return self._fallback_result(video_path, masks)

        # 3. UniDepth
        r2 = subprocess.run([
            sys.executable, os.path.join(self.mega_sam_root, "UniDepth", "scripts", "demo_mega-sam.py"),
            "--img-path", frames_dir, "--outdir", metric_depth_base, "--scene-name", video_id,
        ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
        if r2.returncode != 0:
            pdi_logger.error(f"UniDepth failed (code {r2.returncode}):\n{r2.stderr[-2000:]}")
            return self._fallback_result(video_path, masks)

        # 4. DROID camera tracking
        r3 = subprocess.run([
            sys.executable, os.path.join(self.mega_sam_root, "camera_tracking_scripts", "test_demo.py"),
            "--datapath", frames_dir, "--mono_depth_path", mono_depth_base,
            "--metric_depth_path", metric_depth_base, "--scene_name", video_id,
            "--weights", self.megasam_weights, "--disable_vis",
        ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
        if r3.returncode != 0:
            pdi_logger.error(f"DROID tracking failed (code {r3.returncode}):\n{r3.stderr[-2000:]}")
            return self._fallback_result(video_path, masks)

        # 5a. RAFT flow (CVD prerequisite)
        cvd_npz_path = os.path.join(self.mega_sam_root, "outputs_cvd", f"{video_id}_sgd_cvd_hr.npz")
        use_cvd = False
        if os.path.exists(self.raft_weights):
            r4 = subprocess.run([
                sys.executable, os.path.join(self.mega_sam_root, "cvd_opt", "preprocess_flow.py"),
                "--datapath", frames_dir,
                "--model", self.raft_weights,
                "--scene_name", video_id,
                "--mixed_precision",
            ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
            if r4.returncode != 0:
                pdi_logger.warning(f"RAFT flow failed; skip CVD refinement:\n{r4.stderr[-1000:]}")
            else:
                # 5b. CVD depth refinement
                r5 = subprocess.run([
                    sys.executable, os.path.join(self.mega_sam_root, "cvd_opt", "cvd_opt.py"),
                    "--scene_name", video_id,
                    "--output_dir", "outputs_cvd",
                    "--w_grad", "2.0",
                    "--w_normal", "5.0",
                ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
                if r5.returncode != 0:
                    pdi_logger.warning(f"CVD opt failed; falling back to raw DROID output:\n{r5.stderr[-1000:]}")
                elif os.path.exists(cvd_npz_path):
                    use_cvd = True
                    pdi_logger.info("CVD depth refinement done; using temporally consistent depths")
        else:
            pdi_logger.warning(
                f"RAFT weights missing ({self.raft_weights}); skip CVD. "
                "Download under third_party/mega_sam/cvd_opt/: gdown 1R8m_jMvCun-N45XkMvHlG0P38kXy-h6I"
            )

        # 6. Load result and lift to point maps
        npz_path = cvd_npz_path if use_cvd else os.path.join(self.mega_sam_root, "outputs", f"{video_id}_droid.npz")
        if not os.path.exists(npz_path): return self._fallback_result(video_path, masks)

        data = np.load(npz_path, allow_pickle=True)
        depths = data["depths"]     # (T, H, W)
        cam_c2w = data["cam_c2w"]   # (T, 4, 4)
        fx, fy, cx, cy = self._parse_intrinsic(data["intrinsic"])
        
        T_out = min(depths.shape[0], len(masks))
        pdi_logger.info("Running 3D reprojection and scale normalization...")
        
        pointmaps = self._depth_to_pointmaps(depths[:T_out], cam_c2w[:T_out], fx, fy, cx, cy)

        # Foreground depth Z sequence
        depth_z = []
        for t in range(T_out):
            d = depths[t]
            m = masks[t]
            if m.shape[:2] != d.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            val = np.median(d[m > 0]) if np.any(m > 0) else np.median(d)
            depth_z.append(val)

        depth_z_norm = np.array(depth_z) / (depth_z[0] + 1e-8)
        h_pixel, x_center = _masks_to_h_pixel_x_center(masks[:T_out])

        return PerceptionResult(
            video_id=video_id,
            frames_count=T_out,
            masks=masks[:T_out],
            h_pixel=h_pixel,
            x_center=x_center,
            depth_z=depth_z_norm,
            focal_length=fx,
            camera_poses=cam_c2w[:T_out],
            pointmaps=pointmaps,
            metadata={"engine": "Mega-SAM-Complete-Logic"}
        )

    def _fallback_result(self, video_path: str, masks: np.ndarray) -> PerceptionResult:
        """Safe fallback when Mega-SAM subprocesses fail."""
        T = len(masks)
        h_pixel, x_center = _masks_to_h_pixel_x_center(masks)
        return PerceptionResult(
            video_id=os.path.basename(video_path).split(".")[0],
            frames_count=T,
            masks=masks,
            h_pixel=h_pixel,
            x_center=x_center,
            depth_z=np.ones(T),
            focal_length=1000.0,
            camera_poses=np.eye(4)[None].repeat(T, axis=0),
            pointmaps=np.zeros((T, masks.shape[1], masks.shape[2], 3)),
            metadata={"engine": "Mega-SAM-Fallback"}
        )
