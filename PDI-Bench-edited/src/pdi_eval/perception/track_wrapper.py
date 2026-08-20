import numpy as np
import torch
import os
import cv2
import time
from dataclasses import dataclass, field
from typing import Optional

from .base import BasePerceptor, MultiObjectTrackResult, PerceptionResult

from ..utils.logger import pdi_logger

try:
    from cotracker.predictor import CoTrackerPredictor
except ImportError:
    CoTrackerPredictor = None


TRACKING_MODES = ("joint-query", "exact-group")


@dataclass
class PreparedMultiObjectTracking:
    """Decoded video and deterministic query manifest shared by tracking modes."""

    video_path: str
    video_tensor: torch.Tensor
    object_names: tuple[str, ...]
    object_queries: tuple[np.ndarray, ...]
    background_queries: np.ndarray
    original_hw: tuple[int, int]
    tracker_hw: tuple[int, int]
    scale_xy: tuple[float, float]
    frames_count: int
    timings: dict[str, float] = field(default_factory=dict)


class _FeatureReplayNetwork(torch.nn.Module):
    """Run CoTracker's video backbone once and replay its outputs per query group."""

    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network
        self._cache: list[torch.Tensor] = []
        self._input_specs: list[tuple[tuple[int, ...], torch.dtype, torch.device]] = []
        self._call_index = 0
        self._replaying = False
        self._pass_open = False
        self.backbone_forward_calls = 0

    def begin_pass(self) -> None:
        if self._pass_open:
            raise RuntimeError("CoTracker feature replay pass was not closed")
        self._call_index = 0
        self._pass_open = True

    def end_pass(self) -> None:
        if not self._pass_open:
            raise RuntimeError("CoTracker feature replay pass was not started")
        self._pass_open = False
        if not self._replaying:
            if not self._cache:
                raise RuntimeError("CoTracker did not call its video backbone")
            self._replaying = True
        elif self._call_index != len(self._cache):
            raise RuntimeError(
                "CoTracker feature replay call count changed between groups: "
                f"expected {len(self._cache)}, got {self._call_index}"
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self._pass_open:
            raise RuntimeError("CoTracker called its video backbone outside a replay pass")
        index = self._call_index
        self._call_index += 1
        if self._replaying:
            if index >= len(self._cache):
                raise RuntimeError("CoTracker feature replay call count changed between groups")
            expected = self._input_specs[index]
            actual = (tuple(values.shape), values.dtype, values.device)
            if actual != expected:
                raise RuntimeError(
                    "CoTracker feature replay input changed between groups: "
                    f"expected {expected}, got {actual}"
                )
            cached = self._cache[index]
            return cached
        output = self.network(values)
        self._cache.append(output)
        self._input_specs.append((tuple(values.shape), values.dtype, values.device))
        self.backbone_forward_calls += 1
        return output

class TrackWrapper(BasePerceptor):
    """Co-Tracker v3 wrapper for dense motion cues."""
    
    def __init__(self, checkpoint: Optional[str] = None, device: str = "cuda"):
        super().__init__(device)
        self.model = self._load_model(checkpoint)

    def _load_model(self, checkpoint):
        """Load local checkpoint or fall back to torch.hub."""
        pdi_logger.info(f"Initializing Co-Tracker (device: {self.device})...")
        
        if checkpoint is None or not os.path.exists(checkpoint):
            pdi_logger.warning("No valid local checkpoint; loading cotracker3_offline via torch.hub...")
            return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)

        try:
            model = CoTrackerPredictor(checkpoint=checkpoint).to(self.device)
            pdi_logger.success(f"Loaded local checkpoint: {checkpoint}")
            return model
        except RuntimeError as e:
            pdi_logger.warning(f"Local checkpoint load failed: {str(e)[:50]}...")
            pdi_logger.info("Falling back to torch.hub cotracker3_offline...")
            return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)

    @staticmethod
    def _sync_cuda(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _sample_region_queries(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        count: int,
        label: str,
    ) -> np.ndarray:
        queries = self._sift_sample_queries(gray, mask, region=1, n=count)
        if len(queries) < count // 2:
            extra = self._shi_tomasi_sample_queries(
                gray, mask, region=1, n=count - len(queries)
            )
            if len(extra):
                queries = np.vstack([queries, extra]) if len(queries) else extra
        def deduplicate(values: np.ndarray) -> np.ndarray:
            unique = []
            seen = set()
            for query in values:
                key = (round(float(query[1]), 3), round(float(query[2]), 3))
                if key not in seen:
                    seen.add(key)
                    unique.append(query)
            return np.asarray(unique[:count], dtype=np.float32).reshape(-1, 3)

        queries = deduplicate(queries)
        if len(queries) < 2:
            pdi_logger.warning(f"Very few unique {label} features; using uniform grid")
            grid = self._grid_sample_queries(mask, region=1, n=count)
            candidates = np.vstack([queries, grid]) if len(queries) else grid
            queries = deduplicate(candidates)
        if len(queries) < 2:
            raise ValueError(
                f"{label} mask produced fewer than two unique CoTracker queries"
            )
        return queries

    def prepare_multi(
        self,
        video_path: str,
        initial_masks: np.ndarray,
        object_names: tuple[str, ...] | list[str],
        grid_size: int = 10,
        bg_grid_size: int = 15,
        background_dilation: int = 5,
        max_dim: int = 880,
    ) -> PreparedMultiObjectTracking:
        """Decode once and build a common query manifest for both tracking modes."""
        started = time.perf_counter()
        initial_masks = np.asarray(initial_masks, dtype=bool)
        if initial_masks.ndim != 3:
            raise ValueError(
                f"initial_masks must have shape (N,H,W), got {initial_masks.shape}"
            )
        object_names = tuple(str(name) for name in object_names)
        if len(object_names) != len(initial_masks):
            raise ValueError("object_names must match initial_masks axis 0")
        if background_dilation < 0:
            raise ValueError("background_dilation cannot be negative")
        if grid_size < 1 or bg_grid_size < 1:
            raise ValueError("grid sizes must be positive")
        if max_dim < 1:
            raise ValueError("max_dim must be positive")

        capture = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if not frames:
            raise ValueError(f"Cannot decode video: {video_path}")

        original_hw = tuple(int(value) for value in initial_masks.shape[1:])
        tracker_hw = tuple(int(value) for value in frames[0].shape[:2])
        scale_x = original_hw[1] / tracker_hw[1]
        scale_y = original_hw[0] / tracker_hw[0]
        small_masks = np.stack(
            [
                cv2.resize(
                    mask.astype(np.uint8),
                    (tracker_hw[1], tracker_hw[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                for mask in initial_masks
            ]
        )
        gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        foreground_count = grid_size * grid_size
        object_queries = tuple(
            self._sample_region_queries(gray, mask, foreground_count, name)
            for name, mask in zip(object_names, small_masks)
        )

        union_mask = np.any(small_masks, axis=0).astype(np.uint8)
        if background_dilation:
            size = background_dilation * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            union_mask = cv2.dilate(union_mask, kernel)
        background_region = union_mask == 0
        background_queries = self._sample_region_queries(
            gray,
            background_region,
            bg_grid_size * bg_grid_size,
            "shared background",
        )

        video_np = np.stack(frames)
        video_tensor = (
            torch.from_numpy(video_np)
            .permute(0, 3, 1, 2)[None]
            .to(self.device)
        )
        return PreparedMultiObjectTracking(
            video_path=video_path,
            video_tensor=video_tensor,
            object_names=object_names,
            object_queries=object_queries,
            background_queries=background_queries,
            original_hw=original_hw,
            tracker_hw=tracker_hw,
            scale_xy=(scale_x, scale_y),
            frames_count=len(frames),
            timings={"decode_and_query_seconds": time.perf_counter() - started},
        )

    def _run_queries(
        self,
        video_tensor: torch.Tensor,
        queries_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(queries_np) < 2:
            raise ValueError("CoTracker requires at least two explicit queries")
        queries = torch.from_numpy(queries_np[None].astype(np.float32)).to(self.device)
        autocast = torch.autocast(
            device_type="cuda",
            enabled=self.device.type == "cuda",
        )
        with torch.no_grad(), autocast:
            tracks, visibility = self.model(
                video_tensor.float(),
                queries=queries,
                grid_size=0,
                grid_query_frame=0,
            )
        tracks_np = tracks[0].cpu().numpy()
        visibility_np = visibility[0].cpu().numpy()
        expected = (video_tensor.shape[1], len(queries_np))
        if tracks_np.shape != (*expected, 2) or visibility_np.shape != expected:
            raise RuntimeError(
                "CoTracker returned unexpected shapes: "
                f"tracks={tracks_np.shape}, visibility={visibility_np.shape}, "
                f"expected=({expected[0]},{expected[1]},2)/{expected}"
            )
        return tracks_np, visibility_np

    def _core_model_with_fnet(self):
        core = getattr(self.model, "model", None)
        if core is None or not hasattr(core, "fnet"):
            raise RuntimeError(
                "exact-group mode requires CoTrackerPredictor with an accessible model.fnet"
            )
        return core

    def _scale_and_filter_group(
        self,
        tracks: np.ndarray,
        visibility: np.ndarray,
        queries: np.ndarray,
        scale_xy: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tracks = tracks.copy()
        queries = queries.copy()
        tracks[:, :, 0] *= scale_xy[0]
        tracks[:, :, 1] *= scale_xy[1]
        queries[:, 1] *= scale_xy[0]
        queries[:, 2] *= scale_xy[1]
        keep = self._track_keep_mask(tracks, visibility)
        return tracks[:, keep], visibility[:, keep], queries[keep]

    def track_prepared(
        self,
        prepared: PreparedMultiObjectTracking,
        mode: str,
    ) -> MultiObjectTrackResult:
        """Track one query manifest jointly or with isolated feature-reusing groups."""
        if mode not in TRACKING_MODES:
            raise ValueError(f"tracking mode must be one of {TRACKING_MODES}, got {mode!r}")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sync_cuda(self.device)
        model_started = time.perf_counter()

        raw_object_tracks: list[np.ndarray] = []
        raw_object_visibility: list[np.ndarray] = []
        if mode == "joint-query":
            counts = [len(queries) for queries in prepared.object_queries]
            counts.append(len(prepared.background_queries))
            all_queries = np.concatenate(
                [*prepared.object_queries, prepared.background_queries], axis=0
            )
            all_tracks, all_visibility = self._run_queries(
                prepared.video_tensor, all_queries
            )
            offset = 0
            for count in counts[:-1]:
                raw_object_tracks.append(all_tracks[:, offset:offset + count])
                raw_object_visibility.append(all_visibility[:, offset:offset + count])
                offset += count
            raw_background_tracks = all_tracks[:, offset:offset + counts[-1]]
            raw_background_visibility = all_visibility[:, offset:offset + counts[-1]]
            model_forward_count = 1
            backbone_chunks = 1
        else:
            core = self._core_model_with_fnet()
            original_fnet = core.fnet
            replay_fnet = _FeatureReplayNetwork(original_fnet)
            core.fnet = replay_fnet
            try:
                for queries in prepared.object_queries:
                    replay_fnet.begin_pass()
                    tracks, visibility = self._run_queries(
                        prepared.video_tensor, queries
                    )
                    replay_fnet.end_pass()
                    raw_object_tracks.append(tracks)
                    raw_object_visibility.append(visibility)
                replay_fnet.begin_pass()
                raw_background_tracks, raw_background_visibility = self._run_queries(
                    prepared.video_tensor, prepared.background_queries
                )
                replay_fnet.end_pass()
            finally:
                core.fnet = original_fnet
            model_forward_count = len(prepared.object_queries) + 1
            backbone_chunks = replay_fnet.backbone_forward_calls

        self._sync_cuda(self.device)
        model_seconds = time.perf_counter() - model_started
        filter_started = time.perf_counter()
        object_tracks = []
        object_visibility = []
        object_queries = []
        for tracks, visibility, queries in zip(
            raw_object_tracks,
            raw_object_visibility,
            prepared.object_queries,
        ):
            filtered = self._scale_and_filter_group(
                tracks, visibility, queries, prepared.scale_xy
            )
            object_tracks.append(filtered[0])
            object_visibility.append(filtered[1])
            object_queries.append(filtered[2])
        background = self._scale_and_filter_group(
            raw_background_tracks,
            raw_background_visibility,
            prepared.background_queries,
            prepared.scale_xy,
        )
        filter_seconds = time.perf_counter() - filter_started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )

        metadata = {
            **prepared.timings,
            "model_seconds": model_seconds,
            "filter_seconds": filter_seconds,
            "total_tracking_seconds": model_seconds + filter_seconds,
            "model_forward_count": model_forward_count,
            "video_backbone_forward_passes": 1,
            "video_backbone_chunks": backbone_chunks,
            "peak_gpu_memory_bytes": peak_memory,
            "tracker_hw": list(prepared.tracker_hw),
            "source_hw": list(prepared.original_hw),
            "foreground_query_counts": [len(value) for value in object_queries],
            "background_query_count": len(background[2]),
        }
        pdi_logger.info(
            f"CoTracker {mode}: {model_seconds:.3f}s model, "
            f"{sum(len(value) for value in object_queries)} foreground and "
            f"{len(background[2])} background tracks kept"
        )
        return MultiObjectTrackResult(
            video_id=os.path.basename(prepared.video_path),
            mode=mode,
            object_names=prepared.object_names,
            object_tracks=tuple(object_tracks),
            object_visibility=tuple(object_visibility),
            object_queries=tuple(object_queries),
            background_tracks=background[0],
            background_visibility=background[1],
            background_queries=background[2],
            frames_count=prepared.frames_count,
            metadata=metadata,
        )

    def infer_multi(
        self,
        video_path: str,
        initial_masks: np.ndarray,
        object_names: tuple[str, ...] | list[str],
        mode: str = "joint-query",
        **kwargs,
    ) -> MultiObjectTrackResult:
        prepared = self.prepare_multi(
            video_path,
            initial_masks,
            object_names,
            **kwargs,
        )
        try:
            return self.track_prepared(prepared, mode)
        finally:
            del prepared.video_tensor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def infer(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        grid_size: int = 10,
        bg_grid_size: int = 15,
        **kwargs,
    ) -> PerceptionResult:
        """Joint foreground+background tracking in one forward pass.

        Foreground queries: SIFT -> Shi-Tomasi -> uniform grid (three-level fallback).
        Background: Shi-Tomasi -> uniform grid.
        Merged queries run once; split by n_fg afterward.
        Background tracks go to metadata['bg_tracks'] / metadata['bg_visibility'].
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        frames = []
        max_dim = 880
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        orig_h, orig_w = initial_mask.shape
        curr_h, curr_w = frames[0].shape[:2]
        scale_x, scale_y = orig_w / curr_w, orig_h / curr_h

        video_np = np.stack(frames)
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].to(self.device)

        small_mask = cv2.resize(
            initial_mask.astype(np.uint8), (curr_w, curr_h),
            interpolation=cv2.INTER_NEAREST,
        )
        small_mask = (small_mask > 0).astype(np.uint8)

        first_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        n_fg = grid_size * grid_size
        fg_queries_np = self._sift_sample_queries(first_frame_gray, small_mask, region=1, n=n_fg)
        if len(fg_queries_np) < n_fg // 2:
            pdi_logger.info(f"Few SIFT fg points ({len(fg_queries_np)}); adding Shi-Tomasi")
            extra = self._shi_tomasi_sample_queries(first_frame_gray, small_mask, region=1, n=n_fg - len(fg_queries_np))
            fg_queries_np = np.vstack([fg_queries_np, extra]) if len(fg_queries_np) > 0 else extra
        if len(fg_queries_np) < 2:
            pdi_logger.warning("Very few fg feature points; uniform grid fallback")
            fg_queries_np = self._grid_sample_queries(small_mask, region=1, n=n_fg)

        n_bg = bg_grid_size * bg_grid_size
        bg_queries_np = self._shi_tomasi_sample_queries(first_frame_gray, small_mask, region=0, n=n_bg)
        if len(bg_queries_np) < 2:
            pdi_logger.warning("Very few bg corners; uniform grid fallback")
            bg_queries_np = self._grid_sample_queries(small_mask, region=0, n=n_bg)

        n_fg_pts = len(fg_queries_np)
        all_queries_np = np.vstack([fg_queries_np, bg_queries_np]).astype(np.float32)

        pdi_logger.info(
            f"Co-Tracker ({curr_w}x{curr_h}, "
            f"fg:{n_fg_pts} pts, bg:{len(bg_queries_np)} pts)..."
        )

        if len(all_queries_np) >= 2:
            queries = torch.from_numpy(all_queries_np[None]).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        queries=queries,
                        grid_size=0,
                        grid_query_frame=0,
                    )
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        grid_size=grid_size,
                        grid_query_frame=0,
                    )
            n_fg_pts = tracks.shape[2]

        tracks_np = tracks[0].cpu().numpy()
        tracks_np[:, :, 0] *= scale_x
        tracks_np[:, :, 1] *= scale_y
        vis_np = visibility[0].cpu().numpy()

        fg_tracks = tracks_np[:, :n_fg_pts, :]
        bg_tracks = tracks_np[:, n_fg_pts:, :]
        fg_vis    = vis_np[:, :n_fg_pts]
        bg_vis    = vis_np[:, n_fg_pts:]

        fg_tracks, fg_vis = self._filter_tracks(fg_tracks, fg_vis)
        bg_tracks, bg_vis = self._filter_tracks(bg_tracks, bg_vis)

        breathing_metric = self.calculate_breathing_artifact(fg_tracks)

        del video_tensor
        torch.cuda.empty_cache()

        n_fg_kept = fg_tracks.shape[1]
        tracking_confidence = float(fg_vis.mean()) if fg_vis.size > 0 else 0.0
        pdi_logger.info(f"Tracking done: kept {n_fg_kept} fg points, mean visibility {tracking_confidence:.3f}")

        return PerceptionResult(
            video_id=os.path.basename(video_path),
            frames_count=len(fg_tracks),
            masks=np.zeros((1, 1, 1)),
            h_pixel=np.zeros(len(fg_tracks)),
            x_center=np.zeros(len(fg_tracks)),
            tracks_2d=fg_tracks,
            confidence=fg_vis,
            metadata={
                "breathing_metric": breathing_metric,
                "tracking_confidence": tracking_confidence,
                "bg_tracks": bg_tracks,
                "bg_visibility": bg_vis,
            },
        )

    def _sift_sample_queries(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        region: int,
        n: int,
    ) -> np.ndarray:
        """SIFT keypoints inside mask region (scale/rotation invariant).

        Returns top-n by response as (M, 3) -> [frame=0, x, y].
        """
        sift = cv2.SIFT_create(nfeatures=n * 4)
        kps = sift.detect(gray, None)
        if not kps:
            return np.empty((0, 3), dtype=np.float32)

        kps = sorted(kps, key=lambda k: k.response, reverse=True)
        h, w = mask.shape
        pts = []
        for kp in kps:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= y < h and 0 <= x < w and mask[y, x] == region:
                pts.append([0.0, float(kp.pt[0]), float(kp.pt[1])])
            if len(pts) >= n:
                break

        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)

    def _shi_tomasi_sample_queries(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        region: int,
        n: int,
    ) -> np.ndarray:
        """Shi-Tomasi corners inside mask region.

        Returns (M, 3) -> [frame=0, x, y].
        """
        region_mask = (mask == region).astype(np.uint8) * 255
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=n,
            qualityLevel=0.01,
            minDistance=5,
            mask=region_mask,
        )
        if corners is None:
            return np.empty((0, 3), dtype=np.float32)

        pts = [[0.0, float(c[0][0]), float(c[0][1])] for c in corners]
        return np.array(pts, dtype=np.float32)

    def _grid_sample_queries(
        self,
        mask: np.ndarray,
        region: int,
        n: int,
    ) -> np.ndarray:
        """Uniform spatial grid over region (1=foreground / 0=background).

        sqrt(n) x sqrt(n) cells; one random point per occupied cell.
        Returns (M, 3) [frame=0, x, y], M <= n.
        """
        yy, xx = np.where(mask == region)
        if len(yy) == 0:
            if region == 1:
                pdi_logger.warning("Initial mask has no foreground; bg-only grid tracking")
            return np.empty((0, 3), dtype=np.float32)

        n = min(n, len(yy))
        side = max(1, int(np.ceil(np.sqrt(n))))
        h, w = mask.shape
        cell_h = max(1, h // side)
        cell_w = max(1, w // side)

        rng = np.random.default_rng(42)
        pts = []
        for gy in range(side):
            for gx in range(side):
                y0, y1 = gy * cell_h, min((gy + 1) * cell_h, h)
                x0, x1 = gx * cell_w, min((gx + 1) * cell_w, w)
                in_cell = np.where((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1))[0]
                if len(in_cell) > 0:
                    pick = rng.choice(in_cell)
                    pts.append([0.0, float(xx[pick]), float(yy[pick])])
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break

        if len(pts) < n:
            selected = {(int(point[1]), int(point[2])) for point in pts}
            available = np.asarray(
                [
                    index
                    for index in range(len(yy))
                    if (int(xx[index]), int(yy[index])) not in selected
                ],
                dtype=np.int64,
            )
            if len(available):
                fill = rng.choice(
                    available,
                    size=min(n - len(pts), len(available)),
                    replace=False,
                )
                pts.extend(
                    [0.0, float(xx[index]), float(yy[index])] for index in fill
                )

        return np.array(pts, dtype=np.float32)

    def _filter_tracks(
        self,
        tracks: np.ndarray,
        vis: np.ndarray,
        min_vis_ratio: float = 0.3,
        max_jump_px: float = 120.0,
    ) -> tuple:
        """Drop low-quality tracks.

        Args:
            tracks:        (T, N, 2)
            vis:           (T, N) Co-Tracker visibility
            min_vis_ratio: fraction of frames that must be visible
            max_jump_px:   max per-frame motion; larger treated as jump

        Returns:
            filtered_tracks (T, M, 2), filtered_vis (T, M)
        """
        keep = self._track_keep_mask(
            tracks,
            vis,
            min_vis_ratio=min_vis_ratio,
            max_jump_px=max_jump_px,
        )
        return tracks[:, keep, :], vis[:, keep]

    def _track_keep_mask(
        self,
        tracks: np.ndarray,
        vis: np.ndarray,
        min_vis_ratio: float = 0.3,
        max_jump_px: float = 120.0,
    ) -> np.ndarray:
        """Return the quality selector so queries and tracks stay aligned."""
        frame_count, track_count, _ = tracks.shape
        if track_count == 0:
            return np.zeros(0, dtype=bool)
        finite_ok = np.isfinite(tracks).all(axis=(0, 2)) & np.isfinite(vis).all(axis=0)
        vis_ratio = vis.mean(axis=0)
        vis_ok = vis_ratio >= min_vis_ratio
        if frame_count > 1:
            delta = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
            max_jump = np.max(delta, axis=0)
            jump_ok = max_jump < max_jump_px
        else:
            max_jump = np.zeros(track_count, dtype=np.float64)
            jump_ok = np.ones(track_count, dtype=bool)
        keep = finite_ok & vis_ok & jump_ok
        removed = int((~keep).sum())
        if removed:
            pdi_logger.info(
                f"Track filter: removed {removed}/{track_count} low-quality points "
                f"(low vis:{int((~vis_ok).sum())} jumps:{int((~jump_ok).sum())})"
            )
        if keep.sum() < 2:
            candidates = np.flatnonzero(finite_ok)
            if len(candidates) < 2:
                raise ValueError(
                    f"CoTracker produced only {len(candidates)} finite tracks out of "
                    f"{track_count} queries"
                )
            pdi_logger.warning(
                f"Only {int(keep.sum())}/{track_count} tracks passed thresholds; "
                "keeping the two best finite tracks"
            )
            ranked = sorted(
                candidates.tolist(),
                key=lambda index: (-float(vis_ratio[index]), float(max_jump[index]), index),
            )
            keep = np.zeros(track_count, dtype=bool)
            keep[ranked[:2]] = True
        return keep

    def calculate_breathing_artifact(self, tracks: np.ndarray) -> float:
        """Relative distance CV between two track groups (internal spread)."""
        T, N, _ = tracks.shape
        if N < 2: return 0.0
        
        group_a = tracks[:, :min(5, N//2), :]
        group_b = tracks[:, -min(5, N//2):, :]
        
        centroid_a = np.mean(group_a, axis=1)
        centroid_b = np.mean(group_b, axis=1)
        
        dists = np.linalg.norm(centroid_a - centroid_b, axis=1)
        
        if np.mean(dists) < 1e-6: return 0.0
        return float(np.std(dists) / np.mean(dists))
