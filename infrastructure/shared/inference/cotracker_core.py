"""CoTracker loading, query sampling and feature replay for V1."""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from typing import Optional

from infrastructure.shared.contracts.base import BasePerceptor
from infrastructure.shared.io.logger import pdi_logger

try:
    from cotracker.predictor import CoTrackerPredictor
except ImportError:
    CoTrackerPredictor = None


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


class CoTrackerCore(BasePerceptor):
    """Version-neutral model execution and deterministic query sampling."""

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
        snap_to_mask_pixels: bool = False,
    ) -> np.ndarray:
        candidate_count = max(count * 4, count)
        candidate_groups = (
            self._sift_sample_queries(gray, mask, region=1, n=candidate_count),
            self._shi_tomasi_sample_queries(
                gray, mask, region=1, n=candidate_count
            ),
            self._grid_sample_queries(mask, region=1, n=count),
        )
        populated = [group for group in candidate_groups if len(group)]
        candidates = (
            np.vstack(populated)
            if populated
            else np.empty((0, 3), dtype=np.float32)
        )
        if snap_to_mask_pixels:
            # SIFT centers are subpixel, but eligibility was checked at the
            # rounded mask pixel. Keep the query on that exact pixel center.
            candidates[:, 1:3] = np.rint(candidates[:, 1:3])
        queries = self._spatially_balance_queries(candidates, count)
        if len(queries) < 2:
            raise ValueError(
                f"{label} mask produced fewer than two unique CoTracker queries"
            )
        return queries

    @staticmethod
    def _spatially_balance_queries(values: np.ndarray, count: int) -> np.ndarray:
        unique = []
        seen = set()
        for query in np.asarray(values, dtype=np.float32):
            key = (round(float(query[1]), 3), round(float(query[2]), 3))
            if key not in seen:
                seen.add(key)
                unique.append(query)
        candidates = np.asarray(unique, dtype=np.float32).reshape(-1, 3)
        if len(candidates) <= count:
            return candidates

        coordinates = candidates[:, 1:3].astype(np.float64)
        selected = [0]
        available = np.ones(len(candidates), dtype=bool)
        available[0] = False
        minimum_distance = np.sum(
            (coordinates - coordinates[0]) ** 2, axis=1
        )
        while len(selected) < count:
            scores = np.where(available, minimum_distance, -1.0)
            index = int(np.argmax(scores))
            if scores[index] < 0:
                break
            selected.append(index)
            available[index] = False
            distance = np.sum((coordinates - coordinates[index]) ** 2, axis=1)
            minimum_distance = np.minimum(minimum_distance, distance)
        return candidates[selected]

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
        """Deterministic spatial grid over region (1=foreground / 0=background).

        sqrt(n) x sqrt(n) cells; one center-nearest point per occupied cell.
        Returns (M, 3) [frame=0, x, y], M <= n.
        """
        yy, xx = np.where(mask == region)
        if len(yy) == 0:
            if region == 1:
                pdi_logger.warning("Initial mask has no foreground; bg-only grid tracking")
            return np.empty((0, 3), dtype=np.float32)

        n = min(n, len(yy))
        side = max(1, int(np.ceil(np.sqrt(n))))
        y_min, y_max = int(yy.min()), int(yy.max()) + 1
        x_min, x_max = int(xx.min()), int(xx.max()) + 1
        y_edges = np.linspace(y_min, y_max, side + 1, dtype=int)
        x_edges = np.linspace(x_min, x_max, side + 1, dtype=int)
        pts = []
        for gy in range(side):
            for gx in range(side):
                y0, y1 = y_edges[gy], y_edges[gy + 1]
                x0, x1 = x_edges[gx], x_edges[gx + 1]
                in_cell = np.where((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1))[0]
                if len(in_cell) > 0:
                    center_x = (x0 + x1 - 1) / 2.0
                    center_y = (y0 + y1 - 1) / 2.0
                    distances = (
                        (xx[in_cell] - center_x) ** 2
                        + (yy[in_cell] - center_y) ** 2
                    )
                    pick = in_cell[int(np.argmin(distances))]
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
                fill_count = min(n - len(pts), len(available))
                fill = available[
                    np.linspace(0, len(available) - 1, fill_count, dtype=int)
                ]
                pts.extend(
                    [0.0, float(xx[index]), float(yy[index])] for index in fill
                )

        return np.array(pts, dtype=np.float32)
