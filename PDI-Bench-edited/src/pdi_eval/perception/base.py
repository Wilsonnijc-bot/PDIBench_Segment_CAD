from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional, Dict, Any

@dataclass
class PerceptionResult:
    """Standard perception output: bridge between models and the PDI evaluator."""
    video_id: str
    frames_count: int
    
    # --- 2D (pixel-space audit) ---
    masks: np.ndarray            # (T, H, W) binary mask sequence
    h_pixel: np.ndarray          # (T,) object pixel height h(t)
    x_center: np.ndarray         # (T,) object centroid x(t)
    tracks_2d: Optional[np.ndarray] = None # (T, N, 2) subpixel track paths
    
    # --- 3D (depth audit) ---
    depth_z: Optional[np.ndarray] = None   # (T,) or (T, H, W) depth Z(t)
    focal_length: Optional[float] = None   # implicit focal length f (e.g. from Dust3R)
    camera_poses: Optional[np.ndarray] = None # (T, 4, 4) camera extrinsics
    pointmaps: Optional[np.ndarray] = None # (T, H, W, 3) scene point map (Dust3R-style)
    
    # --- Quality / state ---
    confidence: Optional[np.ndarray] = None   # (T,) or (T, N) confidence
    is_truncated: Optional[np.ndarray] = None # (T,) touches-image-border flags
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiObjectSegmentation:
    """Named object masks from one segmentation pass."""

    video_id: str
    object_names: tuple[str, ...]
    object_ids: np.ndarray
    object_masks: np.ndarray
    h_pixel: np.ndarray
    x_center: np.ndarray
    is_truncated: np.ndarray
    union_masks: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def frames_count(self) -> int:
        return int(self.object_masks.shape[0])

    @property
    def object_count(self) -> int:
        return int(self.object_masks.shape[1])


@dataclass
class SharedGeometryResult:
    """Video-global MegaSAM geometry plus per-object camera-depth sequences."""

    video_id: str
    frames_count: int
    pointmaps: np.ndarray
    camera_poses: np.ndarray
    focal_length: float
    object_depth_z: np.ndarray
    cache_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiObjectTrackResult:
    """CoTracker output partitioned by stable object identity."""

    video_id: str
    mode: str
    object_names: tuple[str, ...]
    object_tracks: tuple[np.ndarray, ...]
    object_visibility: tuple[np.ndarray, ...]
    object_queries: tuple[np.ndarray, ...]
    background_tracks: np.ndarray
    background_visibility: np.ndarray
    background_queries: np.ndarray
    frames_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class BasePerceptor(ABC):
    """Abstract base class for perception backends."""
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def infer(self, video_input: Any, **kwargs) -> PerceptionResult:
        """Subclasses must implement unified inference."""
        pass

    def scale_coords(self, coords: np.ndarray, current_res: tuple, target_res: tuple) -> np.ndarray:
        """Scale coordinates so x, h from different models share target resolution for PDI."""
        h_ratio = target_res[0] / current_res[0]
        w_ratio = target_res[1] / current_res[1]
        scaled_coords = coords.copy().astype(float)
        scaled_coords[..., 0] *= w_ratio # x
        scaled_coords[..., 1] *= h_ratio # y
        return scaled_coords
