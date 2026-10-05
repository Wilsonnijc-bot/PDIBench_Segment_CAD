"""Thin calls into pinned official Simple3D, with one documented device fix."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

import numpy as np
import torch

UPSTREAM = (_workspace_root() / 'infrastructure/vendor/Simple3D')
sys.path.insert(0, str(UPSTREAM))
from feature_extractors.FPFH import FPFHFeatures


def settings():
    # The README's MiniShift configuration, including its original MSND behavior.
    return SimpleNamespace(dataset="minishift", feature="FPFH", device="cuda:0",
                           max_nn=40, num_group=4096, group_size=128,
                           use_MSND=True, num_MSND=2, use_LFSA=True,
                           vis_save=False, level="ALL")


def provenance():
    return {"repository": "https://github.com/hustCYQ/MiniShift-Simple3D",
            "commit": "5951e097ed5f6c52c1e058490f8c64f9b6323501",
            "compatibility_patch": "features.py: move CPU s_map to CUDA idx.device before native neighborhood indexing; no changed arithmetic",
            "settings": vars(settings()),
            "source_sha256": {str(p.relative_to(UPSTREAM)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(UPSTREAM.rglob("*.py"))},
            "scalar_definition": "official MiniShift mean of largest 80 interpolated/smoothed point scores",
            "coreset_fraction": 0.05,
            "small_cloud_behavior": "native upstream FPS can repeat centers when N < 4096 (and < 1024 at score smoothing); no point upsampling"}


def _tensor(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 128 or not np.isfinite(xyz).all():
        raise ValueError("Simple3D requires finite Nx3 XYZ with at least 128 real points")
    return torch.from_numpy(np.ascontiguousarray(xyz)).unsqueeze(0)


@dataclass
class Reference:
    method: FPFHFeatures
    point_count: int


@torch.no_grad()
def build_simple3d_reference(reference_xyz):
    method = FPFHFeatures(args=settings())
    method.collect_features(_tensor(reference_xyz))
    method.run_coreset()
    if not torch.isfinite(method.patch_lib).all():
        raise ValueError("nonfinite official coreset features")
    return Reference(method, len(reference_xyz))


@torch.no_grad()
def score_simple3d(reference, test_xyz):
    # predict() invokes original descriptors, nearest prototypes, LFSA interpolation,
    # native spatial score smoothing, and original scalar aggregation. Dummy labels
    # are only required by its metric bookkeeping; no metrics use these labels.
    method = reference.method
    method.init_para()
    method.predict(_tensor(test_xyz), torch.zeros((1, len(test_xyz))), 0, path=None)
    scores = np.asarray(method.predictions[-1], dtype=np.float32).reshape(-1).copy()
    scalar = float(method.image_preds[-1])
    if scores.shape != (len(test_xyz),) or not np.isfinite(scores).all() or not np.isfinite(scalar):
        raise ValueError("invalid official Simple3D scores")
    method.init_para()
    return scores, scalar
