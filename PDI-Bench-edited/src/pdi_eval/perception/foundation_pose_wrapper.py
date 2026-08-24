"""Numeric artifact boundary for the isolated FoundationPose environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from ..geometry.se3 import rigid_transform_valid
from .base import FoundationPoseResult


POSE_ARCHIVE_SCHEMA = 1
POSE_SOURCE_INVALID = 0
POSE_SOURCE_REGISTER = 1
POSE_SOURCE_TRACK = 2
POSE_SOURCE_REREGISTER = 3


def run_foundation_pose_worker(
    *,
    geometry_npz: str | Path,
    segmentation_npz: str | Path,
    cad_manifest: str | Path,
    output_npz: str | Path,
    config: dict,
    benchmark_root: str | Path,
) -> Path:
    """Invoke the isolated FoundationPose worker through numeric artifacts."""
    root = Path(benchmark_root).resolve()
    output = Path(output_npz).resolve()
    python = Path(
        config.get(
            "foundation_pose_python",
            os.environ.get(
                "PDI_FOUNDATIONPOSE_PYTHON",
                "/root/autodl-tmp/pdi/env/foundationpose/bin/python",
            ),
        )
    ).resolve()
    source = Path(
        config.get(
            "foundation_pose_source",
            os.environ.get(
                "PDI_FOUNDATIONPOSE_SOURCE",
                "/root/autodl-tmp/pdi/cache/src/FoundationPose",
            ),
        )
    ).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"FoundationPose Python is missing: {python}")
    if not (source / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose source is missing: {source}")
    command = [
        str(python),
        "-m",
        "pdi_eval.perception.foundation_pose_worker",
        "--geometry-npz",
        str(Path(geometry_npz).resolve()),
        "--segmentation-npz",
        str(Path(segmentation_npz).resolve()),
        "--cad-manifest",
        str(Path(cad_manifest).resolve()),
        "--output-npz",
        str(output),
        "--scale-policy",
        str(config.get("scale_policy", "video-global-cad")),
        "--registration-iterations",
        str(int(config.get("registration_iterations", 5))),
        "--tracking-iterations",
        str(int(config.get("tracking_iterations", 2))),
        "--reregister-interval",
        str(int(config.get("reregister_interval", 10))),
    ]
    environment = os.environ.copy()
    environment["PDI_FOUNDATIONPOSE_SOURCE"] = str(source)
    python_path = [str(root / "src"), str(source), str(source / "mycpp/build")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            "FoundationPose worker failed with status "
            f"{result.returncode}; see {log_path}\n" + "\n".join(tail)
        )
    load_foundation_pose_archive(output)
    return output


def _optional_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
    *,
    fill: float,
    dtype,
) -> np.ndarray:
    if name not in archive.files:
        return np.full(shape, fill, dtype=dtype)
    value = np.asarray(archive[name], dtype=dtype)
    if value.shape != shape:
        raise ValueError(f"FoundationPose {name} has shape {value.shape}; expected {shape}")
    return value


def load_foundation_pose_archive(
    path: str | Path,
    *,
    expected_link_names: Iterable[str] | None = None,
    expected_frame_count: int | None = None,
) -> FoundationPoseResult:
    """Load and validate a pickle-free FoundationPose worker result."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FoundationPose archive is missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "link_names",
            "frame_indices",
            "frame_times_seconds",
            "T_C_from_L",
            "pose_valid",
            "pose_source",
            "video_depth_scale",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"FoundationPose archive lacks arrays: {missing}")
        link_names = tuple(str(value) for value in np.asarray(archive["link_names"]))
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int32)
        frame_times = np.asarray(archive["frame_times_seconds"], dtype=np.float64)
        transforms = np.asarray(archive["T_C_from_L"], dtype=np.float64)
        pose_valid = np.asarray(archive["pose_valid"], dtype=bool)
        pose_source = np.asarray(archive["pose_source"], dtype=np.uint8)
        depth_scale = float(np.asarray(archive["video_depth_scale"]).reshape(-1)[0])
        metadata = (
            json.loads(str(np.asarray(archive["metadata_json"]).item()))
            if "metadata_json" in archive.files
            else {}
        )
        expected_shape = (len(frame_indices), len(link_names))
        pose_objective = _optional_array(
            archive, "pose_objective", expected_shape, fill=np.inf, dtype=np.float32
        )
        silhouette_iou = _optional_array(
            archive, "silhouette_iou", expected_shape, fill=np.nan, dtype=np.float32
        )
        depth_residual = _optional_array(
            archive,
            "pose_depth_residual",
            expected_shape,
            fill=np.nan,
            dtype=np.float32,
        )

    if not link_names or len(set(link_names)) != len(link_names):
        raise ValueError("FoundationPose link names must be non-empty and unique")
    if expected_link_names is not None and link_names != tuple(expected_link_names):
        raise ValueError(
            f"FoundationPose links {link_names} do not match {tuple(expected_link_names)}"
        )
    if expected_frame_count is not None and len(frame_indices) != expected_frame_count:
        raise ValueError(
            f"FoundationPose has {len(frame_indices)} frames; expected {expected_frame_count}"
        )
    if transforms.shape != (*pose_valid.shape, 4, 4):
        raise ValueError("T_C_from_L and pose_valid axes do not match")
    if pose_valid.shape != (len(frame_indices), len(link_names)):
        raise ValueError("pose_valid does not match frame/link axes")
    if pose_source.shape != pose_valid.shape:
        raise ValueError("pose_source does not match pose_valid")
    if frame_times.shape != (len(frame_indices),):
        raise ValueError("frame_times_seconds must have shape (T,)")
    if len(frame_indices) and (
        not np.array_equal(frame_indices, np.arange(len(frame_indices), dtype=np.int32))
        or not np.isfinite(frame_times).all()
        or np.any(np.diff(frame_times) <= 0.0)
    ):
        raise ValueError("FoundationPose frames/timestamps must be ordered and contiguous")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("FoundationPose video_depth_scale must be finite and positive")
    unknown_sources = set(np.unique(pose_source)).difference(
        {
            POSE_SOURCE_INVALID,
            POSE_SOURCE_REGISTER,
            POSE_SOURCE_TRACK,
            POSE_SOURCE_REREGISTER,
        }
    )
    if unknown_sources:
        raise ValueError(f"FoundationPose archive has unknown pose sources: {unknown_sources}")
    for frame_index, link_index in zip(*np.where(pose_valid)):
        if not rigid_transform_valid(transforms[frame_index, link_index]):
            raise ValueError(
                "FoundationPose marks an invalid rigid transform as valid at "
                f"frame={frame_index}, link={link_names[link_index]}"
            )
    invalid_slots = ~pose_valid
    transforms = transforms.copy()
    transforms[invalid_slots] = np.nan
    return FoundationPoseResult(
        link_names=link_names,
        frame_indices=frame_indices,
        frame_times_seconds=frame_times,
        T_C_from_L=transforms,
        pose_valid=pose_valid,
        pose_source=pose_source,
        pose_objective=pose_objective,
        silhouette_iou=silhouette_iou,
        pose_depth_residual=depth_residual,
        video_depth_scale=depth_scale,
        metadata={"schema": POSE_ARCHIVE_SCHEMA, "archive": str(source), **metadata},
    )
