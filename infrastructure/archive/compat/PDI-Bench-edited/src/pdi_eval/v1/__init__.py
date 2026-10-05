"""Historical multi-object PDI (V1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def evaluate(
    config: dict[str, Any],
    *,
    video_path: str,
    segmentation_npz: str,
    output_dir: str | Path | None = None,
    geometry_cache_dir: str | Path | None = None,
    tracking_modes: Iterable[str] = ("exact-group",),
) -> dict[str, Any]:
    """Run historical per-link PDI with explicit tracking modes."""
    from .pipeline import MultiObjectPDIEvaluationPipeline

    return MultiObjectPDIEvaluationPipeline(config).run(
        video_path=video_path,
        segmentation_npz=segmentation_npz,
        tracking_modes=tracking_modes,
        output_dir=output_dir,
        geometry_cache_dir=geometry_cache_dir,
    )


__all__ = ["evaluate", "MultiObjectPDIEvaluationPipeline", "TRACKING_MODES"]


def __getattr__(name: str):
    if name == "MultiObjectPDIEvaluationPipeline":
        from .pipeline import MultiObjectPDIEvaluationPipeline

        return MultiObjectPDIEvaluationPipeline
    if name == "TRACKING_MODES":
        from .tracking import TRACKING_MODES

        return TRACKING_MODES
    raise AttributeError(name)
