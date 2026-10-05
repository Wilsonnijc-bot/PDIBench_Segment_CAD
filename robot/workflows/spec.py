"""One serializable specification for selection, masking, V1, and replay."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from string import Template

from infrastructure.shared.contracts.contracts import ROOT


@dataclass(frozen=True)
class ExperimentSpec:
    output_root: Path
    selection: Path
    workbook: Path
    video_root: Path
    sam_python: Path
    qwen_python: Path
    qwen_model: Path
    pdi_python: Path
    tracker_checkpoint: Path
    concurrency: int
    gpu_slots: int
    generate_base_masks: bool = False
    segmentation_python: Path | None = None
    reference_dir: Path | None = None
    dinov2_model: Path | None = None
    sam3_checkpoint: Path | None = None
    sam3_bpe: Path | None = None
    workflow: str = "v1"
    tapip3d_python: Path | None = None
    tapip3d_repository: Path | None = None
    tapip3d_checkpoint: Path | None = None

    @property
    def gpu_lock(self) -> Path:
        return self.output_root / "gpu.lock"


def load_spec(path: Path) -> ExperimentSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("experiment spec requires schema_version 1")
    variables = dict(os.environ)
    variables.setdefault("PDI_GPU_ROOT", "/root/autodl-tmp/pdi")
    variables.setdefault("PDI_RUN_ROOT", variables["PDI_GPU_ROOT"] + "/experiments/v1-persistent-10")

    def location(name: str) -> Path:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"experiment spec requires {name}")
        expanded = Template(value).substitute(variables)
        selected = Path(expanded).expanduser()
        # Preserve venv executable symlinks: resolving env/qwen/bin/python can
        # enter the base SAM environment and silently drop Qwen dependencies.
        return selected.absolute() if selected.is_absolute() else (ROOT / selected).absolute()

    if "versions" in raw:
        raise ValueError("versions is no longer supported; this experiment scores V1 only")
    workflow = raw.get("workflow", "v1")
    if workflow not in {"v1", "selected45_link5_link7_four_way", "selected45_link5_only"}:
        raise ValueError(f"unsupported experiment workflow: {workflow}")
    concurrency = int(raw.get("concurrency", 2))
    gpu_slots = int(raw.get("gpu_slots", 1))
    if concurrency < 1 or gpu_slots != 1:
        raise ValueError("concurrency must be positive and this GPU profile requires one GPU slot")
    return ExperimentSpec(
        output_root=location("output_root"), selection=location("selection"),
        workbook=location("workbook"), video_root=location("video_root"),
        sam_python=location("sam_python"), qwen_python=location("qwen_python"),
        qwen_model=location("qwen_model"), pdi_python=location("pdi_python"),
        tracker_checkpoint=location("tracker_checkpoint"),
        concurrency=concurrency, gpu_slots=gpu_slots,
        generate_base_masks=bool(raw.get("generate_base_masks", False)),
        segmentation_python=location("segmentation_python") if raw.get("generate_base_masks") else None,
        reference_dir=location("reference_dir") if raw.get("generate_base_masks") else None,
        dinov2_model=location("dinov2_model") if raw.get("generate_base_masks") else None,
        sam3_checkpoint=location("sam3_checkpoint") if raw.get("generate_base_masks") else None,
        sam3_bpe=location("sam3_bpe") if raw.get("generate_base_masks") else None,
        workflow=workflow,
        tapip3d_python=location("tapip3d_python") if workflow == "selected45_link5_link7_four_way" else None,
        tapip3d_repository=location("tapip3d_repository") if workflow == "selected45_link5_link7_four_way" else None,
        tapip3d_checkpoint=location("tapip3d_checkpoint") if workflow == "selected45_link5_link7_four_way" else None,
    )
