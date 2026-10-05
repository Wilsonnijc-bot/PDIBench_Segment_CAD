"""Edit this file to configure and run persistent masking.

See README.md in this folder for the everyday commands. API keys belong in
the project-root .env.vlm file, never in this file.
"""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

from pathlib import Path
import os
import json
import re


PROJECT_ROOT = (_workspace_root())

# Run setup. A local run uses one case; a global run uses a frozen case list.
RUN_NAME = os.environ.get("PDI_PMASK_RUN_NAME", "my_masking_run")
LEVEL = os.environ.get("PDI_PMASK_LEVEL", "local")  # "local" or "global"
CASES = os.environ.get("PDI_PMASK_CASES", "Cosmos3_0048").split(",")
VIDEO_ROOT = Path(os.environ.get("PDI_PMASK_VIDEO_ROOT", "/root/autodl-tmp/hierarchical-deformation-gdrive"))
WORK_ROOT = Path(os.environ.get("PDI_PMASK_WORK_ROOT", str((_workspace_root() / 'robot/results/persistent_masking/experiments'))))
REVIEW_ROOT = Path(os.environ.get("PDI_PMASK_REVIEW_ROOT", str((_workspace_root() / 'robot/results/persistent_masking_review'))))
INCLUDE_NAIVE_REPLAY = os.environ.get("PDI_PMASK_INCLUDE_NAIVE_REPLAY", "1") == "1"
VLM3_ENABLED = os.environ.get("PDI_PMASK_VLM3_ENABLED", "0") == "1"
VLM3_OBJECT_MASK_ROOT = os.environ.get("PDI_PMASK_VLM3_OBJECT_ROOT")
FFMPEG_BINARY = None  # None = find ffmpeg automatically; or set an absolute path.

# VLM1 finds the first deformed frame. Switch backend to "cloud_api" and fill
# in the API fields to use a hosted model instead of the local GPU model.
VLM1 = {
    "backend": "local_gpu",  # "local_gpu" or "cloud_api"
    "model": os.environ.get("PDI_PMASK_QWEN_MODEL", str((_workspace_root() / 'infrastructure/models/Qwen3.5-9B'))),
    "python": os.environ.get("PDI_PMASK_QWEN_PYTHON", str((_workspace_root() / 'infrastructure/environments/qwen/bin/python'))),
    "max_tokens": 220,
    "api_style": "chat_completions",
    "api_base": "https://openrouter.ai/api/v1",
    "api_key_env": "VLM1_API_KEY",
    "reasoning_effort": "none",
    "timeout_seconds": 240,
}

# VLM2 places SAM3 points. To use a local GPU model, switch backend to
# "local_gpu" and set model/python above just as for VLM1.
VLM2 = {
    "backend": "cloud_api",  # "local_gpu" or "cloud_api"
    "model": "gemini-3.8-flash",
    "python": str((_workspace_root() / 'infrastructure/environments/qwen/bin/python')),
    "max_tokens": 4096,
    "api_style": "chat_completions",
    "api_base": "https://api.302ai.cn/v1",
    "api_key_env": "VLM2_API_KEY",
    "reasoning_effort": None,  # None omits the parameter; "none" disables it when supported.
    "timeout_seconds": 240,
}

# One alternate request when a VLM2 point response is malformed. It uses the
# same 302.ai base URL and API key as VLM2.
VLM2_MALFORMED_FALLBACK = {
    **VLM2,
    "backend": "cloud_api",
    "api_style": "chat_completions",
    "model": "gpt-6-luna",
    "reasoning_effort": "high",
    "temperature": None,
    "max_completion_tokens": 4096,
}


def role_config(role: str) -> dict:
    """Validated VLM settings consumed by the existing pipeline."""
    if role not in {"vlm1", "vlm2", "vlm2_malformed_fallback"}:
        raise ValueError(f"Unknown VLM role: {role}")
    config = dict(VLM1 if role == "vlm1" else
                  VLM2_MALFORMED_FALLBACK if role == "vlm2_malformed_fallback" else VLM2)
    overrides = json.loads(os.environ.get('PDI_VLM_ROLE_OVERRIDES', '{}'))
    config.update(overrides.get(role, {}))
    if config.get("backend") not in {"local_gpu", "cloud_api"}:
        raise ValueError(f"{role}.backend must be local_gpu or cloud_api")
    if not config.get("model"):
        raise ValueError(f"{role}.model is required")
    if config["backend"] == "local_gpu" and not config.get("python"):
        raise ValueError(f"{role}.python is required for a local_gpu backend")
    if config["backend"] == "cloud_api":
        if config.get("api_style") not in {"chat_completions", "responses"}:
            raise ValueError(f"{role}.api_style must be chat_completions or responses")
        for field in ("api_base", "api_key_env"):
            if not config.get(field):
                raise ValueError(f"{role}.{field} is required for a cloud_api backend")
    return config


def public_config(role: str) -> dict:
    """Safe to write to run provenance; never includes the API key itself."""
    return role_config(role)


def run_config() -> dict:
    """Validate the small set of user-facing run choices."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", RUN_NAME):
        raise ValueError("RUN_NAME must use letters, digits, _ or - and start with a letter/digit")
    if LEVEL not in {"local", "global"}:
        raise ValueError("LEVEL must be local or global")
    if not CASES or len(CASES) != len(set(CASES)):
        raise ValueError("CASES must be nonempty and contain no duplicates")
    if LEVEL == "local" and len(CASES) != 1:
        raise ValueError("A local run must have exactly one case")
    for case in CASES:
        if not re.fullmatch(r"(?:LVP|Cosmos3|Cosmos25)_[0-9]{4}", case):
            raise ValueError(f"Unsupported case name: {case}")
    if not isinstance(INCLUDE_NAIVE_REPLAY, bool):
        raise ValueError("INCLUDE_NAIVE_REPLAY must be True or False")
    if VLM3_ENABLED and not VLM3_OBJECT_MASK_ROOT:
        raise ValueError("Optional VLM3 requires VLM3_OBJECT_MASK_ROOT with existing object masks")
    return {
        "name": RUN_NAME,
        "level": LEVEL,
        "cases": list(CASES),
        "video_root": Path(VIDEO_ROOT).expanduser().resolve(),
        "work_root": Path(WORK_ROOT).expanduser().resolve(),
        "review_root": Path(REVIEW_ROOT).expanduser().resolve(),
        "include_naive_replay": INCLUDE_NAIVE_REPLAY,
        "ffmpeg_binary": FFMPEG_BINARY,
        "vlm3_enabled": VLM3_ENABLED,
        "vlm3_object_root": Path(VLM3_OBJECT_MASK_ROOT).expanduser().resolve() if VLM3_OBJECT_MASK_ROOT else None,
    }
