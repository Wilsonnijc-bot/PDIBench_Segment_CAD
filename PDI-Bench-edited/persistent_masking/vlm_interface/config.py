"""Compatibility import; edit persistent_masking/interface/config.py instead.

The vlm_interface folder remains the human-readable prompt/image reference.
Existing pipeline and experiment imports continue to use this path.
"""
from persistent_masking.interface.config import (  # noqa: F401
    PROJECT_ROOT,
    VLM1,
    VLM2,
    public_config,
    role_config,
)
