#!/usr/bin/env python3
"""Prepare frozen SAM3 and shared-geometry inputs for A/B/C/D verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdi_eval.verification import prepare_frozen_inputs, seed_original_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-object", default="link1")
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--original-cache-dir", type=Path)
    args = parser.parse_args()
    result = prepare_frozen_inputs(
        args.video,
        args.segmentation,
        args.output_dir,
        args.selected_object,
    )
    if (args.geometry is None) != (args.original_cache_dir is None):
        parser.error("--geometry and --original-cache-dir must be provided together")
    if args.geometry is not None:
        result["original_cache"] = seed_original_cache(
            args.video,
            args.output_dir / "single_link_segmentation.npz",
            args.geometry,
            args.original_cache_dir,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
