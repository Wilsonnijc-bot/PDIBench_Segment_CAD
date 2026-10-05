"""Run one DINOv2 + SAM3 tracking job in an isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path

from infrastructure.shared.inference.generation.link_crop_wrapper.sam3_link_tracker import track_links


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--dinov2-model", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3-bpe", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--padding-fraction", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    track_links(
        video_path=args.video,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
        dinov2_model=args.dinov2_model,
        sam3_checkpoint=args.sam3_checkpoint,
        sam3_bpe=args.sam3_bpe,
        device=args.device,
        minimum_tracked_fraction=0.0,
        padding_fraction=args.padding_fraction,
        reference_spatial_priors=True,
    )


if __name__ == "__main__":
    main()
