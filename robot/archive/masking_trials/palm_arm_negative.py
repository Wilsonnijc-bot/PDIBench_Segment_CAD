"""Compatibility entry point for automatic arm and wrist negative grounding."""
import argparse
from pathlib import Path
from persistent_masking.gripper_negatives import ROOT, ground

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'models/Qwen3.5-9B')
    parser.add_argument('--work', type=Path, default=ROOT/'negative_work')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.full_context = False
    args.targets = ['arm', 'wrist']
    ground(args)
