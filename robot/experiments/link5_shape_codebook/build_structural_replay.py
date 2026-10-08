"""Build the current two-view Link5 replay; diagnostics stay outside the UI."""
import argparse
from pathlib import Path

from .build_selected_replay import run


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--trial',required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--epoch',type=int,required=True)
    args=parser.parse_args()
    run(args.root,args.destination,args.trial,args.variant,args.epoch)
