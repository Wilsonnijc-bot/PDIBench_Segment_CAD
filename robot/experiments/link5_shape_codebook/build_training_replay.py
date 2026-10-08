"""Refresh the selected normal/test replay without restoring historical tabs."""
import argparse
from pathlib import Path

from .build_selected_replay import run as selected_replay
from .common import read


def run(root,destination):
    destination=Path(destination)
    receipt=destination/'provenance.json'
    if not receipt.exists():
        raise ValueError('Select a checkpoint with build_structural_replay --trial --variant --epoch first.')
    selected=read(receipt)
    return selected_replay(root,destination,selected['trial'],selected['variant'],selected['epoch'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.destination)
