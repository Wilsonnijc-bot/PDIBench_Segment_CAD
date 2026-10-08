"""Refresh completed owned observations on CPU while new cases prepare."""
import argparse
from pathlib import Path

from .common import read,sha,write
from .prepare import MASK_POLICY,refresh_owned_observations
from robot.preprocessing.depth.link5_depth_filter import Config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read(args.config);output=Path(config['output'])
    records=[]
    for case in config['cases']:
        folder=output/'cases'/case['id'];status=folder/'status.json'
        if not status.exists():continue
        receipt=read(status)
        # The active original preparation visits each case once. Only completed
        # old-policy cases are touched; pending/running cases belong to it.
        if receipt.get('status')!='complete' or receipt.get('observation_policy')==MASK_POLICY:continue
        if receipt['source_sha256']!=case['video_sha256'] or sha(case['video'])!=case['video_sha256']:
            raise ValueError('owned fresh input identity changed')
        refreshed=refresh_owned_observations(config,case,folder,receipt)
        records.append(dict(video_id=case['id'],refreshed=refreshed,
            reason='same sparse frames; depth reused' if refreshed else 'sparse selection differs; defer fresh native geometry to resume'))
    write(output/'metadata/depth-support-refresh.json',dict(status='complete',CPU_only=True,records=records,
        mask_erosion_pixels=Config().erosion_pixels,mask_mapping_applied=False))
    print('LINK5_UNTRAINED_DEPTH_SUPPORT_CPU_REFRESH_COMPLETE',flush=True)


if __name__=='__main__':main()
