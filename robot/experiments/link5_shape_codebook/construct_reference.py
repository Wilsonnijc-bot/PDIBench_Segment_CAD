"""Construct the exact observed union of four newly masked, filtered frame0 clouds."""
import argparse
from pathlib import Path

import numpy as np

from .common import FOUR_REFERENCE_IDS, read, reference_ids, sha, write
from robot.preprocessing.depth.link5_depth_filter import Config
from .prepare import MASK_POLICY, validate_mask_guard


def construct(config):
    output=Path(config['output']);ids=reference_ids(config)
    if ids is None:raise ValueError('explicit four-frame0 reference policy required')
    clouds=[];pixels=[];colors=[];sources=[];records=[]
    for source_index,video in enumerate(ids):
        folder=output/'cases'/video
        status=read(folder/'status.json')
        if status['status']!='complete':raise ValueError('reference preparation incomplete: '+video)
        observation=folder/'observations/frame_00000'
        row=read(observation/'observation.json')
        if row['frame_id']!=0 or row['video_id']!=video or row['observation_policy']!=MASK_POLICY:
            raise ValueError('reference must be a freshly filtered frame0 observation')
        if row['mask_erosion_pixels']!=Config().erosion_pixels or row['mask_mapping_applied']:
            raise ValueError('reference has the wrong depth policy')
        successes=[p for p in (folder/'masking').glob('attempt-*')
                   if (p/'result.json').exists() and read(p/'result.json').get('status')=='complete']
        if len(successes)!=1:raise ValueError('ambiguous fresh successful masking attempt')
        mask_path=Path(read(successes[0]/'result.json')['segmentation'])
        validate_mask_guard(config,successes[0])
        with np.load(mask_path,allow_pickle=False) as a:
            mask=a['object_masks'][0,a['object_names'].tolist().index('link5')].astype(bool)
        path=observation/'input.npz'
        with np.load(path,allow_pickle=False) as a:
            if int(a['frame_id'])!=0 or not np.array_equal(mask,a['source_mask']):
                raise ValueError('frame0 source mask differs from fresh SAM result')
            xyz=a['xyz_camera'].copy();yx=a['pixels_yx'].copy()
            if not np.isfinite(xyz).all() or len(xyz)<Config().minimum_pixels:
                raise ValueError('invalid observed normal cloud')
            if not np.array_equal(xyz[:,2],a['depth'][yx[:,0],yx[:,1]]):
                raise ValueError('normal cloud is not exact observed depth')
            rgb=a['rgb'][yx[:,0],yx[:,1]].copy()
        clouds.append(xyz);pixels.append(yx);colors.append(rgb)
        sources.append(np.full(len(xyz),source_index,np.uint8))
        records.append(dict(video_id=video,frame_id=0,point_count=len(xyz),
            input_path=str(path),input_sha256=sha(path),guard_path=str(successes[0]/'link5_guard.json'),
            guard_sha256=sha(successes[0]/'link5_guard.json'),segmentation_sha256=sha(mask_path)))
    xyz=np.concatenate(clouds);rgb=np.concatenate(colors)
    destination=output/'normal_reference';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'construction.json').exists():
        previous=read(destination/'construction.json')
        if previous['normal_video_ids']!=ids or previous['observations']!=records or previous['point_count']!=len(xyz):
            raise ValueError('constructed reference is immutable; use a fresh output for changed inputs')
        if sha(destination/'pooled_frame0.npz')!=previous['pooled_sha256'] or sha(destination/'pooled_frame0.ply')!=previous['ply_sha256']:
            raise ValueError('constructed reference artifact identity changed')
        print('LINK5_FOUR_FRAME0_CLOUD_RETAINED',len(xyz),flush=True)
        return previous
    np.savez_compressed(destination/'pooled_frame0.npz',xyz_camera=xyz,rgb=rgb,
        source_index=np.concatenate(sources),pixels_yx=np.concatenate(pixels),
        video_ids=np.asarray(ids),frame_ids=np.zeros(4,np.int32))
    with (destination/'pooled_frame0.ply').open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(stream,np.column_stack((xyz,rgb)),fmt=['%.9g','%.9g','%.9g','%d','%d','%d'])
    receipt=dict(status='constructed',normal_video_ids=ids,normal_frame_ids=[0]*4,
        observations=records,point_count=len(xyz),construction='concatenation of exact observed frame0 camera-space points',
        synthetic_surfaces=False,later_frames_in_reference=False,per_cloud_pose_registration=False,
        per_cloud_rescaling=False,depth_policy=MASK_POLICY,
        pooled_sha256=sha(destination/'pooled_frame0.npz'),ply_sha256=sha(destination/'pooled_frame0.ply'),
        detector_training_complete=False)
    write(destination/'construction.json',receipt)
    print('LINK5_FOUR_FRAME0_CLOUD_CONSTRUCTED',len(xyz),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    construct(read(parser.parse_args().config))
