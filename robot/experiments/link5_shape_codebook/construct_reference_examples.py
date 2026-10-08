"""Correspondence-preserving deformation replay of the constructed four-frame0 union."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .common import read, sha, write
from .robot_structural import RobotStructuralAugmentation
from .structural_validation import check_example, ply


def run(root):
    root=Path(root);folder=root/'normal_reference';receipt=read(folder/'construction.json')
    if receipt['normal_frame_ids']!=[0]*4 or receipt['later_frames_in_reference']:
        raise ValueError('only the four constructed frame0 clouds may supply examples')
    if sha(folder/'pooled_frame0.npz')!=receipt['pooled_sha256']:raise ValueError('constructed reference changed')
    with np.load(folder/'pooled_frame0.npz',allow_pickle=False) as a:points=a['xyz_camera'].copy()
    # Robust axis fit chooses deformation parameters; it does not crop the cloud.
    low,high=np.quantile(points,[.01,.99],axis=0)
    fit=points[((points>=low)&(points<=high)).all(1)]
    center=np.median(fit,axis=0);values,vectors=np.linalg.eigh(np.cov((fit-center).T))
    axis=vectors[:,-1]
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    start,end=np.quantile((fit-center)@axis,[.01,.99])
    geometry=dict(axis=axis.tolist(),anchor=(center+start*axis).tolist(),length=float(end-start),
        normal_reference_count=4,coordinate_system='original frame0 camera',
        source='PCA of the four-frame0 observed union; robust longitudinal endpoints for deformation parameters only',
        pooled_sha256=receipt['pooled_sha256'])
    if values[-1]/max(values[-2],1e-12)<1.5:raise ValueError('constructed cloud has no clear longitudinal axis')
    destination=folder/'examples';destination.mkdir(parents=True,exist_ok=True)
    np.save(destination/'normal.npy',points,allow_pickle=False);ply(destination/'normal.ply',points)
    normal=torch.from_numpy(points);generator=RobotStructuralAugmentation(geometry);checks=[]
    for name,parameters in [('shortened',dict(alpha=.8,affected_fraction=.8)),('lengthened',dict(alpha=1.2,affected_fraction=.8)),
                           ('mild_bend',dict(angle_degrees=10,start_fraction=.2,direction_angle=.5)),
                           ('strong_bend',dict(angle_degrees=35,start_fraction=.2,direction_angle=.5))]:
        result=generator.axial(normal,**parameters) if 'alpha' in parameters else generator.bend(normal,**parameters)
        check=check_example(normal,result,dict(generator.last_parameters),geometry)
        if not check['passed']:raise ValueError('invalid constructed deformation example: '+name)
        checks.append(dict(example=name,**check))
        np.savez_compressed(destination/(name+'.npz'),normal=points,anomalous=result[0].numpy(),
            gt_offset=result[1].numpy(),gt_mask=result[2].numpy())
        ply(destination/(name+'.ply'),result[0].numpy())
    write(folder/'structural_examples.json',dict(status='passed',checks=checks,geometry=geometry,
        source_video_ids=receipt['normal_video_ids'],source_frame_ids=[0]*4,pooled_sha256=receipt['pooled_sha256'],
        generated_missing_surfaces=False,training_samples=False,
        purpose='deformation replay of the exact pooled four-frame0 normal cloud; native training uses four separate observations'))
    print('LINK5_FOUR_FRAME0_DEFORMATION_EXAMPLES_READY',len(points),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
