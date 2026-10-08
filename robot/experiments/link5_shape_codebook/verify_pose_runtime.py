"""Exercise native FoundationPose CUDA depth kernels and original-CAD renderer."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from .common import ROOT,read,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read(args.config)
    sys.path.insert(0,str(ROOT/'infrastructure/vendor/FoundationPose'))
    from Utils import bilateral_filter_depth,erode_depth,compute_crop_window_tf_batch
    from .pose import Link5Pose
    depth=torch.ones((32,32),dtype=torch.float32,device='cuda')
    # This is the native pose engine's DEPTH preprocessing, never mask erosion
    # or a preprocessing step for saved/scored Link5 observations.
    for operation in (erode_depth,bilateral_filter_depth):
        result=operation(depth)
        if result.shape!=depth.shape or not torch.isfinite(torch.as_tensor(result)).all():
            raise ValueError('native FoundationPose depth kernel failed: '+operation.__name__)
    adapter=Link5Pose({**config['pose'],'debug_directory':str(Path(config['output'])/'metadata/foundationpose-debug')})
    pose=torch.eye(4,device='cuda')[None];pose[:,2,3]=1.0
    k=np.array([[256,0,64],[0,256,64],[0,0,1]],np.float32)
    crops=compute_crop_window_tf_batch(pts=adapter.mesh.vertices,H=128,W=128,poses=pose,K=k,
        crop_ratio=1.2,out_size=(160,160),method='box_3d',mesh_diameter=adapter.estimator.diameter)
    if crops.dtype!=torch.float32 or not torch.isfinite(crops).all():
        raise ValueError('native FoundationPose float32 crop pipeline failed')
    _,rendered,_=adapter.render(K=k,H=128,W=128,ob_in_cams=pose,
        mesh=adapter.mesh,glctx=adapter.estimator.glctx,extra={})
    if not torch.isfinite(rendered).all() or not (rendered>0).any():
        raise ValueError('native FoundationPose CAD renderer failed')
    write(Path(config['output'])/'metadata/foundationpose-runtime-preflight.json',dict(status='passed',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,native_depth_kernels=True,
        mask_erosion_applied=False,original_CAD_rendered=True,native_float32_crops=True,provenance=adapter.provenance))
    print('LINK5_FOUNDATIONPOSE_NATIVE_RUNTIME_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
