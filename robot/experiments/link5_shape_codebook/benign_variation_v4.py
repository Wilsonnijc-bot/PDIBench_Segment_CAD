"""Paired benign reconstruction/pose variability for structural supervision.

Stored observations and the four normal codebook inputs are never changed.
Apply one identical similarity transform and point-noise draw to a clean cloud
and its structural synthetic counterpart. Only their known structural difference
is anomalous. This teaches tolerance rather than normalizing inference clouds.
"""
import math

import torch

DEFAULTS = dict(identity_probability=.25, isotropic_scale=[.75,1.30], rotation_degrees=5.,
                translation_normalized=.04, point_noise_std_normalized=[0.,.006])


def paired_variation(normal, anomalous, mask, seed, settings=None):
    settings={**DEFAULTS,**(settings or {})}
    devices=[normal.device.index] if normal.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if normal.is_cuda:torch.cuda.manual_seed_all(seed)
        if torch.rand(()).item()<settings['identity_probability']:
            return normal,anomalous,normal-anomalous,mask,dict(identity=True)
        def uniform(low,high):return low+(high-low)*torch.rand((),device=normal.device,dtype=normal.dtype)
        scale=uniform(*settings['isotropic_scale'])
        direction=torch.randn(3,device=normal.device,dtype=normal.dtype);direction/=direction.norm()
        angle=uniform(-settings['rotation_degrees'],settings['rotation_degrees'])*math.pi/180
        x,y,z=direction
        skew=torch.stack([x*0,-z,y,z,y*0,-x,-y,x,z*0]).reshape(3,3)
        rotation=torch.eye(3,device=normal.device,dtype=normal.dtype)+angle.sin()*skew+(1-angle.cos())*(skew@skew)
        translation=(torch.rand(3,device=normal.device,dtype=normal.dtype)*2-1)*settings['translation_normalized']
        noise_std=uniform(*settings['point_noise_std_normalized'])
        noise=torch.randn_like(normal)*noise_std
        clean=scale*(normal@rotation.T)+translation+noise
        deformed=scale*(anomalous@rotation.T)+translation+noise
        # Preserve exact identity outside the deliberately deformed region.
        deformed=torch.where(mask.bool()[:,None],deformed,clean)
        offset=clean-deformed
        # Separate float32 matrix products and a subtraction accumulate roundoff
        # proportional to coordinate magnitude. Targets above remain the exact
        # observed clean-minus-deformed difference, regardless of this audit.
        roundoff=8*torch.finfo(normal.dtype).eps*max(1.,float(clean.abs().max()),float(deformed.abs().max()))
        torch.testing.assert_close(offset,scale*((normal-anomalous)@rotation.T),atol=roundoff,rtol=1e-5)
        mask=offset.ne(0).any(dim=-1).to(mask.dtype)
        return clean,deformed,offset,mask,dict(identity=False,isotropic_scale=float(scale),
            rotation_degrees=float(angle*180/math.pi),rotation=rotation.cpu().tolist(),translation=translation.cpu().tolist(),
            point_noise_std_normalized=float(noise_std),seed=seed,coordinate_center='fixed shared normalization origin; no per-cloud centering')
