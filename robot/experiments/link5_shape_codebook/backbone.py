"""Sparse/dense interface to an actual pretrained author-compatible MinkUNet.

This implements only voxel input/output adaptation, never a replacement encoder.
Checkpoint loading is strict. Random weights and PointNet fallback are forbidden.
"""
import importlib
from pathlib import Path
import sys
import subprocess

import torch
from torch import nn

from .common import sha


class MinkUNetAdapter(nn.Module):
    def __init__(self,config):
        super().__init__()
        import MinkowskiEngine as ME
        self.ME=ME
        self.config=config
        actual_revision=subprocess.check_output(['git','-C',config['source_root'],'rev-parse','HEAD'],text=True).strip()
        if actual_revision!=config['source_revision']:raise ValueError('pretrained backbone source revision mismatch')
        changed=subprocess.check_output(['git','-C',config['source_root'],'diff','--name-only','HEAD'],text=True).strip()
        if changed:raise ValueError('official pretrained backbone source was modified: '+changed)
        checkpoint=Path(config['checkpoint'])
        if not checkpoint.is_file():
            raise FileNotFoundError('Author-compatible pretrained MinkUNet checkpoint is required: '+str(checkpoint))
        if sha(checkpoint)!=config['checkpoint_sha256']:
            raise ValueError('pretrained backbone checkpoint hash mismatch')
        sys.path.insert(0,str(Path(config['source_root']).resolve()))
        module=importlib.import_module(config['module'])
        if Path(config['source_root']).resolve() not in Path(module.__file__).resolve().parents:
            raise ValueError('backbone import origin differs from configured source')
        factory=getattr(module,config['class_name'])
        self.provenance=dict(class_name=config['class_name'],source_revision=actual_revision,
            source_sha256=sha(module.__file__),checkpoint_sha256=sha(checkpoint),checkpoint_url=config['checkpoint_url'])
        self.encoder=factory(**config['kwargs'])
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=payload[config['state_key']] if config.get('state_key') else payload
        self.encoder.load_state_dict(state,strict=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.voxel_size=float(config['voxel_size'])
        if self.voxel_size<=0:raise ValueError('invalid fixed voxel size')

    def train(self,mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self,points):
        self.encoder.eval()
        batch=[]
        for xyz in points:
            coords=torch.floor(xyz/self.voxel_size).to(torch.int32)
            unique,inverse=torch.unique(coords,dim=0,return_inverse=True)
            batched=self.ME.utils.batched_coordinates([unique.cpu()],dtype=torch.int32)
            features=torch.ones((len(unique),self.config['kwargs']['in_channels']),device=xyz.device)
            sparse=self.ME.SparseTensor(features=features,coordinates=batched,device=xyz.device)
            encoded=self.encoder(sparse)
            # Coordinate manager may reorder unique voxels. Query features at
            # original coordinates instead of assuming output row correspondence.
            query=self.ME.utils.batched_coordinates([coords.cpu()],dtype=torch.float32).to(xyz.device)
            dense=encoded.features_at_coordinates(query)
            if dense.shape!=(len(xyz),32) or not torch.isfinite(dense).all():
                raise ValueError('actual pretrained backbone must return finite 32-D point features')
            batch.append(dense)
        return torch.stack(batch)
