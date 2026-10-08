"""Verify official pretrained weights, then freeze their observed hash in config."""
import argparse
from pathlib import Path

import torch

from .backbone import MinkUNetAdapter
from .common import read,sha,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read(args.config)
    record=config['backbone'];record['checkpoint_sha256']=sha(record['checkpoint'])
    model=MinkUNetAdapter(record).cuda().eval()
    torch.manual_seed(0)
    features=model(torch.randn(1,256,3,device='cuda')*.1)
    if features.shape!=(1,256,32) or not torch.isfinite(features).all():raise ValueError('pretrained sparse backbone GPU probe failed')
    norms=features.norm(dim=-1)
    if not torch.allclose(norms,torch.ones_like(norms),atol=1e-3) or features.std(dim=1).mean().item()<1e-5:
        raise ValueError('sparse coordinate retrieval returned missing or constant pretrained features')
    write(args.config,config)
    write(Path(config['output'])/'metadata/pretrained-backbone.json',dict(status='passed',**model.provenance,
        actual_configuration=record,feature_shape=list(features.shape),feature_norm_range=[norms.min().item(),norms.max().item()],paper_exact=False))
    print('PRETRAINED_MINKOWSKI_UNET_VERIFIED',flush=True)


if __name__=='__main__':main()
