"""Fail closed on input identity, runtime, checkpoint, and method requirements."""
import argparse
import os
from pathlib import Path
import subprocess

from .common import ROOT, read, sha, write, reference_ids


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();cfg=read(args.config)
    import torch
    from robot.preprocessing.depth.link5_depth_filter import Config, filter_depth
    Config().validate()
    reference_ids(cfg)
    if not torch.cuda.is_available() or (torch.ones(2,device='cuda')+1).sum().item()!=4:raise RuntimeError('CUDA preflight failed')
    if len(cfg['cases'])!=45 or len({c['id'] for c in cfg['cases']})!=45:raise ValueError('exact45 original cases required')
    for case in cfg['cases']:
        if sha(case['video'])!=case['video_sha256']:raise ValueError('frozen video hash mismatch: '+case['id'])
    for key in ('sam3_checkpoint','sam3_bpe','link5_guard_reference','robot_references','dino_directory','mega_sam'):
        if not Path(cfg['resources'][key]).exists():raise FileNotFoundError(cfg['resources'][key])
    from robot.preprocessing.link5_refinement.link5_point_guard import POSITIVE_REFERENCE_FRAME
    positive_reference=Path(cfg['resources'].get('link5_positive_guard_reference',POSITIVE_REFERENCE_FRAME))
    if not positive_reference.is_file():raise FileNotFoundError(positive_reference)
    from infrastructure.deformation_detect.worker import configure,check_environment
    request=Path(cfg['output'])/'metadata/preflight-request.json';write(request,dict(config=cfg))
    for name in ('sam3','geometry'):
        subprocess.run([cfg['environments'][name],'-m','infrastructure.deformation_detect.worker',str(request),
                        '--check-environment',name],cwd=ROOT,check=True)
    if not args.prepare_only:
        from .upstream import build_model
        effective=read(Path(cfg['output'])/'training/effective_config.json')
        build_model(effective)
        for name in ('2023-10-28-18-33-37','2024-01-11-20-02-45'):
            for file in ('model_best.pth','config.yml'):
                if not (ROOT/'infrastructure/vendor/FoundationPose/weights'/name/file).is_file():raise FileNotFoundError(name+'/'+file)
    receipt=dict(status='passed',scope='fresh preparation only' if args.prepare_only else 'all',
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,configuration_sha256=sha(args.config),
        filter_module=filter_depth.__module__,mask_erosion_pixels=Config().erosion_pixels,mask_mapping_applied=False,historical_inputs_used=False)
    write(Path(cfg['output'])/'metadata/preflight.json',receipt)
    print('LINK5_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
