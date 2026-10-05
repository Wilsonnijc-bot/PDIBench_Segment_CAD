"""One isolated coordinator stage, executed in its declared model environment."""
import argparse
import json
import os
from pathlib import Path
import sys

from infrastructure.deformation_detect.coordinator import read, write


def configure(request):
    config=request['config'];resources=config['resources']
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
    if config.get('secrets_file'):load_env_file(Path(config['secrets_file']))
    os.environ.update(PDI_SAM3_CHECKPOINT=resources['sam3_checkpoint'],PDI_SAM3_BPE=resources['sam3_bpe'],
        PDI_DINO_DIRECTORY=resources['dino_directory'],PDI_PALM_REFERENCES=resources['palm_references'],
        PDI_PMASK_QWEN_MODEL=resources['qwen_model'],PDI_PMASK_QWEN_PYTHON=config['environments']['vlm1'],
        FFMPEG_BINARY=resources['ffmpeg'],PDI_VLM_TRANSPORT_ATTEMPTS='1')
    os.environ['PATH']=str(Path(resources['ffmpeg']).parent)+os.pathsep+os.environ.get('PATH','')
    overrides={k:v for k,v in config['vlm'].items() if k!='object_models'}
    attempt=request.get('attempt',1)
    if attempt>1:
        if overrides.get('vlm1_fallback'):overrides['vlm1']=overrides['vlm1_fallback']
        # Existing Gemini -> Luna alternate route, explicit rather than nested retries.
        from robot.preprocessing.link7_persistent.interface.config import role_config
        overrides['vlm2']={**role_config('vlm2_malformed_fallback'),**overrides.get('vlm2_malformed_fallback',{})}
    os.environ['PDI_VLM_ROLE_OVERRIDES']=json.dumps(overrides)
    if request.get('case'):
        os.environ['PDI_CASE_VIDEOS']=json.dumps({request['case']['id']:request['case']['video']})


def check_environment(config, name):
    import importlib
    import torch  # Load libtorch before extension modules such as droid_backends.
    modules={'sam3':['sam3','robot.preprocessing.link7_persistent.pipeline','object.preprocessing.segmentation.segment'],
             'vlm1':['robot.preprocessing.link7_persistent.pipeline','infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm'],
             'geometry':['robot.workflows.score_v1','cotracker','droid_backends','lietorch'],
             'anomaly':['object.scoring.anomalydino','faiss']}
    for module in modules[name]:importlib.import_module(module)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    assert (torch.ones(2,device='cuda')+1).sum().item()==4
    if name=='sam3':
        if 'link2' in config.get('robot_links', []):
            from robot.preprocessing.segmentation.sam3_dinov2_segment import _active_franka_groups, _validate_franka_groups
            from infrastructure.shared.inference.dinov2_reference_boxes import discover_reference_groups
            _validate_franka_groups(_active_franka_groups(discover_reference_groups(Path(config['resources']['robot_references']))))
        from robot.preprocessing.link7_persistent.interface.config import role_config
        keys={'VLM2_API_KEY'}
        for role in ('vlm1','vlm2','vlm2_malformed_fallback'):
            c=role_config(role)
            if c['backend']=='cloud_api':keys.add(c['api_key_env'])
        for c in config['vlm'].values():
            if isinstance(c,dict) and c.get('backend')=='cloud_api':keys.add(c['api_key_env'])
        if any(not os.environ.get(k) for k in keys):raise RuntimeError('Missing required VLM credential environment variable')
        import cv2
        for case in config['cases']:
            cap=cv2.VideoCapture(case['video']);ok,_=cap.read();frames=cap.get(cv2.CAP_PROP_FRAME_COUNT);cap.release()
            if not ok or frames<2:raise ValueError('Unreadable or too-short video: '+case['id'])
    if name=='geometry':
        for relative in ('checkpoints/megasam_final.pth','Depth-Anything/checkpoints/depth_anything_vitl14.pth','cvd_opt/raft-things.pth'):
            if not (Path(config['resources']['mega_sam'])/relative).is_file():raise FileNotFoundError(relative)
    import importlib.metadata
    versions={}
    for package in ('numpy','opencv-python','Pillow','torchvision','transformers','faiss-cpu'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    print(json.dumps({'status':'passed','environment':name,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'packages':versions}))


def isolate_mega(request):
    source=Path(request['config']['resources']['mega_sam']);target=Path(request['directory'])/'mega_sam'
    target.mkdir()
    for path in source.iterdir():
        dest=target/path.name
        if path.name in {'work_space','outputs','outputs_cvd','cache_flow','reconstructions'}:dest.mkdir()
        else:dest.symlink_to(path.resolve(),target_is_directory=path.is_dir())
    for name in ('work_space','outputs','outputs_cvd','cache_flow','reconstructions'):(target/name).mkdir(exist_ok=True)
    os.environ['PDI_MEGA_SAM_ROOT']=str(target)
    return target


def run(request):
    configure(request)
    stage=request['stage']
    if stage in {'link2_masks','link7_initial','vlm1','vlm2','link7_masks','mask_join','robot_score'}:
        from robot.workflows.coordination import run_stage
    else:
        from object.workflows.coordination import run_stage
    result=run_stage(request)
    write(Path(request['directory'])/'result.json',{'status':'complete',**result})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request',type=Path);p.add_argument('--check-environment')
    args=p.parse_args();request=read(args.request)
    try:
        if args.check_environment:
            configure(request);check_environment(request['config'],args.check_environment)
        else:run(request)
        return 0
    except Exception as exc:
        # Keep evidence but redact credentials even if a vendor exception embeds them.
        message=str(exc)
        for name,value in os.environ.items():
            if value and any(token in name.upper() for token in ('API_KEY','TOKEN','PASSWORD','SECRET')):message=message.replace(value,'[REDACTED]')
        if not args.check_environment:write(args.request.parent/'result.json',{'status':'failed','error_type':type(exc).__name__,'error':message})
        print(type(exc).__name__+': '+message,file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())
