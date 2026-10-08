"""One-GPU continuation: freeze four frame0 normals, train, prepare and evaluate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import shutil
import time
from threading import Lock

from .common import MODES, ROOT, mode_folder, read, sha, split_path, write


def evaluation_config(config_path):
    config=read(config_path);output=Path(config['output'])
    receipts={m:read(mode_folder(output,'sanity',m)/'checks.json') for m in MODES}
    if any(r['status']!='passed' for r in receipts.values()):
        raise ValueError('both new detector validations must pass before full-video evaluation')
    workers=config.get('evaluation_execution',{}).get('workers_per_gpu',config.get('preparation',{}).get('workers_per_gpu',5))
    config['full_video']=dict(frame_selection='every_decoded_frame',include_frame0=True,
        primary_video_metric='sum_all_frame_raw_mean_top80',workers=workers,gpu_count=1,workers_per_gpu=workers,
        normalization_sha256=sha(output/'link5_normalization.json'),split_sha256=sha(split_path(output)),
        checkpoint_sha256={m:sha(mode_folder(output,'training',m)/'link5/final.pt') for m in MODES},
        sanity_receipt_sha256={m:sha(mode_folder(output,'sanity',m)/'checks.json') for m in MODES},
        authorization='User requested the refined four-frame0 experiment and every-frame evaluation.')
    path=output/'full_video/config.json';write(path,config)
    return path


def run(config_path,dependency):
    config=read(config_path);output=Path(config['output'])
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or devices[0] in ('','-1','NoDevFiles'):
        raise ValueError('exactly one allocated GPU required')
    if type(config['preparation']['workers_per_gpu']) is not int or config['preparation']['workers_per_gpu']<1:raise ValueError('positive shared-video worker count required')
    training_workers=config.get('training_execution',{}).get('parallel_variants',2)
    if training_workers not in (1,2):raise ValueError('one or two independent training variants required')
    destination=output/'metadata'/f"refined-round-{os.environ['SLURM_JOB_ID']}"
    destination.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='running',gpu_count=1,workers_per_gpu=config['preparation']['workers_per_gpu'],training_workers=training_workers,
                 later_frames_in_reference=False,stages=[])
    progress_lock=Lock()
    def stage(name,command,required=True):
        log=destination/(name+'.log');start=time.time()
        print('LINK5_REFINED_STAGE_STARTED',name,flush=True)
        with log.open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row=dict(stage=name,exit_status=result.returncode,elapsed_seconds=time.time()-start,log=str(log))
        with progress_lock:
            receipt['stages'].append(row);write(destination/'progress.json',receipt)
        print('LINK5_REFINED_STAGE_FINISHED',name,result.returncode,flush=True)
        if required and result.returncode:raise RuntimeError('stage failed: '+name+'; see '+str(log))
        return result.returncode
    module='robot.experiments.link5_shape_codebook.'
    geometry=config['environments']['geometry']
    script=ROOT/'robot/experiments/link5_shape_codebook'
    snapshots=destination/'scripts';snapshots.mkdir(exist_ok=True)
    for name in ('run_detector_job.sh','eris_prepare_job.sh','run_full_video_job.sh'):
        target=snapshots/name
        shutil.copy2(script/name,target);target.chmod(0o444)
    script=snapshots
    detector=lambda kind:['bash',str(script/'run_detector_job.sh'),str(ROOT),str(dependency),str(config_path),kind]
    try:
        stage('preflight',[geometry,'-u','-m',module+'preflight','--config',str(config_path),'--prepare-only'])
        stage('four_reference_alignment',[geometry,'-u','-m',module+'audit_alignment','--config',str(config_path),'--reference-only'])
        stage('fixed_normalization_and_frame0_pose_qc',detector('pose'))
        if config.get('training_execution',{}).get('loader')=='native_equivalent_single_process':
            stage('verify_native_equivalent_loader',[config['environments']['shape_codebook'],'-u','-m',module+'training_loader','--config',str(config_path)])
        with ThreadPoolExecutor(max_workers=training_workers+1) as pool:
            futures=[pool.submit(stage,kind,detector(kind)) for kind in ('train','train_robot_structural')]
            preparation=pool.submit(stage,'prepare_test_videos_shared_workers',['bash',str(script/'eris_prepare_job.sh'),str(ROOT),str(dependency),str(config_path),'prepare'])
            for future in futures:future.result()
            preparation.result()
        failures=[stage(kind,detector(kind),required=False) for kind in ('validate','validate_robot_structural')]
        if any(failures):raise RuntimeError('new detector validation failed; full-video scoring remains disabled')
        full=evaluation_config(config_path)
        stage('full_video_shared_workers',['bash',str(script/'run_full_video_job.sh'),str(ROOT),str(dependency),str(full)])
        if config.get('artifacts', {}).get('point_cloud_replays', False):
            stage('build_full_video_replay',[geometry,'-u','-m',module+'build_full_video_replay','--config',str(full)])
        receipt['status']='complete'
    except Exception as error:
        receipt.update(status='failed',error=str(error));raise
    finally:
        write(destination/'progress.json',receipt)
    print('LINK5_REFINED_ROUND_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dependency',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.dependency)
