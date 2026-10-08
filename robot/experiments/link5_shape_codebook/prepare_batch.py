"""GPU-isolated native preparation, followed by the frame0 alignment gate."""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, read, write, reference_ids


def run_cases(config,output,cases,devices,workers_per_gpu):
    """Independent per-video processes, with several active workers on each GPU."""
    if type(workers_per_gpu) is not int or workers_per_gpu<1:raise ValueError('positive workers_per_gpu required')
    if not devices or len(set(devices))!=len(devices) or any(d in ('','-1','NoDevFiles') for d in devices):
        raise ValueError('distinct allocated CUDA devices required')
    def video(device,case):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
        path=output/'cases'/case['id'];path.mkdir(parents=True,exist_ok=True)
        attempt=len(list(path.glob('preparation-attempt-*.log')))+1
        log=path/f'preparation-attempt-{attempt:04d}.log'
        with log.open('x') as stream:
            result=subprocess.run([config['environments']['geometry'],'-u','-m',
                'robot.experiments.link5_shape_codebook.prepare','--config',config['_config_path'],'--video-id',case['id']],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        row=dict(video_id=case['id'],device=device,exit_status=result.returncode,log=str(log))
        print('PREPARED',case['id'],result.returncode,flush=True)
        return row
    results=[]
    with ExitStack() as stack:
        pools=[stack.enter_context(ThreadPoolExecutor(max_workers=workers_per_gpu)) for _ in devices]
        futures=[pools[i%len(devices)].submit(video,devices[i%len(devices)],case) for i,case in enumerate(cases)]
        for future in as_completed(futures):
            results.append(future.result())
            write(output/'metadata/preparation-progress.json',dict(gpu_count=len(devices),workers_per_gpu=workers_per_gpu,
                workers=len(devices)*workers_per_gpu,completed=len(results),total=len(cases),results=results))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--reference-only',action='store_true',help='prepare only the four frame0-reference videos; no training')
    parser.add_argument('--workers-per-gpu',type=int,help='concurrent video processes sharing each GPU')
    args=parser.parse_args();config=read(args.config)
    devices=[d.strip() for d in os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')]
    workers_per_gpu=args.workers_per_gpu if args.workers_per_gpu is not None else config.get('preparation',{}).get('workers_per_gpu',1)
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    ids=reference_ids(config)
    cases=config['cases']
    if args.reference_only:
        if ids is None:raise ValueError('reference-only preparation requires explicit frame0 references')
        cases=[next(c for c in cases if c['id']==video) for video in ids]
    elif ids:
        cases=sorted(cases,key=lambda c:ids.index(c['id']) if c['id'] in ids else len(ids))
    write(output/'metadata/preparation-concurrency.json',dict(gpu_count=len(devices),allocated_devices=devices,
        workers_per_gpu=workers_per_gpu,workers=len(devices)*workers_per_gpu,per_video_scratch=True))
    print('LINK5_PREPARATION_CONCURRENCY',len(devices),'GPU',workers_per_gpu,'videos per GPU',flush=True)
    results=run_cases({**config,'_config_path':str(args.config)},output,cases,devices,workers_per_gpu)
    write(output/'metadata'/('reference-preparation-exits.json' if args.reference_only else 'preparation-exits.json'),results)
    if any(r['exit_status'] for r in results):raise RuntimeError('native preparation failures')
    if ids:
        from .construct_reference import construct
        construct(config)
    if args.reference_only:
        print('LINK5_FOUR_FRAME0_REFERENCE_COMPLETE',flush=True)
        return
    from .audit_alignment import audit
    if ids and (output/'splits/normal_reference.json').exists() and read(output/'splits/normal_reference.json').get('reference_only'):
        from .audit_alignment import collect_heldout
        collect_heldout(output,config)
    else:audit(output,config)
    print('LINK5_PREPARATION_COMPLETE',flush=True)


if __name__=='__main__':main()
