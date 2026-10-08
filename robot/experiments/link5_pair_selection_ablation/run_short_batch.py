"""Use a validated five-video capacity to finish short clips before expiry."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .prepare_gpu import write


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','cache','sources','queries','output','scratch','mega-root','tracker-checkpoint'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=5);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.workers!=5:raise ValueError('Five total video workers required')
    common=[]
    for name in ('manifest','cache','sources','queries','output','scratch','mega_root','tracker_checkpoint'):
        common+=['--'+name.replace('_','-'),str(getattr(a,name))]
    subprocess.run([sys.executable,'-m','robot.experiments.link5_pair_selection_ablation.run_gpu_batch',*common,'--workers','5','--preflight-only'],check=True)
    meta=a.output.parent/'metadata'
    proof=json.loads((meta/'capacity_passed.json').read_text())
    if proof['status']!='passed' or proof['concurrent_video_workers']!=5 or proof['gpu_count']!=1:
        raise RuntimeError('Measured capacity proof required')
    if a.preflight_only:return
    entries=sorted(json.loads(a.manifest.read_text())['entries'],key=lambda e:e['frame_count'])
    stop=threading.Event();lock=threading.RLock();active={};results=[]
    def cancel(signum,frame):
        stop.set()
        with lock:
            for child in active.values():
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def video(entry):
        case=entry['video_id']
        if stop.is_set():return dict(video_id=case,exit_status=130,status='cancelled_before_launch')
        log=meta/'video_logs'/f'{case}-{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
        t0=time.monotonic()
        with log.open('x') as stream:
            with lock:
                if stop.is_set():return dict(video_id=case,exit_status=130,status='cancelled_before_launch')
                child=subprocess.Popen([sys.executable,'-u','-m','robot.experiments.link5_pair_selection_ablation.prepare_gpu',*common,'--video-id',case],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[case]=child
            while child.poll() is None:
                if stop.wait(1):
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
            rc=child.wait()
        with lock:active.pop(case,None)
        print('VIDEO_EXIT',case,rc,flush=True)
        return dict(video_id=case,exit_status=rc,seconds=time.monotonic()-t0,log=str(log))
    with ThreadPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(video,e) for e in entries]):
            results.append(f.result());write(meta/'short-progress.json',dict(workers=5,gpu_count=1,total=len(entries),results=results))
    write(meta/'short-exits.json',results)
    if stop.is_set():raise SystemExit(130)
    if any(r['exit_status'] for r in results):raise SystemExit(1)
    print('LINK5_SHORT_VIDEO_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()
