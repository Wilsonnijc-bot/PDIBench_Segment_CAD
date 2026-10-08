"""A single allocated GPU must admit five distinct videos simultaneously."""
from threading import Barrier, Lock
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import time
import pytest

from robot.experiments.link5_shape_codebook.prepare_batch import run_cases


@pytest.mark.parametrize('workers',[5,8])
def test_video_processes_share_one_gpu_concurrently(tmp_path,monkeypatch,workers):
    barrier=Barrier(workers,timeout=5);lock=Lock();active=0;peak=0;seen=[]
    def execute(command,**kwargs):
        nonlocal active,peak
        with lock:
            active+=1;peak=max(peak,active);seen.append((command[-1],kwargs['env']['CUDA_VISIBLE_DEVICES'],kwargs['stdout'].name))
        barrier.wait()
        with lock:active-=1
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('robot.experiments.link5_shape_codebook.prepare_batch.subprocess.run',execute)
    cfg=dict(environments=dict(geometry='test-python'),_config_path='test-config.json')
    cases=[dict(id=f'video{i}') for i in range(workers)]
    rows=run_cases(cfg,tmp_path,cases,['3'],workers)
    assert peak==workers
    assert {r['device'] for r in rows}=={'3'}
    assert len({name for _,_,name in seen})==workers
    assert {video for video,_,_ in seen}=={c['id'] for c in cases}


def test_expansion_and_queued_worker_cannot_reconstruct_same_case_twice(tmp_path,monkeypatch):
    from robot.experiments.link5_shape_codebook import prepare
    work=[];active=0;peak=0
    def native(config,case,output):
        nonlocal active,peak
        active+=1;peak=max(peak,active)
        done=output/'cases'/case['id']/'fixture_complete'
        if not done.exists():
            time.sleep(.1);work.append(case['id']);done.write_text('complete')
        active-=1
    monkeypatch.setattr(prepare,'prepare_case',native)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(prepare.prepare_locked,{},dict(id='same-video'),tmp_path) for _ in range(2)]
        for future in futures:future.result()
    assert peak==1 and work==['same-video']
