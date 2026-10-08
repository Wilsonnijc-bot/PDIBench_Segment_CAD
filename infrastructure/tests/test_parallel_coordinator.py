"""Real per-video processes, bounded GPU admission, isolation and resume."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from infrastructure.deformation_detect.coordinator import coordinate, read
from infrastructure.deformation_detect.layout import environment
from infrastructure.deformation_detect.runtime_isolation import validate_origin


def fixture_config(tmp_path, slots=2):
    worker = tmp_path/'fake-python'
    worker.write_text(f'''#!{sys.executable}
import json, os, pathlib, sys, time
r=json.loads(pathlib.Path(sys.argv[-1]).read_text())
folder=pathlib.Path(r['directory'])
start=time.monotonic()
time.sleep(0.03)
data={{'status':'complete','case':r['case']['id'],'stage':r['stage'],'start':start,'end':time.monotonic(),'device':os.environ.get('CUDA_VISIBLE_DEVICES')}}
(folder/'result.json').write_text(json.dumps(data))
''')
    worker.chmod(0o700)
    video=tmp_path/'video.mp4';video.write_text('source')
    prompt=tmp_path/'prompt.txt';prompt.write_text('cube')
    return {'schema_version':1,'run_id':'parallel-test','output':str(tmp_path/'run'),
        'execution':{'workers':3,'gpu_slots':slots},'environments':dict.fromkeys(('sam3','vlm1','geometry','anomaly'),str(worker)),
        'resources':{},'vlm':{},'policy':{'vlm_attempts':4,'vlm_stage_timeout_seconds':60,'stage_timeout_seconds':60},
        'cases':[{'id':name,'video':str(video),'prompt_file':str(prompt)} for name in ('one','two','three')]}


@pytest.mark.parametrize('slots',[1,2])
def test_real_children_respect_slots_and_resume(tmp_path, slots):
    config=fixture_config(tmp_path,slots)
    state=coordinate(config,check=lambda _: {},source_identity={'code':'fixed'})
    assert state['status']=='complete'
    events=[]
    for case in state['cases'].values():
        for stage in case['stages'].values():
            result=read(Path(stage['directory'])/'result.json')
            events += [(result['start'],1),(result['end'],-1)]
    count=maximum=0
    for _,change in sorted(events):
        count+=change;maximum=max(maximum,count)
    assert maximum==slots
    before={str(p):p.read_bytes() for p in Path(config['output']).rglob('result.json')}
    config['execution']['workers']=1
    assert coordinate(config,check=lambda _: {},source_identity={'code':'fixed'})['status']=='complete'
    after={str(p):p.read_bytes() for p in Path(config['output']).rglob('result.json')}
    assert before==after
    # One case's corrupted stage regenerates only its downstream dependency branch.
    damaged=Path(state['cases']['two']['stages']['object_tracks']['directory'])/'result.json'
    damaged.write_text('{}')
    assert coordinate(config,check=lambda _: {},source_identity={'code':'fixed'})['status']=='complete'
    new=[p for p in Path(config['output']).rglob('result.json') if str(p) not in before]
    assert len(new)==3 and all('/two/' in str(p) for p in new)


def test_gpu_lock_is_released_after_process_is_killed(tmp_path):
    code='from infrastructure.deformation_detect.parallel import gpu_admission; import sys,time\nwith gpu_admission(sys.argv[1],1):\n print("held",flush=True);time.sleep(30)'
    first=subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],stdout=subprocess.PIPE,text=True,env=environment())
    assert first.stdout.readline().strip()=='held'
    first.kill();first.wait()
    second=subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],stdout=subprocess.PIPE,text=True,env=environment())
    try:
        assert second.stdout.readline().strip()=='held'
    finally:
        second.terminate();second.wait()


def test_archive_and_foreign_source_rejected_before_execution(tmp_path):
    archive=tmp_path/'infrastructure/archive/legacy.py'
    archive.parent.mkdir(parents=True);archive.write_text('raise AssertionError("legacy executed")')
    code='import sys;sys.path.insert(0,sys.argv[1]);import legacy'
    result=subprocess.run([sys.executable,'-c',code,str(archive.parent)],env=environment(),capture_output=True,text=True)
    assert result.returncode!=0 and 'Legacy source import rejected' in result.stderr
    assert 'legacy executed' not in result.stderr
    with pytest.raises(ImportError,match='Foreign workspace'):
        validate_origin('robot.fake',str(tmp_path/'robot/fake.py'))


def test_import_receipt_includes_nested_process_origins(tmp_path):
    env=environment();env['PDI_IMPORT_ORIGINS_DIR']=str(tmp_path/'origins')
    subprocess.run([sys.executable,'-c','import robot.workflows.coordination'],env=env,check=True)
    records=list((tmp_path/'origins').glob('*.json'))
    assert len(records)==1
    record=read(records[0]);assert record['status']=='passed'
    assert '/infrastructure/archive/' not in json.dumps(record['origins'])
    assert 'robot.workflows.coordination' in record['origins']


def test_parent_sigterm_cleans_active_stage_and_waiting_case(tmp_path):
    config=fixture_config(tmp_path,slots=1)
    worker=Path(config['environments']['sam3'])
    worker.write_text(worker.read_text().replace('time.sleep(0.03)','time.sleep(30)'))
    manifest=tmp_path/'config.json';manifest.write_text(json.dumps(config))
    code='import json,sys;from infrastructure.deformation_detect.coordinator import coordinate;coordinate(json.load(open(sys.argv[1])),check=lambda _: {},source_identity={"code":"fixed"})'
    process=subprocess.Popen([sys.executable,'-c',code,str(manifest)],env=environment(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            receipts=list((tmp_path/'run/cases').rglob('execution.json'))
            if receipts:
                break
            time.sleep(.05)
        assert receipts
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=15)
        assert read(tmp_path/'run/run.json')['status']=='interrupted'
        for receipt in (tmp_path/'run').rglob('execution.json'):
            record=read(receipt)
            assert record.get('finished_at')
            with pytest.raises(ProcessLookupError):os.kill(record['pid'],0)
    finally:
        if process.poll() is None:
            process.kill();process.wait()


def test_per_video_gpu_assignment_is_distinct_and_preserves_allocation(tmp_path, monkeypatch):
    config=fixture_config(tmp_path,slots=3)
    config['execution']['gpu_assignment']='per_video'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b,GPU-c')
    state=coordinate(config,check=lambda _: {},source_identity={'code':'fixed'})
    assert state['status']=='complete'
    receipts=[read(p) for p in (tmp_path/'run/coordination').glob('*/execution.json')]
    assert {r['cuda_visible_devices'] for r in receipts}=={'GPU-a','GPU-b','GPU-c'}
    for name, case in state['cases'].items():
        receipt=read(tmp_path/'run/coordination'/name/'execution.json')
        assert all(read(Path(stage['directory'])/'result.json')['device']==receipt['cuda_visible_devices'] for stage in case['stages'].values())
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-a,GPU-b,GPU-c'


def test_per_video_fails_before_launch_when_allocation_is_insufficient(tmp_path, monkeypatch):
    config=fixture_config(tmp_path,slots=3);config['execution']['gpu_assignment']='per_video'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(ValueError,match='distinct allocated'):
        coordinate(config,check=lambda _: {},source_identity={'code':'fixed'})
    assert not Path(config['output']).exists()


def test_parallel_timeout_retries_only_after_first_pass_finishes(tmp_path,monkeypatch):
    config=fixture_config(tmp_path,slots=2)
    config['execution']['workers']=2
    config['execution']['gpu_assignment']='per_video'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b')
    worker=Path(config['environments']['sam3'])
    worker.write_text(worker.read_text().replace("(folder/'result.json').write_text(json.dumps(data))", """
if r['case']['id']=='one' and r['stage']=='vlm2' and r['attempt']==1:
    data.update(status='failed',failure={'category':'vlm_timeout','defer':True})
(folder/'result.json').write_text(json.dumps(data))
sys.exit(1 if data['status']=='failed' else 0)
"""))
    state=coordinate(config,check=lambda _: {},source_identity={'code':'fixed'})
    assert state['status']=='complete' and state['deferred_cases']==[]
    final=read(Path(state['cases']['one']['stages']['vlm2']['directory'])/'result.json')
    assert final['start']>max(read(Path(state['cases'][name]['stages']['object_anomaly']['directory'])/'result.json')['end'] for name in ('two','three'))
    requests=[read(p) for p in (tmp_path/'run/cases/one').rglob('vlm2/attempt-*/request.json')]
    assert sorted(r['attempt'] for r in requests)==[1,2]
    assert {r['vlm_route'] for r in requests}=={'primary'}
    exits=sorted((tmp_path/'run/coordination/one').glob('*.exit'))
    assert [p.read_text().strip() for p in exits]==['3','0']


def test_parallel_timeout_exhaustion_does_not_loop_forever(tmp_path):
    config=fixture_config(tmp_path,slots=2)
    worker=Path(config['environments']['sam3'])
    worker.write_text(worker.read_text().replace("(folder/'result.json').write_text(json.dumps(data))", """
if r['case']['id']=='one' and r['stage']=='vlm2':
    data.update(status='failed',failure={'category':'vlm_timeout','defer':True})
(folder/'result.json').write_text(json.dumps(data))
sys.exit(1 if data['status']=='failed' else 0)
"""))
    state=coordinate(config,check=lambda _: {},source_identity={'code':'fixed'})
    assert state['cases']['one']['status']=='disabled'
    assert all(state['cases'][name]['status']=='complete' for name in ('two','three'))
    assert len(list((tmp_path/'run/coordination/one').glob('*.exit')))==4
    assert len(list((tmp_path/'run/cases/one').rglob('vlm2/attempt-*/result.json')))==4
