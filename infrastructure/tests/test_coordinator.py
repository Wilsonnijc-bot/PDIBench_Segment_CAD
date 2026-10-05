"""Exercise the real scheduler without invoking GPU or paid VLM services."""
import json
from pathlib import Path
import sys

import pytest

from infrastructure.deformation_detect.coordinator import STAGES, coordinate, load_manifest, locked, read, write


@pytest.fixture
def config(tmp_path):
    video=tmp_path/'video.mp4';video.write_bytes(b'video')
    prompt=tmp_path/'prompt.txt';prompt.write_text('blue cube')
    return {'schema_version':1,'run_id':'test','output':str(tmp_path/'run'),
        'environments':dict.fromkeys(('sam3','vlm1','geometry','anomaly'),sys.executable),
        'resources':{},'vlm':{},'policy':{'vlm_attempts':4,'vlm_stage_timeout_seconds':60,'stage_timeout_seconds':60},
        'cases':[{'id':'one','video':str(video),'prompt_file':str(prompt)}]}


def run(config, executor, **kwargs):
    return coordinate(config,executor=executor,check=lambda _: {},source_identity={'code':'fixed'},**kwargs)


class Executor:
    def __init__(self):self.calls=[];self.fail=set();self.interrupt=None
    def __call__(self,path,python,timeout):
        request=read(path);case=request['case']['id'];stage=request['stage'];self.calls.append((case,stage,request['attempt']))
        assert all((Path(p)/'result.json').exists() for p in request['dependencies'].values())
        if stage==self.interrupt:raise KeyboardInterrupt()
        if (case,stage) in self.fail:
            write(path.parent/'result.json',{'status':'failed','error':'invalid VLM response'});return 1
        # Include dependency identity to represent changed outputs cascading.
        write(path.parent/'result.json',{'status':'complete','stage':stage,'video':Path(request['case']['video']).read_text(),
            'prompt':Path(request['case']['prompt_file']).read_text() if stage=='object_masks' else None,
            'dependencies':request['dependencies']})
        return 0


def test_resume_reuses_verified_outputs_and_repairs_tampering(config):
    execute=Executor();state=run(config,execute)
    assert state['status']=='complete' and len(execute.calls)==len(STAGES)
    execute.calls.clear();run(config,execute);assert execute.calls==[]
    stage=state['cases']['one']['stages']['object_tracks']
    (Path(stage['directory'])/'result.json').write_text('{}')
    run(config,execute)
    assert [s for _,s,_ in execute.calls]==['object_tracks','object_crops','object_anomaly']


def test_prompt_change_does_not_rerun_link7_masking(config):
    execute=Executor();run(config,execute);execute.calls.clear()
    Path(config['cases'][0]['prompt_file']).write_text('red cup')
    run(config,execute)
    assert [s for _,s,_ in execute.calls]==['object_masks','mask_join','object_tracks','object_crops','robot_score','object_anomaly']


def test_exhaustion_disables_case_preserves_evidence_and_continues_batch(config):
    config['cases'].append({**config['cases'][0],'id':'two'})
    execute=Executor();execute.fail.add(('one','object_masks'))
    state=run(config,execute)
    assert state['status']=='completed_with_disabled_cases'
    assert state['cases']['two']['status']=='complete'
    assert [a for c,s,a in execute.calls if c=='one' and s=='object_masks']==[1,2,3,4]
    assert not any(c=='one' and s=='mask_join' for c,s,a in execute.calls)
    execute.calls.clear();run(config,execute);assert execute.calls==[]
    execute.fail.clear();state=run(config,execute,retry_disabled=True)
    assert state['status']=='complete'
    assert execute.calls[0]==('one','object_masks',1)
    attempts=Path(config['output'])/'object/one/object_masks'
    assert len(list(attempts.glob('attempt-*')))==5


def test_interrupted_vlm_attempt_consumes_budget_and_resumes(config):
    execute=Executor();execute.interrupt='vlm2'
    with pytest.raises(KeyboardInterrupt):run(config,execute)
    state=read(Path(config['output'])/'run.json')
    assert state['status']=='interrupted'
    execute.interrupt=None;execute.calls.clear();run(config,execute)
    assert execute.calls[0]==('one','vlm2',2)
    assert not any(s=='vlm1' for _,s,_ in execute.calls)


def test_required_repair_failure_never_reaches_scoring(config):
    execute=Executor();execute.fail.add(('one','mask_join'));state=run(config,execute)
    assert state['cases']['one']['status']=='disabled'
    assert not any(s in ('robot_score','object_crops','object_anomaly') for _,s,_ in execute.calls)


def test_nonvlm_failure_is_not_retried_four_times(config):
    execute=Executor();execute.fail.add(('one','robot_score'));run(config,execute)
    assert [a for _,s,a in execute.calls if s=='robot_score']==[1]


def test_preflight_fails_before_any_worker(config):
    execute=Executor()
    def fail(_):raise ValueError('missing credential')
    with pytest.raises(ValueError,match='missing credential'):coordinate(config,executor=execute,check=fail)
    assert not execute.calls


def test_lock_rejects_concurrent_driver(config):
    with locked(Path(config['output'])):
        with pytest.raises(RuntimeError,match='Another coordinator'):run(config,Executor())


def test_named_single_link_crop_inputs(tmp_path):
    import numpy as np
    from object.preprocessing.crop_pairs.reference_visible_pixels import load_case
    from infrastructure.deformation_detect.coordinator import digest
    folder=tmp_path/'arbitrary';(folder/'occlusion').mkdir(parents=True);(folder/'replay').mkdir()
    import cv2
    video=folder/'replay/source.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for _ in range(2):writer.write(np.zeros((32,32,3),np.uint8))
    writer.release()
    paths={}
    for role,name in [('object','task_object'),('gripper','link7')]:
        p=tmp_path/(role+'.npz');np.savez_compressed(p,object_names=np.array([name]),object_masks=np.ones((2,1,32,32),bool));paths[role]={'path':str(p),'sha256':digest(p)}
    write(folder/'occlusion/detection.json',{'case':'arbitrary','method':'gripper-occlusion-v2','frame_count':2,'inputs':paths,'source_video_sha256':digest(video)})
    _,objects,grippers,_=load_case(folder)
    assert objects.shape==grippers.shape==(2,32,32)


def test_live_orphan_worker_blocks_resume(config):
    import os,socket
    from infrastructure.deformation_detect.coordinator import refuse_live_worker
    folder=Path(config['output']);folder.mkdir()
    write(folder/'execution.json',{'pid':os.getpid(),'host':socket.gethostname()})
    with pytest.raises(RuntimeError,match='still alive'):
        refuse_live_worker({'status':'running','directory':str(folder)})


def test_missing_key_preflight_output_is_not_a_case_failure(config):
    execute=Executor()
    def check(_):raise RuntimeError('sam3 preflight failed: missing VLM credential')
    with pytest.raises(RuntimeError):coordinate(config,executor=execute,check=check)
    assert not (Path(config['output'])/'run.json').exists()


def test_standalone_link7_mask_join_without_prior_experiment(tmp_path):
    import numpy as np
    import cv2
    from infrastructure.deformation_detect.coordinator import digest
    from robot.workflows.coordination import mask_join
    name='newvideo';video=tmp_path/'source.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for _ in range(2):writer.write(np.zeros((32,32,3),np.uint8))
    writer.release()
    before=np.zeros((2,32,32),bool);before[:,2:8,2:8]=True
    obj=np.zeros_like(before);obj[:,20:25,20:25]=True
    work=tmp_path/'persistent/work';work.mkdir(parents=True)
    np.savez_compressed(work/'masks.npz',masks=before)
    write(work/'provenance.json',{'results':{name:{'status':'completed_checks','source_sha256':digest(video),'mask_source':str(work/'masks.npz'),'masks_sha256':digest(work/'masks.npz')}}})
    object_dir=tmp_path/'object/masking';object_dir.mkdir(parents=True)
    np.savez_compressed(object_dir/'segmentation.npz',object_names=np.array(['task_object']),object_ids=np.array([0]),object_masks=obj[:,None])
    write(object_dir/'grounding.json',{'video_sha256':digest(video),'segmentation_sha256':digest(object_dir/'segmentation.npz'),'target_object':'cube'})
    out=tmp_path/'joined';out.mkdir()
    result=mask_join({'directory':str(out),'case':{'id':name,'video':str(video)},'config':{'resources':{'vlm3_reference':str(tmp_path/'unused-reference.png')}},'dependencies':{'link7_masks':str(work.parent),'object_masks':str(object_dir.parent)}})
    assert result['repair_status']=='not_triggered'
    with np.load(out/'segmentation.npz') as z:
        assert z['object_names'].tolist()==['link7']
        assert np.array_equal(z['object_masks'][:,0],before)
    assert read(out/'selection.json')['policy']=='core_not_triggered'


def test_interrupted_last_vlm_attempt_does_not_get_free_retries(config):
    config['policy']['vlm_attempts']=1
    execute=Executor();execute.interrupt='vlm1'
    with pytest.raises(KeyboardInterrupt):run(config,execute)
    execute.interrupt=None;execute.calls.clear();state=run(config,execute)
    assert not execute.calls
    assert state['cases']['one']['status']=='disabled'
    assert state['cases']['one']['stages']['vlm1']['attempt']==1


def test_real_executor_timeout_terminates_worker_and_records_exit(tmp_path):
    import subprocess
    from infrastructure.deformation_detect.coordinator import execute
    interpreter=tmp_path/'slow-python'
    interpreter.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(30)\n')
    interpreter.chmod(0o700)
    request=tmp_path/'request.json';write(request,{})
    with pytest.raises(subprocess.TimeoutExpired):execute(request,str(interpreter),0.1)
    receipt=read(tmp_path/'execution.json')
    assert receipt['exit_code'] is not None and receipt['exit_code']!=0 and receipt['finished_at']


def test_vlm1_parse_failure_retries_only_failed_frame_and_never_means_normal(tmp_path,monkeypatch):
    from PIL import Image
    from robot.preprocessing.link7_persistent import pipeline
    from robot.workflows.coordination import persistent
    from infrastructure.deformation_detect.coordinator import digest
    work=tmp_path/'initial/work';work.mkdir(parents=True)
    image=work/'frame.png';Image.new('RGB',(20,20)).save(image)
    calls=[{'frame':i,'crop':str(image),'sha256':digest(image)} for i in range(2)]
    write(work/'provenance.json',{'config':{'vlm1_response_request':'test','vlm1_system_prompt':'test'},'results':{'case':{'status':'vlm1_pending','calls':calls,'source_frame_count':3,'reference':{'path':str(image)}}}})
    observed=[]
    class Client:
        def __init__(self,*args):pass
        def ask(self,images,prompt,**kwargs):
            frame=1 if 'frame 1.' in prompt else 0
            observed.append(frame)
            if observed==[0,1]:raise ValueError('bad JSON')
            return {'answer':'good'}
    # Isolate the retry integration from the already-tested scientific parser.
    import robot.preprocessing.link7_persistent.palm_recovery as parser
    monkeypatch.setattr(parser,'parse_palm_response',lambda _: {'state':'normal'})
    monkeypatch.setattr(pipeline,'VLMClient',Client)
    resources={'vlm2_examples':[str(image)]*3,'ffmpeg':'unused'}
    request={'directory':str(tmp_path/'attempt1'),'case':{'id':'case'},'config':{'resources':resources},'stage':'vlm1','dependencies':{'link7_initial':str(work.parent)}}
    with pytest.raises(ValueError,match='not no-deformation'):persistent(request)
    request.update(directory=str(tmp_path/'attempt2'),previous_attempt=str(tmp_path/'attempt1'))
    result=persistent(request)
    assert result['native_status']=='no_confirmed_deformation'
    assert observed==[0,1,1]


def test_link2_resume_and_branch_invalidation(config):
    from infrastructure.deformation_detect.coordinator import stages_for
    execute=Executor();run(config,execute);execute.calls.clear()
    config['robot_links']=['link2','link7']
    state=run(config,execute)
    assert [s for _,s,_ in execute.calls]==['link2_masks','robot_score']
    assert stages_for(config)['object_crops']==STAGES['object_crops']
    execute.calls.clear();run(config,execute);assert not execute.calls
    folder=Path(state['cases']['one']['stages']['link2_masks']['directory'])
    (folder/'result.json').write_text('{}')
    run(config,execute)
    assert [s for _,s,_ in execute.calls]==['link2_masks','robot_score']
    execute.calls.clear();config['robot_links']=['link7'];state=run(config,execute)
    assert [s for _,s,_ in execute.calls]==['robot_score']
    assert 'link2_masks' not in state['cases']['one']['stages']


@pytest.mark.parametrize('existing_link2',[False,True])
def test_link2_merge_preserves_selected_link7_and_other_channels(tmp_path, existing_link2):
    import cv2
    import numpy as np
    from infrastructure.deformation_detect.coordinator import digest
    from robot.workflows.coordination import scoring_segmentation
    video=tmp_path/'video.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for _ in range(2):writer.write(np.zeros((32,32,3),np.uint8))
    writer.release()
    names=['link5','link7']+(['link2'] if existing_link2 else [])
    base=np.zeros((2,len(names),32,32),bool);base[:,0,2:8,2:8]=True;base[:,1,20:25,20:25]=True
    link2=np.zeros((2,1,32,32),bool);link2[:,:,12:18,12:18]=True
    for name in ('joined','link2','score'):(tmp_path/name).mkdir()
    selected=tmp_path/'joined/segmentation.npz'
    np.savez_compressed(selected,object_masks=base,object_names=np.array(names),object_ids=np.arange(len(names)))
    source=tmp_path/'link2/segmentation.npz'
    np.savez_compressed(source,object_masks=link2,object_names=np.array(['link2']),object_ids=np.array([0]))
    write(source.parent/'provenance.json',{'video_sha256':digest(video),'segmentation_sha256':digest(source)})
    before=digest(selected)
    request={'directory':str(tmp_path/'score'),'case':{'video':str(video)},'config':{'robot_links':['link2','link7']},
        'dependencies':{'mask_join':str(selected.parent),'link2_masks':str(source.parent)}}
    output=scoring_segmentation(request)
    assert digest(selected)==before
    with np.load(output) as z:
        assert z['object_names'].tolist()==['link5','link7','link2']
        assert len(set(z['object_ids']))==3
        assert np.array_equal(z['object_masks'][:,:2],base[:,:2])
        assert np.array_equal(z['object_masks'][:,2],link2[:,0])
        assert np.array_equal(z['masks'],z['object_masks'].any(axis=1))
    write(source.parent/'provenance.json',{'video_sha256':'wrong','segmentation_sha256':digest(source)})
    with pytest.raises(ValueError,match='identity mismatch'):scoring_segmentation(request)


def test_link2_manifest_requires_references_and_keeps_old_manifest_compatible(tmp_path):
    template=Path(__file__).resolve().parents[2]/'documentation/pipeline/coordinator.example.json'
    data=read(template);path=tmp_path/'manifest.json';write(path,data)
    assert load_manifest(path)['robot_links']==['link2','link7']
    data['resources'].pop('robot_references');write(path,data)
    with pytest.raises(ValueError,match='robot_references'):load_manifest(path)
    data.pop('robot_links');write(path,data)
    assert load_manifest(path)['robot_links']==['link7']
    data['robot_links']=['link5','link7'];write(path,data)
    with pytest.raises(ValueError,match='robot_links'):load_manifest(path)
