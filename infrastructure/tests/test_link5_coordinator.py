"""Link5 guard, scheduler retry routing and named-mask handoff without paid calls."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image
import pytest

from infrastructure.deformation_detect.coordinator import coordinate, digest, load_manifest, read, review, stages_for, write
from infrastructure.shared.contracts.vlm_failure import VLMTimeoutError
from infrastructure.deformation_detect.worker import configure
from robot.preprocessing.link5_refinement.link5_point_guard import review_link5_points
from robot.preprocessing.segmentation.sam3_dinov2_segment import _link5_seed_points, _refine_link5_wrist_prompt
from robot.workflows.coordination import link5_masks, scoring_outcomes, scoring_segmentation


def make_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for _ in range(2):writer.write(np.zeros((32,32,3),np.uint8))
    writer.release()


def test_three_positive_two_negative_prompt_and_native_diagnostics():
    points,labels,_=_link5_seed_points((0,0,32,32),(32,32),2)
    assert labels==[1,1,1,0,0] and points.shape==(5,2)
    calls=[];mask=np.ones((32,32),bool)
    for x,y in points[3:]:mask[y,x]=False
    class Predictor:
        def handle_request(self,request):
            calls.append(request)
            return {'outputs':{'out_obj_ids':[1],'out_binary_masks':[mask]}}
    result,record=_refine_link5_wrist_prompt(Predictor(),'session',0,1,np.ones_like(mask),
        (0,0,32,32),points_override=points,negative_points=2)
    assert calls[0]['point_labels']==labels and len(calls[0]['points'])==5
    assert record['point_labels_accepted'] and np.array_equal(result,mask)
    # Membership disagreement remains a diagnostic, as in the native workflow.
    mask[:]=False
    _,record=_refine_link5_wrist_prompt(Predictor(),'session',0,1,np.ones_like(mask),
        (0,0,32,32),points_override=points,negative_points=2)
    assert not record['point_labels_accepted']
    assert record['mask_acceptance_policy']=='no_point_membership_or_shaft_retention_gate'


@pytest.mark.parametrize('failure',['timeout','null','points'])
def test_guard_saves_failed_evidence_and_does_not_submit_unreviewed_points(tmp_path,monkeypatch,failure):
    reference=tmp_path/'reference.png';Image.new('RGB',(832,480)).save(reference)
    points=np.array([[488,69],[569,69],[767,44],[495,27],[478,72]])
    calls=[]
    class Client:
        model='test'
        def __init__(self,role,config):calls.append((role,config))
        def ask(self,*args,**kwargs):
            if failure=='timeout':raise VLMTimeoutError('VLM request timed out without a response')
            if failure=='null':raise ValueError('missing text')
            return {'answer':'{"decision":"REJECT","negative_points_xy":[[9999,27],[472,82]]}'}
    monkeypatch.setattr('robot.preprocessing.link7_persistent.vlm_client.VLMClient',Client)
    with pytest.raises(VLMTimeoutError if failure=='timeout' else ValueError):
        review_link5_points(Image.new('RGB',(832,480)),(456,0,818,126),points,tmp_path/'guard',
                            reference_frame=reference,required=True)
    result=read(tmp_path/'guard/link5_guard.json')
    assert len(calls)==(3 if failure=='points' else 1)
    assert result['fallback_to_default'] is (failure!='points')
    assert len(result['selected_points_xy'])==5
    if failure=='points':assert result['decision']=='FAILED_PARSE_RETRIES'
    if failure=='timeout':assert result['attempts'][0]['failure']['defer']


def test_guard_luna_route_uses_endpoint_and_long_timeout(tmp_path,monkeypatch):
    reference=tmp_path/'reference.png';Image.new('RGB',(832,480)).save(reference)
    resources=dict.fromkeys(('sam3_checkpoint','sam3_bpe','dino_directory','palm_references','qwen_model'),'unused')
    resources['ffmpeg']='/usr/bin/ffmpeg'
    config={'resources':resources,'environments':{'vlm1':sys.executable},
            'vlm':{'vlm2':{'api_base':'https://api.302.ai/v1'}}}
    monkeypatch.setenv('PDI_VLM_ROLE_OVERRIDES','{}')
    configure({'config':config,'stage':'link5_masks','vlm_route':'alternate','attempt':2})
    calls=[]
    class Client:
        def __init__(self,role,config):self.role=role;calls.append((role,config))
        def ask(self,*args,**kwargs):
            return {'answer':'PASS' if self.role=='link5_negative_guard' else
                    '{"decision":"PLACE","positive_points_xy":[[510,65],[569,69],[767,44]]}'}
    monkeypatch.setattr('robot.preprocessing.link7_persistent.vlm_client.VLMClient',Client)
    points=np.array([[488,69],[569,69],[767,44],[495,27],[478,72]])
    selected,result=review_link5_points(Image.new('RGB',(832,480)),(456,0,818,126),points,
        tmp_path/'guard',reference_frame=reference,required=True)
    assert len(calls)==2 and result['decision']=='REJECT'
    assert selected[0].tolist()==[510,65] and np.array_equal(selected[1:],points[1:])
    assert all(c['model']=='gpt-6-luna' and c['timeout_seconds']>=600
               and c['api_base']=='https://api.302.ai/v1' for _,c in calls)


def test_link5_adapter_selects_guard_and_five_points(tmp_path,monkeypatch):
    video=tmp_path/'video.mp4';make_video(video)
    out=tmp_path/'attempt';out.mkdir();captured=[]
    def segment(args):
        captured.append(args)
        np.savez_compressed(args.output_npz,object_masks=np.ones((2,1,32,32),bool),
                            object_names=np.array(['link5']),object_ids=np.array([0]))
        return {'targets':[{'sam3_status':'complete'}]}
    monkeypatch.setattr('robot.preprocessing.segmentation.sam3_dinov2_segment.run',segment)
    resources=dict.fromkeys(('robot_references','dino_directory','sam3_checkpoint','sam3_bpe','link5_guard_reference'),'unused')
    resources['link5_positive_guard_reference']='three-point-reference.png'
    link5_masks({'directory':str(out),'case':{'video':str(video)},'config':{'resources':resources}})
    args=captured[0]
    assert args.selected_target=='link5' and args.link5_vlm_guard and args.link5_guard_required
    assert args.link5_negative_points==2 and args.link5_guard_reference==Path('unused')
    assert args.link5_positive_guard_reference==Path('three-point-reference.png')
    assert read(out/'provenance.json')['segmentation_sha256']==digest(out/'segmentation.npz')


@pytest.mark.parametrize('existing_link5',[False,True])
def test_merge_fresh_link5_preserves_joint_archive_and_link2(tmp_path,existing_link5):
    video=tmp_path/'video.mp4';make_video(video)
    names=['link7','link3']+(['link5'] if existing_link5 else [])
    base=np.zeros((2,len(names),32,32),bool);base[:,:,1:3,1:3]=True
    selected=tmp_path/'joined/segmentation.npz';selected.parent.mkdir()
    np.savez_compressed(selected,object_masks=base,object_names=names,object_ids=np.arange(len(names)))
    before=digest(selected);deps={'mask_join':str(selected.parent)}
    extras={}
    for link,offset in [('link2',5),('link5',15)]:
        folder=tmp_path/link;folder.mkdir();path=folder/'segmentation.npz'
        masks=np.zeros((2,1,32,32),bool);masks[:,:,offset:offset+3,offset:offset+3]=True
        np.savez_compressed(path,object_masks=masks,object_names=[link],object_ids=[0])
        write(folder/'provenance.json',{'video_sha256':digest(video),'segmentation_sha256':digest(path)})
        deps[f'{link}_masks']=str(folder);extras[link]=masks[:,0]
    out=tmp_path/'score';out.mkdir()
    result=scoring_segmentation({'directory':str(out),'case':{'video':str(video)},
        'config':{'robot_links':['link2','link5','link7']},'dependencies':deps})
    assert digest(selected)==before
    with np.load(result) as z:
        names=z['object_names'].tolist()
        assert np.array_equal(z['object_masks'][:,:2],base[:,:2])
        for link,masks in extras.items():assert np.array_equal(z['object_masks'][:,names.index(link)],masks)
        assert len(set(z['object_ids']))==len(names)
        assert np.array_equal(z['masks'],z['object_masks'].any(axis=1))
    write(Path(deps['link5_masks'])/'provenance.json',{'video_sha256':'wrong','segmentation_sha256':'wrong'})
    with pytest.raises(ValueError,match='Link5 mask identity mismatch'):
        scoring_segmentation({'directory':str(out),'case':{'video':str(video)},
            'config':{'robot_links':['link5','link7']},'dependencies':deps})


@pytest.mark.parametrize('failure',['timeout','semantic'])
def test_link5_scheduler_retries_preserves_other_branches_and_resumes(tmp_path,failure):
    video=tmp_path/'video.mp4';video.write_text('video');prompt=tmp_path/'prompt';prompt.write_text('object')
    config={'schema_version':1,'run_id':'test','output':str(tmp_path/'run'),
        'environments':dict.fromkeys(('sam3','vlm1','geometry','anomaly'),sys.executable),
        'resources':{},'vlm':{},'robot_links':['link2','link5','link7'],
        'policy':{'vlm_attempts':4,'vlm_stage_timeout_seconds':60,'stage_timeout_seconds':60},
        'cases':[{'id':name,'video':str(video),'prompt_file':str(prompt)} for name in ('one','two')]}
    requests=[]
    def execute(path,python,timeout):
        r=read(path);requests.append(r)
        if r['case']['id']=='one' and r['stage']=='link5_masks' and r['attempt']==1:
            write(path.parent/'result.json',{'status':'failed','error':'guard failed',
                'failure':VLMTimeoutError.failure if failure=='timeout' else {}});return 1
        write(path.parent/'result.json',{'status':'complete','stage':r['stage']});return 0
    def run():return coordinate(config,executor=execute,check=lambda _: {},source_identity={'fixed':1})
    state=run();assert state['status']=='complete'
    retry=next(r for r in requests if r['case']['id']=='one' and r['stage']=='link5_masks' and r['attempt']==2)
    assert retry['vlm_route']==('primary' if failure=='timeout' else 'alternate')
    assert retry['model_attempt']==(1 if failure=='timeout' else 2)
    assert Path(retry['previous_attempt']).joinpath('result.json').exists()
    if failure=='timeout':
        later=next(r for r in requests if r['case']['id']=='two' and r['stage']=='object_anomaly')
        assert requests.index(later)<requests.index(retry)
    requests.clear();run();assert not requests
    # Adding/changing Link5 references touches only Link5 and robot scoring.
    reference=tmp_path/'guard.png';reference.write_text('new guard reference')
    config['resources']['link5_guard_reference']=str(reference);run()
    assert {r['stage'] for r in requests}=={'link5_masks','robot_score'}
    requests.clear()
    positive_reference=tmp_path/'positive-reference.png';positive_reference.write_text('new three-point example')
    config['resources']['link5_positive_guard_reference']=str(positive_reference);run()
    assert {r['stage'] for r in requests}=={'link5_masks','robot_score'}
    assert stages_for(config)['robot_score'][1]==('mask_join','link2_masks','link5_masks')
    assert stages_for(config)['object_crops'][1]==('mask_join','object_masks','object_tracks')


def test_link5_insufficient_tracks_is_completed_outcome():
    result=scoring_outcomes({'link5':{'status':'failed','error_type':'insufficient_cotracker_tracks',
                                      'tracking':{'valid':2}}},['link5'])
    assert result['link5']['rigidity'] is None
    assert result['link5']['outcome']=='insufficient_cotracker_tracks'


@pytest.mark.parametrize('answer', ['REJECT', '{"decision":"REJECT","negative_points_xy":[[100,27],[120,82]]}'])
def test_negative_guard_exhaustion_stops_coordinator_after_three_calls(tmp_path, monkeypatch, answer):
    from infrastructure.shared.contracts.vlm_failure import failure_record
    from robot.preprocessing.link5_refinement.link5_point_guard import NegativeGuardExhausted

    reference=tmp_path/'reference.png';Image.new('RGB',(832,480)).save(reference)
    video=tmp_path/'video.mp4';video.write_bytes(b'fixture')
    prompt=tmp_path/'prompt.txt';prompt.write_text('cube')
    config={'run_id':'negative-budget','output':str(tmp_path/'run'),
        'environments':dict.fromkeys(('sam3','vlm1','geometry','anomaly'),sys.executable),
        'resources':{},'vlm':{},'robot_links':['link5','link7'],
        'policy':{'vlm_attempts':4,'vlm_stage_timeout_seconds':60,'stage_timeout_seconds':60},
        'cases':[{'id':'case','video':str(video),'prompt_file':str(prompt)}]}
    calls=[];stages=[]
    class Client:
        def __init__(self,role,config):self.model='test';self.role=role
        def ask(self,*args,**kwargs):
            calls.append(self.role)
            return {'answer':answer}
    monkeypatch.setattr('robot.preprocessing.link7_persistent.vlm_client.VLMClient',Client)
    def execute(path,python,timeout):
        request=read(path);stages.append(request['stage'])
        if request['stage']=='link5_masks':
            try:
                review_link5_points(Image.new('RGB',(832,480)),(456,0,818,126),
                    np.array([[488,69],[569,69],[767,44],[495,27],[478,72]]),path.parent,
                    reference_frame=reference,positive_reference_frame=reference,required=True)
            except NegativeGuardExhausted as error:
                write(path.parent/'result.json',{'status':'failed','error':str(error),'failure':failure_record(error)})
                return 1
        write(path.parent/'result.json',{'status':'complete'})
        return 0
    state=coordinate(config,executor=execute,check=lambda _: {},source_identity={'fixed':1})
    assert calls==['link5_negative_guard']*3
    assert stages.count('link5_masks')==1 and 'robot_score' not in stages
    entry=state['cases']['case']['stages']['link5_masks']
    assert entry['status']=='disabled' and entry['attempt']==1
    assert entry['failure']['retryable'] is False


def test_three_link_manifest_and_link5_replay_review(tmp_path):
    template=Path(__file__).resolve().parents[2]/'documentation/pipeline/coordinator.example.json'
    config=load_manifest(template)
    assert config['robot_links']==['link2','link5','link7']
    assert 'link5_masks' in stages_for(config)
    score=tmp_path/'score';replay=score/'score/replay/interactive_exact-group/link5_exact-group.html'
    replay.parent.mkdir(parents=True);replay.write_text('<html>Link5</html>')
    review(tmp_path,{'cases':{'one':{'status':'complete','stages':{
        'robot_score':{'status':'complete','directory':str(score)}}}}})
    assert 'Link5 replay' in (tmp_path/'index.html').read_text()


def test_robot_score_requests_three_links_and_keeps_unavailable_outcome(tmp_path,monkeypatch):
    from robot.workflows.coordination import robot_score
    out=tmp_path/'attempt';out.mkdir();video=tmp_path/'video.mp4';video.write_bytes(b'video')
    config=tmp_path/'robot.yaml';config.write_text('{}')
    segmentation=tmp_path/'segmentation.npz';segmentation.write_bytes(b'test')
    monkeypatch.setattr('robot.workflows.coordination.scoring_segmentation',lambda _: segmentation)
    monkeypatch.setattr('infrastructure.deformation_detect.worker.isolate_mega',lambda _: (out/'mega_sam').mkdir())
    calls=[]
    def execute(command,check):
        calls.append(command)
        metrics={link:{'status':'complete','breakdown':{'epsilon_rigidity':.1}}
                 for link in ('link2','link5','link7')}
        metrics['link5']={'status':'failed','error_type':'insufficient_cotracker_tracks'}
        write(out/'score/metrics.json',{'modes':{'exact-group':{'objects':metrics}}})
    monkeypatch.setattr('robot.workflows.coordination.subprocess.run',execute)
    result=robot_score({'directory':str(out),'case':{'video':str(video)},
        'config':{'resources':{'robot_config':str(config),'tracker_checkpoint':'tracker'},
                  'robot_links':['link2','link5','link7']}})
    command=calls[0]
    assert [command[i+1] for i,word in enumerate(command) if word=='--score-link']==['link2','link5','link7']
    assert command[command.index('--tracking-mode')+1]=='exact-group'
    assert result['rigidity_by_link']=={'link2':.1,'link5':None,'link7':.1}
    assert not (out/'mega_sam').exists()
