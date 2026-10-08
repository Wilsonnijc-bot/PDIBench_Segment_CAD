"""Reference identity, no temporal leakage, and fresh SAM provenance."""
from pathlib import Path

import numpy as np
import pytest
import cv2

from robot.experiments.link5_shape_codebook.common import FOUR_REFERENCE_IDS, read, reference_ids, sha, write
from robot.experiments.link5_shape_codebook.construct_reference import construct
from robot.experiments.link5_shape_codebook.prepare import save_observation, validate_mask_guard


def fixture(tmp_path):
    reference=tmp_path/'reference.png';reference.write_bytes(b'reference-fixture')
    cfg=dict(output=str(tmp_path),cases=[dict(id=v) for v in FOUR_REFERENCE_IDS],
        normal_reference=dict(video_ids=list(FOUR_REFERENCE_IDS),frame_id=0),
        resources=dict(link5_positive_guard_reference=str(reference)))
    clouds=[]
    for i,video in enumerate(FOUR_REFERENCE_IDS):
        folder=tmp_path/'cases'/video;mask=np.ones((32,32),bool)
        k=np.array([[50,0,16],[0,50,16],[0,0,1]],np.float32)
        row=save_observation(dict(id=video),folder,0,np.full((32,32,3),i,np.uint8),
            np.full((32,32),1+i*.01,np.float32),k,mask,np.eye(4))
        # Deliberately very different later frame: it must never enter the pool.
        save_observation(dict(id=video),folder,7,np.zeros((32,32,3),np.uint8),
            np.full((32,32),50,np.float32),k,mask,np.eye(4))
        with np.load(row['input_path']) as a:clouds.append(a['xyz_camera'].copy())
        attempt=folder/'masking/attempt-0001';attempt.mkdir(parents=True)
        segmentation=attempt/'segmentation.npz'
        np.savez_compressed(segmentation,object_masks=mask[None,None],object_names=np.array(['link5']))
        points=[[1,1],[12,12],[23,23],[8,8],[20,20]]
        write(attempt/'link5_guard.json',dict(positive_placement_policy='vlm_places_all_three_no_default_p1',
            p1_default_used=False,fallback_to_default=False,selected_points_xy=points,
            original_points_xy=[[-1,-1],*points[1:]],decision='REJECT',selected_image='link5_guard_selected.png',
            reviews=dict(positive=dict(decision='PLACE'),negative=dict(decision='PASS')),attempts=[
                dict(role='link5_positive_guard',model='gemini-3.8-flash',answer='PLACE',reasoning_effort='high'),
                dict(role='link5_negative_guard',model='gemini-3.8-flash',answer='PASS',reasoning_effort='high')],
            positive_reference_sha256=sha(reference)))
        write(attempt/'sam3_prompt_diagnostics.json',[dict(target='link5',point_refinement=dict(frame_index=0,points_xy=points,point_labels=[1,1,1,0,0]))])
        for name in ['selected','positive_current','negative_current','positive_reference','negative_reference']:
            cv2.imwrite(str(attempt/f'link5_guard_{name}.png'),np.zeros((32,32,3),np.uint8))
        write(attempt/'result.json',dict(status='complete',segmentation=str(segmentation)))
        write(folder/'status.json',dict(status='complete'))
    return cfg,clouds


def test_exact_four_frame0_union_excludes_later_frames(tmp_path):
    config,clouds=fixture(tmp_path)
    receipt=construct(config)
    with np.load(tmp_path/'normal_reference/pooled_frame0.npz') as a:
        np.testing.assert_array_equal(a['xyz_camera'],np.concatenate(clouds))
        assert a['frame_ids'].tolist()==[0]*4
        assert a['video_ids'].tolist()==list(FOUR_REFERENCE_IDS)
        assert np.max(a['xyz_camera'][:,2])<2
        assert np.bincount(a['source_index']).tolist()==[len(c) for c in clouds]
    assert receipt['later_frames_in_reference'] is False
    assert receipt['synthetic_surfaces'] is False
    # A preparation continuation must preserve the previewed exact archive,
    # rather than replacing it with a different ZIP timestamp/hash.
    before=sha(tmp_path/'normal_reference/pooled_frame0.npz')
    assert construct(config)==receipt
    assert sha(tmp_path/'normal_reference/pooled_frame0.npz')==before


@pytest.mark.parametrize('fault',['old_guard','wrong_sam','wrong_example'])
def test_historical_or_mismatched_masks_cannot_supply_new_reference(tmp_path,fault):
    config,_=fixture(tmp_path)
    p=tmp_path/'cases'/FOUR_REFERENCE_IDS[0]/'masking/attempt-0001'
    if fault=='wrong_sam':
        diagnostics=read(p/'sam3_prompt_diagnostics.json');diagnostics[0]['point_refinement']['points_xy'][0]=[5,5]
        write(p/'sam3_prompt_diagnostics.json',diagnostics)
    else:
        guard=read(p/'link5_guard.json')
        guard['positive_placement_policy' if fault=='old_guard' else 'positive_reference_sha256']='old'
        write(p/'link5_guard.json',guard)
    with pytest.raises(ValueError):construct(config)


def test_later_frame_reference_policy_is_rejected():
    cfg=dict(cases=[dict(id=v) for v in FOUR_REFERENCE_IDS],
        normal_reference=dict(video_ids=list(FOUR_REFERENCE_IDS),frame_id=1))
    with pytest.raises(ValueError,match='frame0 only'):reference_ids(cfg)


def test_replay_can_be_built_immediately_without_trained_detector(tmp_path):
    from robot.experiments.link5_shape_codebook.build_constructed_reference_replay import run
    config,_=fixture(tmp_path);receipt=construct(config)
    viewer=tmp_path/'viewer';run(tmp_path,viewer)
    verified=read(viewer/'verification.json')
    assert verified['frame_ids']==[0]*4
    assert verified['detector_retrained'] is False
    assert verified['observed_normal_point_count']==receipt['point_count']
    assert 'New reference: exactly four frame0' in (viewer/'index.html').read_text()


def test_alignment_selects_only_requested_frame0s_and_holds_out_other_videos(tmp_path):
    from robot.experiments.link5_shape_codebook.audit_alignment import audit
    config,_=fixture(tmp_path)
    extra='LVP_ROBOWM_0001';config['cases'].append(dict(id=extra))
    save_observation(dict(id=extra),tmp_path/'cases'/extra,0,np.zeros((32,32,3),np.uint8),
        np.ones((32,32),np.float32),np.array([[50,0,16],[0,50,16],[0,0,1]],np.float32),
        np.ones((32,32),bool),np.eye(4))
    audit(tmp_path,config)
    split=read(tmp_path/'splits/normal_reference.json')
    assert {r['video_id'] for r in split['train']}==set(FOUR_REFERENCE_IDS)
    assert [r['video_id'] for r in split['test']]==[extra]
    assert {r['frame_id'] for r in split['train']}=={0}
    assert not (tmp_path/'splits/train20_test25.json').exists()


def test_reference_training_does_not_wait_for_tests_or_change_after_preparation(tmp_path):
    from robot.experiments.link5_shape_codebook.audit_alignment import audit,collect_heldout
    config,_=fixture(tmp_path)
    extra='LVP_ROBOWM_0001';config['cases'].append(dict(id=extra))
    audit(tmp_path,config,reference_only=True)
    path=tmp_path/'splits/normal_reference.json';before=sha(path)
    audit_before=sha(tmp_path/'alignment/audit.json')
    split=read(path)
    assert len(split['train'])==4 and split['test']==[]
    assert split['declared_heldout_video_ids']==[extra]
    assert not (tmp_path/'cases'/extra).exists()
    # A later test-preparation stage adds only its own manifest.
    save_observation(dict(id=extra),tmp_path/'cases'/extra,0,np.zeros((32,32,3),np.uint8),
        np.ones((32,32),np.float32),np.array([[50,0,16],[0,50,16],[0,0,1]],np.float32),
        np.ones((32,32),bool),np.eye(4))
    receipt=collect_heldout(tmp_path,config)
    assert [r['video_id'] for r in receipt['test']]==[extra]
    assert receipt['training_split_sha256']==before
    assert sha(path)==before and sha(tmp_path/'alignment/audit.json')==audit_before
    with pytest.raises(ValueError,match='frozen'):audit(tmp_path,config)
    Path(split['train'][0]['input_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):audit(tmp_path,config,reference_only=True)
