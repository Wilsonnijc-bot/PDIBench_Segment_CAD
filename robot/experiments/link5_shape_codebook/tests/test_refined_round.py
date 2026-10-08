"""Training is independent of test preparation; failed validation stops scoring."""
from types import SimpleNamespace
import pytest
from robot.experiments.link5_shape_codebook.common import MODES,mode_folder,read,write,sha
from robot.experiments.link5_shape_codebook import run_refined_round as runner


def test_stage_order_and_both_failed_validations_are_preserved(tmp_path,monkeypatch):
    config=tmp_path/'config.json'
    write(config,dict(output=str(tmp_path),preparation=dict(workers_per_gpu=5),environments=dict(geometry='python')))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3');monkeypatch.setenv('SLURM_JOB_ID','fixture')
    calls=[]
    def invoke(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if command[-1].startswith('validate') else 0)
    monkeypatch.setattr(runner.subprocess,'run',invoke)
    with pytest.raises(RuntimeError,match='validation failed'):runner.run(config,tmp_path/'dependency')
    receipt=read(tmp_path/'metadata/refined-round-fixture/progress.json')
    assert receipt['status']=='failed'
    stages=[r['stage'] for r in receipt['stages']]
    assert stages[:3]==['preflight','four_reference_alignment','fixed_normalization_and_frame0_pose_qc']
    assert set(stages[3:6])=={'train','train_robot_structural','prepare_test_videos_shared_workers'}
    assert stages[6:]==['validate','validate_robot_structural']
    assert len(calls)==8
    assert not (tmp_path/'full_video/config.json').exists()


def test_new_full_video_contract_requires_passed_validation_and_one_gpu(tmp_path):
    from robot.experiments.link5_shape_codebook.full_video import validate_contract
    config_path=tmp_path/'config.json'
    cases=[dict(id=generator+'_'+str(number)) for number in range(15) for generator in ('LVP_ROBOWM','COSMOS2.5','COSMOS3')]
    config=dict(output=str(tmp_path),normal_reference=dict(frame_id=0),cases=cases)
    write(config_path,config);write(tmp_path/'link5_normalization.json',dict(fixture=True))
    write(tmp_path/'splits/normal_reference.json',dict(fixture=True))
    write(tmp_path/'splits/heldout_frame0.json',dict(fixture=True))
    for mode in MODES:
        checkpoint=mode_folder(tmp_path,'training',mode)/'link5/final.pt'
        checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'fixture checkpoint')
        write(mode_folder(tmp_path,'sanity',mode)/'checks.json',dict(status='passed',heldout_manifest_sha256=sha(tmp_path/'splits/heldout_frame0.json')))
    path=runner.evaluation_config(config_path);cfg=read(path);validate_contract(cfg)
    assert cfg['full_video']['gpu_count']==1 and cfg['full_video']['workers_per_gpu']==5
    cfg['full_video']['gpu_count']=5
    with pytest.raises(ValueError,match='one GPU'):validate_contract(cfg)
    cfg=read(path);receipt_path=tmp_path/'sanity/checks.json'
    receipt=read(receipt_path);receipt['status']='failed';write(receipt_path,receipt)
    cfg['full_video']['sanity_receipt_sha256']['paper_original']=sha(receipt_path)
    with pytest.raises(ValueError,match='validation failed'):validate_contract(cfg)
    with pytest.raises(ValueError,match='validations must pass'):runner.evaluation_config(config_path)
