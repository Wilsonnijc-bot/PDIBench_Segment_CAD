from concurrent.futures import ThreadPoolExecutor
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot.experiments.link5_shape_codebook.common import read,sha,write
from robot.experiments.link5_shape_codebook.training_loader import NativeEquivalentLoader,measure
from robot.experiments.link5_shape_codebook.upstream import shape_modules
from robot.experiments.link5_shape_codebook import train_stage


def test_single_process_matches_native_worker_sampling_and_global_rng(tmp_path):
    modules,_=shape_modules();module=modules['dataset']
    directory=tmp_path/'link5/train/normal';directory.mkdir(parents=True)
    for i,count in enumerate((217,233,301,313)):
        np.save(directory/f'{i}.npy',np.random.default_rng(i).normal(size=(count,3)).astype(np.float32))
    factory=lambda:module.PointCloudAnomalyDataset(tmp_path,'link5','train_normal',num_points=128,normalize=False,seed=0)
    receipt=measure(factory,epochs=3,loader_kwargs=dict(multiprocessing_context='fork'))
    assert receipt['status']=='passed' and receipt['batches']==12
    assert receipt['point_order_names_and_torch_rng_identical']
    with pytest.raises(ValueError,match='batch1'):NativeEquivalentLoader(factory(),batch_size=2,num_workers=2)


def test_duplicate_training_launches_reuse_one_verified_result(tmp_path,monkeypatch):
    config=tmp_path/'config.json';write(config,dict(output=str(tmp_path)))
    write(tmp_path/'splits/normal_reference.json',dict(train=[dict(frame_id=0,video_id='fixture')]))
    write(tmp_path/'link5_normalization.json',dict(fixture=True))
    calls=[]
    def invoke(command,**kwargs):
        calls.append(command);time.sleep(.1)
        folder=tmp_path/'training';checkpoint=folder/'link5/final.pt'
        checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint fixture')
        write(folder/'paired_control.json',dict(split_sha256=sha(tmp_path/'splits/normal_reference.json'),normalization_sha256=sha(tmp_path/'link5_normalization.json')))
        write(folder/'normal_manifest.json',read(tmp_path/'splits/normal_reference.json')['train'])
        write(folder/'completion.json',dict(status='complete',epoch=1500,checkpoint_sha256=sha(checkpoint)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(train_stage.subprocess,'run',invoke)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(train_stage.run,config,'paper_original') for _ in range(2)]
        assert [f.result() for f in futures]==[0,0]
    assert len(calls)==1
    (tmp_path/'training/link5/final.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):train_stage.run(config,'paper_original')
