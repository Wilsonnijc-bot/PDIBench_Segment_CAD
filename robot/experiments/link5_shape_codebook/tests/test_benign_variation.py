"""Benign copies cannot change GT region identity or frozen source arrays."""
import torch

from robot.experiments.link5_shape_codebook.benign_variation import paired_variation


def test_shared_benign_noise_cancels_and_unaffected_points_remain_identical():
    normal=torch.arange(1200,dtype=torch.float32).reshape(400,3)/1200
    source=normal.clone();mask=torch.zeros(400);mask[200:]=1
    anomaly=normal.clone();anomaly[200:,0]-=.1
    clean,deformed,offset,label,parameters=paired_variation(normal,anomaly,mask,seed=17,settings=dict(identity_probability=0))
    assert not parameters['identity'] and torch.equal(label,mask)
    torch.testing.assert_close(normal,source,atol=0,rtol=0)
    torch.testing.assert_close(offset,clean-deformed,atol=0,rtol=0)
    torch.testing.assert_close(clean[:200],deformed[:200],atol=0,rtol=0)
    assert torch.equal(offset.ne(0).any(-1),mask.bool())


def test_identity_draw_keeps_original_training_pair_and_rng_state():
    normal=torch.randn(300,3);anomaly=normal+.1;mask=torch.ones(300)
    before=torch.random.get_rng_state().clone()
    clean,deformed,offset,label,parameters=paired_variation(normal,anomaly,mask,seed=9,settings=dict(identity_probability=1))
    assert parameters['identity'] and torch.equal(torch.random.get_rng_state(),before)
    torch.testing.assert_close(clean,normal,atol=0,rtol=0);torch.testing.assert_close(deformed,anomaly,atol=0,rtol=0)
    torch.testing.assert_close(offset,normal-anomaly,atol=0,rtol=0)
