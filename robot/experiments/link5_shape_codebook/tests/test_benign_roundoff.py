"""The resumed audit preserves pair values and exact offset correspondence."""
import torch
from robot.experiments.link5_shape_codebook.benign_variation import paired_variation as old
from robot.experiments.link5_shape_codebook.benign_variation_v4 import paired_variation as corrected


def test_audit_change_leaves_successful_original_pair_bit_identical():
    normal=torch.arange(1200,dtype=torch.float32).reshape(400,3)/1200
    mask=torch.zeros(400);mask[200:]=1
    anomaly=normal.clone();anomaly[200:,0]-=.1
    first=old(normal,anomaly,mask,17,dict(identity_probability=0))
    second=corrected(normal,anomaly,mask,17,dict(identity_probability=0))
    for a,b in zip(first[:4],second[:4]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert first[4]==second[4]


def test_roundoff_with_larger_coordinates_does_not_break_exact_targets():
    torch.manual_seed(71)
    normal=torch.randn(5000,3)*3
    source=normal.clone();mask=torch.zeros(5000);mask[1000:]=1
    anomaly=normal.clone();anomaly[1000:,0]-=.05
    clean,deformed,offset,label,_=corrected(normal,anomaly,mask,707,dict(identity_probability=0))
    torch.testing.assert_close(offset,clean-deformed,rtol=0,atol=0)
    torch.testing.assert_close(clean[:1000],deformed[:1000],rtol=0,atol=0)
    torch.testing.assert_close(normal,source,rtol=0,atol=0)
    assert torch.equal(label.bool(),offset.ne(0).any(-1))
