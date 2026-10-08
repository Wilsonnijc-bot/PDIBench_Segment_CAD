"""Intrinsic restoration must retain axial strain and similarity equivariance."""
import torch
from robot.experiments.link5_shape_codebook.profile_decoder import StructuralProfileDecoder
from robot.experiments.link5_shape_codebook.tests.test_structural_geometry_head import fixture


def setup():
    old,clouds=fixture()
    return StructuralProfileDecoder(old.native,clouds,dict(axis=[1,0,0],anchor=[0,0,0],length=1)),clouds[0][None]


def test_similarity_preserves_probability_and_transforms_offset():
    model,xyz=setup();model.eval()
    angle=.035
    c,s=torch.cos(torch.tensor(angle)),torch.sin(torch.tensor(angle))
    rotation=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]])
    out,logits,_=model(xyz)
    scaled=.83*(xyz@rotation.T)+xyz.new_tensor([.04,-.03,.07])
    second,probability,_=model(scaled)
    torch.testing.assert_close(logits,probability,atol=1e-4,rtol=1e-4)
    torch.testing.assert_close(second,.83*(out@rotation.T),atol=1e-5,rtol=1e-4)


def test_axial_strain_changes_shape_and_reference_buffers_stay_exact():
    model,xyz=setup();short=xyz.clone();short[...,0]*=.75
    before=model.canonical_features(xyz)[0];after=model.canonical_features(short)[0]
    assert after[0,0,4]<before[0,0,4]*.85
    torch.testing.assert_close(model.normal_0,xyz[0],atol=0,rtol=0)
    (model(short)[0].square().mean()+model(short)[1].square().mean()).backward()
    assert model.profile_mlp[0].weight.grad is not None
    assert torch.isfinite(model.profile_mlp[0].weight.grad).all()
