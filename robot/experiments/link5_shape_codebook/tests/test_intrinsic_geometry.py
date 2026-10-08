"""Test reconstruction-scale invariance without losing structural strain."""
import torch
from robot.experiments.link5_shape_codebook.structural_invariant_head import upgrade_geometry_model
from robot.experiments.link5_shape_codebook.tests.test_structural_geometry_head import fixture


def model_and_cloud():
    old,normals=fixture()
    return upgrade_geometry_model(old,dict(axis=[1,0,0],anchor=[0,0,0],length=1)),normals[0][None],old


def test_uniform_scale_keeps_length_thickness_but_shortening_changes_it():
    model,cloud,_=model_and_cloud()
    intrinsic=model.intrinsic_features(cloud)
    scaled=model.intrinsic_features(.8*cloud+cloud.new_tensor([.04,-.02,.05]))
    torch.testing.assert_close(intrinsic[...,4],scaled[...,4],atol=1e-4,rtol=1e-4)
    shortened=cloud.clone();shortened[...,0]*=.75
    shorter=model.intrinsic_features(shortened)
    assert float(shorter[0,0,4])<float(intrinsic[0,0,4])*.85
    torch.testing.assert_close(model.normal_0,cloud[0],rtol=0,atol=0)


def test_bending_changes_intrinsic_curve_profile():
    model,cloud,_=model_and_cloud()
    bent=cloud.clone();bent[...,1]+=.2*cloud[...,0].square()
    assert not torch.allclose(model.intrinsic_features(cloud)[...,7:],model.intrinsic_features(bent)[...,7:])


def test_expanded_head_preserves_warm_predictions_and_learns_new_features():
    model,cloud,old=model_and_cloud()
    expected=old(cloud);actual=model(cloud)
    torch.testing.assert_close(expected[0],actual[0],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(expected[1],actual[1],atol=1e-6,rtol=1e-5)
    (actual[0].square().mean()+actual[1].square().mean()).backward()
    extra=model.geometry_mlp[0].weight.grad[:,-4-model.EXTRA_FEATURES:-4]
    assert torch.isfinite(extra).all() and extra.abs().max()>0
