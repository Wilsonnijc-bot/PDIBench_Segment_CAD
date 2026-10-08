"""Whole-link context and normal-reference isolation checks for the new head."""
import torch
from torch import nn

from robot.experiments.link5_shape_codebook.structural_geometry_head import StructuralGeometryDetector


class NativeFixture(nn.Module):
    feature_dim = 32

    def __init__(self):
        super().__init__(); self.encoder = nn.Linear(3, 32); self.codebook = None

    def forward(self, points):
        self.encoder(points)
        return points.new_zeros((*points.shape[:2],3)), points.new_zeros(points.shape[:2]), {}


def fixture():
    x = torch.linspace(0,1,240)
    cloud = torch.stack([x, .03*torch.sin(x*30), .03*torch.cos(x*30)], dim=-1)
    normals = [cloud+torch.tensor([0.,s,0.]) for s in (0,.005,.01,.015)]
    model = StructuralGeometryDetector(NativeFixture(), normals, dict(axis=[1,0,0],anchor=[0,0,0],length=1))
    return model, normals


def test_length_feature_preserves_shortening_in_fixed_normal_coordinates():
    model, normals = fixture(); normal = normals[0][None]
    shorter = normal.clone(); shorter[...,0]*=.75
    original = model.geometry_features(normal); deformed = model.geometry_features(shorter)
    # Seven point-local/neighbor features precede six global 1%/99% quantiles.
    assert float(deformed[0,0,10]) < float(original[0,0,10])*.8
    torch.testing.assert_close(model.normal_0, normals[0], rtol=0, atol=0)


def test_training_reference_exclusion_removes_exact_self_match_shortcut():
    model, normals = fixture(); normal = normals[0][None]
    all_four = model.geometry_features(normal)
    model.reference_exclusion = 0
    other_three = model.geometry_features(normal)
    assert float(all_four[...,6].max())==0
    assert float(other_three[...,6].median())>.004
    assert len(model._banks[0])==3*len(normals[0])


def test_bending_changes_whole_link_profile_and_head_has_finite_gradients():
    model, normals = fixture(); cloud = normals[0][None]
    bent = cloud.clone(); bent[...,1]+=.3*cloud[...,0].square()
    assert not torch.allclose(model.geometry_features(cloud)[...,7:], model.geometry_features(bent)[...,7:])
    offset, logits, _ = model(bent)
    assert offset.shape==(1,240,3) and logits.shape==(1,240)
    (offset.square().mean()+logits.square().mean()).backward()
    assert model.geometry_mlp[0].weight.grad is not None
    assert torch.isfinite(model.geometry_mlp[0].weight.grad).all()
