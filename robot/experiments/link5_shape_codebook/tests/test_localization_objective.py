import torch

from robot.experiments.link5_shape_codebook.localization_objective import paired_loss, region_mean


def test_exact_restoration_and_clean_prediction_beats_constant_prediction():
    gt = torch.zeros(1, 100, 3); gt[:, :10, 0] = 0.01
    mask = (gt.abs().sum(-1) > 0).float()
    clean = torch.zeros_like(gt)
    ideal_logits = torch.where(mask.bool(), 20.0, -20.0)
    ideal, _ = paired_loss(clean, torch.full_like(mask, -20), gt, ideal_logits, gt, mask)
    constant, _ = paired_loss(torch.full_like(gt, 0.03), torch.full_like(mask, 1.3),
                              torch.full_like(gt, 0.03), torch.full_like(mask, 1.3), gt, mask)
    assert ideal < 1e-4
    assert constant > 1


def test_small_regions_keep_equal_loss_weight():
    mask = torch.zeros(1, 1000, dtype=torch.bool); mask[:, 0] = True
    values = torch.zeros(1, 1000); values[:, 0] = 2
    assert region_mean(values, mask) == 1


def test_normal_loss_still_penalizes_offsets_when_probability_is_zero():
    gt = torch.zeros(1, 20, 3); gt[:, :5, 0] = 0.01
    mask = (gt.abs().sum(-1) > 0).float()
    clean = torch.full_like(gt, 0.02, requires_grad=True)
    loss, _ = paired_loss(clean, torch.full_like(mask, -20), gt, torch.where(mask.bool(), 20.0, -20.0), gt, mask)
    loss.backward()
    assert torch.isfinite(clean.grad).all()
    assert (clean.grad > 0).all()
