"""Read-only check of new default-generator models on saved Link5 deformations.

Shortening/lengthening/bending were not used by these training trials. Retains
the old structural model's exact inputs/labels/scores for an honest comparison.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import raw_metrics, read, sha, split_path, write
from robot.experiments.link5_shape_codebook.localization_objective import install_point_residual
from robot.experiments.link5_shape_codebook.train_localization import VARIANTS, frozen_inputs, localization_metrics, prediction
from robot.experiments.link5_shape_codebook.upstream import build_model, restore_hash_keys


@torch.no_grad()
def run(config_path, variant):
    config = read(config_path); root, split, norm, heldout = frozen_inputs(config)
    destination = root / 'localization' / variant; checkpoint = destination / 'final.pt'
    if read(destination / 'completion.json')['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('new localization checkpoint changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['variant'] != variant or payload['policy']['normal_split_sha256'] != sha(split_path(root)):
        raise ValueError('checkpoint/training reference mismatch')
    if payload['policy']['normalization_sha256'] != sha(root / 'link5_normalization.json'):
        raise ValueError('normalization changed')
    assert torch.cuda.device_count() == 1
    assert float(torch.ones(4, device='cuda').sum()) == 4
    model, modules, source = build_model(payload['config'])
    model.load_state_dict(payload['model'], strict=True)
    restore_hash_keys(model, payload['codebook_hash_state'])
    if variant == 'paired_point_residual':install_point_residual(model)
    model.cuda().eval()
    old_root = root / 'sanity_robot_structural'; checks = read(old_root / 'checks.json')
    video = checks['heldout_normals'][0]['video_id']
    with np.load(old_root / (video + '_heldout_frame0.npz'), allow_pickle=False) as a:
        normal = a['points_normalized'].copy(); old_clean_scores = a['raw_point_scores'].copy()
    clean = prediction(model, torch.from_numpy(normal).cuda())
    threshold = read(destination / 'evaluation_epoch_1500.json')['frozen_clean_threshold']
    rows = []
    for test in checks['synthetic_deformations']:
        path = old_root / f"synthetic_{test['type']}_{test['parameter']}.npz"
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['normal'], normal)
            anomalous = a['anomalous'].copy(); mask = a['gt_mask'].copy(); offset = a['gt_offset'].copy()
            old_scores = a['raw_scores'].copy()
        arrays = prediction(model, torch.from_numpy(anomalous).cuda())
        metrics = localization_metrics(clean['raw_point_scores'], arrays['raw_point_scores'], mask, offset, threshold)
        old_region_ratio = float(old_scores[mask.astype(bool)].mean() / max(old_clean_scores[mask.astype(bool)].mean(), 1e-8))
        name = test['type'] + '_' + test['parameter']
        np.savez_compressed(destination / 'predictions' / (name + '_structural_transfer.npz'), **arrays,
                            normal=normal, anomalous=anomalous, gt_mask=mask, gt_offset=offset,
                            matched_clean_raw_scores=clean['raw_point_scores'], old_structural_raw_scores=old_scores)
        rows.append(dict(type=test['type'], parameter=test['parameter'], source_sha256=sha(path),
                         new_metrics=metrics, old_structural_raw_top80_ratio=test['ratio'],
                         old_structural_matched_region_score_ratio=old_region_ratio))
    receipt = dict(status='complete', variant=variant, checkpoint_sha256=sha(checkpoint), video_id=video, frame_id=0,
                   trained_on_structural_examples=False, normal_metrics=raw_metrics(clean['raw_point_scores']),
                   old_structural_normal_metrics=raw_metrics(old_clean_scores), cases=rows,
                   allocation_id=os.environ.get('SLURM_JOB_ID'), source_sha256=sha(__file__),
                   interpretation='Transfer diagnostic on existing synthetic shortening/lengthening/bending inputs; no training or codebook update.')
    write(destination / 'structural_transfer.json', receipt)
    print('LINK5_LOCALIZATION_TRANSFER_COMPLETE', variant, rows, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args(); run(args.config, args.variant)
