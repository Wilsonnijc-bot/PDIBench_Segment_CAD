"""Strict loading of the new structural-only experimental checkpoints."""
from pathlib import Path

import torch

from robot.experiments.link5_shape_codebook.common import ROOT, read, sha, split_path
from robot.experiments.link5_shape_codebook.localization_objective import install_point_residual
from robot.experiments.link5_shape_codebook.train_localization import load_points
from robot.experiments.link5_shape_codebook.upstream import build_model, restore_hash_keys


def load_checkpoint(config, checkpoint):
    root = Path(config['output']); checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    policy = payload['policy']; norm = payload['normalization']; split = read(split_path(root))
    if policy['normal_manifest']!=split['train'] or len(split['train'])!=4 or any(r['frame_id']!=0 for r in split['train']):
        raise ValueError('checkpoint normal inputs differ from exact four frame0 references')
    if policy['normal_split_sha256']!=sha(split_path(root)) or policy['normalization_sha256']!=sha(root / 'link5_normalization.json'):
        raise ValueError('checkpoint reference provenance mismatch')
    if norm!=read(root / 'link5_normalization.json'):raise ValueError('fixed normalization changed')
    for row in split['train']:
        if sha(row['input_path'])!=row['input_sha256']:raise ValueError('frozen input changed')
    for name,digest in policy['adapter_sources'].items():
        if sha(ROOT / 'robot/experiments/link5_shape_codebook' / name)!=digest:
            raise ValueError('checkpoint numerical adapter source changed: '+name)
    model, _, _ = build_model(payload['config']); model.cuda()
    variant = payload['variant']
    if variant=='structural_point_residual':install_point_residual(model)
    elif variant in ('structural_geometry_head','structural_invariant_geometry_head','structural_profile_decoder'):
        from robot.experiments.link5_shape_codebook.structural_geometry_head import wrap_geometry_model
        samples=[(row,*load_points(row,norm)) for row in split['train']]
        geometry=read(root / 'augmentation/robot_structural_geometry.json')
        if variant=='structural_profile_decoder':
            from robot.experiments.link5_shape_codebook.profile_decoder import wrap_profile_model
            model=wrap_profile_model(model,samples,geometry)
        elif variant=='structural_invariant_geometry_head':
            from robot.experiments.link5_shape_codebook.structural_invariant_head import wrap_invariant_model
            model=wrap_invariant_model(model,samples,geometry)
        else:model=wrap_geometry_model(model,samples,geometry)
        for i,s in enumerate(samples):
            torch.testing.assert_close(payload['model'][f'normal_{i}'],s[1].cpu(),rtol=0,atol=0)
    elif variant!='structural_native':raise ValueError('unknown structural checkpoint architecture')
    model.load_state_dict(payload['model'],strict=True); restore_hash_keys(model,payload['codebook_hash_state']); model.eval()
    if hasattr(model,'reference_exclusion'):model.reference_exclusion=None
    return model,payload
