"""Re-generate upstream synthetic defects on the four actual training clouds.

Read-only with respect to training: uses the same inputs, cached-normal ordering,
generator, and seeds as selected draws; original minibatches were not archived.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import FOUR_REFERENCE_IDS, ROOT, read, sha, write
from robot.experiments.link5_shape_codebook.train_localization import TYPES, SEVERITIES, frozen_inputs, generate, load_points
from robot.experiments.link5_shape_codebook.upstream import shape_modules


@torch.no_grad()
def run(config_path):
    config = read(config_path); root, split, norm, heldout = frozen_inputs(config)
    policy = read(root / 'localization/paired_native/training_policy.json')
    for name, digest in policy['adapter_sources'].items():
        if sha(ROOT / 'robot/experiments/link5_shape_codebook' / name) != digest:raise ValueError('training recipe changed')
    assert torch.cuda.device_count() == 1
    assert float(torch.ones(4, device='cuda').sum()) == 4
    modules, source = shape_modules(); augment = modules['augmentation'].NegativeAugmentation()
    destination = root / 'normal_reference/default_examples'; destination.mkdir(parents=True, exist_ok=True)
    # Find one actual training draw for each reference/type/severity combination.
    stream = np.random.default_rng(0); draws = {}; update = 0
    while len(draws) < 36:
        for sample_index in stream.permutation(4):
            row = split['train'][int(sample_index)]; combination = update % 9
            key = (row['video_id'], TYPES[combination % 3], SEVERITIES[combination // 3])
            draws.setdefault(key, update); update += 1
        if update > 10000:raise ValueError('unable to reconstruct synthetic draw coverage')
    rows = []
    for video in FOUR_REFERENCE_IDS:
        row = next(r for r in split['train'] if r['video_id'] == video)
        xyz, ids = load_points(row, norm); normals = modules['augmentation'].estimate_normals(xyz)
        normal_path = destination / f'{video}_normal.npz'
        np.savez_compressed(normal_path, points_normalized=xyz.cpu().numpy(), sampled_input_indices=ids, normals=normals.cpu().numpy())
        for kind in augment.cfg.types:
            for severity in SEVERITIES if kind in TYPES else (0.01,):
                used = kind in TYPES; draw = draws.get((video, kind, severity))
                seed = 100000 + draw if used else 123
                if used:
                    anomalous, offset, mask = generate(augment, xyz, normals, kind, severity, seed)
                else:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                        anomalous, offset, mask = augment(xyz, normals, atype=kind, severity=severity)
                delta = anomalous - xyz; magnitude = delta.norm(dim=-1)
                key = f'{video}_{kind}_{severity}'
                path = destination / (key + '.npz')
                np.savez_compressed(path, normal=xyz.cpu().numpy(), anomalous=anomalous.cpu().numpy(),
                                    gt_offset=offset.cpu().numpy(), gt_mask=mask.cpu().numpy(), sampled_input_indices=ids)
                rows.append(dict(key=key, video_id=video, frame_id=0, type=kind, severity=severity, seed=seed,
                                 reconstructed_training_draw=draw, used_for_training=used, point_count=len(xyz),
                                 labeled_point_count=int(mask.sum()), labeled_fraction=float(mask.mean()),
                                 coordinate_changed_point_count=int(torch.ne(xyz, anomalous).any(-1).sum()),
                                 actual_max_displacement=float(magnitude.max()),
                                 actual_max_camera_displacement=float(magnitude.max()) * norm['fixed_scale'],
                                 reported_target_max_error=float((xyz-anomalous-offset).abs().max()),
                                 input_sha256=row['input_sha256'], archive=path.name, archive_sha256=sha(path)))
        print('DEFAULT_GENERATOR_REFERENCE_READY', video, len(xyz), flush=True)
    receipt = dict(status='complete', normal_video_ids=list(FOUR_REFERENCE_IDS), normal_frame_ids=[0]*4,
                   normalization=norm, source=source, examples=rows, generator_source_sha256=sha(modules['augmentation'].__file__),
                   source_sha256=sha(__file__), allocation_id=os.environ.get('SLURM_JOB_ID'),
                   model_inference=False, optimizer_updates=0, codebook_updated=False, original_inputs_modified=False,
                   reconstruction='Same GPU implementation, normalized point ordering and draw seeds as current training. Historical minibatches were not saved, so their past byte identity cannot be independently verified.',
                   excluded_modes='Three preview-only moderate-severity modes per reference return unchanged clouds with contradictory restoration labels.')
    write(destination / 'manifest.json', receipt)
    print('LINK5_DEFAULT_GENERATOR_EXAMPLES_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
