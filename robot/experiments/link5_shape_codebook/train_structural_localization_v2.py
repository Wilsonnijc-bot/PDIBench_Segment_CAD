"""Paired localization training with only our axial-strain/beam-bending generator.

Four frozen frame0 clouds supply every optimizer and codebook input. Selected
deformed video frames guide the shape distribution and remain outside training.
"""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import ROOT, raw_metrics, read, sha, split_path, write
from robot.experiments.link5_shape_codebook.localization_objective import install_point_residual, paired_loss
from robot.experiments.link5_shape_codebook.robot_structural import DEFAULTS, RobotStructuralAugmentation
from robot.experiments.link5_shape_codebook.train_localization import frozen_inputs, generate, load_points, localization_metrics, prediction
from robot.experiments.link5_shape_codebook.upstream import build_model, codebook_state

TYPES = ('shortening', 'lengthening', 'mild_bend', 'strong_bend')
VARIANTS = ('structural_native', 'structural_point_residual', 'structural_geometry_head')


def setup(config, guide_manifest, policy_path):
    root, split, norm, heldout = frozen_inputs(config)
    guide = read(guide_manifest); distribution = read(policy_path)
    expected = {'COSMOS2.5_0018':(48, 'bending'), 'COSMOS2.5_0030':(80, 'bending'), 'LVP_ROBOWM_0005':(48, 'shortening')}
    if {r['video_id']:(r['frame_id'],r['guide_family']) for r in guide['guides']} != expected:
        raise ValueError('exact three user-selected deformation guides required')
    if any(r['training_use'] != 'visual guide only; no normal-codebook or direct abnormal optimization input' for r in guide['guides']):
        raise ValueError('guides must remain outside optimization')
    if distribution['status'] != 'frozen' or distribution['guide_manifest_sha256'] != sha(guide_manifest):
        raise ValueError('freeze a guide-backed deformation distribution before training')
    if distribution['generator_source_sha256'] != sha(ROOT / 'robot/experiments/link5_shape_codebook/robot_structural.py'):
        raise ValueError('structural generator changed after guide review')
    for guide_row in guide['guides']:
        receipt = read(root / 'guides/cases' / guide_row['video_id'] / 'status.json')
        if receipt['status']!='complete' or receipt['guide']!=guide_row:
            raise ValueError('all exact guiding clouds must be complete before training')
        for observation in receipt['observations']:
            if sha(observation['input_path'])!=observation['input_sha256']:
                raise ValueError('guide cloud identity changed')
    geometry = read(root / 'augmentation/robot_structural_geometry.json')
    validation = read(root / 'augmentation/robot_structural_validation.json')
    if validation['status'] != 'passed' or validation['augmentation_source_sha256'] != distribution['generator_source_sha256']:
        raise ValueError('structural generator numerical validation required')
    if geometry['split_sha256'] != sha(split_path(root)) or geometry['normalization_sha256'] != sha(root / 'link5_normalization.json'):
        raise ValueError('fixed longitudinal geometry must use the frozen normal reference')
    pose = read(root / 'sanity/pose_checks.json')
    if split['alignment_audit_sha256'] != sha(root / 'alignment/audit.json') or pose['normalization_sha256'] != sha(root / 'link5_normalization.json'):
        raise ValueError('reference alignment/pose provenance changed')
    guide_ids = set(expected)
    independent = [r for r in heldout if r['video_id'] not in guide_ids]
    if len(independent) != 39:raise ValueError('exclude the two calibration-guide video identities from independent heldout summaries')
    return root, split, norm, independent, geometry, distribution


def evaluation_grid(distribution):
    result = [('shortening_5pct', 'shortening', dict(alpha=.95, affected_fraction=.8)),
              ('shortening_25pct', 'shortening', dict(alpha=.75, affected_fraction=.8)),
              ('lengthening_5pct', 'lengthening', dict(alpha=1.05, affected_fraction=.8)),
              ('lengthening_25pct', 'lengthening', dict(alpha=1.25, affected_fraction=.8))]
    for angle in distribution['evaluation_bend_angles_degrees']:
        for direction in (0, np.pi / 2):
            result.append((f'bend_{angle:g}deg_{direction:.3f}', 'bending',
                           dict(angle_degrees=angle, start_fraction=.2, direction_angle=direction)))
    return result


def acceptance(rows, clean_receipts, summary):
    """Predeclared research gate, separate from the original full-video gate."""
    clean = [r['false_positive_fraction'] for r in clean_receipts if r['split']=='heldout']
    strong = [r for r in rows if r['example'] in ('shortening_25pct','lengthening_25pct') or
              (r['type']=='bending' and r['parameters']['angle_degrees']>=35)]
    fractions = {}
    for kind in ('shortening','lengthening','bending'):
        cases = [r for r in strong if r['type']==kind]
        fractions[kind] = float(np.mean([r['raw_top80_ratio']>=1.2 and r['region_vs_outside_ratio'] is not None and r['region_vs_outside_ratio']>=1.2
                                         and r['matched_region_score_ratio']>=1.2 for r in cases]))
    checks = dict(median_localization_auroc=summary['median_point_auroc']>=.8,
                  median_clean_false_positive=bool(np.median(clean)<=.02),
                  p95_clean_false_positive=bool(np.quantile(clean,.95)<=.1),
                  strong_separation_each_family=all(v>=.8 for v in fractions.values()))
    return dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
                strong_family_pass_fractions=fractions, p95_heldout_clean_false_positive=float(np.quantile(clean,.95)),
                thresholds=dict(median_point_auroc=.8, median_clean_false_positive=.02, p95_clean_false_positive=.1,
                                strong_raw_top80_ratio=1.2, strong_region_outside_ratio=1.2,
                                strong_matched_clean_region_ratio=1.2, passing_fraction_each_strong_family=.8),
                production_gate_changed=False, interpretation='Research synthetic sensitivity/localization gate; real guide frames have no dense defect GT.')


@torch.no_grad()
def evaluate(model, split, norm, heldout, generator, distribution, destination, epoch, final):
    model.eval(); clean = {}; rows = []
    if hasattr(model, 'reference_exclusion'):model.reference_exclusion = None
    for row in split['train'] + heldout:
        xyz, indices = load_points(row, norm); arrays = prediction(model, xyz)
        clean[row['video_id']] = arrays['raw_point_scores']
        if final and row['video_id'] == 'LVP_ROBOWM_0001':
            np.savez_compressed(destination / 'predictions/clean_LVP_ROBOWM_0001.npz', **arrays, sampled_input_indices=indices)
    threshold = float(np.quantile(np.concatenate([clean[r['video_id']] for r in split['train']]), .995))
    clean_receipts = [dict(video_id=r['video_id'], split='train' if r in split['train'] else 'heldout',
                          false_positive_fraction=float((clean[r['video_id']] > threshold).mean()), **raw_metrics(clean[r['video_id']]))
                      for r in split['train'] + heldout]
    for row in heldout:
        xyz, indices = load_points(row, norm)
        for name, kind, parameters in evaluation_grid(distribution):
            result = generator.axial(xyz, **parameters) if kind != 'bending' else generator.bend(xyz, **parameters)
            anomalous, offset, mask = result
            torch.testing.assert_close(xyz - anomalous, offset, rtol=0, atol=1e-7)
            arrays = prediction(model, anomalous)
            metrics = localization_metrics(clean[row['video_id']], arrays['raw_point_scores'], mask.cpu().numpy(), offset.cpu().numpy(), threshold)
            rows.append(dict(video_id=row['video_id'], frame_id=0, type=kind, example=name, parameters=parameters, **metrics))
            if final and row['video_id'] == 'LVP_ROBOWM_0001':
                np.savez_compressed(destination / f'predictions/{name}_LVP_ROBOWM_0001.npz', **arrays,
                                    normal=xyz.cpu().numpy(), anomalous=anomalous.cpu().numpy(),
                                    gt_offset=offset.cpu().numpy(), gt_mask=mask.cpu().numpy(), sampled_input_indices=indices)
    local = [r for r in rows if r['positive_count'] >= 32 and r['negative_count'] >= 32]
    if not local:raise ValueError('no spatial localization cases in evaluation')
    summary = dict(synthetic_cases=len(rows), localization_cases=len(local),
                   median_point_auroc=float(np.median([r['point_auroc'] for r in local])),
                   median_point_average_precision=float(np.median([r['point_average_precision'] for r in local])),
                   median_region_vs_outside_ratio=float(np.median([r['region_vs_outside_ratio'] for r in local])),
                   median_matched_region_score_ratio=float(np.median([r['matched_region_score_ratio'] for r in local])),
                   median_raw_top80_ratio=float(np.median([r['raw_top80_ratio'] for r in rows])),
                   median_heldout_clean_false_positive_fraction=float(np.median([r['false_positive_fraction'] for r in clean_receipts if r['split']=='heldout'])))
    receipt = dict(status='complete', epoch=epoch, training_inputs=4, independent_heldout_frame0_inputs=len(heldout),
                   excluded_calibration_video_ids=['COSMOS2.5_0018','COSMOS2.5_0030','LVP_ROBOWM_0005'],
                   generator='our shortening/lengthening/circular-arc bending only', fixed_train_clean_quantile=.995,
                   frozen_clean_threshold=threshold, synthetic_cases=rows, clean_clouds=clean_receipts, summary=summary,
                   original_full_video_gate_passed=False,
                   acceptance=acceptance(rows, clean_receipts, summary),
                   interpretation='Controlled synthetic localization diagnostic. Guide frames are not dense labeled validation data.')
    write(destination / f'evaluation_epoch_{epoch:04d}.json', receipt)
    print('STRUCTURAL_LOCALIZATION_EVALUATION', epoch, summary, flush=True)
    model.train();return receipt


def run(config_path, guide_manifest, distribution_path, variant, epochs):
    config = read(config_path)
    root, split, norm, heldout, geometry, distribution = setup(config, guide_manifest, distribution_path)
    destination = root / 'structural_localization' / 'report_fix_v2' / variant; destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.train.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (destination / 'final.pt').exists() or (destination / 'losses.csv').exists():raise ValueError('never overwrite a previous training attempt')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('one allocated CUDA GPU required')
    assert float(torch.ones(4, device='cuda').sum()) == 4
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    cfg = read(root / 'training_robot_structural/effective_config.json')
    settings = {**DEFAULTS, **distribution['settings']}
    cfg['augmentation'] = dict(types=list(TYPES), robot_structural=settings, structural_geometry=geometry)
    cfg['augmentation_mode'] = 'paired_robot_structural'
    model, modules, source = build_model(cfg); model.cuda()
    if variant == 'structural_point_residual':install_point_residual(model)
    generator = RobotStructuralAugmentation(geometry, settings)
    samples = [(row, *load_points(row, norm)) for row in split['train']]
    if variant == 'structural_geometry_head':
        from robot.experiments.link5_shape_codebook.structural_geometry_head import wrap_geometry_model
        model = wrap_geometry_model(model, samples, geometry)
    sources = [ROOT / 'robot/experiments/link5_shape_codebook' / n for n in
               ('train_structural_localization_v2.py','train_localization.py','robot_structural.py','localization_objective.py','structural_geometry_head.py')]
    policy = dict(variant=variant, epochs=epochs, optimizer='Adam', learning_rate=.001, batch_size=1, seed=0,
                  normal_manifest=split['train'], normal_split_sha256=sha(split_path(root)), normalization_sha256=sha(root / 'link5_normalization.json'),
                  augmentation_types=TYPES, augmentation_distribution=distribution, guide_manifest=read(guide_manifest),
                  guide_manifest_sha256=sha(guide_manifest), distribution_sha256=sha(distribution_path),
                  guide_clouds_used_for_optimization=False, later_frames_in_normal_codebook=False,
                  objective='paired clean zero offsets/mask; balanced deformed/background restoration and BCE; raw-score margin 0.5*GT displacement',
                  architecture={'structural_native':'original upstream', 'structural_point_residual':'original upstream + frozen point feature residual at attention output',
                                'structural_geometry_head':'original upstream network + learned geometry-conditioned restoration/probability head; fixed normal XYZ, nearest normal geometry and whole-link profile'}[variant],
                  auxiliary_reference_policy='all four frozen normals; exclude current source from geometric-neighbor queries during training' if variant=='structural_geometry_head' else None,
                  pretrained_backbone_frozen=True, upstream_sources=source, detector_config=cfg,
                  adapter_sources={p.name:sha(p) for p in sources}, allocation_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('HOSTNAME'),
                  execution='cached four observed point arrays on GPU; native batch1, no unused PCA normal calculation',
                  independent_heldout_count=len(heldout), old_checkpoints_and_sanity_gates_modified=False)
    policy['acceptance_thresholds'] = dict(median_point_auroc=.8, median_clean_false_positive=.02, p95_clean_false_positive=.1,
        strong_raw_top80_ratio=1.2, strong_region_outside_ratio=1.2, strong_matched_clean_region_ratio=1.2,
        passing_fraction_each_strong_family=.8)
    write(destination / 'training_policy.json', policy)
    model.codebook.reset()
    for row, xyz, indices in samples:model.update_codebook(xyz[None])
    xyz = samples[0][1]
    if hasattr(model, 'reference_exclusion'):model.reference_exclusion = 0
    anomalous, gt_offset, gt_mask = generate(generator, xyz, None, 'shortening', None, 99999)
    clean_offset, clean_logits, _ = model(xyz[None]); offset, logits, _ = model(anomalous[None])
    probe, values = paired_loss(clean_offset, clean_logits, offset, logits, gt_offset[None], gt_mask[None])
    if not torch.isfinite(probe):raise ValueError('nonfinite paired structural probe')
    probe.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite paired structural gradient')
    model.zero_grad(set_to_none=True)
    write(destination / 'preflight.json', dict(status='passed', optimizer_updates=0, loss=values,
          actual_backbone=model.encoder.provenance, codebook_sizes=[int(b.size.item()) for b in model.codebook.books],
          device=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda))
    del anomalous, gt_offset, gt_mask, clean_offset, clean_logits, offset, logits, probe
    print('STRUCTURAL_LOCALIZATION_PREFLIGHT_PASSED', variant, flush=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    (destination / 'predictions').mkdir(exist_ok=True); (destination / 'training_examples').mkdir(exist_ok=True)
    starting = time.monotonic(); stream = np.random.default_rng(0); updates = 0; counts = {}; examples = []
    with (destination / 'losses.csv').open('w', newline='') as log:
        writer = csv.DictWriter(log, fieldnames=['epoch','offset','mask','ranking','total','seconds']); writer.writeheader()
        for epoch in range(1, epochs+1):
            parts = []
            for sample_index in stream.permutation(4):
                row, xyz, indices = samples[int(sample_index)]; kind = TYPES[updates % 4]; seed = 100000+updates
                if hasattr(model, 'reference_exclusion'):model.reference_exclusion = int(sample_index)
                anomalous, gt_offset, gt_mask = generate(generator, xyz, None, kind, None, seed)
                parameters = dict(generator.last_parameters)
                name = row['video_id'] + '_' + kind
                if not (destination / 'training_examples' / (name+'.npz')).exists():
                    np.savez_compressed(destination / 'training_examples' / (name+'.npz'), normal=xyz.cpu().numpy(),
                                        anomalous=anomalous.cpu().numpy(), gt_offset=gt_offset.cpu().numpy(), gt_mask=gt_mask.cpu().numpy(),
                                        sampled_input_indices=indices)
                    examples.append(dict(video_id=row['video_id'], frame_id=0, type=kind, parameters=parameters, seed=seed,
                                         optimizer_update=updates+1, source_input_sha256=row['input_sha256'], archive=name+'.npz',
                                         archive_sha256=sha(destination / 'training_examples' / (name+'.npz'))))
                    write(destination / 'training_examples/manifest.json', dict(status='actual saved optimizer inputs', examples=examples))
                model.update_codebook(xyz[None])
                clean_offset, clean_logits, _ = model(xyz[None]); offset, logits, _ = model(anomalous[None])
                loss, values = paired_loss(clean_offset, clean_logits, offset, logits, gt_offset[None], gt_mask[None])
                if not torch.isfinite(loss):raise ValueError('nonfinite structural loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite structural gradient')
                optimizer.step(); parts.append(values); updates+=1; counts[kind]=counts.get(kind,0)+1
            means = {k:float(np.mean([v[k] for v in parts])) for k in parts[0]}
            writer.writerow(dict(epoch=epoch, **means, seconds=time.monotonic()-starting)); log.flush()
            if epoch<=3 or epoch%100==0:print('STRUCTURAL_LOCALIZATION_TRAIN', variant, epoch, means, 'seconds', time.monotonic()-starting, flush=True)
            if epoch in (100, 300, epochs):
                pre_payload = dict(model=model.state_dict(), codebook_hash_state=codebook_state(model), config=cfg, variant=variant, epoch=epoch,
                    optimizer=optimizer.state_dict(), policy=policy, normalization=norm, augmentation_counts=counts, optimizer_updates=updates)
                pre_path = destination / f'epoch_{epoch:04d}.{os.getpid()}.pre_eval.tmp'; torch.save(pre_payload, pre_path)
                pre_path.replace(destination / f'epoch_{epoch:04d}.pt')
                receipt = evaluate(model, split, norm, heldout, generator, distribution, destination, epoch, epoch==epochs)
                payload = dict(model=model.state_dict(), codebook_hash_state=codebook_state(model), config=cfg, variant=variant, epoch=epoch,
                               optimizer=optimizer.state_dict(), policy=policy, normalization=norm, augmentation_counts=counts,
                               optimizer_updates=updates, evaluation_summary=receipt['summary'])
                path = destination / f'epoch_{epoch:04d}.{os.getpid()}.tmp'; torch.save(payload, path)
                path.replace(destination / f'epoch_{epoch:04d}.pt')
    final = destination / 'final.pt'; final.write_bytes((destination / f'epoch_{epochs:04d}.pt').read_bytes())
    setup(config, guide_manifest, distribution_path)
    write(destination / 'completion.json', dict(status='complete', epoch=epochs, variant=variant, checkpoint_sha256=sha(final), optimizer_updates=updates,
          augmentation_counts=counts, elapsed_seconds=time.monotonic()-starting, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
          evaluation_summary=receipt['summary'], original_full_video_gate_passed=False,
          research_acceptance=receipt['acceptance'],
          normal_split_sha256=sha(split_path(root)), normalization_sha256=sha(root / 'link5_normalization.json')))
    print('LINK5_STRUCTURAL_LOCALIZATION_COMPLETE', variant, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--guide-manifest', type=Path, required=True)
    parser.add_argument('--distribution', type=Path, required=True); parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--epochs', type=int, default=1500)
    args = parser.parse_args()
    if args.epochs<1:raise ValueError('positive epochs required')
    torch.set_num_threads(6)
    run(args.config, args.guide_manifest, args.distribution, args.variant, args.epochs)
