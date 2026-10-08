"""Paired clean/deformed localization trials using the upstream real generators.

Two isolated variants share the four frozen frame0 inputs, augmentation stream,
optimizer and explicit point supervision. The only difference is a point-feature
residual at attention output. Existing checkpoints and pipeline gates are untouched.
"""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import normalize, observed_sample, raw_metrics, read, reference_ids, sha, split_path, write
from robot.experiments.link5_shape_codebook.localization_objective import install_point_residual, paired_loss, raw_score
from robot.experiments.link5_shape_codebook.upstream import build_model, codebook_state

TYPES = ('sink', 'concavity', 'bulges')
SEVERITIES = (0.001, 0.01, 0.1)
VARIANTS = ('paired_native', 'paired_point_residual')


def frozen_inputs(config):
    root = Path(config['output'])
    split = read(split_path(root)); norm = read(root / 'link5_normalization.json')
    ids = reference_ids(config)
    if not split.get('reference_only') or len(split['train']) != 4 or split['test'] or set(ids) != {r['video_id'] for r in split['train']}:
        raise ValueError('exact frozen four-frame0 split required')
    if read(root / 'alignment/audit.json')['status'] != 'passed' or read(root / 'sanity/pose_checks.json')['status'] != 'passed':
        raise ValueError('reference alignment and pose gates required')
    heldout = read(root / 'splits/heldout_frame0.json')
    if heldout['training_split_sha256'] != sha(split_path(root)) or {r['video_id'] for r in heldout['test']} != set(split['declared_heldout_video_ids']):
        raise ValueError('heldout coverage/provenance mismatch')
    for row in split['train'] + heldout['test']:
        if row['frame_id'] != 0 or sha(row['input_path']) != row['input_sha256']:
            raise ValueError('frame0 input identity changed')
    return root, split, norm, heldout['test']


def load_points(row, norm):
    with np.load(row['input_path'], allow_pickle=False) as a:
        points, indices = observed_sample(normalize(a['xyz_camera'], norm), 10000, seed=0)
    return torch.from_numpy(points).cuda(), indices


def generate(augment, xyz, normals, kind, severity, seed):
    # Isolate augmentation random draws from FPS/model state; both variants get
    # identical points and labels for the same draw, including random centers.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        anomalous, offset, mask = augment(xyz, normals, atype=kind, severity=severity)
    torch.testing.assert_close(xyz - anomalous, offset, rtol=0, atol=1e-7)
    if not mask.bool().any() or not torch.isfinite(anomalous).all() or float(offset.abs().max()) == 0:
        raise ValueError('synthetic generator produced no real deformation')
    return anomalous, offset, mask


@torch.no_grad()
def prediction(model, xyz):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        offset, logits, aux = model(xyz[None])
    return dict(raw_point_scores=raw_score(offset, logits)[0].cpu().numpy(),
                predicted_offset=offset[0].cpu().numpy(), validity_logits=logits[0].cpu().numpy(),
                points_normalized=xyz.cpu().numpy(), selected_scale=aux['best_scale'].cpu().numpy())


def localization_metrics(clean_scores, scores, mask, offsets, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    mask = np.asarray(mask, bool); outside = ~mask
    target = np.abs(offsets).sum(-1)
    pair_margin = scores[mask] - clean_scores[mask]
    result = dict(positive_count=int(mask.sum()), negative_count=int(outside.sum()), positive_fraction=float(mask.mean()),
                  region_score=float(scores[mask].mean()), clean_region_score=float(clean_scores[mask].mean()),
                  outside_score=float(scores[outside].mean()) if outside.any() else None,
                  matched_region_score_ratio=float(scores[mask].mean() / max(clean_scores[mask].mean(), 1e-8)),
                  region_vs_outside_ratio=float(scores[mask].mean() / max(scores[outside].mean(), 1e-8)) if outside.any() else None,
                  positive_pair_margin_fraction=float((pair_margin > 0).mean()),
                  recall_at_frozen_clean_threshold=float((scores[mask] > threshold).mean()),
                  outside_false_positive_fraction=float((scores[outside] > threshold).mean()) if outside.any() else None,
                  raw_top80_ratio=float(raw_metrics(scores)['raw_mean_top80'] / max(raw_metrics(clean_scores)['raw_mean_top80'], 1e-8)),
                  target_mean_displacement_l1=float(target[mask].mean()),
                  point_auroc=float(roc_auc_score(mask, scores)) if outside.any() else None,
                  point_average_precision=float(average_precision_score(mask, scores)) if outside.any() else None)
    k = min(int(mask.sum()), len(mask))
    result['precision_at_positive_count'] = float(mask[np.argsort(scores)[-k:]].mean())
    return result


def evaluate(model, modules, config, split, norm, heldout, destination, epoch, save_predictions=False):
    model.eval(); augment = modules['augmentation'].NegativeAugmentation()
    clean_predictions = {}; rows = []
    for row in split['train'] + heldout:
        xyz, ids = load_points(row, norm)
        arrays = prediction(model, xyz)
        clean_predictions[row['video_id']] = arrays['raw_point_scores']
        if save_predictions and row['video_id'] == 'LVP_ROBOWM_0001':
            np.savez_compressed(destination / 'predictions/clean_LVP_ROBOWM_0001.npz', **arrays, sampled_input_indices=ids)
    train_scores = np.concatenate([clean_predictions[row['video_id']] for row in split['train']])
    threshold = float(np.quantile(train_scores, 0.995))
    clean_receipts = [dict(video_id=row['video_id'], split='train' if row in split['train'] else 'heldout',
                          false_positive_fraction=float((clean_predictions[row['video_id']] > threshold).mean()),
                          **raw_metrics(clean_predictions[row['video_id']])) for row in split['train'] + heldout]
    for row in heldout:
        xyz, ids = load_points(row, norm)
        normals = modules['augmentation'].estimate_normals(xyz)
        for severity in SEVERITIES:
            for kind in TYPES:
                seed = 123 + SEVERITIES.index(severity) * 10 + TYPES.index(kind)
                anomalous, offset, mask = generate(augment, xyz, normals, kind, severity, seed)
                arrays = prediction(model, anomalous)
                metrics = localization_metrics(clean_predictions[row['video_id']], arrays['raw_point_scores'], mask.cpu().numpy(), offset.cpu().numpy(), threshold)
                rows.append(dict(video_id=row['video_id'], frame_id=0, type=kind, severity=severity, seed=seed, **metrics))
                if save_predictions and row['video_id'] == 'LVP_ROBOWM_0001':
                    np.savez_compressed(destination / f'predictions/{kind}_{severity}_LVP_ROBOWM_0001.npz', **arrays,
                                        normal=xyz.cpu().numpy(), anomalous=anomalous.cpu().numpy(),
                                        gt_offset=offset.cpu().numpy(), gt_mask=mask.cpu().numpy(), sampled_input_indices=ids)
    local = [r for r in rows if r['negative_count'] >= 32 and r['positive_count'] >= 32]
    summary = dict(synthetic_cases=len(rows), localization_cases=len(local),
                   median_point_auroc=float(np.median([r['point_auroc'] for r in local])),
                   median_point_average_precision=float(np.median([r['point_average_precision'] for r in local])),
                   median_region_vs_outside_ratio=float(np.median([r['region_vs_outside_ratio'] for r in local])),
                   median_matched_region_score_ratio=float(np.median([r['matched_region_score_ratio'] for r in local])),
                   median_raw_top80_ratio=float(np.median([r['raw_top80_ratio'] for r in rows])),
                   median_heldout_clean_false_positive_fraction=float(np.median([r['false_positive_fraction'] for r in clean_receipts if r['split'] == 'heldout'])))
    receipt = dict(status='complete', epoch=epoch, fixed_train_clean_score_quantile=0.995, frozen_clean_threshold=threshold,
                   synthetic_generators='exact upstream sink/concavity/bulges; default severities; no fake missing-surface labels',
                   training_inputs=4, heldout_frame0_inputs=len(heldout), synthetic_cases=rows, clean_clouds=clean_receipts,
                   summary=summary, source_sha256=sha(__file__),
                   interpretation='Research localization diagnostic, not the original full-video sanity gate. Gaussian tails outside the labeled region may have small offsets.')
    write(destination / f'evaluation_epoch_{epoch:04d}.json', receipt)
    print('LOCALIZATION_EVALUATION', epoch, summary, flush=True)
    model.train()
    return receipt


def run(config_path, variant, epochs):
    config = read(config_path)
    root, split, norm, heldout = frozen_inputs(config)
    destination = root / 'localization' / variant
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.train.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (destination / 'final.pt').exists() or (destination / 'losses.csv').exists():
        raise ValueError('preserve previous trial; use a separate explicit experiment configuration')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('one allocated CUDA GPU required')
    assert float(torch.ones(8, device='cuda').sum()) == 8
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    cfg = read(root / 'training/effective_config.json')
    cfg['augmentation'] = dict(types=list(TYPES), severities=list(SEVERITIES))
    cfg['augmentation_mode'] = 'paired_localization'
    model, modules, source = build_model(cfg); model.cuda()
    if variant == 'paired_point_residual':
        install_point_residual(model)
    augment = modules['augmentation'].NegativeAugmentation()
    samples = []
    for row in split['train']:
        xyz, ids = load_points(row, norm)
        samples.append((row, xyz, modules['augmentation'].estimate_normals(xyz)))
    # Audit all default generator modes on an actual reference. Preserve the
    # bad-mode finding explicitly; they do not enter either training stream.
    audit = []
    for kind in augment.cfg.types:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(123); torch.cuda.manual_seed_all(123)
            xyz = samples[0][1]
            a, o, m = augment(xyz, samples[0][2], atype=kind, severity=0.01)
        audit.append(dict(type=kind, input_changed=bool(torch.ne(xyz, a).any()),
                          correspondence_max_error=float((xyz - a - o).abs().max()), labeled_points=int(m.sum())))
    write(destination / 'generator_audit.json', dict(rows=audit, selected_types=TYPES, default_severities=SEVERITIES))
    policy = dict(variant=variant, epochs=epochs, optimizer='Adam', learning_rate=0.001, batch_size=1, seed=0,
                  normal_manifest=split['train'], normal_split_sha256=sha(split_path(root)), normalization_sha256=sha(root / 'link5_normalization.json'),
                  augmentation_types=TYPES, severities=SEVERITIES, augmentation_correspondence_verified=True,
                  input_policy='At most10000 observed points, seed0 permutation, no repetition; cached PCA normals in this exact ordering.',
                  objective='L_offset +0.5 L_mask +0.5 L_rank; paired clean zero offsets/mask, region-balanced restoration/mask, score margin0.5*GT displacement.',
                  architecture='original upstream' if variant == 'paired_native' else 'original upstream + frozen point feature residual at attention output',
                  pretrained_backbone_frozen=True, upstream_sources=source, detector_config=cfg,
                  adapter_sources={p.name:sha(p) for p in (Path(__file__), Path(__file__).with_name('localization_objective.py'))},
                  default_missing_modes_excluded='unchanged clouds with nonzero targets violate observed correspondence',
                  old_production_checkpoints_modified=False, original_sanity_gates_modified=False,
                  allocation_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('HOSTNAME'))
    write(destination / 'training_policy.json', policy)
    model.codebook.reset()
    for row, xyz, normals in samples:
        model.update_codebook(xyz[None])
    # Exercise the actual paired objective/backward before any optimizer step.
    xyz, normals = samples[0][1:]
    anomalous, gt_offset, gt_mask = generate(augment, xyz, normals, 'sink', 0.01, 99999)
    clean_offset, clean_logits, _ = model(xyz[None])
    offset, logits, _ = model(anomalous[None])
    probe_loss, probe_parts = paired_loss(clean_offset, clean_logits, offset, logits, gt_offset[None], gt_mask[None])
    if not torch.isfinite(probe_loss):raise ValueError('paired preflight loss nonfinite')
    probe_loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('paired preflight gradient nonfinite')
    model.zero_grad(set_to_none=True)
    write(destination / 'preflight.json', dict(status='passed', optimizer_updates=0, loss=probe_parts,
          actual_backbone=model.encoder.provenance, codebook_sizes=[int(b.size.item()) for b in model.codebook.books],
          device=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda))
    del anomalous, gt_offset, gt_mask, clean_offset, clean_logits, offset, logits, probe_loss
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    print('LINK5_LOCALIZATION_PREFLIGHT_PASSED', variant, [int(b.size.item()) for b in model.codebook.books], torch.cuda.get_device_name(), flush=True)
    starting = time.monotonic(); counts = {}; stream = np.random.default_rng(0); updates = 0
    destination.joinpath('predictions').mkdir(exist_ok=True)
    with (destination / 'losses.csv').open('w', newline='') as log:
        writer = csv.DictWriter(log, fieldnames=['epoch', 'offset', 'mask', 'ranking', 'total', 'seconds']); writer.writeheader()
        for epoch in range(1, epochs + 1):
            parts = []
            for sample_index in stream.permutation(len(samples)):
                row, xyz, normals = samples[int(sample_index)]
                combination = updates % 9
                kind = TYPES[combination % 3]; severity = SEVERITIES[combination // 3]
                anomalous, gt_offset, gt_mask = generate(augment, xyz, normals, kind, severity, 100000 + updates)
                model.update_codebook(xyz[None])
                clean_offset, clean_logits, _ = model(xyz[None])
                offset, logits, _ = model(anomalous[None])
                loss, values = paired_loss(clean_offset, clean_logits, offset, logits, gt_offset[None], gt_mask[None])
                if not torch.isfinite(loss):raise ValueError('nonfinite localization loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('nonfinite localization gradient')
                optimizer.step(); parts.append(values); updates += 1
                key = f'{kind}:{severity}'; counts[key] = counts.get(key, 0) + 1
            means = {key:float(np.mean([p[key] for p in parts])) for key in parts[0]}
            writer.writerow(dict(epoch=epoch, **means, seconds=time.monotonic() - starting)); log.flush()
            if epoch <= 3 or epoch % 100 == 0:
                print('LOCALIZATION_TRAIN', variant, epoch, means, 'seconds', time.monotonic() - starting, flush=True)
            if epoch in (300, epochs):
                receipt = evaluate(model, modules, config, split, norm, heldout, destination, epoch, save_predictions=epoch == epochs)
                payload = dict(model=model.state_dict(), codebook_hash_state=codebook_state(model), config=cfg,
                               variant=variant, epoch=epoch, optimizer=optimizer.state_dict(), policy=policy, normalization=norm,
                               augmentation_counts=counts, optimizer_updates=updates, evaluation_summary=receipt['summary'])
                temporary = destination / f'epoch_{epoch:04d}.{os.getpid()}.tmp'; torch.save(payload, temporary)
                temporary.replace(destination / f'epoch_{epoch:04d}.pt')
    final_path = destination / 'final.pt'
    final_path.write_bytes((destination / f'epoch_{epochs:04d}.pt').read_bytes())
    write(destination / 'completion.json', dict(status='complete', epoch=epochs, variant=variant,
          checkpoint_sha256=sha(final_path), optimizer_updates=updates, augmentation_counts=counts,
          elapsed_seconds=time.monotonic() - starting, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
          normal_split_sha256=sha(split_path(root)), normalization_sha256=sha(root / 'link5_normalization.json'),
          evaluation_summary=receipt['summary'], original_full_video_gate_passed=False))
    frozen_inputs(config)
    print('LINK5_LOCALIZATION_TRIAL_COMPLETE', variant, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--epochs', type=int, default=1500)
    args = parser.parse_args()
    if args.epochs < 1:raise ValueError('positive epochs required')
    run(args.config, args.variant, args.epochs)
