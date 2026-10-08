"""Unmodified frame0 overlay, cross-video scale audit and label-blind split."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .common import cloud_metrics, observed_sample, read, sha, write, reference_ids
from robot.preprocessing.depth.link5_depth_filter import Config
from .prepare import MASK_POLICY


def pair_distance(a, b):
    d1 = cKDTree(a).query(b, workers=1)[0]
    d2 = cKDTree(b).query(a, workers=1)[0]
    d = np.concatenate([d1, d2])
    return float(np.median(d)), float(np.quantile(d, .95))


def audit(output, config, reference_only=False):
    explicit=reference_ids(config)
    if reference_only and not explicit:raise ValueError('explicit frame0 references required')
    cases=[c for c in config['cases'] if c['id'] in explicit] if reference_only else config['cases']
    frozen=output/'splits/normal_reference.json'
    if not reference_only and frozen.exists() and read(frozen).get('reference_only'):
        raise ValueError('training reference is frozen; collect heldout observations separately')
    if reference_only and frozen.exists():
        split=read(frozen)
        if split.get('reference_only') is not True or {r['video_id'] for r in split['train']}!=set(explicit):
            raise ValueError('existing training split differs from requested reference-only split')
        if split['alignment_audit_sha256']!=sha(output/'alignment/audit.json'):raise ValueError('reference audit changed')
        for row in split['train']:
            if row['frame_id']!=0 or row['input_sha256']!=sha(row['input_path']):raise ValueError('frozen training input changed')
        return split
    rows, clouds, errors = [], [], []
    for case in cases:
        folder = output/'cases'/case['id']/'observations'/'frame_00000'
        try:
            row = read(folder/'observation.json')
            if row.get('mask_erosion_pixels')!=Config().erosion_pixels or row.get('mask_mapping_applied') is not False or row.get('observation_policy')!=MASK_POLICY:
                raise ValueError('observation predates refined Link5 depth policy; prepare a new run')
            with np.load(folder/'input.npz', allow_pickle=False) as archive:
                points = archive['xyz_camera'].copy()
                row['intrinsics_K'] = archive['K'].tolist()
            row.update(cloud_metrics(points), input_sha256=sha(folder/'input.npz'))
            rows.append(row)
            clouds.append(observed_sample(points, 2000)[0])
        except Exception as exc:
            errors.append(dict(video_id=case['id'], error=str(exc)))
    destination = output/'alignment'
    destination.mkdir(parents=True, exist_ok=True)
    if not rows:
        write(destination/'audit.json', dict(status='failed', errors=errors))
        raise ValueError('no fresh frame0 observations')
    dimensions = np.array([r['bbox_dimensions'] for r in rows])
    diagonal = np.linalg.norm(dimensions, axis=1)
    median_diagonal = float(np.median(diagonal))
    scale_ratio = diagonal/median_diagonal
    scale_ok = (scale_ratio >= .65) & (scale_ratio <= 1.5)
    n = len(rows)
    med, p95 = np.zeros((n,n)), np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            med[i,j], p95[i,j] = pair_distance(clouds[i], clouds[j])
            med[j,i], p95[j,i] = med[i,j], p95[i,j]
    centers = np.array([r['centroid'] for r in rows])
    displacement = np.linalg.norm(centers[:,None]-centers[None], axis=-1)
    compatible = ((med <= .05*median_diagonal) & (p95 <= .15*median_diagonal)
                  & (displacement <= .25*median_diagonal) & scale_ok[:,None] & scale_ok[None])
    # Pick a natural overlap medoid; no ICP, scale fitting or pose changes.
    candidate_indices=[i for i,r in enumerate(rows) if explicit is None or r['video_id'] in explicit]
    if not candidate_indices:raise ValueError('requested frame0 references are unavailable')
    anchor = max(candidate_indices,key=lambda i:int(compatible[i,candidate_indices].sum()))
    eligible = np.flatnonzero(compatible[anchor])
    for i, row in enumerate(rows):
        row.update(scale_ratio=float(scale_ratio[i]), scale_failure=not bool(scale_ok[i]),
                   anchor_pair_median=float(med[anchor,i]), anchor_pair_p95=float(p95[anchor,i]),
                   centroid_displacement=float(displacement[anchor,i]), naturally_aligned=bool(compatible[anchor,i]))
    np.savez_compressed(destination/'pairwise.npz', median=med, p95=p95, centroid_distance=displacement,
                        video_ids=np.array([r['video_id'] for r in rows]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    overlay_ids = list(eligible[:8]) or [anchor]
    for i in overlay_ids:
        for ax, (x,y) in zip(axes, [(0,1),(0,2),(1,2)]):
            ax.scatter(clouds[i][:,x], clouds[i][:,y], s=1, alpha=.3, label=rows[i]['video_id'])
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel('camera '+'XYZ'[x]); ax.set_ylabel('camera '+'XYZ'[y])
    for ax,(x,y) in zip(axes,[(0,1),(0,2),(1,2)]):
        center=np.asarray(rows[anchor]['centroid']);basis=np.asarray(rows[anchor]['principal_axes'])
        for channel,color in enumerate(['red','green','blue']):
            direction=basis[:,channel]*.15*median_diagonal
            ax.arrow(center[x],center[y],direction[x],direction[y],color=color,width=.002*median_diagonal)
    axes[-1].legend(fontsize=7)
    fig.suptitle('Frame0 camera coordinates: unmodified observed clouds, no registration')
    fig.tight_layout(); fig.savefig(destination/'frame0_overlay.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(15,5))
    for axis, (name, data) in zip(axes, [('Bounding-box dimensions',dimensions), ('Centroids',centers)]):
        for channel in range(3):
            axis.plot(data[:,channel], '.', label='XYZ'[channel])
        axis.set_title(name+' in original camera units'); axis.legend()
        axis.set_xticks(range(n),[r['video_id'] for r in rows],rotation=90,fontsize=5)
    fig.tight_layout();fig.savefig(destination/'frame0_geometry.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    image=axes[0].imshow(p95/median_diagonal,vmin=0,vmax=.5,cmap='viridis')
    axes[0].set_title('Pairwise symmetric p95 / median bbox diagonal');fig.colorbar(image,ax=axes[0])
    axes[1].plot([r['valid_point_count'] for r in rows],'.');axes[1].set_title('Observed valid point count')
    axes[2].plot([r['mask_area'] for r in rows],'.');axes[2].set_title('Fresh guarded source-mask area')
    fig.tight_layout();fig.savefig(destination/'frame0_quality.png',dpi=180);plt.close(fig)
    requested_indices=[i for i,r in enumerate(rows) if explicit and r['video_id'] in explicit]
    passed = len(rows)==len(cases) and not errors and (
        len(requested_indices)==4 and all(compatible[anchor,i] for i in requested_indices)
        if explicit else len(eligible)>=20)
    report = dict(status='passed' if passed else 'failed', rows=rows, errors=errors,
                  natural_cluster_size=len(eligible), anchor_video=rows[anchor]['video_id'],
                  median_bbox_diagonal=median_diagonal, frame0_foundationpose_applied=False,
                  thresholds=dict(scale_ratio=[.65,1.5], pair_median_diagonal=.05, pair_p95_diagonal=.15, centroid_diagonal=.25),
                  threshold_policy='declared geometry-only trial QC; no labels or anomaly scores',
                  requested_normal_video_ids=explicit,normal_reference_frame_id=0,
                  rejected_candidates=[r['video_id'] for r in rows if not r['naturally_aligned']])
    write(destination/'audit.json',report)
    if not passed:
        raise ValueError(f'frame0 alignment gate failed: {len(rows)}/{len(cases)} clouds; requested normals must naturally overlap; inspect overlay')
    # Greedy coverage in visibility metrics within the naturally aligned cluster.
    # Start with medoid, then retain normal partial-view diversity without labels.
    visibility = np.array([[r['mask_area'], r['valid_point_count'], r['mask_bbox_coverage']] for r in rows])
    visibility = (visibility-visibility.mean(0))/np.maximum(visibility.std(0),1e-8)
    selected = ([anchor]+[next(i for i,r in enumerate(rows) if r['video_id']==video) for video in explicit
                          if video!=rows[anchor]['video_id']]) if explicit else [anchor]
    while len(selected)<(len(explicit) if explicit else 20):
        choices = [i for i in eligible if i not in selected]
        nearest = np.linalg.norm(visibility[choices,None]-visibility[selected][None],axis=-1).min(1)
        selected.append(choices[int(nearest.argmax())])
    train = [rows[i] for i in selected]
    heldout = [r for i,r in enumerate(rows) if i not in selected]
    path=output/'splits'/('normal_reference.json' if explicit else 'train20_test25.json')
    write(path,dict(class_name='link5', train=train, test=heldout,normal_reference_frame_id=0,
          reference_only=reference_only, declared_heldout_video_ids=[c['id'] for c in config['cases'] if c['id'] not in (explicit or [])],
          requested_normal_video_ids=explicit,
          policy='four user-selected frame0 observations only; remaining observations held out' if explicit else '20 naturally overlapping frame0 observations with visibility diversity; remaining25 primary heldout',
          alignment_audit_sha256=sha(destination/'audit.json'), per_sample_pose_normalization=False))
    print('LINK5_FRAME0_ALIGNMENT_PASSED',len(eligible),flush=True)


def collect_heldout(output,config):
    """Add test inputs separately, preserving the hashed training split and audit."""
    audit(output,config,reference_only=True)
    split=read(output/'splits/normal_reference.json');rows=[]
    for video in split['declared_heldout_video_ids']:
        folder=output/'cases'/video/'observations/frame_00000'
        row=read(folder/'observation.json')
        if row['frame_id']!=0 or row['video_id']!=video or row.get('observation_policy')!=MASK_POLICY or row.get('mask_mapping_applied') is not False or row.get('mask_erosion_pixels')!=Config().erosion_pixels:
            raise ValueError('heldout input policy mismatch: '+video)
        with np.load(folder/'input.npz',allow_pickle=False) as archive:
            row.update(cloud_metrics(archive['xyz_camera']),input_sha256=sha(folder/'input.npz'))
        rows.append(row)
    receipt=dict(status='complete',test=rows,training_split_sha256=sha(output/'splits/normal_reference.json'),
                 policy='heldout frame0 observations only; cannot modify the frozen normal reference')
    write(output/'splits/heldout_frame0.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--reference-only',action='store_true')
    args=parser.parse_args();config=read(args.config)
    audit(Path(config['output']),config,reference_only=args.reference_only)


if __name__=='__main__':main()
