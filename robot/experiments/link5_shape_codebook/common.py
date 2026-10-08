"""Coordinate and artifact contracts, independent of model runtimes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODES = ('paper_original','robot_structural')
FOUR_REFERENCE_IDS = ('COSMOS3_0046', 'COSMOS2.5_0044', 'LVP_ROBOWM_0044', 'COSMOS3_0056')


def reference_ids(config):
    ids = config.get('normal_reference', {}).get('video_ids')
    if ids is None:return None  # historical geometry-selected train20 policy
    if len(ids)!=4 or len(set(ids))!=4 or set(ids)!=set(FOUR_REFERENCE_IDS):
        raise ValueError('this refinement requires the four user-selected normal videos')
    if config['normal_reference'].get('frame_id')!=0:
        raise ValueError('normal references must use frame0 only')
    if not set(ids)<={c['id'] for c in config['cases']}:
        raise ValueError('normal reference video absent from frozen cases')
    return list(ids)


def split_path(output):
    current=Path(output)/'splits/normal_reference.json'
    return current if current.exists() else Path(output)/'splits/train20_test25.json'


def normal_manifest_path(training):
    current=Path(training)/'normal_manifest.json'
    return current if current.exists() else Path(training)/'train20_manifest.json'


def mode_folder(output,role,mode):
    if mode not in MODES:raise ValueError('unknown augmentation mode: '+str(mode))
    return Path(output)/(role if mode=='paper_original' else role+'_robot_structural')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def check_se3(transform):
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('pose must be a finite 4x4 matrix')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('invalid homogeneous pose')
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=2e-3) or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=2e-3):
        raise ValueError('pose contains scale, reflection or nonrigid motion')
    return t


def transform_points(points, transform):
    t = check_se3(transform)
    return np.asarray(points, dtype=np.float64) @ t[:3, :3].T + t[:3, 3]


def reference_align(points, t_ref, t_camera_from_link):
    return transform_points(points, check_se3(t_ref) @ np.linalg.inv(check_se3(t_camera_from_link)))


def normalize(points, record):
    scale = float(record['fixed_scale'])
    center = np.asarray(record['fixed_center'])
    if scale <= 0 or not np.isfinite(scale) or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('invalid fixed normalization')
    return ((np.asarray(points) - center) / scale).astype(np.float32)


def observed_sample(points, target=10000, seed=0):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or len(points) < 192:
        raise ValueError('need at least 192 finite observed points for industrial patches')
    # Never invent missing surfaces or repeat points to meet the target.
    ids = np.random.default_rng(seed).choice(len(points), min(target, len(points)), replace=False)
    return points[ids], ids


def cloud_metrics(points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(0)
    values, axes = np.linalg.eigh(np.cov((points - center).T))
    order = np.argsort(values)[::-1]
    axes = axes[:, order]
    dimensions = np.ptp(points, axis=0)
    return dict(centroid=center.tolist(), bbox_min=points.min(0).tolist(), bbox_max=points.max(0).tolist(),
                bbox_dimensions=dimensions.tolist(), principal_axes=axes.tolist(),
                principal_variances=values[order].tolist(), pca_dimensions=np.ptp((points-center) @ axes, axis=0).tolist(),
                valid_point_count=len(points))


def raw_metrics(scores):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('invalid raw point scores')
    descending = np.sort(scores)[::-1]
    return dict(raw_max=float(descending[0]), raw_mean=float(scores.mean()), raw_p95=float(np.quantile(scores, .95)),
                raw_mean_top20=float(descending[:20].mean()), raw_mean_top80=float(descending[:80].mean()),
                raw_mean_top5percent=float(descending[:max(1, int(np.ceil(.05*len(scores))))].mean()),
                scored_point_count=len(scores))
