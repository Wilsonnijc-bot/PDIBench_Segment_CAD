"""Reuse native CVD depth and cached source masks; filter every original frame."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np

from .prepare_gpu import sha, write
from robot.preprocessing.depth.link5_depth_filter import Config, filter_depth, settings


def filter_queries(valid, xy):
    h, w = valid.shape
    uv = np.rint(xy).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    accepted = np.zeros(len(xy), bool)
    accepted[inside] = valid[uv[inside, 1], uv[inside, 0]]
    return accepted


def process(job):
    entry, root, cache = job
    root, cache = Path(root), Path(cache)
    case = entry['video_id']
    try:
        cfg = Config().validate()
        receipt_path = root / 'shared_inputs' / (case + '.json')
        raw_receipt = json.loads(receipt_path.read_text())
        raw_path = receipt_path.with_suffix('.npz')
        if raw_receipt['raw_sha256'] != sha(raw_path):
            raise ValueError('Raw input checksum mismatch')
        for key in ('mask_sha256', 'source_video_sha256'):
            if raw_receipt[key] != entry[key]:
                raise ValueError('Raw input provenance mismatch: ' + key)
        mask_path = cache / entry['mask']
        if sha(mask_path) != entry['mask_sha256']:
            raise ValueError('Source-mask checksum mismatch')
        native_name = case.rsplit('_', 1)[1] + '_sgd_cvd_hr.npz'
        natives = []
        for native in root.glob('scratch*/' + case + '/mega_sam/outputs_cvd/' + native_name):
            geometry_path = native.parents[2] / 'geometry.json'
            if not geometry_path.exists():
                continue
            geometry = json.loads(geometry_path.read_text())
            if all(geometry.get(k) == entry[k] for k in ('mask_sha256', 'source_video_sha256')):
                if geometry['sha256'] != sha(native):
                    raise ValueError('Receipted native CVD checksum mismatch')
                natives.append((native, geometry))
        if not natives:
            raise ValueError('Exact receipted native CVD depth unavailable; no depth approximation')
        if len({g['sha256'] for _, g in natives}) != 1:
            raise ValueError('Multiple different valid native geometries; investigate provenance')
        native, geometry = natives[0]
        output = root / 'depth_support' / (case + '.npz')
        prior_path = output.with_suffix('.json')
        filter_source = Path(__import__(filter_depth.__module__, fromlist=['filter_depth']).__file__)
        policy = dict(filter_source_sha256=sha(filter_source), adapter_source_sha256=sha(__file__), settings=settings(cfg))
        if output.exists() and prior_path.exists():
            prior = json.loads(prior_path.read_text())
            if (prior['raw_input_sha256'] == raw_receipt['raw_sha256'] and prior['native_depth_sha256'] == geometry['sha256']
                    and prior['policy'] == policy and prior['support_sha256'] == sha(output)):
                return dict(video_id=case, status='complete', reused=True)
            raise ValueError('Prior filtered support differs; preserved for investigation')
        with np.load(mask_path, allow_pickle=False) as archive:
            names = archive['object_names'].tolist()
            source_masks = archive['object_masks'][:, names.index('link5')].astype(bool)
        with np.load(raw_path, allow_pickle=False) as archive:
            xy = archive['tracks_2d']; ids = archive['point_ids']
            raw_poses = archive['camera_poses']; raw_intrinsic = archive['intrinsic']
        with np.load(native, allow_pickle=False) as archive:
            depths = archive['depths']; poses = archive['cam_c2w']; intrinsic = archive['intrinsic']
        if len(depths) != entry['frame_count'] or len(source_masks) != len(depths) or len(xy) != len(depths):
            raise ValueError('All original source frames required')
        if not np.array_equal(poses, raw_poses) or not np.array_equal(intrinsic, raw_intrinsic):
            raise ValueError('Native geometry is not the raw input geometry')
        if intrinsic.shape == (4,):
            fx, fy, cx, cy = intrinsic
            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif intrinsic.shape == (3, 3):
            k = intrinsic
        else:
            raise ValueError('Unexpected intrinsic layout')
        valid = np.empty(depths.shape, bool)
        reasons = np.empty(depths.shape, np.uint8)
        query_support = np.empty(xy.shape[:2], bool)
        query_density = np.full(xy.shape[:2], np.nan, np.float32)
        quality = np.empty(len(depths), bool)
        diagnostics = []
        for t, depth in enumerate(depths):
            support = filter_depth(depth, source_masks[t], k, cfg)
            valid[t] = support['valid']; reasons[t] = support['rejection_reason']
            info = support['info']
            quality[t] = (info['final_valid_pixel_count'] >= cfg.minimum_pixels
                          and info['retained_fraction'] >= cfg.minimum_depth_retained_fraction)
            query_support[t] = filter_queries(valid[t], xy[t]) & quality[t]
            uv = np.rint(xy[t]).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            query_density[t, inside] = support['density_score'][uv[inside, 1], uv[inside, 0]]
            diagnostics.append(dict(frame=t, display_frame=t + 1, qc_passed=bool(quality[t]),
                                    supported_query_count=int(query_support[t].sum()), **info))
        output.parent.mkdir(exist_ok=True)
        temporary = output.with_suffix('.partial.npz')
        if temporary.exists() or output.exists():
            raise ValueError('Existing incomplete support preserved; investigate before retry')
        np.savez_compressed(temporary, valid_pixels=valid, rejection_reasons=reasons,
                            query_support=query_support, query_density=query_density,
                            frame_qc_passed=quality, point_ids=ids)
        temporary.replace(output)
        write(prior_path, dict(video_id=case, status='complete', raw_input_sha256=raw_receipt['raw_sha256'],
              native_depth_path=str(native), native_depth_sha256=geometry['sha256'],
              mask_sha256=entry['mask_sha256'], source_video_sha256=entry['source_video_sha256'],
              support_sha256=sha(output), policy=policy, frame_indices=list(range(len(depths))),
              initialization='frame0 visibility plus refined depth support; no gradient or visible-only fallback',
              availability='raw visibility > 0.5 AND current-frame depth support at both pair endpoints',
              insufficient_frame_policy='cloud QC failure disables depth availability; fewer than 3 pairs carries previous score',
              xyz_modified=False, raw_visibility_modified=False, frames=diagnostics))
        return dict(video_id=case, status='complete', frame0_qc_passed=bool(quality[0]),
                    depth_qc_failed_frames=int((~quality).sum()))
    except Exception as exc:
        return dict(video_id=case, status='failed', error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads((args.manifest or args.root / 'manifest.json').read_text())
    cv2.setNumThreads(1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, [(e, str(args.root), str(args.cache)) for e in manifest['entries']]))
    write(args.root / 'depth_support/summary.json', rows)
    print('DEPTH_SUPPORT_COMPLETE', sum(r['status'] == 'complete' for r in rows), len(rows), flush=True)
    if any(r['status'] != 'complete' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
