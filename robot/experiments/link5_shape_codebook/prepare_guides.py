"""Prepare exact user-selected deformation guides without changing training data.

The native guarded SAM track and full-sequence MegaSAM CVD reconstruction supply
frame0 plus the exact requested frame. Both receive the current Link5 filter.
These observations are visual calibration guides, never normal codebook inputs.
"""
import argparse
import fcntl
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import ROOT, read, sha, split_path, write
from robot.preprocessing.depth.link5_depth_filter import Config, settings
from robot.experiments.link5_shape_codebook.prepare import MASK_POLICY, native_mask, save_observation


def run(config_path, manifest_path, video_id):
    config = read(config_path); root = Path(config['output'])
    manifest = read(manifest_path)
    rows = [r for r in manifest['guides'] if r['video_id'] == video_id]
    if len(rows) != 1:raise ValueError('one exact guide required')
    guide = rows[0]
    if guide['training_use'] != 'visual guide only; no normal-codebook or direct abnormal optimization input':
        raise ValueError('guide cannot become an optimization example')
    case = dict(id=video_id, video=guide['video'], video_sha256=guide['video_sha256'])
    if sha(case['video']) != case['video_sha256']:raise ValueError('guide video identity changed')
    folder = root / 'guides/cases' / video_id; folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(config, root, folder, guide, case)


def run_locked(config, root, folder, guide, case):
    frozen = {str(p):sha(p) for p in (split_path(root), root / 'link5_normalization.json')}
    split = read(split_path(root))
    frozen.update({r['input_path']:sha(r['input_path']) for r in split['train']})
    status_path = folder / 'status.json'
    if status_path.exists() and read(status_path).get('status') == 'complete':
        receipt = read(status_path)
        if receipt['guide'] != guide or receipt['mask_policy'] != MASK_POLICY:
            raise ValueError('completed guide identity changed')
        for row in receipt['observations']:
            if sha(row['input_path']) != row['input_sha256']:raise ValueError('completed guide cloud changed')
        print('LINK5_GUIDE_RETAINED', case['id'], flush=True);return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('one allocated GPU required')
    assert float(torch.ones(4, device='cuda').sum()) == 4
    started = time.monotonic()
    status = dict(status='running', guide=guide, mask_policy=MASK_POLICY,
                  allocation_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('HOSTNAME'),
                  normal_reference_modified=False, direct_abnormal_training=False)
    write(status_path, status)
    try:
        # Existing current-round tracks are reusable after the original guard
        # validation. A newly supplied video gets its own guarded SAM attempt.
        existing = root / 'cases' / case['id']
        mask_folder = existing if (existing / 'masking').is_dir() else folder
        mask_path = native_mask(config, case, mask_folder)
        with np.load(mask_path, allow_pickle=False) as a:
            masks = a['object_masks'][:, a['object_names'].tolist().index('link5')].astype(bool)
        if len(masks) != guide['source_frame_count']:raise ValueError('guide mask track is not the full source video')
        work = folder / 'geometry-work'; work.mkdir(exist_ok=True)
        from infrastructure.deformation_detect.worker import isolate_mega
        if not (work / 'mega_sam').exists():isolate_mega(dict(config=config, directory=str(work)))
        else:os.environ['PDI_MEGA_SAM_ROOT'] = str(work / 'mega_sam')
        from infrastructure.shared.inference.mega_sam_wrapper import MegaSamWrapper
        wrapper = MegaSamWrapper(device='cuda')
        native = Path(wrapper.mega_sam_root) / 'outputs_cvd' / f"{Path(case['video']).stem}_sgd_cvd_hr.npz"
        receipt_path = folder / 'full_sequence_geometry.json'
        if native.exists() and receipt_path.exists():
            prior = read(receipt_path)
            if prior['source_sha256'] != case['video_sha256'] or prior['mask_sha256'] != sha(mask_path) or prior['native_cvd_sha256'] != sha(native):
                raise ValueError('owned guide CVD provenance mismatch')
        else:
            geometry = wrapper.infer_shared(case['video'], masks[:, None], cache_dir=None)
            if geometry.metadata['cache_hit'] or geometry.frames_count != len(masks) or not native.is_file():
                raise ValueError('full-sequence CVD required; raw depth fallback forbidden')
            write(receipt_path, dict(status='complete', frame_count=len(masks), source_sha256=case['video_sha256'],
                  mask_sha256=sha(mask_path), native_cvd_sha256=sha(native), geometry_metadata=geometry.metadata,
                  reconstruction_code_identity=wrapper._reconstruction_code_identity()))
            del geometry
        with np.load(native, allow_pickle=False) as a:
            depths = a['depths'].copy(); intrinsic = a['intrinsic'].copy(); poses = a['cam_c2w'].copy()
        if len(depths) != len(masks) or len(poses) != len(masks):raise ValueError('guide geometry coverage mismatch')
        if intrinsic.shape not in ((4,), (3, 3)):raise ValueError('unknown CVD intrinsic layout')
        fx, fy, cx, cy = wrapper._parse_intrinsic(intrinsic)
        k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        indices = {0, guide['frame_id']}; cap = cv2.VideoCapture(case['video']); observations = []
        index = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:break
                if index in indices:
                    depth = depths[index].astype(np.float32); h,w = depth.shape
                    rgb = cv2.resize(bgr, (w,h), interpolation=cv2.INTER_LINEAR)[..., ::-1].copy()
                    row = save_observation(case, folder, index, rgb, depth, k, masks[index], poses[index])
                    row['input_sha256'] = sha(row['input_path']); observations.append(row)
                index += 1
        finally:cap.release()
        if index != len(masks) or {r['frame_id'] for r in observations} != indices:
            raise ValueError('exact requested frame was not decoded and retained')
        for path, digest in frozen.items():
            if sha(path) != digest:raise ValueError('frozen normal input changed during guide preparation')
        write(folder / 'observations.json', observations)
        status.update(status='complete', observations=observations, mask_source=str(mask_path), mask_sha256=sha(mask_path),
                      full_sequence_frame_count=len(masks), filter_settings=settings(Config()),
                      preprocessing_source_sha256={p.name:sha(p) for p in (ROOT / 'robot/experiments/link5_shape_codebook/prepare_guides.py',
                                                   ROOT / 'robot/preprocessing/depth/link5_depth_filter.py')},
                      normal_inputs_sha256=frozen, elapsed_seconds=time.monotonic() - started)
        print('LINK5_GUIDE_COMPLETE', case['id'], guide['frame_id'], flush=True)
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:write(status_path, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    run(args.config, args.manifest, args.video_id)
