"""Full-sequence shared inputs; cached masks and immutable explicit queries only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import cv2
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_sources(entry, cache, sources, queries_root):
    case = entry['video_id']
    generator, number = case.rsplit('_', 1)
    video = sources / ('LVP' if generator == 'LVP_ROBOWM' else generator) / (number + '.mp4')
    mask = cache / entry['mask']
    initialization = queries_root / case / 'pairs.json'
    for path, expected in ((video, entry['source_video_sha256']), (mask, entry['mask_sha256'])):
        if sha(path) != expected:
            raise ValueError(f'Input SHA-256 mismatch: {path}')
    queries = json.loads(initialization.read_text())
    for key in ('source_video_sha256', 'mask_sha256'):
        if queries[key] != entry[key]:
            raise ValueError(f'Query initialization provenance mismatch: {key}')
    xy = np.asarray(queries['query_points_xy'], np.float32)
    ids = np.asarray(queries['point_ids'], np.int64)
    if xy.shape != (len(ids), 2) or len(ids) < 5 or len(set(ids.tolist())) != len(ids) or not np.isfinite(xy).all():
        raise ValueError('Invalid initialized query coordinates/IDs')
    return video, mask, initialization, xy, ids


def scale_queries(xy, source_hw, grid_hw):
    sh, sw = source_hw
    gh, gw = grid_hw
    scale = np.array([gw / sw, gh / sh], np.float32)
    return np.c_[np.zeros(len(xy), np.float32), xy * scale].astype(np.float32), scale


def prepare(args, entry):
    import torch
    from cotracker.predictor import CoTrackerPredictor
    from infrastructure.shared.inference.cotracker_core import CoTrackerCore
    from infrastructure.shared.inference.mega_sam_wrapper import MegaSamWrapper
    from infrastructure.deformation_detect.worker import isolate_mega

    class ExplicitTracker(CoTrackerCore):
        def _load_model(self, checkpoint):
            # Fail on invalid local weights; no network/hub fallback.
            return CoTrackerPredictor(checkpoint=str(checkpoint)).to(self.device).eval()

        def infer(self, video_input, **kwargs):
            return self._run_queries(video_input, kwargs['queries'])

    case = entry['video_id']
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch / case
    scratch.mkdir(parents=True, exist_ok=True)
    video, mask_path, initialization, xy, ids = validate_sources(entry, args.cache, args.sources, args.queries)
    destination = output / (case + '.npz')
    receipt_path = destination.with_suffix('.json')
    if receipt_path.exists() and destination.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('status') == 'complete' and receipt.get('raw_sha256') == sha(destination)
                and receipt.get('query_initialization_sha256') == sha(initialization)
                and all(receipt.get(k) == entry[k] for k in ('video_id', 'mask_sha256', 'source_video_sha256'))):
            print('SHARED_INPUT_REUSED', case, flush=True)
            return
        raise ValueError('Existing shared inputs have invalid provenance; preserved for investigation')
    started = time.monotonic()
    random.seed(0)
    np.random.seed(0)
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    with np.load(mask_path, allow_pickle=False) as archive:
        names = archive['object_names'].tolist()
        masks_source = archive['object_masks'][:, names.index('link5')].astype(bool)
    if masks_source.shape[0] != entry['frame_count']:
        raise ValueError('Mask cache frame count differs from manifest')
    # No shape-codebook observations, training, depth gating, SAM or VLM calls.
    if not (scratch / 'mega_sam').exists():
        isolate_mega({'config': {'resources': {'mega_sam': str(args.mega_root)}}, 'directory': str(scratch)})
    os.environ['PDI_MEGA_SAM_ROOT'] = str(scratch / 'mega_sam')
    wrapper = MegaSamWrapper(device='cuda')
    native = Path(wrapper.mega_sam_root) / 'outputs_cvd' / f'{video.stem}_sgd_cvd_hr.npz'
    native_receipt = scratch / 'geometry.json'
    if native.exists() and native_receipt.exists():
        prior = json.loads(native_receipt.read_text())
        if prior['sha256'] != sha(native) or any(prior[k] != entry[k] for k in ('mask_sha256', 'source_video_sha256')):
            raise ValueError('Owned native geometry provenance mismatch')
    else:
        if native.exists():
            raise ValueError('Unreceipted native geometry exists; preserved for investigation')
        geometry = wrapper.infer_shared(str(video), masks_source[:, None], cache_dir=None)
        if geometry.frames_count != len(masks_source) or geometry.metadata['cache_hit'] or not native.is_file():
            raise ValueError('Full-sequence CVD geometry missing; no raw-DROID/fallback scores')
        del geometry
        write(native_receipt, dict(sha256=sha(native), **{k: entry[k] for k in ('mask_sha256', 'source_video_sha256')},
                                  code_identity=wrapper._reconstruction_code_identity()))
    with np.load(native, allow_pickle=False) as archive:
        depths = archive['depths']
        poses = archive['cam_c2w']
        intrinsic = archive['intrinsic']
    if intrinsic.shape not in ((4,), (3, 3)):
        raise ValueError('Unexpected native camera intrinsics layout')
    fx, fy, cx, cy = wrapper._parse_intrinsic(intrinsic)
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError('Invalid camera intrinsics')
    count, gh, gw = depths.shape
    if count != len(masks_source) or poses.shape != (count, 4, 4):
        raise ValueError('Native geometry does not cover every original frame')
    pointmaps = wrapper._depth_to_pointmaps(depths, poses, fx, fy, cx, cy)
    del depths
    queries, scale = scale_queries(xy, masks_source.shape[1:], (gh, gw))
    masks = np.empty((count, gh, gw), bool)
    frames = np.empty((count, gh, gw, 3), np.uint8)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    try:
        for index in range(count):
            ok, bgr = cap.read()
            if not ok or bgr.shape[:2] != masks_source.shape[1:]:
                raise ValueError(f'Source decode/resolution mismatch at zero-based frame {index}')
            frames[index] = cv2.resize(bgr, (gw, gh), interpolation=cv2.INTER_LINEAR)[..., ::-1]
            masks[index] = cv2.resize(masks_source[index].astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
        if cap.read()[0]:
            raise ValueError('Extra decoded source frames were omitted by manifest')
    finally:
        cap.release()
    geometry_seconds = time.monotonic() - started
    tracker = ExplicitTracker(checkpoint=str(args.tracker_checkpoint), device='cuda')
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].to('cuda')
    tracks, visibility = tracker._run_queries(tensor, queries)
    # Preserve unfiltered outputs before checking numerical validity.
    np.savez(scratch / f'raw_tracker-{time.time_ns()}.npz', tracks_2d=tracks, visibility=visibility, queries=queries, point_ids=ids)
    temporary = destination.with_suffix('.partial.npz')
    if temporary.exists():
        raise ValueError('A previous partial raw NPZ exists; preserved for investigation')
    np.savez(temporary, pointmaps=pointmaps, tracks_2d=tracks, visibility=visibility, masks=masks,
             queries=queries, point_ids=ids, camera_poses=poses, intrinsic=intrinsic)
    nonfinite = {name: int((~np.isfinite(array)).sum()) for name, array in
                 [('pointmaps', pointmaps), ('tracks_2d', tracks), ('visibility', visibility)]}
    if any(nonfinite.values()):
        write(scratch / 'nonfinite.json', nonfinite)
        raise ValueError(f'Nonfinite raw inputs retained: {nonfinite}')
    temporary.replace(destination)
    receipt = dict(status='complete', **{k: entry[k] for k in ('video_id', 'mask_sha256', 'source_video_sha256')},
        raw_sha256=sha(destination), query_initialization_sha256=sha(initialization), query_ids=ids.tolist(),
        source_hw=list(masks_source.shape[1:]), tracker_pointmap_hw=[gh, gw], source_to_grid_xy_scale=scale.tolist(),
        frame_indices=list(range(count)), fps=fps, query_time=0, track_filtering='none', mask_interpolation='nearest',
        rgb_interpolation='linear', pair_initialization_uses_frame0_only=True, refined_depth_filter_applied=False,
        anchor_gate='baseline_v1_world_Z_gradient_and_visibility_with_visible_only_fallback', seed=0,
        preprocessing='full original source sequence; native CVD world pointmaps; RGB on native pointmap grid',
        models={'tracker_checkpoint': str(args.tracker_checkpoint), 'tracker_sha256': sha(args.tracker_checkpoint),
                'predictor_source_sha256': sha(__import__('inspect').getfile(CoTrackerPredictor)),
                'mega_code': wrapper._reconstruction_code_identity(),
                'mega_checkpoints': {str(p): sha(p) for p in (Path(wrapper.da_ckpt), Path(wrapper.megasam_weights), Path(wrapper.raft_weights))}},
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         numpy=np.__version__, opencv=cv2.__version__, hostname=platform.node(), slurm_job_id=os.getenv('SLURM_JOB_ID')),
        geometry_seconds=geometry_seconds, total_seconds=time.monotonic() - started,
        process_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        process_peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    write(receipt_path, receipt)
    print('SHARED_INPUT_COMPLETE', case, count, receipt['total_seconds'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cache', 'sources', 'queries', 'output', 'scratch', 'mega-root', 'tracker-checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())['entries']
    entry = next(e for e in entries if e['video_id'] == args.video_id)
    prepare(args, entry)


if __name__ == '__main__':
    main()
