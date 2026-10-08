"""Export display samples of existing CVD geometry for the previous replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from infrastructure.shared.replay.rigidity_replay import build_scene_context
from .prepare_gpu import sha, write


def process(job):
    root, case = Path(job[0]), job[1]
    raw_path = root / 'shared_inputs' / (case + '.npz')
    support_path = root / 'depth_support' / (case + '.npz')
    receipt = json.loads(raw_path.with_suffix('.json').read_text())
    support_receipt = json.loads(support_path.with_suffix('.json').read_text())
    if sha(raw_path) != receipt['raw_sha256'] or sha(support_path) != support_receipt['support_sha256']:
        raise ValueError('Replay input checksum mismatch: ' + case)
    with np.load(raw_path, allow_pickle=False) as archive:
        pointmaps = archive['pointmaps']; poses = archive['camera_poses']; intrinsic = archive['intrinsic']
        masks = archive['masks']
    with np.load(support_path, allow_pickle=False) as archive:
        valid = archive['valid_pixels']; qc = archive['frame_qc_passed']
    if valid.shape != pointmaps.shape[:3] or masks.shape != valid.shape:
        raise ValueError('Replay cloud grids disagree')
    valid &= qc[:, None, None]
    # Remove rejected Link5 pixels from the context as well as the bright cloud.
    # Only this display copy changes; saved pointmaps and score inputs stay intact.
    pointmaps[masks & ~valid] = np.nan
    video = root / 'report/videos' / (case + '.mp4')
    if sha(video) != receipt['source_video_sha256']:
        raise ValueError('Replay source checksum mismatch')
    cv2.setNumThreads(1)
    clouds, colors, scene, scene_colors, bounds = build_scene_context(
        valid[:, None], pointmaps, poses, video, 900)
    payload = dict(clouds=clouds, cloud_colors=colors, scene_clouds=scene,
                   scene_colors=scene_colors, scene_bounds=bounds,
                   camera_poses=poses.tolist(), intrinsic=intrinsic.tolist(),
                   image_hw=list(pointmaps.shape[1:3]), robot_clouds=[[] for _ in clouds])
    destination = root / 'replay_context' / (case + '.json.gz')
    destination.parent.mkdir(exist_ok=True)
    with gzip.open(destination, 'wt', encoding='utf-8', compresslevel=3) as stream:
        json.dump(payload, stream, separators=(',', ':'), allow_nan=False)
    write(destination.with_suffix('.receipt.json'), dict(video_id=case, frames=len(clouds),
          raw_input_sha256=receipt['raw_sha256'], support_sha256=support_receipt['support_sha256'],
          source_video_sha256=receipt['source_video_sha256'], context_sha256=sha(destination),
          sampling='Existing build_scene_context, seed17, up to2700 Link5 /900 context points per frame',
          depth_filter='saved refined support on every frame; rejected Link5 pixels omitted',
          coordinate_system='source camera', geometry_modified=False, purpose='display only'))
    print('REPLAY_CONTEXT_COMPLETE', case, flush=True)
    return case


def run(root, workers):
    manifest = json.loads((root / 'manifest.json').read_text())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(process, [(str(root), e['video_id']) for e in manifest['entries']]))
    write(root / 'replay_context/completion.json', dict(status='complete', videos=len(cases), cases=cases))
    print('ALL_REPLAY_CONTEXTS_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); run(args.root, args.workers)
