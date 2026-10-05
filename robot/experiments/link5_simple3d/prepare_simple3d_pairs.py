"""Freeze ten later-bin tests and common masks from the existing selected-45 replay."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys

import cv2
import numpy as np

ROOT = (_workspace_root())
sys.path[:0] = [str(ROOT), str((_workspace_root()))]
from infrastructure.shared.experimental.simple3d.simple3d_pair_visibility import Config, common_visible, erode, select_frames
from infrastructure.shared.experimental.simple3d.simple3d_pipeline import sha, write_json
from infrastructure.shared.contracts.contracts import video_path


def prepare(output, config, video_root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old = (_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929')
    selection = json.loads((old/'selection.json').read_text())
    rows = []
    for e in selection['videos']:
        video_id = f"{e['dataset']}_{e['video_number']}"
        source = old/'cases'/video_id
        archive = source/'base_segmentation.npz'
        tracker = source/'v1_cotracker3/cotracker_exact-group.npz'
        tracking_manifest = json.loads((source/'v1_cotracker3/manifest.json').read_text())
        if tracking_manifest['input']['sha256'] != e['sha256']:
            raise ValueError('saved tracks/source video identity mismatch: '+video_id)
        with np.load(archive, allow_pickle=False) as z:
            masks = z['object_masks'][:, list(z['object_names']).index('link5')].astype(bool)
        with np.load(tracker, allow_pickle=False) as z:
            k = list(z['object_names']).index('link5')
            start, end = z['object_offsets'][k:k+2]
            tracks, visibility = z['tracks'][:, start:end], z['visibility'][:, start:end]
            point_ids = z['point_ids'][start:end]
        if len(masks) != len(tracks) or tracks.shape[:2] != visibility.shape:
            raise ValueError('saved mask/track frame grids differ: '+video_id)
        cap = cv2.VideoCapture(str(source/'v1_cotracker3/replay/interactive_exact-group/source.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
        if shape != masks.shape or fps <= 0:
            raise ValueError('existing replay source and mask frame grid differ: '+video_id)
        base, tests, raw, areas = select_frames(masks, config)
        dest = output/video_id
        dest.mkdir()
        ids = ([base] if base is not None else []) + [s['test_frame_id'] for s in tests if s['test_frame_id'] is not None]
        np.savez_compressed(dest/'selected_masks.npz', masks=masks[ids], frame_ids=ids)
        np.savez_compressed(dest/'saved_correspondences.npz', tracks=tracks[ids], visibility=visibility[ids],
                            point_ids=point_ids, frame_ids=ids)
        for s in tests:
            t = s['test_frame_id']
            s.update(reference_frame_id=base, timestamp_seconds=t/fps if t is not None else None,
                     reference_timestamp_seconds=base/fps if base is not None else None)
            pair = dest/f"bin_{s['temporal_bin']:02d}"
            pair.mkdir()
            s['pair_inputs'] = str(pair.relative_to(output))
            if base is None or t is None:
                s['status'] = 'invalid'; s['failure_reason'] = s['failure_reason'] or 'no valid base reference frame'
                continue
            for role, m in [('reference', masks[base]), ('test', masks[t])]:
                cv2.imwrite(str(pair/f'{role}_original.png'), m.astype(np.uint8)*255)
            try:
                a, b, mapping = common_visible(masks[base], masks[t], tracks[base], tracks[t], visibility[base], visibility[t], config)
                mapping['source_point_ids'] = point_ids[mapping['inlier_track_ids']].tolist()
                s['correspondence'] = mapping
                s['common_areas'] = [int(a.sum()), int(b.sum())]
                s['eroded_common_areas'] = []
                for role, m in [('reference', a), ('test', b)]:
                    cv2.imwrite(str(pair/f'{role}_common.png'), m.astype(np.uint8)*255)
                    eroded = erode(m, config.erosion_px)
                    cv2.imwrite(str(pair/f'{role}_eroded.png'), eroded.astype(np.uint8)*255)
                    s['eroded_common_areas'].append(int(eroded.sum()))
                if min(s['eroded_common_areas']) < config.minimum_pixels:
                    raise ValueError('too little pair-specific support after erosion')
                s['status'] = 'ready'
            except Exception as exc:
                s.update(status='invalid', failure_reason=str(exc))
        row = dict(video_id=video_id, video_path=str(video_path(video_root, e)), video_sha256=e['sha256'],
                   fps=fps, frame_count=len(masks), source_mask_path=str(archive), source_mask_sha256=sha(archive),
                   source_tracks_path=str(tracker), source_tracks_sha256=sha(tracker),
                   selected_masks_path=str((dest/'selected_masks.npz').relative_to(output)),
                   selected_masks_sha256=sha(dest/'selected_masks.npz'), reference_frame_id=base,
                   tests=tests, frame_areas=[dict(frame_id=i, timestamp_seconds=i/fps, raw_mask_area=int(r),
                       eroded_mask_area=int(a)) for i, (r, a) in enumerate(zip(raw, areas))])
        rows.append(row)
        write_json(dest/'selection.json', row)
        print(video_id, 'ready pairs', sum(s['status']=='ready' for s in tests), '/10', flush=True)
    manifest = dict(config=asdict(config), rows=rows, source_selection_sha256=sha(old/'selection.json'),
        selection_policy='same frozen 45 videos; base first valid; maximum valid raw mask area in each of ten bins covering [10%,100%)',
        expected_comparisons=10*len(rows), segmentation_rerun=False, tracking_rerun=False,
        old_geometry_reused=False)
    write_json(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video-root', type=Path, default=Path('/root/autodl-tmp/motionsmoothness-robot/videos'))
    p.add_argument('--erosion-px', type=int, default=2)
    p.add_argument('--visibility-mode', choices=['strict_hulls','smaller_view_anchor'],default='smaller_view_anchor')
    p.add_argument('--track-quality-gate', action='store_true', help='Reproduce the superseded track-consensus quality vetoes')
    p.add_argument('--correspondence-px', type=float, default=6)
    p.add_argument('--depth-window', type=int, default=5)
    p.add_argument('--depth-mad-multiplier', type=float, default=8)
    p.add_argument('--depth-relative-threshold', type=float, default=.05)
    p.add_argument('--minimum-depth-retained-fraction', type=float, default=.5)
    a = p.parse_args()
    prepare(a.output, Config(erosion_px=a.erosion_px, visibility_mode=a.visibility_mode, track_quality_gate=a.track_quality_gate, correspondence_px=a.correspondence_px,
        depth_window=a.depth_window, depth_mad_multiplier=a.depth_mad_multiplier,
        depth_relative_threshold=a.depth_relative_threshold,
        minimum_depth_retained_fraction=a.minimum_depth_retained_fraction).validate(), a.video_root)
