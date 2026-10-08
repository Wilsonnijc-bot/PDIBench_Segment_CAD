"""Review the five excluded cases using unchanged cached Link5 masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from robot.replay.build_link7_mask_replay import HTML, horizontal_runs
from .common import sha, write
from .common import probe


def run(root, cache, sources, *, case_ids=None, destination_name='excluded_mask_replays'):
    manifest = json.loads((root / 'manifest.json').read_text())
    excluded = manifest['excluded']
    if case_ids is not None:
        included = {r['video_id'] for r in manifest['entries']}
        if not set(case_ids) <= included:
            raise ValueError('Review cases must have completed rigidity runs')
        cases = {case: 'Included in completed rigidity run' for case in case_ids}
    else:
        cases = excluded
    review_title = 'Completed Link5 masking replays' if case_ids is not None else 'Five excluded masking replays'
    entries = {r['video_id']: r for r in json.loads((cache / 'mask_cache_manifest.json').read_text())['entries']}
    destination = root / destination_name
    destination.mkdir(exist_ok=True)
    def process(case):
        entry = entries[case]
        mask_path = cache / entry['mask']
        source = sources / case / 'v1/replay/interactive_exact-group/source.mp4'
        if sha(mask_path) != entry['mask_sha256'] or sha(source) != entry['source_video_sha256']:
            raise ValueError('Mask replay provenance mismatch: ' + case)
        with np.load(mask_path, allow_pickle=False) as archive:
            masks = archive['object_masks'][:, archive['object_names'].tolist().index('link5')].astype(bool)
        frame_count, height, width = masks.shape
        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS); count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count != frame_count or frame_count != entry['frame_count']:
            raise ValueError('Full mask trajectory required')
        folder = destination / case
        folder.mkdir(exist_ok=True)
        playback = folder / 'source.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
                        '-map', '0:v:0', '-an', '-c:v', 'libx264', '-profile:v', 'high',
                        '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'veryfast',
                        '-threads', '2', '-fps_mode', 'passthrough', '-movflags', '+faststart',
                        str(playback)], check=True)
        before, after = probe(source), probe(playback)
        for k in ('width', 'height', 'avg_frame_rate', 'nb_frames'):
            if before['streams'][0][k] != after['streams'][0][k]:
                raise ValueError('Playback changed: ' + k)
        original_pts = [float(f['best_effort_timestamp_time']) for f in before['frames']]
        playback_pts = [float(f['best_effort_timestamp_time']) for f in after['frames']]
        if len(original_pts) != len(playback_pts) or not np.allclose(original_pts, playback_pts, rtol=0, atol=1e-6):
            raise ValueError('Playback timestamps changed')
        runs = [horizontal_runs(np.ascontiguousarray(mask, dtype=np.uint8).tobytes(), height, width) for mask in masks]
        # Independently reconstruct the display encoding and compare every pixel.
        for mask, row in zip(masks, runs):
            decoded = np.zeros_like(mask)
            for y, start, end in row:
                decoded[y, start:end] = True
            if not np.array_equal(mask, decoded):
                raise ValueError('Display mask encoding changed pixels')
        payload = dict(case=case, frames=frame_count, height=height, width=width, fps=fps,
                       base=[[] for _ in masks], refined=runs)
        (folder / 'masks.js').write_text('window.MASK_REPLAY=' + json.dumps(payload, separators=(',', ':')) + ';')
        metadata = dict(video_id=case, review_reason=cases[case], exclusion_reason=excluded.get(case), frame_count=frame_count, fps=fps,
            mask_path=str(mask_path.resolve()), mask_sha256=entry['mask_sha256'],
            source_video_path=str(source.resolve()), source_video_sha256=entry['source_video_sha256'],
            playback_sha256=sha(playback), exact_mask_pixel_roundtrip_verified=True, original_pts_verified=True,
            selected_cached_attempt=mask_path.parent.name, inference_rerun=False)
        write(folder / 'provenance.json', metadata)
        page = HTML.replace('COSMOS3_0015', case).replace('link7', 'Link5').replace('Link7', 'Link5')
        for key, value in dict(__WIDTH__=width, __HEIGHT__=height, __FRAMES__=frame_count,
                               __LAST_FRAME__=frame_count-1).items():
            page = page.replace(key, str(value))
        start = page.index('<div class="notice"')
        end = page.index('<div class="layout">', start)
        status = 'Completed rigidity case' if case_ids is not None else 'Excluded from rigidity scoring'
        page = page[:start] + '<div class="notice"><strong>' + status + '</strong><p>' + cases[case] + '. Saved mask: ' + mask_path.parent.name + '.</p></div>\n' + page[end:]
        page = page.replace('189 source frames · 24 fps', f'{frame_count} source frames · {fps:g} fps')
        page = page.replace('Selected 45 · V1 · ', 'Completed case · ' if case_ids is not None else 'Excluded case · ')
        page = page.replace('src="../v1/replay/interactive_exact-group/source.mp4"', 'src="source.mp4" muted')
        page = page.replace('Inspect the correction', 'Inspect the saved mask').replace('Split comparison', 'Source / mask split')
        page = page.replace('Base SAM3', 'Source only').replace('Persistent V1', 'Link5 overlay')
        page = page.replace('data-mode="split" aria-pressed="true"', 'data-mode="split" aria-pressed="false"')
        page = page.replace('data-mode="refined" aria-pressed="false"', 'data-mode="refined" aria-pressed="true"')
        page = page.replace("let mode='split'", "let mode='refined'")
        start = page.index('<div class="legend">'); end = page.index('</aside></div>', start)
        page = page[:start] + '<div class="legend"><div><span class="swatch refined"></span>Exact cached Link5 mask</div></div><p>Play or scrub to inspect every frame. Source-only view hides the overlay; split view compares source and mask.</p><div class="links"><a href="provenance.json">Mask provenance</a><a href="../index.html">All five excluded cases</a></div>' + page[end:]
        page = page.replace('All five excluded cases', 'All review cases')
        start = page.index('<p class="foot">'); end = page.index('</p>', start)+4
        page = page[:start] + '<p class="foot">Unchanged cached mask trajectory. Display frame = source frame + 1. No new masking or rigidity inference.</p>' + page[end:]
        page = page.replace('Number(frameSlider.value)/data.fps', '(Number(frameSlider.value)+.2)/data.fps')
        (folder / 'index.html').write_text(page)
        print('EXCLUDED_MASK_REPLAY_COMPLETE', case, frame_count, flush=True)
        return metadata
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(process, cases))
    write(destination / 'manifest.json', dict(status='complete', cases=rows, published=False))
    options = ''.join('<option value="' + r['video_id'] + '">' + r['video_id'] + '</option>' for r in rows)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Five excluded Link5 masking replays</title><style>:root{color-scheme:dark;font:15px system-ui;background:#101719;color:#e7f0eb}body{margin:0}header{padding:15px 24px;display:flex;gap:20px;align-items:center;border-bottom:1px solid #344548}h1{font-size:21px;margin:0}select{font:inherit;padding:8px;background:#223c38;color:#edfff6;border:1px solid #5c8277}iframe{width:100%;height:calc(100vh - 75px);min-height:850px;border:0}</style><header><h1>Five excluded masking replays</h1><label>Video <select id="case">' + options + '</select></label></header><iframe id="replay" title="Link5 masking replay"></iframe><script>const select=document.getElementById("case"),frame=document.getElementById("replay");function change(){frame.src=select.value+"/index.html"}select.onchange=change;change();</script></html>'
    page = page.replace('Five excluded Link5 masking replays', review_title).replace('Five excluded masking replays', review_title)
    (destination / 'index.html').write_text(page)
    print('EXCLUDED_MASK_VIEWER_COMPLETE', destination / 'index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True); parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--cases', nargs='+'); parser.add_argument('--destination', default='excluded_mask_replays')
    args = parser.parse_args(); run(args.root, args.cache, args.sources, case_ids=args.cases, destination_name=args.destination)
