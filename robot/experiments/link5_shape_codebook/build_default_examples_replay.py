"""Add four-reference upstream-generator geometry previews to the existing replay."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .build_full_video_replay import packed
from .common import FOUR_REFERENCE_IDS, normalize, read, sha, write


def run(root, destination):
    root = Path(root); destination = Path(destination)
    folder = root / 'normal_reference/default_examples'; manifest = read(folder / 'manifest.json')
    if manifest['status'] != 'complete' or manifest['normal_video_ids'] != list(FOUR_REFERENCE_IDS) or manifest['normal_frame_ids'] != [0]*4:
        raise ValueError('four actual training frame0 references required')
    href = lambda p:os.path.relpath(p, destination)
    payload = dict(normals={}, examples=[], manifest=href(folder / 'manifest.json'), normalization=manifest['normalization'])
    packets = destination / 'default-examples'; packets.mkdir(exist_ok=True)
    for video in FOUR_REFERENCE_IDS:
        with np.load(folder / f'{video}_normal.npz', allow_pickle=False) as a:
            normal = a['points_normalized'].copy(); indices = a['sampled_input_indices'].copy()
        source = root / 'cases' / video / 'observations/frame_00000/input.npz'
        with np.load(source, allow_pickle=False) as a:
            np.testing.assert_array_equal(normal, normalize(a['xyz_camera'], manifest['normalization'])[indices])
        payload['normals'][video] = dict(points=packed(normal, '<f4'), count=len(normal), source=href(source))
        for item in [r for r in manifest['examples'] if r['video_id'] == video]:
            path = folder / item['archive']
            if sha(path) != item['archive_sha256'] or sha(source) != item['input_sha256']:
                raise ValueError('generator example/input identity changed')
            with np.load(path, allow_pickle=False) as a:
                np.testing.assert_array_equal(a['normal'], normal)
                points = a['anomalous'].copy(); mask = a['gt_mask'].astype(np.uint8)
                error = float(abs(normal - points - a['gt_offset']).max())
                if item['used_for_training']:
                    np.testing.assert_allclose(normal - points, a['gt_offset'], rtol=0, atol=1e-7)
                elif not np.array_equal(points, normal):
                    raise ValueError('excluded-mode preview does not reproduce unchanged coordinates')
            packet = packets / (item['key'] + '.js')
            packet.write_text('window.LINK5_DEFAULT_FRAME('+json.dumps(dict(key=item['key'],points=packed(points, '<f4'),mask=packed(mask, 'u1')),separators=(',', ':'))+');\n')
            payload['examples'].append({**item, 'packet':href(packet), 'archive':href(path), 'verified_target_error':error})
    if len(payload['examples']) != 48 or sum(e['used_for_training'] for e in payload['examples']) != 36:
        raise ValueError('default generator replay coverage incomplete')
    (destination / 'default-generation.js').write_text('window.LINK5_DEFAULT_GENERATION='+json.dumps(payload,separators=(',', ':'))+';\n')
    (destination / 'default-generation-app.js').write_text(Path(__file__).with_name('default_generation_app.js').read_text())
    page = (destination / 'index.html').read_text()
    if 'id="generation-tab"' not in page:
        page = page.replace('<button id="training-tab">', '<button id="generation-tab">Default synthetic defects</button><button id="training-tab">', 1)
        page = page.replace('</main>', Path(__file__).with_name('default_generation_body.html').read_text()+'</main>', 1)
        page = page.replace('</body>', '<script src="default-generation.js"></script><script src="default-generation-app.js"></script></body>', 1)
        (destination / 'index.html').write_text(page)
    write(destination / 'default-generation-verification.json', dict(status='verified', reference_videos=list(FOUR_REFERENCE_IDS),
          frame_ids=[0]*4, used_method_examples=36, excluded_unchanged_method_previews=12,
          points_match_training_inputs=True, actual_displacement_is_displayed=True, model_inference=False,
          new_masks_or_depth=False, source_manifest_sha256=sha(folder / 'manifest.json')))
    print('LINK5_DEFAULT_GENERATION_REPLAY_READY', len(payload['examples']), destination, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(); run(args.root, args.destination)
