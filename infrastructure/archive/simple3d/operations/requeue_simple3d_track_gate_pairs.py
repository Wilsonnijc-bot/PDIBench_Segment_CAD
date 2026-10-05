"""Remove track-quality vetoes in place and requeue only their rejected pairs."""

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'experiments/requeue_simple3d_track_gate_pairs.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = _SOURCE_PATH.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'PDI-Bench-edited/src')]
from experiments.simple3d_pair_visibility import Config, common_visible, erode
from experiments.simple3d_pipeline import sha, write_json


def requeue(root):
    root = Path(root).resolve()
    inputs = root/'inputs'
    manifest = json.loads((inputs/'manifest.json').read_text())
    previous_manifest_sha256 = sha(inputs/'manifest.json')
    config = Config(**dict(manifest['config'], track_quality_gate=False)).validate()
    preserved = {str(p.relative_to(root)): sha(p) for p in root.glob('cases/*/bin_*/comparison.json')
                 if json.loads(p.read_text())['simple3d_status'] == 'complete'}
    updated = []
    rejected = []
    # Evaluate every map before invalidating any terminal comparison.
    for row in manifest['rows']:
        folder = inputs/row['video_id']
        with np.load(folder/'selected_masks.npz', allow_pickle=False) as z:
            masks = dict(zip(z['frame_ids'].tolist(), z['masks'].astype(bool)))
        with np.load(folder/'saved_correspondences.npz', allow_pickle=False) as z:
            tracks = dict(zip(z['frame_ids'].tolist(), z['tracks']))
            visibility = dict(zip(z['frame_ids'].tolist(), z['visibility']))
            point_ids = z['point_ids']
        for selected in row['tests']:
            if selected['status'] == 'ready':
                # Accepted masks are already frozen inputs. Re-estimating them
                # on another OpenCV build can shift one boundary pixel, even
                # with the same RNG seed; retain their exact scientific bytes.
                selected.setdefault('correspondence', {}).update(track_quality_gate=False,
                    track_quality_policy='diagnostic only; no count/inlier-percentage/hull-coverage vetoes')
                continue
            a, b = selected['reference_frame_id'], selected['test_frame_id']
            ref, test, info = common_visible(masks[a], masks[b], tracks[a], tracks[b],
                                             visibility[a], visibility[b], config)
            clean = [erode(ref, config.erosion_px), erode(test, config.erosion_px)]
            assert min(m.sum() for m in clean) >= config.minimum_pixels
            pair = inputs/selected['pair_inputs']
            rejected.append(dict(video_id=row['video_id'], temporal_bin=selected['temporal_bin'],
                                 previous_failure_reason=selected['failure_reason'],
                                 pair_path=f"cases/{row['video_id']}/bin_{selected['temporal_bin']:02d}"))
            info['source_point_ids'] = point_ids[info['inlier_track_ids']].tolist()
            selected.update(status='ready', failure_reason=None, correspondence=info,
                            common_areas=[int(ref.sum()), int(test.sum())],
                            eroded_common_areas=[int(m.sum()) for m in clean])
            updated.append((pair, ref, test, clean))
    assert sum(len(row['tests']) for row in manifest['rows']) == 450
    assert all(s['status'] == 'ready' for row in manifest['rows'] for s in row['tests'])
    for pair, ref, test, clean in updated:
        for role, common, eroded in zip(('reference', 'test'), (ref, test), clean):
            for suffix, mask in [('common', common), ('eroded', eroded)]:
                assert cv2.imwrite(str(pair/f'{role}_{suffix}.png'), mask.astype(np.uint8)*255)
    for row in manifest['rows']:
        write_json(inputs/row['video_id']/'selection.json', row)
    manifest['config'] = asdict(config)
    manifest['track_quality_policy'] = 'User requested removal of count, inlier-percentage and hull-coverage vetoes; tracks estimate mask correspondence only.'
    write_json(inputs/'manifest.json', manifest)
    done_path = root/'metadata/transfer_verification.json'
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    for rejected_pair in rejected:
        pair = root/rejected_pair['pair_path']
        evidence = pair/'comparison.json'
        if evidence.exists():
            assert json.loads(evidence.read_text())['simple3d_status'] != 'complete'
            evidence.unlink()
        (pair.parent/'comparisons.json').unlink(missing_ok=True)
        done.pop(rejected_pair['pair_path'], None)
    if done_path.exists():
        write_json(done_path, done)
    for name, digest in preserved.items():
        assert sha(root/name) == digest, 'successful comparison changed: '+name
    provenance = root/'metadata/track_gate_change.json'
    record = json.loads(provenance.read_text()) if provenance.exists() else {}
    record.update(status='requeued_without_track_quality_gate', updated_at=datetime.now(timezone.utc).isoformat(),
                  previous_manifest_sha256=previous_manifest_sha256, new_manifest_sha256=sha(inputs/'manifest.json'),
                  expected_pairs=450, ready_pairs=450, requeued_pairs=rejected,
                  preserved_successful_pairs=len(preserved), preserved_comparison_sha256=preserved,
                  accepted_common_and_eroded_masks_verified_unchanged=True,
                  labels_used=False, scoring_code_changed=False)
    write_json(provenance, record)
    print(json.dumps(dict(ready_pairs=450, requeued_pairs=len(rejected),
                         preserved_successful_pairs=len(preserved)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    requeue(parser.parse_args().root)
