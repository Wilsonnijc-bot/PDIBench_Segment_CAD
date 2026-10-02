"""Pairing and batch I/O, kept separate from AnomalyDINO inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

DEFAULT_EXCLUSIONS = (
    'COSMOS2.5_0001', 'COSMOS3_0001', 'LVP_ROBOWM_0001',
    'COSMOS2.5_0065', 'COSMOS2.5_0010', 'COSMOS2.5_0005')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_selected(crop_root: Path, exclusions=DEFAULT_EXCLUSIONS):
    crop_root = Path(crop_root).resolve()
    index_path = crop_root / 'selection_index.json'
    selected = json.loads(index_path.read_text())
    known = {row['case'] for row in selected}
    if len(known) != len(selected) or set(exclusions) - known:
        raise ValueError('Duplicate cases or unknown exclusions')
    manifest = {'crop_root': crop_root.name, 'selection_index_sha256': sha256(index_path),
                'excluded_videos': list(exclusions), 'videos': [], 'pairs': []}
    for row in selected:
        video = row['case']
        frames = row['selected_frames']
        status = ('excluded_by_user' if video in exclusions else
                  'unavailable_reference' if not frames else 'ready')
        manifest['videos'].append({'video_id': video, 'status': status,
                                   'selected_pair_count': len(frames),
                                   'selection_status': row['status']})
        if status != 'ready':
            continue
        for frame in frames:
            directory = Path(video) / frame['crop_directory']
            reference = directory / 'frame0_shape_crop.png'
            query = directory / 'current_available.png'
            for path in (reference, query):
                if not (crop_root / path).is_file():
                    raise FileNotFoundError(crop_root / path)
            manifest['pairs'].append({
                'video_id': video, 'frame': frame['frame'],
                'reference_crop': reference.as_posix(), 'query_crop': query.as_posix(),
                'reference_sha256': sha256(crop_root / reference),
                'query_sha256': sha256(crop_root / query),
                'selection_reason': frame['reason'],
                'occlusion_flagged': frame['occlusion_flagged']})
    manifest['pair_count'] = len(manifest['pairs'])
    return manifest


def read_pairs(path: Path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return {}, [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    return ({}, value) if isinstance(value, list) else (value, value['pairs'])


def run_pairs(scorer, pairs, input_root: Path, output_root: Path, *, manifest=None):
    """Initialize scorer outside this function; write one row per supplied pair."""
    output_root = Path(output_root)
    input_root = Path(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Validate all inputs and hashes before starting model inference.
    for pair in pairs:
        for role in ('reference', 'query'):
            path = input_root / pair[role + '_crop']
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = pair.get(role + '_sha256')
            if expected is not None and sha256(path) != expected:
                raise ValueError(f'Input changed: {path}')
    rows = []
    started = time.monotonic()
    temporary = output_root / 'pairs.jsonl.tmp'
    with temporary.open('w') as stream:
        for number, pair in enumerate(pairs, 1):
            begin = time.monotonic()
            result = scorer.score(input_root / pair['reference_crop'],
                                  input_root / pair['query_crop'])
            row = dict(pair, anomaly_score=float(result['anomaly_score']),
                       inference_seconds=time.monotonic() - begin)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            rows.append(row)
            print(f"[{number}/{len(pairs)}] {pair['video_id']} frame={pair.get('frame')} "
                  f"score={row['anomaly_score']:.8f}", flush=True)
    temporary.replace(output_root / 'pairs.jsonl')
    groups = {}
    for row in rows:
        groups.setdefault(row['video_id'], []).append(row['anomaly_score'])
    videos = [{ 'video_id': video, 'pair_count': len(scores),
                'mean_anomaly_score': sum(scores) / len(scores),
                'max_anomaly_score': max(scores)} for video, scores in sorted(groups.items())]
    with (output_root / 'videos.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'video_id', 'pair_count', 'mean_anomaly_score', 'max_anomaly_score'])
        writer.writeheader()
        writer.writerows(videos)
    summary = {'status': 'completed', 'pair_count': len(rows), 'scored_video_count': len(videos),
               'settings': scorer.settings, 'seconds': time.monotonic() - started,
               'reference_cache_hits': scorer.reference_cache_hits,
               'reference_cache_misses': scorer.reference_cache_misses,
               'scalar_definition': 'Official mean_top1p of raw patch-grid normalized-L2/2 1NN distances',
               'video_aggregation': 'Descriptive mean and maximum of supplied pair scores; not an upstream video metric'}
    if manifest:
        summary['videos'] = manifest.get('videos', [])
        summary['excluded_videos'] = manifest.get('excluded_videos', [])
        summary['selection_index_sha256'] = manifest.get('selection_index_sha256')
    write_json(output_root / 'summary.json', summary)
    print('COMPLETED: ' + json.dumps({'pairs': len(rows), 'videos': len(videos)}), flush=True)
    return rows, summary
