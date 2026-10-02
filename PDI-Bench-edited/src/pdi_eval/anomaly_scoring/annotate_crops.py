"""Attach verified pair scores to the existing selected-crop export and gallery."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .runner import read_pairs, sha256, write_json


def annotate(crop_root: Path, results_root: Path):
    crop_root, results_root = Path(crop_root), Path(results_root)
    summary = json.loads((results_root / 'summary.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('The scoring run is incomplete')
    _, scores = read_pairs(results_root / 'pairs.jsonl')
    if len(scores) != summary['pair_count']:
        raise ValueError('Pair count differs from the completed run')
    by_pair = {}
    for row in scores:
        key = (row['video_id'], row['frame'])
        if key in by_pair or not math.isfinite(row['anomaly_score']):
            raise ValueError('Duplicate pair or nonfinite score')
        for role in ('reference', 'query'):
            path = crop_root / row[role + '_crop']
            if sha256(path) != row[role + '_sha256']:
                raise ValueError(f'The scored crop changed: {path}')
        by_pair[key] = dict(row)
    index_path = crop_root / 'selection_index.json'
    index = json.loads(index_path.read_text())
    score_export_path = crop_root / 'anomalydino_scores.json'
    previous = json.loads(score_export_path.read_text()) if score_export_path.is_file() else {}
    original_index_hash = (previous.get('selection_index_sha256_before_annotation', sha256(index_path))
                           if previous.get('source_pair_results_sha256') == sha256(results_root / 'pairs.jsonl')
                           else sha256(index_path))
    case_updates = []
    matched = set()
    exclusions = set(summary['excluded_videos'])
    videos = []
    for case in index:
        name = case['case']
        selection_path = crop_root / name / 'selection.json'
        selection = json.loads(selection_path.read_text())
        if selection['selected_frames'] != case['selected_frames']:
            raise ValueError(f'Selection index differs from case details: {name}')
        status = ('excluded_by_user' if name in exclusions else
                  'unavailable_reference' if not case['selected_frames'] else 'scored')
        case['anomalydino_status'] = status
        selection['anomalydino_status'] = status
        for row in case['selected_frames']:
            key = (name, row['frame'])
            score = by_pair.get(key)
            if status == 'scored':
                if score is None:
                    raise ValueError(f'Missing score for selected pair: {key}')
                directory = Path(name) / row['crop_directory']
                if (score['reference_crop'] != (directory / 'frame0_shape_crop.png').as_posix()
                        or score['query_crop'] != (directory / 'current_available.png').as_posix()):
                    raise ValueError(f'Scored paths differ from the selected pair: {key}')
                row['anomalydino'] = {'status': 'scored', 'anomaly_score': score['anomaly_score']}
                matched.add(key)
            else:
                if score is not None:
                    raise ValueError(f'Unexpected score for excluded pair: {key}')
                row['anomalydino'] = {'status': status}
        values = [row['anomalydino']['anomaly_score'] for row in case['selected_frames']
                  if row['anomalydino']['status'] == 'scored']
        total = math.fsum(values) if values else None
        case['anomalydino_sum'] = selection['anomalydino_sum'] = total
        case['anomalydino_pair_count'] = selection['anomalydino_pair_count'] = len(values)
        videos.append({'video_id': name, 'status': status, 'anomaly_score_sum': total,
                       'scored_pair_count': len(values)})
        selection['selected_frames'] = case['selected_frames']
        case_updates.append((selection_path, selection))
    if matched != set(by_pair):
        raise ValueError('Some scored pairs are absent from the current selection')
    # All inputs and relationships are checked before modifying the crop export.
    export = {'method': 'AnomalyDINO 1-shot', 'settings': summary['settings'],
              'video_aggregation_definition': 'Sum of the selected scored pair anomaly scores',
              'videos': videos,
              'pair_count': len(scores), 'excluded_videos': sorted(exclusions),
              'source_pair_results_sha256': sha256(results_root / 'pairs.jsonl'),
              'selection_index_sha256_before_annotation': original_index_hash,
              'pairs': list(by_pair.values())}
    write_json(crop_root / 'anomalydino_scores.json', export)
    with (crop_root / 'anomalydino_videos.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'video_id', 'status', 'anomaly_score_sum', 'scored_pair_count'])
        writer.writeheader()
        writer.writerows(videos)
    for path, selection in case_updates:
        write_json(path, selection)
    write_json(index_path, index)
    from pdi_eval.object_deformation_wrapper.frame_selection_gallery import write_gallery
    write_gallery(crop_root)
    return {'scored_pairs': len(scores), 'excluded_pairs': sum(
        len(c['selected_frames']) for c in index if c['anomalydino_status'] == 'excluded_by_user')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crop-root', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(annotate(args.crop_root, args.results_root)))


if __name__ == '__main__':
    main()
