#!/usr/bin/env python3
"""Prepare pixel-preserving paired crops and compare controlled DINO scores."""
from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/normalize_object_crop_pairs.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = _SOURCE_PATH.parents[1]
sys.path.insert(0, str(ROOT / 'PDI-Bench-edited/src'))

from pdi_eval.anomaly_scoring.runner import prepare_selected, sha256, write_json


def prepare(run: Path):
    import numpy as np
    from PIL import Image
    from pdi_eval.object_deformation_wrapper.paired_crops import METHOD, normalize_pair
    from pdi_eval.object_deformation_wrapper.reference_visible_pixels import write_preview
    from pdi_eval.object_deformation_wrapper.frame_selection_gallery import write_gallery

    fresh = ROOT / 'results/vlm3-seven-cases-20261003'
    publication = fresh / 'metadata/publication'
    baseline_path = publication / 'merged_anomalydino_scores.json'
    baseline = json.loads(baseline_path.read_text())
    index_path = publication / 'merged_selection_index.json'
    index = json.loads(index_path.read_text())
    assert len(index) == 45 and baseline['pair_count'] == 380
    known_scores = {(r['video_id'], r['frame']): r for r in baseline['pairs']}
    metadata = run / 'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline_path, metadata / 'baseline_scores.json')
    shutil.copy2(index_path, metadata / 'baseline_selection_index.json')
    shutil.copy2(ROOT / 'docs/analysis/OBJECT_ANOMALY_STATISTICS.json', metadata / 'baseline_statistics.json')
    crops = run / 'crops'
    crops.mkdir()
    (run / 'inputs').symlink_to('crops', target_is_directory=True)
    sources, pair_count = {}, 0
    for case in index:
        name = case['case']
        old_root = fresh / 'crops' if name in baseline['updated_cases'] else ROOT / 'results/object-reference-crops-20261001'
        source = old_root / name
        destination = crops / name
        destination.mkdir()
        source_selection = json.loads((source / 'selection.json').read_text())
        assert source_selection['selected_frames'] == case['selected_frames'], name
        for filename in ('manifest.json', 'selection.json', 'frame0_reference.png'):
            if (source / filename).is_file():
                shutil.copy2(source / filename, destination / filename)
        geometry = {'method': METHOD, 'pairs': {}}
        source_hashes = {}
        for selected in case['selected_frames']:
            t, folder = selected['frame'], selected['crop_directory']
            old, new = source / folder, destination / folder
            new.mkdir(parents=True)
            arrays = {}
            for role, filename in (('query', 'current_available.png'), ('reference', 'frame0_shape_crop.png')):
                path = old / filename
                source_hashes[(Path(folder) / filename).as_posix()] = sha256(path)
                if (name, t) in known_scores:
                    assert sha256(path) == known_scores[(name, t)][role + '_sha256']
                with Image.open(path) as image:
                    arrays[role] = np.array(image.convert('RGBA'))
            query, reference, record = normalize_pair(arrays['query'], arrays['reference'])
            for role, array, filename in (('query', query, 'current_available.png'),
                                          ('reference', reference, 'frame0_shape_crop.png')):
                old_array = arrays[role]
                old_visible = old_array[old_array[:, :, 3] > 0]
                new_visible = array[array[:, :, 3] > 0]
                np.testing.assert_array_equal(old_visible, new_visible)
                x0, y0, x1, y1 = record['roles'][role]['content_bbox_xyxy']
                x, y = record['roles'][role]['content_offset_xy']
                np.testing.assert_array_equal(array[y:y+y1-y0, x:x+x1-x0, 3],
                                              old_array[y0:y1, x0:x1, 3])
                Image.fromarray(array).save(new / filename)
                record[role + '_sha256'] = sha256(new / filename)
                record[role + '_source_sha256'] = sha256(old / filename)
            # Integer translation/padding is idempotent; no geometry is inferred.
            q2, r2, _ = normalize_pair(query, reference)
            np.testing.assert_array_equal(q2, query)
            np.testing.assert_array_equal(r2, reference)
            geometry['pairs'][folder] = record
            shutil.copy2(old / 'original_frame.png', new / 'original_frame.png')
            source_hashes[(Path(folder) / 'original_frame.png').as_posix()] = sha256(old / 'original_frame.png')
            write_preview(destination / 'frame0_reference.png', new / 'current_available.png',
                          new / 'frame0_shape_crop.png', new / 'preview.png', t)
            pair_count += 1
        write_json(destination / 'pair_geometry.json', geometry)
        sources[name] = {'root': str(source.relative_to(ROOT)), 'source_sha256': source_hashes,
                         'selection_sha256': sha256(source / 'selection.json'),
                         'manifest_sha256': sha256(source / 'manifest.json')}
    write_json(crops / 'selection_index.json', index)
    pairs = prepare_selected(crops, baseline['excluded_videos'])
    assert pairs['pair_count'] == 380 and pair_count == 440
    write_json(metadata / 'pair_manifest.json', pairs)
    write_json(metadata / 'crop_verification.json', {
        'method': METHOD, 'case_count': 45, 'normalized_pairs': pair_count,
        'scoring_pairs': pairs['pair_count'], 'visible_rgba_exact': True,
        'alpha_shape_exact_under_integer_translation': True, 'idempotent': True,
        'frame_selection_unchanged': True, 'resampling': 'none', 'sources': sources,
        'baseline_scores_sha256': sha256(baseline_path),
        'baseline_selection_index_sha256': sha256(index_path)})
    shutil.copy2(ROOT / 'PDI-Bench-edited/src/pdi_eval/object_deformation_wrapper/REFERENCE_VISIBLE_PIXELS.md', crops / 'README.md')
    write_gallery(crops)
    print(json.dumps({'normalized_pairs': pair_count, 'scoring_pairs': pairs['pair_count']}))


def analyze(run: Path):
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from analyze_object_occlusion_v2_auroc import workbook_labels
    from pdi_eval.anomaly_scoring.annotate_crops import annotate

    annotate(run / 'crops', run / 'outputs')
    old = json.loads((run / 'metadata/baseline_scores.json').read_text())
    new = json.loads((run / 'crops/anomalydino_scores.json').read_text())
    labels = workbook_labels(ROOT / 'selected_45_matched_videos_styledv2.xlsx')
    assert old['settings'] == new['settings'], 'Model configuration changed'
    old_pairs = {(r['video_id'], r['frame']): r for r in old['pairs']}
    new_pairs = {(r['video_id'], r['frame']): r for r in new['pairs']}
    assert set(old_pairs) == set(new_pairs) and len(new_pairs) == 380
    old_videos = {r['video_id']: r for r in old['videos']}
    rows = []
    for video in new['videos']:
        name = video['video_id']
        before = old_videos[name]
        assert video['status'] == before['status']
        row = dict(video, human_label_v2=labels[name]['label'],
                   baseline_sum=before['anomaly_score_sum'])
        row['delta_sum'] = (video['anomaly_score_sum'] - before['anomaly_score_sum']
                            if video['anomaly_score_sum'] is not None else None)
        rows.append(row)
    scored = [r for r in rows if r['scored_pair_count'] == 10]
    binary = [r for r in scored if r['human_label_v2'] != .5]
    def statistics(key):
        positives = [r[key] for r in binary if r['human_label_v2'] == 1]
        negatives = [r[key] for r in binary if r['human_label_v2'] == 0]
        auc = sum((p > n) + .5*(p == n) for p in positives for n in negatives) / (len(positives)*len(negatives))
        correlations = {}
        for name, function in (('pearson', pearsonr), ('spearman', spearmanr), ('kendall_tau_b', kendalltau)):
            result = function([r[key] for r in scored], [r['human_label_v2'] for r in scored])
            correlations[name] = {'coefficient': float(result.statistic), 'p_two_sided': float(result.pvalue)}
        return {'binary_auroc': auc, 'positive_count': len(positives), 'negative_count': len(negatives),
                'ambiguous_scored_count': len(scored)-len(binary),
                'unscored_count': len(rows)-len(scored), 'partial_count': 0,
                'primary': {'n': len(scored), **correlations}}
    baseline, normalized = statistics('baseline_sum'), statistics('anomaly_score_sum')
    pair_deltas = [new_pairs[k]['anomaly_score']-old_pairs[k]['anomaly_score'] for k in new_pairs]
    report = {'method': 'shared-canvas-native-pixels-v1', 'baseline': baseline,
              'normalized': normalized, 'videos': rows, 'pair_count': len(pair_deltas),
              'mean_pair_delta': math.fsum(pair_deltas)/len(pair_deltas),
              'max_absolute_pair_delta': max(map(abs, pair_deltas)),
              'settings_equal': True, 'selected_frames_equal': True,
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (
                  run / 'metadata/baseline_scores.json', run / 'crops/anomalydino_scores.json',
                  ROOT / 'selected_45_matched_videos_styledv2.xlsx')}}
    write_json(run / 'metadata/comparison.json', report)
    write_json(run / 'metadata/OBJECT_ANOMALY_STATISTICS.json', {
        **normalized, 'label_column': 'Object deformation (0/1)2', 'label_cells': 'Sheet1!AF2:AF46',
        'score': 'sum of ten unrounded paired-canvas AnomalyDINO scores',
        'crop_geometry': report['method'], 'baseline_binary_auroc': baseline['binary_auroc'],
        'source_sha256': report['source_sha256']})
    lines = ['# AnomalyDINO with matched crop canvases', '',
             f'Binary AUROC: **{baseline["binary_auroc"]:.6f} → {normalized["binary_auroc"]:.6f}**.', '',
             'Controlled comparison: the same 380 selected pairs in 38 videos, the same model/checkpoint/settings, and the same revised labels and exclusions. Binary AUROC uses 37 videos (11 deformed, 26 non-deformed); one moderate label is omitted. No thresholds or parameters were fitted against these labels.', '',
             'The new inputs trim transparent margins, then use identical centered canvas sizes per pair. Every visible RGB/alpha pixel, silhouette and native pixel area is preserved exactly. There is no interpolation, warping, equal-area matching or physical-scale estimation. Existing frame-0 correspondence estimates are unchanged. Both images now receive the same DINO shorter-edge resize factor. Black alpha compositing, eight reference rotations and no PCA masking remain unchanged.', '',
             f'Mean pair-score change: {report["mean_pair_delta"]:+.6f}; largest absolute pair-score change: {report["max_absolute_pair_delta"]:.6f}. A lower score is not itself evidence of improved anomaly detection. This is an evaluation on the existing selected dataset, not held-out validation.', '',
             '| Case | Label | Previous sum | Normalized sum | Change |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['delta_sum'] is not None:
            lines.append(f'| {row["video_id"]} | {row["human_label_v2"]:g} | {row["baseline_sum"]:.6f} | {row["anomaly_score_sum"]:.6f} | {row["delta_sum"]:+.6f} |')
        else:
            lines.append(f'| {row["video_id"]} | {row["human_label_v2"]:g} | — | — | {row["status"]} |')
    (run / 'metadata/OBJECT_ANOMALY_REPORT.md').write_text('\n'.join(lines)+'\n')
    body = '<h1>Matched crop canvases · AnomalyDINO</h1>'
    body += f'<p>Binary AUROC <strong>{baseline["binary_auroc"]:.3f} → {normalized["binary_auroc"]:.3f}</strong> · Same 380 pairs, model settings, frame selection and labels.</p>'
    body += '<p>Transparent margins are trimmed, then both crops are centered on canvases of the same size. Visible pixels and shapes are unchanged; existing frame-0 correspondence remains estimated.</p><p><a href="crops/selection_gallery.html">Inspect the normalized crop pictures</a> · <a href="metadata/OBJECT_ANOMALY_REPORT.md">Detailed comparison</a> · <a href="metadata/comparison.json">Exact results</a></p>'
    body += '<table><thead><tr><th>Case</th><th>Label</th><th>Previous sum</th><th>Normalized sum</th><th>Change</th></tr></thead><tbody>'
    for row in rows:
        numbers = ([f'{row["baseline_sum"]:.6f}', f'{row["anomaly_score_sum"]:.6f}', f'{row["delta_sum"]:+.6f}']
                   if row['delta_sum'] is not None else ['—', '—', row['status']])
        body += '<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in [row['video_id'], row['human_label_v2'], *numbers])+'</tr>'
    body += '</tbody></table><p>These results describe the existing dataset. A lower anomaly score alone does not mean a better detector.</p>'
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched crop canvases</title><style>body{font:16px system-ui;background:#f5f6f2;color:#24312f;max-width:1100px;margin:32px auto;padding:0 20px}p{line-height:1.6}a{color:#246757}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:10px;border-bottom:1px solid #cbd3cb}td:first-child{overflow-wrap:anywhere}@media(max-width:700px){table{font-size:12px}td,th{padding:6px}}</style>'+body+'</html>'
    (run / 'index.html').write_text(page)
    print(json.dumps({'baseline_auroc': baseline['binary_auroc'], 'normalized_auroc': normalized['binary_auroc'], 'mean_pair_delta': report['mean_pair_delta']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'analyze'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.run.resolve())
