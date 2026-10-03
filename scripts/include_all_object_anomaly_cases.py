#!/usr/bin/env python3
"""Restore the three-0001 exclusion policy and score missing eligible cases."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PDI-Bench-edited/src'))
from pdi_eval.anomaly_scoring.runner import DEFAULT_EXCLUSIONS, prepare_selected, sha256, write_json

NEW_CASES = ['COSMOS2.5_0005', 'COSMOS2.5_0010', 'COSMOS2.5_0065', 'COSMOS2.5_0021']


def copy_owned(source, destination):
    # Immutable images can share bytes; JSON/CSV/HTML must remain independent.
    if Path(source).suffix in {'.png', '.mp4', '.npz', '.js'}:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def prepare(run):
    import numpy as np
    from pdi_eval.object_deformation_wrapper.mask_sync import sync_case
    from pdi_eval.object_deformation_wrapper.frame_selection_gallery import write_gallery
    baseline = ROOT / 'results/paired-crop-normalization-20261003'
    metadata = run / 'metadata'
    metadata.mkdir(parents=True)
    shutil.copytree(baseline / 'crops', run / 'crops', copy_function=copy_owned)
    (run / 'inputs').symlink_to('crops', target_is_directory=True)
    shutil.copy2(baseline / 'crops/anomalydino_scores.json', metadata / 'baseline_scores.json')
    name = 'COSMOS2.5_0021'
    original_case = ROOT / 'results/object-deformation-selected45-20260929/cases' / name
    case = run / 'inputs-object/cases' / name
    case.mkdir(parents=True)
    for folder in ('masking', 'score', 'replay', 'occlusion'):
        shutil.copytree(original_case / folder, case / folder, copy_function=copy_owned)
    repair_path = ROOT / 'results/vlm3-overmask-20261002' / name / 'repair.json'
    repair = json.loads(repair_path.read_text())
    old_detection = json.loads((original_case / 'occlusion/detection.json').read_text())
    base = Path(old_detection['inputs']['gripper']['path'])
    assert repair['accepted'] and sha256(base) == repair['link7_input']['sha256']
    assert sha256(Path(repair['output_masks'])) == repair['output_masks_sha256']
    with np.load(base, allow_pickle=False) as archive:
        masks = archive['object_masks'][:, 5].astype(bool)
        assert hashlib.sha256(np.ascontiguousarray(masks).tobytes()).hexdigest() == repair['original_mask_array_sha256']
    preserved = {str(p.relative_to(case)): sha256(p) for p in [
        case / 'masking/segmentation.npz', case / 'score/cotracker_exact-group.npz',
        case / 'score/rigidity.json', case / 'replay/source.mp4']}
    summary = sync_case(case=case, repair_record=repair_path, base_segmentation=base,
                        gripper_root=run / 'gripper', crop_root=run / 'crops')
    assert summary['crops']['frame0_reference_quality'] == 'clean_low_contact'
    assert len(summary['selected_frames']) == 10
    for key, value in preserved.items():
        assert sha256(case / key) == value
    # Every other selection and normalized pair remains exactly the current one.
    old_index = json.loads((baseline / 'crops/selection_index.json').read_text())
    index = json.loads((run / 'crops/selection_index.json').read_text())
    old_rows = {r['case']: r for r in old_index}
    for row in index:
        if row['case'] != name:
            assert row['selected_frames'] == old_rows[row['case']]['selected_frames']
    manifest = prepare_selected(run / 'crops')
    assert manifest['pair_count'] == 420
    existing = json.loads((metadata / 'baseline_scores.json').read_text())
    scored = {(p['video_id'], p['frame']): p for p in existing['pairs']}
    for pair in manifest['pairs']:
        key = pair['video_id'], pair['frame']
        if key in scored:
            for role in ('reference', 'query'):
                assert pair[role+'_sha256'] == scored[key][role+'_sha256']
    pending = [p for p in manifest['pairs'] if (p['video_id'], p['frame']) not in scored]
    assert len(pending) == 40 and {p['video_id'] for p in pending} == set(NEW_CASES)
    write_json(metadata / 'all_pair_manifest.json', manifest)
    write_json(metadata / 'pair_manifest.json', {**manifest, 'pairs': pending, 'pair_count': len(pending)})
    write_json(metadata / 'case0021_sync.json', summary)
    write_json(metadata / 'case0021_provenance.json', {
        'source_repair': str(repair_path), 'source_repair_sha256': sha256(repair_path),
        'mask_source_sha256': repair['output_masks_sha256'], 'preserved_source_sha256': preserved,
        'new_vlm_inference': False, 'sync_default_used': True})
    write_gallery(run / 'crops')
    print(json.dumps({'included_videos': 42, 'new_pairs_to_score': len(pending),
                      'case0021_selected_frames': summary['selected_frames']}))


def analyze(run):
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from analyze_object_occlusion_v2_auroc import workbook_labels
    from pdi_eval.anomaly_scoring.annotate_crops import annotate
    old = json.loads((run / 'metadata/baseline_scores.json').read_text())
    summary = json.loads((run / 'outputs/summary.json').read_text())
    new = [json.loads(line) for line in (run / 'outputs/pairs.jsonl').read_text().splitlines()]
    assert summary['status'] == 'completed' and len(new) == 40
    assert summary['settings'] == old['settings']
    expected = json.loads((run / 'metadata/all_pair_manifest.json').read_text())
    pairs = old['pairs'] + new
    assert len(pairs) == 420 and len({(p['video_id'], p['frame']) for p in pairs}) == 420
    lookup = {(p['video_id'], p['frame']): p for p in pairs}
    for pair in expected['pairs']:
        score = lookup[pair['video_id'], pair['frame']]
        for role in ('reference', 'query'):
            assert score[role+'_sha256'] == pair[role+'_sha256'] == sha256(run / 'crops' / pair[role+'_crop'])
    # Native annotation validates all 420 scores against the displayed pair files.
    combined = run / 'metadata/combined-results'
    combined.mkdir(exist_ok=True)
    write_json(combined / 'summary.json', {**summary, 'pair_count': 420, 'videos': expected['videos'],
                                         'excluded_videos': list(DEFAULT_EXCLUSIONS)})
    (combined / 'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    annotate(run / 'crops', combined)
    current = json.loads((run / 'crops/anomalydino_scores.json').read_text())
    labels = workbook_labels(ROOT / 'selected_45_matched_videos_styledv2.xlsx')
    def stats(score_data):
        rows = [dict(r, human_label_v2=labels[r['video_id']]['label']) for r in score_data['videos'] if r['status']=='scored']
        assert all(r['scored_pair_count']==10 for r in rows)
        binary = [r for r in rows if r['human_label_v2'] != .5]
        pos = [r['anomaly_score_sum'] for r in binary if r['human_label_v2']==1]
        neg = [r['anomaly_score_sum'] for r in binary if r['human_label_v2']==0]
        auc = sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
        correlations = {}
        for name, function in [('pearson',pearsonr),('spearman',spearmanr),('kendall_tau_b',kendalltau)]:
            result = function([r['anomaly_score_sum'] for r in rows], [r['human_label_v2'] for r in rows])
            correlations[name] = {'coefficient':float(result.statistic),'p_two_sided':float(result.pvalue)}
        return {'binary_auroc':auc,'positive_count':len(pos),'negative_count':len(neg),
                'ambiguous_scored_count':len(rows)-len(binary),'unscored_count':45-len(rows),
                'partial_count':0,'scored_video_count':len(rows),'primary':{'n':len(rows),**correlations}}
    baseline_stats, latest_stats = stats(old), stats(current)
    earliest = json.loads((ROOT / 'results/object-reference-crops-20261001/anomalydino_scores.json').read_text())
    vlm3 = json.loads((ROOT / 'results/vlm3-seven-cases-20261003/metadata/publication/merged_anomalydino_scores.json').read_text())
    old_scored = {r['video_id'] for r in earliest['videos'] if r['status']=='scored'}
    same_old_cohort = {**vlm3, 'videos':[r for r in vlm3['videos'] if r['video_id'] in old_scored]}
    history = [
        {'stage':'Original ten-frame scoring','scored_videos':35,**stats(earliest)},
        {'stage':'Refresh COSMOS2.5_0056 on the original scored cohort','scored_videos':35,**stats(same_old_cohort)},
        {'stage':'Add repaired LVP_ROBOWM_0010,0015,0060','scored_videos':38,**stats(vlm3)},
        {'stage':'Matched-canvas normalization, same 38 videos','scored_videos':38,**baseline_stats},
        {'stage':'Include every case except the three 0001 videos','scored_videos':42,**latest_stats}]
    old_videos = {r['video_id']:r for r in old['videos']}
    rows = []
    for video in current['videos']:
        name = video['video_id'];before = old_videos[name]
        rows.append(dict(video, human_label_v2=labels[name]['label'],
                         baseline_sum=before['anomaly_score_sum'],
                         cohort_status='newly included' if name in NEW_CASES else video['status']))
    comparison = {'comparison_type':'cohort-expansion','method':'shared-canvas-native-pixels-v1',
                  'baseline':baseline_stats,'normalized':latest_stats,'videos':rows,'history':history,
                  'added_cases':NEW_CASES,'excluded_videos':list(DEFAULT_EXCLUSIONS),
                  'pair_count':420,'crop_pair_count':450,'settings_equal':True,'selected_frames_equal':True,
                  'same_cohort_normalization_auroc':baseline_stats['binary_auroc'],
                  'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [
                      run / 'metadata/baseline_scores.json',run / 'crops/anomalydino_scores.json',
                      ROOT / 'selected_45_matched_videos_styledv2.xlsx']}}
    write_json(run / 'metadata/comparison.json', comparison)
    write_json(run / 'metadata/OBJECT_ANOMALY_STATISTICS.json',{
        **latest_stats,'label_column':'Object deformation (0/1)2','label_cells':'Sheet1!AF2:AF46',
        'score':'sum of ten unrounded matched-canvas AnomalyDINO scores',
        'crop_geometry':comparison['method'],'excluded_videos':list(DEFAULT_EXCLUSIONS),
        'baseline_binary_auroc':baseline_stats['binary_auroc'],'history':history,
        'source_sha256':comparison['source_sha256']})
    lines=['# Object anomaly scores: all cases except the three 0001 videos','',
           f'Current binary AUROC: **{latest_stats["binary_auroc"]:.6f}**, using {latest_stats["positive_count"]} deformed and {latest_stats["negative_count"]} non-deformed videos.','',
           'All 42 requested videos are scored (420 pairs); one moderate label is omitted only from binary AUROC. The only exclusions are COSMOS2.5_0001, COSMOS3_0001 and LVP_ROBOWM_0001. No score is imputed or set to zero.','',
           '| Stage | Scored videos | Binary AUROC |','|---|---:|---:|']
    for stage in history:lines.append(f'| {stage["stage"]} | {stage["scored_videos"]} | {stage["binary_auroc"]:.6f} |')
    lines += ['', 'The original 0.790514 and later 0.779720 describe different scored cohorts. Refreshing COSMOS2.5_0056 alone gives 0.802372 on the old 35-video cohort; adding the three repaired LVP videos gives 0.779720 on 38 videos. Matched-canvas normalization leaves that 38-video AUROC unchanged. The new current value adds four videos, so it is a cohort expansion, not a controlled normalization-improvement claim.','',
              'Newly included: COSMOS2.5_0005, COSMOS2.5_0010, COSMOS2.5_0065 and COSMOS2.5_0021. The first three already had valid normalized crop pairs. COSMOS2.5_0021 uses its hash-verified accepted VLM3 frame0 repair from the earlier GPU experiment; the default synchronized handoff refreshed occlusion detection, replay and V5 ten-frame crops. Its frame-0 reference now has zero link7 overlap. No new VLM points or mask inference were fabricated.','',
              'Model/checkpoint/inference settings match the preceding normalized run. The 38 existing videos retain their exact crop files and scores. The 40 newly eligible pairs are scored using the same model. Visible pixels and contours are preserved through integer translation and shared-canvas padding; existing correspondence estimates remain estimated. No threshold or parameter was fitted against these labels.','',
              '| Case | Label | Status | Pair count | Anomaly sum |','|---|---:|---|---:|---:|']
    body='<h1>Object anomaly scores · all cases except 0001</h1>'
    body+=f'<p><strong>Binary AUROC {latest_stats["binary_auroc"]:.6f}</strong> · 420 pairs across 42 scored videos. Only the three 0001 videos are excluded.</p>'
    body+='<p><a href="crops/selection_gallery.html">Inspect crop pictures</a> · <a href="metadata/OBJECT_ANOMALY_REPORT.md">Detailed report</a> · <a href="metadata/comparison.json">Exact results</a></p><table><tr><th>Stage</th><th>Scored videos</th><th>AUROC</th></tr>'
    for stage in history:body+=f'<tr><td>{html.escape(stage["stage"])}</td><td>{stage["scored_videos"]}</td><td>{stage["binary_auroc"]:.6f}</td></tr>'
    body+='</table><p>The 0.790514 and 0.779720 values use different cohorts. Normalization on the same 38 videos left AUROC unchanged. One moderate label is omitted from binary AUROC.</p><table><tr><th>Case</th><th>Label</th><th>Status</th><th>Pairs</th><th>Anomaly sum</th></tr>'
    for row in rows:
        total=f'{row["anomaly_score_sum"]:.6f}' if row['anomaly_score_sum'] is not None else '—'
        lines.append(f'| {row["video_id"]} | {row["human_label_v2"]:g} | {row["cohort_status"]} | {row["scored_pair_count"]} | {total} |')
        body+='<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in [row['video_id'],row['human_label_v2'],row['cohort_status'],row['scored_pair_count'],total])+'</tr>'
    body+='</table>'
    (run / 'metadata/OBJECT_ANOMALY_REPORT.md').write_text('\n'.join(lines)+'\n')
    (run / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All-case object anomaly results</title><style>body{font:16px system-ui;background:#f5f6f2;color:#24312f;max-width:1100px;margin:32px auto;padding:0 20px}p{line-height:1.6}a{color:#246757}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:10px;border-bottom:1px solid #cbd3cb}td:first-child{overflow-wrap:anywhere}@media(max-width:700px){table{font-size:12px}td,th{padding:6px}}</style>'+body+'</html>')
    print(json.dumps({'binary_auroc':latest_stats['binary_auroc'],'scored_videos':42,'scored_pairs':420,'added_cases':NEW_CASES},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','analyze']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.run.resolve())
