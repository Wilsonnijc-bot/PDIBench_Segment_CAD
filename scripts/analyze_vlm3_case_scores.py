#!/usr/bin/env python3
"""Merge only refreshed crop scores and recompute the site's object AUROC."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
from analyze_object_occlusion_v2_auroc import workbook_labels
ROOT=Path(__file__).resolve().parents[1]


def correlations(rows):
    x=[r['anomaly_score_sum'] for r in rows];y=[r['human_label_v2'] for r in rows]
    result={'n':len(rows)}
    for name,fn in [('pearson',pearsonr),('spearman',spearmanr),('kendall_tau_b',kendalltau)]:
        s=fn(x,y);result[name]={'coefficient':float(s.statistic),'p_two_sided':float(s.pvalue)}
    return result


def analyze(run: Path, cases: list[str]):
    root=run.resolve();out=root/'metadata/publication';out.mkdir(parents=True,exist_ok=True)
    old_path=ROOT/'results/object-reference-crops-20261001/anomalydino_scores.json'
    new_path=root/'crops/anomalydino_scores.json'
    old=json.loads(old_path.read_text());new=json.loads(new_path.read_text())
    assert set(v['video_id'] for v in new['videos'])==set(cases)
    assert old['settings']==new['settings']
    videos={v['video_id']:v for v in old['videos']};videos.update({v['video_id']:v for v in new['videos']})
    pairs=[p for p in old['pairs'] if p['video_id'] not in cases]+new['pairs']
    # Every displayed number remains tied to the exact original/current crop bytes.
    for p in pairs:
        crop_root=root/'crops' if p['video_id'] in cases else old_path.parent
        for role in ('reference','query'):
            actual=hashlib.sha256((crop_root/p[role+'_crop']).read_bytes()).hexdigest()
            assert actual==p[role+'_sha256']
    for name,v in videos.items():
        values=[p['anomaly_score'] for p in pairs if p['video_id']==name]
        assert v['scored_pair_count']==len(values)
        assert v['anomaly_score_sum']==(math.fsum(values) if values else None)
    merged={**old,'videos':[videos[n] for n in sorted(videos)],'pairs':pairs,'pair_count':len(pairs),
        'source_runs':{'baseline':str(old_path.relative_to(ROOT)),'updated':str(new_path.relative_to(ROOT))},
        'updated_cases':cases}
    merged_path=out/'merged_anomalydino_scores.json';merged_path.write_text(json.dumps(merged,indent=2)+'\n')
    with (out/'merged_anomalydino_videos.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['video_id','status','anomaly_score_sum','scored_pair_count']);writer.writeheader();writer.writerows(merged['videos'])
    old_index=json.loads((old_path.parent/'selection_index.json').read_text())
    new_index=json.loads((root/'crops/selection_index.json').read_text())
    selections={r['case']:r for r in old_index};selections.update({r['case']:r for r in new_index})
    (out/'merged_selection_index.json').write_text(json.dumps([selections[n] for n in sorted(selections)],indent=2)+'\n')
    labels=workbook_labels(ROOT/'selected_45_matched_videos_styledv2.xlsx')
    assert set(labels)==set(videos) and len(videos)==45
    rows=[dict(v,human_label_v2=labels[name]['label']) for name,v in videos.items()
          if v['status']=='scored' and v['scored_pair_count']==10]
    binary=[r for r in rows if r['human_label_v2']!=.5]
    pos=[r['anomaly_score_sum'] for r in binary if r['human_label_v2']==1]
    neg=[r['anomaly_score_sum'] for r in binary if r['human_label_v2']==0]
    auc=sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
    partial=[v for v in videos.values() if v['status']=='scored' and v['scored_pair_count']!=10]
    unscored=[v for v in videos.values() if v['status']!='scored']
    stats=dict(label_column='Object deformation (0/1)2',label_cells='Sheet1!AF2:AF46',
        score='sum of ten unrounded AnomalyDINO pair scores; partial selections excluded from AUROC',
        primary=correlations(rows),binary_sensitivity=correlations(binary),binary_auroc=auc,
        positive_count=len(pos),negative_count=len(neg),ambiguous_scored_count=len(rows)-len(binary),
        unscored_count=len(unscored),partial_count=len(partial),partial_videos=partial,
        updated_cases=cases,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'selected_45_matched_videos_styledv2.xlsx',old_path,new_path,merged_path]})
    (out/'OBJECT_ANOMALY_STATISTICS.json').write_text(json.dumps(stats,indent=2)+'\n')
    lines=['# Object crop anomaly results after seven-case VLM3 refresh','',
        f'Binary AUROC: **{auc:.3f}** across {len(pos)} deformed and {len(neg)} non-deformed videos. One moderate label is omitted from binary analysis when scored. No score threshold was fitted.','',
        f'{len(rows)} complete ten-pair videos enter the correlation; {len(unscored)} have no score; {len(partial)} have partial selections and are omitted from the ten-pair comparison. Missing values are never set to zero.','',
        '| Statistic | Coefficient | Two-sided p |','|---|---:|---:|']
    for label,key in [('Pearson','pearson'),('Spearman','spearman')]:
        value=stats['primary'][key];lines.append(f'| {label} | {value["coefficient"]:.4f} | {value["p_two_sided"]:.5f} |')
    lines+=['','## Refreshed cases','',
        'All other crop selections and pair scores retain their original bytes. Existing exclusions remain. The refreshed cases use exact VLM3 seed frames, accepted SAM3 masks, synchronized occlusion evidence, and available-pixel crop selection with the original 0/2/2/2/4 quotas and the selective late-area gate.','',
        '| Case | Status | Scored pairs | Anomaly sum |','|---|---|---:|---:|']
    for name in cases:
        v=videos[name];total=f'{v["anomaly_score_sum"]:.6f}' if v['anomaly_score_sum'] is not None else '—'
        lines.append(f'| {name} | {v["status"]} | {v["scored_pair_count"]} | {total} |')
    lines+=['','Labels: revised workbook column AF, matched by generator and video number. AnomalyDINO can respond to appearance, viewpoint, segmentation and mapping differences as well as deformation. These results describe the selected dataset; the correlations do not establish deformation-specific causation or out-of-sample accuracy.','',
        '[Exact statistics](OBJECT_ANOMALY_STATISTICS.json) · [Updated crop gallery](../objects/selection_gallery.html)','']
    (out/'OBJECT_ANOMALY_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:stats[k] for k in ['binary_auroc','positive_count','negative_count','unscored_count','partial_count']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True)
    args=p.parse_args();analyze(args.run,args.cases)
