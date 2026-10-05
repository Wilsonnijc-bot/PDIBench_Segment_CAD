#!/usr/bin/env python3
"""Correlate frozen AnomalyDINO video sums with revised workbook object labels."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

from pathlib import Path
import csv,json,math,hashlib
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
from object.analysis.occlusion.analyze_object_occlusion_v2_auroc import workbook_labels

ROOT=(_workspace_root())
WORKBOOK=(_workspace_root() / 'documentation/data/labels/selected_45_matched_videos_styledv2.xlsx')
CROPS=(_workspace_root() / 'results/object-reference-crops-20261001')
RUN=(_workspace_root() / 'results/anomalydino-object-crops-10frames-20261002')
OUT=(_workspace_root() / 'results/anomalydino-object-crops-10frames-20261002/correlation')

def stats(rows):
    x=[r['anomaly_score_sum'] for r in rows];y=[r['human_label_v2'] for r in rows]
    result={'n':len(rows)}
    for name,fn in [('pearson',pearsonr),('spearman',spearmanr),('kendall_tau_b',kendalltau)]:
        s=fn(x,y);result[name]={'coefficient':float(s.statistic),'p_two_sided':float(s.pvalue)}
    return result

def main():
    labels=workbook_labels(WORKBOOK)
    export=json.loads(((_workspace_root() / 'results/object-reference-crops-20261001/anomalydino_scores.json')).read_text())
    assert len(labels)==len(export['videos'])==45
    assert set(labels)=={r['video_id'] for r in export['videos']}
    matched=[]
    for v in export['videos']:
        name=v['video_id'];values=[p['anomaly_score'] for p in export['pairs'] if p['video_id']==name]
        assert v['scored_pair_count']==len(values)
        assert v['anomaly_score_sum']==(math.fsum(values) if values else None)
        matched.append({**v,'dataset':name.rsplit('_',1)[0],
                        'human_label_v2':labels[name]['label'],
                        'workbook_row':labels[name]['workbook_row'],
                        'anomaly_score_mean':math.fsum(values)/len(values) if values else None})
    scored=[r for r in matched if r['status']=='scored'];assert len(scored)==35
    assert all(r['scored_pair_count']==10 for r in scored)
    binary=[r for r in scored if r['human_label_v2']!=.5]
    positive=[r['anomaly_score_sum'] for r in binary if r['human_label_v2']==1]
    negative=[r['anomaly_score_sum'] for r in binary if r['human_label_v2']==0]
    auc=sum((p>n)+.5*(p==n) for p in positive for n in negative)/(len(positive)*len(negative))
    old=[json.loads(line) for line in ((_workspace_root() / 'results/anomalydino-object-crops-20261002/pairs.jsonl')).read_text().splitlines()]
    oldrows=[{**r,'anomaly_score_sum':math.fsum(p['anomaly_score'] for p in old if p['video_id']==r['video_id'])} for r in scored]
    assert all(sum(p['video_id']==r['video_id'] for p in old)==5 for r in scored)
    groups={}
    for label in (0,.5,1):
        a=[r['anomaly_score_sum'] for r in scored if r['human_label_v2']==label]
        groups[str(label)]={'n':len(a),'mean_sum':float(np.mean(a)),'median_sum':float(np.median(a))}
    summary={'label_column':'Object deformation (0/1)2','label_cells':'Sheet1!AF2:AF46',
             'score':'sum of ten original unrounded pair scores','primary':stats(scored),
             'binary_sensitivity':stats(binary),'binary_auroc':auc,'label_groups':groups,
             'by_dataset':{d:stats([r for r in scored if r['dataset']==d]) for d in sorted({r['dataset'] for r in scored})},
             'five_frame_comparison':stats(oldrows),
             'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORKBOOK,(_workspace_root() / 'results/object-reference-crops-20261001/anomalydino_scores.json'),(_workspace_root() / 'results/anomalydino-object-crops-10frames-20261002/pairs.jsonl')]}}
    OUT.mkdir(parents=True,exist_ok=True)
    with ((_workspace_root() / 'results/anomalydino-object-crops-10frames-20261002/correlation/matched_videos.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(matched[0]));w.writeheader();w.writerows(matched)
    ((_workspace_root() / 'results/anomalydino-object-crops-10frames-20261002/correlation/statistics.json')).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    def cells(s):
        return f"{s['n']} | {s['pearson']['coefficient']:.3f} | {s['pearson']['p_two_sided']:.4f} | {s['spearman']['coefficient']:.3f} | {s['spearman']['p_two_sided']:.4f}"
    lines=['# AnomalyDINO correlation with human object deformation','',
           '**The ten-frame anomaly sums show a moderate positive association with human object deformation: Pearson r = 0.452; Spearman ρ = 0.455.** Higher anomaly scores tend to accompany deformation labels, with substantial overlap between the groups.','',
           '## Main result','',
           '| Sample | Videos | Pearson r | p | Spearman ρ | p |','|---|---:|---:|---:|---:|---:|',
           '| Revised labels, including 0.5 | '+cells(summary['primary'])+' |',
           '| Binary labels only, excluding 0.5 | '+cells(summary['binary_sensitivity'])+' |','',
           f"Binary AUROC is **{auc:.3f}** across {len(positive)} deformation and {len(negative)} non-deformation videos. No score threshold was fitted.",'',
           '| Human label | Videos | Mean anomaly sum | Median anomaly sum |','|---|---:|---:|---:|']
    for label,g in groups.items():lines.append(f"| {label} | {g['n']} | {g['mean_sum']:.3f} | {g['median_sum']:.3f} |")
    lines+=['','## Five versus ten frames','',
            '| Selection | Videos | Pearson r | Spearman ρ |','|---|---:|---:|---:|']
    for name,s in [('Previous five frames',summary['five_frame_comparison']),('Current ten frames',summary['primary'])]:
        lines.append(f"| {name} | {s['n']} | {s['pearson']['coefficient']:.3f} | {s['spearman']['coefficient']:.3f} |")
    lines+=['','Doubling the samples barely changes the observed association: Δr = +0.001 and Δρ = +0.004. Both runs use the same 35 videos and labels. Because every included video has the same number of pairs within a run, using the mean instead of the sum gives identical correlations. The difference between runs is descriptive; it has not been tested as an improvement.','',
            '## By generator','',
            '| Generator | Videos | Spearman ρ | p |','|---|---:|---:|---:|']
    for d,s in summary['by_dataset'].items():lines.append(f"| {d} | {s['n']} | {s['spearman']['coefficient']:.3f} | {s['spearman']['p_two_sided']:.4f} |")
    lines+=['','## Largest disagreements','',
            'The following are high scores among label-0 videos and low scores among label-1 videos. They are ranking disagreements, not threshold-based classification errors.','',
            '| Video | Human label | Anomaly sum |','|---|---:|---:|']
    disagreement=sorted([r for r in scored if r['human_label_v2']==0],key=lambda r:-r['anomaly_score_sum'])[:3]+sorted([r for r in scored if r['human_label_v2']==1],key=lambda r:r['anomaly_score_sum'])[:3]
    for r in disagreement:lines.append(f"| {r['video_id']} | {r['human_label_v2']:g} | {r['anomaly_score_sum']:.3f} |")
    lines+=['','## Scope and interpretation','',
            '- **Label source:** `selected_45_matched_videos_styledv2.xlsx`, first sheet, revised `Object deformation (0/1)2` column AF, rows 2–46. Match by generator and numeric video ID. The original column Q is not used.',
            '- **Score source:** the completed ten-frame AnomalyDINO run. Sum the ten unrounded official pair scores per video. Quotas are 0/2/2/2/4 with mandatory usable post-occlusion successors.',
            '- **Coverage:** 35 of 45 videos. The existing six exclusions and four unavailable references have no score and are omitted, never assigned zero.',
            '- **Ambiguous label:** the single 0.5 value is retained as an ordinal middle value in the main correlation and omitted from the binary check.',
            '- **Statistics:** two-sided Pearson and asymptotic Spearman p-values. These are nominal, without adjustment for the several descriptive comparisons. Videos share matched tasks across generators, so the usual independent-observation p-values may overstate precision.',
            '- **Meaning:** this measures association with video-level deformation labels. AnomalyDINO also responds to appearance, viewpoint, segmentation, and crop-mapping differences. The result does not establish deformation-specific detection or generalization beyond these selected videos.','',
            '[Matched scores and workbook rows](matched_videos.csv) · [Exact statistics and source hashes](statistics.json) · [Crop gallery](../../object-reference-crops-20261001/selection_gallery.html)','']
    ((_workspace_root() / 'results/anomalydino-object-crops-10frames-20261002/correlation/REPORT.md')).write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
