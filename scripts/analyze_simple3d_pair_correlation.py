"""Read-only Selected45 AB analysis for terminal pair-visible results."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_simple3d_correlation import read_labels,metric,sha256
from experiments.simple3d_pipeline import write_csv,write_json


def analyze(root):
    root=Path(root)
    summary=json.loads((root/'metadata/summary.json').read_text())
    if summary['videos_processed']!=45 or summary['terminal_pairs']!=450:
        raise ValueError('label correlation requires all45 videos/450 pairs terminal; no partial-run inference')
    workbook=ROOT/'selected_45_matched_videos_styledv2.xlsx'
    selection=ROOT/'results/link5-only-selected45-updated-mask-20260929/selection.json'
    labels=read_labels(workbook,selection)
    records=json.loads((root/'comparisons.json').read_text())
    grouped=defaultdict(list)
    for row in records:grouped[row['video_id']].append(row)
    assert set(grouped)==set(labels)
    rows=[]
    for video,lab in labels.items():
        values=[r['scalar_anomaly_score'] for r in grouped[video] if r['simple3d_status']=='complete']
        assert len(grouped[video])==10
        rows.append(dict(video_id=video,dataset=lab['dataset'],human_label=lab['AB'],workbook_cell='AB'+str(lab['workbook_row']),
            expected_comparisons=10,successful_comparisons=len(values),
            mean_score=float(np.mean(values)) if values else None,max_score=max(values) if values else None,
            coverage_status='complete' if len(values)==10 else 'partial' if values else 'no_usable_scores'))
    metrics=[]
    for subset in ('all_scored','complete_videos'):
        for aggregation in ('mean_score','max_score'):
            for policy in ('ordinal','binary'):
                metrics.append(metric('link5_pair_visible',rows,subset=subset,aggregation=aggregation,labels=policy))
    for dataset in sorted({r['dataset'] for r in rows}):
        metrics.append(metric('link5_pair_visible',rows,dataset=dataset))
    write_csv(root/'metadata/video_scores_and_labels.csv',rows)
    write_csv(root/'metadata/label_metrics.csv',metrics)
    result=dict(status='complete',label_column='Selected 45!AB (Forearm deformation)',
        primary_aggregation='mean official frame scalar over successful pairs; no fabricated scores for invalid pairs',
        binary_policy='exclude0.5 labels, use0/1 only',videos_with_scores=sum(r['mean_score'] is not None for r in rows),
        complete_videos=sum(r['coverage_status']=='complete' for r in rows),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (workbook,selection,root/'comparisons.json')},
        metrics=metrics,limitation='Pair validity and uneven coverage can bias the scored cohort; metrics include coverage and complete-video sensitivity. No label tuning.')
    write_json(root/'metadata/label_correlation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'results/simple3d-20261003-run2/link5_pair_visible')
    print(json.dumps(analyze(p.parse_args().root),indent=2))
