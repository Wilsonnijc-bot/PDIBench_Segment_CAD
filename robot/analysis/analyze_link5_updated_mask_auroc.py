#!/usr/bin/env python3
"""Audit displayed link5 rigidity against forearm labels, reusing rank analysis."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse, csv, hashlib, json, math, re
from pathlib import Path
import robot.analysis.analyze_selected45_deformation_labels as shared
ROOT=(_workspace_root())
SCORES=(_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929')
OUTPUT=SCORES
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def labels(workbook,selection,column):
 shared.WORKBOOK=workbook;rows=shared.read_workbook_rows();out={}
 for item in selection['videos']:
  row=rows[int(item['workbook_row'])];case=f"{item['dataset']}_{int(item['video_number']):04d}"
  assert row['C']==shared.DATASET_NAMES[item['dataset']] and int(row['A'])==int(item['video_number']),case
  out[case]=shared.label_value(row.get(column))
 return out

def metrics(points,missing_highest=False):
 available=[(c,l,s) for c,l,s in points if l is not None and s is not None and math.isfinite(s)]
 missing=[(c,l) for c,l,s in points if l is not None and (s is None or not math.isfinite(s))]
 if missing_highest:
  highest=max(s for c,l,s in available)+1
  available += [(c,l,highest) for c,l in missing]
 auc,pairs=shared.ordered_concordance(available,severe_vs_none=True)
 concordance,ordered_pairs=shared.ordered_concordance(available)
 return {'auroc':auc,'valid_pairs':pairs,'wins_with_half_ties':auc*pairs,'ordered_concordance':concordance,'ordered_pairs':ordered_pairs,'valid_score_cases':len(available),'counts':{str(l):sum(label==l for c,label,s in available) for l in (0.,.5,1.)},'missing_score_cases':[c for c,l in missing],'missing_label_cases':[c for c,l,s in points if l is None],'missing_highest':missing_highest}
def replay_score(p):
 if not p.exists():return None
 with p.open() as stream:prefix=stream.read(25000)
 m=re.search(r'const data=.*?"score":([0-9.eE+-]+)',prefix,re.S)
 if not m:raise ValueError(f'Cannot identify replay score: {p}')
 return float(m.group(1))
def analyze(primary='v2'):
 selection=json.loads(((_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/selection.json')).read_text());summary=json.loads(((_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/summary.json')).read_text());rows=[]
 workbooks={'original':(_workspace_root() / 'documentation/data/labels/selected_45_matched_videos_styled.xlsx'),'v2':(_workspace_root() / 'documentation/data/labels/selected_45_matched_videos_styledv2.xlsx')}
 lbl={version:labels(p,selection,'AB') for version,p in workbooks.items()}
 for item in selection['videos']:
  case=f"{item['dataset']}_{int(item['video_number']):04d}";state=summary['cases'][case];path=(_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/cases')/case/'v1_cotracker3/metrics.json';m=json.loads(path.read_text())['modes']['exact-group']['objects']['link5'];score=m['breakdown']['epsilon_rigidity'] if m['status']=='complete' else None
  assert score==state['epsilon_rigidity'],case
  replay=(_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/cases')/case/'v1_cotracker3/replay/interactive_exact-group/link5_exact-group.html';raw=replay_score(replay)
  assert raw is None or abs(raw-score)<=.00005001,case
  rows.append({'case':case,'workbook_row':item['workbook_row'],'forearm_original':lbl['original'][case],'forearm_v2':lbl['v2'][case],'epsilon_rigidity':score,'replay_full_precision_score':raw,'status':m['status'],'metrics_sha256':sha(path)})
 results={version:metrics([(r['case'],r['forearm_'+version],r['epsilon_rigidity']) for r in rows]) for version in workbooks}
 precision={version:metrics([(r['case'],r['forearm_'+version],r['replay_full_precision_score']) for r in rows]) for version in workbooks}
 out={'score_source':str(SCORES.relative_to(ROOT)),'output_location':str(OUTPUT.relative_to(ROOT)),'label_column':'AB','positive_label':1,'negative_label':0,'moderate_label_policy':'exclude 0.5 from binary AUROC','score_field':'modes.exact-group.objects.link5.breakdown.epsilon_rigidity','primary_workbook':primary,'workbooks':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for k,p in workbooks.items()},'selection_sha256':sha((_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/selection.json')),'summary_sha256':sha((_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/summary.json')),'results':results,'full_precision_replay_sensitivity':precision,'label_disagreement_cases':[r['case'] for r in rows if r['forearm_original']!=r['forearm_v2']]}
 OUTPUT.mkdir(exist_ok=True);((_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/LINK5_FOREARM_AUROC.json')).write_text(json.dumps(out,indent=2)+'\n')
 with ((_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/LINK5_FOREARM_AUROC.csv')).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 result=results[primary];lines=['# Link5 rigidity versus forearm deformation','',f'**AUROC: {result["auroc"]:.3f}** using the **{primary} workbook**, forearm column **AB**, and the updated-mask link5 scores displayed on the review page. Higher rigidity predicts label 1; label 0 is the negative class. Label 0.5 is excluded.','',f'All **45 cases** have valid scores: **{result["counts"]["1.0"]} severe**, **{result["counts"]["0.0"]} none**, and **{result["counts"]["0.5"]} moderate.** Binary AUROC uses {result["valid_pairs"]} positive–negative pairs, with ties counting as half.','', '| Forearm labels | AUROC | Severe / none / moderate | Pairs |','| --- | ---: | ---: | ---: |']
 for version in ['v2','original']:
  r=results[version];lines.append(f'| {version} workbook | {r["auroc"]:.3f} | {r["counts"]["1.0"]} / {r["counts"]["0.0"]} / {r["counts"]["0.5"]} | {r["valid_pairs"]} |')
 lines+=['',f'**Result:** AUROC {result["auroc"]:.3f} is close to chance (0.5): updated-mask link5 provides little severe-versus-none separation on this selected set. This is descriptive performance, not a fitted threshold or a cross-validated estimate.','', '**Provenance:** Scores are from [`summary.json`](summary.json), checked against every case’s `v1_cotracker3/metrics.json`. The separate `link5-joint-prompt-20260929` folder holds SAM prompt tuning trials; it is not the score source. It does not use the older four-way run’s link5 scores or the original V1 link5 scores.','', 'The top-level workbooks differ in nine AB labels. Embedded `selection.json` labels and copied `input/` workbooks are not used as ground truth. Workbook rows are checked against dataset and video number. Reported scores are `epsilon_rigidity`, not PDI grades.','',f'**Precision check:** Using full-precision scores embedded in the original replay HTML gives AUROC {precision[primary]["auroc"]:.3f} ({primary} labels), versus {result["auroc"]:.3f} using reported metrics. Reported values follow the score field used by the existing deformation-label analyses.','', '**Audit files:** [case-level labels and scores](LINK5_FOREARM_AUROC.csv), [exact results and input hashes](LINK5_FOREARM_AUROC.json).','', 'Reproduce: `python3 scripts/analyze_link5_updated_mask_auroc.py --primary '+primary+'` from the repository root.','']
 ((_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/LINK5_FOREARM_AUROC.md')).write_text('\n'.join(lines));print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--primary',choices=['v2','original'],default='v2');analyze(p.parse_args().primary)
