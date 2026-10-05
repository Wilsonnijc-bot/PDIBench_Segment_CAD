#!/usr/bin/env python3
"""Recompute the overview from the score runs used by the displayed replays."""
import hashlib,json,shutil
from pathlib import Path
from analyze_link5_updated_mask_auroc import labels,metrics
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(out):
 selected=ROOT/'results/selected-45-v1';four=ROOT/'results/link5-link7-four-way-selected45-20260927';link5root=ROOT/'results/link5-only-selected45-updated-mask-20260929'
 sel=json.loads((four/'selection.json').read_text());summ=json.loads((four/'summary.json').read_text())
 gripper=labels(ROOT/'selected_45_matched_videos_styledv2.xlsx',sel,'AC');upper=labels(ROOT/'selected_45_matched_videos_styled.xlsx',sel,'AD')
 rows=[]
 for path,name in [('v1_cotracker3','V1 + CoTracker3'),('v2_tapip3d','V2 + TAPIP3D')]:
  data=[(c,gripper[c],s['paths'][path]['link7']['epsilon_rigidity']) for c,s in summ['cases'].items()]
  r=metrics(data);rows.append({'part':'Gripper · link7','method':name,'statistics':r,'policy':'Measured only · 7 unavailable excluded' if path=='v1_cotracker3' else '45 measured scores','source':'four_way','score_path':path})
  if path=='v1_cotracker3':rows.append({'part':'Gripper · link7','method':name,'statistics':metrics(data,True),'policy':'Sensitivity · 7 unavailable assigned the highest score','source':'four_way','score_path':path})
 data=[]
 for c in summ['cases']:
  p=selected/'cases'/c/'v1/metrics.json';score=None
  if p.exists():
   d=json.loads(p.read_text())['modes']['exact-group']['objects']['link2'];score=d['breakdown']['epsilon_rigidity'] if d['status']=='complete' else None
  data.append((c,upper[c],score))
 link2=metrics(data);assert round(link2['auroc'],3)==.941
 l5=json.loads((link5root/'LINK5_FOREARM_AUROC.json').read_text());assert l5['primary_workbook']=='v2'
 rows=[{'part':'Upper arm · link2','method':'V1 + CoTracker3','statistics':link2,'policy':'1 missing score; 2 nonnumeric labels excluded','source':'selected_v1','score_path':'v1/link2'}, {'part':'Forearm · link5','method':'V1 + CoTracker3 · updated mask','statistics':l5['results']['v2'],'policy':'45 measured scores','source':'updated_link5','score_path':l5['score_source']}]+rows
 assert [round(r['statistics']['auroc'],3) for r in rows]==[.941,.538,.845,.890,.842]
 dest=out/'analysis';dest.mkdir(exist_ok=True)
 docs={'selected_v1':selected/'DEFORMATION_LABEL_ANALYSIS.md','updated_link5':link5root/'LINK5_FOREARM_AUROC.md','four_way':four/'rigidity-vs-deformation-labels.md'}
 for p in docs.values():shutil.copy2(p,dest/p.name)
 for name in ['LINK5_FOREARM_AUROC.json','LINK5_FOREARM_AUROC.csv']:shutil.copy2(link5root/name,dest/name)
 audit={'definition':'Label 1 versus 0; exclude 0.5; higher reported epsilon_rigidity predicts deformation; ties count as half','rows':rows,'workbooks':{'link2':'selected_45_matched_videos_styled.xlsx, AD','link5':'selected_45_matched_videos_styledv2.xlsx, AB','link7':'selected_45_matched_videos_styledv2.xlsx, AC'},'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in list(docs.values())+[four/'summary.json',four/'selection.json']},'link5_precision_sensitivity':l5['full_precision_replay_sensitivity']['v2'],'v1_link7_replays':'results/link5-link7-four-way-selected45-20260927/cases/*/v1_cotracker3/replay/interactive_exact-group/link7_exact-group.html'}
 (dest/'DETECTION_AUROC.json').write_text(json.dumps(audit,indent=2))
 text='<section id="detection-summary" class="review-section auroc-summary"><h2>Detection AUROC · selected videos</h2><p>Label 1 (severe deformation) versus label 0 (almost none). Moderate label 0.5 is excluded. Higher rigidity predicts deformation; ties count as half.</p><div class="table-scroll"><table><thead><tr><th>Robot part</th><th>Detection method</th><th>AUROC</th><th>Severe / none</th><th>Availability treatment</th></tr></thead><tbody>'
 for r in rows:
  stats=r['statistics'];counts=stats['counts'];text+=f'<tr><th scope="row">{r["part"]}</th><td>{r["method"]}</td><td><strong>{stats["auroc"]:.3f}</strong></td><td>{counts["1.0"]} / {counts["0.0"]}</td><td>{r["policy"]}</td></tr>'
 text+='</tbody></table></div><p class="summary-note">The 0.890 link7 result is an assumed-score sensitivity check, not measured detection performance. All seven unavailable V1 + CoTracker3 cases have severe labels; assigning them a score above all measurements raises AUROC from 0.845 to 0.890. V2 + TAPIP3D scores all 45 videos.</p><p class="summary-note">Link2 uses the original workbook (AD); link5 uses v2 (AB); link7 uses v2 (AC). Gripper labels agree between the two workbooks. Link5 uses reported, rounded ε rigidity: AUROC 0.538; full-precision replay scores give 0.542. These values describe the selected videos.</p><details><summary>Analysis sources and exact results</summary><div class="links">'
 for key,p in docs.items():text+=f'<a target="_blank" href="analysis/{p.name}">{ {"selected_v1":"Link2 analysis","updated_link5":"New link5 analysis","four_way":"Link7 four-way analysis"}[key]}</a>'
 text+='<a target="_blank" href="analysis/DETECTION_AUROC.json">Exact AUROCs and provenance</a></div></details></section>'
 return text
