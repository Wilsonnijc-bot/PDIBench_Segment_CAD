#!/usr/bin/env python3
"""Render exact recorded Gemini SAM points and representative mask comparisons."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import base64, hashlib, html, json, subprocess, shutil
from fractions import Fraction
from pathlib import Path
ROOT=(_workspace_root())
OUT=(_workspace_root() / '.tmp/selected-review-site/representative')
PERSISTENT=('Cosmos3_0065','Cosmos3_0015','LVP_0010')
GUARDS=('COSMOS3_0001','COSMOS2.5_0056','LVP_ROBOWM_0021')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def raw_frame(video,dest,frame=0):
 subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{frame})','-frames:v','1','-y',str(dest)],check=True)
def figure_svg(raw,points,labels,box,crop,title,subtitle,dest,changed=()):
 data=base64.b64encode(raw.read_bytes()).decode();w,h=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height','-of','json',str(raw)]))['streams'][0].values()
 shapes=[f'<rect x="{box[0]}" y="{box[1]}" width="{box[2]-box[0]}" height="{box[3]-box[1]}" fill="none" stroke="#6ec5ec" stroke-width="2" stroke-dasharray="7 5"/>']
 counts={0:0,1:0}
 for i,((x,y),label) in enumerate(zip(points,labels)):
  counts[label]+=1;name=('P' if label else 'N')+str(counts[label]);color='#39e58c' if label else '#ff6262';dy=25 if label else -17
  if i in changed:shapes.append(f'<circle cx="{x}" cy="{y}" r="12" fill="none" stroke="#ffd45c" stroke-width="3"/>')
  shapes.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" stroke="#13242c" stroke-width="2"/><text x="{x+10}" y="{y+dy}" fill="{color}" font-size="20" font-weight="700" stroke="#13242c" stroke-width="3" paint-order="stroke">{name}</text>')
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="610" viewBox="0 0 1200 610"><rect width="1200" height="610" fill="#132028"/><text x="24" y="34" fill="#edf3f6" font-family="sans-serif" font-size="23" font-weight="700">{html.escape(title)}</text><text x="24" y="67" fill="#c7d6df" font-family="sans-serif" font-size="18">{html.escape(subtitle)}</text><svg x="16" y="88" width="1168" height="470" viewBox="{' '.join(map(str,crop))}" preserveAspectRatio="xMidYMid meet"><image href="data:image/png;base64,{data}" width="{w}" height="{h}"/>{''.join(shapes)}</svg><text x="24" y="587" fill="#39e58c" font-family="sans-serif" font-size="18">● Positive · include forearm</text><text x="350" y="587" fill="#ff6262" font-family="sans-serif" font-size="18">● Negative · exclude adjacent parts</text><text x="785" y="587" fill="#ffd45c" font-family="sans-serif" font-size="18">○ Point moved by Gemini</text></svg>'''
 dest.write_text(svg)
def build(register):
 OUT.mkdir(parents=True,exist_ok=True);cards=[];provenance=[]
 for folder in ((_workspace_root() / '.tmp/selected-review-site/representative/link5')).glob('*'):
  if folder.is_dir() and folder.name not in GUARDS:shutil.rmtree(folder)
 for name in PERSISTENT:
  case=(_workspace_root() / 'results/selected-45-v1/persistent_work/local/selected_45_v1')/name
  seed=case/'sam'/name/'seed.json';d=json.loads(seed.read_text());metrics=case/'sam'/name/'seed_metrics.json';m=json.loads(metrics.read_text())
  assert d['status']=='ready' and m['membership_ok']
  assert all(c['requested_model']=='gemini-3.8-flash' for c in d['calls'])
  assert m['points']==d['points'] and m['labels']==d['labels']
  art=case/'artifacts'/name;naive=case/'naive'/name/'naive_masking.mp4';refined=art/'masking.mp4'
  streams=[]
  for p in [naive,refined]:streams.append(json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate,nb_frames','-of','json',str(p)]))['streams'][0])
  assert streams[0]==streams[1]
  fps=float(Fraction(streams[0]['r_frame_rate']));n=int(streams[0]['nb_frames'])
  id=name.replace('Cosmos25_','COSMOS2.5_').replace('Cosmos3_','COSMOS3_').replace('LVP_','LVP_ROBOWM_')
  cards.append(f'''<article class="representative persistent" data-case="{id}" data-fps="{fps}" data-frames="{n}" data-seed="{d['frame']}"><h3>{id}</h3><p>Gemini 3.8 Flash SAM3 prompt · seed frame {d['frame']} · 3 positive points and 2 negative points.</p><figure class="prompt-picture"><img loading="lazy" src="{register(art/'points.png')}" alt="{id}: exact Gemini positive and negative points used by SAM3"><figcaption>Green: positive · red: negative. These are the recorded Gemini points at the SAM3 reseed frame.</figcaption></figure><div class="mask-controls"><button class="mask-play">Play comparison</button><button class="mask-seed">Go to SAM prompt frame</button><input class="mask-frame" type="range" min="0" max="{n-1}" value="0" aria-label="{id} mask comparison frame"><output>Frame 0 / {n-1}</output></div><div class="mask-pair"><figure><figcaption>Naive masking · frame-zero initialization</figcaption><div class="mask-crop" style="aspect-ratio:{d['source_hw'][1]}/{d['source_hw'][0]}"><video muted preload="none" playsinline src="{register(naive)}"></video></div></figure><figure><figcaption>Persistent masking · Gemini SAM3 prompt</figcaption><div class="mask-crop" style="aspect-ratio:{d['source_hw'][1]}/{d['source_hw'][0]}"><video muted preload="none" playsinline src="{register(refined)}"></video></div></figure></div><details class="provenance"><summary>Recorded Gemini prompt</summary><a target="_blank" href="{register(seed)}">Exact points and Gemini responses</a><a target="_blank" href="{register(metrics)}">SAM seed validation</a></details></article>''')
  provenance.append({'type':'persistent','case':id,'seed':str(seed.relative_to(ROOT)),'seed_sha256':sha(seed),'points':d['points'],'labels':d['labels'],'frame':d['frame'],'model':'gemini-3.8-flash','naive_sha256':sha(naive),'refined_sha256':sha(refined)})
 section7='<h3 class="type-title">Link7 · persistent masking</h3><p>Three representative cases. Inspect the exact Gemini prompt, then play or scrub the naive and persistent masks on the same frame.</p>'+''.join(cards)
 cards=[]
 for name in GUARDS:
  case=(_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/cases')/name;b=case/'base_generation';record=b/'link5_guard.json';g=json.loads(record.read_text());diagnostic=b/'sam3_prompt_diagnostics.json';d=json.loads(diagnostic.read_text())[0];ref=d['point_refinement']
  assert all(a['requested_model']=='gemini-3.8-flash' for a in g['attempts'])
  assert g['selected_points_xy']==ref['points_xy'] and not g['fallback_to_default']
  folder=(_workspace_root() / '.tmp/selected-review-site/representative/link5')/name;folder.mkdir(parents=True,exist_ok=True);raw=folder/'source_frame.png';video=case/'v1_cotracker3/replay/interactive_exact-group/source.mp4';raw_frame(video,raw)
  pts=g['original_points_xy']+g['selected_points_xy'];box=d['box_xyxy'];minx=max(0,min(x for x,y in pts)-80);maxx=max(x for x,y in pts)+80;maxy=max(y for x,y in pts)+100;crop=[minx,0,maxx-minx,maxy]
  changed=[i for i,(a,z) in enumerate(zip(g['original_points_xy'],g['selected_points_xy'])) if a!=z]
  negative=g['reviews']['negative']['decision'];positive=g['reviews']['positive']['decision'];summary=f'Negative: {negative} · Positive: {positive} · '+('keep all points' if not changed else 'updated '+', '.join(('P'+str(i+1) if i<3 else 'N'+str(i-2)) for i in changed))
  figure_svg(raw,g['original_points_xy'],ref['point_labels'],box,crop,name+' · initial SAM3 points','Before Gemini review · 3 positive points + 3 negative points',folder/'initial.svg')
  figure_svg(raw,g['selected_points_xy'],ref['point_labels'],box,crop,name+' · Gemini 3.8 Flash decision',summary,folder/'decision.svg',changed)
  meta={'case':name,'type':'link5_guard','model':'gemini-3.8-flash','source_record':str(record.relative_to(ROOT)),'source_record_sha256':sha(record),'diagnostic_sha256':sha(diagnostic),'source_video_sha256':sha(video),'frame_index':0,'frame_sha256':sha(raw),'original_points_xy':g['original_points_xy'],'selected_points_xy':g['selected_points_xy'],'point_labels':ref['point_labels'],'decisions':{'negative':negative,'positive':positive},'changed_point_indices':changed,'gemini_answers':[{'role':a['role'],'answer':a['answer']} for a in g['attempts']]}
  (folder/'provenance.json').write_text(json.dumps(meta,indent=2));provenance.append(meta)
  cards.append(f'''<article class="representative guard" data-case="{name}"><h3>{name}</h3><p>{summary}</p><div class="point-pair"><figure><a target="_blank" href="representative/link5/{name}/initial.svg"><img loading="lazy" width="1200" height="610" src="representative/link5/{name}/initial.svg" alt="{name}: all initial positive and negative points"></a><figcaption>1 · Initial positive and negative points</figcaption></figure><figure><a target="_blank" href="representative/link5/{name}/decision.svg"><img loading="lazy" width="1200" height="610" src="representative/link5/{name}/decision.svg" alt="{name}: Gemini decisions and exact points used by SAM3"></a><figcaption>2 · Gemini decision and points used by SAM3</figcaption></figure></div><details class="provenance"><summary>Recorded Gemini decision</summary><a target="_blank" href="{register(record)}">Gemini review and responses</a><a target="_blank" href="representative/link5/{name}/provenance.json">Exact point coordinates</a></details></article>''')
 section5='<h3 class="type-title">Link5 · Gemini masking guard</h3><p>These recorded prompts come from the latest updated-mask scoring run. Earlier joint-prompt experiments are SAM tuning trials, separate from detection results. Three representative decisions: replace negative points, replace both point types, and keep points unchanged. Both pictures use the same source frame and crop; gold rings mark points Gemini moved.</p>'+''.join(cards)
 ((_workspace_root() / '.tmp/selected-review-site/representative/selection.json')).write_text(json.dumps({'persistent_cases':list(PERSISTENT),'link5_cases':list(GUARDS),'records':provenance},indent=2))
 return '<section id="mask-refinement" class="review-section"><h2>Representative SAM3 prompt and mask refinements</h2>'+section7+section5+'</section>'
