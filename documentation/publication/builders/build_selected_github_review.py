#!/usr/bin/env python3
"""Publish selected, unchanged replay evidence with lossless HTML transport."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse, copy, gzip, hashlib, json, re, shutil, sys
from pathlib import Path
from urllib.parse import urlsplit
try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.path.insert(0, str((_workspace_root() / '.tmp/reviewdeps')))
    from bs4 import BeautifulSoup
ROOT=(_workspace_root())
OUT=(_workspace_root() / '.tmp/selected-review-site')
VIDEOS=(_workspace_root() / '.tmp/selected-review-videos')
RELEASE='https://github.com/Wilsonnijc-bot/robot_object_deformation_detect/releases/download/selected-review-20261002/'
records={}; hashes={}; html_queue=[]
def register(p):
    p=Path(p).resolve()
    key=str(p.relative_to(ROOT))
    if key in records:return records[key]['url']
    if not p.is_file():raise FileNotFoundError(p)
    raw=p.read_bytes(); h=hashlib.sha256(raw).hexdigest();suffix=p.suffix
    name=h+suffix
    url=('replay.html?asset='+name+'.gz') if suffix=='.html' else ('assets/'+name)
    if suffix in ['.mp4','.png','.jpg']:url=RELEASE+name
    if 'results/object-reference-crops-20261001/' in key and suffix in ['.png','.jpg']:
        url=RELEASE.replace('selected-review-20261002/',f'selected-review-objects-20261002-{int(h[:8],16)%3}/')+name
    records[key]={'sha256':h,'bytes':len(raw),'url':url}
    if (h,suffix) not in hashes:
        hashes[h,suffix]=p
        if suffix=='.html':
            target=(_workspace_root() / '.tmp/selected-review-site/assets')/(name+'.gz')
            if not target.exists():target.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
            html_queue.append(p)
        elif suffix in ['.mp4','.png','.jpg']:shutil.copy2(p,VIDEOS/name)
        else:shutil.copy2(p,(_workspace_root() / '.tmp/selected-review-site/assets')/name)
    return url

def remap(node,base):
    for e in [node]+list(node.find_all(True)):
        for attr in ['href','src','data-src']:
            val=e.get(attr)
            if val and not val.startswith(('http:','https:','#','data:')):
                e[attr]=register(base/val)
        if e.name=='option' and e.get('value','').endswith('.html'):e['value']=register(base/e['value'])
    return node

def parse(p):return BeautifulSoup(p.read_text(),'html.parser')
def section(id,title,text):return f'<section id="{id}" class="review-section"><h2>{title}</h2>{text}</section>'
def evidence(title,steps,metadata,video=None,note=''):
    body=f'<details class="evidence"><summary>{title}</summary><p>{note}</p><div class="stepper">'
    for i,(label,p) in enumerate(steps):body+=f'<button type="button" data-image="{register(p)}">{label}</button>'
    if steps:body+=f'<figure><img loading="lazy" src="{register(steps[0][1])}" alt="{title} · {steps[0][0]}"><figcaption>{steps[0][0]}</figcaption></figure>'
    body+='</div>'
    if video:body+=f'<video controls preload="none" playsinline src="{register(video)}"></video>'
    body+='<details><summary>Exact prompt and provenance</summary>'
    for p in metadata:body+=f'<a href="{register(p)}" target="_blank">{p.name}</a> '
    body+='</details></details>'
    return body

def main():
    OUT.mkdir(parents=True,exist_ok=True);((_workspace_root() / '.tmp/selected-review-site/assets')).mkdir(exist_ok=True);VIDEOS.mkdir(exist_ok=True)
    base=(_workspace_root() / 'results/selected-45-v1');s=parse(base/'review.html');originals=[base/'review.html']
    completed=copy.deepcopy(s.find('h2',string='Completed cases').find_next_sibling('ul'))
    for li in completed.find_all('li',recursive=False):
        for a in li.find_all('a'):
            if a.get_text(strip=True) not in ['link2']:a.decompose()
        for span in li.select('.links span'):
            links=[x for x in ['link2','link7'] if x in span.text]
            if links:span.string='Unscored: '+', '.join(links)
            else:span.decompose()
        li['data-case']=li.find('strong').text
    remap(completed,base)
    failed=copy.deepcopy(s.find('h2',string='Failed cases').find_next_sibling('ul'));remap(failed,base)
    base4=(_workspace_root() / 'results/link5-link7-four-way-selected45-20260927');p=base4/'index.html';originals.append(p);table=copy.deepcopy(parse(p).find('table'))
    for row in table.find_all('tr'):
        cells=row.find_all(['th','td'],recursive=False)
        for i,c in enumerate(cells):
            if i not in [0,1,3,6]:c.decompose()
    remap(table,base4)
    part2=section('v2tapip','Link7 · v2_tapip3d', '<div class="table-scroll">'+str(table)+'</div>')
    base5=(_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929');p=base5/'index.html';originals.append(p);table5=copy.deepcopy(parse(p).find('table'));remap(table5,base5)
    part3=section('link5','Link5 · updated mask · all 45 cases','<div class="table-scroll">'+str(table5)+'</div>')
    from documentation.publication.builders.build_representative_mask_review import build
    representative_review=build(register)
    # Group extracted cells and replay links by the original video's case identity.
    completed_by_case={li.find('strong').text:li for li in completed.find_all('li',recursive=False)}
    failed_by_case={li.find('strong').text:li for li in failed.find_all('li',recursive=False)}
    v2_by_case={row.find('th').text:row for row in table.select('tbody tr')}
    cases=[]
    for row in table5.select('tbody tr'):
        cells=row.find_all(['th','td'],recursive=False);case_id=cells[0].get_text(strip=True)
        source=base5/'cases'/case_id/'v1_cotracker3/replay/interactive_exact-group/source.mp4'
        vid=register(source) if source.is_file() else None
        original_links=completed_by_case.get(case_id)
        def selected_link(link):
            return original_links.find('a',string=link) if original_links else None
        def heading(title,anchor):
            if anchor:
                a=copy.deepcopy(anchor);a.clear();a.append(title);a['class']='replay-title'
                return '<th>'+str(a)+'</th>'
            return '<th><span class="unavailable">'+title+'<small>Unscored / unavailable</small></span></th>'
        v2_cells=v2_by_case[case_id].find_all(['th','td'],recursive=False)
        video=f'<details><summary>Original video</summary><video controls preload="none" playsinline src="{vid}"></video></details>' if vid else ''
        headers=heading('Link2 · V1',selected_link('link2'))+heading('Link7 · V1 + CoTracker3',v2_cells[2].find('a'))+heading('Link7 · V2 + TAPIP3D',v2_cells[-1].find('a'))+heading('Link5 · updated mask',cells[3].find('a'))
        details=f'<details class="case-group" data-case="{case_id}"><summary><strong>{case_id}</strong> · {cells[1].get_text(strip=True)}</summary>{video}<div class="table-scroll"><table class="replay-titles"><thead><tr>{headers}</tr></thead></table></div>'
        if case_id in failed_by_case:details+='<h3>Original link2 run · failure diagnostics</h3>'+str(failed_by_case[case_id])
        details+='</details>';cases.append(details)
    grouped=section('cases','Replays by original video','<p>Select a link title to open its interactive point-cloud replay.</p>'+''.join(cases))+representative_review
    from documentation.publication.builders.build_review_object_sections import build as build_objects
    object_sections,object_originals=build_objects(OUT,register,parse,remap)
    grouped+=object_sections;originals+=object_originals
    # Recursively collect HTML dependencies. Preserve each original HTML byte stream in gzip.
    for p in html_queue:
        raw=p.read_text()
        for val in re.findall(r'(?:href|src)=["\']([^"\']+)["\']',raw):
            if not val.startswith(('http:','https:','#','data:')):register(p.parent/val)
        for val in re.findall(r'"evidence_file"\s*:\s*"([^"]+)"',raw):register(p.parent/val)
    from documentation.publication.builders.build_review_auroc_summary import build as build_auroc
    auroc_summary=build_auroc(OUT)
    original_style=s.find('style').text
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PDI selected review · point clouds and SAM prompts</title><style>'''+original_style+'''
nav{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0 30px}.review-section{border-top:1px solid #bdc9d3;margin-top:32px;padding-top:8px}.table-scroll{overflow:auto}td,th{padding:10px 12px}.score{font-variant-numeric:tabular-nums;font-weight:650}.replay{display:block}.evidence{padding:12px 0;border-bottom:1px solid #dce2e8}.evidence summary{cursor:pointer;font-weight:650}.stepper button{font:inherit;margin:5px;padding:8px;background:#eef5ff;border:1px solid #bacbd8;color:#17212b;cursor:pointer}.stepper img{width:100%;max-height:650px;object-fit:contain}.stepper figure{max-width:100%;margin:12px 0}#viewer{position:fixed;inset:2vh 2vw;background:#f6f7f8;z-index:100;border:1px solid #aebac5;box-shadow:0 0 0 100vmax #17212b99;display:none}#viewer.open{display:flex;flex-direction:column}#viewer header{padding:10px;display:flex;align-items:center;gap:18px}#viewer iframe{width:100%;flex:1;border:0}#viewer button{padding:8px 16px;font:inherit}#viewer-label{flex:1}input[type=search]{font:inherit;padding:10px;width:min(100%,480px);margin-bottom:16px}a:focus-visible,button:focus-visible{outline:3px solid #07579c;outline-offset:3px}details>summary{list-style:disclosure-closed}details[open]>summary{list-style:disclosure-open}.case-group{border-bottom:1px solid #dce2e8;padding:16px 0}.case-group>summary{cursor:pointer;font-size:18px}.case-group[open]>summary{margin-bottom:18px}.case-group h3{font-size:16px}.hidden{display:none!important}.notice{margin:10px 0 24px}.auroc-summary h2{margin-top:10px}.auroc-summary th,.auroc-summary td{white-space:normal}.summary-note{font-size:14px;color:#425463}.replay-titles th{padding:0;background:transparent}.replay-title{display:block;padding:15px 12px;margin:0;border-radius:0;background:#eef3f7;color:#07579c}.replay-title:hover{background:#dcebf7}.unavailable{display:block;padding:15px 12px;color:#687985;background:#eef0f2}.unavailable small{display:block;font-size:11px;font-weight:400}.representative{margin:28px 0 48px;border-top:1px solid #dce2e8;padding-top:18px}.representative h3{font-size:21px}.type-title{font-size:24px;margin-top:38px}.prompt-picture{max-width:900px}.prompt-picture img{width:100%;max-height:none}.mask-pair,.point-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.mask-pair figure,.point-pair figure{max-width:none;min-width:0}.mask-pair figcaption{margin-bottom:8px;font-weight:650}.point-pair a{display:block;padding:0;background:transparent}.point-pair img{width:100%;height:auto;max-height:none}.mask-crop{position:relative;overflow:hidden;background:#132028}.mask-crop video{position:absolute;width:200%;max-width:none;height:100%;left:-100%;top:0;margin:0;object-fit:fill}.mask-controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:16px 0}.mask-controls button{font:inherit;padding:8px 12px;border:1px solid #b5c8d6;background:#eef5ff;color:#07579c;cursor:pointer}.mask-frame{flex:1;min-width:140px}.mask-controls output{font-size:13px;font-variant-numeric:tabular-nums}.provenance{margin-top:14px}.provenance a{margin:8px 8px 0 0}@media(max-width:760px){.mask-pair,.point-pair{grid-template-columns:1fr}.mask-controls output{width:100%}.replay-titles{min-width:730px}}
</style></head><body><main><h1>PDI · selected review</h1><p class="sub">Original evidence · selected links · interactive 3D point clouds</p><nav><a href="#detection-summary">Detection AUROC</a><a href="#cases">45 original videos · grouped replays</a><a href="#mask-refinement">Representative SAM3 refinements</a><a href="#object-reference">Object reference crops</a><a href="#object-occlusion">Occlusion catches</a></nav>'''+auroc_summary+'''<label>Find a case <input id="search" type="search" placeholder="e.g. COSMOS3_0010"></label><p class="notice">Choose a replay to play, scrub, rotate the point cloud, and inspect point pairs alongside the original video. Failed and unscored cases are preserved.</p>'''+grouped+'''</main><div id="viewer" role="dialog" aria-modal="true" aria-label="Interactive point cloud replay"><header><strong id="viewer-label">Replay</strong><a id="full" target="_blank">Open full page</a><button id="close" type="button">Close</button></header><iframe title="Interactive point cloud"></iframe></div><script>
const viewer=document.getElementById('viewer');let focusBack;
document.addEventListener('click',e=>{const a=e.target.closest('a');if(a&&a.getAttribute('href')?.startsWith('replay.html?')){e.preventDefault();focusBack=a;viewer.classList.add('open');viewer.querySelector('iframe').src=a.href;document.getElementById('full').href=a.href;document.getElementById('viewer-label').textContent=(a.closest('.case-group')?.dataset.case||'')+' · '+a.textContent;document.getElementById('close').focus();document.body.style.overflow='hidden';}const b=e.target.closest('button[data-image]');if(b){const box=b.closest('.stepper');box.querySelector('img').src=b.dataset.image;box.querySelector('img').alt=b.textContent;box.querySelector('figcaption').textContent=b.textContent;}});
function closeReplay(){viewer.classList.remove('open');viewer.querySelector('iframe').removeAttribute('src');document.body.style.overflow='';focusBack?.focus()}document.getElementById('close').onclick=closeReplay;document.addEventListener('keydown',e=>{if(e.key==='Escape')closeReplay();if(e.key==='Tab'&&viewer.classList.contains('open')){const els=viewer.querySelectorAll('a,button,iframe');if(e.shiftKey&&document.activeElement===els[0]){e.preventDefault();els[els.length-1].focus()}else if(!e.shiftKey&&document.activeElement===els[els.length-1]){e.preventDefault();els[0].focus()}}});
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.case-group').forEach(el=>el.classList.toggle('hidden',!el.textContent.toLowerCase().includes(q)))});
document.querySelectorAll('.persistent').forEach(box=>{const videos=[...box.querySelectorAll('video')],fps=Number(box.dataset.fps),n=Number(box.dataset.frames),slider=box.querySelector('.mask-frame'),readout=box.querySelector('output'),button=box.querySelector('.mask-play');let running=false,raf=0;function paint(){const frame=Math.min(n-1,Math.round(videos[0].currentTime*fps));slider.value=frame;readout.textContent=`Frame ${frame} / ${n-1}`;if(running){if(Math.abs(videos[0].currentTime-videos[1].currentTime)>.1)videos[1].currentTime=videos[0].currentTime;if(videos[0].ended){stop();return}raf=requestAnimationFrame(paint)}}function stop(){running=false;videos.forEach(v=>v.pause());button.textContent='Play comparison';cancelAnimationFrame(raf)}async function load(){await Promise.all(videos.map(v=>v.readyState>=1?Promise.resolve():new Promise((resolve,reject)=>{v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',()=>reject(Error('Mask video failed to load')),{once:true});v.preload='metadata';v.load()})))}async function seek(frame){stop();await load();await Promise.all(videos.map(v=>new Promise(resolve=>{const t=frame/fps;if(Math.abs(v.currentTime-t)<.001){resolve();return}v.addEventListener('seeked',resolve,{once:true});v.currentTime=t})));paint()}button.onclick=async()=>{if(running){stop();return}try{await load();if(videos[0].ended)await seek(0);videos[1].currentTime=videos[0].currentTime;await Promise.all(videos.map(v=>v.play()));running=true;button.textContent='Pause comparison';paint()}catch(error){readout.textContent=error.message;stop()}};slider.onchange=()=>seek(Number(slider.value)).catch(e=>readout.textContent=e.message);box.querySelector('.mask-seed').onclick=()=>seek(Number(box.dataset.seed)).catch(e=>readout.textContent=e.message)});
</script></body></html>'''
    ((_workspace_root() / '.tmp/selected-review-site/index.html')).write_text(page)
    manifest={'files':records,'source_indexes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals},'selection':{'selected45':['link2'],'four_way':['v1_cotracker3/link7','v2_tapip3d/link7'],'updated_link5':'all','representative_persistent_cases':['Cosmos3_0065','Cosmos3_0015','LVP_0010'],'representative_link5_cases':['COSMOS3_0001','COSMOS2.5_0056','LVP_ROBOWM_0021']},'replay_transport':'gzip, exact bytes after decompression'}
    ((_workspace_root() / '.tmp/selected-review-site/manifest.json')).write_text(json.dumps(manifest,indent=2))
    keep={v['url'].split('asset=')[1] if v['url'].startswith('replay.html?') else v['url'][7:] for v in records.values() if v['url'].startswith(('replay.html?','assets/'))}
    for asset in ((_workspace_root() / '.tmp/selected-review-site/assets')).iterdir():
        if asset.name not in keep:asset.unlink()
    ((_workspace_root() / '.tmp/selected-review-site/.nojekyll')).touch()
    ((_workspace_root() / '.tmp/selected-review-site/replay.html')).write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loading original interactive replay</title><style>html,body{margin:0;height:100%;background:#12181a;color:#edf2ed;font:16px system-ui}iframe{width:100%;height:100%;border:0}p{padding:24px}</style><p id="status">Loading original replay…</p><script>
(async()=>{try{const asset=new URLSearchParams(location.search).get('asset');if(!/^[a-f0-9]{64}\\.html\\.gz$/.test(asset||''))throw Error('Invalid replay');const manifest=await(await fetch('manifest.json',{cache:'no-store'})).json();const entry=Object.entries(manifest.files).find(([k,v])=>v.url==='replay.html?asset='+asset);if(!entry)throw Error('Unknown replay');const response=await fetch('assets/'+asset);if(!response.ok)throw Error('Replay download failed');const raw=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).text();const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(raw)))).map(b=>b.toString(16).padStart(2,'0')).join('');if(hash!==entry[1].sha256)throw Error('Replay integrity check failed');const doc=new DOMParser().parseFromString(raw,'text/html');const origin=new URL(entry[0], 'https://source.invalid/');for(const el of doc.querySelectorAll('[href],[src]'))for(const attr of ['href','src']){const val=el.getAttribute(attr);if(!val||/^(https?:|data:|#)/.test(val))continue;const path=new URL(val,origin).pathname.slice(1);const mapped=manifest.files[path];if(mapped)el.setAttribute(attr,new URL(mapped.url,location.href).href);}const ev=raw.match(/\"evidence_file\"\s*:\s*\"([^\"]+)\"/);if(ev){const key=new URL(ev[1],origin).pathname.slice(1);if(manifest.files[key]){const fix=doc.createElement('script');fix.textContent='document.getElementById(\"evidence\").href='+JSON.stringify(new URL(manifest.files[key].url,location.href).href)+';';doc.body.append(fix);}}const base=doc.createElement('base');base.href=location.href;doc.head.prepend(base);const frame=document.createElement('iframe');frame.onload=()=>{const id=decodeURIComponent(location.hash.slice(1));if(id)frame.contentDocument?.getElementById(id)?.scrollIntoView()};frame.title='Original interactive replay';frame.srcdoc='<!doctype html>'+doc.documentElement.outerHTML;document.body.replaceChildren(frame);}catch(error){document.getElementById('status').textContent='Could not load replay: '+error.message;}})();</script></html>''')
    print(json.dumps({'files':len(records),'site_MB':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())/1e6,'videos_MB':sum(p.stat().st_size for p in VIDEOS.glob('*'))/1e6,'video_assets':len(list(VIDEOS.glob('*')))},indent=2))
if __name__=='__main__':main()
