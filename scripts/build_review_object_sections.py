#!/usr/bin/env python3
"""Append the original crop gallery and occlusion catches to the review."""
import base64, gzip, hashlib, json, shutil
from bs4 import BeautifulSoup
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def scored_cases_first(page):
 cases=page.select('section.case')
 footer=page.select_one('main > .footer')
 for case in sorted(cases,key=lambda item:item.get('data-scored')!='1'):
  footer.insert_before(case.extract())

def build(out, register, parse, remap):
 gallery=ROOT/'results/object-reference-crops-20261001/selection_gallery.html'
 catches=ROOT/'results/object-deformation-selected45-20260929/occlusion/replay/index.html'
 analysis=ROOT/'results/anomalydino-object-crops-10frames-20261002/correlation'
 stats=json.loads((analysis/'statistics.json').read_text())
 assert stats['binary_sensitivity']['n']==34
 page=parse(gallery)
 scored_cases_first(page)
 packs={};packdir=ROOT/'web-assets/object-image-packs';packdir.mkdir(parents=True,exist_ok=True)
 for el in page.select('[src],[href]'):
  for attr in ['src','href']:
   val=el.get(attr)
   if not val or val.startswith(('http:','https:','#','data:')):continue
   path=gallery.parent/val
   if path.suffix in ('.png','.jpg'):
    case=Path(val).parts[0];packs.setdefault(case,{})[val]=base64.b64encode(path.read_bytes()).decode()
    el['data-image-path']=val;el['data-image-case']=case
    el[attr]='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7' if attr=='src' else '#'
   else:el[attr]=register(path)
 urls={}
 for case,data in packs.items():
  raw=gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0);name=hashlib.sha256(raw).hexdigest()+'.json.gz';(packdir/name).write_bytes(raw)
  urls[case]='https://raw.githubusercontent.com/Wilsonnijc-bot/PDIBench_Segment_CAD/main/web-assets/object-image-packs/'+name
 script=page.new_tag('script');script.string='const imagePacks='+json.dumps(urls)+""";
 const imageJobs=new Map();async function loadImages(caseName){if(imageJobs.has(caseName))return imageJobs.get(caseName);const job=(async()=>{const r=await fetch(imagePacks[caseName]);if(!r.ok)throw Error('Image pack download failed');const compressed=await r.arrayBuffer();const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',compressed))).map(b=>b.toString(16).padStart(2,'0')).join('');if(!imagePacks[caseName].endsWith(digest+'.json.gz'))throw Error('Image pack integrity check failed');const data=await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).json();for(const el of document.querySelectorAll('[data-image-case]'))if(el.dataset.imageCase===caseName){const raw=atob(data[el.dataset.imagePath]);const bytes=Uint8Array.from(raw,c=>c.charCodeAt(0));const url=URL.createObjectURL(new Blob([bytes],{type:el.dataset.imagePath.endsWith('.jpg')?'image/jpeg':'image/png'}));el.setAttribute(el.tagName==='IMG'?'src':'href',url);}})();imageJobs.set(caseName,job);return job;}
 const imageObserver=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){loadImages(e.target.dataset.imageCase).catch(err=>{e.target.alt=err.message;imageJobs.delete(e.target.dataset.imageCase)});imageObserver.unobserve(e.target);}},{rootMargin:'600px'});document.querySelectorAll('img[data-image-case]').forEach(el=>imageObserver.observe(el));
 document.addEventListener('click',async event=>{const link=event.target.closest('a[data-image-path]');if(link&&link.getAttribute('href')==='#'){event.preventDefault();await loadImages(link.dataset.imageCase);window.open(link.href,'_blank');}});
 """;page.body.append(script)
 base=page.new_tag('base',href='../');page.head.insert(0,base)
 summary=BeautifulSoup(f'<section aria-label="Object anomaly AUROC"><h2>Detection AUROC · {stats["binary_auroc"]:.3f}</h2><p>AnomalyDINO · sum of ten unrounded pair scores per video. Revised object-deformation labels: v2 workbook column AF. 11 deformed versus 23 non-deformed videos; one moderate label excluded. Ten videos have no score.</p><p><a target="_blank" href="analysis/OBJECT_ANOMALY_REPORT.md">Analysis and coverage</a> · <a target="_blank" href="analysis/OBJECT_ANOMALY_STATISTICS.json">Exact statistics</a></p></section>','html.parser')
 page.select_one('header').append(summary.section)
 folder=out/'objects';folder.mkdir(exist_ok=True);(folder/'image-pack-manifest.json').write_text(json.dumps({'image_packs':urls,'format':'gzip-compressed JSON mapping original relative image paths to exact base64 bytes','source_gallery':str(gallery.relative_to(ROOT))},indent=2));(folder/'selection_gallery.html').write_text(str(page))
 dest=out/'analysis';dest.mkdir(exist_ok=True)
 shutil.copy2(analysis/'REPORT.md',dest/'OBJECT_ANOMALY_REPORT.md')
 shutil.copy2(analysis/'statistics.json',dest/'OBJECT_ANOMALY_STATISTICS.json')
 table=parse(catches).find('table');remap(table,catches.parent)
 result='<section id="object-reference" class="review-section"><h2>Object reference crops and anomaly scores</h2><iframe class="object-gallery" title="Original object crop gallery with AUROC" src="objects/selection_gallery.html" loading="lazy" style="width:100%;height:1000px;border:0" onload="const f=this;const resize=()=>{f.style.height=Math.ceil(f.contentDocument.body.getBoundingClientRect().height)+\'px\'};resize();new ResizeObserver(resize).observe(f.contentDocument.body)"></iframe></section>'
 result+='<section id="object-occlusion" class="review-section"><h2>Object occlusion catches</h2><p>Inspect the detector’s catches alongside the original video, magnified object view, mask evidence, and rigidity history. Frame numbers are zero based.</p><div class="table-scroll">'+str(table)+'</div><p>Use Previous catch and Next catch to jump between flagged intervals. These are occlusion-risk annotations; original rigidity scores are unchanged.</p></section>'
 return result,[gallery,catches]
