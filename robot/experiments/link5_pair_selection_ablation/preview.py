"""Compare both methods on the same saved current-mask query initialization."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .selectors import select_pairs,METHODS
COLORS={'longitudinal':'#df9800','transverse':'#00cddd','diagonal':'#ce45e4','transition_upper':'#ff4037','transition_lower':'#ff4037','transition_width':'#ce45e4','global_long':'#a6f329','other_width':'#00cddd','other_diagonal':'#00cddd'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--initialization',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();manifest=json.loads(a.manifest.read_text());summary=[]
    for entry in manifest['entries']:
        case=entry['video_id'];src=a.initialization/case;dst=a.output/case;dst.mkdir(parents=True,exist_ok=True)
        data=json.loads((src/'pairs.json').read_text());assert data['mask_sha256']==entry['mask_sha256'];xy=np.array(data['query_points_xy']);mask=np.array(Image.open(src/'mask.png'))>0
        for file in ['frame0.png','mask.png','mask_points.png']:shutil.copy2(src/file,dst/file)
        yy,xx=np.where(mask);box=(max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(mask.shape[1],int(xx.max())+26),min(mask.shape[0],int(yy.max())+26))
        records={}
        for method in METHODS:
            pairs,stats=select_pairs(method,xy,None,mask);again,_=select_pairs(method,xy,None,mask);assert pairs==again
            record=dict(case=case,method=method,scope='2D frame-zero preview; no visibility/depth eligibility, no rigidity score',point_ids=data['point_ids'],query_points_xy=xy.tolist(),mask_sha256=entry['mask_sha256'],pairs=pairs,stats=stats)
            (dst/(method+'.json')).write_text(json.dumps(record,indent=2));records[method]=stats
            im=Image.open(src/'frame0.png').convert('RGB');d=ImageDraw.Draw(im)
            for x,y in xy:d.ellipse((x-2,y-2,x+2,y+2),fill='white')
            for edge in pairs:d.line([tuple(xy[edge['i']]),tuple(xy[edge['j']])],fill=COLORS[edge['kind']],width=3)
            im.crop(box).save(dst/(method+'.png'))
        summary.append(dict(case=case,cohort=entry['cohort'],methods=records))
    (a.output/'preview_summary.json').write_text(json.dumps(summary,indent=2))
    options=''.join(f'<option>{r["case"]}</option>' for r in summary)
    body='''<!doctype html><meta charset="utf-8"><title>Link5 pair selection ablations</title><style>body{max-width:1200px;margin:30px auto;padding:20px;font:17px/1.6 system-ui;background:#f8f8f5;color:#25363a}img{width:100%}select{font:inherit}a{color:#176c7a}</style><h1>Link5: two pair-selection methods</h1><p>41 selected videos + COSMOS2.5_0018 (additional). Excluded: LVP_ROBOWM_0010, LVP_ROBOWM_0030, COSMOS3_0021, COSMOS3_0035. Original artifacts preserved.</p><p>Same current frame-zero masks and query points for both methods. These are geometry previews; actual depth/visibility gating can change the final graph. No new rigidity scores have been computed.</p><select id="case" onchange="show(this.value)">OPTIONS</select><h2>Mask and initialized points</h2><img id="points"><h2>Balanced v0 — earlier proposal</h2><p>Target: 10 longitudinal + 10 cross-width + 10 local diagonals; five longitudinal sections; maximum endpoint degree 3.</p><p id="balanced_stats"></p><img id="balanced"><h2>Pair selection refine v1 — transition-focused</h2><p>Target: 6 upper + 6 lower transition bridges, 4 transition-width, 6 long, 4 other width and 4 other diagonal pairs. Maximum endpoint degree 3.</p><p id="refine_stats"></p><img id="refine"><p id="links"></p><script>const data=DATA;function show(c){document.getElementById('case').value=c;let r=data.find(x=>x.case===c);document.getElementById('points').src=c+'/mask_points.png';for(let [id,m] of [['balanced','balanced_v0'],['refine','refine_v1']]){document.getElementById(id).src=c+'/'+m+'.png';document.getElementById(id+'_stats').textContent=r.methods[m].pair_count+'/30 pairs; '+r.methods[m].distinct_endpoints+' distinct endpoints; deficits: '+JSON.stringify(r.methods[m].deficits)}document.getElementById('links').innerHTML=`<a href="${c}/mask.png">Exact mask</a> · <a href="${c}/balanced_v0.json">Balanced pairs + point IDs</a> · <a href="${c}/refine_v1.json">Refine v1 pairs + point IDs</a>`}show('COSMOS2.5_0010')</script>'''
    body=body.replace('41 selected videos + COSMOS2.5_0018 (additional). Excluded: LVP_ROBOWM_0010, LVP_ROBOWM_0030, COSMOS3_0021, COSMOS3_0035.',
        f"{manifest['primary_count']} selected videos + {manifest['additional_count']} additional. Excluded: {', '.join(sorted(manifest['excluded']))}.")
    body=body.replace("show('COSMOS2.5_0010')",'show('+json.dumps(summary[0]['case'])+')')
    (a.output/'index.html').write_text(body.replace('OPTIONS',options).replace('const data=DATA','const data='+json.dumps(summary)))
    print(json.dumps({m:sum(r['methods'][m]['pair_count']==30 for r in summary) for m in METHODS}));print(a.output/'index.html')
if __name__=='__main__':main()
