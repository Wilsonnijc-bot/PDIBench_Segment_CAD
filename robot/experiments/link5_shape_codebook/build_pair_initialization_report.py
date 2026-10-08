"""Render the CPU frame-zero initialization experiment, without inference."""
from pathlib import Path
import json,hashlib,csv,html,io
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'results/link5_shape_codebook/round_four_frame0/pair_initialization_frame0'
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
COLORS={'transition_upper':'#ed332c','transition_lower':'#ed332c','transition_width':'#bb379b',
        'global_long':'#98d51c','other_width':'#189ec0','other_diagonal':'#189ec0'}
def font(n):return ImageFont.truetype(FONT,n)

def render(r):
    folder=OUT/r['case'];raw=Image.open(folder/'frame0.png').convert('RGB');mask=np.array(Image.open(folder/'mask.png'))>0
    pts=np.array(r['query_points_xy']);g=r['geometry'];origin=np.array(g['origin']);basis=np.array(g['basis']);L=g['length'];umin=g['umin'];center=g['transition'];half=r['config']['transition_half_width']
    yy,xx=np.where(mask);lo=np.array([xx.min()-25,yy.min()-25]);hi=np.array([xx.max()+25,yy.max()+25]);mid=(lo+hi)/2
    cropw=max(hi[0]-lo[0],(hi[1]-lo[1])*3.6);croph=cropw/3.6
    box=(int(mid[0]-cropw/2),int(mid[1]-croph/2),int(mid[0]+cropw/2),int(mid[1]+croph/2))
    scale=np.array([1440/(box[2]-box[0]),400/(box[3]-box[1])]);point=(pts-box[:2])*scale
    clean=np.array(raw);overlay=clean.copy();overlay[mask]=(clean[mask]*.60+np.array([20,210,165])*.40).astype(np.uint8)
    mask_crop=Image.fromarray(overlay).crop(box).resize((1440,400));pair_crop=raw.crop(box).resize((1440,400));dr=ImageDraw.Draw(pair_crop,'RGBA')
    points_crop=raw.crop(box).resize((1440,400));pd=ImageDraw.Draw(points_crop)
    for x,y in point:
        pd.ellipse((x-5,y-5,x+5,y+5),fill='#00ffff',outline='#11282e',width=2)
    points_canvas=Image.new('RGB',(1520,1030),'#f8f8f5');label=ImageDraw.Draw(points_canvas)
    label.text((40,18),r['case']+' | frame 0',font=font(30),fill='#203238')
    label.text((40,65),'Saved mask, unchanged (green overlay)',font=font(24),fill='#203238')
    points_canvas.paste(mask_crop,(40,105))
    label.text((40,530),f'{r["query_count"]} automatically initialized points (cyan)',font=font(24),fill='#203238')
    points_canvas.paste(points_crop,(40,570))
    label.text((40,988),'Frame-zero query positions; these are not tracked trajectories or visibility measurements.',font=font(20),fill='#203238')
    points_canvas.save(folder/'mask_points.png')
    corners=[]
    for su,sv in [(center-half,-250),(center+half,-250),(center+half,250),(center-half,250)]:
        q=origin+np.array([umin+su*L,sv])@basis.T;corners.append(tuple((q-box[:2])*scale))
    dr.polygon(corners,fill=(255,170,30,27))
    for q in point:
        x,y=q;dr.ellipse((x-2,y-2,x+2,y+2),fill=(210,210,210,255))
    for p in sorted(r['pairs'],key=lambda p:p['kind'].startswith('transition')):
        dr.line([tuple(point[p['i']]),tuple(point[p['j']])],fill=COLORS[p['kind']],width=3 if not p['kind'].startswith('transition') else 4)
    for i in sorted({i for p in r['pairs'] for i in [p['i'],p['j']]}):
        x,y=point[i];dr.ellipse((x-4,y-4,x+4,y+4),fill='white',outline='#263138',width=1)
    canvas=Image.new('RGB',(1520,1240),'#f8f8f5');draw=ImageDraw.Draw(canvas)
    draw.text((40,18),r['case']+' | current mask cache | frame 0 only',font=font(27),fill='#203238')
    draw.text((40,60),'Saved mask, unchanged (green overlay). Binary mask available separately.',font=font(21),fill='#203238');canvas.paste(mask_crop,(40,95))
    draw.text((40,510),f'Automatic pair initialization: {r["stats"]["pair_count"]} pairs; {r["stats"]["transition_pairs"]} around the detected transition',font=font(24),fill='#203238');canvas.paste(pair_crop,(40,548))
    draw.text((40,961),'Red: upper/lower transition bridges | Magenta: transition width | Green: long | Blue: elsewhere',font=font(20),fill='#203238')
    fig,ax=plt.subplots(figsize=(10,1.55),layout='constrained');grid=np.array(g['grid']);ax.plot(grid,g['smooth_width'],color='#237b83',lw=2);ax.axvline(g['neck'],color='#5e6f74',ls=':',label='narrow section');ax.axvline(center,color='#e84331',label='chosen transition');ax.axvspan(center-half,center+half,color='#e9b956',alpha=.25);ax.set(xlabel='Normalized mask long axis',ylabel='Width (px)',xlim=(0,1));ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,loc='upper right');buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);buf.seek(0);chart=Image.open(buf).convert('RGB').resize((1000,180));canvas.paste(chart,(30,1004))
    lines=[f'Queries: {r["query_count"]}  | endpoints: {r["stats"]["distinct_endpoints"]}',f'Maximum endpoint degree: {r["stats"]["max_degree"]}',f'Transition at {center:.2f} of mask length',f'Status: {r["status"]}']
    for j,line in enumerate(lines):draw.text((1050,1022+j*30),line,font=font(18),fill='#203238')
    flag=', '.join(g['flags']) or 'none';draw.text((40,1204),'Flags: '+flag,font=font(18),fill='#9b4c14')
    canvas.save(folder/'preview.png')
    with (folder/'pairs.csv').open('w') as f:
        keys=['pair','kind','query_id_i','query_id_j','x_i','y_i','x_j','y_j','distance_source_px'];wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
        for k,p in enumerate(r['pairs'],1):
            a,b=pts[[p['i'],p['j']]];wr.writerow(dict(pair=k,kind=p['kind'],query_id_i=r['point_ids'][p['i']],query_id_j=r['point_ids'][p['j']],x_i=a[0],y_i=a[1],x_j=b[0],y_j=b[1],distance_source_px=p['distance_px']))

def main():
    results=json.loads((OUT/'metadata/results.json').read_text());execution=json.loads((OUT/'metadata/execution.json').read_text())
    # This historical initialization report is independent of deleted ablations.
    # An explicit exclusions file may accompany its own saved input metadata.
    excluded_path=OUT/'metadata/excluded.json'
    excluded=json.loads(excluded_path.read_text()) if excluded_path.is_file() else []
    results=[r for r in results if r['case'] not in excluded]
    for r in results:render(r)
    summaries={r['case']:{'case':r['case'],'status':r['status'],'queries':r['query_count'],**r['stats'],'flags':r['geometry']['flags'],'transition':r['geometry']['transition'],'side':r['geometry']['broad_side']} for r in results}
    options=''.join(f'<option>{html.escape(c)}</option>' for c in summaries)
    rows=''.join(f'<tr><td><button onclick="selectCase(\'{c}\')">{c}</button></td><td>{r["queries"]}</td><td>{r["pair_count"]}</td><td>{r["transition_pairs"]}</td><td>{r["distinct_endpoints"]}</td><td>{r["max_degree"]}</td><td>{r["transition"]:.2f}</td><td>{html.escape(", ".join(r["flags"]))}</td></tr>' for c,r in summaries.items())
    body=f'''<h1>Automatic pairs around the forearm transition</h1>
<p>Current Link5 masks from <code>round_four_frame0</code> · frame zero only · tested on ERIS allocation 5660284, erishpc-gpu-001, one H200 allocated.</p>
<p><strong>No per-frame manual point placement.</strong> The red pairs reproduce the intent of the user's drawing by crossing the automatically located narrow-to-broad transition along both flanks. The same rule and parameters were used on all 46 masks. The mask silhouette is a geometric prior, not proof of a physical hinge.</p>
<p><strong>Result:</strong> {execution['complete']}/46 cases supplied all 30 pairs. Two supplied 27 because the lower-flank quota could not be met under the current constraints. This is an initialization feasibility test, not deformation-detection validation. Eight CPU workers took {execution['seconds']:.2f} seconds inside the existing H200 allocation; GPU inference and GPU memory use were zero.</p>
<h2>Inspect masks and initialized points</h2><label for="case">Video </label><select id="case" onchange="selectCase(this.value)">{options}</select> <label for="mode">View </label><select id="mode" onchange="selectCase(document.getElementById('case').value)"><option value="mask_points.png">Mask + initialized points</option><option value="preview.png">Mask + proposed pairs</option></select><p id="status"></p><img id="preview" alt="Saved frame-zero mask and automatic initialization"><p id="links"></p>
<h2>Reproducible rule</h2><ol>
<li>Read only <code>object_masks[0,0]</code> from the current manifest-selected mask file, including its selected attempt. Verify mask and source-video SHA-256. Display/save the raw mask unchanged.</li>
<li>Use the largest connected component for geometry and erode its query support by two source-image pixels. Use the existing SIFT + Shi–Tomasi + grid candidate sampler and farthest-point balancing, with a nominal 100 queries at maximum image dimension 880. Reject resize-rounded points outside support. These are newly initialized query IDs, not IDs from old CoTracker replays.</li>
<li>Compute the mask's PCA long axis; measure 2.5–97.5% transverse width in 101 sections and smooth with a fixed seven-bin kernel. Find the narrow central segment (25–70% of length). Choose the side with greater sustained excess width beyond that neck. Locate its strongest positive width increase, using a 12%-length window and excluding terminal 18% regions.</li>
<li>Define a transition band ±14% of mask length. Reserve <strong>6 upper-flank and 6 lower-flank bridges</strong>, each with endpoints on opposite sides of the transition, plus <strong>4 cross-width pairs</strong> in the band.</li>
<li>Keep <strong>6 long pairs</strong> spanning at least half the link, <strong>4 cross-width pairs elsewhere</strong> and <strong>4 local diagonals elsewhere</strong>. The total target is 30. Prefer distinct measurement positions, moderate baselines and less-used endpoints; cap each point at three pairs.</li>
<li>Reject pairs shorter than max(8 source pixels, 18% of local mask width). Do not silently relax quotas or invent points when support is insufficient. Exact thresholds, IDs and endpoints are in each case's JSON/CSV and the frozen execution script.</li></ol>
<h2>Failures and limits that matter</h2>
<p><strong>COSMOS2.5_0044 and COSMOS3_0056:</strong> 27 pairs, with three lower-flank transition bridges missing. More queries or an explicitly adaptive quota could address feasibility, but neither was silently introduced.</p>
<p><strong>LVP_ROBOWM_0035 and LVP_ROBOWM_0054:</strong> broad-side selection is ambiguous; for 0035, the heuristic chose the left shoulder at 0.19. <strong>LVP_ROBOWM_0065:</strong> it chose 0.18, at the search boundary, toward the left shoulder. Inspect these previews before treating the highlighted region as the intended proximal bend zone. They illustrate that a reproducible mask-only rule is not always anatomically correct.</p>
<p>The masks touch an image edge in these frame-zero views; width near clipping is incomplete. Small disconnected mask pieces are displayed in the raw mask but excluded from query support. No 3D depth, visibility, tracking persistence or rigidity score was computed. Reliable-depth gating and temporal validation remain necessary before adopting the pairs in a detector. Pair sets concentrated near a suspect region must also be evaluated with a suitable aggregation rule; a pooled median can suppress a minority of changed pairs.</p>
<h2>42 retained cases</h2><p>Excluded by user: LVP_ROBOWM_0010, LVP_ROBOWM_0030, COSMOS3_0021, COSMOS3_0035. Original run statistics above cover all 46; this viewer now shows 41 selected videos and one additional video only.</p><div class="scroll"><table><thead><tr><th>Video</th><th>Queries</th><th>Pairs</th><th>Transition pairs</th><th>Endpoints</th><th>Max degree</th><th>Transition</th><th>Flags</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><a href="metadata/results.json">All numerical results</a> · <a href="metadata/execution.json">Timing and execution scope</a> · <a href="metadata/run.log">ERIS log</a> · <a href="metadata/validation.json">Validation</a></p>'''
    style='body{font:17px/1.6 system-ui;color:#26343a;background:#f8f8f5;margin:0}main{max-width:1200px;margin:auto;padding:32px}h1{font-size:36px}h2{margin-top:38px}a{color:#166c7c}img{width:100%;height:auto}select,button{font:inherit;padding:6px;border:1px solid #a5b6b8;background:white}code{font-size:.85em;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ccd6d3;white-space:nowrap}.scroll{overflow:auto}li{margin:12px 0}'
    js='const DATA='+json.dumps(summaries)+';function selectCase(c){document.getElementById("case").value=c;const d=DATA[c];document.getElementById("preview").src=c+"/"+document.getElementById("mode").value;document.getElementById("status").textContent=`${d.pair_count}/30 pairs · ${d.transition_pairs} transition pairs · ${d.queries} automatic queries · max point degree ${d.max_degree}`;document.getElementById("links").innerHTML=`<a href="${c}/mask.png">Exact binary mask</a> · <a href="${c}/frame0.png">Unannotated source frame</a> · <a href="${c}/pairs.json">Query coordinates and pair logic</a> · <a href="${c}/pairs.csv">Pair endpoints CSV</a>`;}selectCase("COSMOS2.5_0010");'
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Current Link5 masks: automatic pair initialization</title><style>'+style+'</style><main>'+body+'</main><script>'+js+'</script></html>')
    print(OUT/'index.html')

if __name__=='__main__':main()
