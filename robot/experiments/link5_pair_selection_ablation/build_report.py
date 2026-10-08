"""Create an independent rigidity comparison and synchronized evidence viewer."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .prepare_gpu import sha, write
from .selectors import METHODS


def build(root, sources):
    manifest=json.loads((root/'manifest.json').read_text())
    scores=json.loads((root/'rigidity/summary.json').read_text())
    by_key={(r['case'],r['method']):r for r in scores}
    videos=[];paired=[];diagnostics=[];graph_comparisons=[]
    viewer=root/'report';viewer.mkdir(exist_ok=True)
    for entry in manifest['entries']:
        case=entry['video_id'];record=dict(case=case,cohort=entry['cohort'],methods={})
        pairrow=dict(video_id=case,cohort=entry['cohort'])
        for method in METHODS:
            result=by_key.get((case,method),dict(status='failed',error='Missing score row'))
            pairrow[method+'_status']=result['status'];pairrow[method+'_score']=result.get('rigidity_score')
            pairrow[method+'_pairs']=result.get('pairs');pairrow[method+'_carried_frames']=result.get('carried_frames')
            if result['status']!='complete':
                record['methods'][method]=result;continue
            folder=root/'rigidity'/case/method
            evidence=json.loads((folder/'evidence.json').read_text())
            with np.load(folder/'trajectories.npz',allow_pickle=False) as archive:
                xy=archive['tracks_2d'];vis=archive['raw_visibility'];ids=archive['point_ids'];indices=archive['pair_track_indices']
            raw=root/'shared_inputs'/(case+'.json')
            receipt=json.loads(raw.read_text())
            h,w=receipt['tracker_pointmap_hw']
            record.update(fps=receipt['fps'],frame_count=len(xy),grid_hw=[h,w])
            record['methods'][method]=dict(result=result,history=evidence['rigidity_history'],frames=evidence['frame_diagnostics'],
                coverage=evidence['selection_stats'],pairs=evidence['selected_pairs'],geometric_selection=evidence['geometric_selection'],
                query_ids=ids.tolist(),xy=np.round(xy.astype(float),3).tolist(),visible=(vis>.5).tolist(),pair_indices=indices.tolist())
            if case in ('COSMOS3_0010','COSMOS3_0015','LVP_ROBOWM_0015','COSMOS2.5_0015'):
                center={'COSMOS3_0010':101,'COSMOS3_0015':73}.get(case,int(np.argmax(evidence['rigidity_history'])))
                for frame in range(max(0,center-2),min(len(xy),center+3)):
                    diag=evidence['frame_diagnostics'][frame]
                    diagnostics.append(dict(video_id=case,method=method,**diag,center_frame_zero_based=center))
        if all(pairrow[m+'_status']=='complete' for m in METHODS):
            pairrow['refine_minus_balanced']=pairrow['refine_v1_score']-pairrow['balanced_v0_score']
            a,b=[record['methods'][m] for m in METHODS]
            sets=[{tuple(sorted((p['point_id_i'],p['point_id_j']))) for p in d['pairs']} for d in (a,b)]
            endpoints=[{i for pair in pairs for i in pair} for pairs in sets]
            pairrow['shared_pairs']=len(sets[0]&sets[1]);pairrow['shared_endpoints']=len(endpoints[0]&endpoints[1])
            pairrow['availability_difference_frames']=sum(x['available_pair_count']!=y['available_pair_count'] for x,y in zip(a['frames'][1:],b['frames'][1:]))
            pairrow['carry_difference_frames']=sum(x['carried']!=y['carried'] for x,y in zip(a['frames'][1:],b['frames'][1:]))
            comparison=dict(video_id=case,cohort=entry['cohort'],score_difference=pairrow['refine_minus_balanced'],
                shared_pairs=pairrow['shared_pairs'],shared_endpoints=pairrow['shared_endpoints'],
                availability_difference_frames=pairrow['availability_difference_frames'],carry_difference_frames=pairrow['carry_difference_frames'],
                methods={m:dict(selected_pair_count=len(record['methods'][m]['pairs']),
                    mean_available_pairs=float(np.mean([f['available_pair_count'] for f in record['methods'][m]['frames'][1:]])),
                    carried_frames=record['methods'][m]['result']['carried_frames'],
                    quota_deficits=record['methods'][m]['coverage'].get('deficits',{}),
                    geometry_flags=record['methods'][m]['coverage'].get('geometry_flags',[])) for m in METHODS},
                attribution='Identical raw inputs and gate; differences arise from graph selection and its visibility/carry consequences. Depth or tracking error versus real deformation requires visual assessment.')
            graph_comparisons.append(comparison)
        paired.append(pairrow)
        if 'fps' in record:
            generator,number=case.rsplit('_',1)
            source=sources/('LVP' if generator=='LVP_ROBOWM' else generator)/(number+'.mp4')
            if sha(source)!=entry['source_video_sha256']:raise ValueError('Viewer source mismatch: '+case)
            localvideo=viewer/'videos'/(case+'.mp4');localvideo.parent.mkdir(exist_ok=True)
            if not localvideo.exists():shutil.copy2(source,localvideo)
            elif sha(localvideo)!=entry['source_video_sha256']:raise ValueError('Existing viewer source mismatch')
            record['video']='videos/'+case+'.mp4'
            cap=cv2.VideoCapture(str(source));ok,bgr=cap.read();cap.release()
            if not ok:raise ValueError('Cannot decode frame0 for overlay: '+case)
            h,w=record['grid_hw'];frame=cv2.resize(bgr,(w,h))
            for method in METHODS:
                data=record['methods'][method]
                if 'xy' not in data:continue
                overlay=frame.copy();coords=np.rint(data['xy'][0]).astype(int)
                for i,j in data['pair_indices']:
                    cv2.line(overlay,tuple(coords[i]),tuple(coords[j]),(55,210,255) if method=='balanced_v0' else (210,100,225),2,cv2.LINE_AA)
                for i in np.unique(data['pair_indices']):
                    cv2.circle(overlay,tuple(coords[i]),3,(255,255,255),-1)
                    cv2.putText(overlay,str(data['query_ids'][i]),tuple(coords[i]+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1,cv2.LINE_AA)
                destination=root/'rigidity'/case/method/'frame0_actual_graph.png'
                if not cv2.imwrite(str(destination),overlay):raise RuntimeError('Overlay write failed')
        videos.append(record)
    keys=list(dict.fromkeys(k for row in paired for k in row))
    with (root/'rigidity/paired_scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(paired)
    write(root/'rigidity/paired_scores.json',paired)
    write(root/'rigidity/neighboring_frame_diagnostics.json',diagnostics)
    write(root/'rigidity/graph_comparison.json',graph_comparisons)
    summary=dict(primary_videos=manifest['primary_count'],additional_videos=manifest['additional_count'],
        successful_method_video_scores=sum(r['status']=='complete' for r in scores),expected_method_video_scores=2*len(manifest['entries']),
        excluded=manifest['excluded'],published=False,metric='Mean per-frame MAD(distance ratio)/(median(distance ratio)+1e-6), excluding frame0',
        interpretation='Paired graph ablation only; no detection improvement inferred without labeled full-video validation.',
        cohorts={cohort:{method:dict(successful=sum(r['status']=='complete' for r in scores if r['cohort']==cohort and r['method']==method),
            failed=sum(r['status']!='complete' for r in scores if r['cohort']==cohort and r['method']==method),
            mean_score=float(np.mean([r['rigidity_score'] for r in scores if r['cohort']==cohort and r['method']==method and r['status']=='complete']))
            if any(r['status']=='complete' for r in scores if r['cohort']==cohort and r['method']==method) else None)
            for method in METHODS} for cohort in ('selected45','additional')})
    write(root/'rigidity/comparison_summary.json',summary)
    (viewer/'data.js').write_text('window.RIGIDITY_DATA='+json.dumps(dict(summary=summary,videos=videos),separators=(',',':'),allow_nan=False)+';')
    (viewer/'index.html').write_text(HTML)
    (root/'REPORT.md').write_text('\n'.join([
        '# Link5 pair-selection rigidity comparison','',
        f"{summary['successful_method_video_scores']}/{summary['expected_method_video_scores']} method/video scores complete; {manifest['primary_count']} primary videos and {manifest['additional_count']} additional video.",'',
        'Metric: '+summary['metric']+'. Both methods use identical cached masks, saved frame-zero query IDs, raw CoTracker visibility/tracks and full-sequence CVD world pointmaps. Baseline v1 anchor gate, tiny-baseline rejection, fewer-than-three-pair carry and temporal mean are unchanged. No refined shape-codebook depth gate is applied.','',
        'The five explicit handoff exclusions include COSMOS2.5_0010, so the older 41-primary count is superseded by 40. Its old artifacts remain preserved; no new diagnostic score is invented for the excluded video.','',
        'Actual graphs may differ from CPU source-resolution previews because native-grid resizing and the common frame-zero gradient/visibility gate affect eligibility. Quota deficits are retained.','',
        'Interpretation: graph choice changes which spatial deformations and reconstruction/tracking errors enter MAD. More pairs alone do not establish detection improvement. A coherent scale change produces nearly equal distance ratios and can be suppressed by MAD; minority pair changes can also be suppressed. Occlusion can carry a previous score. The unchanged temporal mean can dilute brief events.','',
        'Inspect neighboring_frame_diagnostics.json and the synchronized viewer around zero-based101/display102 for COSMOS3_0010 and zero-based73/display74 for COSMOS3_0015. For the other two guide videos, automatically centered neighborhoods are score peaks, not manually labeled deformation onset.','',
        'Per-method evidence includes coverage/geometry flags, actual point IDs/baselines, frame diagnostics, distance-ratio trajectories and selected graph PNGs. Finite 3D samples are numerical availability, separate from raw tracker visibility; finite depth can still be geometrically wrong.','',
        'Sources, model/checkpoint hashes, preprocessing, query coordinate transforms, seeds and environment are in shared_inputs/*.json. Original high-precision arrays and trajectories are preserved; viewer coordinate rounding is display-only.','',
        'Viewer: report/index.html. Tables: rigidity/paired_scores.csv and rigidity/rigidity_scores.csv. This report is local and has not been published.','']))
    print('RIGIDITY_REPORT_COMPLETE',summary['successful_method_video_scores'],summary['expected_method_video_scores'],flush=True)


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Link5 rigidity ablation</title><style>
:root{color-scheme:light}body{margin:0;background:#f5f5f0;color:#223438;font:15px/1.5 system-ui,sans-serif}main{max-width:1320px;margin:32px auto;padding:0 24px}h1{font-size:32px;letter-spacing:-.7px;margin:0}h2{font-size:19px;margin:22px 0 10px}.muted{color:#587074}select,button,input{font:inherit}select,button{padding:8px;border:1px solid #acb9b7;background:white;border-radius:3px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;display:block;background:#101a1b}canvas{width:100%;display:block;background:white}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #d1dcd8}th{font-size:12px;position:sticky;top:0;background:#e8edE7}.scroll{overflow:auto;max-height:480px}a{color:#176776}.metrics{white-space:pre-wrap;min-height:68px}.controls{display:flex;align-items:center;gap:12px;margin:12px 0;flex-wrap:wrap}input{flex:1;min-width:170px}.notice{border-left:3px solid #a58136;padding:10px 16px;background:#efeadb}@media(max-width:760px){.compare{grid-template-columns:1fr}main{padding:0 14px}h1{font-size:26px}}
</style><main><h1>Link5 · pair-selection rigidity</h1><p id="summary" class="muted"></p><p class="notice">Raw rigidity scores. The pair graph is the ablation; tracking and 3D geometry are shared. Detection improvement has not been established.</p><p><a href="../rigidity/paired_scores.csv">Paired CSV</a> · <a href="../rigidity/summary.json">Scores JSON</a> · <a href="../REPORT.md">Experiment report</a></p><h2>Full-video evidence</h2><select id="case" aria-label="Video"></select><div class="controls"><button id="play">Play / pause</button><button id="prev">Previous frame</button><button id="next">Next frame</button><input id="frame" type="range" min="0" value="0" aria-label="Zero-based frame"><span id="frameLabel"></span></div><video id="video" muted playsinline preload="metadata"></video><div class="compare"><section><h2>Balanced v0</h2><canvas id="balanced_v0"></canvas><p class="metrics" id="balanced_v0_metrics"></p></section><section><h2>Pair selection refine v1</h2><canvas id="refine_v1"></canvas><p class="metrics" id="refine_v1_metrics"></p></section></div><canvas id="chart" height="190"></canvas><p id="coverage" class="muted"></p><p id="evidenceLinks"></p><h2>Paired comparison</h2><p>40 primary selected-45 videos and one additional COSMOS2.5_0018. Positive Δ means refine v1 reports a higher raw rigidity score; it does not establish better detection.</p><div class="scroll"><table><thead><tr><th>Video</th><th>Cohort</th><th>Balanced</th><th>Refine v1</th><th>Δ</th><th>Pairs B/R</th><th>Carry B/R</th></tr></thead><tbody id="table"></tbody></table></div><h2>Interpretation and exclusions</h2><p>The viewer shows the actual native-grid graphs after the common frame-zero gate. Inspect tracking drift, implausible world distances, changing visibility, carry flags and graph coverage together. Finite pointmaps do not certify correct depth. The unchanged MAD can suppress coherent scaling or minority changes; temporal averaging can dilute a brief event.</p><p id="excluded"></p><p class="muted">Display frame = zero-based source frame + 1. Guide neighborhoods: COSMOS3_0010 source101/display102; COSMOS3_0015 source73/display74. COSMOS2.5_0010 is excluded by the current handoff. Not published.</p></main><script src="data.js"></script><script>
const D=window.RIGIDITY_DATA,$=id=>document.getElementById(id),methods=['balanced_v0','refine_v1'];let current,index=0,raf;
$('summary').textContent=`${D.summary.successful_method_video_scores}/${D.summary.expected_method_video_scores} scores complete · ${D.summary.primary_videos} primary + ${D.summary.additional_videos} additional · mean MAD/median, frame0 excluded`;
const fmt=x=>Number.isFinite(x)?x.toFixed(6):'failed';
for(const v of D.videos){let option=document.createElement('option');option.value=v.case;option.textContent=v.case+(v.cohort==='additional'?' · additional':'');$('case').append(option);let row=document.createElement('tr'),a=v.methods.balanced_v0.result||v.methods.balanced_v0,b=v.methods.refine_v1.result||v.methods.refine_v1;for(const value of [v.case,v.cohort,fmt(a.rigidity_score),fmt(b.rigidity_score),fmt(b.rigidity_score-a.rigidity_score),`${a.pairs??'–'} / ${b.pairs??'–'}`,`${a.carried_frames??'–'} / ${b.carried_frames??'–'}`]){let cell=document.createElement('td');cell.textContent=value;row.append(cell)}row.style.cursor='pointer';row.onclick=()=>select(v.case);$('table').append(row)}
$('excluded').textContent='Excluded: '+Object.keys(D.summary.excluded).join(', ')+'. Existing artifacts preserved.';
function select(name){current=D.videos.find(v=>v.case===name);$('case').value=name;index=0;$('video').pause();$('video').src=current.video||'';$('frame').max=Math.max(0,(current.frame_count||1)-1);$('frame').value=0;$('evidenceLinks').replaceChildren();for(const method of methods){for(const [file,label] of [['evidence.json','point IDs and evidence'],['frame0_actual_graph.png','frame0 graph'],['trajectories.npz','raw trajectories']]){const a=document.createElement('a');a.href='../rigidity/'+name+'/'+method+'/'+file;a.textContent=method+' '+label;$('evidenceLinks').append(a,document.createTextNode(' · '))}}draw();}
function draw(){if(!current)return;$('frame').value=index;$('frameLabel').textContent=`source ${index} · display ${index+1}`;let coverage=[];for(const method of methods){const data=current.methods[method],canvas=$(method),ctx=canvas.getContext('2d');let [h,w]=current.grid_hw||[200,400];canvas.width=w;canvas.height=h;if($('video').readyState>=2)ctx.drawImage($('video'),0,0,w,h);if(!data.xy){$(method+'_metrics').textContent=data.error||'Failed';continue}const xy=data.xy[index]||data.xy[0],vis=data.visible[index]||data.visible[0];ctx.lineWidth=Math.max(1,w/500);for(const [i,j] of data.pair_indices){ctx.strokeStyle=vis[i]&&vis[j]?(method==='balanced_v0'?'#ffce56':'#ea92ff'):'#777a';ctx.beginPath();ctx.moveTo(...xy[i]);ctx.lineTo(...xy[j]);ctx.stroke()}for(const i of new Set(data.pair_indices.flat())){ctx.fillStyle=vis[i]?'#fff':'#999';ctx.beginPath();ctx.arc(...xy[i],2.2,0,Math.PI*2);ctx.fill()}const f=data.frames[index];$(method+'_metrics').textContent=`video ${fmt(data.result.rigidity_score)} · frame ${fmt(f.score)}\navailable pairs ${f.available_pair_count}/${f.selected_pair_count} · visible points ${f.raw_visible_point_count} · finite 3D ${f.finite_sampled_3d_point_count}${f.carried?' · CARRIED':''}`;coverage.push(`${method}: deficits ${JSON.stringify(data.coverage.deficits||{})}; geometry flags ${(data.coverage.geometry_flags||[]).join(', ')||'none'}`)}$('coverage').textContent=coverage.join(' | ');chart();}
function chart(){const c=$('chart'),ctx=c.getContext('2d');c.width=Math.max(600,c.clientWidth*devicePixelRatio);c.height=190*devicePixelRatio;const pad=30*devicePixelRatio,w=c.width,h=c.height;ctx.clearRect(0,0,w,h);let peak=Math.max(.001,...methods.flatMap(m=>current.methods[m].history||[]));ctx.font=`${12*devicePixelRatio}px system-ui`;ctx.fillStyle='#587074';ctx.fillText(`frame score · max ${peak.toFixed(4)}`,pad,20*devicePixelRatio);for(const m of methods){const ys=current.methods[m].history;if(!ys)continue;ctx.strokeStyle=m==='balanced_v0'?'#bb8b18':'#9b4bb0';ctx.lineWidth=2*devicePixelRatio;ctx.beginPath();ys.forEach((y,i)=>{const x=pad+i/Math.max(1,ys.length-1)*(w-2*pad),v=h-pad-y/peak*(h-2*pad);i?ctx.lineTo(x,v):ctx.moveTo(x,v)});ctx.stroke()}ctx.strokeStyle='#294e57';ctx.beginPath();const x=pad+index/Math.max(1,(current.frame_count||1)-1)*(w-2*pad);ctx.moveTo(x,pad);ctx.lineTo(x,h-pad);ctx.stroke();}
function seek(frame){if(!current.fps)return;index=Math.max(0,Math.min(current.frame_count-1,frame));$('video').pause();$('video').currentTime=(index+.1)/current.fps;draw()}
$('case').onchange=e=>select(e.target.value);$('frame').oninput=e=>seek(+e.target.value);$('prev').onclick=()=>seek(index-1);$('next').onclick=()=>seek(index+1);$('play').onclick=()=>{$('video').paused?$('video').play():$('video').pause()};$('video').onseeked=draw;$('video').onloadeddata=draw;function tick(){if(current?.fps&&!$('video').paused){index=Math.min(current.frame_count-1,Math.floor($('video').currentTime*current.fps));draw()}raf=requestAnimationFrame(tick)}tick();select(D.videos.find(v=>v.case==='COSMOS3_0010'&&v.frame_count)?.case||D.videos[0].case);
</script></html>'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--sources',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.sources)


if __name__=='__main__':main()
