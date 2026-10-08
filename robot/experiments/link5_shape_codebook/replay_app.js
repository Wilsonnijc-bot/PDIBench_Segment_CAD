/* Anomaly adapter for the project's published offline Plotly/video replay. */
'use strict';
const data=window.LINK5_REPLAY,$=id=>document.getElementById(id);
const video=$('video'),slider=$('frame'),play=$('play');
const defaultCamera={eye:{x:0,y:0,z:-2.25},up:{x:0,y:-1,z:0},projection:{type:'perspective'}};
let frame=0,camera=defaultCamera,orbit=true,showReference=true,activePoint=null,playing=false;
let renderRequest=0,renderBusy=false,renderDirty=false,renderToken=0,seekInFlight=false,seekPending=false,requestedFrame=0,timer=null;
const packets=new Map(),loads=new Map(),images=new Map();
const refinement=window.LINK5_REFINEMENT,depthPackets=new Map(),depthLoads=new Map();
const depthHref=path=>new URL(path,new URL('../../refinement/replay/',document.baseURI)).href;
function depthFrame(c,f){return refinement?.cases.find(r=>r.video_id===c.video_id)?.frames.find(r=>r.frame_id===f.frame_id)}
function refinedView(){return $('depth').value!=='original'}
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const fmt=x=>x==null?'unavailable':Number(x).toPrecision(6);
function currentCase(){return data.cases.find(c=>c.number===$('number').value&&c.generator===$('generator').value)}
function mode(){return $('mode').value}
function decode(s,Type){const raw=atob(s),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Type(bytes.buffer)}
window.LINK5_REFINEMENT_FRAME=p=>{p.camera=decode(p.camera,Float32Array);p.aligned=p.aligned?decode(p.aligned,Float32Array):null;for(const key of ['reason','old','rgb'])p[key]=decode(p[key],Uint8Array);p.pixels=decode(p.pixels,Uint16Array);p.observed_indices=decode(p.observed_indices,Uint32Array);p.scores={};p.refined=true;depthPackets.set(p.video_id+'/'+p.frame_id,p);while(depthPackets.size>24)depthPackets.delete(depthPackets.keys().next().value)};
function loadDepth(c,f){const row=depthFrame(c,f);if(!row?.packet)return Promise.resolve(null);const key=c.video_id+'/'+f.frame_id;if(depthPackets.has(key))return Promise.resolve(depthPackets.get(key));if(depthLoads.has(key))return depthLoads.get(key);const promise=new Promise(resolve=>{const script=document.createElement('script');script.src=depthHref(row.packet);script.onload=()=>{resolve(depthPackets.get(key)||null);depthLoads.delete(key);script.remove()};script.onerror=()=>{resolve(null);depthLoads.delete(key);script.remove()};document.head.append(script)});depthLoads.set(key,promise);return promise}
window.LINK5_FRAME=p=>{p.camera=decode(p.camera,Float32Array);p.aligned=p.aligned?decode(p.aligned,Float32Array):null;p.pixels=decode(p.pixels,Uint16Array);p.rgb=decode(p.rgb,Uint8Array);p.observed_indices=decode(p.observed_indices,Uint32Array);for(const k of Object.keys(p.scores))p.scores[k]=decode(p.scores[k],Float32Array);packets.set(p.video_id+'/'+p.frame_id,p);while(packets.size>24)packets.delete(packets.keys().next().value)};
function loadPacket(c,f){if(!f?.packet)return Promise.resolve(null);const key=c.video_id+'/'+f.frame_id;if(packets.has(key))return Promise.resolve(packets.get(key));if(loads.has(key))return loads.get(key);const promise=new Promise(resolve=>{const script=document.createElement('script');script.src=f.packet;script.onload=()=>{resolve(packets.get(key)||null);loads.delete(key);script.remove()};script.onerror=()=>{resolve(null);loads.delete(key);script.remove()};document.head.append(script)});loads.set(key,promise);return promise}
function loadImage(path){if(!path)return Promise.resolve(null);if(images.has(path))return images.get(path);const p=new Promise(resolve=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>resolve(null);i.src=path});images.set(path,p);while(images.size>24)images.delete(images.keys().next().value);return p}
function heat(v){const q=clamp(v/data.color_max[mode()],0,1),a=q<.5?[88,166,164]:[233,199,108],b=q<.5?[233,199,108]:[238,97,84],t=q<.5?q*2:(q-.5)*2;return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`}
function idsFor(p){if(!p)return[];const n=Number($('density').value),s=p.scores[mode()];const eligible=Array.from({length:p.count},(_,i)=>i).filter(i=>!p.refined||$('depth').value==='comparison'||p.reason[i]===0);if(eligible.length<=n)return eligible;const ids=new Set();for(let i=0;i<n;i++)ids.add(eligible[Math.floor(i*eligible.length/n)]);if(s)Array.from(s.keys()).sort((a,b)=>s[b]-s[a]).slice(0,80).forEach(i=>ids.add(i));if(activePoint!=null&&eligible.includes(activePoint))ids.add(activePoint);return Array.from(ids)}
// Same Plotly scatter3d/scattergl trace pattern as rigidity_replay.html.
function pointTrace(p,ids,reference=false,colorOverride=null,nameOverride=null){
  const xyz=$('space').value==='aligned'?p.aligned:p.camera,scores=p.scores[mode()];
  if(!xyz&&orbit)return null;
  const color=colorOverride||(reference?'#80928b':ids.map(i=>scores?heat(scores[i]):`rgb(${p.rgb[i*3]},${p.rgb[i*3+1]},${p.rgb[i*3+2]})`));
  return {type:orbit?'scatter3d':'scattergl',mode:'markers',x:ids.map(i=>orbit?xyz[i*3]:p.pixels[i*2+1]),
    y:ids.map(i=>orbit?xyz[i*3+1]:p.pixels[i*2]),...(orbit?{z:ids.map(i=>xyz[i*3+2])}:{}),
    marker:{size:reference?1.5:orbit?2.4:3,color,opacity:reference?.22:1},
    name:nameOverride||(reference?'Normal frame0':p.refined?'Filtered observed Link5':'Previous scored Link5'),showlegend:true,
    customdata:ids.map(i=>[p.observed_indices[i],scores?.[i]??null,i]),
    hovertemplate:reference?'Normal frame0<extra></extra>':p.refined?'Observed depth pixel %{customdata[0]}<br>Updated support · no new score<extra></extra>':'Point %{customdata[0]}<br>Previous raw anomaly %{customdata[1]:.6g}<extra></extra>'};
}
function render(){renderDirty=true;if(renderRequest||renderBusy)return;renderRequest=requestAnimationFrame(()=>{renderRequest=0;renderDirty=false;renderBusy=true;Promise.resolve(renderFrame()).catch(e=>{$('detail').textContent='Replay error: '+e.message}).finally(()=>{renderBusy=false;if(renderDirty)render();if(seekPending&&!seekInFlight)queueSeek()})})}
async function renderFrame(){
  const token=++renderToken,c=currentCase(),f=c.frames[frame],m=f.models[mode()],summary=c.summaries[mode()];
  $('title').textContent=c.video_id+' · Link5 '+(refinedView()?'depth refinement':'deformation');$('version').textContent=refinedView()?'Local filtered-depth preview':mode()==='robot_structural'?'Robot structural augmentation':'Original repository augmentation';
  if(!seekPending)slider.value=frame;
  $('time').textContent=`Frame ${frame} / ${c.frame_count-1} · ${(frame/c.fps).toFixed(2)} s`;
  $('final').textContent=fmt(summary?.sum_all_frame_raw_mean_top80);$('current').textContent=fmt(m?.raw_mean_top80);$('coverage').textContent=`${summary?.scored_frame_count??0} / ${c.frame_count}`;
  $('final-label').textContent='Previous run · sum of raw top-80';$('current-label').textContent='Previous run · current frame score';$('coverage-label').textContent='Previous run · scored frames';
  $('detail').textContent=`${f.valid_point_count??'Pending'} observed points. ${frame===0?'Frame0 stays in its original camera pose.':'FoundationPose: '+(f.pose?.status??'pending')+(f.pose?.cad_mask_iou!=null?' · CAD-mask IoU '+f.pose.cad_mask_iou.toFixed(3):'')}. Raw point color: 0 to ${fmt(data.color_max[mode()])}, shared across videos.${m?.origin==='frame0_sanity'?' This is an earlier frame0 sanity result; full-video scoring is pending.':''}`;
  const row=depthFrame(c,f),p=await (refinedView()?loadDepth(c,f):loadPacket(c,f));if(token!==renderToken)return;
  if(refinedView())$('detail').textContent=row?`${f.valid_point_count?.toLocaleString()??'Unavailable'} previous points → ${row.filter.final_valid_pixel_count.toLocaleString()} retained (${(row.filter.retained_fraction*100).toFixed(1)}% of raw mask grid). Two-pixel erosion removed ${row.filter.erosion_rejected_pixel_count}; 3D density rejected ${row.filter.density_rejected_pixel_count}. ${row.status==='usable'?'':'Insufficient support. '}Retained XYZ is unchanged; displayed scores and curves belong to the previous GPU run. ${$('space').value==='aligned'?'Using its saved rigid pose.':''}`:'No re-filtered observation is available for this frame.';
  const rgb=await loadImage(f.images.rgb);if(token!==renderToken)return;
  const traces=[],ids=idsFor(p);
  if(showReference&&orbit&&$('space').value==='aligned'){
    const ref=await (refinedView()?loadDepth(c,c.frames[0]):loadPacket(c,c.frames[0]));if(token!==renderToken)return;
    if(ref){const eligible=Array.from({length:ref.count},(_,i)=>i).filter(i=>!ref.refined||ref.reason[i]===0),n=Math.min(1600,eligible.length),sample=Array.from({length:n},(_,i)=>eligible[Math.floor(i*eligible.length/n)]);const t=pointTrace(ref,sample,true);if(t)traces.push(t)}
  }
  if(p){if(p.refined&&$('depth').value==='comparison'){for(const [reason,color,name] of [[0,'#46dcc8','Retained'],[2,'#f2b542','Two-pixel erosion'],[3,'#f45341','Sparse 3D depth support']]){const t=pointTrace(p,ids.filter(i=>p.reason[i]===reason),false,color,name);if(t)traces.push(t)}}else{const t=pointTrace(p,ids);if(t)traces.push(t)}if(activePoint!=null&&activePoint<p.count){const t=pointTrace(p,[activePoint]);if(t){t.marker={size:8,color:'#ffffff'};t.name='Selected point';traces.push(t)}}}
  const bg='#1c2525',layout=orbit?{paper_bgcolor:bg,scene:{aspectmode:'data',camera,
    xaxis:{title:'X · fixed normalization'},yaxis:{title:'Y · fixed normalization'},zaxis:{title:'Z · fixed normalization'},bgcolor:bg},
    margin:{l:0,r:0,t:0,b:0},uirevision:'keep-camera-'+$('space').value}:
    {paper_bgcolor:bg,plot_bgcolor:bg,margin:{l:0,r:0,t:0,b:0},xaxis:{range:[0,rgb?.naturalWidth??640],visible:false},
     yaxis:{range:[rgb?.naturalHeight??360,0],visible:false,scaleanchor:'x',scaleratio:1}};
  layout.legend={font:{color:'#acbeb4'},x:0,y:1};layout.font={color:'#c7d4cb'};
  await Plotly.react('cloud',traces,layout,{responsive:true,displaylogo:false});
  if(!$('cloud').dataset.bound){$('cloud').on('plotly_relayout',ev=>{if(ev['scene.camera'])camera=ev['scene.camera']});$('cloud').on('plotly_click',ev=>{if(ev.points[0]?.customdata){activePoint=ev.points[0].customdata[2];render()}});$('cloud').dataset.bound='1'}
  $('points').replaceChildren();
  if(p?.refined){const el=document.createElement('p');el.textContent='Updated depth support has no anomaly predictions. Select “Previous scored cloud” to inspect the saved point scores.';$('points').append(el)}
  if(p?.scores[mode()])Array.from(p.scores[mode()].keys()).sort((a,b)=>p.scores[mode()][b]-p.scores[mode()][a]).slice(0,80).forEach((i,j)=>{const el=document.createElement('div');el.className='point'+(activePoint===i?' active':'');el.textContent=`${j+1}. Point ${p.observed_indices[i]} · raw ${fmt(p.scores[mode()][i])}`;el.onclick=()=>{activePoint=activePoint===i?null:i;render()};$('points').append(el)});
  const canvas=$('overlay'),ctx=canvas.getContext('2d');canvas.width=rgb?.naturalWidth??640;canvas.height=rgb?.naturalHeight??360;ctx.clearRect(0,0,canvas.width,canvas.height);
  if($('image').value==='depth'&&row?.images.overlay){const overlay=await loadImage(depthHref(row.images.overlay));if(token!==renderToken)return;if(overlay)ctx.drawImage(overlay,0,0,canvas.width,canvas.height)}
  if($('image').value==='heat'){const scored=p?.refined?await loadPacket(c,f):p;if(token!==renderToken)return;if(scored?.scores[mode()]){ctx.globalAlpha=.8;for(let i=0;i<scored.count;i++){ctx.fillStyle=heat(scored.scores[mode()][i]);ctx.fillRect(scored.pixels[i*2+1],scored.pixels[i*2],2,2)}ctx.globalAlpha=1}}
  if($('image').value==='mask'){const mask=await loadImage(f.images.mask);if(mask){ctx.drawImage(mask,0,0,canvas.width,canvas.height);const a=ctx.getImageData(0,0,canvas.width,canvas.height);for(let i=0;i<a.data.length;i+=4){const yes=a.data[i]>127;a.data[i]=74;a.data[i+1]=225;a.data[i+2]=195;a.data[i+3]=yes?105:0}ctx.putImageData(a,0,0)}}
  moveHistoryCursor();
}
// Reuse the published replay's lightweight history cursor and video-owned
// frame synchronization. A frame's cloud is committed after decoding its MP4.
const historyPlot=$('history'),historyCursor=document.createElement('div');historyCursor.className='history-cursor';
function moveHistoryCursor(){const layout=historyPlot._fullLayout;if(!layout)return;if(!historyCursor.isConnected)historyPlot.append(historyCursor);const x=layout.xaxis,y=layout.yaxis,pixel=x.l2p(frame);historyCursor.style.height=y._length+'px';historyCursor.style.transform=`translate(${x._offset+pixel}px,${y._offset}px)`;historyCursor.hidden=pixel<0||pixel>x._length}
async function chart(){const c=currentCase(),values=c.frames.map(f=>f.models[mode()]?.origin==='full_video'&&f.models[mode()].status==='complete'?f.models[mode()].raw_mean_top80:null);
  await Plotly.react('history',[{x:c.frames.map(f=>f.frame_id),y:values,type:'scatter',mode:'lines+markers',line:{color:'#efbd68',width:2},marker:{size:4},connectgaps:false,hovertemplate:'Frame %{x}<br>Raw top-80 %{y:.6g}<extra></extra>'}],
    {paper_bgcolor:'#1c2525',plot_bgcolor:'#1c2525',font:{color:'#c7d4cb'},margin:{l:65,r:15,t:15,b:42},xaxis:{title:'Source frame',gridcolor:'#34423e',range:[0,c.frame_count-1]},yaxis:{title:'Raw top-80 score',gridcolor:'#34423e'}},{responsive:true,displaylogo:false});
  if(!historyPlot.dataset.bound){historyPlot.on('plotly_click',ev=>seek(ev.points[0].x));historyPlot.on('plotly_afterplot',moveHistoryCursor);historyPlot.dataset.bound='1'}moveHistoryCursor()}
function decodedFrame(){const c=currentCase();return clamp(Math.floor(video.currentTime*c.fps+1e-6),0,c.frame_count-1)}
function queueSeek(){requestAnimationFrame(()=>{if(!seekPending||seekInFlight||video.readyState<2||video.seeking)return;if(decodedFrame()===requestedFrame){finishSeek();return}seekInFlight=true;video.currentTime=(requestedFrame+.2)/currentCase().fps})}
function finishSeek(){if(video.seeking)return;seekInFlight=false;frame=decodedFrame();seekPending=frame!==requestedFrame;render();if(seekPending)queueSeek()}
function seek(value){stop();requestedFrame=clamp(Math.round(value),0,currentCase().frame_count-1);seekPending=true;slider.value=requestedFrame;queueSeek()}
function stop(){playing=false;play.textContent='Play';video.pause();if(timer)cancelAnimationFrame(timer);timer=null}
function tick(){if(!playing)return;const next=decodedFrame();if(next!==frame){frame=next;render()}if(video.ended){stop();return}timer=requestAnimationFrame(tick)}
play.onclick=()=>{if(playing){stop();return}if(seekPending)return;playing=true;play.textContent='Pause';video.play().catch(stop);timer=requestAnimationFrame(tick)};
slider.oninput=()=>seek(Number(slider.value));video.onseeked=finishSeek;video.onloadeddata=()=>{if(seekPending)queueSeek();else render()};video.onended=stop;
function openCase(relative=0){stop();const c=currentCase();activePoint=null;slider.max=c.frame_count-1;requestedFrame=Math.round(relative*(c.frame_count-1));frame=requestedFrame;seekInFlight=false;seekPending=true;video.src=c.source_video;video.load();chart();table();render()}
function table(){const t=$('matched');t.innerHTML='<thead><tr><th>Video #</th><th>LVP_ROBOWM</th><th>COSMOS2.5</th><th>COSMOS3</th></tr></thead>';const body=document.createElement('tbody');for(const n of data.numbers){const tr=document.createElement('tr');if(n===$('number').value)tr.className='active';const heading=document.createElement('th');heading.textContent=n;tr.append(heading);for(const g of ['LVP_ROBOWM','COSMOS2.5','COSMOS3']){const c=data.cases.find(c=>c.number===n&&c.generator===g),s=c.summaries[mode()],label=window.LINK5_LABELS?.rows.find(r=>r.video_id===c.video_id)?.forearm_AB,td=document.createElement('td'),button=document.createElement('button');button.textContent=fmt(s?.sum_all_frame_raw_mean_top80);button.onclick=()=>{$('number').value=n;$('generator').value=g;openCase()};const small=document.createElement('small');small.textContent=`${s?.scored_frame_count??0}/${c.frame_count} frames${label!=null?' · human AB '+label:''}`;td.append(button,small);tr.append(td)}body.append(tr)}t.append(body)}
function correlations(){const a=window.LINK5_LABELS;if(!a){$('analysis-status').textContent='Correlation will be calculated after the selected full videos finish.';return}
  $('analysis-status').textContent=`${a.status==='complete'?'Completed':'Partial'} · ${a.selected_videos} selected videos · AB only · descriptive Pearson and Spearman correlations`;
  const table=document.createElement('table');table.innerHTML='<thead><tr><th>Detector</th><th>Group</th><th>Videos</th><th>Pearson r</th><th>Spearman rho</th></tr></thead>';const body=document.createElement('tbody');
  for(const m of data.modes)for(const [group,v] of Object.entries(a.statistics.forearm_AB[m])){const tr=document.createElement('tr');for(const value of [m==='paper_original'?'Original':'Robot structural',group,`${v.n}/${v.selected_n}`,fmt(v.pearson_r),fmt(v.spearman_rho)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}table.append(body);$('correlation-table').replaceChildren(table);
  const traces=['LVP_ROBOWM','COSMOS2.5','COSMOS3'].map(g=>{const rows=a.rows.filter(r=>r.generator===g&&r[mode()]!=null);return{type:'scatter',mode:'markers',name:g,x:rows.map(r=>r.forearm_AB),y:rows.map(r=>r[mode()]),text:rows.map(r=>r.video_id),marker:{size:9},hovertemplate:'%{text}<br>AB label %{x}<br>Full-video sum %{y:.6g}<extra></extra>'}});
  Plotly.react('correlation-chart',traces,{paper_bgcolor:'#12181a',plot_bgcolor:'#1c2525',font:{color:'#c7d4cb'},margin:{l:70,r:20,t:25,b:50},xaxis:{title:'Human forearm label · AB',tickvals:[0,.5,1]},yaxis:{title:'Sum of all frame raw top-80 scores'},legend:{orientation:'h',y:1.15}},{responsive:true,displaylogo:false})
}
data.numbers.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;$('number').append(o)});
$('run-status').textContent=`${data.cases.length} full videos · ${data.numbers.length} matched numbers · ${data.expected_frames.toLocaleString()} source frames. Original ${data.counts.full_video_scored_frames.paper_original}/${data.expected_frames}; robot structural ${data.counts.full_video_scored_frames.robot_structural}/${data.expected_frames}. ${data.sensitivity_status}`;
['number','generator'].forEach(id=>$(id).onchange=()=>openCase(frame/Math.max(1,Number(slider.max))));
$('mode').onchange=()=>{activePoint=null;chart();table();correlations();render()};['space','image','density'].forEach(id=>$(id).onchange=render);
$('reset-view').onclick=()=>{camera=defaultCamera;if(orbit)Plotly.relayout('cloud',{'scene.camera':defaultCamera});render()};
$('view-mode').onclick=()=>{orbit=!orbit;$('view-mode').textContent=orbit?'Camera view':'3D orbit';render()};
$('reference').onclick=()=>{showReference=!showReference;$('reference').textContent=showReference?'Hide normal frame0':'Show normal frame0';render()};
$('reference').textContent='Hide normal frame0';$('refresh').onclick=()=>location.reload();
if(!refinement){$('depth').value='original';$('image').value='heat';$('depth').disabled=true}
$('depth').onchange=()=>{activePoint=null;$('image').value=refinedView()?'depth':'heat';render()};
correlations();openCase();
