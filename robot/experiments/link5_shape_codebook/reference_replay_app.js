'use strict';
const R=window.LINK5_REFERENCE,$=id=>document.getElementById(id);
function unpack(value,Type=Float32Array){const raw=atob(value),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Type(bytes.buffer)}
R.normals.forEach(c=>{c.camera=unpack(c.camera);c.normalized=unpack(c.normalized);if(c.refined){c.refined.camera=unpack(c.refined.camera);c.refined.normalized=unpack(c.refined.normalized)}});
R.examples.forEach(c=>{c.points=unpack(c.points);c.offset=unpack(c.offset);c.mask=unpack(c.mask,Uint8Array);c.magnitude=Array.from({length:c.count},(_,i)=>Math.hypot(...c.offset.slice(i*3,i*3+3)))});
const camera={eye:{x:0,y:0,z:-2.25},up:{x:0,y:-1,z:0}},bg='#1c2525';
function layout(key){return{paper_bgcolor:bg,font:{color:'#c7d4cb'},scene:{aspectmode:'data',camera,xaxis:{title:'X'},yaxis:{title:'Y'},zaxis:{title:'Z'},bgcolor:bg},margin:{l:0,r:0,t:15,b:0},uirevision:key,legend:{font:{size:11},x:0,y:1}}}
function trace(points,name,color,ids=null,size=2){ids=ids||Array.from({length:points.length/3},(_,i)=>i);return{type:'scatter3d',mode:'markers',name,x:ids.map(i=>points[i*3]),y:ids.map(i=>points[i*3+1]),z:ids.map(i=>points[i*3+2]),marker:{size,color,opacity:.8},customdata:ids,hovertemplate:name+'<br>Point %{customdata}<br>(%{x:.6g}, %{y:.6g}, %{z:.6g})<extra></extra>'}}
let imageMode='rgb';
function normal(){const selected=$('normal-choice').value,rows=selected==='all'?R.normals:[R.normals.find(r=>r.video_id===selected)],space=$('coordinates').value;
  const refined=$('normal-depth').value==='refined';
  const traces=rows.map(c=>{const cloud=refined&&c.refined?c.refined:c,index=R.normals.indexOf(c),n=$('density').value==='all'?cloud.count:Math.min(1000,cloud.count),ids=Array.from({length:n},(_,i)=>Math.floor(i*cloud.count/n));return trace(cloud[space],c.video_id,`hsl(${index*137.508%360},52%,65%)`,ids,rows.length===1?2.2:1.5)});
  Plotly.react('normal-cloud',traces,layout('normal-'+space),{responsive:true,displaylogo:false});
  const c=rows[0];$('normal-caption').textContent=c.video_id+' · exact frame0 '+(imageMode==='support'?'updated depth support':imageMode==='rgb'?'RGB':'original SAM mask');$('normal-image').src=imageMode==='support'?(c.support||c.refined?.overlay):c[imageMode];
  $('normal-detail').textContent=refined&&c.refined?`${c.count.toLocaleString()} original training points → ${c.refined.count.toLocaleString()} filtered observed points. Two original-image pixels of support erosion, followed by 3D density rejection. This is a preview; codebooks have not been retrained. ${selected==='all'?'All ${R.normals.length} filtered clouds are overlaid.':''}`:`${c.count.toLocaleString()} exact stored points; mask area ${c.mask_area.toLocaleString()} pixels; mask erosion ${c.mask_erosion_pixels}. ${selected==='all'?'All ${R.normals.length} previous training clouds are overlaid. RGB shows '+c.video_id+'.':''}`;
  if(R.constructed)$('normal-detail').textContent=`${c.count.toLocaleString()} freshly filtered frame0 points; 2-pixel source erosion and ray-normalized 3D density filtering. ${selected==='all'?R.pooled.point_count.toLocaleString()+' total points from exactly four frame0 clouds.':'No later frame enters this reference.'} ${R.training_complete?'Both detector variants trained on these four frame0 inputs.':'Detector retraining is pending.'}`;
  $('input-link').href=c.input;$('training-link').href=c.training_npy;$('filtered-link').href=R.constructed?R.pooled.archive:(c.refined?.archive||c.input);
}
for(const c of R.normals){const option=document.createElement('option');option.value=c.video_id;option.textContent=c.video_id;$('normal-choice').append(option)}
const table=$('normal-table');table.innerHTML='<thead><tr><th>Normal video · frame0</th><th>Observed points</th><th>Original input</th><th>Training array</th><th>Training file SHA256</th></tr></thead>';
for(const c of R.normals){const row=document.createElement('tr');row.innerHTML=`<td><button>${c.video_id}</button></td><td>${c.count}</td><td><a href="${c.input}">NPZ</a></td><td><a href="${c.training_npy}">NPY</a></td><td><code>${c.training_sha256}</code></td>`;row.querySelector('button').onclick=()=>{$('normal-choice').value=c.video_id;normal();$('normal-cloud').scrollIntoView({behavior:'smooth',block:'center'})};table.append(row)}
const scales=['8 × 192','32 × 64','64 × 32'];
if(R.constructed){
  $('normal-depth').disabled=true;$('normal-depth').selectedOptions[0].textContent='Fresh filtered reference';
  $('normal-choice').options[0].textContent='Constructed union · four frame0 clouds';
  if(R.normalization.available===false)$('coordinates').options[1].disabled=true;
  $('codebook-table').innerHTML=`<p>${R.training_complete?'Both detectors trained on four frame0 inputs; their Phase1 normal codebook tensors and hash state match.':'New detector training is pending.'} This normal cloud contains ${R.pooled.point_count.toLocaleString()} observed points from four frame0 inputs.</p><a href="${R.pooled.ply}">Download constructed PLY</a> · <a href="${R.pooled.receipt}">Construction provenance</a>`;
  if(R.training_complete)$('codebook-table').innerHTML+='<table class="data-table"><tr><th>Patch scale</th><th>Shared Phase1 entries</th><th>Final original entries</th><th>Final structural entries</th></tr>'+scales.map((s,i)=>`<tr><td>${s}</td><td>${R.phase1_sizes[i]}</td><td>${R.baseline_sizes[i]}</td><td>${R.structural_sizes[i]}</td></tr>`).join('')+`</table><a href="${R.phase1_archive}">Exact Phase1 codebook tensors</a>`;
  if(!R.examples.length){$('deformation-tab').disabled=true;$('deformation-tab').textContent='Deformation examples pending';}
}else{
  $('codebook-table').innerHTML='<table class="data-table"><tr><th>Patch scale</th><th>Shared Phase1 entries</th><th>Final original entries</th><th>Final structural entries</th></tr>'+scales.map((s,i)=>`<tr><td>${s}</td><td>${R.phase1_sizes[i]}</td><td>${R.baseline_sizes[i]}</td><td>${R.structural_sizes[i]}</td></tr>`).join('')+'</table><p>Feature vectors have 32 dimensions. Both Phase1 codebook tensors were verified equivalent. <a href="../../training/phase1_normal_codebook.pt">Exact Phase1 codebook tensors</a></p>';
}
const N=R.normalization,G=R.geometry;const fmt=x=>Number(x).toFixed(8);
$('normal-math').textContent=R.constructed?`P_reference = concatenate(P_frame0,COSMOS3_0046, P_frame0,COSMOS2.5_0044, P_frame0,LVP_ROBOWM_0044, P_frame0,COSMOS3_0056).\nEvery point uses fresh SAM + MegaSAM + the same refined depth filter.\nNo later-frame training samples, shape completion, per-cloud rescaling or registration.`:`Pₖ = {pᵢ in original frame0 camera space: intact Link5 mask + MegaSAM + existing depth filter}, k = 1,…,${R.normals.length}.\nqᵢ = (pᵢ − c) / s.\nc = (${N.fixed_center.map(fmt).join(', ')}); s = ${fmt(N.fixed_scale)}.\nNo per-cloud centering, rescaling, or FoundationPose registration.`;$('normal-math').style.whiteSpace='pre-wrap';
if(G){$('geometry').textContent=`${R.constructed?'Original camera coordinates':'Fixed normalized coordinates'}:\nu = (${G.axis.map(fmt).join(', ')}).\na = (${G.anchor.map(fmt).join(', ')}).\nL = ${fmt(G.length)}.\nAnchor: observed normal end estimated by pooled frame0 PCA and 1%/99% longitudinal endpoints.`;$('geometry').style.whiteSpace='pre-wrap';}
if(R.examples.length)$('example-source').textContent=`Exact saved example source: ${R.example_source}, frame0 · ${R.examples[0].count.toLocaleString()} corresponding observed points. Gray normal points are from the stored normal.npy file.`;
R.examples.forEach(e=>{const o=document.createElement('option');o.value=e.name;o.textContent=e.name.replaceAll('_',' ');$('example-choice').append(o)});$('example-choice').value='shortened';
const globalMax=R.examples.reduce((maximum,e)=>e.magnitude.reduce((m,v)=>Math.max(m,v),maximum),0);
function deformation(){if(!R.examples.length)return;const e=R.examples.find(e=>e.name===$('example-choice').value),n=R.examples[0],beta=Number($('amount').value)/100;
  const positions=Float32Array.from(e.points,(v,i)=>n.points[i]+beta*(v-n.points[i]));
  const traces=[trace(n.points,'Exact saved normal','#7b8e85',null,1.3)];
  const colors=$('target').value==='mask'?Array.from(e.mask,v=>v?'#f3c77c':'#76bdb5'):$('target').value==='offset'?e.magnitude.map(v=>v*beta):'#e96b54';
  const t=trace(positions,'Deformation · '+e.name,colors);if($('target').value==='offset'){t.marker.colorscale='YlOrRd';t.marker.cmin=0;t.marker.cmax=globalMax;t.marker.showscale=true;t.marker.colorbar={title:'GT offset L2',thickness:12}}traces.push(t);
  const affected=Array.from(e.mask.keys()).filter(i=>e.mask[i]);const arrows=affected.filter((_,j)=>j%Math.max(1,Math.ceil(affected.length/30))===0);
  const ax=[],ay=[],az=[];for(const i of arrows){ax.push(positions[i*3],n.points[i*3],null);ay.push(positions[i*3+1],n.points[i*3+1],null);az.push(positions[i*3+2],n.points[i*3+2],null)}
  traces.push({type:'scatter3d',mode:'lines',x:ax,y:ay,z:az,line:{color:'#edf2ed',width:2},name:'GT offset: deformed → normal',hoverinfo:'skip'});
  if(beta>0&&arrows.length)traces.push({type:'cone',x:arrows.map(i=>n.points[i*3]),y:arrows.map(i=>n.points[i*3+1]),z:arrows.map(i=>n.points[i*3+2]),
    u:arrows.map(i=>n.points[i*3]-positions[i*3]),v:arrows.map(i=>n.points[i*3+1]-positions[i*3+1]),w:arrows.map(i=>n.points[i*3+2]-positions[i*3+2]),
    anchor:'tip',sizemode:'absolute',sizeref:.025,colorscale:[[0,'#edf2ed'],[1,'#edf2ed']],showscale:false,showlegend:false,hoverinfo:'skip'});
  Plotly.react('deformation-cloud',traces,layout('examples'),{responsive:true,displaylogo:false});
  const p=e.parameters;$('example-detail').textContent=e.name==='normal'?'Exact unchanged normal cloud.':`${e.name.replaceAll('_',' ')} · ${Math.round(beta*100)}% of saved deformation · ${e.checks.affected_point_count}/${e.count} affected points. ${p.alpha!=null?'α='+p.alpha+', affected fraction='+p.affected_fraction:'distal angle='+p.angle_degrees+'°, start fraction='+p.start_fraction+', κ='+fmt(p.kappa)+', direction angle='+p.direction_angle+' rad'}. Saved GT offset error ${e.checks.exact_target_max_error}; unaffected offset ${e.checks.unaffected_offset_max}; cross-section radial error ${e.checks.cross_section_radial_norm_max_error.toExponential(3)}.`;
  $('example-ply').href=e.ply;$('example-archive').href=e.archive;
}
let timer=null;function stop(){if(timer)clearInterval(timer);timer=null;$('example-play').textContent='Replay deformation'}
$('example-play').onclick=()=>{if(timer){stop();return}$('amount').value=0;$('example-play').textContent='Pause';timer=setInterval(()=>{const next=Number($('amount').value)+2;$('amount').value=next;deformation();if(next>=100)stop()},100)};
$('amount').oninput=()=>{stop();deformation()};['example-choice','target'].forEach(id=>$(id).onchange=()=>{stop();$('amount').value=100;deformation()});
['normal-choice','coordinates','density'].forEach(id=>$(id).onchange=normal);$('show-rgb').onclick=()=>{imageMode='rgb';normal()};$('show-mask').onclick=()=>{imageMode='mask';normal()};
function section(name){['normal','deformation','guard'].forEach(n=>$(n+'-section').classList.toggle('hidden',n!==name));if(name!=='guard')stopGuard()}
$('normal-tab').onclick=()=>{section('normal');normal()};$('deformation-tab').onclick=()=>{section('deformation');deformation()};
$('normal-reset').onclick=()=>Plotly.relayout('normal-cloud',{'scene.camera':camera});$('deformation-reset').onclick=()=>Plotly.relayout('deformation-cloud',{'scene.camera':camera});
$('normal-depth').onchange=normal;
$('show-support').onclick=()=>{imageMode='support';normal()};
normal();

let guardTimer=null,maskGeneration=0;
function vlmName(model){return /gemini/i.test(model)?'Gemini':/luna/i.test(model)?'Luna':model==='gpt-6.1-sol'?'GPT-6.1 Sol':model||'Unavailable'}
function guardModel(prompt,role){const call=prompt.calls.find(a=>a.role===`link5_${role}_guard`);return call?.model||call?.requested_model||''}
for(const c of R.normals){const o=document.createElement('option');o.value=c.video_id;const names=[...new Set(c.prompt.calls.map(a=>vlmName(a.model)))];o.textContent=c.video_id+' · '+names.join(' / ');$('guard-choice').append(o)}
const guardParams=new URLSearchParams(location.search);let guardInitial=true;
if(R.normals.some(c=>c.video_id===guardParams.get('video')))$('guard-choice').value=guardParams.get('video');
function loadImage(src){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('Cannot load '+src));img.src=src})}
async function guardMask(c){const generation=++maskGeneration,mode=$('guard-mask-mode').value,canvas=$('guard-mask-canvas'),ctx=canvas.getContext('2d');
  try{const img=await loadImage(mode==='source_mask'?c.source_mask:mode==='mask'?c.mask:c.rgb);if(generation!==maskGeneration)return;
    canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);
    if(mode==='overlay'){const mask=await loadImage(c.mask);if(generation!==maskGeneration)return;const layer=document.createElement('canvas');layer.width=canvas.width;layer.height=canvas.height;const lc=layer.getContext('2d');lc.drawImage(mask,0,0);const pixels=lc.getImageData(0,0,layer.width,layer.height);for(let i=0;i<pixels.data.length;i+=4){const on=pixels.data[i]>0;pixels.data[i]=65;pixels.data[i+1]=235;pixels.data[i+2]=220;pixels.data[i+3]=on?130:0}lc.putImageData(pixels,0,0);ctx.drawImage(layer,0,0)}
  }catch(error){if(generation===maskGeneration)$('guard-mask-detail').textContent=error.message}
}
function guard(){const i=$('guard-choice').selectedIndex,c=R.normals[i],variants=c.prompt_variants||[{name:'Previous GPU run',prompt:c.prompt}];
  const menu=$('guard-variant');if(menu.dataset.video!==c.video_id){const previous=menu.selectedOptions[0]?.textContent;menu.replaceChildren();variants.forEach((v,j)=>{const o=document.createElement('option');o.value=j;o.textContent=v.name;menu.append(o)});const same=variants.findIndex(v=>v.name===previous),requested=guardInitial?variants.findIndex(v=>v.id===guardParams.get('vlm')):-1;menu.value=requested>=0?requested:same>=0?same:Math.max(0,variants.findLastIndex(v=>/Gemini/.test(v.name)&&!v.prompt.failed));menu.dataset.video=c.video_id}
  const p=variants[Number(menu.value)].prompt,imageMenu=$('guard-image-mode');
  for(const option of imageMenu.options)option.disabled=!p.images[option.value];
  if(guardInitial&&p.images[guardParams.get('view')])imageMenu.value=guardParams.get('view');
  if(!p.images[imageMenu.value])imageMenu.value='selected';
  guardInitial=false;const view=imageMenu.value;
  $('guard-position').textContent=`${i+1} / ${R.normals.length}`;$('guard-image').src=p.images[view];$('guard-image-link').href=p.images[view];
  $('guard-caption').textContent=c.video_id+' · frame0 · '+$('guard-image-mode').selectedOptions[0].textContent;
  const positiveModel=guardModel(p,'positive'),negativeModel=guardModel(p,'negative');
  const modelLabel=(model,role)=>{const call=p.calls.find(a=>role?a.role===`link5_${role}_guard`:(a.model||a.requested_model)===model);return vlmName(model)+(model?' ('+model+')':'')+(call?.reasoning_effort?' · '+call.reasoning_effort:' · effort unspecified')+(call?.output_token_limit?' · '+call.output_token_limit.toLocaleString()+' output tokens':'')+(call?.timeout_seconds?' · timeout '+call.timeout_seconds+' s':'')+(role&&call?.elapsed_seconds!=null?' · response '+call.elapsed_seconds.toFixed(1)+' s':'')};
  $('guard-model').textContent=view.startsWith('positive')?'Positive-point VLM: '+modelLabel(positiveModel,'positive'):view.startsWith('negative')?'Negative-point VLM: '+modelLabel(negativeModel,'negative'):positiveModel===negativeModel?'Point-review VLM: '+modelLabel(positiveModel):'Positive-point VLM: '+modelLabel(positiveModel,'positive')+' · Negative-point VLM: '+modelLabel(negativeModel,'negative');
  $('guard-mask-detail').textContent=`Source mask ${c.source_image_hw[1]} × ${c.source_image_hw[0]}: ${c.mask_area.toLocaleString()} pixels. Training grid ${c.native_depth_hw[1]} × ${c.native_depth_hw[0]}; nearest-neighbor resolution alignment; ${c.mask_erosion_pixels} source-pixel erosion for depth support. ${c.count.toLocaleString()} observed 3D points used as a normal reference.`;
  $('guard-status').textContent=p.preview_only?`Local ${variants[Number(menu.value)].name} review: ${p.decision}. ${p.failed?'Failed review; displayed points are unreviewed defaults.':'Updated positive and negative proposals are shown.'} These points have not been supplied to SAM; the adjacent mask belongs to the previous GPU run.`:`Negative review: ${p.reviews.negative.decision==='PASS'?'kept proposed N1/N2':'replaced N1/N2'}. Positive review: ${p.p1_vlm?'placed P1/P2/P3 explicitly':p.reviews.positive.decision==='PASS'?'kept proposed P2/P3':'corrected P2/P3'}. Successful masking: ${p.successful_attempt}. Final points match SAM diagnostics; original mask matches the guarded segmentation exactly.`;
  $('guard-points').innerHTML='<thead><tr><th>Point</th><th>SAM label</th><th>Proposed (x, y)</th><th>Used by SAM (x, y)</th><th>VLM</th><th>VLM action</th></tr></thead><tbody>'+p.selected.map((xy,j)=>{const same=xy.every((v,k)=>v===p.original[j][k]),unplaced=p.original[j].some(v=>v<0);return `<tr><td>${j<3?'P'+(j+1):'N'+(j-2)}</td><td>${p.labels[j]?'Positive · 1':'Negative · 0'}</td><td>${unplaced?'No default · VLM placement required':p.original[j].join(', ')}</td><td>${xy.some(v=>v<0)?'Unavailable':xy.join(', ')}</td><td>${j===0&&!p.p1_vlm?'—':vlmName(j<3?positiveModel:negativeModel)}</td><td>${j===0&&!p.p1_vlm?'Deterministic proposal; not VLM reviewed':unplaced?'Placed by VLM':same?'Kept':'Replaced'}</td></tr>`}).join('')+'</tbody>';
  if(p.preview_only)$('guard-points').querySelector('thead tr th:nth-child(4)').textContent='New proposal (x, y)';
  $('guard-answers').textContent=p.calls.map(a=>a.role+' · '+(a.model||a.requested_model)+'\n'+(a.answer||a.response_error||a.parse_error||'No answer')).join('\n\n');$('guard-record').href=p.record;$('guard-diagnostics').href=p.diagnostics;
  $('guard-mask-link').href=c.source_mask;$('guard-grid-mask-link').href=c.mask;guardMask(c);
}
function stopGuard(){if(guardTimer)clearInterval(guardTimer);guardTimer=null;$('guard-play').textContent='Replay all '+R.normals.length}
function stepGuard(delta){$('guard-choice').selectedIndex=($('guard-choice').selectedIndex+delta+R.normals.length)%R.normals.length;guard()}
$('guard-prev').onclick=()=>{stopGuard();stepGuard(-1)};$('guard-next').onclick=()=>{stopGuard();stepGuard(1)};
$('guard-play').onclick=()=>{if(guardTimer){stopGuard();return} $('guard-choice').selectedIndex=0;guard();$('guard-play').textContent='Pause replay';guardTimer=setInterval(()=>{if($('guard-choice').selectedIndex===R.normals.length-1){stopGuard();return}stepGuard(1)},3500)};
['guard-choice','guard-variant','guard-image-mode','guard-mask-mode'].forEach(id=>$(id).onchange=()=>{stopGuard();guard()});
$('guard-tab').onclick=()=>{stop();section('guard');guard()};
if(location.hash==='#guard')$('guard-tab').click();
if(location.hash==='#deformation')$('deformation-tab').click();
