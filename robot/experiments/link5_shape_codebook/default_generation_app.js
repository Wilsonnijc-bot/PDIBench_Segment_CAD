'use strict';
const DG=window.LINK5_DEFAULT_GENERATION;
for(const n of Object.values(DG.normals))n.points=unpack(n.points);
let defaultPacket=null,defaultTimer=null,defaultBaseKey='',defaultSync=false;
const priorDefaultSection=section;
section=function(name){priorDefaultSection(name);$('generation-section').classList.toggle('hidden',name!=='generation');if(name!=='generation')stopDefaultAnimation();};
for(const video of Object.keys(DG.normals)){const option=document.createElement('option');option.value=video;option.textContent=video;$('generation-video').append(option);}
function selectedDefault(){return DG.examples.find(e=>e.video_id===$('generation-video').value&&e.type===$('generation-type').value&&e.severity===Number($('generation-severity').value));}
function stopDefaultAnimation(){if(defaultTimer)clearInterval(defaultTimer);defaultTimer=null;$('generation-play').textContent='Animate deformation';}
function loadDefaultExample(){stopDefaultAnimation();const used=['sink','concavity','bulges'].includes($('generation-type').value);
  for(const option of $('generation-severity').options)option.disabled=!used&&option.value!=='0.01';if(!used)$('generation-severity').value='0.01';
  $('generation-amount').value=100;defaultPacket=null;const e=selectedDefault();$('generation-detail').textContent='Loading generated geometry…';
  const script=document.createElement('script');script.src=e.packet;script.onload=()=>script.remove();script.onerror=()=>{script.remove();$('generation-detail').textContent='Could not load '+e.packet;};document.head.append(script);
}
window.LINK5_DEFAULT_FRAME=p=>{if(p.key!==selectedDefault().key)return;p.points=unpack(p.points);p.mask=unpack(p.mask,Uint8Array);defaultPacket=p;defaultBaseKey='';drawDefaultExample();};
function defaultLayout(e,points,normal){const result=layout('default-'+e.key+'-'+$('generation-zoom').checked);
  if($('generation-zoom').checked){const ids=Array.from(defaultPacket.mask.keys()).filter(i=>defaultPacket.mask[i]);for(const [axis,j] of [['x',0],['y',1],['z',2]]){let lo=Infinity,hi=-Infinity;for(const i of ids){lo=Math.min(lo,normal[i*3+j],points[i*3+j]);hi=Math.max(hi,normal[i*3+j],points[i*3+j]);}const pad=Math.max(.004,(hi-lo)*.1);if(ids.length)result.scene[axis+'axis'].range=[lo-pad,hi+pad];}}
  return result;
}
function synchronizeDefaultViews(){for(const [source,target] of [['generation-original','generation-deformed'],['generation-deformed','generation-original']]){const graph=$(source);if(graph.defaultCameraListener)continue;graph.defaultCameraListener=true;graph.on('plotly_relayout',event=>{if(defaultSync||!event['scene.camera'])return;defaultSync=true;Plotly.relayout(target,{'scene.camera':event['scene.camera']}).finally(()=>{defaultSync=false;});});}}
function drawDefaultExample(){const e=selectedDefault(),p=defaultPacket;if(!p||p.key!==e.key)return;const normal=DG.normals[e.video_id].points,beta=Number($('generation-amount').value)/100;
  const points=Float32Array.from(p.points,(v,i)=>normal[i]+beta*(v-normal[i]));const layers=[];
  if($('generation-overlay').checked)layers.push(trace(normal,'Original overlay','#83958c',null,1.3));
  const colors=$('generation-color').value==='mask'?Array.from(p.mask,v=>v?'#ed725d':'#72bcb4'):Array.from({length:e.point_count},(_,i)=>Math.hypot(...points.slice(i*3,i*3+3).map((v,j)=>v-normal[i*3+j])));
  const deformed=trace(points,'Actual generated geometry',colors,null,2.4);
  if($('generation-color').value==='displacement'){deformed.marker.colorscale='YlOrRd';deformed.marker.cmin=0;deformed.marker.cmax=.1;deformed.marker.showscale=true;deformed.marker.colorbar={title:'Actual motion · normalized',thickness:12};}
  layers.push(deformed);
  const affected=Array.from(p.mask.keys()).filter(i=>p.mask[i]),arrows=affected.filter((_,j)=>j%Math.max(1,Math.ceil(affected.length/25))===0),x=[],y=[],z=[];
  for(const i of arrows){x.push(points[i*3],normal[i*3],null);y.push(points[i*3+1],normal[i*3+1],null);z.push(points[i*3+2],normal[i*3+2],null);}
  if(e.used_for_training)layers.push({type:'scatter3d',mode:'lines',x,y,z,line:{color:'#edf2ed',width:2},name:'Restoration toward original',hoverinfo:'skip'});
  const key=e.key+'-'+$('generation-zoom').checked;
  const base=key!==defaultBaseKey?Plotly.react('generation-original',[trace(normal,'Original normal cloud','#a3b5ab',null,2)],defaultLayout(e,p.points,normal),{responsive:true,displaylogo:false}):Promise.resolve();defaultBaseKey=key;
  Promise.all([base,Plotly.react('generation-deformed',layers,defaultLayout(e,p.points,normal),{responsive:true,displaylogo:false})]).then(synchronizeDefaultViews);
  $('generation-detail').textContent=`${e.video_id} · frame0 · ${e.type} · severity ${e.severity} · ${e.point_count.toLocaleString()} observed points · ${e.labeled_point_count.toLocaleString()} labeled (${(e.labeled_fraction*100).toFixed(2)}%) · ${Math.round(beta*100)}% of actual deformation. ${e.used_for_training?'This generator/severity was used in the completed default-generator comparison.':'EXCLUDED PREVIEW: coordinates are unchanged; colored labels do not represent a real deformation.'}`;
  $('generation-check').textContent=e.used_for_training?`Maximum actual displacement ${e.actual_max_displacement.toPrecision(5)} in normalized units; ${e.actual_max_camera_displacement.toPrecision(5)} in stored camera units. Same generator, point ordering and seed ${e.seed} as selected training draw ${e.reconstructed_training_draw+1}; re-generated for inspection. Original historical minibatches were not archived.`:`Actual maximum displacement is zero. The returned restoration target differs from the real coordinate difference by up to ${e.reported_target_max_error.toPrecision(5)}. This inconsistent example is excluded from training.`;
  $('generation-archive').href=e.archive;$('generation-input').href=DG.normals[e.video_id].source;$('generation-manifest').href=DG.manifest;
}
$('generation-tab').onclick=()=>{section('generation');if(defaultPacket)drawDefaultExample();else loadDefaultExample();};
['generation-video','generation-type','generation-severity'].forEach(id=>$(id).onchange=loadDefaultExample);
['generation-color','generation-overlay','generation-zoom'].forEach(id=>$(id).onchange=drawDefaultExample);
$('generation-amount').oninput=()=>{stopDefaultAnimation();drawDefaultExample();};
$('generation-play').onclick=()=>{if(defaultTimer){stopDefaultAnimation();return;}if(!defaultPacket)return;$('generation-amount').value=0;$('generation-play').textContent='Pause';defaultTimer=setInterval(()=>{const value=Math.min(100,Number($('generation-amount').value)+5);$('generation-amount').value=value;drawDefaultExample();if(value===100)stopDefaultAnimation();},180);};
$('generation-reset').onclick=()=>['generation-original','generation-deformed'].forEach(id=>Plotly.relayout(id,{'scene.camera':camera}));
const defaultParams=new URLSearchParams(location.search);
if(DG.normals[defaultParams.get('video')])$('generation-video').value=defaultParams.get('video');
if(Array.from($('generation-type').options).some(o=>o.value===defaultParams.get('atype')))$('generation-type').value=defaultParams.get('atype');
if(['0.001','0.01','0.1'].includes(defaultParams.get('severity')))$('generation-severity').value=defaultParams.get('severity');
if(location.hash==='#generation')$('generation-tab').click();
