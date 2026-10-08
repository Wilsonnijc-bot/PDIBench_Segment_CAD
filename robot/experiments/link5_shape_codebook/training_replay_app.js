'use strict';
const T=window.LINK5_TRAINING;
const modeName=m=>m==='paper_original'?'Original augmentation':'Robot structural augmentation';
const showSection=section;
let predictionTimer=null,packetGeneration=0,predictionPacket=null;
function stopPrediction(){if(predictionTimer)clearInterval(predictionTimer);predictionTimer=null;$('prediction-play').textContent='Replay examples';}
section=function(name){showSection(name);['training','prediction'].forEach(n=>$(n+'-section').classList.toggle('hidden',n!==name));if(name!=='prediction')stopPrediction();if(name!=='deformation')stop();};
function training(){const component=$('training-loss').value;
  const lines=Object.entries(T.models).map(([m,v],i)=>({type:'scatter',mode:'lines',name:modeName(m),x:v.losses.map(r=>r.epoch),y:v.losses.map(r=>r[component]),line:{color:i?'#f3c77c':'#76bdb5',width:1.7}}));
  Plotly.react('training-chart',lines,{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:'#c7d4cb'},xaxis:{title:'Epoch'},yaxis:{title:component},margin:{l:65,r:20,t:20,b:55},legend:{orientation:'h'},uirevision:'loss'},{responsive:true,displaylogo:false});
  $('training-summary').innerHTML='<tr><th>Detector</th><th>Epochs</th><th>Final total loss</th><th>Validation</th><th>Saved output</th></tr>'+Object.entries(T.models).map(([m,v])=>`<tr><td>${modeName(m)}</td><td>${v.epoch}</td><td>${v.final_loss.total.toPrecision(6)}</td><td>${v.validation} · deformation sensitivity</td><td><a href="${v.checkpoint}">Checkpoint</a> · <a href="${v.loss_csv}">Loss CSV</a> · <a href="${v.validation_receipt}">Validation</a></td></tr>`).join('');
}
function predictionRows(){return T.predictions.filter(r=>r.mode===$('prediction-mode').value);}
function choices(){const menu=$('prediction-choice'),old=menu.selectedOptions[0]?.textContent;menu.replaceChildren();
  const rows=predictionRows(),groups={};for(const kind of ['synthetic_validation','heldout_frame0']){const group=document.createElement('optgroup');group.label=kind==='synthetic_validation'?'Synthetic validation deformations':'Heldout frame0 predictions';groups[kind]=group;menu.append(group);}
  for(const r of rows){const o=document.createElement('option');o.value=r.key;o.textContent=r.name;groups[r.kind].append(o);}
  const match=rows.find(r=>r.name===old),bend=rows.find(r=>r.name==='bending 35');menu.value=(match||bend||rows.find(r=>r.kind==='synthetic_validation')||rows[0]).key;
}
function selectedPrediction(){return T.predictions.find(r=>r.key===$('prediction-choice').value);}
window.LINK5_TRAINING_FRAME=p=>{if(p.key!==$('prediction-choice').value)return;p.points=unpack(p.points);p.scores=unpack(p.scores);if(p.normal)p.normal=unpack(p.normal);if(p.mask)p.mask=unpack(p.mask,Uint8Array);if(p.offset)p.offset=unpack(p.offset);predictionPacket=p;drawPrediction();};
function prediction(){const r=selectedPrediction(),generation=++packetGeneration;predictionPacket=null;
  $('prediction-detail').textContent='Loading saved prediction…';const script=document.createElement('script');script.src=r.packet;script.onload=()=>script.remove();script.onerror=()=>{script.remove();if(generation===packetGeneration)$('prediction-detail').textContent='Could not load saved prediction: '+r.packet;};document.head.append(script);
}
function scoreTrace(points,name,scores,max,title){const t=trace(points,name,Array.from(scores));t.marker.colorscale='Viridis';t.marker.cmin=0;t.marker.cmax=Math.max(max,1e-8);t.marker.showscale=true;t.marker.colorbar={title,thickness:12};t.hovertemplate=name+'<br>Point %{customdata}<br>Score %{marker.color:.7g}<extra></extra>';return t;}
function drawPrediction(){const r=selectedPrediction(),p=predictionPacket;if(!p||p.key!==r.key)return;
  $('prediction-truth').disabled=!p.normal;
  const model=T.models[r.mode],view=$('prediction-view');for(const option of view.options)option.disabled=option.value!=='score'&&!p.offset;if(!p.offset)view.value='score';
  const truth=$('prediction-truth').checked,inputs=[];if(truth&&p.normal)inputs.push(trace(p.normal,'Saved normal target','#7b8e85',null,1.4));inputs.push(trace(p.points,'Actual inference input','#76bdb5'));
  Plotly.react('prediction-input',inputs,layout('prediction-input-'+r.key),{responsive:true,displaylogo:false});
  let predictions=[],label='Actual raw anomaly scores';
  if(view.value==='score')predictions=[scoreTrace(p.points,'Raw anomaly',p.scores,model.color_max,'Raw score')];
  else if(view.value==='offset'){const magnitude=Float32Array.from({length:p.count},(_,i)=>Math.hypot(...p.offset.slice(i*3,i*3+3)));predictions=[scoreTrace(p.points,'Predicted offset L2',magnitude,model.offset_max,'Offset L2')];label='Predicted offset magnitude';}
  else{const amount=Number($('prediction-amount').value)/100,corrected=Float32Array.from(p.points,(v,i)=>v+amount*p.offset[i]);predictions=[trace(p.points,'Inference input','#7b8e85',null,1.3),trace(corrected,'Input + predicted offset','#e96b54')];label=`Predicted corrected geometry · ${Math.round(amount*100)}% of saved offset`;}
  $('prediction-amount').disabled=view.value!=='corrected';if(truth&&p.normal)predictions.unshift(trace(p.normal,'Saved normal target','#7b8e85',null,1.2));
  Plotly.react('prediction-cloud',predictions,layout('prediction-output-'+r.key),{responsive:true,displaylogo:false});$('prediction-label').textContent=label;
  $('prediction-detail').textContent=`${modeName(r.mode)} · ${r.name} · ${p.count.toLocaleString()} exact inference points in shared normalized training coordinates. ${r.kind==='synthetic_validation'?'Synthetic validation deformation of '+r.video_id+' frame0; this is separate from the four-cloud union.':'Heldout '+r.video_id+' frame0; this cloud did not enter training.'} Checkpoint: epoch ${model.epoch}; deformation validation ${model.validation}.`;
  $('prediction-metrics').textContent=`Raw top80 mean ${r.metrics.raw_mean_top80.toPrecision(7)} · point mean ${r.metrics.raw_mean.toPrecision(7)} · p95 ${r.metrics.raw_p95.toPrecision(7)} · maximum ${r.metrics.raw_max.toPrecision(7)}${r.ratio!=null?' · deformation / normal = '+r.ratio.toFixed(6)+' (required 1.20)':''}.`;
  $('prediction-archive').href=r.archive;$('prediction-receipt').href=model.validation_receipt;
}
function stepPrediction(delta){const menu=$('prediction-choice');menu.selectedIndex=(menu.selectedIndex+delta+menu.options.length)%menu.options.length;prediction();}
$('training-tab').onclick=()=>{section('training');training();};$('prediction-tab').onclick=()=>{section('prediction');prediction();};$('training-loss').onchange=training;
$('prediction-mode').onchange=()=>{stopPrediction();choices();prediction();};$('prediction-choice').onchange=()=>{stopPrediction();prediction();};
$('prediction-prev').onclick=()=>{stopPrediction();stepPrediction(-1);};$('prediction-next').onclick=()=>{stopPrediction();stepPrediction(1);};
$('prediction-play').onclick=()=>{if(predictionTimer){stopPrediction();return;}$('prediction-play').textContent='Pause';predictionTimer=setInterval(()=>stepPrediction(1),3500);};
['prediction-view','prediction-truth','prediction-amount'].forEach(id=>$(id).oninput=drawPrediction);
$('prediction-reset').onclick=()=>['prediction-input','prediction-cloud'].forEach(id=>Plotly.relayout(id,{'scene.camera':camera}));
choices();if(location.hash==='#training')$('training-tab').click();if(location.hash==='#prediction')$('prediction-tab').click();
