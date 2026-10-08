'use strict';
const ST=window.LINK5_STRUCTURAL;
let structuralPacket=null;
const priorStructuralSection=section;
section=function(name){priorStructuralSection(name);$('structural-section').classList.toggle('hidden',name!=='structural');};
const structuralPercent=n=>(n*100).toFixed(2)+'%';
function structuralRows(){const rows=ST.entries.filter(e=>e.kind===$('structural-kind').value);if($('structural-kind').value==='normal_prediction')rows.sort((a,b)=>b.validation_metrics.false_positive_fraction-a.validation_metrics.false_positive_fraction);return rows;}
function selectedStructural(){return ST.entries.find(e=>e.key===$('structural-choice').value);}
function structuralChoices(){const menu=$('structural-choice');menu.replaceChildren();for(const e of structuralRows()){const option=document.createElement('option');option.value=e.key;option.textContent=e.name;menu.append(option);}const initial=structuralRows().find(e=>e.name.includes('shortening_25pct'));if(initial)menu.value=initial.key;}
window.LINK5_STRUCTURAL_FRAME=p=>{if(p.key!==$('structural-choice').value)return;for(const name of ['points','normal','scores','offset','gt','source'])if(p[name])p[name]=unpack(p[name]);if(p.mask)p.mask=unpack(p.mask,Uint8Array);structuralPacket=p;drawStructural();};
function loadStructural(){const e=selectedStructural();structuralPacket=null;$('structural-detail').textContent='Loading saved cloud…';const script=document.createElement('script');script.src=e.packet;script.onload=()=>script.remove();script.onerror=()=>{$('structural-detail').textContent='Could not load '+e.packet;script.remove();};document.head.append(script);}
function structuralHeat(points,scores,name,max){const t=trace(points,name,Array.from(scores));t.marker.colorscale='Viridis';t.marker.cmin=0;t.marker.cmax=Math.max(max,1e-8);t.marker.showscale=true;t.marker.colorbar={title:name,thickness:12};return t;}
function drawStructural(){const e=selectedStructural(),p=structuralPacket;if(!p||p.key!==e.key)return;const view=$('structural-view');for(const o of view.options)o.disabled=o.value==='mask'?!p.mask:!p.scores;if(view.selectedOptions[0]?.disabled)view.value=p.scores?'score':'mask';
  $('structural-target').disabled=!p.normal;const show=$('structural-target').checked&&p.normal,inputs=[];if(show)inputs.push(trace(p.normal,p.mask?'Normal target':'Observed frame0 companion','#81928b',null,1.4));inputs.push(trace(p.points,'Actual input','#76bdb5'));Plotly.react('structural-input',inputs,layout('st-input-'+e.key),{responsive:true,displaylogo:false});
  const output=[];if(show)output.push(trace(p.normal,p.mask?'Normal target':'Observed frame0 companion','#81928b',null,1.4));let title;
  if(view.value==='score'){output.push(structuralHeat(p.points,p.scores,'Raw score',ST.color_max));title='Actual raw anomaly score · fixed color range';}
  else if(view.value==='mask'){output.push(trace(p.points,'Deliberately moved region',Array.from(p.mask,v=>v?'#ed725d':'#76bdb5')));title=p.scores?'Known GT region · compare with predicted score':'Actual optimizer input · GT, before inference';}
  else if(view.value==='offset'){const magnitude=Float32Array.from({length:p.count},(_,i)=>Math.hypot(...p.offset.slice(i*3,i*3+3)));output.push(structuralHeat(p.points,magnitude,'Predicted offset L2',ST.color_max));title='Saved predicted restoration magnitude';}
  else{output.push(trace(p.points,'Input','#81928b',null,1.4));output.push(trace(Float32Array.from(p.points,(v,i)=>v+p.offset[i]),'Input + predicted offset','#ed725d'));title='Actual input + saved predicted restoration';}
  Plotly.react('structural-output',output,layout('st-output-'+e.key),{responsive:true,displaylogo:false});$('structural-output-label').textContent=title;
  $('structural-detail').textContent=`${e.name} · ${p.count.toLocaleString()} points · ${e.kind==='optimizer_example'?'Captured optimizer pair derived only from this normal frame0; shared benign variation can be present. No detector output is claimed for this archived pair.':e.kind==='guide_prediction'?'Observed guide/frame0 prediction after native rigid pose alignment; no dense real GT.':'Actual pose-aligned inference; normal observations did not enter optimization.'}`;
  const m=e.validation_metrics;let metrics=e.scored?`Raw top80 ${e.metrics.raw_mean_top80.toPrecision(6)} · mean ${e.metrics.raw_mean.toPrecision(6)} · normal-only point threshold ${ST.threshold.toPrecision(6)}.`:'Ground truth is exactly normal minus the displayed synthetic input.';
  if(m?.point_auroc!=null)metrics+=` Point AUROC ${m.point_auroc.toFixed(4)} · region/background ${m.region_vs_outside_ratio.toFixed(3)}× · matched-region/clean ${m.matched_region_score_ratio.toFixed(3)}× · defect recall ${structuralPercent(m.recall_at_frozen_clean_threshold)}.`;
  if(m?.false_positive_fraction!=null)metrics+=` Normal-point false alarms ${structuralPercent(m.false_positive_fraction)}.`;
  if(m?.cad_mask_iou!=null)metrics+=` CAD-mask pose IoU ${m.cad_mask_iou.toFixed(3)}.`;
  $('structural-metrics').textContent=metrics;$('structural-archive').href=e.archive;$('structural-rgb').src=e.rgb||'';$('structural-rgb').hidden=!e.rgb;
}
function structuralHeader(){const a=ST.acceptance,s=ST.summary;$('structural-status').textContent=`${ST.trial} · ${ST.variant} · selected epoch ${ST.epoch}. Research gate: ${a.status.toUpperCase()}. Threshold sensitivity check: ${ST.effective_research_check?'PASSED':'NOT PASSED'}. Four normal frame0 inputs; zero guide optimization inputs.`;
  $('structural-summary').innerHTML=`<tr><th>Point AUROC</th><th>Region / background</th><th>Matched region / clean</th><th>Normal FPR median / p95</th><th>Strong defect recall: short / long / bend</th></tr><tr><td>${s.median_point_auroc.toFixed(4)}</td><td>${s.median_region_vs_outside_ratio.toFixed(3)}×</td><td>${s.median_matched_region_score_ratio.toFixed(3)}×</td><td>${structuralPercent(s.median_heldout_clean_false_positive_fraction)} / ${structuralPercent(a.p95_heldout_clean_false_positive)}</td><td>${['shortening','lengthening','bending'].map(k=>structuralPercent(s.strong_family_metrics[k].median_defect_point_recall)).join(' / ')}</td></tr>`;
  $('structural-comparisons').innerHTML='<tr><th>Trial / head</th><th>Epoch</th><th>AUROC</th><th>Normal FPR median / p95</th><th>Research gate</th><th>Evidence</th></tr>'+ST.comparisons.map(c=>`<tr><td>${c.trial} / ${c.variant}</td><td>${c.epoch}</td><td>${c.summary.median_point_auroc.toFixed(4)}</td><td>${structuralPercent(c.summary.median_heldout_clean_false_positive_fraction)} / ${structuralPercent(c.acceptance.p95_heldout_clean_false_positive)}</td><td>${c.acceptance.status}${c.effective?' · sensitivity passed':''}</td><td><a href="${c.report}">Report</a></td></tr>`).join('');
  $('structural-checkpoint').href=ST.checkpoint;$('structural-policy').href=ST.policy;$('structural-validation').href=ST.validation;
}
function stepStructural(delta){const menu=$('structural-choice');menu.selectedIndex=(menu.selectedIndex+delta+menu.options.length)%menu.options.length;loadStructural();}
$('structural-tab').onclick=()=>{section('structural');structuralHeader();if(structuralPacket)drawStructural();else loadStructural();};
$('structural-kind').onchange=()=>{structuralChoices();loadStructural();};$('structural-choice').onchange=loadStructural;
$('structural-prev').onclick=()=>stepStructural(-1);$('structural-next').onclick=()=>stepStructural(1);
['structural-view','structural-target'].forEach(id=>$(id).oninput=drawStructural);
$('structural-reset').onclick=()=>['structural-input','structural-output'].forEach(id=>Plotly.relayout(id,{'scene.camera':camera}));
structuralChoices();if(location.hash==='#structural')$('structural-tab').click();
