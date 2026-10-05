// Separate audit layer. Existing replay data, score and history remain intact.
const occlusionAudit=__OCCLUSION_DATA__;
(()=>{
 const o=occlusionAudit, pct=x=>x===null?'Unavailable':(x*100).toFixed(2)+'%';
 const style=document.createElement('style');style.textContent=`#occlusion-status{padding:7px 10px;border:1px solid #82948a;color:#e7eee9;white-space:nowrap;font-size:13px}#occlusion-status.caught{background:#66382f;border-color:#ff947b;color:#fff0e9}.transport{flex-wrap:wrap}#occlusion-explanation{font-size:13px;line-height:1.5;color:#bdccc3}`;document.head.appendChild(style);
 const transport=document.querySelector('.transport'),status=document.createElement('span');status.id='occlusion-status';transport.appendChild(status);
 const summary=document.querySelector('.summary'),metric=document.createElement('div');
 const label=document.createElement('small');label.textContent='Filtered rigidity · occlusion and failed masks excluded';
 const value=document.createElement('strong');value.id='occlusion-filtered-score';value.textContent=pct(o.score.filtered_rigidity_score);metric.append(label,value);summary.appendChild(metric);
 document.getElementById('final-label').textContent='Original naive rigidity · unchanged';
 const explanation=document.createElement('p');explanation.id='occlusion-explanation';
 explanation.textContent=`Filtered mean: ${o.score.retained_count} retained / ${o.score.retained_count+o.score.excluded_count} scored frames; ${o.score.excluded_count} excluded, including ${o.score.failed_mask_count||0} failed-mask frames. Reference frame is excluded from both means. Other frame values, including carried scores, are unchanged. Not caught does not guarantee a reliable score. `;
 const evidence=document.createElement('a');evidence.href='../score/rigidity_occlusion_filtered.json';evidence.textContent='Filtered-score evidence';explanation.appendChild(evidence);summary.after(explanation);
 let shaded=false;
 const previousRender=render;
 render=function(){previousRender();const row=o.frames[frame],failed=row.mask_valid===false;status.textContent=failed?'FAILED LINK7 MASK · EXCLUDED':row.flagged?'OCCLUSION: CAUGHT':row.status==='assessed'?'Occlusion: not caught':'Occlusion: unassessable';status.classList.toggle('caught',row.flagged||failed);const severityOnly=row.severity_flagged&&!row.legacy_flagged;status.title=failed?`Link7 mask covers ${(row.gripper_area_fraction*100).toFixed(1)}% of the image; skipped by occlusion and filtered scoring`:row.flagged?(severityOnly?(row.severity_continued?'Continued replacement supported by mask evidence':'Severity-supported replacement'):(row.continued?'Continued occlusion episode':'Missing object pixels replaced by link7')):'Current saved detection result';
 if(!shaded&&document.getElementById('history').data){shaded=true;Plotly.relayout('history',{shapes:o.flagged_intervals.map(([a,b])=>({type:'rect',x0:a-.5,x1:b+.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(240,128,104,.22)',line:{width:0},layer:'below'}))});}
 };
 // The original chart promise calls render after its initial setup. If it has
 // already resolved, update this audit layer immediately as well.
 if(document.getElementById('history').data)render();
})();
