'use strict';
const replay=window.LINK5_SELECTED;
const element=id=>document.getElementById(id);
const initialCamera={eye:{x:.15,y:-1.8,z:.85},up:{x:0,y:0,z:1}};
let normalPoints=null,currentTest=null,syncing=false;
function unpack(encoded){const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));return new Float32Array(bytes.buffer);}
function currentEntry(){return replay.tests.find(t=>t.key===element('test-choice').value);}
function fail(message){element('load-error').hidden=false;element('load-error').textContent=message;}
function loadPacket(path){const script=document.createElement('script');script.src=path;script.onload=()=>script.remove();script.onerror=()=>{script.remove();fail('Could not load this saved point cloud.');};document.head.append(script);}
function loadTest(){currentTest=null;element('test-info').textContent='Loading…';element('test-status').textContent='';element('load-error').hidden=true;loadPacket(currentEntry().packet);}
window.LINK5_SELECTED_NORMAL=packet=>{normalPoints=unpack(packet.points);draw();};
window.LINK5_SELECTED_TEST=packet=>{if(packet.key!==element('test-choice').value)return;currentTest={key:packet.key,points:unpack(packet.points),scores:unpack(packet.scores)};draw();};
function ranges(){const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];for(const points of [normalPoints,currentTest.points])for(let i=0;i<points.length;i++){const axis=i%3;low[axis]=Math.min(low[axis],points[i]);high[axis]=Math.max(high[axis],points[i]);}return low.map((v,i)=>{const pad=Math.max((high[i]-v)*.07,.04);return [v-pad,high[i]+pad];});}
function layout(bounds,revision){const axis=i=>({range:bounds[i],visible:false,showspikes:false});return {paper_bgcolor:'#151a1c',plot_bgcolor:'#151a1c',font:{color:'#c3cdc5',family:'Helvetica Neue, sans-serif'},margin:{l:0,r:24,t:8,b:0},showlegend:false,uirevision:revision,scene:{xaxis:axis(0),yaxis:axis(1),zaxis:axis(2),aspectmode:'data',camera:initialCamera,bgcolor:'#151a1c',dragmode:'orbit'}};}
function trace(points,color,name){const x=[],y=[],z=[];for(let i=0;i<points.length;i+=3){x.push(points[i]);y.push(points[i+1]);z.push(points[i+2]);}return {type:'scatter3d',mode:'markers',name,x,y,z,marker:{size:2,color,opacity:.95},hoverinfo:'skip'};}
const plotConfig={responsive:true,displayModeBar:false,displaylogo:false,scrollZoom:true};
function draw(){if(!normalPoints||!currentTest)return;const e=currentEntry(),bounds=ranges(),flags=element('flagged-view').checked;
  const normal=trace(normalPoints,'#adc2b6','Normal reference');
  const output=trace(currentTest.points,Array.from(currentTest.scores),'Anomaly score');
  output.customdata=Array.from(currentTest.scores);output.hovertemplate='Anomaly score: %{customdata:.5f}<extra></extra>';
  if(flags){output.marker.color=Array.from(currentTest.scores,s=>s>replay.threshold?'#f19c63':'#708580');output.marker.showscale=false;}
  else{output.marker.colorscale=[[0,'#596f6b'],[.32,'#b4c3b8'],[1/3,'#ddb578'],[1,'#f08248']];output.marker.cmin=0;output.marker.cmax=replay.color_max;output.marker.showscale=true;output.marker.colorbar={title:{text:'Score'},thickness:10,len:.6,tickvals:[0,replay.threshold,replay.color_max],ticktext:['0',replay.threshold.toPrecision(3)+' · threshold',replay.color_max.toPrecision(3)+'+'],outlinewidth:0};}
  const tasks=[Plotly.react('normal-cloud',[normal],layout(bounds,e.key),plotConfig),Plotly.react('test-cloud',[output],layout(bounds,e.key),plotConfig)];
  Promise.all(tasks).then(connectViews).catch(()=>fail('The 3D view could not render. Reload this page.'));
  element('normal-info').textContent='4 frame0 clouds · '+(normalPoints.length/3).toLocaleString()+' points';
  element('test-info').textContent=e.point_count.toLocaleString()+' points';
  element('test-status').textContent=(100*e.flagged_fraction).toFixed(1)+'% of test points flagged'+(e.weak_pose?' · weak pose fit':'');
  element('color-note').textContent=flags?'Gray: below threshold · orange: flagged':'Gray: low score · orange: above threshold · shared scale';
}
function connectViews(){for(const [from,to] of [['normal-cloud','test-cloud'],['test-cloud','normal-cloud']]){const plot=element(from);if(plot._link5Sync)return;plot._link5Sync=true;plot.on('plotly_relayout',async event=>{if(syncing||!event['scene.camera'])return;syncing=true;try{await Plotly.relayout(to,{'scene.camera':event['scene.camera']});}finally{syncing=false;}});}}
function step(delta){const menu=element('test-choice');menu.selectedIndex=(menu.selectedIndex+delta+menu.options.length)%menu.options.length;loadTest();}
for(const entry of replay.tests){const option=document.createElement('option');option.value=entry.key;option.textContent=entry.label;element('test-choice').append(option);}
element('test-choice').value=replay.default_test;
element('checkpoint-status').textContent=replay.effective?'Checkpoint · validation passed':'Current checkpoint · validation failed';
element('checkpoint').href=replay.checkpoint;
element('test-choice').onchange=loadTest;
element('previous').onclick=()=>step(-1);element('next').onclick=()=>step(1);
element('flagged-view').onchange=draw;
element('reset').onclick=()=>{for(const id of ['normal-cloud','test-cloud'])Plotly.relayout(id,{'scene.camera':initialCamera});};
loadPacket(replay.normal.packet);loadTest();
