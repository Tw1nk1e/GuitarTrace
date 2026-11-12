'use strict';
// Presentation only. Geometry, history, project format and exports remain in editor.js.
const iconPaths = {
  image:'M3 4H21V20H3ZM3 16L8 11L13 16L17 12L21 16M8 8H8.01',
  folder:'M3 6H10L12 8H21V20H3Z', save:'M4 3H17L21 7V21H3V3ZM7 3V9H16V3M7 21V14H17V21',
  download:'M12 3V15M7 10L12 15L17 10M4 16V21H20V16',
  select:'M5 3L19 13L12 14L9 21ZM16 16L20 21', auto:'M4 8V4H8M16 4H20V8M20 16V20H16M8 20H4V16M12 7L13.5 10.5L17 12L13.5 13.5L12 17L10.5 13.5L7 12L10.5 10.5Z',
  circle:'M21 12A9 9 0 1 1 3 12A9 9 0 1 1 21 12',
  rings:'M21 12A9 9 0 1 1 3 12A9 9 0 1 1 21 12M16 12A4 4 0 1 1 8 12A4 4 0 1 1 16 12',
  line:'M4 20L20 4M3 19L5 21M19 3L21 5', rect:'M4 5H20V19H4Z',
  erase:'M3 14L13 4L21 12L13 20H9ZM8 9L16 17M13 20H22',
  bend:'M3 18C7 18 6 6 12 6S17 18 21 18M3 16V20M21 16V20',
  nodes:'M5 17L12 6L19 17M3 15H7V19H3ZM10 4H14V8H10ZM17 15H21V19H17Z',
  duplicate:'M9 9H21V21H9ZM15 6V3H3V15H6', delete:'M3 6H21M9 6V3H15V6M6 6L7 21H17L18 6M10 10V17M14 10V17',
  undo:'M9 4L4 9L9 14M4 9H15A6 6 0 0 1 15 21', redo:'M15 4L20 9L15 14M20 9H9A6 6 0 0 0 9 21',
  join:'M10 14L14 10M8 16L6 18A3 3 0 0 1 2 14L7 9A3 3 0 0 1 11 9M13 15A3 3 0 0 0 17 15L22 10A3 3 0 0 0 18 6L16 8',
  disconnect:'M3 3L21 21M8 16L6 18A3 3 0 0 1 2 14L6 10M18 14L22 10A3 3 0 0 0 18 6L14 10',
  minus:'M5 12H19', plus:'M5 12H19M12 5V19', fit:'M3 9V3H9M15 3H21V9M21 15V21H15M9 21H3V15M7 12H17'
};
for (const button of document.querySelectorAll('[data-icon]')) {
  const icon=node('svg',{viewBox:'0 0 24 24',class:'icon','aria-hidden':'true'});
  icon.append(node('path',{d:iconPaths[button.dataset.icon]}));button.prepend(icon);
  if(!button.title)button.title=button.textContent.trim() || button.getAttribute('aria-label');
}
const toolHints={
  select:'Click to select / drag to move · Alt+click to cycle overlaps · Delete to remove',
  pan:'Drag to pan · Space+drag or middle mouse button',
  auto:'Click a part in the image · SAM requests permission once per session',
  circle:'Drag from the centre to the edge · Esc to cancel',
  rings:'Drag from the ring centre · use the inner handle to adjust the hole',
  line:'Drag between endpoints · Esc to cancel',
  rect:'Drag between opposite corners · Esc to cancel',
  erase:'Drag across a line to cut a segment · Esc to cancel',
  bend:'Drag the highlighted segment · endpoints stay fixed · Esc to cancel',
  nodes:'Select two endpoints to join · drag a node to edit'
};
const geometryFields=['objectX','objectY','objectRotation','objectScale'];
let propertyEdit=null;
function selectedBox() {
  return $('stage').querySelector(`g[data-object="${selected}"]`)?.getBBox();
}
function updateInterface() {
  document.body.dataset.busy=String(busy);
  $('work').setAttribute('aria-busy',String(busy));
  $('sessionState').textContent=busy?'PROCESSING…':$('status').classList.contains('err')?'ERROR':'READY';
  $('toolHint').textContent=toolHints[tool];$('toolHint').title=toolHints[tool];
  $('status').title=$('status').textContent;
  $('propertyMode').textContent=selected?'OBJECT':'NONE';
  $('objectProperties').hidden=!selected;
  $('eraserProperties').hidden=tool!=='erase';
  $('nodeProperties').hidden=tool!=='nodes';
  $('opacityValue').textContent=`${Math.round((state?.opacity ?? 1)*100)}%`;
  updateSizeControls();
  if(!state) { $('selection').textContent='Select an object on the canvas';return; }
  const photo=result.editor.photo;
  $('photoMeta').textContent=`${photo.width} × ${photo.height}`;
  $('vectorMeta').textContent=`${state.objects.length} objects`;
  $('documentSize').textContent=`${photo.width} × ${photo.height} px`;
  $('objectCount').textContent=state.objects.length;
  $('missingPhoto').hidden=!!result.photo;
  if(!selected) { $('selection').textContent='Select an object to edit its stroke and geometry.';return; }
  const obj=state.objects.find(o=>o.id===selected),box=selectedBox();
  if(!obj || !box)return;
  $('objectType').textContent=({path:'Path',line:'Line',circle:'Circle',rect:'Rectangle',compound:'Compound'})[obj.type];
  const linked=(state.joints||[]).some(j=>j.some(r=>r.object===selected));
  for(const id of geometryFields)$(id).disabled=busy || !!edit || !!drag || spacePan || linked;
  const values={objectX:box.x,objectY:box.y,objectRotation:0,objectScale:100};
  for(const [id,value] of Object.entries(values))if(document.activeElement!==$(id))$(id).value=Number(value.toFixed(3));
  $('objectW').textContent=box.width.toFixed(2);$('objectH').textContent=box.height.toFixed(2);
  $('transformNote').textContent=linked?'Disconnect linked nodes before transforming this object.':'Drag an outline to move. Rotation and scale fields apply relative to the current geometry.';
}
// Live property preview, one history entry on commit, Esc restores the base snapshot.
function finishProperty(commit=true) {
  if(!propertyEdit)return;
  const base=propertyEdit.base;propertyEdit=null;
  if(!commit)state=base;
  else if(JSON.stringify(base.objects)!==JSON.stringify(state.objects)){past.push(base);future=[];}
  render();
}
for(const id of geometryFields){
  const input=$(id);
  input.oninput=()=>{
    if(!state || !selected || input.disabled || !input.value || !input.validity.valid)return;
    const value=Number(input.value);if(!Number.isFinite(value))return;
    if(!propertyEdit){const box=selectedBox();if(!box)return;propertyEdit={base:state,object:state.objects.find(o=>o.id===selected),box:{x:box.x,y:box.y,width:box.width,height:box.height}};}
    const {base,object,box}=propertyEdit,center={x:box.x+box.width/2,y:box.y+box.height/2};
    const changed=transformObject(object,center,id==='objectScale'?value/100:1,id==='objectX'?value-box.x:0,id==='objectY'?value-box.y:0,id==='objectRotation'?value*Math.PI/180:0);
    if(!changed)return;
    const next={...base,objects:base.objects.map(o=>o.id===object.id?changed:o)},project=projectData();project.document.objects=next.objects;
    try {GBCVProject.validate(project);}catch(error){status(error.message,true);return;}
    state=next;render();
  };
  input.onchange=()=>finishProperty();input.onblur=()=>{finishProperty();if(state)render();};
  input.onkeydown=e=>{if(e.key==='Escape'){e.preventDefault();finishProperty(false);input.blur();}else if(e.key==='Enter')input.blur();};
}
$('view').addEventListener('pointermove',e=>{
  if(!state)return;const p=drawingPoint(e);
  $('coordinates').textContent=`X: ${p.x.toFixed(1)}  Y: ${p.y.toFixed(1)}`;
});
$('view').addEventListener('pointerleave',()=>{$('coordinates').textContent='X: —  Y: —';});
function resolvedStrokeWidth() {
  const obj=state?.objects.find(o=>o.id===selected);
  if(!obj)return 1;
  if(obj.strokeWidth!==undefined)return obj.strokeWidth;
  const b=selectedBox();if(!b)return 1;
  const c={x:b.x+b.width/2,y:b.y+b.height/2},p=photoPoint(c),px=photoPoint({x:c.x+.01,y:c.y}),py=photoPoint({x:c.x,y:c.y+.01});
  const area=Math.abs((px.x-p.x)*(py.y-p.y)-(px.y-p.y)*(py.x-p.x))/.0001;
  return Number(obj.primitives[0].styles[state.style]['stroke-width'])*Math.sqrt(area);
}
function updateSizeControls() {
  const width=resolvedStrokeWidth(),diameter=eraserRadius()*2;
  for(const prefix of ['stroke','erase']){
    const value=prefix==='stroke'?width:diameter,disabled=prefix==='stroke'?$('strokeWidth').disabled:$('eraseSize').disabled;
    for(const suffix of ['Range','Minus','Plus'])$(prefix+suffix).disabled=disabled;
    $(prefix+'Range').value=value;
    const scale=Math.min(1,48/Math.max(.001,value));
    $(prefix+'PreviewScale').textContent=scale===1?'Preview 1:1':`Preview 1:${(1/scale).toFixed(1)}`;
    const shape=$(prefix+'Preview').firstElementChild;
    if(prefix==='stroke')shape.style.strokeWidth=value*scale;
    else shape.setAttribute('r',value*scale/2);
  }
}
function previewSize(prefix,value) {
  if(prefix==='stroke'){
    if($('strokeWidth').disabled)return;
    $('strokeWidth').value=Math.max(.001,Math.min(100,value));$('strokeWidth').oninput();
  }else{
    if($('eraseSize').disabled)return;
    $('eraseSize').value=Math.round(Math.max(1,Math.min(200,value)));eraserCursor();updateSizeControls();
  }
}
for(const prefix of ['stroke','erase']){
  $(prefix+'Range').oninput=()=>previewSize(prefix,Number($(prefix+'Range').value));
  $(prefix+'Range').onchange=()=>{if(prefix==='stroke')finishWidth();};
  $(prefix+'Range').onblur=()=>{if(prefix==='stroke')finishWidth();};
  $(prefix+'Range').onkeydown=e=>{
    if(e.key==='Escape'){e.preventDefault();if(prefix==='stroke')finishWidth(false);$(prefix+'Range').blur();updateSizeControls();}
    if(e.key==='Enter')$(prefix+'Range').blur();
  };
  for(const [suffix,direction] of [['Minus',-1],['Plus',1]])$(prefix+suffix).onclick=()=>{
    const value=prefix==='stroke'?resolvedStrokeWidth():eraserRadius()*2,step=prefix==='stroke' && value<5 ? .25 : 1;
    previewSize(prefix,Math.round((value+direction*step)*1000)/1000);
    if(prefix==='stroke')finishWidth();
  };
}
$('eraseSize').oninput=()=>{eraserCursor();updateSizeControls();};
function setSplit(percent){
  percent=Math.max(25,Math.min(75,percent));
  // The divider takes 14px; distribute the remaining space without grid overflow.
  $('work').style.setProperty('--split',`calc((100% - 14px) * ${percent/100})`);
  $('splitter').setAttribute('aria-valuenow',Math.round(percent));
  if(state)render();
}
$('splitter').onpointerdown=e=>{e.preventDefault();$('splitter').setPointerCapture(e.pointerId);};
$('splitter').onpointermove=e=>{if(!$('splitter').hasPointerCapture(e.pointerId))return;const r=$('work').getBoundingClientRect();setSplit((e.clientX-r.left)/(r.width-14)*100);};
$('splitter').onpointerup=e=>{if($('splitter').hasPointerCapture(e.pointerId))$('splitter').releasePointerCapture(e.pointerId);};
$('splitter').onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home'].includes(e.key))return;e.preventDefault();setSplit(e.key==='Home'?50:Number($('splitter').getAttribute('aria-valuenow'))+(e.key==='ArrowLeft'?-5:5));};
$('drop').onkeydown=e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();$('file').click();}};
// Status updates can happen without a full render (network errors and completed saves).
new MutationObserver(()=>{ $('status').title=$('status').textContent;$('sessionState').textContent=busy?'PROCESSING…':$('status').classList.contains('err')?'ERROR':'READY'; }).observe($('status'),{childList:true,attributes:true,attributeFilter:['class']});
setSplit(50);refresh();
