'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let result = null, file = null, busy = false, selected = null, tool = 'select';
let state = null, past = [], future = [], drag = null, spacePan = false;
let edit = null, manualSerial = 0;
let savedFingerprint = null, documentName = 'guitar';
let eraserPointer = null;
let bendHover = null;
let nodeSelection = [], widthStart = null;
let samAllowedSid = null, autoRequest = null;
const toolIds = ['select', 'nodes', 'auto', 'erase', 'bend', 'circle', 'rings', 'line', 'rect'];
const z = {s: 1, x: 0, y: 0};
const labels = {silhouette: 'Silhouette', board: 'Fretboard edge', nut: 'Nut', fret: 'Fret', post: 'Tuner bushing', key: 'Tuner outline', control: 'Control knob', hardware: 'Hardware', manual: 'Manual element'};
function node(tag, attrs = {}) {
  const el = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}
function status(text, err = false) { $('status').textContent = text; $('status').className = err ? 'err' : ''; $('busyLabel').textContent = text; }
function closeSession() { if (result?.id) navigator.sendBeacon(`/api/s/${result.id}/close`); }
function projectData() {
  return {format:GBCVProject.FORMAT,version:GBCVProject.VERSION,
    document:{...result.editor,objects:state.objects,joints:state.joints || []},
    display:{style:state.style,photoVisible:state.photoVisible,opacity:state.opacity},viewport:{...z},manualSerial};
}
function fingerprint() {
  // Unused IDs consumed by cancelled gestures are not document changes.
  const data=projectData();delete data.manualSerial;
  return JSON.stringify(data);
}
function isDirty() { return !!state && (savedFingerprint===null || fingerprint()!==savedFingerprint); }
function updateDirty() {
  const dirty=isDirty(),text=!state?'':dirty?'Unsaved changes':'Project saved';
  if($('dirty').textContent!==text)$('dirty').textContent=text;
  $('dirty').dataset.dirty=String(dirty);
}
function mayReplace() { return !isDirty() || window.confirm('Discard unsaved changes and continue?'); }
function refresh() {
  $('busyOverlay').hidden = !busy;
  $('drop').setAttribute('aria-disabled', String(busy));
  $('cancelAuto').hidden = !autoRequest;
  $('autoProgress').hidden = !busy;
  for (const id of ['open', 'openProject']) $(id).disabled = busy || !!edit;
  for (const id of [...toolIds, 'photoVisible', 'opacity', 'style', 'eraseSize', 'zin', 'zout', 'zfit']) $(id).disabled = busy || !!edit || !state;
  $('saveProject').disabled = busy || !!edit || !!drag || !state;
  $('delete').disabled = busy || !!edit || !!drag || spacePan || !selected;
  $('duplicate').disabled = $('delete').disabled;
  $('undo').disabled = busy || !!edit || !!drag || spacePan || !past.length;
  $('redo').disabled = busy || !!edit || !!drag || spacePan || !future.length;
  $('download').disabled = $('downloadPng').disabled = busy || !!edit || !state;
  for (const id of toolIds) $(id).setAttribute('aria-pressed', tool === id);
  $('view').classList.toggle('draw', !['select', 'pan'].includes(tool) && !spacePan && !drag);
  $('view').classList.toggle('pan', tool === 'pan' || spacePan || !!drag);
  $('view').classList.toggle('erasing', tool==='erase' && !spacePan && !drag);
  $('view').style.cursor='';
  eraserCursor();
  const obj = state?.objects.find(o => o.id === selected);
  const locked=busy || !!edit || !!drag || spacePan || !obj;
  $('strokeWidth').disabled=$('resetWidth').disabled=locked;
  if(document.activeElement!==$('strokeWidth'))$('strokeWidth').value=obj?.strokeWidth ?? '';
  const available=state && nodeSelection.length?GBCVNodes.points(state.objects.filter(o=>nodeSelection.some(r=>r.object===o.id))):[];
  nodeSelection=nodeSelection.filter(r=>available.some(p=>GBCVNodes.key(p)===GBCVNodes.key(r)));
  $('join').disabled=locked || tool!=='nodes' || nodeSelection.length!==2 || nodeSelection.some(r=>!available.find(p=>GBCVNodes.key(p)===GBCVNodes.key(r))?.endpoint);
  $('disconnect').disabled=locked || !nodeSelection.some(r=>(state.joints||[]).some(j=>j.some(p=>GBCVNodes.key(p)===GBCVNodes.key(r))));
  $('selection').textContent = obj ? `${labels[obj.category]} · ${obj.id}${obj.type === 'compound' ? ' · compound object' : ''}` : 'Nothing selected';
  updateDirty();
  if (typeof updateInterface === 'function') updateInterface();
}
// Objects and primitives are immutable. History stores document snapshots, never SVG DOM.
function change(next) {
  past.push(state); state = next; future = []; render();
}
function undo() {
  if (busy || edit || drag || spacePan || !past.length) return;
  bendHover=null;
  nodeSelection=[];
  future.push(state); state = past.pop(); selected = null; render();
}
function redo() {
  if (busy || edit || drag || spacePan || !future.length) return;
  bendHover=null;
  nodeSelection=[];
  past.push(state); state = future.pop(); selected = null; render();
}
function removeSelected() {
  if (busy || edit || drag || spacePan || !selected) return;
  const objects = state.objects.filter(o => o.id !== selected);
  const joints=GBCVNodes.reconcile(state.objects,objects,state.joints||[]);
  selected = null; change({...state, objects,joints});
}
function copyObject(source,dx,dy,id=`manual-${++manualSerial}`) {
  const obj=transformObject(source,{x:0,y:0},1,dx,dy);
  if (!obj) return null;
  return {...obj,id,order:Math.max(0,...state.objects.map(o=>o.order))+1};
}
function duplicateSelected() {
  if (!state || busy || edit || drag || spacePan || !selected) return;
  const source=state.objects.find(o=>o.id===selected),obj=source && copyObject(source,5,5);
  if (!obj) return;
  const project=projectData();project.document.objects=[...state.objects,obj];
  try {GBCVProject.validate(project);}catch(error){status(error.message,true);return;}
  selected=obj.id;change({...state,objects:[...state.objects,obj]});
}
function eraserRadius() { return Math.max(1,Math.min(200,Number($('eraseSize').value)||12))/2; }
function eraserCursor() {
  const cursor=$('eraserCursor');
  cursor.hidden=!state || busy || tool!=='erase' || spacePan || !!drag || !eraserPointer;
  if (cursor.hidden) return;
  const r=$('view').getBoundingClientRect(),size=eraserRadius()*2*z.s;
  cursor.style.left=`${eraserPointer.clientX-r.left}px`;cursor.style.top=`${eraserPointer.clientY-r.top}px`;
  cursor.style.width=cursor.style.height=`${size}px`;
}
function svgGeometry(p) { const {bends,...geometry}=p.geometry;return geometry; }
function objectWidth(obj) {
  if(obj.strokeWidth===undefined)return null;
  // A scalar SVG stroke: image-pixel width at the object's centre (local area scale).
  const ps=obj.primitives.flatMap(p=>p.type==='circle'?[{x:+p.geometry.cx,y:+p.geometry.cy}]:p.type==='rect'?[{x:+p.geometry.x,y:+p.geometry.y}]:GBCVBend.commands(p).flatMap(c=>c.ps));
  const c={x:ps.reduce((s,p)=>s+p.x,0)/ps.length,y:ps.reduce((s,p)=>s+p.y,0)/ps.length};
  const [a,b,d,e,f,g,h,i,j]=result.editor.photo.projection,w=h*c.x+i*c.y+j;
  const determinant=a*(f*j-g*i)-b*(e*j-g*h)+d*(e*i-f*h);
  return obj.strokeWidth/Math.sqrt(Math.abs(determinant/(w*w*w)));
}
function primitive(p,obj) { const width=objectWidth(obj);return node(p.type, {...svgGeometry(p), ...p.styles[state.style],...(width===null?{}:{'stroke-width':width})}); }
// This exact layer is used by both views and both exports. Selection/photo never enter it.
function drawing() {
  const layer = node('g');
  for (const obj of [...state.objects].sort((a, b) => a.order - b.order)) {
    const group = node('g', {'data-object': obj.id, 'data-category': obj.category});
    for (const p of obj.primitives) group.append(primitive(p,obj));
    layer.append(group);
  }
  return layer;
}
// Visual markers and acquisition targets are deliberately separate. Targets are
// generous in SCREEN pixels and overlapping targets resolve to the nearest node.
function controlMarker(root,point,scale,attrs,fill='white',radius=7) {
  root.append(node('circle',{cx:point.x,cy:point.y,r:14/scale,class:'handle handle-hit',
    style:'fill:transparent;stroke:none;pointer-events:all',...attrs}));
  root.append(node('circle',{cx:point.x,cy:point.y,r:radius/scale,class:'handle handle-mark',
    style:`vector-effect:none;stroke-width:${2/scale};fill:${fill};pointer-events:none`}));
}
function nearestControl(e,attribute) {
  const view=$('view').getBoundingClientRect();let nearest=null,distance=14;
  for(const marker of $('stage').querySelectorAll(`circle[${attribute}]`)){
    const p=photoPoint({x:+marker.getAttribute('cx'),y:+marker.getAttribute('cy')});
    const d=Math.hypot(view.left+z.x+z.s*p.x-e.clientX,view.top+z.y+z.s*p.y-e.clientY);
    if(d<=distance){nearest=marker;distance=d;}
  }
  return nearest;
}
function hoverNode(e) {
  const nearest=e && tool==='nodes' && !busy && !edit && !drag && !spacePan?nearestControl(e,'data-node'):null;
  for(const marker of $('stage').querySelectorAll('[data-node]'))marker.style.fill=marker===nearest?'#ff5a1f28':'transparent';
  $('view').style.cursor=nearest?'pointer':'';
}
function interactions(root, scale, handles=false) {
  const hits = node('g'), highlights = node('g');
  for (const obj of state.objects) {
    for (const p of obj.primitives) {
      const hit = node(p.type, {...svgGeometry(p), class: 'hit', 'data-object': obj.id});
      hit.style.strokeWidth = `${Math.max(10, Number(objectWidth(obj) ?? p.styles[state.style]['stroke-width']) * scale)}px`;
      if (obj.category === 'manual' && ['circle','rect'].includes(p.type)) { hit.style.fill = 'transparent'; hit.style.pointerEvents = 'all'; }
      hits.append(hit);
      if (obj.id === selected && tool!=='bend') highlights.append(node(p.type, {...svgGeometry(p), class: 'highlight'}));
    }
  }
  root.append(hits, highlights);
  if (tool==='bend' && bendHover && !spacePan && !drag) {
    const p=state.objects.find(o=>o.id===bendHover.id)?.primitives[bendHover.primitive];
    if(p) root.append(node('path',{d:GBCVBend.highlight(p,bendHover.hit),class:'highlight bend-highlight'}));
  }
  if (!handles || ['erase','bend','auto'].includes(tool)) return;
  const obj = state.objects.find(o => o.id === selected);
  if(tool==='nodes') {
    const ids=new Set([selected,...nodeSelection.map(r=>r.object)]);
    for(const o of state.objects.filter(o=>ids.has(o.id)))for(const p of o.primitives)for(const c of GBCVBend.commands(p))if(c.op==='C'){
      for(const [a,b] of [[c.ps[0],c.ps[1]],[c.ps[2],c.ps[3]]])root.append(node('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'highlight',style:'stroke:#4c8fac;stroke-width:1;stroke-dasharray:3 3'}));
    }
    for(const r of GBCVNodes.points(state.objects.filter(o=>ids.has(o.id)))){
      const s=photoPoint(r.point),t=photoPoint({x:r.point.x+1,y:r.point.y}),sc=Math.max(.001,Math.hypot(t.x-s.x,t.y-s.y)*z.s);
      const chosen=nodeSelection.some(p=>GBCVNodes.key(p)===GBCVNodes.key(r));
      const linked=(state.joints||[]).some(j=>j.some(p=>GBCVNodes.key(p)===GBCVNodes.key(r)));
      controlMarker(root,r.point,sc,{'data-node':JSON.stringify(r)},chosen?'#ff5a1f':linked?'#86d7ff':r.end.startsWith('control-')?'#dcefff':'white',r.end.startsWith('control-')?6:8);
    }return;
  }
  if (obj) for (const [name, point] of Object.entries(objectHandles(obj))) {
    const screen = photoPoint(point), next = photoPoint({x:point.x+1,y:point.y});
    const localScale = Math.max(.001, Math.hypot(next.x-screen.x, next.y-screen.y)*z.s);
    root.append(node('circle', {cx:point.x,cy:point.y,r:5/localScale,class:'handle',
      style:`vector-effect:none;stroke-width:${1.5/localScale}`,'data-handle':name,'data-object':obj.id}));
  }
  if(obj && tool==='select'){
    const b=$('stage').querySelector(`g[data-object="${obj.id}"]`).getBBox(),c={x:b.x+b.width/2,y:b.y+b.height/2};
    const p=photoPoint({x:c.x,y:b.y}),q=photoPoint({x:c.x,y:b.y+1}),sc=Math.max(.001,Math.hypot(q.x-p.x,q.y-p.y)*z.s);
    const y=b.y-26/sc;
    root.append(node('line',{x1:c.x,y1:b.y,x2:c.x,y2:y,class:'highlight'}));
    root.append(node('circle',{cx:c.x,cy:y,r:6/sc,class:'handle','data-handle':'rotate','data-object':obj.id,style:`vector-effect:none;stroke-width:${1.5/sc};fill:#86d7ff`}));
  }
}
function render() {
  if (!state) { refresh(); return; }
  $('style').value = state.style; $('photoVisible').checked = state.photoVisible; $('opacity').value = state.opacity;
  const [, , vw] = result.editor.viewBox.split(/\s+/).map(Number);
  // Rasterize SVG at the current screen density, not at photo resolution and then
  // magnify its CSS-3D texture. Keep original geometry for editing and export.
  const stage = node('svg', {xmlns: NS, style: 'color:#f0efec;position:absolute;left:0;top:0;transform-origin:0 0;overflow:hidden'});
  const photo = result.editor.photo;
  const image = document.createElement('img');
  if(result.photo) image.src = result.photo;
  image.width = photo.width; image.height = photo.height; image.draggable = false;
  image.style.cssText = `display:block;pointer-events:none;opacity:${state.photoVisible ? state.opacity : 0}`;
  stage.append(drawing());
  $('stage').replaceChildren(...(result.photo?[image,stage]:[stage]));
  interactions(stage, overlayViewport().scale, true);
  const preview = node('svg', {xmlns: NS, viewBox: result.editor.viewBox, style: 'color:#f0efec'});
  preview.append(drawing());
  const previewScale = Math.max(0.01, ($('svgbox').clientWidth - 16) / vw);
  interactions(preview, previewScale);
  $('svgbox').replaceChildren(preview);
  apply(); refresh();
}
function overlayViewport() {
  const view=$('view'),pad=48;
  const corners=[[-pad,-pad],[view.clientWidth+pad,-pad],[-pad,view.clientHeight+pad],[view.clientWidth+pad,view.clientHeight+pad]]
    .map(([x,y])=>drawingFromPhoto({x:(x-z.x)/z.s,y:(y-z.y)/z.s}));
  // Clip the native SVG viewport to the visible portion of the drawing. A huge
  // full-document 3D texture would otherwise hit Chromium's texture-size limit.
  // Do not clamp to the export viewBox: manual objects/handles may lie outside it.
  const vx=Math.min(...corners.map(p=>p.x)),vy=Math.min(...corners.map(p=>p.y));
  const vw=Math.max(.01,Math.max(...corners.map(p=>p.x))-vx),vh=Math.max(.01,Math.max(...corners.map(p=>p.y))-vy);
  const c={x:vx+vw/2,y:vy+vh/2},p=photoPoint(c),px=photoPoint({x:c.x+.01,y:c.y}),py=photoPoint({x:c.x,y:c.y+.01});
  const scale=Math.max(.001,z.s*Math.max(Math.hypot(px.x-p.x,px.y-p.y),Math.hypot(py.x-p.x,py.y-p.y))/.01);
  return {x:vx,y:vy,width:vw,height:vh,scale};
}
function apply() {
  $('stage').style.transform = `translate(${z.x}px, ${z.y}px)`;
  if(result && state){
    const svg=$('stage').querySelector('svg'),image=$('stage').querySelector('img');
    if(image){image.style.width=`${result.editor.photo.width*z.s}px`;image.style.height=`${result.editor.photo.height*z.s}px`;}
    if(svg){
      const {x,y,width,height,scale:r}=overlayViewport();
      svg.setAttribute('viewBox',`${x} ${y} ${width} ${height}`);svg.setAttribute('width',width*r);svg.setAttribute('height',height*r);
      const [a,b,c,d,e,f,g,h,i]=result.editor.photo.projection;
      const tx=a*x+b*y+c,ty=d*x+e*y+f,den=g*x+h*y+i;
      // Ordinary photos need no composited 3D layer at all.
      svg.style.transform=g===0 && h===0?`matrix(${z.s*a/r/i},${z.s*d/r/i},${z.s*b/r/i},${z.s*e/r/i},${z.s*tx/i},${z.s*ty/i})`:
        `matrix3d(${[z.s*a/r,z.s*d/r,0,g/r,z.s*b/r,z.s*e/r,0,h/r,0,0,1,0,z.s*tx,z.s*ty,0,den].join(',')})`;
    }
  }
  $('zoomValue').textContent=`${Math.round(z.s*100)}%`;updateDirty();
}
function zoomFit() {
  if (!result) return;
  const {width: w, height: h} = result.editor.photo, v = $('view').getBoundingClientRect();
  z.s = Math.min(v.width / w, v.height / h); z.x = (v.width - w * z.s) / 2; z.y = (v.height - h * z.s) / 2;
  render();
}
function zoomAt(x, y, factor) {
  const s = Math.min(20, Math.max(.05, z.s * factor));
  z.x = x - (x - z.x) * s / z.s; z.y = y - (y - z.y) * s / z.s; z.s = s; render();
}
function zoomCenter(factor) { const v = $('view').getBoundingClientRect(); zoomAt(v.width / 2, v.height / 2, factor); }
// All manual geometry lives in the same SVG coordinates as automatic objects.
// Invert the photo homography, not a bounding-box approximation (perspective is supported).
function photoPoint(p) {
  const [a,b,c,d,e,f,g,h,i] = result.editor.photo.projection, w = g*p.x+h*p.y+i;
  return {x:(a*p.x+b*p.y+c)/w,y:(d*p.x+e*p.y+f)/w};
}
function drawingPoint(e) {
  const r = $('view').getBoundingClientRect(), u=(e.clientX-r.left-z.x)/z.s, v=(e.clientY-r.top-z.y)/z.s;
  return drawingFromPhoto({x:u,y:v});
}
function drawingFromPhoto({x:u,y:v}) {
  const [a,b,c,d,f,g,h,i,j] = result.editor.photo.projection;
  const A=a-u*h, B=b-u*i, C=u*j-c, D=d-v*h, E=f-v*i, F=v*j-g, det=A*E-B*D;
  return {x:(C*E-B*F)/det,y:(A*F-C*D)/det};
}
function manualHandles(obj) {
  const p=obj.primitives[0], g=p.geometry;
  if (p.type === 'circle') {
    const handles={radius:{x:g.cx+g.r,y:g.cy}};
    if (obj.primitives.length===2) handles.inner={x:g.cx,y:g.cy-obj.primitives[1].geometry.r};
    return handles;
  }
  if (p.type === 'line') return {start:{x:g.x1,y:g.y1},end:{x:g.x2,y:g.y2}};
  return {start:{x:g.x,y:g.y},end:{x:g.x+g.width,y:g.y+g.height}};
}
function usesManualHandles(obj) { return obj.category==='manual' && !obj.primitives.some(p=>p.type==='path'); }
function objectHandles(obj) {
  if (usesManualHandles(obj)) return manualHandles(obj);
  // Measure the actual SVG paths (including Bezier extrema), not their control points.
  const el=$('stage').querySelector(`g[data-object="${obj.id}"]`);
  if (!el) return {};
  const b=el.getBBox();
  return {nw:{x:b.x,y:b.y},ne:{x:b.x+b.width,y:b.y},
    se:{x:b.x+b.width,y:b.y+b.height},sw:{x:b.x,y:b.y+b.height}};
}
// Bake one uniform SVG-space transform into the existing geometry. This keeps paths,
// compound objects, hit testing, project v1 and both exports on the same data path.
// Stroke widths remain the chosen drawing style, as with manual radius editing.
function transformObject(source, anchor, scale, dx=0, dy=0, angle=0) {
  if (scale===1 && dx===0 && dy===0 && angle===0) return source;
  const obj=structuredClone(source);
  const x=v=>anchor.x+(Number(v)-anchor.x)*scale+dx;
  const y=v=>anchor.y+(Number(v)-anchor.y)*scale+dy;
  let valid=true;
  const check=v=>{if(!Number.isFinite(v) || Math.abs(v)>1e6)valid=false;return v;};
  for (const p of obj.primitives) {
    const g=p.geometry;
    if(angle){
      const rotate=q=>({x:anchor.x+(q.x-anchor.x)*Math.cos(angle)-(q.y-anchor.y)*Math.sin(angle),y:anchor.y+(q.x-anchor.x)*Math.sin(angle)+(q.y-anchor.y)*Math.cos(angle)});
      if(p.type==='circle'){const q=rotate({x:+g.cx,y:+g.cy});g.cx=check(q.x);g.cy=check(q.y);}
      else if(p.type==='line'){for(const [kx,ky] of [['x1','y1'],['x2','y2']]){const q=rotate({x:+g[kx],y:+g[ky]});g[kx]=check(q.x);g[ky]=check(q.y);}}
      else {
        if(p.type==='rect'){const x=+g.x,y=+g.y,w=+g.width,h=+g.height;p.type='path';p.geometry={d:`M${x} ${y}L${x+w} ${y}L${x+w} ${y+h}L${x} ${y+h}Z`};}
        const cs=GBCVBend.commands(p);for(const c of cs)c.ps=c.ps.map(q=>{const r=rotate(q);return {x:check(r.x),y:check(r.y)};});
        p.geometry.d=GBCVBend.path(cs);
      }continue;
    }
    if (p.type==='path') {
      let index=0;
      g.d=g.d.replace(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g,
        v=>String(check(index++%2===0?x(v):y(v))));
    } else {
      const keys=p.type==='circle'?['cx','cy']:p.type==='line'?['x1','y1','x2','y2']:['x','y'];
      keys.forEach((key,i)=>{g[key]=check(i%2===0?x(g[key]):y(g[key]));});
      for (const key of ['r','width','height']) if (key in g) g[key]=check(Number(g[key])*scale);
    }
  }
  obj.type=obj.primitives.length===1?obj.primitives[0].type:'compound';
  return valid?obj:null;
}
function manualPrimitive(type, geometry, inner=false) {
  const styles={};
  for (const [name,widths] of Object.entries(result.editor.manualStyles)) styles[name]={
    fill:'none',stroke:'currentColor','stroke-width':widths[inner?'inner':'outer'],
    'stroke-linecap':'round','stroke-linejoin':'round'};
  return {type,geometry,styles};
}
function makeManual(kind, a, b, id) {
  let primitives;
  if (kind==='circle' || kind==='rings') {
    const g={cx:a.x,cy:a.y,r:Math.hypot(b.x-a.x,b.y-a.y)};
    primitives=[manualPrimitive('circle',g)];
    if (kind==='rings') primitives.push(manualPrimitive('circle',{...g,r:g.r*.5},true));
  } else if (kind==='line') primitives=[manualPrimitive('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y})];
  else primitives=[manualPrimitive('rect',{x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),width:Math.abs(b.x-a.x),height:Math.abs(b.y-a.y)})];
  return {id,category:'manual',type:primitives.length===2?'compound':primitives[0].type,
    order:Math.max(0,...state.objects.map(o=>o.order))+1,primitives};
}
function beginEdit(e, obj=null, handle=null) {
  if(obj && tool==='select' && !e.altKey && (state.joints||[]).some(j=>j.some(r=>r.object===obj.id))){status('This object has linked nodes. Move them with Nodes, or disconnect them before transforming the object.');return;}
  if(tool==='erase' && !globalThis.GBCVErase) {status('Eraser module failed to load. Save the project and reload the page (Ctrl+F5).',true);return;}
  edit={pointerId:e.pointerId,base:state,selection:selected,obj,handle,kind:tool,
    copying:!!obj && e.altKey,altClick:e.altKey?{clientX:e.clientX,clientY:e.clientY,target:e.target,button:0,altKey:true}:null,
    handles:obj && handle?objectHandles(obj):null,
    start:drawingPoint(e),screen:{x:e.clientX,y:e.clientY},id:obj?.id || (tool==='erase'?null:`manual-${++manualSerial}`),moved:false};
  if (tool==='erase') edit.last=photoPoint(edit.start);
  if(handle==='rotate'){const b=$('stage').querySelector(`g[data-object="${obj.id}"]`).getBBox();edit.center={x:b.x+b.width/2,y:b.y+b.height/2};edit.copying=false;}
  $('view').setPointerCapture(e.pointerId); refresh();
  if (tool==='erase') updateEdit(e);
}
function bendAt(e) {
  const ids=[...new Set([e.target,...document.elementsFromPoint(e.clientX,e.clientY)]
    .map(el=>el.closest('[data-object]')?.getAttribute('data-object')).filter(Boolean))];
  const id=ids.includes(selected)?selected:ids[0],obj=state.objects.find(o=>o.id===id);
  if(!obj)return null;
  const r=$('view').getBoundingClientRect();
  const project=p=>{const q=photoPoint(p);return {x:r.left+z.x+z.s*q.x,y:r.top+z.y+z.s*q.y};};
  let best=null;
  obj.primitives.forEach((p,primitive)=>{
    const hit=GBCVBend.nearest(p,{x:e.clientX,y:e.clientY},project);
    if(hit && hit.distance<=100 && (!best || hit.distance<best.hit.distance))best={id,primitive,hit};
  });
  return best;
}
function beginBend(e) {
  if(!globalThis.GBCVBend) {status('Bend module failed to load. Save the project and reload the page (Ctrl+F5).',true);return;}
  const target=bendAt(e);if(!target)return;
  const obj=state.objects.find(o=>o.id===target.id),p=obj.primitives[target.primitive];
  const prepared=GBCVBend.prepare(p,target.hit);
  if(!prepared){status('Endpoints are fixed. Grab the segment farther from its end.');return;}
  beginEdit(e,obj);edit.copying=false;edit.altClick=null;edit.bend={...target,prepared};
  bendHover=null;selected=obj.id;render();
}
function updateEdit(e) {
  try { updateEditGeometry(e); }
  catch(error) {
    // A missing/stale module or bad geometry must not strand the captured pointer
    // with every toolbar button disabled. Roll back just this gesture.
    finishEdit(false);
    status(`Edit cancelled: ${error.message}. The project is unchanged; tools are available again.`,true);
  }
}
function updateEditGeometry(e) {
  if (!edit || drag || edit.pointerId!==e.pointerId) return;
  const q=drawingPoint(e), a=edit.start;
  if (edit.kind==='erase') {
    const point=photoPoint(q),objects=GBCVErase.objects(state.objects,edit.last,point,eraserRadius(),result.editor.photo.projection);
    edit.last=point;
    if (objects!==state.objects) {edit.moved=true;state={...state,objects,joints:GBCVNodes.reconcile(state.objects,objects,state.joints||[])};if(!objects.some(o=>o.id===selected))selected=null;render();}
    return;
  }
  if (!edit.moved && Math.hypot(e.clientX-edit.screen.x,e.clientY-edit.screen.y)<4) return;
  edit.moved=true;
   if(edit.kind==='nodes'){
    // A generous acquisition radius must not make a node jump to an off-centre
    // click. Preserve the grab offset in screen/photo space, including perspective.
    const origin=photoPoint(edit.nodeRef.point),start=photoPoint(edit.start),cursor=photoPoint(q);
    let target=drawingFromPhoto({x:origin.x+cursor.x-start.x,y:origin.y+cursor.y-start.y});
    const screen=photoPoint(target);let distance=9;
    const excluded=new Set(((edit.base.joints||[]).find(j=>j.some(r=>GBCVNodes.key(r)===GBCVNodes.key(edit.nodeRef))) || [edit.nodeRef]).map(GBCVNodes.key));
    if(edit.nodeRef.endpoint)for(const p of GBCVNodes.points(edit.base.objects).filter(p=>!p.end.startsWith('control-') && !excluded.has(GBCVNodes.key(p)))){
      const s=photoPoint(p.point),d=Math.hypot(s.x-screen.x,s.y-screen.y)*z.s;if(d<distance){distance=d;target=p.point;}
    }
    state={...edit.base,objects:GBCVNodes.moveLinked(edit.base.objects,edit.base.joints||[],edit.nodeRef,target)};render();return;
  }
  let obj;
  if(edit.handle==='rotate'){
    const c=edit.center;let angle=Math.atan2(q.y-c.y,q.x-c.x)-Math.atan2(a.y-c.y,a.x-c.x);
    if(e.shiftKey)angle=Math.round(angle/(Math.PI/12))*Math.PI/12;
    obj=transformObject(edit.obj,c,1,0,0,angle);if(!obj)return;
  }
  else if (edit.kind==='bend') {
    const {primitive,prepared}=edit.bend;
    const p=GBCVBend.deform(edit.obj.primitives[primitive],prepared,{x:q.x-a.x,y:q.y-a.y});
    if(!p){status('Bend limit reached: this movement would cause a reversal or loop.',true);return;}
    obj={...edit.obj,primitives:edit.obj.primitives.map((old,i)=>i===primitive?p:old)};
    obj.type=obj.primitives.length===1?p.type:'compound';
  }
  else if (!edit.obj) obj=makeManual(edit.kind,a,q,edit.id);
  else if (!edit.handle) {
    obj=transformObject(edit.obj,{x:0,y:0},1,q.x-a.x,q.y-a.y);
    if (!obj) return;
  }
  else if (!usesManualHandles(edit.obj)) {
    const opposite={nw:'se',ne:'sw',se:'nw',sw:'ne'}[edit.handle];
    const anchor=edit.handles[opposite], corner=edit.handles[edit.handle];
    const vx=corner.x-anchor.x,vy=corner.y-anchor.y, length=vx*vx+vy*vy;
    if (!length) return;
    // Use the gesture delta: grabbing near a handle must not cause a size jump.
    const scale=Math.max(.01,Math.min(100,1+((q.x-a.x)*vx+(q.y-a.y)*vy)/length));
    obj=transformObject(edit.obj,anchor,scale);
    if (!obj) return;
  } else {
    obj=structuredClone(edit.obj);
    const p=obj.primitives[0], g=p.geometry, handle=edit.handle;
    if (p.type==='circle') {
      const radius=Math.max(.01,Math.hypot(q.x-g.cx,q.y-g.cy));
      if (handle==='inner') obj.primitives[1].geometry.r=Math.min(g.r*.99,radius);
      else {const ratio=obj.primitives[1]?.geometry.r/g.r;g.r=radius;if(obj.primitives[1])obj.primitives[1].geometry.r=radius*ratio;}
    } else if (p.type==='line') {g[handle==='start'?'x1':'x2']=q.x;g[handle==='start'?'y1':'y2']=q.y;}
    else {
      const anchor=manualHandles(edit.obj)[handle==='start'?'end':'start'];
      g.x=Math.min(q.x,anchor.x);g.y=Math.min(q.y,anchor.y);g.width=Math.abs(q.x-anchor.x);g.height=Math.abs(q.y-anchor.y);
    }
  }
  if (edit.copying) {
    if (!edit.copyId) edit.copyId=`manual-${++manualSerial}`;
    obj={...obj,id:edit.copyId,order:Math.max(0,...edit.base.objects.map(o=>o.order))+1};edit.id=obj.id;
  }
  state={...edit.base,objects:edit.obj && !edit.copying?edit.base.objects.map(o=>o.id===obj.id?obj:o):[...edit.base.objects,obj]};
  selected=obj.id; render();
}
function finishEdit(commit=false) {
  if (!edit) return;
  bendHover=null;
  const current=edit; edit=null;
  const object=state.objects.find(o=>o.id===current.id);
  let valid=current.kind==='erase' || object && object.primitives.every(p=>{
    const g=p.geometry;
    if (p.type==='path') return true;
    if (p.type==='rect') return g.width>0 && g.height>0;
    if (p.type==='circle') return g.r>0;
    return Math.hypot(g.x2-g.x1,g.y2-g.y1)>0;
  });
  const changed=JSON.stringify(state.objects)!==JSON.stringify(current.base.objects);
  // Keep edits inside the existing project complexity/security limits, atomically.
  if (commit && changed)try {GBCVProject.validate(projectData());}catch(error){valid=false;status(error.message,true);}
  if (commit && current.moved && valid && changed) {past.push(current.base);future=[];if(current.kind!=='erase'){selected=current.id;if(!['bend','nodes'].includes(current.kind))tool='select';}}
  else {state=current.base;selected=current.selection;}
  if ($('view').hasPointerCapture(current.pointerId)) $('view').releasePointerCapture(current.pointerId);
  render();
  if (commit && current.altClick && !current.moved) selectAt(current.altClick);
}
function beginPan(e, pointerId=e.pointerId) {
  finishEdit(false);
  drag = {pointerId,button:e.button,x:e.clientX,y:e.clientY,zx:z.x,zy:z.y};
  $('view').setPointerCapture(pointerId);$('view').classList.add('dragging');refresh();
}
// Compare closed geometry, independent of ID, path start/direction and line style.
// SAM can return the same outline with a different first vertex or tiny pixel noise.
function duplicateContour(obj) {
  const scratch=node('svg',{style:'position:absolute;left:-10000px;top:0;visibility:hidden;pointer-events:none'});
  document.body.append(scratch);
  function measure(p) {
    if(!['path','circle','rect'].includes(p.type) || (p.type==='path' && !/Z\s*$/i.test(p.geometry.d)))return null;
    const el=node(p.type,{...p.geometry,fill:'black',stroke:'black','stroke-width':1.2,'stroke-linecap':'round','stroke-linejoin':'round'});scratch.append(el);
    const box=el.getBBox(),length=el.getTotalLength();
    return {el,box,length};
  }
  const samples=m=>Array.from({length:128},(_,i)=>m.el.getPointAtLength(i*m.length/128));
  const near=(points,shape)=>points.every(p=>shape.el.isPointInStroke(new DOMPoint(p.x,p.y)));
  function sameArea(a,b) {
    // Global detection and click detection use different mask clean-up kernels.
    // A few corner/string pixels can differ even on the same hardware contour.
    const x=Math.min(a.box.x,b.box.x),y=Math.min(a.box.y,b.box.y);
    const w=Math.max(a.box.x+a.box.width,b.box.x+b.box.width)-x,h=Math.max(a.box.y+a.box.height,b.box.y+b.box.height)-y;
    let intersection=0,union=0;
    for(let i=0;i<32;i++)for(let j=0;j<32;j++) {
      const p=new DOMPoint(x+(i+.5)*w/32,y+(j+.5)*h/32);
      const insideA=a.el.isPointInFill(p),insideB=b.el.isPointInFill(p);
      if(insideA || insideB)union++;
      if(insideA && insideB)intersection++;
    }
    return union>8 && intersection/union>=.95;
  }
  try {
    const target=measure(obj.primitives[0]);if(!target)return null;
    const targetSamples=samples(target);
    for(const existing of state.objects)for(const p of existing.primitives) {
      const other=measure(p);if(!other)continue;
      if(['x','y','width','height'].some(k=>Math.abs(other.box[k]-target.box[k])>.75))continue;
      if(Math.abs(other.length-target.length)>Math.max(1,target.length*.05))continue;
      const points=samples(other);
      if((near(targetSamples,other) && near(points,target)) || sameArea(target,other))return existing.id;
    }
    return null;
  } finally {scratch.remove();}
}
async function autoSelect(e) {
  if(busy || edit || drag || spacePan || !state)return;
  if(!result?.id) {status('Auto select requires a live session with the source image, mask and SAM. Local projects do not include this data. Manual tools remain available.',true);return;}
  if(!result.photo || !state.photoVisible || state.opacity===0) {status('Show the source image to select parts from it.',true);return;}
  const rect=$('view').getBoundingClientRect(),photo=result.editor.photo;
  const x=(e.clientX-rect.left-z.x)/z.s/photo.width,y=(e.clientY-rect.top-z.y)/z.s/photo.height;
  if(x<0 || y<0 || x>=1 || y>=1) {status('Click a part inside the source image.',true);return;}
  const sid=result.id;
  if(samAllowedSid!==sid && !window.confirm('Allow the existing SAM model (CPU) for this editor session? Each Auto select click will run processing without another prompt. No models will be downloaded.'))return;
  const request={sid,controller:new AbortController()};autoRequest=request;
  busy=true;refresh();status('SAM selection in progress…');
  try {
    if(samAllowedSid!==sid) {
      const permission=await fetch(`/api/s/${encodeURIComponent(sid)}/sam-permission`,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:true}),signal:request.controller.signal});
      if(!permission.ok)throw new Error(permission.status===404?'Source session expired. Upload the image again.':'Could not enable SAM for this session. The drawing is unchanged.');
      if(autoRequest!==request)return;
      samAllowedSid=sid;
    }
    let response,data;
    try {
      response=await fetch(`/api/s/${encodeURIComponent(sid)}/auto-select`,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y}),signal:request.controller.signal});
    } catch {throw new Error('Could not reach the selection server. The drawing is unchanged; try again later or use manual tools.');}
    try {data=await response.json();} catch {throw new Error('The selection service returned an invalid response. The drawing is unchanged; use manual tools.');}
    if(autoRequest!==request || result?.id!==sid)return;
    if(response.status===428)samAllowedSid=null;
    if(response.status===404)throw new Error('Source session expired. Auto select is unavailable; the drawing is unchanged and manual tools remain available.');
    if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Could not select the part');
    const obj=data.object;
    if(!obj || obj.category!=='hardware' || obj.type!=='path' || obj.primitives?.length!==1)throw new Error('Invalid auto-selection response');
    // Validate server geometry with the same strict allowlist as project import.
    const candidate={...obj,id:`auto-${manualSerial+1}`,order:Math.max(0,...state.objects.map(o=>o.order))+1};
    const project=projectData();project.manualSerial=manualSerial+1;project.document.objects=[...state.objects,candidate];GBCVProject.validate(project);
    const duplicate=duplicateContour(candidate);
    if(duplicate) {selected=duplicate;render();status('This outline already exists. Selected the existing object; no duplicate added.');return;}
    manualSerial++;selected=candidate.id;
    change({...state,objects:[...state.objects,candidate]});render();
    status('Outline added. Click another part, or switch to Select to edit.');
  } catch(error) {if(autoRequest===request)status(error.message,true);}
  finally {if(autoRequest===request){autoRequest=null;busy=false;refresh();}}
}
function cancelAutoSelect() {
  if(!autoRequest)return;
  autoRequest.controller.abort();autoRequest=null;busy=false;refresh();
  status('Waiting cancelled; the drawing is unchanged. An active SAM task will finish on the server; the model may remain busy until then.');
}
$('cancelAuto').onclick=cancelAutoSelect;
function selectAt(e) {
  if (!state || busy || edit || drag || spacePan || !['select','bend','nodes'].includes(tool) || e.button !== 0) return;
  // Alt-click cycles coincident objects, so a covered contour is still reachable.
  const ids = [...new Set([e.target, ...document.elementsFromPoint(e.clientX, e.clientY)]
    .map(el => el.closest('[data-object]')?.getAttribute('data-object')).filter(Boolean))];
  selected = e.altKey && ids.length ? ids[(ids.indexOf(selected) + 1) % ids.length] : ids[0] || null;
  render();
}
$('view').addEventListener('wheel', e => {
  e.preventDefault(); if (!state || busy || drag || edit) return;
  const v = $('view').getBoundingClientRect(); zoomAt(e.clientX - v.left, e.clientY - v.top, e.deltaY < 0 ? 1.2 : 1 / 1.2);
}, {passive: false});
$('view').addEventListener('pointerdown', e => {
  if (e.button === 1) e.preventDefault();
  if (busy || !state || drag || e.target.closest('.zoom')) return;
  const pan = e.button === 1 || (e.button === 0 && (spacePan || tool === 'pan'));
  if (!pan) {
    if (edit || e.button!==0 || e.ctrlKey || e.metaKey) return;
    if(tool==='nodes'){
      if(e.altKey){selectAt(e);return;}
      const marker=nearestControl(e,'data-node');
      if(marker){const ref=JSON.parse(marker.getAttribute('data-node'));selected=ref.object;
        const has=nodeSelection.some(r=>GBCVNodes.key(r)===GBCVNodes.key(ref));
        nodeSelection=has?(e.shiftKey?nodeSelection.filter(r=>GBCVNodes.key(r)!==GBCVNodes.key(ref)):nodeSelection):[...nodeSelection.slice(-1),ref];
        beginEdit(e,state.objects.find(o=>o.id===selected));edit.copying=false;edit.altClick=null;edit.nodeRef=ref;render();
      }else selectAt(e);return;
    }
    if (tool==='auto') {autoSelect(e);return;}
    if (tool==='bend') {e.preventDefault();if(e.altKey){selectAt(e);return;}beginBend(e);return;}
    if (tool!=='select') {e.preventDefault();beginEdit(e);return;}
    const handle=e.target.closest('[data-handle]');
    if (handle) {beginEdit(e,state.objects.find(o=>o.id===handle.getAttribute('data-object')),e.altKey?null:handle.getAttribute('data-handle'));return;}
    if (e.altKey) {
      const obj=state.objects.find(o=>o.id===e.target.closest('[data-object]')?.getAttribute('data-object'));
      if (obj)beginEdit(e,obj);
      return;
    }
    selectAt(e);
    const obj=state.objects.find(o=>o.id===selected);
    if (obj && !e.altKey) beginEdit(e,obj);
    return;
  }
  e.preventDefault();
  beginPan(e);
});
$('view').addEventListener('pointermove', e => {
  eraserPointer={clientX:e.clientX,clientY:e.clientY};eraserCursor();
  if(tool==='nodes')hoverNode(e);
  // Pointer Events report a second mouse button as a move, not another down.
  if (edit && e.button===1 && (e.buttons&4)) {beginPan(e);return;}
  if (edit) {updateEdit(e);return;}
  if (tool==='bend' && globalThis.GBCVBend && !drag && !spacePan && !busy && state) {bendHover=bendAt(e);render();}
  if (!drag || e.pointerId !== drag.pointerId) return;
  if (!(e.buttons & (drag.button===1?4:1))) {endPan();return;}
  z.x = drag.zx + e.clientX - drag.x; z.y = drag.zy + e.clientY - drag.y; apply();
});
$('view').addEventListener('pointerleave',()=>{eraserPointer=null;eraserCursor();hoverNode(null);if(bendHover){bendHover=null;render();}});
function endPan() {
  const id = drag?.pointerId;
  drag = null;
  if (id !== undefined && $('view').hasPointerCapture(id)) $('view').releasePointerCapture(id);
  $('view').classList.remove('dragging'); refresh();
}
$('view').addEventListener('pointerup', e => {
  if (edit && e.button===0 && e.pointerId===edit.pointerId) {updateEdit(e);finishEdit(true);}
  if (drag && e.button === drag.button && e.pointerId === drag.pointerId) endPan();
});
for (const type of ['pointercancel','lostpointercapture']) $('view').addEventListener(type, () => {finishEdit(false);endPan();});
// Cancel the browser's middle-button autoscroll (also when the canvas is empty).
for (const type of ['mousedown', 'auxclick']) $('view').addEventListener(type, e => { if (e.button === 1) e.preventDefault(); });
$('svgbox').addEventListener('click', selectAt);
for (const id of toolIds) $(id).onclick = () => {
  if (edit) return;
  tool=id;bendHover=null;
  if(id==='nodes')status('Nodes: select two open-path endpoints, then Join points. Blue nodes are linked; Disconnect separates selected endpoints. The 9px screen snap does not create a link. Closed paths, circles and rectangles use Select.');
  if(id==='bend')status('Bend: drag a highlighted segment. Endpoints and tangents stay fixed. Alt+click cycles overlaps; Esc cancels. Circles, rectangles and closing Z segments cannot be bent.');
  if(id==='erase')status('Eraser: cuts lines and M/L/C/Z paths. Skips circles and rectangles. Size is the diameter in source-image pixels.');
  render();
};
$('eraseSize').oninput=eraserCursor;
$('zin').onclick = () => zoomCenter(1.4); $('zout').onclick = () => zoomCenter(1 / 1.4); $('zfit').onclick = zoomFit;
$('delete').onclick = removeSelected; $('undo').onclick = undo; $('redo').onclick = redo;
$('duplicate').onclick=duplicateSelected;
$('join').onclick=()=>{
  if(busy || edit || drag || nodeSelection.length!==2)return;
  const joined=GBCVNodes.join(state.objects,state.joints||[],...nodeSelection);
  if(!joined)return;
  const candidate={...state,...joined},project=projectData();project.document.objects=candidate.objects;project.document.joints=candidate.joints;
  try{GBCVProject.validate(project);change(candidate);}catch(e){status(e.message,true);}
};
$('disconnect').onclick=()=>{
  if(busy || edit || drag)return;
  const keys=new Set(nodeSelection.map(GBCVNodes.key));
  const joints=(state.joints||[]).map(j=>j.filter(r=>!keys.has(GBCVNodes.key(r)))).filter(j=>j.length>1);
  if(JSON.stringify(joints)!==JSON.stringify(state.joints||[]))change({...state,joints});
};
function finishWidth(commit=true){
  if(!widthStart)return;
  const base=widthStart;widthStart=null;
  if(!commit)state=base;
  else if(JSON.stringify(base.objects)!==JSON.stringify(state.objects)){past.push(base);future=[];}
  render();
}
$('strokeWidth').oninput=()=>{
  if(!selected || busy || edit || drag)return;
  const text=$('strokeWidth').value,width=Number(text);
  if(text!=='' && (!Number.isFinite(width) || width<.001 || width>100))return;
  if(!widthStart)widthStart=state;
  state={...state,objects:state.objects.map(o=>{if(o.id!==selected)return o;const next={...o};if(text==='')delete next.strokeWidth;else next.strokeWidth=width;return next;})};render();
};
$('strokeWidth').onchange=()=>finishWidth();
$('strokeWidth').onblur=()=>finishWidth();
$('strokeWidth').onkeydown=e=>{if(e.key==='Escape'){e.preventDefault();finishWidth(false);$('strokeWidth').blur();}if(e.key==='Enter'){$('strokeWidth').blur();}};
$('resetWidth').onclick=()=>{if(!selected || busy || edit || drag)return;const obj=state.objects.find(o=>o.id===selected);if(obj.strokeWidth===undefined)return;change({...state,objects:state.objects.map(o=>{if(o.id!==selected)return o;const n={...o};delete n.strokeWidth;return n;})});};
$('photoVisible').onchange = () => change({...state, photoVisible: $('photoVisible').checked});
$('style').onchange = () => change({...state, style: $('style').value});
// Range input previews continuously, but its final change is one undo step.
let opacityStart = null;
$('opacity').oninput = () => {
  if (opacityStart === null) opacityStart = state;
  state = {...state, opacity: Number($('opacity').value)}; render();
};
$('opacity').onchange = () => {
  if (opacityStart && opacityStart.opacity !== state.opacity) { past.push(opacityStart); future = []; }
  opacityStart = null; refresh();
};
document.addEventListener('keydown', e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable) return;
  if (e.key==='Escape' && autoRequest) {e.preventDefault();cancelAutoSelect();return;}
  if (e.key==='Escape' && edit) {e.preventDefault();finishEdit(false);return;}
  if (e.key==='Escape' && drag) {e.preventDefault();z.x=drag.zx;z.y=drag.zy;endPan();apply();return;}
  if (e.code === 'Space' && !e.ctrlKey && !e.metaKey && !e.altKey) { e.preventDefault(); finishEdit(false);spacePan = true; refresh(); return; }
  if (edit) {e.preventDefault();return;}
  if (drag || spacePan) { if (['Delete', 'Backspace', 'Escape', 'z', 'Z', 'y', 'Y', 'd', 'D'].includes(e.key)) e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicateSelected(); }
  else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); removeSelected(); }
   else if (e.key === 'Escape') { selected = null; nodeSelection=[];render(); }
});
document.addEventListener('keyup', e => { if (e.code === 'Space') { spacePan = false; refresh(); } });
window.addEventListener('blur', () => { spacePan = false; finishEdit(false);endPan(); });
function exportSvg() {
  const svg = node('svg', {xmlns: NS, viewBox: result.editor.viewBox, style: 'color:#1b1b1b;background:#fff'});
  svg.append(drawing()); return svg;
}
function save(blob, extension) {
  const a = document.createElement('a'), url = URL.createObjectURL(blob);
  a.href = url; a.download = documentName + extension; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('download').onclick = () => save(new Blob([new XMLSerializer().serializeToString(exportSvg())], {type: 'image/svg+xml'}), '.svg');
$('downloadPng').onclick = () => {
  const svg = exportSvg(), [, , vw, vh] = result.editor.viewBox.split(/\s+/).map(Number);
  const w = Math.round(1500 * Math.min(1, vw / vh)), h = Math.round(1500 * Math.min(1, vh / vw));
  svg.setAttribute('width', w); svg.setAttribute('height', h);
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type: 'image/svg+xml'}));
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h); ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url); canvas.toBlob(blob => { if (blob) save(blob, '.png'); else status('Could not create PNG', true); });
  };
  img.onerror = () => { URL.revokeObjectURL(url); status('Could not create PNG', true); };
  img.src = url;
};
async function upload(f) {
  if (!f || busy || edit || drag || !mayReplace()) return;
  busy = true; refresh(); status('Processing… typically 5–20 seconds');
  const body = new FormData(); body.append('file', f);
  try {
    const response = await fetch('/api/trace', {method: 'POST', body}), data = await response.json();
    if (!response.ok) throw new Error(data.detail || response.statusText);
    loadResult(data, f);
  } catch (e) { status('Error: ' + e.message, true); }
  finally { busy = false; refresh(); }
}
function loadResult(data, source = null) {
  cancelAutoSelect();samAllowedSid=data.sam_enabled?data.id:null;
  bendHover=null;
  nodeSelection=[];widthStart=null;
  finishEdit(false);endPan(); spacePan = false;tool='select';
  if (result && result.id !== data.id) closeSession();
  result = data; file = source; selected = null; past = []; future = []; opacityStart = null;
  documentName=source?.name.replace(/\.[^.]+$/,'') || 'guitar';savedFingerprint=null;
  state = {objects: data.editor.objects,joints:[], style: 'comic', photoVisible: true, opacity: 1};
  $('drop').hidden = true; $('work').hidden = false; $('tools').hidden = false;
  zoomFit(); status(`Ready in ${data.stats.seconds} s · click an outline to select`);
}
function saveProject() {
  if(!state || busy || edit || drag)return;
  try {
    const text=GBCVProject.stringify(projectData());
    save(new Blob([text],{type:'application/json'}),'.gbcv.json');
    savedFingerprint=fingerprint();refresh();
    status('Project saved without the source image.');
  } catch(e) {status(e.message,true);}
}
async function openProjectFile(f) {
  if(!f || busy || edit || drag)return false;
  busy=true;refresh();status('Opening project…');
  try {
    if(f.size>GBCVProject.MAX_BYTES)throw new Error('Project exceeds 10 MB');
    const project=GBCVProject.parse(await f.text());
    if(!mayReplace())return false;
    // Validation is complete before replacing anything. No server/session/ML is needed.
    closeSession();
    samAllowedSid=null;
    result={id:null,photo:null,editor:project.document};file=null;
    bendHover=null;
    documentName=f.name.replace(/(?:\.gbcv)?\.json$/i,'') || 'guitar';
    nodeSelection=[];widthStart=null;
    state={objects:project.document.objects,joints:project.document.joints||[],...project.display};Object.assign(z,project.viewport);
    manualSerial=project.manualSerial;selected=null;tool='select';spacePan=false;past=[];future=[];opacityStart=null;
    savedFingerprint=fingerprint();
    $('drop').hidden=true;$('work').hidden=false;$('tools').hidden=false;render();
    status('Project opened without the source image or ML. All manual editing tools are available.');
    return true;
  } catch(e) {status(e.message,true);return false;}
  finally {busy=false;refresh();}
}
$('saveProject').onclick=saveProject;
$('openProject').onclick=()=>$('projectFile').click();
$('projectFile').onchange=async e=>{await openProjectFile(e.target.files[0]);e.target.value='';};
$('open').onclick = $('drop').onclick = () => { if (!busy) $('file').click(); };
$('file').onchange = e => { upload(e.target.files[0]); e.target.value = ''; };
document.addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('over'); });
document.addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('over'); });
document.addEventListener('drop', e => {
  e.preventDefault();document.body.classList.remove('over');const f=e.dataTransfer.files[0];
  if(f && /\.json$/i.test(f.name))openProjectFile(f);else upload(f);
});
// Do not overwrite a restored viewport on resize.
window.addEventListener('resize', () => { if (result) render(); });
window.addEventListener('beforeunload', e=>{if(isDirty() || edit?.moved){e.preventDefault();e.returnValue='';}});
window.addEventListener('pagehide', closeSession);
refresh();
const sessionId = new URLSearchParams(location.search).get('session');
if (sessionId) {
  busy = true; refresh(); status('Loading session…');
  fetch(`/api/s/${encodeURIComponent(sessionId)}`).then(async response => {
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || response.statusText);
    loadResult(data);
  }).catch(e => status('Error: ' + e.message, true)).finally(() => { busy = false; refresh(); });
}
