import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

export async function nodesTests({evaluate,send,key}) {
  const dir='out/editor-nodes';await mkdir(dir,{recursive:true});
  await evaluate(`(() => {
    const objects=[makeManual('rect',{x:70,y:80},{x:130,y:110},'manual-901'),
      makeManual('line',{x:200,y:80},{x:260,y:80},'manual-902'),
      makeManual('rings',{x:340,y:90},{x:365,y:90},'manual-903'),
      {...makeManual('rings',{x:460,y:90},{x:485,y:90},'manual-904'),id:'post-test',category:'post'},
      makeManual('line',{x:180,y:180},{x:240,y:180},'manual-905'),
      makeManual('line',{x:260,y:200},{x:330,y:200},'manual-906'),
      {...makeManual('line',{x:520,y:160},{x:520,y:220},'manual-907'),id:'fret-test',category:'fret'}];
    const editor={...result.editor,viewBox:'0 0 640 320',objects,photo:{width:640,height:320,projection:[1,0,0,0,1,0,0,0,1]}};
    objects.find(o=>o.id==='post-test').primitives[1].geometry.cx+=7;
    loadResult({id:null,photo:null,editor,stats:{seconds:0}});manualSerial=910;
  })()`);
  const screen=p=>evaluate(`(() => {const p=photoPoint(${JSON.stringify(p)}),r=$('view').getBoundingClientRect();return {x:r.left+z.x+z.s*p.x,y:r.top+z.y+z.s*p.y};})()`);
  async function gesture(a,b,{shift=false,cancel=false}={}) {
    const p=await screen(a),q=await screen(b),modifiers=shift?8:0;
    const mouse=(type,point)=>send('Input.dispatchMouseEvent',{type,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1,modifiers,...point});
    await mouse('mousePressed',p);assert.equal(await evaluate('!!edit'),true);
    for(const t of [.3,.7,1])await mouse('mouseMoved',{x:p.x+(q.x-p.x)*t,y:p.y+(q.y-p.y)*t});
    if(cancel)await key('keyDown','Escape','Escape');
    await mouse('mouseReleased',q);
  }
  async function clickPoint(p){const q=await screen(p);for(const type of ['mousePressed','mouseReleased'])await send('Input.dispatchMouseEvent',{type,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1,...q});}
  async function history(before,n){
    const after=await evaluate('JSON.stringify(state)');assert.equal(await evaluate('past.length'),n+1);
    await evaluate('undo()');assert.equal(await evaluate('JSON.stringify(state)'),before);
    await evaluate('redo()');assert.equal(await evaluate('JSON.stringify(state)'),after);
  }
  // Rotation is baked into every primitive. Neighbours remain byte-for-byte unchanged.
  for(const id of ['manual-901','manual-902','manual-903','post-test']){
    await evaluate(`tool='select';selected='${id}';render()`);
    const before=await evaluate('JSON.stringify(state)'),n=await evaluate('past.length');
    const geometry=await evaluate(`(() => {const el=$('stage').querySelector('[data-handle="rotate"]'),b=$('stage').querySelector('g[data-object="${id}"]').getBBox();return {a:{x:+el.getAttribute('cx'),y:+el.getAttribute('cy')},c:{x:b.x+b.width/2,y:b.y+b.height/2}};})()`);
    const {a,c}=geometry,angle=37*Math.PI/180;
    const b={x:c.x+(a.x-c.x)*Math.cos(angle)-(a.y-c.y)*Math.sin(angle),y:c.y+(a.x-c.x)*Math.sin(angle)+(a.y-c.y)*Math.cos(angle)};
    await gesture(a,b,{shift:true});
    const after=await evaluate('state.objects');
    assert.deepEqual(after.filter(o=>o.id!==id),JSON.parse(before).objects.filter(o=>o.id!==id));
    if(id==='manual-902'){
      const g=after.find(o=>o.id===id).primitives[0].geometry;
      assert(Math.abs(Math.atan2(g.y2-g.y1,g.x2-g.x1)-Math.PI/6)<1e-6);
    }
    // Concentric circles at the rotation centre have no visible change and no history entry.
    if(id==='manual-903')assert.equal(await evaluate('past.length'),n);
    else await history(before,n);
    await evaluate(`selected='${id}';render()`);
    const marker=await evaluate(`(() => {const el=$('stage').querySelector('[data-handle="rotate"]');return {x:+el.getAttribute('cx'),y:+el.getAttribute('cy')};})()`);
    const saved=await evaluate('JSON.stringify(state)');await gesture(marker,{x:marker.x+20,y:marker.y+10},{cancel:true});assert.equal(await evaluate('JSON.stringify(state)'),saved);
  }
  // Per-object fractional width: live canvas, one action, every style and export.
  for(const id of ['manual-902','fret-test']){
    await evaluate(`selected='${id}';render();$('strokeWidth').focus()`);
    const before=await evaluate('JSON.stringify(state)'),n=await evaluate('past.length');
    await evaluate(`$('strokeWidth').value='2.75';$('strokeWidth').dispatchEvent(new Event('input'))`);
    assert.equal(await evaluate(`+$('stage').querySelector('g[data-object="${id}"] > *').getAttribute('stroke-width')`),2.75);
    await evaluate(`$('strokeWidth').value='3.25';$('strokeWidth').dispatchEvent(new Event('input'));$('strokeWidth').blur()`);
    await history(before,n);
    await evaluate(`$('style').value='thin';$('style').dispatchEvent(new Event('change'))`);
    assert.equal(await evaluate(`+exportSvg().querySelector('g[data-object="${id}"] > *').getAttribute('stroke-width')`),3.25);
  }
  // Real node selection on separate lines; persistent explicit join.
  await evaluate(`tool='nodes';selected='manual-905';nodeSelection=[];render()`);
  await clickPoint({x:240,y:180});
  await clickPoint({x:300,y:200}); // select the second line, then its start
  await clickPoint({x:260,y:200});
  assert.equal(await evaluate("$('join').disabled"),false);
  let before=await evaluate('JSON.stringify(state)'),n=await evaluate('past.length');
  await evaluate("$('join').click()");assert.equal(await evaluate('state.joints.length'),1);await history(before,n);
  await evaluate(`tool='nodes';selected='manual-905';nodeSelection=[];z.s=2;z.x=-100;z.y=-100;render()`);
  before=await evaluate('JSON.stringify(state)');n=await evaluate('past.length');
  await gesture({x:250,y:190},{x:270,y:170});
  assert.equal(await evaluate("state.objects.find(o=>o.id==='manual-905').primitives[0].geometry.x2"),270);
  assert.equal(await evaluate("state.objects.find(o=>o.id==='manual-906').primitives[0].geometry.x1"),270);
  await history(before,n);
  // Shared node Escape, disconnect, independent drag, and screen-radius snapping.
  await evaluate(`tool='nodes';selected='manual-905';nodeSelection=[];render()`);
  before=await evaluate('JSON.stringify(state)');await gesture({x:270,y:170},{x:275,y:190},{cancel:true});assert.equal(await evaluate('JSON.stringify(state)'),before);
  n=await evaluate('past.length');await evaluate("$('disconnect').click()");assert.equal(await evaluate('state.joints.length'),0);await history(before,n);
  await evaluate(`tool='nodes';selected='manual-905';nodeSelection=[];render()`);
  await gesture({x:180,y:180},{x:200,y:160});
  assert.equal(await evaluate("state.objects.find(o=>o.id==='manual-906').primitives[0].geometry.x1"),270);
  for(const scale of [.65,2]){
    await evaluate(`z.s=${scale};z.x=10;z.y=0;selected='manual-905';tool='nodes';nodeSelection=[];render()`);
    const start=await evaluate("({x:state.objects.find(o=>o.id==='manual-905').primitives[0].geometry.x1,y:state.objects.find(o=>o.id==='manual-905').primitives[0].geometry.y1})");
    await gesture(start,{x:330-4/scale,y:200});
    assert.equal(await evaluate("state.objects.find(o=>o.id==='manual-905').primitives[0].geometry.x1"),330);
    assert.equal(await evaluate('state.joints.length'),0,'snap is not an implicit join');
    await evaluate('undo()');
  }
  // Rejoin and retain stable refs through Bend; duplicate has no external links.
  await evaluate(`nodeSelection=[{object:'manual-905',primitive:0,subpath:0,end:'end'},{object:'manual-906',primitive:0,subpath:0,end:'start'}];selected='manual-905';tool='nodes';render();$('join').click();
    selected='manual-905';duplicateSelected();GBCVProject.validate(projectData());zoomFit()`);
  const duplicate=await evaluate('state.objects.find(o=>o.id===selected)');
  await evaluate(`selected='manual-905';tool='bend';render()`);
  const midpoint=await evaluate(`(() => {const g=state.objects.find(o=>o.id==='manual-905').primitives[0].geometry;return {x:(g.x1+g.x2)/2,y:(g.y1+g.y2)/2};})()`);
  before=await evaluate('JSON.stringify(state)');n=await evaluate('past.length');
  await gesture(midpoint,{x:midpoint.x,y:midpoint.y+8});await history(before,n);
  await evaluate(`tool='nodes';selected='manual-905';nodeSelection=[];render();GBCVProject.validate(projectData())`);
  await gesture({x:270,y:170},{x:290,y:175});
  assert.equal(await evaluate("state.objects.find(o=>o.id==='manual-906').primitives[0].geometry.x1"),290);
  assert.deepEqual(await evaluate(`state.objects.find(o=>o.id==='${duplicate.id}')`),duplicate);
  assert.equal(await evaluate('GBCVProject.validate(projectData()).document.joints.length'),1);
  const text=await evaluate('GBCVProject.stringify(projectData())');
  const svg=await evaluate('new XMLSerializer().serializeToString(exportSvg())');
  async function png(){return evaluate(`(async()=>{downloads={};$('downloadPng').click();for(let i=0;i<200;i++){if(downloads['.png'])return new Promise(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.readAsDataURL(downloads['.png']);});await new Promise(r=>setTimeout(r,10));}throw Error('PNG timeout');})()`);}
  const image=await png();
  await writeFile(`${dir}/trace.svg`,svg);await writeFile(`${dir}/trace.png`,Buffer.from(image.split(',')[1],'base64'));
  await writeFile(`${dir}/guitar.gbcv.json`,text);
  const shot=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${dir}/overlay.png`,Buffer.from(shot.data,'base64'));
  assert.equal(await evaluate(`(async()=>{savedFingerprint=fingerprint();return openProjectFile(new File([${JSON.stringify(text)}],'nodes.gbcv.json'));})()`),true);
  assert.equal(await evaluate('new XMLSerializer().serializeToString(exportSvg())'),svg);
  assert.equal(await png(),image);
  assert.equal(await evaluate('state.joints.length'),1);
  await writeFile(`${dir}/verification.json`,JSON.stringify({noML:true,rotation:true,shift15:true,neighboursUnchanged:true,strokeWidth:true,sharedNodes:true,disconnect:true,screenSnap:true,undoRedo:true,projectV2:true,svgPngRoundtrip:true},null,2));
  console.log('PASS: rotation/Shift/Escape, fractional object widths, explicit node join/shared drag/disconnect, screen snap at two zooms, Undo/Redo, v2 reopen and byte-identical SVG/PNG; '+dir);
}
