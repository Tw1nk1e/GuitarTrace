// Eraser and duplication integration tests, called after the existing regressions.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

export async function finishTests({evaluate,send,key,panGesture,realExample}) {
  const artifacts=realExample?'out/editor-finish-real':'out/editor-finish';
  await mkdir(artifacts,{recursive:true});
  const screen=p=>evaluate(`(() => {const p=photoPoint(${JSON.stringify(p)}),r=$('view').getBoundingClientRect();return {x:r.left+z.x+z.s*p.x,y:r.top+z.y+z.s*p.y};})()`);
  const mouse=(type,p,modifiers=0)=>send('Input.dispatchMouseEvent',{type,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1,modifiers,...p});
  async function gesture(a,b,{alt=false,ending=null}={}) {
    const p=await screen(a),q=await screen(b),modifiers=alt?1:0;
    await mouse('mouseMoved',p,modifiers);await mouse('mousePressed',p,modifiers);
    assert.equal(await evaluate('!!edit'),true);
    for(const t of [.3,.7,1])await mouse('mouseMoved',{x:p.x+(q.x-p.x)*t,y:p.y+(q.y-p.y)*t},modifiers);
    if(ending==='Escape')await key('keyDown','Escape','Escape');
    else if(ending==='space')await key('keyDown',' ','Space');
    else if(ending==='middle') {
      await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'middle',buttons:5,clickCount:1,...q});
      await send('Input.dispatchMouseEvent',{type:'mouseMoved',button:'middle',buttons:5,x:q.x+5,y:q.y+5});
      await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'middle',buttons:1,x:q.x+5,y:q.y+5});
    } else if(ending)await evaluate(ending==='blur'?"window.dispatchEvent(new Event('blur'))":`$('view').dispatchEvent(new PointerEvent('${ending}',{pointerId:edit.pointerId}))`);
    await mouse('mouseReleased',q,modifiers);
    if(ending==='space')await key('keyUp',' ','Space');
  }
  async function focus(id,pixels=10) {
    await evaluate(`selected='${id}';tool='select';render();(() => {
      const el=$('stage').querySelector('g[data-object="${id}"]'),b=el.getBBox(),c={x:b.x+b.width/2,y:b.y+b.height/2};
      const p=photoPoint(c),q=photoPoint({x:c.x+1,y:c.y}),r=$('view').getBoundingClientRect();
      z.s=${pixels}/Math.hypot(p.x-q.x,p.y-q.y);z.x=r.width*.5-z.s*p.x;z.y=r.height*.5-z.s*p.y;render();})()`);
  }
  async function history(before,after,n) {
    assert.equal(await evaluate('past.length'),n+1,'single gesture/action');
    await key('keyDown','z','KeyZ',2);assert.equal(await evaluate('JSON.stringify(state.objects)'),before);
    await key('keyDown','z','KeyZ',10);assert.equal(await evaluate('JSON.stringify(state.objects)'),after);
  }
  // Five independent copies of a real/synthetic concentric post, with exact styles/radii.
  await focus('post-0',5);
  const original=await evaluate("state.objects.find(o=>o.id==='post-0')");
  assert.equal(original.primitives.length,2);
  const copyIds=[];
  for(let i=0;i<5;i++) {
    await evaluate("selected='post-0';render();document.activeElement.blur()");
    const before=await evaluate('JSON.stringify(state.objects)'),n=await evaluate('past.length');
    await key('keyDown','d','KeyD',2);
    const id=await evaluate('selected');copyIds.push(id);
    const after=await evaluate('JSON.stringify(state.objects)');await history(before,after,n);
    const copy=await evaluate(`state.objects.find(o=>o.id==='${id}')`);
    assert.notEqual(copy.id,original.id);assert.equal(copy.type,'compound');
    copy.primitives.forEach((p,k)=>{
      assert.equal(p.geometry.r,Number(original.primitives[k].geometry.r));
      assert.equal(p.geometry.cx,Number(original.primitives[k].geometry.cx)+5);
      assert.equal(p.geometry.cy,Number(original.primitives[k].geometry.cy)+5);
      assert.deepEqual(p.styles,original.primitives[k].styles);
    });
    assert.equal(copy.primitives[0].geometry.cx,copy.primitives[1].geometry.cx);
    assert.equal(copy.primitives[0].geometry.cy,copy.primitives[1].geometry.cy);
    // Select a single copy through a test-only hit target filter, as other tests overlap rings.
    await evaluate(`window.finishHit=document.createElement('style');finishHit.textContent='#stage svg > g:first-child {pointer-events:none} #stage .hit:not([data-object="${id}"]) {pointer-events:none!important}';document.head.append(finishHit);selected='${id}';render()`);
    const c={x:copy.primitives[0].geometry.cx,y:copy.primitives[0].geometry.cy-copy.primitives[0].geometry.r};
    await gesture(c,{x:c.x+8+i*9,y:c.y+8+(i%2)*12});
    await evaluate('finishHit.remove()');
    assert.deepEqual(await evaluate("state.objects.find(o=>o.id==='post-0')"),original);
  }
  assert.equal(new Set(copyIds).size,5);
  // Alt-click still cycles; Alt-drag makes a fresh copy without touching the source.
  const sourceId=copyIds[0];await focus(sourceId,8);
  await evaluate(`selected='${sourceId}';render();window.finishHit=document.createElement('style');finishHit.textContent='#stage svg > g:first-child {pointer-events:none} #stage .hit:not([data-object="${sourceId}"]) {pointer-events:none!important}';document.head.append(finishHit)`);
  const source=await evaluate(`state.objects.find(o=>o.id==='${sourceId}')`),g=source.primitives[0].geometry;
  const sourcePoint={x:g.cx,y:g.cy-g.r};
  const beforeAlt=await evaluate('JSON.stringify(state.objects)'),nAlt=await evaluate('past.length');
  await gesture(sourcePoint,{x:sourcePoint.x+10,y:sourcePoint.y+6},{alt:true});
  const altId=await evaluate('selected');assert(!copyIds.includes(altId));assert.notEqual(altId,sourceId);
  const afterAlt=await evaluate('JSON.stringify(state.objects)');await history(beforeAlt,afterAlt,nAlt);
  assert.deepEqual(await evaluate(`state.objects.find(o=>o.id==='${sourceId}')`),source);
  const altCopy=await evaluate(`state.objects.find(o=>o.id==='${altId}')`);
  altCopy.primitives.forEach((p,i)=>{assert(Math.abs(p.geometry.cx-source.primitives[i].geometry.cx-10)<.02);assert(Math.abs(p.geometry.cy-source.primitives[i].geometry.cy-6)<.02);assert.equal(p.geometry.r,source.primitives[i].geometry.r);assert.deepEqual(p.styles,source.primitives[i].styles);});
  for(const ending of ['Escape','pointercancel','lostpointercapture','blur','space','middle']) {
    await evaluate(`selected='${sourceId}';render()`);
    const before=await evaluate('JSON.stringify(state.objects)'),n=await evaluate('past.length');
    await gesture(sourcePoint,{x:sourcePoint.x+3,y:sourcePoint.y+4},{alt:true,ending});
    assert.equal(await evaluate('JSON.stringify(state.objects)'),before,`copy cancel ${ending}`);
    assert.equal(await evaluate('past.length'),n);
    await focus(sourceId,8);
  }
  await evaluate('finishHit.remove()');
  // Existing long fret: carve a short gap, keeping endpoints and all unrelated geometry.
  await focus('fret-1',15);
  const midpoint=await evaluate(`(() => {const el=$('stage').querySelector('.hit[data-object="fret-1"]'),p=el.getPointAtLength(el.getTotalLength()*.5);return {x:p.x,y:p.y};})()`);
  await evaluate("$('eraseSize').value=4;$('erase').click()");
  const beforeFret=await evaluate('JSON.stringify(state.objects)'),nFret=await evaluate('past.length');
  await gesture({x:midpoint.x-1,y:midpoint.y},{x:midpoint.x+1,y:midpoint.y});
  assert.equal(await evaluate('tool'),'erase');
  const afterFret=await evaluate('JSON.stringify(state.objects)');await history(beforeFret,afterFret,nFret);
  const cut=await evaluate("state.objects.find(o=>o.id==='fret-1')");assert(cut.primitives[0].geometry.d.match(/M/g).length>=2);
  const untouched=JSON.parse(beforeFret).filter(o=>o.id!=='fret-1');
  assert.deepEqual(JSON.parse(afterFret).filter(o=>o.id!=='fret-1'),untouched);
  const fretShot=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${artifacts}/fret-detail.png`,Buffer.from(fretShot.data,'base64'));
  // Silhouette: a small local cut changes its closed cubic outline into an open path.
  await focus('silhouette-0',1);
  // Reframe a single point so even a long contour's cut is on screen.
  const bodyPoint=await evaluate(`(() => {const el=$('stage').querySelector('.hit[data-object="silhouette-0"]'),p=el.getPointAtLength(el.getTotalLength()*.03);return {x:p.x,y:p.y};})()`);
  await evaluate(`(() => {const p=photoPoint(${JSON.stringify(bodyPoint)}),r=$('view').getBoundingClientRect();z.s=2;z.x=r.width/2-z.s*p.x;z.y=r.height/2-z.s*p.y;render();})();$('eraseSize').value=3;$('erase').click()`);
  const beforeBody=await evaluate('JSON.stringify(state.objects)'),nBody=await evaluate('past.length');
  await gesture(bodyPoint,bodyPoint);
  const afterBody=await evaluate('JSON.stringify(state.objects)');await history(beforeBody,afterBody,nBody);
  assert.equal(await evaluate("state.objects.find(o=>o.id==='silhouette-0').primitives[0].geometry.d.includes('Z')"),false);
  assert.deepEqual(JSON.parse(afterBody).filter(o=>o.id!=='silhouette-0'),JSON.parse(beforeBody).filter(o=>o.id!=='silhouette-0'));
  const bodyShot=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${artifacts}/body-detail.png`,Buffer.from(bodyShot.data,'base64'));
  // Manual closed cubic fixture and a compound neighbour; limits are explicit and tested.
  await evaluate(`(() => {const source=state.objects.find(o=>o.id==='hardware-0'),p=source.primitives[0],id='erase-curve-test';
    change({...state,objects:[...state.objects,{id,category:'hardware',type:'compound',order:9999,primitives:[
      {...p,geometry:{d:'M50 50C50 70 70 70 70 50Z'}},{...p,geometry:{d:'M90 50L100 50'}}]}]});})()`);
  await focus('erase-curve-test',8);await evaluate("$('eraseSize').value=3;$('erase').click()");
  const beforeCurve=await evaluate('JSON.stringify(state.objects)'),nCurve=await evaluate('past.length');
  await gesture({x:59.5,y:65},{x:60.5,y:65});
  const afterCurve=await evaluate('JSON.stringify(state.objects)');await history(beforeCurve,afterCurve,nCurve);
  const curve=await evaluate("state.objects.find(o=>o.id==='erase-curve-test')");
  assert(curve.primitives[0].geometry.d.includes('C'));assert(!curve.primitives[0].geometry.d.includes('Z'));
  assert.deepEqual(curve.primitives[1],JSON.parse(beforeCurve).find(o=>o.id==='erase-curve-test').primitives[1]);
  // An erased manual line becomes an editable multi-subpath object, not a broken rectangle.
  await evaluate(`{const obj=makeManual('line',{x:50,y:95},{x:70,y:95},'manual-'+(++manualSerial));change({...state,objects:[...state.objects,obj]});selected=obj.id;}`);
  const manualPathId=await evaluate('selected');await focus(manualPathId,12);
  await evaluate("$('eraseSize').value=3;$('erase').click()");
  await gesture({x:60,y:95},{x:60,y:95});
  assert.equal(await evaluate(`state.objects.find(o=>o.id==='${manualPathId}').type`),'path');
  await evaluate(`tool='select';selected='${manualPathId}';render();window.finishHit=document.createElement('style');finishHit.textContent='#stage svg > g:first-child {pointer-events:none} #stage .hit:not([data-object="${manualPathId}"]) {pointer-events:none!important}';document.head.append(finishHit)`);
  await gesture({x:54,y:95},{x:55,y:96});
  await evaluate('finishHit.remove()');
  const manualPath=await evaluate(`state.objects.find(o=>o.id==='${manualPathId}')`);
  const firstPoint=manualPath.primitives[0].geometry.d.match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g).slice(0,2).map(Number);
  assert(Math.abs(firstPoint[0]-51)<.02 && Math.abs(firstPoint[1]-96)<.02);
  await evaluate(`selected='${manualPathId}';duplicateSelected()`);
  const manualPathCopyId=await evaluate('selected');assert.notEqual(manualPathId,manualPathCopyId);
  assert.equal(await evaluate(`state.objects.find(o=>o.id==='${manualPathCopyId}').type`),'path');
  // Cancelling an already effective erase restores every primitive and preserves history.
  for(const ending of ['Escape','pointercancel','lostpointercapture','blur','space','middle']) {
    await focus('fret-0',10);await evaluate("$('eraseSize').value=4;$('erase').click()");
    const p=await evaluate(`(() => {const el=$('stage').querySelector('.hit[data-object="fret-0"]'),p=el.getPointAtLength(el.getTotalLength()*.4);return {x:p.x,y:p.y};})()`);
    const before=await evaluate('JSON.stringify(state.objects)'),n=await evaluate('past.length');
    await gesture(p,{x:p.x+1,y:p.y},{ending});
    assert.equal(await evaluate('JSON.stringify(state.objects)'),before,`erase cancel ${ending}`);assert.equal(await evaluate('past.length'),n);
  }
  // Panning ON a line in eraser mode must never cut it or add history.
  await panGesture('middle',false,'erase');await panGesture('left',true,'erase');
  // Cursor diameter is in image pixels and follows zoom; it never enters the export.
  await evaluate("tool='erase';$('eraseSize').value=14;zoomFit();refresh()");
  const pointer=await screen(midpoint);await mouse('mouseMoved',pointer);
  assert.equal(await evaluate("$('eraserCursor').hidden"),false);
  const cursorBefore=await evaluate("parseFloat($('eraserCursor').style.width)/z.s");assert(Math.abs(cursorBefore-14)<1e-4,`cursor diameter: ${cursorBefore}`);
  await evaluate('zoomCenter(1.4)');assert(Math.abs(await evaluate("parseFloat($('eraserCursor').style.width)/z.s")-14)<1e-4);
  assert.equal(await evaluate('exportSvg().querySelector("#eraserCursor,mask,image")===null'),true);
  // Remove just the synthetic cubic fixture; keep erasures/copies for the reopen suite.
  await evaluate("selected='erase-curve-test';tool='select';removeSelected();zoomFit()");
  // Keep real edits/copies for the existing close/reopen and byte-identical SVG/PNG suite.
  const text=await evaluate('GBCVProject.stringify(projectData())');await writeFile(`${artifacts}/guitar.gbcv.json`,text);
  await writeFile(`${artifacts}/trace.svg`,await evaluate('new XMLSerializer().serializeToString(exportSvg())'));
  await evaluate("downloads={};$('downloadPng').click()");
  const png=await evaluate(`new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{if(downloads['.png']){clearInterval(t);const r=new FileReader();r.onload=()=>resolve(r.result);r.readAsDataURL(downloads['.png']);}else if(++n>200){clearInterval(t);reject(Error('PNG timeout'));}},20);})`);
  await writeFile(`${artifacts}/trace.png`,Buffer.from(png.split(',')[1],'base64'));
  const shot=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${artifacts}/overlay.png`,Buffer.from(shot.data,'base64'));
  await writeFile(`${artifacts}/verification.json`,JSON.stringify({copyIds,altId,modelInference:false,concentricCopies:true,sourceUnchanged:true,fretGap:true,bodyCut:true,closedBezierOpened:true,compoundNeighbourUnchanged:true,gestureUndo:true,cancellation:true,panSafe:true,photoPixelCursor:true},null,2));
  console.log(`PASS: five concentric duplicates + placement, Ctrl+D/Alt-drag and cancellation; local fret/body/cubic cuts, compound neighbour intact; undo/redo, photo-pixel cursor, safe panning; ${artifacts}`);
}
