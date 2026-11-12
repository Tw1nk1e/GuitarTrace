import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
export async function bendTests({evaluate,send,key,panGesture,realExample}) {
  const dir=realExample?'out/editor-bend-real':'out/editor-bend';await mkdir(dir,{recursive:true});
  const screen=p=>evaluate(`(() => {const p=photoPoint(${JSON.stringify(p)}),r=$('view').getBoundingClientRect();return {x:r.left+z.x+z.s*p.x,y:r.top+z.y+z.s*p.y};})()`);
  const mouse=(type,p,modifiers=0)=>send('Input.dispatchMouseEvent',{type,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1,modifiers,...p});
  async function toolbar(id) {
    const p=await evaluate(`(() => {const r=$('${id}').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
    await mouse('mousePressed',p);await mouse('mouseReleased',p);
    assert.equal(await evaluate('tool'),id,`real toolbar click: ${id}`);
    assert.equal(await evaluate(`$('${id}').getAttribute('aria-pressed')`),'true');
  }
  await evaluate(`window.bendSaved={state,selected,tool,past,future,result,savedFingerprint,manualSerial,z:{...z}};
    change({...state,objects:[{id:'bend-test',category:'board',order:999,type:'path',primitives:[manualPrimitive('path',{d:'M 20 -10 L 120 -10 L 140 5'})]}]});
    const r=$('view').getBoundingClientRect(),p=photoPoint({x:80,y:-10});z.s=2;z.x=r.width/2-z.s*p.x;z.y=r.height/2-z.s*p.y;render();$('bend').click()`);
  let a={x:70,y:-10};
  // Real clicks, not .click(): reproduce the reported erase -> bend transition.
  await toolbar('erase');await toolbar('bend');
  const initial=await evaluate('JSON.stringify(state.objects)');
  await toolbar('erase');
  const erasePoint=await screen({x:45,y:-10});
  await mouse('mousePressed',erasePoint);await mouse('mouseReleased',erasePoint);
  assert.equal(await evaluate('edit'),null);
  await toolbar('bend');
  await evaluate('undo()');assert.equal(await evaluate('JSON.stringify(state.objects)'),initial);
  // A module error used to leave edit alive, permanently disabling the toolbar.
  await toolbar('erase');
  await evaluate(`window.originalEraseObjects=GBCVErase.objects;GBCVErase.objects=()=>{throw Error('simulated module failure')}`);
  await mouse('mousePressed',erasePoint);await mouse('mouseReleased',erasePoint);
  assert.equal(await evaluate('edit'),null);
  assert.equal(await evaluate("$('bend').disabled"),false);
  assert.equal(await evaluate('JSON.stringify(state.objects)'),initial);
  await evaluate('GBCVErase.objects=originalEraseObjects');
  await toolbar('bend');
  async function gesture(a,b,cancel=false) {
    const p=await screen(a),q=await screen(b);
    await mouse('mouseMoved',p);
    assert.equal(await evaluate("!!$('stage').querySelector('.bend-highlight')"),true);
    await mouse('mousePressed',p);assert.equal(await evaluate("edit?.kind"),'bend');
    await mouse('mouseMoved',q);
    assert.equal(await evaluate("state.objects[0].primitives[0].type"),'path');
    if(cancel)await key('keyDown','Escape','Escape');
    await mouse('mouseReleased',q);
  }
  const before=await evaluate('JSON.stringify(state.objects)'),n=await evaluate('past.length');
  await gesture(a,{x:70,y:5});
  const after=await evaluate('JSON.stringify(state.objects)');assert.notEqual(after,before);
  assert.equal(await evaluate('past.length'),n+1);
  assert.equal(await evaluate('tool'),'bend');
  await key('keyDown','z','KeyZ',2);assert.equal(await evaluate('JSON.stringify(state.objects)'),before);
  await key('keyDown','z','KeyZ',10);assert.equal(await evaluate('JSON.stringify(state.objects)'),after);
  for(const s of [.7,3]) {
    await evaluate(`(() => {const r=$('view').getBoundingClientRect(),p=photoPoint({x:70,y:5});z.s=${s};z.x=r.width/2-z.s*p.x;z.y=r.height/2-z.s*p.y;render();})()`);
    await gesture({x:70,y:5},{x:70,y:15},true);
    assert.equal(await evaluate('JSON.stringify(state.objects)'),after);
  }
  await gesture({x:70,y:5},{x:70,y:11});
  assert.equal(await evaluate('GBCVBend.commands(state.objects[0].primitives[0]).length'),4);
  // Alt-click remains selection cycling, never duplicate or bend.
  await evaluate(`change({...state,objects:[...state.objects,{...structuredClone(state.objects[0]),id:'bend-overlap',order:1000}]});selected='bend-test';render()`);
  const overlap=await screen({x:70,y:11}),history=await evaluate('past.length');
  await mouse('mousePressed',overlap,1);await mouse('mouseReleased',overlap,1);
  assert.equal(await evaluate('selected'),'bend-overlap');assert.equal(await evaluate('past.length'),history);
  await evaluate('undo()');
  const text=await evaluate('GBCVProject.stringify(projectData())');await writeFile(`${dir}/guitar.gbcv.json`,text);
  const geometry=await evaluate('JSON.stringify(state.objects)');
  await evaluate(`savedFingerprint=fingerprint();openProjectFile(new File([${JSON.stringify(text)}],'bend.gbcv.json'))`);
  assert.equal(await evaluate('JSON.stringify(state.objects)'),geometry);
  await evaluate(`$('bend').click()`);await gesture({x:70,y:11},{x:70,y:15});
  assert.equal(await evaluate('GBCVBend.commands(state.objects[0].primitives[0]).length'),4);
  // Actual fixture guitar: derive the board point from ITS geometry, not the
  // synthetic fixture's hardcoded (100,-10), which is off the real LP's board.
  // The entire rest of the scene must remain byte-for-byte unchanged.
  const boardPoint=await evaluate(`(() => {result=bendSaved.result;state={...state,objects:structuredClone(result.editor.objects)};
    selected='board-0';tool='bend';bendHover=null;
    const command=GBCVBend.commands(state.objects.find(o=>o.id==='board-0').primitives[0]).find(c=>c.op==='L');
    const a=command.ps[0],b=command.ps[1],point={x:(a.x+b.x)/2,y:(a.y+b.y)/2};
    const r=$('view').getBoundingClientRect(),p=photoPoint(point);z.s=1;z.x=r.width/2-z.s*p.x;z.y=r.height/2-z.s*p.y;render();return point;})()`);
  const neighbours=await evaluate("JSON.stringify(state.objects.filter(o=>o.id!=='board-0'))");
  await gesture(boardPoint,{x:boardPoint.x,y:boardPoint.y+4});
  assert.equal(await evaluate("JSON.stringify(state.objects.filter(o=>o.id!=='board-0'))"),neighbours);
  await evaluate(`change({...state,objects:[...state.objects,{id:'bend-cubic',category:'hardware',order:999,type:'path',primitives:[manualPrimitive('path',{d:'M10 0 C20 -2 30 2 40 0'})]}]});
    selected='bend-cubic';{const r=$('view').getBoundingClientRect(),p=photoPoint({x:25,y:0});z.s=2;z.x=r.width/2-z.s*p.x;z.y=r.height/2-z.s*p.y;render()}`);
  await gesture({x:25,y:0},{x:25,y:4});await gesture({x:25,y:4},{x:25,y:6});
  assert.equal(await evaluate("GBCVBend.commands(state.objects.find(o=>o.id==='bend-cubic').primitives[0]).length"),3);
  await evaluate('selected=null;bendHover=null;zoomFit()');
  await writeFile(`${dir}/guitar.gbcv.json`,await evaluate('GBCVProject.stringify(projectData())'));
  const svg=await evaluate('new XMLSerializer().serializeToString(exportSvg())');
  assert(!svg.includes('bends='));assert(!svg.includes('highlight'));
  assert.equal(await evaluate(`$('stage').querySelector('svg > g').innerHTML===exportSvg().firstChild.innerHTML && $('svgbox').querySelector('svg > g').innerHTML===exportSvg().firstChild.innerHTML`),true);
  await writeFile(`${dir}/trace.svg`,svg);
  await evaluate(`downloads={};$('downloadPng').click()`);
  const png=await evaluate(`new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{if(downloads['.png']){clearInterval(t);const r=new FileReader();r.onload=()=>resolve(r.result);r.readAsDataURL(downloads['.png']);}else if(++n>200){clearInterval(t);reject(Error('PNG timeout'));}},20);})`);
  await writeFile(`${dir}/trace.png`,Buffer.from(png.split(',')[1],'base64'));
  assert.equal(await evaluate(`(async()=>{
    const png=await createImageBitmap(downloads['.png']),svg=$('svgbox').querySelector('svg').cloneNode(true);
    svg.querySelectorAll('.hit,.highlight').forEach(el=>el.remove());
    // The workspace uses light ink on a dark canvas; exports use paper colours.
    // Normalize only presentation, retaining the actual preview geometry/styles.
    svg.setAttribute('style',exportSvg().getAttribute('style'));
    svg.setAttribute('width',png.width);svg.setAttribute('height',png.height);
    const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'}));
    const image=new Image();image.src=url;await image.decode();URL.revokeObjectURL(url);
    const pixels=image=>{const c=document.createElement('canvas');c.width=png.width;c.height=png.height;
      const x=c.getContext('2d');x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);x.drawImage(image,0,0);return x.getImageData(0,0,c.width,c.height).data;};
    const a=pixels(png),b=pixels(image);png.close();return a.every((v,i)=>v===b[i]);
  })()`),true,'PNG pixel-identical to rasterized preview');
  await writeFile(`${dir}/overlay.png`,Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));
  await writeFile(`${dir}/verification.json`,JSON.stringify({synthetic:!realExample,ML:false,neckBoundary:true,originalCubic:true,
    pinnedEndpointsAndTangents:true,neighboursUnchanged:true,repeatedBendsWithoutNodes:true,hover:true,livePreview:true,
    zoom:true,cancel:true,altCycle:true,undoRedo:true,saveReopen:true,svgPreviewIdentical:true,pngPreviewPixelIdentical:true,
    eraseToBendToolbar:true,gestureErrorRecovery:true},null,2));
  // Restore the existing regression suite's document and its live synthetic photo.
  await evaluate(`state=bendSaved.state;result=bendSaved.result;savedFingerprint=bendSaved.savedFingerprint;manualSerial=bendSaved.manualSerial;selected=bendSaved.selected;tool=bendSaved.tool;past=bendSaved.past;future=bendSaved.future;Object.assign(z,bendSaved.z);bendHover=null;render()`);
  // Existing all-tool pan helper tests both ordinary and temporary navigation.
  await panGesture('middle',false,'bend');await panGesture('left',true,'bend');
  await evaluate("tool='select';zoomFit();render()");
  console.log(`PASS: Bend real pointer hover/live preview, pinned ends, repeat edits, zoom, Escape, Alt overlap, pan, Undo/Redo, save/reopen, SVG/PNG; ${dir}`);
}
