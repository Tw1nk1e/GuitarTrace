import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';

// Real file download -> close tab/context -> new app page -> local file input.
export async function projectTests({evaluate,send,key,reopen,closeWarning,realExample}) {
  const artifacts=realExample?'out/editor-project-real':'out/editor-project';
  await mkdir(artifacts,{recursive:true});
  const head=await evaluate(`(() => {const g=state.objects.find(o=>o.category==='post').primitives[0].geometry;return {x:Number(g.cx),y:Number(g.cy)};})()`);
  const autoIds=await evaluate("state.objects.filter(o=>/^auto-/.test(o.id)).map(o=>o.id)");
  await evaluate(`(() => {const p=photoPoint(${JSON.stringify(head)}),q=photoPoint({x:${head.x+1},y:${head.y}}),r=$('view').getBoundingClientRect();
    z.s=5/Math.hypot(q.x-p.x,q.y-p.y);z.x=r.width*.5-z.s*p.x;z.y=r.height*.5-z.s*p.y;render();})()`);
  async function screen(point) {
    return evaluate(`(() => {const p=photoPoint(${JSON.stringify(point)}),r=$('view').getBoundingClientRect();return {x:r.left+z.x+z.s*p.x,y:r.top+z.y+z.s*p.y};})()`);
  }
  async function gesture(a,b) {
    const p=await screen(a),q=await screen(b);
    await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...p});
    await send('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,...q});
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,...q});
  }
  async function input(selector,filename) {
    const {root}=await send('DOM.getDocument');
    const {nodeId}=await send('DOM.querySelector',{nodeId:root.nodeId,selector});
    await send('DOM.setFileInputFiles',{nodeId,files:[path.resolve(filename)]});
    await evaluate(`new Promise((resolve,reject)=>{let n=0;const timer=setInterval(()=>{
      if(!busy){clearInterval(timer);resolve();}else if(++n>200){clearInterval(timer);reject(Error('File input timeout'));}
    },20);})`);
  }
  async function captureExports() {
    await evaluate(`window.downloads={};save=(blob,ext)=>{downloads[ext]=blob;};$('download').click();$('downloadPng').click();`);
    const png=await evaluate(`new Promise((resolve,reject)=>{let n=0;const timer=setInterval(()=>{
      if(downloads['.png']){clearInterval(timer);const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.readAsDataURL(downloads['.png']);}
      else if(++n>200){clearInterval(timer);reject(Error('PNG timeout'));}
    },20);})`);
    return {svg:await evaluate("downloads['.svg'].text()"),png};
  }
  await evaluate(`tool='select';selected=null;render()`);
  for(let i=0;i<2;i++) {
    const a={x:head.x+15*i,y:head.y+12},b={x:a.x+4,y:a.y};
    await evaluate("$('rings').click()");await gesture(a,b);
  }
  const added=await evaluate('state.objects.slice(-2).map(o=>o.id)');
  assert.equal(added.length,2);
  assert.equal(await evaluate(`state.objects.slice(-2).every(o=>o.category==='manual' && o.type==='compound')`),true);
  // Move an already editable object and remove unwanted automatic hardware.
  const existing=await evaluate(`state.objects.find(o=>o.category==='manual' && o.type==='compound').id`);
  const center=await evaluate(`(() => {const g=state.objects.find(o=>o.id==='${existing}').primitives[0].geometry;return {x:g.cx,y:g.cy};})()`);
  await gesture(center,{x:center.x+3,y:center.y+3});
  const moved=await evaluate(`state.objects.find(o=>o.id==='${existing}').primitives[0].geometry`);
  assert(Math.abs(moved.cx-center.x-3)<.01 && Math.abs(moved.cy-center.y-3)<.01);
  await evaluate("selected='hardware-0';removeSelected()");
  assert.equal(await evaluate('isDirty()'),true);
  assert.equal(await evaluate("$('dirty').textContent"),'Unsaved changes');
  await evaluate("saveProject()");
  assert.equal(await evaluate('isDirty()'),false,await evaluate("$('status').textContent"));
  const text=await evaluate("downloads['.gbcv.json'].text()");
  const project=JSON.parse(text);
  assert.equal(project.version,2);assert(!text.includes('data:image'));assert(!text.includes('base64'));assert(!text.includes('http:'));
  for(const id of autoIds)assert(project.document.objects.some(o=>o.id===id));
  assert(!('history' in project));assert(!('id' in project.document));
  const file=path.join(artifacts,'guitar.gbcv.json');await writeFile(file,text);
  const before=await captureExports();
  await writeFile(path.join(artifacts,'before.svg'),before.svg);
  await writeFile(path.join(artifacts,'before.png'),Buffer.from(before.png.split(',')[1],'base64'));
  const beforeScene=await evaluate('JSON.stringify(state)'),beforeView=await evaluate('JSON.stringify(z)');
  const beforeProjection=JSON.stringify(project.document.photo);
  assert.equal(await evaluate('isDirty()'),false,'SVG/PNG/photo downloads do not change the saved state');
  await reopen();
  assert.equal(await evaluate('state'),null,'old browser context has really gone');
  // Any accidental session/ML/API request now fails the test immediately.
  await evaluate(`window.fetch=()=>{throw Error('Unexpected fetch while restoring a project');};
    navigator.sendBeacon=()=>{throw Error('Unexpected beacon in a local project');};`);
  await input('#projectFile',file);
  assert.equal(await evaluate('JSON.stringify(state)'),beforeScene);
  assert.equal(await evaluate('JSON.stringify(z)'),beforeView);
  assert.equal(await evaluate('JSON.stringify(result.editor.photo)'),beforeProjection);
  assert.equal(await evaluate('result.photo'),null);
  assert.equal(await evaluate('past.length+future.length'),0);
  for(const id of autoIds)assert.equal(await evaluate(`state.objects.some(o=>o.id==='${id}')`),true,'SAM contour survived file reopen');
  assert.equal(await evaluate('isDirty()'),false);
  assert.equal(await evaluate(`state.objects.some(o=>o.id==='hardware-0')`),false);
  const after=await captureExports();
  assert.equal(after.svg,before.svg,'exact SVG equality without a photo');
  assert.equal(after.png,before.png,'exact PNG equality after closing/reopening');
  await writeFile(path.join(artifacts,'restored.svg'),after.svg);
  await writeFile(path.join(artifacts,'restored.png'),Buffer.from(after.png.split(',')[1],'base64'));
  // Drawing is editable even before an image is reattached; projection stays fixed.
  await evaluate("$('line').click()");
  await gesture({x:head.x-8,y:head.y+8},{x:head.x+4,y:head.y+8});
  assert.equal(await evaluate('isDirty()'),true);
  await key('keyDown','z','KeyZ',2);
  assert.equal(await evaluate('JSON.stringify(state)'),beforeScene);
  assert.equal(await evaluate('isDirty()'),false);
  // Malformed and malicious files must leave the current scene and savepoint intact.
  const mutations=[
    p=>p.version=999,
    p=>p.document.objects.push(structuredClone(p.document.objects[0])),
    p=>p.document.photo.src='https://attacker.invalid/secret',
    p=>p.document.objects[0].primitives[0].styles.comic.stroke='url(https://attacker.invalid)',
    p=>p.document.objects[0].primitives[0].geometry.onload='globalThis.pwned=true',
  ];
  for(const mutate of mutations) {
    const malicious=structuredClone(project);mutate(malicious);
    const ok=await evaluate(`openProjectFile(new File([${JSON.stringify(JSON.stringify(malicious))}],'malicious.json',{type:'application/json'}))`);
    assert.equal(ok,false);assert.equal(await evaluate('JSON.stringify(state)'),beforeScene);assert.equal(await evaluate('isDirty()'),false);
  }
  assert.equal(await evaluate(`openProjectFile(new File(['{'],'broken.json'))`),false);
  assert.equal(await evaluate(`openProjectFile({size:GBCVProject.MAX_BYTES+1,text:()=>{throw Error('MUST NOT READ oversized file');}})`),false);
  assert.equal(await evaluate(`openProjectFile(new File(['{"__proto__":{"pwned":true}}'],'evil.json'))`),false);
  assert.equal(await evaluate('({}).pwned===undefined && globalThis.pwned===undefined'),true);
  assert.equal(await evaluate("$('attachPhoto')===null && $('savePhoto')===null && $('photoFile')===null"),true,'underlay import/download were intentionally removed');
  assert.equal(await evaluate('typeof attachPhoto'), 'undefined');
  assert.equal(await evaluate('typeof savePhoto'), 'undefined');
  assert.equal(await evaluate('JSON.stringify(state)'),beforeScene);
  assert.equal(await evaluate('JSON.stringify(z)'),beforeView);
  assert.equal(await evaluate('isDirty()'),false);
  // Fresh edits have fresh IDs and independent Undo/Redo after import.
  await evaluate("$('circle').click()");
  const point={x:head.x-10,y:head.y+10};await gesture(point,{x:point.x+5,y:point.y});
  const fresh=await evaluate('selected');assert(!project.document.objects.some(o=>o.id===fresh));
  assert.equal(await evaluate('isDirty()'),true);assert.equal(await evaluate('past.length'),1);
  await key('keyDown','z','KeyZ',2);assert.equal(await evaluate('JSON.stringify(state)'),beforeScene);assert.equal(await evaluate('isDirty()'),false);
  await key('keyDown','z','KeyZ',10);assert.equal(await evaluate('isDirty()'),true);
  // Real beforeunload dialog, not just a synthetic event. Cancel and keep working.
  await closeWarning();
  assert.equal(await evaluate('isDirty()'),true);
  // Replacement confirmation also protects the current unsaved scene.
  await evaluate("window.confirm=()=>false");
  assert.equal(await evaluate(`openProjectFile(new File([${JSON.stringify(text)}],'saved.gbcv.json'))`),false);
  assert.equal(await evaluate(`state.objects.some(o=>o.id==='${fresh}')`),true);
  await evaluate(`{const original=save;save=()=>{throw Error('Download failed');};saveProject();save=original;}`);
  assert.equal(await evaluate('isDirty()'),true,'failed download must not mark a project as saved');
  await evaluate('saveProject()');assert.equal(await evaluate('isDirty()'),false);
  const final=await evaluate("downloads['.gbcv.json'].text()");await writeFile(path.join(artifacts,'continued.gbcv.json'),final);
  const shot=await send('Page.captureScreenshot',{format:'png'});await writeFile(path.join(artifacts,'restored-overlay.png'),Buffer.from(shot.data,'base64'));
  await writeFile(path.join(artifacts,'verification.json'),JSON.stringify({closedContext:true,restoredWithoutML:true,svgByteIdentical:true,pngByteIdentical:true,idsAndDisplayPreserved:true,photoLocalOnly:true,maliciousInputsRejected:true,newUndoRedo:true,realBeforeUnloadDialog:true},null,2));
  console.log(`PASS: project save -> closed page -> fresh page -> real file input; byte-identical SVG/PNG without photo; removed underlay controls; new edits/IDs/Undo/Redo; real beforeunload warning and cancelled replacement; malicious input rejection. Artifacts: ${artifacts}`);
}
