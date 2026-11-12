import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

export async function autoTests({evaluate,send,clickObject,panGesture,realExample}) {
  const metadata=await evaluate("fetch('/test-auto').then(r=>r.json())");
  async function dispatchClick() {
    const p=await evaluate(`(() => {const r=$('view').getBoundingClientRect(),photo=result.editor.photo;
      return {x:r.left+z.x+z.s*${metadata.x}*photo.width,y:r.top+z.y+z.s*${metadata.y}*photo.height};})()`);
    await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...p});
    await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,...p});
  }
  async function idle() {await evaluate(`new Promise((resolve,reject)=>{let n=0;const timer=setInterval(()=>{if(!busy){clearInterval(timer);resolve();}else if(++n>300){clearInterval(timer);reject(Error('Auto selection timeout'));}},20);})`);}
  async function click() {await dispatchClick();await idle();}
  const calls=()=>evaluate("fetch('/test-auto').then(r=>r.json()).then(v=>v.calls)");
  const baseline=await calls();
  assert.equal(baseline,0,'opening, movement and tool changes must not start a model');
  await panGesture('middle',false,'auto');await panGesture('left',true,'auto');
  assert.equal(await calls(),baseline,'panning must not run SAM');
  await evaluate(`window.originalConfirm=window.confirm;window.autoPrompts=[];window.confirm=message=>{autoPrompts.push(message);return false;};$('auto').click();zoomFit()`);
  const cancelled=await evaluate('JSON.stringify(state)'),history=await evaluate('past.length');
  await click();assert.equal(await calls(),baseline);assert.equal(await evaluate('JSON.stringify(state)'),cancelled);assert.equal(await evaluate('past.length'),history);
  assert((await evaluate('autoPrompts[0]')).includes('editor session'));
  assert.equal(await evaluate("fetch('/test-auto').then(r=>r.json()).then(v=>v.sam_enabled)"),false);
  // A local project cannot silently manufacture a missing source mask or re-trace.
  await evaluate("window.originalSid=result.id;result.id=null");await click();
  assert.equal(await calls(),baseline);assert((await evaluate("$('status').textContent")).includes('live session'));
  await evaluate("result.id=originalSid;window.confirm=message=>{autoPrompts.push(message);return true;}");
  const promptsBefore=await evaluate('autoPrompts.length');
  // Recover a previously removed contour, rather than asking SAM to duplicate it.
  if(realExample)assert.notEqual(await evaluate(`duplicateContour(${JSON.stringify(metadata.object)})`),null,'saved stop-bar overlaps an existing detected contour');
  await evaluate(`{const obj=${JSON.stringify(metadata.object)};let id;
    while((id=duplicateContour(obj))){selected=id;removeSelected();}}`);
  // Deduplication also handles reversed/cyclic path starts and different IDs/styles.
  assert.equal(await evaluate(`(() => {
    const obj=${JSON.stringify(metadata.object)},a=structuredClone(obj),b=structuredClone(obj);
    a.id='dedup-test';a.primitives[0].geometry.d='M10 10L20 10L20 20L10 20Z';
    b.primitives[0].geometry.d='M20 20L20 10L10 10L10 20Z';
    change({...state,objects:[...state.objects,a]});const id=duplicateContour(b);undo();return id;
  })()`),'dedup-test');
  assert.equal(await evaluate(`(() => {
    const obj=${JSON.stringify(metadata.object)},a=structuredClone(obj),b=structuredClone(obj);
    a.id='distinct-test';a.primitives[0].geometry.d='M10 10L20 10L20 20L10 20Z';
    b.primitives[0].geometry.d='M15 10L20 15L15 20L10 15Z';
    change({...state,objects:[...state.objects,a]});const id=duplicateContour(b);undo();return id;
  })()`),null,'same bounding box alone must not suppress a distinct contour');
  const before=await evaluate('JSON.stringify(state.objects)'),startHistory=await evaluate('past.length');
  await evaluate("$('auto').click()");await click();
  assert.equal(await calls(),baseline+1);
  assert.equal(await evaluate('tool'),'auto','keep selecting without re-entering the tool');
  assert.equal(await evaluate('autoPrompts.length'),promptsBefore+1);
  const id=await evaluate('selected');assert.match(id,/^auto-\d+$/);
  assert.equal(await evaluate('past.length'),startHistory+1);
  const added=await evaluate(`state.objects.find(o=>o.id==='${id}')`);
  assert.equal(added.category,'hardware');assert.equal(added.type,'path');
  assert.deepEqual(added.primitives,metadata.object.primitives,'exact existing mask -> part -> renderer');
  assert.equal(await evaluate(`JSON.stringify(state.objects.filter(o=>o.id!=='${id}'))`),before,'manual edits and all other objects unchanged');
  const after=await evaluate('JSON.stringify(state)');
  await evaluate("$('auto').click()");await click();
  assert.equal(await calls(),baseline+1,'repeat click uses the session cache');assert.equal(await evaluate('JSON.stringify(state)'),after);
  assert.equal(await evaluate('autoPrompts.length'),promptsBefore+1,'no repeated consent');
  assert.equal(await evaluate('past.length'),startHistory+1);
  assert((await evaluate("$('status').textContent")).includes('no duplicate added'));
  for(const mode of ['none','unavailable']) {
    await evaluate(`fetch('/test-auto/mode/${mode}',{method:'POST'})`);
    await click();assert.equal(await evaluate('JSON.stringify(state)'),after);assert.equal(await evaluate('past.length'),startHistory+1);
    assert.equal(await evaluate("$('status').className"),'err');
  }
  await evaluate("fetch('/test-auto/mode/good',{method:'POST'})");
  // Cancellation drops the response, NOT the server lock around an active model.
  const cancelBefore=await evaluate('JSON.stringify(state)'),cancelHistory=await evaluate('past.length');
  await evaluate("fetch('/test-auto/mode/slow',{method:'POST'})");
  await dispatchClick();
  await evaluate(`new Promise((resolve,reject)=>{let n=0;const timer=setInterval(async()=>{const m=await fetch('/test-auto').then(r=>r.json());if(m.active){clearInterval(timer);resolve();}else if(++n>100){clearInterval(timer);reject(Error('No slow request'));}},20);})`);
  assert.equal(await evaluate("!$('cancelAuto').hidden && !$('autoProgress').hidden && busy"),true);
  const slowCalls=await calls();
  await dispatchClick();assert.equal(await calls(),slowCalls,'rapid clicks do not infer in parallel');
  await evaluate("$('cancelAuto').click()");
  assert.equal(await evaluate('busy'),false);
  assert.equal(await evaluate('JSON.stringify(state)'),cancelBefore);
  assert.equal(await evaluate('past.length'),cancelHistory);
  await click(); // a real second HTTP request is rejected by the server, not queued
  assert.equal(await calls(),slowCalls);
  assert.equal(await evaluate("$('status').className"),'err');
  await evaluate("fetch('/test-auto/mode/good',{method:'POST'})");
  await evaluate(`new Promise((resolve,reject)=>{let n=0;const timer=setInterval(async()=>{const m=await fetch('/test-auto').then(r=>r.json());if(!m.active){clearInterval(timer);resolve();}else if(++n>100){clearInterval(timer);reject(Error('Slow request still active'));}},20);})`);
  assert.equal(await evaluate('JSON.stringify(state)'),cancelBefore,'late cancelled response never changes the drawing');
  assert.equal(await evaluate("fetch('/test-auto').then(r=>r.json()).then(v=>v.max_active)"),1);
  await click();assert.equal(await calls(),slowCalls,'completed cancelled request supplies a reusable cache entry');
  // Even a transport that ignores AbortSignal cannot merge a stale response.
  await evaluate(`window.transportFetch=window.fetch;window.fetch=(url,options)=>url.includes('/auto-select')?new Promise(resolve=>{window.lateAuto=resolve;}):transportFetch(url,options)`);
  await dispatchClick();
  await evaluate(`new Promise(resolve=>{const timer=setInterval(()=>{if(window.lateAuto){clearInterval(timer);resolve();}},10);})`);
  await evaluate("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))");
  await evaluate(`lateAuto(new Response(${JSON.stringify(JSON.stringify({object:metadata.object}))},{status:200}));window.fetch=transportFetch`);
  await idle();assert.equal(await evaluate('JSON.stringify(state)'),cancelBefore);
  assert.equal(await evaluate('autoPrompts.length'),promptsBefore+1);
  const count=await calls();
  await evaluate(`window.originalFetch=window.fetch;window.fetch=(url,options)=>url.includes('/auto-select')?Promise.reject(Error('Offline')):originalFetch(url,options)`);
  await click();assert.equal(await evaluate('JSON.stringify(state)'),after);
  await evaluate('window.fetch=originalFetch');assert.equal(await calls(),count);
  for(const [body,status] of [['Service failed',500],[JSON.stringify({detail:'expired'}),404]]) {
    await evaluate(`window.fetch=(url,options)=>url.includes('/auto-select')?Promise.resolve(new Response(${JSON.stringify(body)},{status:${status}})):originalFetch(url,options)`);
    await click();assert.equal(await evaluate('JSON.stringify(state)'),after);
    assert.equal(await evaluate("$('status').className"),'err');
  }
  // Reject malformed server object as well as malformed imported projects.
  const evil=structuredClone(metadata.object);evil.primitives[0].geometry.onload='globalThis.pwned=true';
  await evaluate(`window.fetch=(url,options)=>url.includes('/auto-select')?Promise.resolve(new Response(${JSON.stringify(JSON.stringify({object:evil}))},{status:200})):originalFetch(url,options)`);
  await click();assert.equal(await evaluate('JSON.stringify(state)'),after);assert.equal(await evaluate('globalThis.pwned'),undefined);
  await evaluate("window.fetch=originalFetch;window.confirm=originalConfirm;$('select').click()");
  await clickObject(id,true);
  await evaluate("$('delete').click()");assert.equal(await evaluate(`state.objects.some(o=>o.id==='${id}')`),false);
  await evaluate('undo()');assert.equal(await evaluate('JSON.stringify(state)'),after);
  await evaluate('undo()');assert.equal(await evaluate('JSON.stringify(state.objects)'),before);
  await evaluate('redo()');assert.equal(await evaluate('JSON.stringify(state)'),after);
  assert.equal(await evaluate(`$('stage').querySelector('svg > g').innerHTML===exportSvg().firstChild.innerHTML && $('svgbox').querySelector('svg > g').innerHTML===exportSvg().firstChild.innerHTML`),true);
  const text=await evaluate('GBCVProject.stringify(projectData())');assert(JSON.parse(text).document.objects.some(o=>o.id===id));
  const artifacts=realExample?'out/editor-auto-real':'out/editor-auto';await mkdir(artifacts,{recursive:true});
  await writeFile(`${artifacts}/guitar.gbcv.json`,text);
  await writeFile(`${artifacts}/trace.svg`,await evaluate('new XMLSerializer().serializeToString(exportSvg())'));
  await evaluate("downloads={};$('downloadPng').click()");
  const png=await evaluate(`new Promise(resolve=>{const timer=setInterval(()=>{if(downloads['.png']){clearInterval(timer);const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.readAsDataURL(downloads['.png']);}},20);})`);
  await writeFile(`${artifacts}/trace.png`,Buffer.from(png.split(',')[1],'base64'));
  await clickObject(id,true);
  await evaluate("status('Auto select: outline added from a fixture / saved mask. No new ML inference.')");
  const shot=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${artifacts}/overlay.png`,Buffer.from(shot.data,'base64'));
  await writeFile(`${artifacts}/verification.json`,JSON.stringify({id,modelInference:false,source:realExample?'real LP + saved SAM mask':'synthetic mask',sessionConsent:true,cacheReused:true,serialRequests:true,cancelAndLateResponse:true,preservedEdits:true,deduplicated:true,failuresAtomic:true,undoRedo:true,exactPreviewExport:true,projectSaved:true},null,2));
  console.log(`PASS: session opt-in auto selection (NO ML), cache, serial requests, cancellation/late response, selection/delete/Undo/Redo, no duplicate, preserved edits, failure/no-session/pan guards; ${artifacts}`);
}
