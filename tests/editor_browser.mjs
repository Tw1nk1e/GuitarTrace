// Optional dependency-free Chromium smoke test. First run the synthetic fixture server.
// node tests/editor_browser.mjs [path-to-chrome-or-edge] [real-example-url]
// With editor_auto_fixture_server.py: append --auto-test (and --synthetic for its synthetic URL).
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, writeFile, rm} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {manualTests} from './editor_manual_browser.mjs';
import {projectTests} from './editor_project_browser.mjs';
import {autoTests} from './editor_auto_browser.mjs';
import {transformTests} from './editor_transform_browser.mjs';
import {finishTests} from './editor_finish_browser.mjs';
import {bendTests} from './editor_bend_browser.mjs';
import {nodesTests} from './editor_nodes_browser.mjs';

const profile = await mkdtemp(path.join(process.env.LOCALAPPDATA, 'Temp', 'opencode', 'editor-browser-'));
const browser = spawn(process.argv[2] || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, '--window-size=1400,1000', 'about:blank']);
let socket, nextId = 0, session, errors = [], stderr = '';
const waiting = new Map();
const timeout = setTimeout(() => { console.error('Browser test timed out'); browser.kill(); process.exit(1); }, 120000);
try {
  const endpoint = await new Promise((resolve, reject) => {
    browser.on('error', reject);
    browser.stderr.on('data', chunk => { stderr += chunk; const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/); if (match) resolve(match[1]); });
    browser.on('exit', code => reject(new Error(`Browser exited: ${code}\n${stderr}`)));
  });
  socket = new WebSocket(endpoint);
  await new Promise(resolve => socket.addEventListener('open', resolve, {once: true}));
  socket.addEventListener('message', event => {
    const msg = JSON.parse(event.data);
    if (msg.id) { const task = waiting.get(msg.id); waiting.delete(msg.id); msg.error ? task.reject(msg.error) : task.resolve(msg.result); }
    if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails);
  });
  const send = (method, params = {}, target = session) => new Promise((resolve, reject) => {
    const id = ++nextId; waiting.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params, ...(target ? {sessionId: target} : {})}));
  });
  let {targetId} = await send('Target.createTarget', {url: 'about:blank'});
  ({sessionId: session} = await send('Target.attachToTarget', {targetId, flatten: true}));
  await send('Runtime.enable');
  async function evaluate(expression) {
    const response = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  }
  await send('Page.enable');
  const loaded = new Promise(resolve => socket.addEventListener('message', function onLoad(event) {
    const msg = JSON.parse(event.data);
    if (msg.method === 'Page.loadEventFired' && msg.sessionId === session) { socket.removeEventListener('message', onLoad); resolve(); }
  }));
  const sourceUrl=process.argv[3] || 'http://127.0.0.1:8765/';
  const realExample = process.argv.includes('--synthetic')?null:process.argv[3];
  await send('Page.navigate', {url: sourceUrl});
  await loaded;
  if (realExample) await evaluate(`new Promise((resolve,reject) => {let n=0; const timer=setInterval(() => {
    if (state && !busy) {clearInterval(timer);resolve();} else if (++n>200) {clearInterval(timer);reject(Error($('status').textContent));}
  },25);})`);
  else await evaluate(`(async () => { const canvas=document.createElement('canvas'); canvas.width=640; canvas.height=320;
    const blob=await new Promise(resolve=>canvas.toBlob(resolve)); await upload(new File([blob], 'synthetic.png', {type:'image/png'})); })()`);
  assert.equal(await evaluate('!!state'), true, await evaluate("$('status').textContent"));
  if(process.argv.includes('--quality-gallery')) {
    const {renderQualityGallery}=await import('./quality_gallery_browser.mjs');
    await renderQualityGallery({evaluate});
  }
  await evaluate(`Promise.all([...document.images].map(img => img.decode()))`);
  await evaluate(`window.mouseLog=[];document.addEventListener('click',e=>{mouseLog.push([e.button,e.clientX,e.clientY,e.target.tagName,e.target.getAttribute('data-object')]);if(mouseLog.length>5)mouseLog.shift();},true)`);
  async function clickObject(id, overlay = false, alt = false) {
    const point = await evaluate(`(() => {
      const el = document.querySelector('${overlay ? '#stage' : '#svgbox'} .hit[data-object="${id}"]');
      const p = el.getPointAtLength(el.getTotalLength() * .2);
      ${overlay ? `const H = result.editor.photo.projection, den = H[6]*p.x+H[7]*p.y+H[8];
        const v = $('view').getBoundingClientRect();
        return {x:v.left+z.x+z.s*(H[0]*p.x+H[1]*p.y+H[2])/den, y:v.top+z.y+z.s*(H[3]*p.x+H[4]*p.y+H[5])/den};` :
        `const q = new DOMPoint(p.x,p.y).matrixTransform(el.getScreenCTM()); return {x:q.x,y:q.y};`}
    })()`);
    await send('Input.dispatchMouseEvent', {type:'mousePressed', button:'left', clickCount:1, modifiers: alt ? 1 : 0, ...point});
    await send('Input.dispatchMouseEvent', {type:'mouseReleased', button:'left', clickCount:1, modifiers: alt ? 1 : 0, ...point});
    // At compact viewport sizes even synthetic contours' 10px hit areas can
    // overlap at this sample point. Use the existing Alt cycle in either fixture.
    for (let n=0; n<100 && await evaluate('selected') !== id; n++) {
      await send('Input.dispatchMouseEvent', {type:'mousePressed',button:'left',clickCount:1,modifiers:1,...point});
      await send('Input.dispatchMouseEvent', {type:'mouseReleased',button:'left',clickCount:1,modifiers:1,...point});
    }
    if (await evaluate('selected') !== id) {
      const shot=await send('Page.captureScreenshot',{format:'png'});
      await mkdir('out/editor-manual',{recursive:true});await writeFile('out/editor-manual/failure.png',Buffer.from(shot.data,'base64'));
      console.error(JSON.stringify({point,debug:await evaluate(`({mouseLog,tool,edit:!!edit,drag:!!drag,spacePan,hits:document.elementsFromPoint(${point.x},${point.y}).map(el=>[el.tagName,el.getAttribute('data-object')]),svg:$('svgbox').getBoundingClientRect().toJSON()})`),errors}));
    }
    assert.equal(await evaluate('selected'), id, `mouse selection ${id}, overlay=${overlay}`);
  }
  const count = await evaluate('state.objects.length');
  for (const id of ['fret-1','hardware-0','post-0','key-0','control-0','nut-0','board-0','silhouette-0']) {
    await clickObject(id);
    await evaluate("$('delete').click()");
    assert.equal(await evaluate('state.objects.length'), count - 1);
    assert.equal(await evaluate(`!!document.querySelector('#stage [data-object="${id}"]')`), false);
    await evaluate("$('undo').click()");
    assert.equal(await evaluate('state.objects.length'), count);
    await evaluate("$('redo').click()");
    assert.equal(await evaluate('state.objects.length'), count - 1);
    await evaluate("$('undo').click()");
  }
  // Check the true perspective overlay by clicking calculated photo-pixel locations.
  await clickObject('fret-1', true);
  await clickObject('post-0', true);
  await evaluate(`change({...state, objects:[...state.objects, {...state.objects.find(o=>o.id==='hardware-0'), id:'overlap-test', order:999}]})`);
  await clickObject('overlap-test');
  await clickObject('hardware-0', false, true);
  await evaluate('undo()');
  const beforePan = await evaluate('({history:past.length, x:z.x, y:z.y})');
  await evaluate("tool='pan';render()");
  const panPoint = await evaluate("(() => {const r=$('view').getBoundingClientRect();return {x:r.left+30,y:r.top+30};})()");
  await send('Input.dispatchMouseEvent', {type:'mousePressed', button:'left', clickCount:1, ...panPoint});
  await send('Input.dispatchMouseEvent', {type:'mouseMoved', button:'left', buttons:1, x:panPoint.x+25, y:panPoint.y+20});
  await send('Input.dispatchMouseEvent', {type:'mouseReleased', button:'left', x:panPoint.x+25, y:panPoint.y+20});
  assert.equal(await evaluate('z.x'), beforePan.x+25);
  assert.equal(await evaluate('past.length'), beforePan.history);
  await evaluate("$('select').click(); zoomFit()");
  async function key(type, key, code, modifiers = 0) {
    await send('Input.dispatchKeyEvent', {type, key, code, modifiers});
  }
  async function panGesture(button, space, mode, extra = '') {
    await evaluate(`tool='${mode}'; zoomFit(); selected='fret-1'; refresh()`);
    // Start ON an object: temporary panning must win over hit testing.
    const p = await evaluate(`(() => {const el=document.querySelector('#stage .hit[data-object="fret-1"]');
      const p=el.getPointAtLength(el.getTotalLength()*.2), q=new DOMPoint(p.x,p.y).matrixTransform(el.getScreenCTM());return {x:q.x,y:q.y};})()`);
    const before = await evaluate('({x:z.x,y:z.y,selected,objects:state.objects.length,past:past.length,future:future.length})');
    if (space) await key('keyDown', ' ', 'Space');
    await send('Input.dispatchMouseEvent', {type:'mousePressed',button,buttons:button==='middle'?4:1,clickCount:1,...p});
    assert.equal(await evaluate('!!drag'), true);
    await key('keyDown','Delete','Delete');
    await key('keyDown','Backspace','Backspace');
    await key('keyDown','z','KeyZ',2);
    await evaluate("$('delete').click(); removeSelected(); undo(); redo()");
    if (extra === 'releaseSpace') await key('keyUp',' ','Space');
    if (extra === 'switchTool') await evaluate("tool='pan';render()");
    await send('Input.dispatchMouseEvent', {type:'mouseMoved',button,buttons:button==='middle'?4:1,x:p.x+25,y:p.y+20});
    // CDP pointer coordinates are rounded to float32 at fractional grid positions.
    assert(Math.abs(await evaluate('z.x')-(before.x+25))<.001);
    assert(Math.abs(await evaluate('z.y')-(before.y+20))<.001);
    assert.equal(await evaluate('selected'), before.selected);
    assert.equal(await evaluate('state.objects.length'), before.objects);
    assert.equal(await evaluate('past.length'), before.past);
    assert.equal(await evaluate('future.length'), before.future);
    await send('Input.dispatchMouseEvent', {type:'mouseReleased',button,buttons:0,x:p.x+25,y:p.y+20});
    if (space) await key('keyUp',' ','Space');
    assert.equal(await evaluate('drag'), null);
    assert.equal(await evaluate('spacePan'), false);
  }
  for (const mode of ['select','pan']) {
    await panGesture('middle',false,mode);
    await panGesture('left',true,mode);
  }
  await panGesture('left',true,'select','releaseSpace');
  await panGesture('middle',false,'select','switchTool');
  for (const ending of ['pointercancel', 'lostpointercapture', 'blur']) {
    await evaluate("tool='select'; refresh()");
    await key('keyDown',' ','Space');
    await send('Input.dispatchMouseEvent', {type:'mousePressed',button:'left',buttons:1,clickCount:1,...panPoint});
    assert.equal(await evaluate('!!drag'), true);
    await evaluate(ending === 'blur' ? "window.dispatchEvent(new Event('blur'))" : `$('view').dispatchEvent(new PointerEvent('${ending}',{pointerId:drag.pointerId}))`);
    assert.equal(await evaluate('drag'), null, ending);
    await send('Input.dispatchMouseEvent', {type:'mouseReleased',button:'left',buttons:0,...panPoint});
    await key('keyUp',' ','Space');
  }
  // Default actions are cancelled, not merely hidden by an active drag.
  assert.equal(await evaluate(`['mousedown','auxclick'].every(type => {
    const e=new MouseEvent(type,{button:1,bubbles:true,cancelable:true}); $('view').dispatchEvent(e); return e.defaultPrevented;
  })`), true);
  await evaluate("tool='select'; zoomFit(); refresh()");
  await clickObject('fret-1',true);
  await key('keyDown',' ','Space');
  assert.equal(await evaluate("$('view').classList.contains('pan')"), true);
  await evaluate("window.dispatchEvent(new Event('blur'))");
  assert.equal(await evaluate('spacePan || !!drag'), false);
  await evaluate("$('opacity').focus()");
  await key('keyDown',' ','Space');
  assert.equal(await evaluate('spacePan'), false, 'space in a form control is not a canvas shortcut');
  await key('keyUp',' ','Space');
  await evaluate('document.activeElement.blur()');
  const wheelPoint = await evaluate("(() => {const r=$('view').getBoundingClientRect();return {x:Math.round(r.left+r.width*.35),y:Math.round(r.top+r.height*.4)};})()");
  const anchor = await evaluate(`(() => {const r=$('view').getBoundingClientRect();return {x:(${wheelPoint.x}-r.left-z.x)/z.s,y:(${wheelPoint.y}-r.top-z.y)/z.s,s:z.s};})()`);
  await send('Input.dispatchMouseEvent',{type:'mouseWheel',...wheelPoint,deltaX:0,deltaY:-120});
  // CDP dispatch may return before the wheel frame is rendered.
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  const afterWheel = await evaluate(`(() => {const r=$('view').getBoundingClientRect();return {x:(${wheelPoint.x}-r.left-z.x)/z.s,y:(${wheelPoint.y}-r.top-z.y)/z.s,s:z.s};})()`);
  assert(afterWheel.s > anchor.s);
  assert(Math.abs(afterWheel.x-anchor.x)<1e-7 && Math.abs(afterWheel.y-anchor.y)<1e-7, 'wheel keeps cursor anchor');
  await evaluate('zoomFit()');
  const original = await evaluate('new XMLSerializer().serializeToString(exportSvg())');
  await evaluate("$('photoVisible').click()");
  assert.equal(await evaluate('new XMLSerializer().serializeToString(exportSvg())'), original);
  await evaluate("$('undo').click()");
  assert.equal(await evaluate('state.photoVisible'), true);
  await evaluate("$('opacity').value=.4; $('opacity').dispatchEvent(new Event('input')); $('opacity').value=.6; $('opacity').dispatchEvent(new Event('input')); $('opacity').dispatchEvent(new Event('change'))");
  await evaluate("$('undo').click()");
  assert.equal(await evaluate('state.opacity'), 1);
  await evaluate("$('redo').click()");
  assert.equal(await evaluate('state.opacity'), .6);
  await evaluate("$('style').value='thin'; $('style').dispatchEvent(new Event('change'))");
  await evaluate("$('undo').click()");
  assert.equal(await evaluate('state.style'), 'comic');
  await clickObject('fret-1'); await evaluate("$('delete').click()");
  assert.equal(await evaluate('future.length'), 0);
  assert.equal(await evaluate('new XMLSerializer().serializeToString(exportSvg()).includes(\'data-object="fret-1"\')'),false);
  await evaluate('undo()');
  const manualIds=await manualTests({evaluate,send,key,panGesture});
  assert.equal(await evaluate(`$('stage').querySelector('svg > g').innerHTML === $('svgbox').querySelector('svg > g').innerHTML && $('stage').querySelector('svg > g').innerHTML === exportSvg().firstChild.innerHTML`), true);
  // Capture downloads without altering the production export path.
  await evaluate(`window.downloads = {}; save = (blob, ext) => { window.downloads[ext] = blob; }; $('download').click(); $('downloadPng').click();`);
  const png = await evaluate(`new Promise((resolve,reject) => { let attempts=0; const timer=setInterval(async () => { if (downloads['.png']) { clearInterval(timer); const reader=new FileReader(); reader.onload=()=>resolve(reader.result); reader.readAsDataURL(downloads['.png']); } else if (++attempts>100) {clearInterval(timer);reject(Error('No PNG'));} },20); })`);
  const svg = await evaluate("downloads['.svg'].text()");
  assert(svg.includes('data-object="fret-1"')); assert(!svg.includes('<image')); assert(!svg.includes('highlight')); assert(!svg.includes('class="hit"'));
  assert(!svg.includes('data-handle='));
  for (const id of manualIds) assert(svg.includes(`data-object="${id}"`));
  assert(svg.includes('data-category="manual"'));
  assert(png.startsWith('data:image/png;base64,'));
  const artifacts = realExample ? 'out/editor-manual-real' : 'out/editor-manual';
  await mkdir(artifacts, {recursive: true});
  await writeFile(`${artifacts}/trace.svg`, svg);
  await writeFile(`${artifacts}/trace.png`, Buffer.from(png.split(',')[1], 'base64'));
  await clickObject('fret-0');
  const screenshot = await send('Page.captureScreenshot', {format:'png'});
  await writeFile(`${artifacts}/overlay.png`, Buffer.from(screenshot.data,'base64'));
  await evaluate(`(() => {const o=state.objects.find(o=>o.id==='${manualIds[0]}'),g=o.primitives[0].geometry;
    const p=photoPoint({x:g.cx,y:g.cy}),q=photoPoint({x:g.cx+1,y:g.cy}),r=$('view').getBoundingClientRect();
    z.s=5/Math.hypot(q.x-p.x,q.y-p.y);z.x=r.width*.5-z.s*p.x;z.y=r.height*.5-z.s*p.y;selected=o.id;render();})()`);
  const headshot=await send('Page.captureScreenshot',{format:'png'});
  await writeFile(`${artifacts}/head-edit.png`,Buffer.from(headshot.data,'base64'));
  await writeFile(`${artifacts}/verification.json`,JSON.stringify({manualIds,automaticObjectsUnchanged:true,undoRedo:true,projection:true,svgPreviewIdentical:true,pngExport:true,noML:true},null,2));
  assert.deepEqual(errors, []);
  console.log('PASS: manual circle/rings/line/rect creation, movement, radius/end/corner handles, stable IDs, full Undo/Redo, cancellation/short clicks, automatic geometry unchanged; selection, Alt overlap cycling, all-tool middle/Space panning, cursor-anchored zoom, SVG/PNG exports, exact preview equality; no browser exceptions.');
  console.log(`Artifacts: ${artifacts}/{trace.svg,trace.png,overlay.png} (${realExample ? 'real guitar, saved SAM replay' : 'synthetic fixture'}, no ML).`);
  async function reopen() {
    await send('Target.closeTarget',{targetId},null);
    ({targetId}=await send('Target.createTarget',{url:'about:blank'},null));
    ({sessionId:session}=await send('Target.attachToTarget',{targetId,flatten:true},null));
    await send('Runtime.enable');await send('Page.enable');
    const load=new Promise(resolve=>socket.addEventListener('message',function listener(event){
      const m=JSON.parse(event.data);if(m.method==='Page.loadEventFired' && m.sessionId===session){socket.removeEventListener('message',listener);resolve();}
    }));
    await send('Page.navigate',{url:new URL(sourceUrl).origin+'/'});await load;
  }
  async function closeWarning() {
    const dialog=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{socket.removeEventListener('message',listener);reject(Error('No beforeunload dialog'));},5000);
      function listener(event) {const m=JSON.parse(event.data);if(m.method==='Page.javascriptDialogOpening' && m.sessionId===session){clearTimeout(timer);socket.removeEventListener('message',listener);resolve(m.params);}}
      socket.addEventListener('message',listener);
    });
    const navigation=send('Page.navigate',{url:'about:blank'});
    const info=await dialog;assert.equal(info.type,'beforeunload');
    await send('Page.handleJavaScriptDialog',{accept:false});await navigation;
  }
  if(process.argv.includes('--auto-test'))await autoTests({evaluate,send,clickObject,panGesture,realExample});
  await transformTests({evaluate,send,key,clickObject,realExample});
  await finishTests({evaluate,send,key,panGesture,realExample});
  await bendTests({evaluate,send,key,panGesture,realExample});
  await projectTests({evaluate,send,key,reopen,closeWarning,realExample});
  await nodesTests({evaluate,send,key});
  assert.deepEqual(errors,[]);
  await send('Browser.close', {}, null);
} finally {
  clearTimeout(timeout); socket?.close(); browser.kill();
  // Only this test's newly created disposable browser profile is removed.
  await rm(profile, {recursive:true, force:true, maxRetries:10, retryDelay:200});
}
