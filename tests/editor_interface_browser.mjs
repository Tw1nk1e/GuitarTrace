// Layout, context properties and saved real-photo screenshots. Never calls ML/APIs.
// Start tests/editor_auto_fixture_server.py, then: node tests/editor_interface_browser.mjs
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {zoomTests} from './editor_zoom_browser.mjs';

const dir='out/editor-redesign';
await mkdir(dir,{recursive:true});
const profile=await mkdtemp(path.join(process.env.LOCALAPPDATA,'Temp','opencode','editor-ui-'));
const browser=spawn(process.argv[2]||'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank']);
let socket,session,serial=0,stderr='';const waiting=new Map(),errors=[];
const timeout=setTimeout(()=>{console.error('Interface browser test timed out');browser.kill();process.exitCode=1;},90000);
try {
  const endpoint=await new Promise((resolve,reject)=>{
    browser.on('error',reject);browser.stderr.on('data',chunk=>{stderr+=chunk;const m=stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m)resolve(m[1]);});
    browser.on('exit',code=>reject(Error(`Browser exited ${code}: ${stderr}`)));
  });
  socket=new WebSocket(endpoint);await new Promise(resolve=>socket.addEventListener('open',resolve,{once:true}));
  socket.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=waiting.get(m.id);waiting.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);});
  const send=(method,params={},target=session)=>new Promise((resolve,reject)=>{const id=++serial;waiting.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params,...(target?{sessionId:target}:{})}));});
  const {targetId}=await send('Target.createTarget',{url:'about:blank'});
  ({sessionId:session}=await send('Target.attachToTarget',{targetId,flatten:true}));
  await send('Page.enable');await send('Runtime.enable');
  async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
  async function resize(width,height){await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');}
  async function screenshot(name){await writeFile(`${dir}/${name}.png`,Buffer.from((await send('Page.captureScreenshot',{format:'png'})).data,'base64'));}
  async function assertEnglishUI(){
    assert.equal(await evaluate('document.documentElement.lang'),'en');
    assert.equal(await evaluate(`/[А-Яа-яЁё]/.test(document.title+document.body.textContent+[...document.querySelectorAll('[title],[aria-label],[placeholder]')].map(el=>['title','aria-label','placeholder'].map(a=>el.getAttribute(a)||'').join(' ')).join(' '))`),false,'all visible copy, tooltips and accessible labels are English');
  }
  const loaded=new Promise(resolve=>socket.addEventListener('message',function listener(e){const m=JSON.parse(e.data);if(m.method==='Page.loadEventFired' && m.sessionId===session){socket.removeEventListener('message',listener);resolve();}}));
  await send('Page.navigate',{url:process.argv[3]||'http://127.0.0.1:8770/'});await loaded;
  await resize(1366,768);
  assert.equal(await evaluate('state'),null);
  await assertEnglishUI();
  assert.equal(await evaluate("$('drop').querySelector('svg,h1,h2,.eyebrow,.empty-art')"),null,'neutral empty state has no illustration or promotional heading');
  assert.equal(await evaluate("$('select').disabled && $('download').disabled && !$('openProject').disabled"),true);
  await screenshot('empty-1366');
  // Guard both inference and all other server calls while opening the local saved data.
  await evaluate(`window.fetch=()=>{throw Error('Network/ML forbidden in local UI tests');};navigator.sendBeacon=()=>false;`);
  const text=await readFile('out/editor-project-real/guitar.gbcv.json','utf8');
  assert.equal(await evaluate(`openProjectFile(new File([${JSON.stringify(text)}],'guitar.gbcv.json'))`),true);
  assert.equal(await evaluate('isDirty()'),false);
  assert.equal(await evaluate("!$('missingPhoto').hidden"),true);
  await screenshot('no-underlay-1366');
  const photo=await readFile('out/editor-project-real/underlay.png');
  // Test fixture supplies the saved server-result pixels; the app has no photo-attach UI.
  await evaluate(`result.photo='data:image/png;base64,${photo.toString('base64')}';render()`);
  assert.equal(await evaluate("document.querySelector('header,.identity,.brand-mark')===null && $('toggleProperties')===null && $('pan')===null && $('attachPhoto')===null && $('savePhoto')===null"),true);
  assert.equal(await evaluate("/Guitar Archive|GuitarArchive/i.test(document.body.textContent+document.title)"),false);
  await evaluate(`Promise.all([...document.images].map(i=>i.decode()));`);
  await evaluate('zoomFit();savedFingerprint=fingerprint();refresh()');
  await assertEnglishUI();
  assert.equal(await evaluate("(()=>{try{GBCVProject.parse('{');}catch(e){return e.message.startsWith('Invalid project: ') && !/[А-Яа-яЁё]/.test(e.message);}})()"),true,'project validation errors are English');
  const original=await evaluate('JSON.stringify(state.objects)');
  const checks=[];
  for(const [w,h] of [[1920,1080],[1366,768]]){
    await resize(w,h);await evaluate('selected=null;zoomFit();refresh()');
    const layout=await evaluate(`(() => {const ids=['work','view','svgbox','properties','zfit','open','openProject','saveProject','download','downloadPng'];return {overflow:document.documentElement.scrollWidth>innerWidth,rects:Object.fromEntries(ids.map(id=>[id,$(id).getBoundingClientRect().toJSON()]))};})()`);
    assert.equal(layout.overflow,false);
    for(const [id,r] of Object.entries(layout.rects)){assert(r.width>20 && r.height>20,`${id} visible at ${w}`);assert(r.x>=0 && r.right<=w+1 && r.y>=0 && r.bottom<=h+1,`${id} within viewport at ${w}`);}
    assert(layout.rects.view.right<layout.rects.svgbox.x);
    assert(layout.rects.svgbox.right<layout.rects.properties.x);
    assert(layout.rects.view.height>h*.65,'large usable canvas');
    checks.push({width:w,height:h,...layout});
    await screenshot(`document-${w}`);
  }
  await evaluate(`window.uiTestObjectId=state.objects.filter(o=>o.category==='hardware').map(o=>{const b=$('stage').querySelector('g[data-object="'+o.id+'"]').getBBox();return {id:o.id,area:b.width*b.height};}).sort((a,b)=>b.area-a.area)[0].id;selected=uiTestObjectId;render()`);
  assert.equal(await evaluate("$('objectProperties').hidden"),false);
  async function input(id,value){await evaluate(`$('${id}').focus();$('${id}').value=${JSON.stringify(String(value))};$('${id}').dispatchEvent(new Event('input'));`);}
  async function commit(id){await evaluate(`$('${id}').dispatchEvent(new Event('change'));$('${id}').blur()`);}
  const beforeX=await evaluate("Number($('objectX').value)"),n=await evaluate('past.length');
  await input('objectX',beforeX+12);await input('objectX',beforeX+15);await commit('objectX');
  assert.equal(await evaluate('past.length'),n+1,'continuous property edit is one history entry');
  assert.equal(await evaluate('isDirty()'),true);
  await evaluate('undo()');assert.equal(await evaluate('JSON.stringify(state.objects)'),original);
  await evaluate('redo()');assert.notEqual(await evaluate('JSON.stringify(state.objects)'),original);
  await evaluate('undo();selected=uiTestObjectId;render()');
  await input('objectRotation',15);await commit('objectRotation');
  assert.notEqual(await evaluate('JSON.stringify(state.objects)'),original);
  await evaluate('undo();selected=uiTestObjectId;render()');
  await input('objectScale',120);await commit('objectScale');
  assert.notEqual(await evaluate('JSON.stringify(state.objects)'),original);
  await evaluate('undo();selected=uiTestObjectId;render()');
  await input('objectY',120);
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape'});
  assert.equal(await evaluate('JSON.stringify(state.objects)'),original,'Esc cancels a live property edit');
  await input('strokeWidth',2);await commit('strokeWidth');
  const sizeHistory=await evaluate('past.length');
  await evaluate("$('strokePlus').click()");
  assert.equal(await evaluate('state.objects.find(o=>o.id===selected).strokeWidth'),2.25);
  assert.equal(await evaluate('past.length'),sizeHistory+1);
  await evaluate('undo();selected=uiTestObjectId;render()');
  await input('strokeRange',3);await input('strokeRange',4);await commit('strokeRange');
  assert.equal(await evaluate('past.length'),sizeHistory+1,'range gesture is one Undo step');
  assert.equal(await evaluate("+$('strokePreview').firstElementChild.style.strokeWidth"),4);
  await evaluate('undo();selected=uiTestObjectId;render()');
  assert.equal(await evaluate('state.objects.find(o=>o.id===selected).strokeWidth'),2);
  await input('strokeRange',6);
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape'});
  assert.equal(await evaluate('state.objects.find(o=>o.id===selected).strokeWidth'),2,'Esc cancels live width slider');
  const beforeShortcut=await evaluate('JSON.stringify(state.objects)');
  await evaluate("$('objectX').focus()");
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'d',code:'KeyD',modifiers:2});
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Delete',code:'Delete'});
  assert.equal(await evaluate('JSON.stringify(state.objects)'),beforeShortcut,'shortcuts in inputs do not edit the canvas');
  await evaluate('document.activeElement.blur()');
  assert.equal(await evaluate(`$('stage').querySelector('svg > g').innerHTML===$('svgbox').querySelector('svg > g').innerHTML && exportSvg().firstChild.innerHTML===$('svgbox').querySelector('svg > g').innerHTML`),true);
  await evaluate("document.querySelector('.properties-scroll').scrollTop=0");
  await screenshot('selected-dirty-1366');
  await writeFile(`${dir}/trace.svg`,await evaluate('new XMLSerializer().serializeToString(exportSvg())'));
  await resize(1920,1080);await evaluate('zoomFit()');await screenshot('selected-dirty-1920');
  const geometry=await evaluate('JSON.stringify(state.objects)'),viewport=await evaluate('JSON.stringify(z)');
  await evaluate('setSplit(35)');
  assert.equal(await evaluate('JSON.stringify(state.objects)'),geometry);
  assert.equal(await evaluate('JSON.stringify(z)'),viewport,'layout changes do not move photo/overlay coordinates');
  assert.equal(await evaluate("$('properties').getBoundingClientRect().width>200 && !$('documentInfo').hidden"),true,'document settings and object properties stay visible');
  await screenshot('split-1920');
  await evaluate("$('erase').click();$('erasePlus').click()");
  assert.equal(await evaluate("+$('eraseSize').value"),13);
  await input('eraseRange',40);await commit('eraseRange');
  assert.equal(await evaluate("+$('eraseSize').value"),40);
  assert.equal(await evaluate("+$('erasePreview').firstElementChild.getAttribute('r')"),20);
  await evaluate("setSplit(50);zoomFit();document.querySelector('.properties-scroll').scrollTop=0");
  await screenshot('eraser-size-1920');
  await evaluate(`$('select').click();setSplit(50);savedFingerprint=fingerprint();
    window.uiFetchCount=0;window.fetch=()=>{uiFetchCount++;return new Promise(resolve=>window.uiReleaseUpload=resolve);};
    window.uiUpload=upload(new File(['fixture'],'fixture.png',{type:'image/png'}));true;`);
  assert.equal(await evaluate("$('open').disabled && $('objectX').disabled && !$('autoProgress').hidden"),true);
  await screenshot('loading-1920');
  await evaluate(`(async()=>{uiReleaseUpload({ok:false,json:async()=>({detail:'Could not read file. The project is unchanged.'})});await uiUpload;})()`);
  assert.equal(await evaluate('JSON.stringify(state.objects)'),geometry,'upload failure retains the current geometry');
  assert.equal(await evaluate("!busy && $('status').classList.contains('err')"),true);
  await screenshot('error-1920');
  await evaluate("window.downloads={};save=(blob,ext)=>downloads[ext]=blob;saveProject()");
  assert.equal(await evaluate('isDirty()'),false);
  await screenshot('saved-1920');
  await assertEnglishUI();
  assert.equal(await evaluate('uiFetchCount'),1,'only the intercepted upload request, never a real network call');
  await zoomTests({evaluate,send});
  assert.deepEqual(errors,[]);
  await writeFile(`${dir}/verification.json`,JSON.stringify({noML:true,noNetwork:true,checks,properties:{live:true,undoRedo:true,singleHistoryEntry:true,escape:true,shortcuts:true},previewExportSameGeometry:true,states:['empty','no-underlay','ready','selected','dirty','loading','error','saved'],browserExceptions:errors},null,2));
  console.log(`PASS: local real guitar, 1920×1080 / 1366×768, compact toolbar/no branding, permanent separated settings, visual size controls/live history/Esc/shortcut guards, split, all UI states, exact geometry/export parity, no ML/network/browser exceptions. Screenshots: ${dir}`);
  await send('Browser.close',{},null);
} finally {
  clearTimeout(timeout);socket?.close();browser.kill();
  // Only this test's own disposable browser profile.
  await rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});
}
