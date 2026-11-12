// Real pointer gestures through the existing editor; no inference or dependencies.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

export async function transformTests({evaluate,send,key,clickObject,realExample}) {
  const artifacts=realExample?'out/editor-transform-real':'out/editor-transform';
  await mkdir(artifacts,{recursive:true});
  const screen=p=>evaluate(`(() => {const p=photoPoint(${JSON.stringify(p)}),r=$('view').getBoundingClientRect();
    return {x:r.left+z.x+z.s*p.x,y:r.top+z.y+z.s*p.y};})()`);
  const mouse=(type,p)=>send('Input.dispatchMouseEvent',{type,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1,...p});
  async function gesture(a,b,ending=null) {
    const p=await screen(a),q=await screen(b);
    await mouse('mousePressed',p);
    assert.equal(await evaluate('!!edit'),true,`uses the existing edit gesture: ${JSON.stringify({a,b,p,q,ending,debug:await evaluate('({selected,tool,view:$("view").getBoundingClientRect().toJSON()})')})}`);
    for(const t of [.25,.5,1])await mouse('mouseMoved',{x:p.x+(q.x-p.x)*t,y:p.y+(q.y-p.y)*t});
    if(ending==='returnToStart') {await mouse('mouseMoved',p);await mouse('mouseReleased',p);return;}
    if(ending==='Escape')await key('keyDown','Escape','Escape');
    else if(ending)await evaluate(ending==='blur'?"window.dispatchEvent(new Event('blur'))":`$('view').dispatchEvent(new PointerEvent('${ending}',{pointerId:edit.pointerId}))`);
    await mouse('mouseReleased',q);
  }
  function verify(before,after,anchor,s,dx=0,dy=0) {
    assert.equal(after.id,before.id);assert.equal(after.order,before.order);
    const near=(a,b)=>assert(Math.abs(a-b)<.015,`${a} != ${b}`);
    const x=n=>anchor.x+(Number(n)-anchor.x)*s+dx,y=n=>anchor.y+(Number(n)-anchor.y)*s+dy;
    before.primitives.forEach((p,i)=>{
      const a=p.geometry,b=after.primitives[i].geometry;
      assert.deepEqual(after.primitives[i].styles,p.styles);
      if(p.type==='path') {
        assert.equal(a.d.replace(/[^MLCZ]/g,''),b.d.replace(/[^MLCZ]/g,''));
        const numbers=d=>d.match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g).map(Number);
        const aa=numbers(a.d),bb=numbers(b.d);assert.equal(aa.length,bb.length);
        aa.forEach((n,k)=>near(bb[k],k%2?y(n):x(n)));
      } else {
        const keys=p.type==='circle'?['cx','cy']:p.type==='line'?['x1','y1','x2','y2']:['x','y'];
        keys.forEach((k,j)=>near(b[k],j%2?y(a[k]):x(a[k])));
        for(const k of ['r','width','height'])if(k in a)near(b[k],Number(a[k])*s);
      }
    });
  }
  async function history(before,after,n) {
    assert.equal(await evaluate('past.length'),n+1,'one gesture = one action');
    await key('keyDown','z','KeyZ',2);assert.equal(await evaluate('JSON.stringify(state.objects)'),before);
    await key('keyDown','z','KeyZ',10);assert.equal(await evaluate('JSON.stringify(state.objects)'),after);
  }
  const ids=await evaluate(`['post-0','control-0','key-0','hardware-0',...state.objects.filter(o=>/^auto-/.test(o.id)).map(o=>o.id)]`);
  // Earlier manual regression tests deliberately overlay posts with manual rings.
  // Isolate pointer targets, not document geometry; overlap cycling is tested separately.
  await evaluate("window.transformHitStyle=document.createElement('style');document.head.append(transformHitStyle)");
  for(const [index,id] of ids.entries()) {
    await evaluate(`transformHitStyle.textContent='#stage svg > g:first-child {pointer-events:none} #stage .hit:not([data-object="${id}"]) {pointer-events:none!important}'`);
    // Keep the test object under the pointer at different zooms and pan offsets.
    await evaluate(`tool='select';selected='${id}';render();(() => {
      const h=objectHandles(state.objects.find(o=>o.id===selected)),c={x:(h.nw.x+h.se.x)/2,y:(h.nw.y+h.se.y)/2};
      const p=photoPoint(c),q=photoPoint({x:c.x+1,y:c.y}),r=$('view').getBoundingClientRect();
      const corners=Object.values(h).map(photoPoint),xs=corners.map(p=>p.x),ys=corners.map(p=>p.y);
      z.s=Math.min(${index%2?6:9}/Math.hypot(q.x-p.x,q.y-p.y),r.width*.45/(Math.max(...xs)-Math.min(...xs)),r.height*.45/(Math.max(...ys)-Math.min(...ys)));
      z.x=r.width*.48-z.s*p.x;z.y=r.height*.45-z.s*p.y;selected=null;render();})()`);
    await clickObject(id,true);
    const before=await evaluate('JSON.stringify(state.objects)'),n=await evaluate('past.length');
    const original=await evaluate(`state.objects.find(o=>o.id==='${id}')`);
    const a=await evaluate(`(() => {const el=$('stage').querySelector('.hit[data-object="${id}"]'),p=el.getPointAtLength(el.getTotalLength()*.37);return {x:p.x,y:p.y};})()`);
    await gesture(a,{x:a.x+2,y:a.y-1.5});
    const moved=await evaluate(`state.objects.find(o=>o.id==='${id}')`);
    verify(original,moved,{x:0,y:0},1,2,-1.5);
    const after=await evaluate('JSON.stringify(state.objects)');await history(before,after,n);
    assert.deepEqual(JSON.parse(after).filter(o=>o.id!==id),JSON.parse(before).filter(o=>o.id!==id),'other objects untouched');
    await evaluate(`selected='${id}';render()`);
    const handles=await evaluate(`objectHandles(state.objects.find(o=>o.id==='${id}'))`);
    const h=handles.se,anchor=handles.nw;
    const target={x:anchor.x+(h.x-anchor.x)*1.25,y:anchor.y+(h.y-anchor.y)*1.25};
    const count=await evaluate('past.length');await gesture(h,target);
    const scaled=await evaluate(`state.objects.find(o=>o.id==='${id}')`);
    verify(moved,scaled,anchor,1.25);
    const scaledScene=await evaluate('JSON.stringify(state.objects)');await history(after,scaledScene,count);
    assert.deepEqual(JSON.parse(scaledScene).filter(o=>o.id!==id),JSON.parse(after).filter(o=>o.id!==id));
    // The projected DOM and the exact photo homography agree after editing.
    const projectionError=await evaluate(`(() => {const el=$('stage').querySelector('.hit[data-object="${id}"]'),p=el.getPointAtLength(el.getTotalLength()*.37);
      // getScreenCTM is affine and omits CSS perspective; measure a tiny DOM marker instead.
      const marker=node('circle',{cx:p.x,cy:p.y,r:.0001});$('stage').querySelector('svg').append(marker);
      const dom=marker.getBoundingClientRect(),photo=photoPoint(p),r=$('view').getBoundingClientRect();marker.remove();
      return Math.hypot(dom.x+dom.width/2-r.left-z.x-z.s*photo.x,dom.y+dom.height/2-r.top-z.y-z.s*photo.y);})()`);
    // CSS layout quantizes large projected canvases to subpixels at high zoom.
    assert(projectionError<.5,`photo projection: ${id}, error ${projectionError}px`);
    for(const ending of ['Escape','pointercancel','lostpointercapture','blur','returnToStart']) {
      await evaluate(`selected='${id}';render()`);
      const hs=await evaluate(`objectHandles(state.objects.find(o=>o.id==='${id}'))`);
      await gesture(hs.se,{x:hs.se.x+3,y:hs.se.y+3},ending);
      assert.equal(await evaluate('JSON.stringify(state.objects)'),scaledScene,ending);
      assert.equal(await evaluate('past.length'),count+1);
    }
  }
  await evaluate('transformHitStyle.remove()');
  // Validate and record geometry with the unchanged project-v1 schema.
  const text=await evaluate('GBCVProject.stringify(projectData())');
  await writeFile(`${artifacts}/guitar.gbcv.json`,text);
  const svg=await evaluate('new XMLSerializer().serializeToString(exportSvg())');
  assert(!svg.includes('data-handle'));assert(!svg.includes('highlight'));
  assert.equal(await evaluate(`$('stage').querySelector('svg > g').innerHTML===exportSvg().firstChild.innerHTML && $('svgbox').querySelector('svg > g').innerHTML===exportSvg().firstChild.innerHTML`),true);
  await writeFile(`${artifacts}/trace.svg`,svg);
  await evaluate("downloads={};$('downloadPng').click()");
  const png=await evaluate(`new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{if(downloads['.png']){clearInterval(t);const r=new FileReader();r.onload=()=>resolve(r.result);r.readAsDataURL(downloads['.png']);}else if(++n>200){clearInterval(t);reject(Error('PNG timeout'));}},20);})`);
  await writeFile(`${artifacts}/trace.png`,Buffer.from(png.split(',')[1],'base64'));
  await evaluate('zoomFit()');
  const shot=await send('Page.captureScreenshot',{format:'png'});await writeFile(`${artifacts}/overlay.png`,Buffer.from(shot.data,'base64'));
  await writeFile(`${artifacts}/verification.json`,JSON.stringify({ids,modelInference:false,uniformScale:1.25,svgSpaceTranslation:[2,-1.5],otherObjectsUnchanged:true,compoundGeometry:true,oneGestureOneAction:true,cancellation:true,photoProjection:true,projectV1:true,svgPngExport:true},null,2));
  console.log(`PASS: automatic object drag + proportional corner scale, all primitive coordinates/compound shapes, SVG/photo projection at different zoom/pan, single-step Undo/Redo, cancellation, project v1 and SVG/PNG; ${artifacts}`);
}
