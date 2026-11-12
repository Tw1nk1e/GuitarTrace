// Called by editor_browser.mjs on both the synthetic perspective and real LP.
import assert from 'node:assert/strict';

export async function manualTests({evaluate, send, key, panGesture}) {
  const automatic = await evaluate('JSON.stringify(state.objects.filter(o=>o.category!=="manual"))');
  const initial = await evaluate('past.length');
  const ids = [];
  const head = await evaluate(`(() => {const g=state.objects.find(o=>o.category==='post').primitives[0].geometry;
    return {x:Number(g.cx),y:Number(g.cy)};})()`);
  await evaluate(`(() => {const p=photoPoint(${JSON.stringify(head)}),n=photoPoint({x:${head.x+1},y:${head.y}}),r=$('view').getBoundingClientRect();
    z.s=4/Math.hypot(n.x-p.x,n.y-p.y);z.x=r.width*.5-z.s*p.x;z.y=r.height*.5-z.s*p.y;render();})()`);
  const screen = p => evaluate(`(() => {const p=photoPoint(${JSON.stringify(p)}),r=$('view').getBoundingClientRect();return {x:r.left+z.x+z.s*p.x,y:r.top+z.y+z.s*p.y};})()`);
  async function mouse(type, p, button='left') {
    await send('Input.dispatchMouseEvent',{type,button,buttons:type==='mouseReleased'?0:1,clickCount:1,...p});
  }
  async function gesture(a,b) {
    await mouse('mousePressed',await screen(a));
    await mouse('mouseMoved',await screen(b));
    await mouse('mouseReleased',await screen(b));
  }
  async function historyCheck(before, after) {
    await key('keyDown','z','KeyZ',2);
    assert.equal(await evaluate('JSON.stringify(state.objects)'),before);
    await key('keyDown','z','KeyZ',10);
    assert.equal(await evaluate('JSON.stringify(state.objects)'),after);
  }
  async function checkedGesture(a,b) {
    const before=await evaluate('JSON.stringify(state.objects)'), history=await evaluate('past.length');
    await gesture(a,b);
    const after=await evaluate('JSON.stringify(state.objects)');
    assert.equal(await evaluate('past.length'),history+1,'one gesture = one undo step');
    await historyCheck(before,after);
  }
  // Place three rings near real photo posts, then drag to their actual centres.
  const posts=await evaluate(`state.objects.filter(o=>o.category==='post').slice(0,3).map(o=>({x:Number(o.primitives[0].geometry.cx),y:Number(o.primitives[0].geometry.cy)}))`);
  for (let i=0;i<3;i++) {
    const target=posts[i] || {x:head.x-12*i,y:head.y};
    const a={x:target.x-3,y:target.y+2},b={x:a.x+4,y:a.y};
    await evaluate("$('rings').click()");
    await checkedGesture(a,b);
    const id=await evaluate('state.objects.at(-1).id');ids.push(id);
    assert.equal(await evaluate('tool'),'select');
    // The centre is inside the manual hit area, away from radius handles.
    await checkedGesture(a,target);
    let g=await evaluate(`state.objects.find(o=>o.id==='${id}').primitives[0].geometry`);
    assert(Math.abs(g.cx-target.x)<.01 && Math.abs(g.cy-target.y)<.01,`move onto photo post: ${JSON.stringify({g,target,debug:await evaluate('({status:$("status").textContent,selected,tool})')})}`);
    await evaluate(`selected='${id}';render()`);
    await checkedGesture({x:g.cx+g.r,y:g.cy},{x:g.cx+6,y:g.cy});
    await evaluate(`selected='${id}';render()`);
    g=await evaluate(`state.objects.find(o=>o.id==='${id}').primitives[1].geometry`);
    await checkedGesture({x:g.cx,y:g.cy-g.r},{x:g.cx,y:g.cy-1.5});
    const circles=await evaluate(`state.objects.find(o=>o.id==='${id}').primitives.map(p=>p.geometry)`);
    assert.equal(circles[0].cx,circles[1].cx);assert.equal(circles[0].cy,circles[1].cy);
    assert(Math.abs(circles[0].r-6)<.01 && Math.abs(circles[1].r-1.5)<.01);
  }
  for (const kind of ['circle','line','rect']) {
    const a={x:head.x-40,y:head.y+25}, b={x:a.x+12,y:a.y+8};
    await evaluate(`$('${kind}').click()`);
    await checkedGesture(a,b);
    const id=await evaluate('state.objects.at(-1).id');ids.push(id);
    const center=await evaluate(`(() => {const g=state.objects.find(o=>o.id==='${id}').primitives[0].geometry;
      return '${kind}'==='circle'?{x:g.cx,y:g.cy}:'${kind}'==='line'?{x:(g.x1+g.x2)/2,y:(g.y1+g.y2)/2}:{x:g.x+g.width/2,y:g.y+g.height/2};})()`);
    await checkedGesture(center,{x:center.x+2,y:center.y+3});
    await evaluate(`selected='${id}';render()`);
    const handles=await evaluate(`manualHandles(state.objects.find(o=>o.id==='${id}'))`);
    const name=kind==='circle'?'radius':'end', h=handles[name];
    await checkedGesture(h,{x:h.x+3,y:h.y+4});
    if (kind!=='circle') {
      await evaluate(`selected='${id}';render()`);
      const start=await evaluate(`manualHandles(state.objects.find(o=>o.id==='${id}')).start`);
      await checkedGesture(start,{x:start.x-2,y:start.y-3});
    }
    await evaluate(`selected='${id}';render()`);
    const before=await evaluate('JSON.stringify(state.objects)');
    await key('keyDown','Delete','Delete');
    assert.equal(await evaluate(`state.objects.some(o=>o.id==='${id}')`),false);
    const after=await evaluate('JSON.stringify(state.objects)');
    await historyCheck(before,after);
    await evaluate('undo()');
  }
  // Invalid clicks and unfinished/cancelled drawings never enter history.
  const anchor={x:head.x-20,y:head.y+35};
  for (const kind of ['circle','rings','line','rect']) {
    for (const ending of ['short','returnToStart','Escape','blur','pointercancel','lostpointercapture','middle','space']) {
      await evaluate(`$('${kind}').click()`);
      const before=await evaluate('JSON.stringify(state.objects)'),history=await evaluate('past.length');
      const p=await screen(anchor);
      await mouse('mousePressed',p);
      if (ending==='short') await mouse('mouseMoved',{x:p.x+1,y:p.y+1});
      else {
        await mouse('mouseMoved',{x:p.x+20,y:p.y+20});
        if (ending==='returnToStart') await mouse('mouseMoved',p);
        else if (ending==='Escape') await key('keyDown','Escape','Escape');
        else if (ending==='blur') await evaluate("window.dispatchEvent(new Event('blur'))");
        else if (ending==='middle') {
          await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'middle',buttons:5,clickCount:1,x:p.x+20,y:p.y+20});
          await send('Input.dispatchMouseEvent',{type:'mouseMoved',button:'middle',buttons:5,x:p.x+30,y:p.y+25});
          await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'middle',buttons:1,x:p.x+30,y:p.y+25});
        } else if (ending==='space') await key('keyDown',' ','Space');
        else await evaluate(`$('view').dispatchEvent(new PointerEvent('${ending}',{pointerId:edit.pointerId}))`);
      }
      await mouse('mouseReleased',p);
      if (ending==='space') await key('keyUp',' ','Space');
      assert.equal(await evaluate('JSON.stringify(state.objects)'),before,`${kind}: ${ending}`);
      assert.equal(await evaluate('past.length'),history);
      assert.equal(await evaluate('edit'),null);
    }
  }
  // IDs do not get reused even after undo and creation on a new branch.
  await evaluate("$('circle').click()");
  await gesture(anchor,{x:anchor.x+4,y:anchor.y});
  const discarded=await evaluate('selected');
  await evaluate("undo(); $('circle').click()");
  await gesture(anchor,{x:anchor.x+5,y:anchor.y});
  assert.notEqual(await evaluate('selected'),discarded);
  assert.equal(await evaluate('future.length'),0);
  await evaluate('undo()');
  // Re-edit an existing photo-aligned object at another zoom, then roll it back.
  await evaluate(`zoomCenter(1.25);selected='${ids[0]}';render()`);
  const center=await evaluate(`(() => {const g=state.objects.find(o=>o.id==='${ids[0]}').primitives[0].geometry;return {x:g.cx,y:g.cy};})()`);
  await checkedGesture(center,{x:center.x+2,y:center.y+2});
  const moved=await evaluate(`state.objects.find(o=>o.id==='${ids[0]}').primitives[0].geometry`);
  assert(Math.abs(moved.cx-center.x-2)<.01 && Math.abs(moved.cy-center.y-2)<.01,`zoom does not alter geometry deltas: ${JSON.stringify({center,moved})}`);
  await evaluate('undo()');
  const final=await evaluate('JSON.stringify(state.objects)');
  const operations=await evaluate('past.length')-initial;
  for(let i=0;i<operations;i++)await evaluate('undo()');
  assert.equal(await evaluate('JSON.stringify(state.objects)'),automatic);
  for(let i=0;i<operations;i++)await evaluate('redo()');
  assert.equal(await evaluate('JSON.stringify(state.objects)'),final);
  assert.equal(await evaluate('JSON.stringify(state.objects.filter(o=>o.category!=="manual"))'),automatic,'automatic geometry unchanged');
  assert.equal(new Set(ids).size,ids.length);
  for (const style of ['thin','png']) {
    await evaluate(`$('style').value='${style}';$('style').dispatchEvent(new Event('change'))`);
    assert.equal(await evaluate(`Number(document.querySelector('#stage g[data-object="${ids[0]}"] circle').getAttribute('stroke-width'))`),await evaluate(`result.editor.manualStyles.${style}.outer`));
    await evaluate('undo()');
  }
  for(const mode of ['circle','rings','line','rect']) {
    await panGesture('middle',false,mode);
    await panGesture('left',true,mode);
  }
  await evaluate("tool='select';zoomFit();render()");
  assert.equal(await evaluate(`$('stage').querySelector('svg > g').innerHTML === $('svgbox').querySelector('svg > g').innerHTML && $('stage').querySelector('svg > g').innerHTML === exportSvg().firstChild.innerHTML`),true);
  return ids;
}
