// High-zoom rendering / hit-target regression. Uses saved pixels only, never ML.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

export async function zoomTests({evaluate,send}) {
  const dir='out/editor-zoom';await mkdir(dir,{recursive:true});
  await evaluate(`window.zoomSaved={result,state,past,future,selected,tool,z:{...z},savedFingerprint,manualSerial,nodeSelection};`);
  const shot=async name=>{const {data}=await send('Page.captureScreenshot',{format:'png'});await writeFile(dir+'/'+name+'.png',Buffer.from(data,'base64'));return data;};
  const screen=p=>evaluate(`(() => {const q=photoPoint(${JSON.stringify(p)}),r=$('view').getBoundingClientRect();return {x:r.left+z.x+z.s*q.x,y:r.top+z.y+z.s*q.y};})()`);
  const mouse=(type,p)=>send('Input.dispatchMouseEvent',{type,button:'left',buttons:type==='mouseReleased'?0:1,clickCount:1,...p});
  const click=async p=>{const q=await screen(p);await mouse('mousePressed',q);await mouse('mouseReleased',q);};
  // Recreate a low-resolution photo from the already attached, sanitized underlay.
  if(await evaluate('!!result.photo')){
    await evaluate(`(async()=>{
      const img=$('stage').querySelector('img');await img.decode();const ratio=1126/result.editor.photo.width;
      const c=document.createElement('canvas');c.width=1126;c.height=Math.round(result.editor.photo.height*ratio);c.getContext('2d').drawImage(img,0,0,c.width,c.height);
      const H=result.editor.photo.projection.map((v,i)=>i<6?v*ratio:v);
      result={...result,photo:c.toDataURL(),editor:{...result.editor,photo:{width:c.width,height:c.height,projection:H}}};
      selected=state.objects.find(o=>o.category==='key').id;tool='select';render();
      const b=selectedBox(),p=photoPoint({x:b.x+b.width/2,y:b.y+b.height/2}),r=$('view').getBoundingClientRect();z.s=18.26;z.x=r.width/2-z.s*p.x;z.y=r.height/2-z.s*p.y;render();
      await Promise.all([...document.images].map(i=>i.decode()));
    })()`);
    await shot('low-resolution-1826');
    await evaluate(`selected=state.objects.find(o=>o.category==='fret').id;tool='nodes';nodeSelection=[];render();{const point=GBCVNodes.points(state.objects.filter(o=>o.id===selected))[0].point,p=photoPoint(point),v=$('view');z.x=v.clientWidth/2-z.s*p.x;z.y=v.clientHeight/2-z.s*p.y;render();}`);
    const hover=await evaluate("(() => {const r=$('view').getBoundingClientRect();return {x:r.left+r.width/2+10,y:r.top+r.height/2};})()");
    await mouse('mouseMoved',hover);
    await shot('low-resolution-nodes-1826');
  }
  const checks=[];
  for(const perspective of [false,true]){
    await evaluate(`(() => {
      const objects=[makeManual('line',{x:20,y:20},{x:28,y:20},'manual-980'),makeManual('line',{x:20,y:21.2},{x:28,y:21.2},'manual-981')];
      for(const o of objects)for(const p of o.primitives)for(const s of Object.values(p.styles))s['stroke-width']=.15;
      const editor={...result.editor,viewBox:'0 0 1126 751',objects,photo:{width:1126,height:751,projection:${perspective?'[1,.1,0,.02,1,0,.0004,.0002,1]':'[1,0,0,0,1,0,0,0,1]'}}};
      loadResult({id:null,photo:null,editor,stats:{seconds:0}});manualSerial=990;
    })()`);
    for(const scale of [8,18.26,20]){
      await evaluate(`(() => {const p=photoPoint({x:24,y:20.6}),v=$('view');z.s=${scale};z.x=v.clientWidth/2-z.s*p.x;z.y=v.clientHeight/2-z.s*p.y;selected=null;tool='select';render();})()`);
      const geometry=await evaluate('JSON.stringify(state.objects)');
      await click({x:24,y:20});assert.equal(await evaluate('selected'),'manual-980','nearby later-painted line must not steal the click');
      await click({x:24,y:21.2});assert.equal(await evaluate('selected'),'manual-981');
      if(scale>=18){await click({x:24,y:20.6});assert.equal(await evaluate('selected'),null,'gap between lines is not an enormous hidden hit target');}
      assert.equal(await evaluate('JSON.stringify(state.objects)'),geometry);
      await evaluate("selected='manual-980';tool='nodes';nodeSelection=[GBCVNodes.points(state.objects.filter(o=>o.id==='manual-981'))[0]];render()");
      const p=await screen({x:20,y:20});
      assert.equal(await evaluate(`document.elementFromPoint(${p.x},${p.y})?.hasAttribute('data-node')`),true,'node centre matches visible geometry');
      assert(await evaluate("$('stage').querySelector('.handle-mark').getBoundingClientRect().width")>=14,'visible node is comfortably sized at every zoom');
      const other=await screen({x:20,y:21.2});
      const overlap={x:p.x+(other.x-p.x)*.43,y:p.y+(other.y-p.y)*.43};
      await mouse('mousePressed',overlap);
      assert.equal(await evaluate('edit?.nodeRef?.object'),'manual-980','overlapping acquisition regions choose the nearer node, not the last painted one');
      await mouse('mouseReleased',overlap);
      // Start 12px from the node centre (outside its visible dot), in its 28px target.
      // Preserve that offset while dragging: the geometry must not jump to the click.
      const grab={x:p.x-11,y:p.y-5},target=await screen({x:20.6,y:19.6}),n=await evaluate('past.length');
      const release={x:target.x-11,y:target.y-5};
      await mouse('mousePressed',grab);assert.equal(await evaluate("edit?.nodeRef?.object"),'manual-980','nearest node wins, independently of paint order');
      await mouse('mouseMoved',release);await mouse('mouseReleased',release);
      assert.equal(await evaluate('past.length'),n+1);
      const moved=await evaluate('state.objects[0].primitives[0].geometry');
      assert(Math.abs(moved.x1-20.6)<.001 && Math.abs(moved.y1-19.6)<.001,'off-centre acquisition preserves grab offset');
      assert.equal(await evaluate("JSON.stringify(state.objects[1])"),JSON.stringify(JSON.parse(geometry)[1]));
      await evaluate('undo()');assert.equal(await evaluate('JSON.stringify(state.objects)'),geometry);
      await evaluate("selected='manual-980';tool='select';render()");
      const png=await shot(`${perspective?'perspective':'affine'}-${scale}`);
      // Away from the central rotation stem and endpoint handles.
      const middle=await screen({x:22,y:20});
      const thickness=await evaluate(`(async()=>{const img=new Image();img.src='data:image/png;base64,${png}';await img.decode();const c=document.createElement('canvas');c.width=img.width;c.height=img.height;const ctx=c.getContext('2d');ctx.drawImage(img,0,0);const x=Math.round(${middle.x}),y=Math.round(${middle.y}),data=ctx.getImageData(x,y-40,1,81).data;let orange=0;for(let i=0;i<data.length;i+=4)if(data[i]>170 && data[i+1]>40 && data[i+1]<160 && data[i+2]<100)orange++;return orange;})()`);
      assert(thickness>=1 && thickness<=5,`highlight stays sharp and ~2 screen px, not CSS-magnified: ${thickness}px`);
      const layout=await evaluate('overlayViewport()');
      assert(layout.width*layout.scale<3000 && layout.height*layout.scale<3000,'native raster surface is viewport bounded, not the entire magnified document');
      checks.push({perspective,scale,highlightPixels:thickness,viewport:layout});
      // Pan without a full render: the cropped SVG/photo mapping must stay aligned.
      await evaluate('z.x+=17;z.y-=11;apply()');
      await click({x:24,y:20});assert.equal(await evaluate('selected'),'manual-980');
    }
  }
  await evaluate(`({result,state,past,future,selected,tool,savedFingerprint,manualSerial,nodeSelection}=zoomSaved);Object.assign(z,zoomSaved.z);render();`);
  await writeFile(dir+'/verification.json',JSON.stringify({noML:true,checks,selection:true,nodeDragging:true,offCentreGrab:true,nearestNode:true,hitDiameterPx:28,undo:true,panAlignment:true},null,2));
  console.log('PASS: low-resolution / 800%, 1826%, 2000%; affine + perspective; sharp 2px highlights, bounded hit targets, accurate node drag + Undo, pan alignment, viewport-sized SVG texture. '+dir);
}
