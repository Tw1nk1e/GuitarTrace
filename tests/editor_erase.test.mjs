import '../src/gbcv/web/erase.js';
import {test} from 'node:test';
import assert from 'node:assert/strict';
const E=globalThis.GBCVErase,H=[1,0,0,0,1,0,0,0,1];
const path=d=>({type:'path',geometry:{d},styles:{thin:{'stroke-width':1}}});
const at=(x,y)=>({x,y});
const erase=(p,a,b=a,r=2,h=H)=>E.erasePrimitive(p,a,b,r,h);
const numbers=d=>d.match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g).map(Number);
test('line is cut locally, leaving two open subpaths with original endpoints/styles',()=>{
  const p={...path(''),type:'line',geometry:{x1:0,y1:0,x2:100,y2:0}},q=erase(p,at(50,0));
  assert.equal(q.type,'path');assert.equal(q.styles,p.styles);
  const n=numbers(q.geometry.d);assert.equal(n[0],0);assert.equal(n.at(-2),100);
  assert(Math.abs(n[2]-48)<.04);assert(Math.abs(n[4]-52)<.04);
  assert.equal(q.geometry.d.match(/M/g).length,2);assert(!q.geometry.d.includes('Z'));
});
test('swept disk has no gaps when pointer events are far apart',()=>{
  const p=path('M0 0L100 0'),q=erase(p,at(10,1),at(90,1),2);
  const n=numbers(q.geometry.d);assert(n[2]<10 && n[4]>90);
});
test('untouched primitive/object identities and byte-exact paths are preserved',()=>{
  const p=path('M 0.00 0 L 100.0 0');assert.equal(erase(p,at(50,20)),p);
  const objects=[{id:'a',primitives:[p]}];assert.equal(E.objects(objects,at(50,20),at(50,30),2,H),objects);
});
test('partly cut closed polygon is open and still includes its original closing edge',()=>{
  const q=erase(path('M0 0L100 0L100 100L0 100Z'),at(50,0));
  assert(!q.geometry.d.includes('Z'));assert.equal(q.geometry.d.match(/M/g).length,1);
  assert(q.geometry.d.includes('L0 0'),'original Z edge survives as an explicit line');
});
test('cubic clipping retains exact de Casteljau subcurves rather than polylines',()=>{
  const p=path('M0 0C0 100 100 100 100 0Z'),q=erase(p,at(50,75),at(50,75),5);
  assert(q.geometry.d.includes('C'));assert(!q.geometry.d.includes('Z'));
  // Far from the brush the original curve remains byte-for-byte untouched.
  assert.equal(erase(p,at(50,100),at(50,100),2),p);
  const ps=[at(0,0),at(0,100),at(100,100),at(100,0)];
  const [left,right]=E.split(ps,.5);assert.deepEqual(left.at(-1),at(50,75));assert.deepEqual(right[0],left.at(-1));
});
test('photo-pixel radius uses perspective projection, not viewport zoom',()=>{
  const h=[2,0,10,0,2,5,.001,0,1],x=50,photoX=(2*x+10)/(1+.001*x),photoY=5/(1+.001*x);
  const q=erase(path('M0 0L100 0'),at(photoX,photoY),at(photoX,photoY),4,h),n=numbers(q.geometry.d);
  assert(n[2]>47 && n[2]<49);assert(n[4]>51 && n[4]<53);
});
test('fully erased primitive disappears, compound neighbour stays identical',()=>{
  const a=path('M0 0L1 0'),b=path('M50 0L60 0');
  const source=[{id:'control-0',category:'control',type:'compound',order:4,primitives:[a,b]}];
  const result=E.objects(source,at(0,0),at(1,0),3,H);
  assert.equal(result[0].id,source[0].id);assert.equal(result[0].type,'path');assert.equal(result[0].primitives[0],b);
  assert.equal(E.objects(result,at(50,0),at(60,0),3,H).length,0);
});
test('unsupported circles/rectangles and neighbours are never deleted',()=>{
  for(const type of ['circle','rect']) {const p={type,geometry:{}};assert.equal(erase(p,at(0,0),at(100,100),200),p);}
});
test('multiple subpaths and repeated cuts remain disconnected',()=>{
  const p=path('M0 0L100 0M0 20L100 20Z'),q=erase(p,at(25,0));
  assert(q.geometry.d.includes('M0 20L100 20Z'));
  const next=erase(q,at(75,0));assert.equal(next.geometry.d.match(/M/g).length,4);
});
test('near miss and tangent brush do not remove distant sections',()=>{
  const p=path('M0 0C0 100 100 100 100 0');
  assert.equal(erase(p,at(50,77.1),at(50,77.1),2),p);
});
