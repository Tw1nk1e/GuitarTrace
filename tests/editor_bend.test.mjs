import assert from 'node:assert/strict';
import test from 'node:test';
import '../src/gbcv/web/bend.js';
import '../src/gbcv/web/project.js';
const B=globalThis.GBCVBend;
const near=(a,b)=>assert(Math.hypot(a.x-b.x,a.y-b.y)<1e-7,JSON.stringify({a,b}));
const line={type:'line',geometry:{x1:0,y1:0,x2:120,y2:0},styles:{}};
const curve={type:'path',geometry:{d:'M 0 0 C 30 10 80 -10 120 0 L 140 0'},styles:{}};
function bend(p,index,t,delta) {
  const hit={index,t,point:B.point(B.commands(p)[index].ps,t)};
  const prepared=B.prepare(p,hit);assert(prepared);
  const out=B.deform(p,prepared,delta);assert(out);
  const cs=B.commands(out);
  near(B.point(cs[prepared.index+prepared.side].ps,prepared.t),{x:hit.point.x+delta.x,y:hit.point.y+delta.y});
  const before=prepared.cs,first=prepared.index,last=first+1;
  near(cs[first].ps[0],before[first].ps[0]);near(cs[first].ps[1],before[first].ps[1]);
  near(cs[last].ps[2],before[last].ps[2]);near(cs[last].ps[3],before[last].ps[3]);
  const sub=(a,b)=>({x:a.x-b.x,y:a.y-b.y});
  near(sub(cs[first].ps[3],cs[first].ps[2]),sub(before[first].ps[3],before[first].ps[2]));
  near(sub(cs[last].ps[1],cs[last].ps[0]),sub(before[last].ps[1],before[last].ps[0]));
  return out;
}
test('line: exact grabbed point, pinned endpoints/tangents, repeated bends without nodes',()=>{
  let p=bend(line,1,.35,{x:2,y:20});
  for(let i=0;i<20;i++)p=bend(p,i%2?1:2,.5,{x:0,y:i%2?2:-2});
  assert.equal(B.commands(p).length,3);assert.deepEqual(p.geometry.bends,[1]);
});
test('original cubic: exact subdivision, adjacent segment unchanged; two independent spans',()=>{
  const p=bend(curve,1,.43,{x:0,y:14});
  assert.deepEqual(B.commands(p).at(-1).ps,B.commands(curve).at(-1).ps);
  const q=bend(p,3,.5,{x:0,y:3});assert.deepEqual(q.geometry.bends,[1,3]);
  const r=bend(q,1,.6,{x:0,y:-4});assert.deepEqual(r.geometry.bends,[1,3]);
});
test('safe limits, fixed end grabs, zero drag, screen-space nearest under zoom/perspective',()=>{
  const hit={index:1,t:.5,point:{x:60,y:0}},prepared=B.prepare(line,hit);
  assert.equal(B.deform(line,prepared,{x:0,y:0}),line);
  assert.equal(B.deform(line,prepared,{x:300,y:20}),null);
  assert.equal(B.prepare(line,{...hit,t:0}),null);
  for(const scale of [.1,1,10]) {
    const project=p=>({x:scale*(p.x+12)/(1+p.x*.001),y:scale*(p.y+4)/(1+p.x*.001)});
    const h=B.nearest(line,project({x:37,y:0}),project);assert(Math.abs(h.t-37/120)<1e-6);
  }
});
