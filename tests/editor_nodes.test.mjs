import '../src/gbcv/web/bend.js';
import '../src/gbcv/web/nodes.js';
import '../src/gbcv/web/erase.js';
import '../src/gbcv/web/project.js';
import {test} from 'node:test';
import assert from 'node:assert/strict';
const N=GBCVNodes;
const style={fill:'none',stroke:'currentColor','stroke-width':1,'stroke-linejoin':'round','stroke-linecap':'round'};
const styles={thin:style,comic:style,png:style};
const line=(id,x1,y1,x2,y2)=>({id,category:'manual',order:0,type:'line',primitives:[{type:'line',geometry:{x1,y1,x2,y2},styles}]});
const ref=(object,end,subpath=0)=>({object,primitive:0,subpath,end});
const scene=()=>[line('a',0,0,20,0),line('b',30,10,50,10)];
test('explicit end join, shared drag, immutability and independent disconnected/copy ends',()=>{
  const before=scene(),text=JSON.stringify(before),j=N.join(before,[],ref('a','end'),ref('b','start'));
  assert.deepEqual(j.objects[0].primitives[0].geometry,{x1:0,y1:0,x2:25,y2:5});
  const moved=N.moveLinked(j.objects,j.joints,ref('a','end'),{x:21,y:17});
  assert.equal(moved[1].primitives[0].geometry.x1,21);assert.equal(moved[1].primitives[0].geometry.y1,17);
  const independent=N.moveLinked(moved,[],ref('a','end'),{x:9,y:8});
  assert.equal(independent[1].primitives[0].geometry.x1,21);assert.equal(JSON.stringify(before),text);
  assert.equal(N.join(before,[],ref('a','end'),ref('a','end')),null);
});
test('cubic anchors translate adjacent controls, preserve untouched controls and bend metadata',()=>{
  const objects=[{...line('a',0,0,1,1),type:'path',primitives:[{type:'path',geometry:{d:'M0 0 C2 3 7 3 10 0 C12 -3 17 -3 20 0',bends:[1]},styles}]}];
  const result=N.move(objects,ref('a','anchor-1'),{x:13,y:4});
  const cs=GBCVBend.commands(result[0].primitives[0]);
  assert.deepEqual(cs[1].ps,[{x:0,y:0},{x:2,y:3},{x:10,y:7},{x:13,y:4}]);
  assert.deepEqual(cs[2].ps,[{x:13,y:4},{x:15,y:1},{x:17,y:-3},{x:20,y:0}]);
  assert.deepEqual(result[0].primitives[0].geometry.bends,[1]);
  const control=N.move(objects,ref('a','control-1-1'),{x:4,y:6});
  assert.deepEqual(GBCVBend.commands(control[0].primitives[0])[1].ps[1],{x:4,y:6});
});
test('open subpaths supported, closed paths ignored; joining non-ends rejected',()=>{
  const o={...line('a',0,0,1,1),type:'path',primitives:[{type:'path',geometry:{d:'M0 0 L10 0 L20 0 M30 0 C35 0 40 0 45 0 M50 0 L60 0 Z'},styles}]};
  const ps=N.points([o]);assert(!ps.some(p=>p.subpath===2));
  assert.equal(ps.filter(p=>p.endpoint).length,4);
  assert.equal(N.join([o],[],ref('a','anchor-1'),ref('a','start',1)),null);
});
test('Bend splits preserve end refs; eraser rebinds surviving original ends and drops destroyed ones',()=>{
  const j=N.join(scene(),[],ref('a','end'),ref('b','start'));
  const p=j.objects[0].primitives[0],prep=GBCVBend.prepare(p,{index:1,t:.5,point:{x:12.5,y:2.5}});
  const bent={...j.objects[0],type:'path',primitives:[GBCVBend.deform(p,prep,{x:0,y:2})]};
  const objects=[bent,j.objects[1]],moved=N.moveLinked(objects,j.joints,ref('a','end'),{x:27,y:4});
  assert.equal(moved[1].primitives[0].geometry.x1,27);
  const cut=GBCVErase.objects(j.objects,{x:10,y:0},{x:10,y:0},2,[1,0,0,0,1,0,0,0,1]);
  const rebound=N.reconcile(j.objects,cut,j.joints);assert.equal(rebound.length,1);assert.equal(rebound[0][0].subpath,1);
  const erased=GBCVErase.objects(j.objects,{x:25,y:5},{x:25,y:5},3,[1,0,0,0,1,0,0,0,1]);
  assert.deepEqual(N.reconcile(j.objects,erased,j.joints),[]);
});
test('v2 links and per-object width roundtrip; invalid references and v1 extensions rejected',()=>{
  const j=N.join(scene(),[],ref('a','end'),ref('b','start'));j.objects[0].strokeWidth=2.75;
  const value={format:GBCVProject.FORMAT,version:2,document:{version:1,viewBox:'0 0 640 320',...j,
    photo:{width:640,height:320,projection:[1,0,0,0,1,0,0,0,1]},manualStyles:{thin:{outer:1,inner:.5},comic:{outer:2,inner:1},png:{outer:3,inner:2}}},
    display:{style:'comic',photoVisible:false,opacity:1},viewport:{s:1,x:0,y:0},manualSerial:0};
  assert.deepEqual(GBCVProject.parse(GBCVProject.stringify(value)),value);
  for(const mutation of [v=>v.version=1,v=>v.document.joints[0][0].object='missing',v=>v.document.joints[0][0].end='anchor-1',v=>v.document.objects[0].strokeWidth=-1,v=>v.document.objects[0].primitives[0].geometry.x2++]){
    const v=structuredClone(value);mutation(v);assert.throws(()=>GBCVProject.validate(v));
  }
});
