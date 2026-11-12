// node --test tests/editor_project.test.mjs (no npm dependencies)
import '../src/gbcv/web/project.js';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const P=globalThis.GBCVProject;
function sample() {
  const style={fill:'none',stroke:'currentColor','stroke-width':'1.7','stroke-linejoin':'round','stroke-linecap':'round'};
  return {format:P.FORMAT,version:1,document:{version:1,viewBox:'-10 -50 1000 100',objects:[{
    id:'manual-2',category:'manual',type:'circle',order:9,
    primitives:[{type:'circle',geometry:{cx:500,cy:0,r:5},styles:{thin:{...style},comic:{...style},png:{...style}}}]
  }],manualStyles:{thin:{outer:.6,inner:.35},comic:{outer:1.7,inner:.9},png:{outer:3.5,inner:2.2}},
  photo:{width:2000,height:800,projection:[2,0,10,0,2,400,0,0,1]}},
  display:{style:'comic',photoVisible:true,opacity:.6},viewport:{s:.4,x:1,y:20},manualSerial:27};
}
test('lossless, versioned roundtrip and independent clone; no photos/session/history',()=>{
  const value=sample(),parsed=P.parse(P.stringify(value));assert.deepEqual(parsed,value);
  parsed.document.objects[0].primitives[0].geometry.r=7;
  assert.equal(value.document.objects[0].primitives[0].geometry.r,5);
  assert(!P.stringify(value).includes('data:image'));assert(!P.stringify(value).includes('session'));
});
test('automatic SVG path/string geometry retain exact representation',()=>{
  const value=sample(),obj=value.document.objects[0];
  obj.category='hardware';obj.type='path';obj.primitives[0].type='path';obj.primitives[0].geometry={d:'M1.25 -2 L3 4C4 4 5 5 6 6Z'};
  assert.deepEqual(P.parse(P.stringify(value)),value);
});
test('bend groups roundtrip; invalid, overlapping and non-cubic group indices rejected',()=>{
  const value=sample(),obj=value.document.objects[0];obj.type='path';
  const p=obj.primitives[0];p.type='path';p.geometry={d:'M0 0 C10 0 20 5 30 5 C40 5 50 0 60 0 L70 0',bends:[1]};
  assert.deepEqual(P.parse(P.stringify(value)),value);
  for(const bends of [[0],[2],[3],[99],[1,1],[1,2],['1'],[1.5],null,{}]) {
    p.geometry.bends=bends;assert.throws(()=>P.validate(value),JSON.stringify(bends));
  }
});
test('auto IDs roundtrip with the shared allocator; stale counters are rejected',()=>{
  const p=sample(),obj=p.document.objects[0];obj.id='auto-28';obj.category='hardware';obj.type='path';
  obj.primitives[0].type='path';obj.primitives[0].geometry={d:'M0 0L4 0L4 4Z'};
  assert.throws(()=>P.validate(p));p.manualSerial=28;
  assert.deepEqual(P.parse(P.stringify(p)),p);
});
const cases={
  'unknown version':p=>p.version=999,
  'wrong format':p=>p.format='svg',
  'image URL':p=>p.document.photo.src='https://attacker.invalid/photo',
  'raw SVG':p=>p.document.svg='<svg onload="alert(1)">',
  'array instead of record':p=>p.viewport=[],
  'null field':p=>p.document=null,
  'duplicate IDs':p=>p.document.objects.push(structuredClone(p.document.objects[0])),
  'bad ID':p=>p.document.objects[0].id='x" onload="alert(1)',
  'ID allocator collision':p=>p.manualSerial=0,
  'wrong category':p=>p.document.objects[0].category='script',
  'invalid object type':p=>p.document.objects[0].type='script',
  'SVG image primitive':p=>{const o=p.document.objects[0];o.type='image';o.primitives[0].type='image';},
  'event handler attribute':p=>p.document.objects[0].primitives[0].geometry.onload='alert(1)',
  'style attribute':p=>p.document.objects[0].primitives[0].styles.comic.style='background:url(https://attacker.invalid)',
  'CSS URL paint':p=>p.document.objects[0].primitives[0].styles.comic.stroke='url(https://attacker.invalid)',
  'wrong geometry type':p=>p.document.objects[0].primitives[0].geometry.r='5',
  'negative radius':p=>p.document.objects[0].primitives[0].geometry.r=-1,
  'negative order':p=>p.document.objects[0].order=-1,
  'unbounded coordinate':p=>p.document.objects[0].primitives[0].geometry.cx=1e20,
  'nonfinite number':p=>p.viewport.x=Infinity,
  'wrong boolean':p=>p.display.photoVisible='false',
  'invalid opacity':p=>p.display.opacity=1.01,
  'arbitrary style':p=>p.display.style='evil',
  'invalid zoom':p=>p.viewport.s=0,
  'singular projection':p=>p.document.photo.projection=[1,0,0,1,0,0,0,0,1],
  'projection pole':p=>p.document.photo.projection=[1,0,0,0,1,0,-.01,0,1],
  'huge canvas':p=>p.document.photo.width=200000,
  'pixel bomb':p=>{p.document.photo.width=20000;p.document.photo.height=20000;},
  'too many objects':p=>p.document.objects=Array(5001).fill(p.document.objects[0]),
  'manual path mixed with rings':p=>{const o=p.document.objects[0];o.type='compound';o.primitives.push({...o.primitives[0],type:'path',geometry:{d:'M0 0L1 1'}});},
};
for(const [name,mutate] of Object.entries(cases))test(`reject ${name}`,()=>{const p=sample();mutate(p);assert.throws(()=>P.validate(p));});
for(const path of ['M0 0<script/>','M0 0L1e999 2','M0 0C1 2','M0 0ZL1 1','M0 0M1 1','M0 0A5 5 0 0 0 9 9','M0 0 LNaN 2','X'.repeat(200001)]) {
  test('reject bad/oversized path '+path.slice(0,25),()=>{
    const p=sample(),o=p.document.objects[0];o.category='hardware';o.type='path';o.primitives[0].type='path';o.primitives[0].geometry={d:path};
    assert.throws(()=>P.validate(p));
  });
}
test('reject malformed, oversized JSON and prototype keys without pollution',()=>{
  assert.throws(()=>P.parse('{'));
  assert.throws(()=>P.parse(' '.repeat(P.MAX_BYTES+1)));
  const text=P.stringify(sample()).replace('"viewport":{','"viewport":{"__proto__":{"polluted":true},');
  assert.throws(()=>P.parse(text));assert.equal({}.polluted,undefined);
  assert.throws(()=>P.parse(P.stringify(sample()).replace('"viewport":{','"viewport":{"constructor":{},')));
});
test('empty document after deleting everything is a valid project',()=>{
  const p=sample();p.document.objects=[];assert.deepEqual(P.parse(P.stringify(p)),p);
});
test('erased manual line supports disconnected open cubic/line subpaths',()=>{
  const p=sample(),o=p.document.objects[0];o.type='path';o.primitives[0].type='path';
  o.primitives[0].geometry={d:'M0 0L10 0M20 0C21 0 22 1 23 1M40 0L50 0Z'};
  assert.deepEqual(P.parse(P.stringify(p)),p);
});
test('two-ring primitive must stay concentric and have a smaller inner radius',()=>{
  const p=sample(),o=p.document.objects[0];o.type='compound';o.primitives.push(structuredClone(o.primitives[0]));
  o.primitives[1].geometry.r=2;assert.deepEqual(P.validate(p),p);
  o.primitives[1].geometry.cx++;assert.throws(()=>P.validate(p));
});
function png(w,h) {
  const bytes=new Uint8Array(33),v=new DataView(bytes.buffer);
  bytes.set([137,80,78,71,13,10,26,10]);v.setUint32(8,13);bytes.set([73,72,68,82],12);v.setUint32(16,w);v.setUint32(20,h);
  return bytes;
}
test('raster header dimensions checked before decoding',()=>{
  assert.deepEqual(P.rasterInfo(png(2000,800)),{width:2000,height:800,type:'image/png'});
  assert.throws(()=>P.rasterInfo(png(20000,20000)));
  assert.throws(()=>P.rasterInfo(new TextEncoder().encode('<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>')));
  assert.throws(()=>P.rasterInfo(new Uint8Array(P.MAX_PHOTO_BYTES+1)));
});
function webp(canvas,frame,animated=false) {
  const bytes=new Uint8Array(48),v=new DataView(bytes.buffer),ascii=(p,s)=>bytes.set(new TextEncoder().encode(s),p);
  ascii(0,'RIFF');v.setUint32(4,40,true);ascii(8,'WEBP');ascii(12,'VP8X');v.setUint32(16,10,true);
  bytes[20]=animated?2:0;
  for(const [p,n] of [[24,canvas[0]-1],[27,canvas[1]-1]]){bytes[p]=n&255;bytes[p+1]=(n>>8)&255;bytes[p+2]=(n>>16)&255;}
  ascii(30,'VP8 ');v.setUint32(34,10,true);bytes.set([0x9d,1,0x2a],41);v.setUint16(44,frame[0],true);v.setUint16(46,frame[1],true);
  return bytes;
}
test('WebP checks actual frame size as well as advertised canvas, rejects animation',()=>{
  assert.deepEqual(P.rasterInfo(webp([640,480],[640,480])),{width:640,height:480,type:'image/webp'});
  assert.throws(()=>P.rasterInfo(webp([640,480],[16000,16000])));
  assert.throws(()=>P.rasterInfo(webp([640,480],[640,480],true)));
});
test('PNG animation and corrupt chunk lengths rejected',()=>{
  const bytes=new Uint8Array(45);bytes.set(png(64,64));bytes.set(new TextEncoder().encode('acTL'),37);
  assert.throws(()=>P.rasterInfo(bytes));
  const corrupt=png(64,64);new DataView(corrupt.buffer).setUint32(8,0xffffffff);
  assert.throws(()=>P.rasterInfo(corrupt));
});
