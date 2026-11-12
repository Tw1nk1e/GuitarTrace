'use strict';
// Data only: never import SVG markup, CSS, URLs, scripts, or an image from a project.
// Also loaded directly by Node's dependency-free validator tests.
globalThis.GBCVProject = (() => {
  const FORMAT='guitar-blueprint-cv-project', VERSION=2;
  const MAX_BYTES=10*1024*1024, MAX_PHOTO_BYTES=25*1024*1024, MAX_PIXELS=40_000_000;
  const styles=['thin','comic','png'];
  const categories=['silhouette','board','nut','fret','post','key','control','hardware','manual'];
  const numberPattern=/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
  function fail(message) { throw new Error('Invalid project: '+message); }
  function record(value, keys) {
    if (!value || typeof value!=='object' || Array.isArray(value) || Object.getPrototypeOf(value)!==Object.prototype) fail('expected an object');
    const actual=Object.keys(value);
    if (actual.length!==keys.length || actual.some(k=>!keys.includes(k))) fail('unknown or missing fields');
  }
  function finite(value, min=-1e6, max=1e6, string=false) {
    if (string && typeof value==='string' && value.length<=32 && numberPattern.test(value)) value=Number(value);
    if (typeof value!=='number' || !Number.isFinite(value) || value<min || value>max) fail('invalid number');
    return value;
  }
  function integer(value,min,max) { finite(value,min,max);if(!Number.isSafeInteger(value))fail('expected an integer');return value; }
  function choice(value, choices) { if(!choices.includes(value))fail('unsupported value'); }
  function validate(value) {
    record(value,['format','version','document','display','viewport','manualSerial']);
    if(value.format!==FORMAT || ![1,VERSION].includes(value.version))fail('unsupported format or version');
    const doc=value.document;
    record(doc,['version','viewBox','objects','manualStyles','photo',...('joints' in doc?['joints']:[])]);
    if(value.version===1 && 'joints' in doc)fail('connections require version 2');
    if(doc.version!==1)fail('scene version');
    if(typeof doc.viewBox!=='string' || doc.viewBox.length>140)fail('viewBox');
    const box=doc.viewBox.trim().split(/\s+/);
    if(box.length!==4)fail('viewBox');
    const [x,y,w,h]=box.map(n=>finite(n,-1e6,1e6,true));
    if(w<=0 || h<=0)fail('empty viewBox');
    record(doc.photo,['width','height','projection']);
    const photo=doc.photo;
    integer(photo.width,1,20000);integer(photo.height,1,20000);
    if(photo.width*photo.height>MAX_PIXELS)fail('canvas size');
    if(!Array.isArray(photo.projection) || photo.projection.length!==9)fail('projection');
    photo.projection.forEach(n=>finite(n));
    const norm=Math.max(...photo.projection.map(Math.abs));
    const [a,b,c,d,e,f,g,i,j]=photo.projection.map(n=>n/norm);
    const det=a*(e*j-f*i)-b*(d*j-f*g)+c*(d*i-e*g);
    if(!Number.isFinite(det) || Math.abs(det)<1e-14 || Math.abs(j)<1e-12)fail('singular projection');
    const denominators=[[x,y],[x+w,y],[x,y+h],[x+w,y+h]].map(([u,v])=>g*u+i*v+j);
    if(denominators.some(n=>!Number.isFinite(n) || Math.abs(n)<1e-8) || denominators.some(n=>Math.sign(n)!==Math.sign(denominators[0])))fail('projection pole inside the canvas');
    record(doc.manualStyles,styles);
    for(const name of styles) {
      record(doc.manualStyles[name],['outer','inner']);
      finite(doc.manualStyles[name].outer,.001,100);finite(doc.manualStyles[name].inner,.001,100);
    }
    if(!Array.isArray(doc.objects) || doc.objects.length>5000)fail('too many objects');
    const ids=new Set();let primitiveCount=0, pathNumbers=0, maxSerial=0;
    for(const obj of doc.objects) {
      record(obj,['id','type','category','order','primitives',...('strokeWidth' in obj?['strokeWidth']:[])]);
      if('strokeWidth' in obj){if(value.version===1)fail('stroke width requires version 2');finite(obj.strokeWidth,.001,100);}
      if(typeof obj.id!=='string' || !/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(obj.id) || ids.has(obj.id))fail('invalid or duplicate ID');
      ids.add(obj.id);
      const match=obj.id.match(/^(?:manual|auto)-(\d+)$/);if(match)maxSerial=Math.max(maxSerial,Number(match[1]));
      choice(obj.category,categories);integer(obj.order,0,1e9);
      if(!Array.isArray(obj.primitives) || !obj.primitives.length || obj.primitives.length>16)fail('primitives');
      primitiveCount+=obj.primitives.length;if(primitiveCount>20000)fail('too many primitives');
      if(obj.type!==(obj.primitives.length===1?obj.primitives[0]?.type:'compound'))fail('object type');
      for(const p of obj.primitives) {
        record(p,['type','geometry','styles']);choice(p.type,['path','circle','line','rect']);
        const keys={path:['d'],circle:['cx','cy','r'],line:['x1','y1','x2','y2'],rect:['x','y','width','height']}[p.type];
        record(p.geometry,p.type==='path' && 'bends' in p.geometry?['d','bends']:keys);
        if(p.type==='path') {
          const text=p.geometry.d;
          if(typeof text!=='string' || text.length>200000)fail('path size');
          const tokens=text.match(/[MLCZ]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g) || [];
          if(text.replace(/[MLCZ]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g,'').replace(/[\s,]/g,'') || tokens[0]!=='M')fail('path commands');
          let pos=0,active=false,drawn=false;const commands=[];
          while(pos<tokens.length) {
            const command=tokens[pos++];
            commands.push(command);
            if(command==='Z') {if(!active || !drawn)fail('path closure');active=false;continue;}
            if(command==='M') {if(active && !drawn)fail('empty subpath');active=true;drawn=false;}
            else {if(!active)fail('subpath without M');drawn=true;}
            const count={M:2,L:2,C:6}[command];if(!count || pos+count>tokens.length)fail('path coordinates');
            for(let k=0;k<count;k++)finite(tokens[pos++],-1e6,1e6,true);
            pathNumbers+=count;if(pathNumbers>500000)fail('paths are too complex');
          }
          if(tokens.length<3 || !drawn)fail('empty path');
          if('bends' in p.geometry) {
            const spans=p.geometry.bends;
            if(!Array.isArray(spans) || spans.length>commands.length/2)fail('bend groups');
            let previous=-2;
            for(const index of spans) {
              integer(index,1,commands.length-2);
              if(index<=previous+1 || commands[index]!=='C' || commands[index+1]!=='C')fail('bend segments');
              previous=index;
            }
          }
        } else {
          for(const key of keys)finite(p.geometry[key],-1e6,1e6,true);
          if(p.type==='circle' && finite(p.geometry.r,0,1e6,true)<=0)fail('radius');
          if(p.type==='rect' && (finite(p.geometry.width,0,1e6,true)<=0 || finite(p.geometry.height,0,1e6,true)<=0))fail('rectangle size');
        }
        record(p.styles,styles);
        for(const name of styles) {
          const s=p.styles[name];record(s,['fill','stroke','stroke-width','stroke-linejoin','stroke-linecap']);
          if(s.fill!=='none' || s.stroke!=='currentColor')fail('custom paints and references are not allowed');
          finite(s['stroke-width'],.001,100,true);
          choice(s['stroke-linejoin'],['round','miter','bevel']);choice(s['stroke-linecap'],['round','butt','square']);
        }
      }
      if(obj.category==='manual') {
        // Editing code expects numeric geometry, not strings from automatic SVGs.
        for(const p of obj.primitives)if(p.type!=='path')for(const n of Object.values(p.geometry))finite(n);
        if(obj.primitives.length>1) {
          if(obj.primitives.length!==2 || obj.primitives.some(p=>p.type!=='circle'))fail('ring');
          const [a,b]=obj.primitives.map(p=>p.geometry);
          if(a.cx!==b.cx || a.cy!==b.cy || b.r>=a.r)fail('concentric circles');
        }
      }
    }
    if('joints' in doc) {
      if(!Array.isArray(doc.joints) || doc.joints.length>10000)fail('node connections');
      const used=new Set(),byId=new Map(doc.objects.map(o=>[o.id,o])),paths=new Map();let count=0;
      for(const joint of doc.joints){
        if(!Array.isArray(joint) || joint.length<2 || (count+=joint.length)>20000)fail('connection');
        let origin=null;
        for(const r of joint){
          record(r,['object','primitive','subpath','end']);
          const obj=byId.get(r.object);if(!obj)fail('node object');
          integer(r.primitive,0,obj.primitives.length-1);integer(r.subpath,0,100000);choice(r.end,['start','end']);
          const k=JSON.stringify([r.object,r.primitive,r.subpath,r.end]);if(used.has(k))fail('duplicate connection');used.add(k);
          const p=obj.primitives[r.primitive],g=p.geometry;let point;
          if(p.type==='line' && r.subpath===0)point=r.end==='start'?[+g.x1,+g.y1]:[+g.x2,+g.y2];
          else if(p.type==='path'){
            if(!paths.has(p))paths.set(p,(g.d.match(/M[^M]*/g)||[]).map(part=>{
              if(part.includes('Z'))return null;
              const ns=part.match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g).map(Number);return {start:ns.slice(0,2),end:ns.slice(-2)};
            }));
            point=paths.get(p)[r.subpath]?.[r.end];
          }
          if(!point || (origin && Math.hypot(point[0]-origin[0],point[1]-origin[1])>1e-7))fail('connected endpoints do not coincide');
          origin=point;
        }
      }
    }
    integer(value.manualSerial,0,1e9);if(value.manualSerial<maxSerial)fail('ID counter');
    record(value.display,['style','photoVisible','opacity']);choice(value.display.style,styles);
    if(typeof value.display.photoVisible!=='boolean')fail('image visibility');finite(value.display.opacity,0,1);
    record(value.viewport,['s','x','y']);finite(value.viewport.s,.00001,100);finite(value.viewport.x);finite(value.viewport.y);
    return structuredClone(value);
  }
  function parse(text) {
    if(typeof text!=='string' || new TextEncoder().encode(text).length>MAX_BYTES)fail('file exceeds 10 MB');
    let value;try{value=JSON.parse(text);}catch{fail('could not parse JSON');}
    return validate(value);
  }
  function stringify(value) { const text=JSON.stringify(validate(value));if(new TextEncoder().encode(text).length>MAX_BYTES)fail('file exceeds 10 MB');return text; }
  // Read raster dimensions BEFORE decoding. Never pass SVG or arbitrary MIME to a decoder.
  function rasterInfo(bytes) {
    if(!(bytes instanceof Uint8Array) || bytes.length>MAX_PHOTO_BYTES)throw new Error('Source image exceeds 25 MB');
    const b=bytes, v=new DataView(b.buffer,b.byteOffset,b.byteLength);
    let width,height,type;
    const ascii=(p,n)=>String.fromCharCode(...b.subarray(p,p+n));
    if(b.length>=33 && ascii(0,8)==='\x89PNG\r\n\x1a\n' && v.getUint32(8)===13 && ascii(12,4)==='IHDR') {
      type='image/png';width=v.getUint32(16);height=v.getUint32(20);
      let p=8;
      while(p+12<=b.length) {
        const length=v.getUint32(p);if(length>b.length-p-12)throw new Error('Corrupt PNG image');
        if(ascii(p+4,4)==='acTL')throw new Error('Animated source images are not supported');
        p+=length+12;
      }
    } else if(b.length>=4 && b[0]===255 && b[1]===216) {
      type='image/jpeg';let p=2;
      while(p+4<=b.length) {
        if(b[p++]!==255)break;while(p<b.length && b[p]===255)p++;
        const marker=b[p++];if(marker===0xda || marker===0xd9)break;
        if(p+2>b.length)break;
        const length=v.getUint16(p);if(length<2 || p+length>b.length)break;
        if([0xc0,0xc1,0xc2].includes(marker) && length>=8) {height=v.getUint16(p+3);width=v.getUint16(p+5);break;}
        p+=length;
      }
    } else if(b.length>=30 && ascii(0,4)==='RIFF' && ascii(8,4)==='WEBP') {
      type='image/webp';let p=12,canvas=null,frame=null;
      if(v.getUint32(4,true)!==b.length-8)throw new Error('Corrupt WebP image');
      while(p+8<=b.length) {
        const kind=ascii(p,4),size=v.getUint32(p+4,true),start=p+8;
        if(size>b.length-start)throw new Error('Corrupt WebP image');
        if(kind==='ANIM' || kind==='ANMF' || (kind==='VP8X' && (b[start]&2)))throw new Error('Animated source images are not supported');
        if(kind==='VP8X' && size>=10)canvas=[1+b[start+4]+(b[start+5]<<8)+(b[start+6]<<16),1+b[start+7]+(b[start+8]<<8)+(b[start+9]<<16)];
        let dimensions;
        if(kind==='VP8L' && size>=5 && b[start]===0x2f)dimensions=[1+b[start+1]+((b[start+2]&63)<<8),1+(b[start+2]>>6)+(b[start+3]<<2)+((b[start+4]&15)<<10)];
        if(kind==='VP8 ' && size>=10 && ascii(start+3,3)==='\x9d\x01\x2a')dimensions=[v.getUint16(start+6,true)&0x3fff,v.getUint16(start+8,true)&0x3fff];
        if(dimensions) {if(frame)throw new Error('Multiple frames in source image');frame=dimensions;}
        p=start+size+(size&1);
      }
      if(p!==b.length || !frame || (canvas && (canvas[0]!==frame[0] || canvas[1]!==frame[1])))throw new Error('Inconsistent WebP dimensions');
      [width,height]=frame;
    }
    if(!width || !height || width>20000 || height>20000 || width*height>MAX_PIXELS)throw new Error('Expected a PNG, JPEG or WebP image, up to 40 MP');
    return {width,height,type};
  }
  return {FORMAT,VERSION,MAX_BYTES,MAX_PHOTO_BYTES,MAX_PIXELS,validate,parse,stringify,rasterInfo};
})();
