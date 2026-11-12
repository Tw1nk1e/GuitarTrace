'use strict';
// References address open subpath ends, not command indices: Bend may split a span.
globalThis.GBCVNodes = (() => {
  const key=r=>JSON.stringify([r.object,r.primitive,r.subpath,r.end]);
  function points(objects) {
    const out=[];
    for(const o of objects)o.primitives.forEach((p,primitive)=>{
      const cs=GBCVBend.commands(p);let subpath=-1,run=[];
      const flush=()=>{
        if(run.length<2 || run.at(-1).op==='Z')return;
        run.forEach((c,i)=>out.push({object:o.id,primitive,subpath,end:i===0?'start':i===run.length-1?'end':`anchor-${i}`,
          point:c.ps.at(-1),endpoint:i===0 || i===run.length-1}));
        run.forEach((c,i)=>{if(c.op==='C')for(const k of [1,2])out.push({object:o.id,primitive,subpath,end:`control-${i}-${k}`,point:c.ps[k],endpoint:false});});
      };
      for(const c of cs){if(c.op==='M'){flush();run=[];subpath++;}run.push(c);}flush();
    });
    return out;
  }
  function move(objects,ref,q) {
    return objects.map(o=>{
      if(o.id!==ref.object)return o;
      const primitives=o.primitives.map((p,i)=>{
        if(i!==ref.primitive)return p;
        if(p.type==='line')return {...p,geometry:{...p.geometry,[ref.end==='start'?'x1':'x2']:q.x,[ref.end==='start'?'y1':'y2']:q.y}};
        const cs=GBCVBend.commands(p);let sub=-1,start=-1,finish=-1;
        cs.forEach((c,i)=>{if(c.op==='M')sub++;if(sub===ref.subpath){if(start<0)start=i;finish=i;}});
        if(ref.end.startsWith('control-')){
          const [,index,k]=ref.end.split('-').map((v,i)=>i?Number(v):v);cs[start+index].ps[k]={...q};
          return {...p,geometry:{...p.geometry,d:GBCVBend.path(cs)}};
        }
        const index=ref.end==='start'?start:ref.end==='end'?finish:start+Number(ref.end.slice(7));
        const old=cs[index].ps.at(-1),delta={x:q.x-old.x,y:q.y-old.y};
        cs[index].ps[cs[index].ps.length-1]={...q};
        if(cs[index].op==='C'){const a=cs[index].ps[2];cs[index].ps[2]={x:a.x+delta.x,y:a.y+delta.y};}
        if(cs[index+1] && cs[index+1].op!=='M'){
          cs[index+1].ps[0]={...q};
          if(cs[index+1].op==='C'){const a=cs[index+1].ps[1];cs[index+1].ps[1]={x:a.x+delta.x,y:a.y+delta.y};}
        }
        return {...p,geometry:{...p.geometry,d:GBCVBend.path(cs)}};
      });return {...o,primitives};
    });
  }
  function moveLinked(objects,joints,ref,q) {
    const joint=joints.find(j=>j.some(r=>key(r)===key(ref)));
    for(const r of joint || [ref])objects=move(objects,r,q);
    return objects;
  }
  function join(objects,joints,a,b) {
    const ps=points(objects),pa=ps.find(p=>key(p)===key(a)),pb=ps.find(p=>key(p)===key(b));
    if(!pa?.endpoint || !pb?.endpoint || key(a)===key(b))return null;
    const group=[a,b],rest=[];
    for(const j of joints){if(j.some(r=>key(r)===key(a) || key(r)===key(b)))group.push(...j);else rest.push(j);}
    const refs=[...new Map(group.map(r=>[key(r),{object:r.object,primitive:r.primitive,subpath:r.subpath,end:r.end}])).values()];
    const q={x:(pa.point.x+pb.point.x)/2,y:(pa.point.y+pb.point.y)/2};
    for(const r of refs)objects=move(objects,r,q);
    return {objects,joints:[...rest,refs]};
  }
  // Cutting changes subpaths. Rebind only surviving ORIGINAL endpoints by position.
  function reconcile(before,after,joints) {
    const old=points(before),next=points(after).filter(p=>p.endpoint);
    return joints.map(j=>j.flatMap(r=>{
      const p=old.find(p=>key(p)===key(r));
      const matches=n=>n.object===r.object && Math.hypot(n.point.x-p.point.x,n.point.y-p.point.y)<1e-7;
      const n=p && (next.find(n=>key(n)===key(r) && matches(n)) || next.find(n=>matches(n) && n.end===r.end));
      return n?[{object:n.object,primitive:n.primitive,subpath:n.subpath,end:n.end}]:[];
    })).map(j=>[...new Map(j.map(r=>[key(r),r])).values()]).filter(j=>j.length>1);
  }
  return {key,points,move,moveLinked,join,reconcile};
})();
