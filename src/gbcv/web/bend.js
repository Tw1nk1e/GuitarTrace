'use strict';
// Two cubic pieces per bent span. Only the three inner controls move together:
// fixed endpoints AND endpoint tangents, smooth join, no new nodes on later bends.
// geometry.bends records first-command indices of these pairs in the ordinary M/L/C/Z path.
globalThis.GBCVBend = (() => {
  const tokens=/[MLCZ]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g;
  const mix=(a,b,t)=>({x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t});
  function split(ps,t) {
    let row=ps;const left=[ps[0]],right=[ps.at(-1)];
    while(row.length>1){row=row.slice(1).map((p,i)=>mix(row[i],p,t));left.push(row[0]);right.unshift(row.at(-1));}
    return [left,right];
  }
  const point=(ps,t)=>split(ps,t)[0].at(-1);
  function commands(p) {
    if(p.type==='line') {const g=p.geometry;return [{op:'M',ps:[{x:+g.x1,y:+g.y1}]},{op:'L',ps:[{x:+g.x1,y:+g.y1},{x:+g.x2,y:+g.y2}]}];}
    if(p.type!=='path')return [];
    const ts=p.geometry.d.match(tokens)||[],out=[];let i=0,current,start;
    const read=()=>({x:+ts[i++],y:+ts[i++]});
    while(i<ts.length) {
      const op=ts[i++];let ps;
      if(op==='M'){current=read();start=current;ps=[current];}
      else if(op==='Z'){ps=[current,start];current=start;}
      else {ps=[current,...Array.from({length:op==='C'?3:1},read)];current=ps.at(-1);}
      out.push({op,ps});
    }
    return out;
  }
  const cubic=ps=>ps.length===4?ps:[ps[0],mix(ps[0],ps[1],1/3),mix(ps[0],ps[1],2/3),ps[1]];
  const smooth=t=>t*t*(3-2*t);
  function path(cs) {
    const xy=p=>`${p.x} ${p.y}`;
    return cs.map(c=>c.op==='Z'?'Z':c.op+(c.op==='M'?c.ps:c.ps.slice(1)).map(xy).join(' ')).join(' ');
  }
  // Nearest point is measured in SCREEN coordinates, including perspective and zoom.
  function nearest(p,q,project) {
    let best=null;const cs=commands(p);
    cs.forEach((c,index)=>{
      if(c.op==='M' || c.op==='Z')return;
      const bounds=c.ps.map(project);
      if(bounds.every(p=>Number.isFinite(p.x) && Number.isFinite(p.y)) &&
        (q.x<Math.min(...bounds.map(p=>p.x))-10 || q.x>Math.max(...bounds.map(p=>p.x))+10 ||
         q.y<Math.min(...bounds.map(p=>p.y))-10 || q.y>Math.max(...bounds.map(p=>p.y))+10))return;
      const distance=t=>{const a=project(point(c.ps,t));return (a.x-q.x)**2+(a.y-q.y)**2;};
      let t=0,d=Infinity;
      for(let k=0;k<=48;k++){const v=k/48,n=distance(v);if(n<d){d=n;t=v;}}
      let lo=Math.max(0,t-1/48),hi=Math.min(1,t+1/48);
      for(let k=0;k<24;k++){const a=lo+(hi-lo)/3,b=hi-(hi-lo)/3;if(distance(a)<distance(b))hi=b;else lo=a;}
      t=(lo+hi)/2;d=distance(t);
      if(!best || d<best.distance)best={index,t,distance:d,point:point(c.ps,t)};
    });
    return best;
  }
  function prepare(p,hit) {
    const cs=commands(p),groups=p.geometry.bends||[];
    const group=groups.find(i=>hit.index===i || hit.index===i+1);
    if(group!==undefined) {
      const side=hit.index===group?0:1,weight=side?1-smooth(hit.t):smooth(hit.t);
      if(weight<.025)return null; // Fixed ends aren't bend handles.
      return {cs,groups,index:group,side,t:hit.t,weight,origin:hit.point};
    }
    if(hit.t<.03 || hit.t>.97)return null;
    const [a,b]=split(cubic(cs[hit.index].ps),hit.t);
    cs.splice(hit.index,1,{op:'C',ps:a},{op:'C',ps:b});
    return {cs,groups:[...groups.map(i=>i>hit.index?i+1:i),hit.index].sort((a,b)=>a-b),index:hit.index,
      side:0,t:1,weight:1,origin:hit.point};
  }
  function safe(before,after) {
    const a=before[0][0],b=before[1][3],dx=b.x-a.x,dy=b.y-a.y,len=dx*dx+dy*dy;
    if(len<1e-12)return false;
    // If the source runs monotonically along its chord, retain that property.
    // It proves the new span has no reversal/loop, including an angled drag.
    const monotone=pair=>pair.every(ps=>ps.slice(1).every((p,i)=>(p.x-ps[i].x)*dx+(p.y-ps[i].y)*dy>=-1e-9*len));
    if(monotone(before) && !monotone(after))return false;
    // Conservative sampled intersection guard for non-monotone original cubics.
    const samples=[...Array.from({length:49},(_,i)=>point(after[0],i/48)),...Array.from({length:48},(_,i)=>point(after[1],(i+1)/48))];
    const cross=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
    for(let i=1;i<samples.length;i++)for(let j=i+2;j<samples.length;j++) {
      const a=samples[i-1],b=samples[i],c=samples[j-1],d=samples[j];
      if(cross(a,b,c)*cross(a,b,d)<0 && cross(c,d,a)*cross(c,d,b)<0)return false;
    }
    return after.flat().every(p=>Number.isFinite(p.x) && Number.isFinite(p.y) && Math.abs(p.x)<=1e6 && Math.abs(p.y)<=1e6);
  }
  function deform(p,prepared,delta) {
    if(!delta.x && !delta.y)return p;
    const {index,weight}=prepared,cs=structuredClone(prepared.cs);
    const shift=p=>({x:p.x+delta.x/weight,y:p.y+delta.y/weight});
    cs[index].ps[2]=shift(cs[index].ps[2]);cs[index].ps[3]=shift(cs[index].ps[3]);
    cs[index+1].ps[0]=cs[index].ps[3];cs[index+1].ps[1]=shift(cs[index+1].ps[1]);
    if(!safe([prepared.cs[index].ps,prepared.cs[index+1].ps],[cs[index].ps,cs[index+1].ps]))return null;
    return {...p,type:'path',geometry:{d:path(cs),bends:prepared.groups}};
  }
  function highlight(p,hit) {
    const cs=commands(p),group=(p.geometry.bends||[]).find(i=>hit.index===i || hit.index===i+1);
    const parts=group===undefined?[cs[hit.index]]:cs.slice(group,group+2);
    return path([{op:'M',ps:[parts[0].ps[0]]},...parts]);
  }
  return {commands,path,point,nearest,prepare,deform,highlight};
})();
