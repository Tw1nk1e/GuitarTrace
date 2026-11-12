'use strict';
// Geometry only. The brush is a swept disk in photo pixels, never a paint/mask layer.
// Subdivide only to locate intersections; surviving pieces remain exact cubic Beziers.
globalThis.GBCVErase = (() => {
  const number=/[MLCZ]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g;
  const mix=(a,b,t)=>({x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t});
  function split(points,t) {
    const left=[points[0]],right=[points.at(-1)];let row=points;
    while(row.length>1) {row=row.slice(1).map((p,i)=>mix(row[i],p,t));left.push(row[0]);right.unshift(row.at(-1));}
    return [left,right];
  }
  function piece(points,a,b) {
    if(a===0 && b===1)return points;
    const head=b===1?points:split(points,b)[0];
    return a===0?head:split(head,a/b)[1];
  }
  function distance(p,a,b) {
    const dx=b.x-a.x,dy=b.y-a.y,den=dx*dx+dy*dy;
    const t=den?Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/den)):0;
    return Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy);
  }
  function intervals(points,a,b,r,projection) {
    const H=projection;
    const denominators=points.map(p=>H[6]*p.x+H[7]*p.y+H[8]);
    // Do not attempt a curve crossing the photo homography's pole.
    if(denominators.some(w=>!Number.isFinite(w) || Math.abs(w)<1e-10 || Math.sign(w)!==Math.sign(denominators[0])))return [[0,1]];
    const project=p=>{const w=H[6]*p.x+H[7]*p.y+H[8];return {x:(H[0]*p.x+H[1]*p.y+H[2])/w,y:(H[3]*p.x+H[4]*p.y+H[5])/w};};
    const kept=[];
    function keep(lo,hi) {
      if(kept.length && kept.at(-1)[1]===lo)kept.at(-1)[1]=hi;
      else kept.push([lo,hi]);
    }
    function visit(ps,lo,hi,depth) {
      const q=ps.map(project),xs=q.map(p=>p.x),ys=q.map(p=>p.y);
      const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;
      const centre={x:x+w/2,y:y+h/2},half=Math.hypot(w,h)/2;
      // Projected positive-weight Beziers lie in the convex hull of their controls.
      if(distance(centre,a,b)>r+half) {keep(lo,hi);return;}
      if(q.every(p=>distance(p,a,b)<=r))return;
      if(depth>=24 || half<=.02) {
        if(distance(project(split(ps,.5)[0].at(-1)),a,b)>r)keep(lo,hi);
        return;
      }
      const [left,right]=split(ps,.5),mid=(lo+hi)/2;
      visit(left,lo,mid,depth+1);visit(right,mid,hi,depth+1);
    }
    visit(points,0,1,0);return kept;
  }
  function parse(d) {
    const tokens=d.match(number) || [],paths=[];let i=0,current,start,path;
    while(i<tokens.length) {
      const command=tokens[i++];
      const point=()=>({x:Number(tokens[i++]),y:Number(tokens[i++])});
      if(command==='M') {current=point();start=current;path={start,segments:[],closed:false};paths.push(path);}
      else if(command==='Z') {path.segments.push([current,start]);current=start;path.closed=true;}
      else if(command==='L') {const next=point();path.segments.push([current,next]);current=next;}
      else if(command==='C') {const controls=[point(),point(),point()];path.segments.push([current,...controls]);current=controls[2];}
      else throw Error('Unsupported path command');
    }
    return paths;
  }
  const coords=p=>`${p.x} ${p.y}`;
  const command=ps=>(ps.length===2?'L':'C')+ps.slice(1).map(coords).join(' ');
  const same=(a,b)=>a.x===b.x && a.y===b.y;
  function erasePrimitive(p,a,b,r,projection) {
    if(!['line','path'].includes(p.type))return p;
    const g=p.geometry,paths=p.type==='path'?parse(g.d):[{start:{x:Number(g.x1),y:Number(g.y1)},closed:false,
      segments:[[{x:Number(g.x1),y:Number(g.y1)},{x:Number(g.x2),y:Number(g.y2)}]]}];
    let touched=false;const output=[];
    for(const path of paths) {
      const runs=[];let run=null,cut=false;
      for(const ps of path.segments) {
        const ranges=intervals(ps,a,b,r,projection);
        if(ranges.length!==1 || ranges[0][0]!==0 || ranges[0][1]!==1)cut=true;
        for(const [lo,hi] of ranges) {
          const segment=piece(ps,lo,hi);
          if(!run || lo!==0 || !same(run.at(-1).at(-1),segment[0])) {run=[];runs.push(run);}
          run.push(segment);
          if(hi!==1)run=null;
        }
        if(!ranges.length)run=null;
      }
      touched ||= cut;
      if(!cut) {
        output.push('M'+coords(path.start)+path.segments.slice(0,path.closed?-1:undefined).map(command).join('')+(path.closed?'Z':''));
      } else {
        // Join surviving pieces across the original closed-path seam, without closing the cut.
        if(path.closed && runs.length>1 && same(runs.at(-1).at(-1).at(-1),runs[0][0][0])) {
          const last=runs.pop();runs[0]=[...last,...runs[0]];
        }
        for(const part of runs)output.push('M'+coords(part[0][0])+part.map(command).join(''));
      }
    }
    if(!touched)return p; // Unhit primitives retain even their original string representation.
    const d=output.join('');return d?{...p,type:'path',geometry:{d}}:null;
  }
  function objects(objects,a,b,r,projection) {
    let changed=false;
    const next=objects.flatMap(obj=>{
      const primitives=obj.primitives.map(p=>erasePrimitive(p,a,b,r,projection)).filter(Boolean);
      if(primitives.length===obj.primitives.length && primitives.every((p,i)=>p===obj.primitives[i]))return [obj];
      changed=true;
      return primitives.length?[{...obj,primitives,type:primitives.length===1?primitives[0].type:'compound'}]:[];
    });
    return changed?next:objects;
  }
  return {objects,erasePrimitive,split};
})();
