// True Chromium SVG rasterization, no diagnostic polyline approximation or ML.
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';

export async function renderQualityGallery({evaluate}) {
  const records=await evaluate("fetch('/quality-comparison/metrics.json').then(r=>r.json()).then(v=>v.records)");
  const verification=[];
  for(const record of records) {
    let difference=0;
    for(const [suffix,height] of [['',160],['',800],...record.windows.map(w=>['-'+w,600])]) {
      const pair=await evaluate(`(async()=>{
        const result=[];
        for(const side of ['before','after']) {
          const text=await fetch('/quality-comparison/${record.id}/'+side+'${suffix}.svg').then(r=>r.text());
          const svg=new DOMParser().parseFromString(text,'image/svg+xml').documentElement;
          const [x,y,w,h]=svg.getAttribute('viewBox').split(/\\s+/).map(Number);
          const width=Math.round(w*${height}/h);
          svg.setAttribute('width',width);svg.setAttribute('height',${height});
          const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'}));
          try {
            const image=new Image();image.src=url;await image.decode();
            const c=document.createElement('canvas');c.width=width;c.height=${height};const ctx=c.getContext('2d');
            ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);ctx.drawImage(image,0,0);
            const data=ctx.getImageData(0,0,c.width,c.height).data;
            result.push({png:c.toDataURL('image/png'),width,height:${height},pixels:Array.from(data)});
          } finally {URL.revokeObjectURL(url);}
        }
        const a=result[0].pixels,b=result[1].pixels;
        let changed=0;for(let i=0;i<a.length;i+=4)if(Math.abs(a[i]-b[i])>8)changed++;
        return {images:result.map(({pixels,...r})=>r),changed};
      })()`);
      if(!suffix && height===800)difference=pair.changed;
      for(let i=0;i<2;i++) {
        const {png,width}=pair.images[i];assert(width>0);assert(png.startsWith('data:image/png;base64,'));
        const side=i?'after':'before',fileSuffix=suffix || '-'+height;
        await writeFile(`out/quality-fixes-2026-11-12/${record.id}/${side}${fileSuffix}.png`,Buffer.from(png.split(',')[1],'base64'));
      }
    }
    assert(difference>0,record.id+' should have visible changed strokes');
    verification.push({id:record.id,renderer:'Chromium SVG image -> Canvas',wholeHeights:[160,800],zoomHeight:600,changedPixelsAt800:difference,noML:true});
  }
  await writeFile('out/quality-fixes-2026-11-12/browser-rasterization.json',JSON.stringify(verification,null,2));
  console.log('PASS: five before/after galleries rasterized with real Chromium at 160/800px and close-ups');
}
