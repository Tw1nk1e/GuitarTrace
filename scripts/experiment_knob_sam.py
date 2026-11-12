"""Exactly one local SAM encoder/decoder experiment, centred on a CV proposal.

uv run --no-sync python scripts/experiment_knob_sam.py
Not a production fallback; no threshold tuning or automatic addition to SVG.
"""

import json
from unittest.mock import patch

import cv2
import numpy as np

from debug_fretboard import ROOT, geometry, load_image, no_ml, save, tr
from gbcv import knobs, sam


def main():
    dest=ROOT/'out/knob-recovery/local-sam'
    if dest.exists():
        raise SystemExit('Experiment already exists; repeat inference is forbidden')
    if not sam.WEIGHTS.exists():
        raise SystemExit('Installed SAM weights missing; no download allowed')
    dest.mkdir(parents=True)
    # ID selects an already measured rejected photo hypothesis for this
    # diagnostic ONLY. No coordinates/IDs are used in the production detector.
    rows=json.loads((ROOT/'out/knob-recovery/lp_ref_1/geometry.json').read_text())
    q=rows[4]
    assert q['source']=='ellipse' and q['reason']=='open_or_weak_outer'
    with patch('gbcv.segment.probability',no_ml),patch.object(knobs,'recover',return_value=[]):
        t=geometry(load_image(ROOT/'samples/dl/lp_ref_1.webp',mode='RGBA'),cv2.imread(str(ROOT/'out/lp_ref_1/mask.png'),0))
    origin=t.outline.min(0)-4
    centre_mm=np.asarray(q['centre'])*.5+origin
    centre=t.ruler.to_px([centre_mm])[0]
    radius_mm=.5*max(q['axes'])
    offsets=np.array([[0,0],[-1.6,0],[1.6,0],[0,-1.6],[0,1.6]])*radius_mm
    points=t.ruler.to_px(centre_mm+offsets)
    bounds=t.ruler.to_px([centre_mm-50,centre_mm+50])
    left,top=np.maximum(0,np.floor(bounds.min(0))).astype(int)
    right,bottom=np.minimum([t.photo.shape[1],t.photo.shape[0]],np.ceil(bounds.max(0))).astype(int)
    crop=t.photo[top:bottom,left:right]
    points-=[left,top]
    session=sam.Session(crop)
    results=session.together(points[:1],negative=points[1:])
    np.savez_compressed(dest/'masks.npz',masks=np.asarray([m for m,_ in results]),
                        scores=np.asarray([s for _,s in results]),points=points,
                        crop=[left,top,right,bottom],k=session.k)
    photo=tr.masked_photo(t)[top:bottom,left:right]
    save(dest/'photo.png',photo)
    rows=[]
    for i,(mask,score) in enumerate(results):
        mask=cv2.resize(mask,(right-left,bottom-top),interpolation=cv2.INTER_NEAREST)
        big=np.zeros_like(t.mask,dtype=bool)
        big[top:bottom,left:right]=mask>0
        contour=tr.part_from_mask(t,big)
        image=photo.copy()
        image[mask>0]=(.6*image[mask>0]+.4*np.array([40,180,240])).astype(np.uint8)
        for j,(x,y) in enumerate(points):
            cv2.drawMarker(image,(round(x),round(y)),(0,200,0) if j==0 else (250,0,180),cv2.MARKER_CROSS,12,2)
        save(dest/f'candidate-{i}.png',image)
        rows.append(dict(id=i,score=float(score),area_px=int(mask.sum()),
                         contour_mm=None if contour is None else contour.tolist()))
    (dest/'results.json').write_text(json.dumps(rows,indent=2))
    print([(r['id'],r['score'],r['area_px']) for r in rows])


if __name__=='__main__':
    main()
