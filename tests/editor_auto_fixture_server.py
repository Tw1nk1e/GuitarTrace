"""Browser adapter fixture: replaces sam_at, never runs SAM or BiRefNet.

uv run python tests/editor_auto_fixture_server.py [--real-cache]
Real mode uses the existing LP photo + saved SAM mask; not a fresh prediction.
"""
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import uvicorn
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles

from editor_sample import sample
from gbcv import server, trace as tr
from gbcv.editor import added_part

ROOT = Path(__file__).resolve().parents[1]
comparison = ROOT / 'out/quality-fixes-2026-11-12'
if comparison.exists():
    server.app.mount('/quality-comparison', StaticFiles(directory=comparison), name='quality-comparison')
sys.path.insert(0, str(ROOT/'scripts'))
real = '--real-cache' in sys.argv
control = {'mode':'good','calls':0,'active':0,'max_active':0}
release = threading.Event()
saved_mask = None


def forbidden(*args, **kwargs):
    raise RuntimeError('ML is forbidden in the browser adapter fixture')


def mask_for(t):
    if saved_mask is not None:
        return saved_mask
    mask = np.zeros(t.mask.shape, np.uint8)
    cv2.ellipse(mask, (260,160), (22,12), 20, 0, 360, 1, -1)
    return mask.astype(bool)


def fake_sam(t, x, y):
    control['calls'] += 1
    control['active'] += 1
    control['max_active'] = max(control['max_active'], control['active'])
    try:
        if control['mode'] == 'slow':
            if not release.wait(10):
                raise RuntimeError('Fixture timeout')
        if control['mode'] == 'none':
            return []
        if control['mode'] == 'unavailable':
            raise RuntimeError('SAM disabled')
        return [(mask_for(t), .95)]
    finally:
        control['active'] -= 1


@server.app.get('/test-auto')
def metadata():
    t = next(iter(server.sessions.values())).trace
    mask = mask_for(t)
    yy, xx = np.nonzero(mask & (t.mask>0))
    k = np.argmin((xx-np.median(xx))**2+(yy-np.median(yy))**2)
    h,w = t.mask.shape
    return {'x':float(xx[k]/w),'y':float(yy[k]/h),
            'object':added_part(t,tr.part_from_mask(t,mask)),
            'sam_enabled':next(iter(server.sessions.values())).sam_enabled,**control}


@server.app.post('/test-auto/mode/{mode}')
def mode(mode: str):
    if mode not in ('good','none','unavailable','slow'):
        raise HTTPException(400)
    control['mode'] = mode
    if mode == 'slow':
        release.clear()
    else:
        release.set()
    # Fixture-only: failures and a slow uncached click must reach the fake model.
    for s in server.sessions.values():
        s.auto_cache.clear()
    return control


if __name__ == '__main__':
    with patch('gbcv.sam.Session',forbidden), patch('gbcv.segment.probability',forbidden), patch.object(tr,'sam_at',fake_sam):
        if real:
            from audit_knob_export import replay
            t = replay()
            with np.load(ROOT/'out/debug-sam/lp_ref_1/before/sam-cache.npz',allow_pickle=False) as cache:
                x0,y0,x1,y1 = cache['crop'].astype(int)
                # Fixture-only: #155 is the preserved stop-bar mask, marked 'kept'
                # in before/candidates.json. Do not substitute a rejected shadow/blob.
                assert float(cache['scores'][155]) >= .7
                mask = cache['masks'][155]
                big = np.zeros(t.mask.shape,bool)
                big[y0:y1,x0:x1] = cv2.resize(mask.astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)>0
                if tr.part_from_mask(t,big) is not None:
                    saved_mask = big
            assert saved_mask is not None, 'No suitable saved mask; refusing to run a model'
            server.sessions['auto-cache'] = server.Session(t,time.time())
            print('Saved LP mask adapter (NO ML): http://127.0.0.1:8771/?session=auto-cache',flush=True)
            uvicorn.run(server.app,host='127.0.0.1',port=8771,log_level='warning')
        else:
            with patch.object(tr,'trace',side_effect=lambda rgba:sample(.0005)):
                uvicorn.run(server.app,host='127.0.0.1',port=8770,log_level='warning')
