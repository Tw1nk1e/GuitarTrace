"""Export-only before/after from cached SVG contours, never ML inference.

uv run --no-sync python scripts/validate_knob_export.py
"""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

from validate_knobs import ROOT, NAMES, geometry, load_image, no_ml, save, tr, knobs, render, sample_path

DEST = ROOT / 'out/knob-export'


def text(root):
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    return ET.tostring(root, encoding='unicode')


def preview(root, height):
    """Supersampled diagnostic raster of SVG paths/circles with actual stroke weights."""
    x, y, w, h = map(float, root.attrib['viewBox'].split())
    scale = 4*height/h
    image = np.full((4*height, round(w*scale), 3), 255, np.uint8)
    for e in root:
        if e.tag.endswith('circle'):
            theta = np.linspace(0, 2*np.pi, 256)
            r = float(e.attrib['r'])
            p = np.c_[r*np.cos(theta), r*np.sin(theta)] + [float(e.attrib['cx']),float(e.attrib['cy'])]
            closed = True
        else:
            p = sample_path(e.attrib['d'], step=.15)
            closed = e.attrib['d'].endswith('Z')
        p = np.round((p-[x,y])*scale*256).astype(np.int32)
        cv2.polylines(image, [p], closed, (27,27,27), max(1,round(float(e.attrib['stroke-width'])*scale)), cv2.LINE_AA, shift=8)
    return cv2.resize(image, (round(w*height/h),height), interpolation=cv2.INTER_AREA)


def main():
    manifest = json.loads((ROOT/'out/evaluation/manifest.json').read_text())
    DEST.mkdir(parents=True,exist_ok=True)
    metrics, cards = {}, []
    with patch('gbcv.segment.probability',no_ml), patch('gbcv.sam.Session',no_ml):
        for name in NAMES:
            folder = DEST/name
            folder.mkdir(exist_ok=True)
            record = next(r for r in manifest['records'] if r['id']==name)
            source = ROOT/record['source']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256']
            with patch.object(knobs,'recover',return_value=[]):
                t = geometry(load_image(source,mode='RGBA'),cv2.imread(str(ROOT/'out'/name/'mask.png'),0))
            before_text = (ROOT/'out/knob-recovery'/name/'after.svg').read_text(encoding='utf-8')
            before = ET.fromstring(before_text)
            count = record['cached_body_paths'] + (1 if name=='lp_ref_1' else 0)
            t.parts = [sample_path(e.attrib['d'],step=.4) for e in list(before)[-count:]]
            added = knobs.recover(t,float(t.board[1:3,0].max()))
            assert not added, 'Export audit must not add detections'
            controls = [q for q in t.knobs if knobs.circle_geometry(q)]
            # Use the production exporter, but keep every unrelated cached SVG
            # element verbatim: neither old board geometry nor path re-sampling
            # should masquerade as part of this improvement.
            generated = list(ET.fromstring(tr.svg(t,'png')))
            body_count = sum(len(knobs.circle_geometry(q)) for q in controls) + count-len(controls)
            body = iter(generated[-body_count:])
            after = copy.deepcopy(before)
            for e in list(after)[-count:]:
                after.remove(e)
            changed = []
            for i,(c,old) in enumerate(zip(t.parts,list(before)[-count:])):
                q = next((q for q in controls if q.contour is c),None)
                if q is not None:
                    changed.append(i)
                    for _ in knobs.circle_geometry(q):
                        after.append(next(body))
                else:
                    next(body)
                    after.append(copy.deepcopy(old))
            offset = len(before)-count
            unchanged_before = list(before)[:offset]+[e for i,e in enumerate(list(before)[offset:]) if i not in changed]
            circle_count = sum(len(knobs.circle_geometry(q)) for q in controls)
            unchanged_after = [e for e in after if not e.tag.endswith('circle') or e in list(after)[:offset]]
            # Above removes only new circles (all cached body details are paths).
            assert len(unchanged_before)==len(unchanged_after)
            assert all(a.tag==b.tag and a.attrib==b.attrib for a,b in zip(unchanged_before,unchanged_after))
            if not changed:
                assert text(before)==text(after)
            (folder/'before.svg').write_text(before_text,encoding='utf-8')
            (folder/'after.svg').write_text(text(after) if changed else before_text,encoding='utf-8')
            photo = tr.masked_photo(t)
            save(folder/'before-overlay.png',render(before,t,photo))
            save(folder/'after-overlay.png',render(after,t,photo))
            if controls:
                all_points = np.vstack([q.contour for q in controls])
                lo,hi = all_points.min(0)-35,all_points.max(0)+35
            else:
                x0,y0,x1,y1 = record['crops']['body']
                bounds = t.ruler.to_mm(np.array([[x0,y0],[x1,y1]],float))
                lo,hi = bounds.min(0),bounds.max(0)
            for label,root in [('before',before),('after',after)]:
                zoom = copy.deepcopy(root)
                zoom.set('viewBox',' '.join(map(str,[*lo,*(hi-lo)])))
                (folder/f'{label}-knobs.svg').write_text(text(zoom),encoding='utf-8')
                save(folder/f'{label}-knobs.png',preview(zoom,800))
                for height in (160,800):
                    save(folder/f'{label}-{height}.png',preview(root,height))
            metrics[name] = dict(confirmed_knobs=len(controls),changed_body_indices=changed,new_circles=circle_count,
                                 unrelated_svg_elements_unchanged=True,ml_inference=False,added_detections=0)
            assert len(controls)==(3 if name=='lp_ref_1' else 0)
            cards.append(f'<section><h2>{name}: {len(controls)} подтверждённых ручки</h2><p>Слева до, справа после.</p>'
                         + ''.join(f'<h3>{title}</h3><div class="pair"><img src="{name}/before-{suffix}"><img src="{name}/after-{suffix}"></div>'
                                   for title,suffix in [('Полный SVG','svg'),('Область ручек (если не подтверждены — корпус)','knobs.svg'),('160 px по высоте','160.png'),('800 px по высоте','800.png'),('Наложение на маскированное фото','overlay.png')])
                         .replace('before-svg','before.svg').replace('after-svg','after.svg')+'</section>')
            print(name,metrics[name])
    (DEST/'verification.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Экспорт ручек: до / после</title><style>body{font:16px sans-serif;margin:24px}section{border-top:1px solid #aaa;margin-top:30px}.pair{display:flex;gap:16px;overflow:auto}.pair img{width:48%;object-fit:contain;align-self:start}h3{margin-top:24px}</style><h1>Только SVG-представление подтверждённых ручек</h1><p>Без ML, без новых деталей, без изменения порогов поиска. Исходные контуры — сохранённые SVG предыдущей проверки; все остальные SVG-элементы проверены на точное равенство. PNG — диагностическая растеризация с исходными толщинами; SVG отображает браузер.</p><p>LP 1: 3/4 ручки, все три теперь круги. Четвёртая не добавлена. Рядом с нижней правой ручкой сохранён отдельный маленький шумный контур: он не подтверждён как часть ручки и не удалялся. Fender 1/2 и RGMS8: нет подтверждения ручек, SVG оставлен без изменений. <a href="verification.json">Проверки</a> · <a href="production-replay.json">Replay штатного пути на сохранённых SAM-масках</a></p>'+''.join(cards),encoding='utf-8')


if __name__=='__main__':
    main()
