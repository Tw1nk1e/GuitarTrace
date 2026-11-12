"""LP export acceptance on saved SAM masks; no model sessions/inference.

Run after validate_knob_export.py. Fixture indices below are audit assertions,
not production rules. Outputs are confined to out/knob-export.
"""
import json
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

from validate_knob_export import ROOT, DEST
from validate_knobs import tr, knobs, no_ml, load_image, save, render


def replay():
    cache = np.load(ROOT/'out/debug-sam/lp_ref_1/before/sam-cache.npz', allow_pickle=False)

    class Replay:
        def __init__(self, image):
            self.k = float(cache['k'])
            x0, y0, x1, y1 = cache['crop']
            assert image.shape[:2] == (y1-y0, x1-x0)

        def each(self, points, min_score):
            np.testing.assert_allclose(points, cache['points'], atol=1e-8)
            return [(m, float(s), int(i)) for m, s, i in
                    zip(cache['masks'], cache['scores'], cache['prompts']) if s >= min_score]

    with patch('gbcv.sam.Session', Replay), patch('gbcv.segment.probability', no_ml):
        return tr.trace(load_image(ROOT/'samples/dl/lp_ref_1.webp', mode='RGBA'))


def crop(t, images, lo, hi, factor=5):
    xy = t.ruler.to_px(np.array([lo, hi]))
    left, top = np.maximum(0, np.floor(xy.min(0))).astype(int)
    right, bottom = np.minimum([t.photo.shape[1], t.photo.shape[0]], np.ceil(xy.max(0))).astype(int)
    return np.concatenate([cv2.resize(im[top:bottom,left:right], None, fx=factor, fy=factor,
                                      interpolation=cv2.INTER_NEAREST) for im in images], axis=1)


def main():
    folder = DEST/'lp_ref_1'
    folder.mkdir(parents=True, exist_ok=True)
    t = replay()
    assert len(t.parts) == 18 and len(t.knobs) == 3
    after_text = tr.svg(t, 'png')
    labels = t.knobs
    t.knobs = []
    before_text = tr.svg(t, 'png')
    t.knobs = labels
    before, after = ET.fromstring(before_text), ET.fromstring(after_text)
    changed = [i for i,c in enumerate(t.parts) if any(q.contour is c for q in labels)]
    offset = len(before)-len(t.parts)
    unchanged_b = list(before)[:offset]+[e for i,e in enumerate(list(before)[offset:]) if i not in changed]
    unchanged_a = list(after)[:offset]+[e for e in list(after)[offset:] if e.tag.endswith('path')]
    assert len(unchanged_a) == len(unchanged_b)
    assert all(a.tag == b.tag and a.attrib == b.attrib for a,b in zip(unchanged_a, unchanged_b))
    for label,content in [('before',before_text), ('after',after_text)]:
        (folder/f'production-{label}.svg').write_text(content, encoding='utf-8')

    photo = tr.masked_photo(t)
    images = [photo, render(before,t,photo), render(after,t,photo)]
    origin = t.outline.min(0)-4
    x1, y1 = float(t.board[1:3,0].max())+40, float(t.outline[:,1].max())+4
    res = .5
    grid, mask = tr._rect(t.photo,t.mask,t.ruler,origin[0],x1,origin[1],y1,res)
    measurements, cards = [], []
    for control in labels:
        part_index = next(i for i,c in enumerate(t.parts) if c is control.contour)
        x,y,r = knobs.circle_geometry(control)[0]
        # Independent photo-edge sampling (not comparison to a SAM mask alone).
        seed = knobs.Candidate(tuple((np.array([x,y])-origin)/res), (r/res,r/res), 0., 'audit')
        knobs.verify(grid,mask,seed)
        assert seed.reason == 'supported'
        measured = knobs.fit_circle(seed.contour*res+origin)
        assert measured is not None
        mx,my,mr = measured
        centre_error = np.hypot(mx-x,my-y)*t.ruler.d
        radius_error = abs(mr-r)/mr
        assert centre_error < 2.5 and radius_error < .1
        measurements.append(dict(part_index=part_index,centre_photo_px=t.ruler.to_px(np.array([[x,y]]))[0].tolist(),
                                 radius_photo_px=r*t.ruler.d,centre_vs_photo_edge_px=centre_error,
                                 relative_radius_error=radius_error,photo_evidence=seed.reason))
        path = f'control-{part_index}-comparison.png'
        save(folder/path, crop(t,images,[x-r-6,y-r-6],[x+r+6,y+r+6]))
        cards.append(f'<h2>Ручка, контур {part_index}</h2><p>Фото / до / после. Ошибка центра относительно фотограницы: {centre_error:.2f} px; радиуса: {radius_error:.1%}.</p><img src="lp_ref_1/{path}">')

    # Inspect the actual source contour responsible for the bump; preserve it.
    # The photo shows a separate dark protruding pointer with a bright edge,
    # outside the knob rim. It is NOT another tracing of the circular disk.
    small = t.parts[12]
    x,y,r = knobs.circle_geometry(labels[-1])[0]
    radial = np.linalg.norm(small-[x,y],axis=1)/r
    assert radial.min() > .7 and radial.max() > 1.1
    original = list(before)[offset+12]
    assert any(e.tag == original.tag and e.attrib == original.attrib for e in after)
    marked = photo.copy()
    cv2.polylines(marked,[np.round(t.ruler.to_px(small)).astype(np.int32)],True,(0,150,255),1,cv2.LINE_AA)
    save(folder/'pointer-evidence.png',crop(t,[photo,marked],[x-r-7,y-r-7],[x+r+7,y+r+7],6))
    cards.append('<h2>Оставшийся выступ</h2><p>Фото / исходный контур 12 (синий). Видимый отдельный выступ за ободом, похожий на указатель положения. Дубликат ручки не подтверждён; контур сохранён неизменным.</p><img src="lp_ref_1/pointer-evidence.png">')
    result = dict(ml_inference=False,entry='trace.trace -> trace.svg (cached SAM replay)',
                  confirmed_controls=3,body_parts=18,changed_body_indices=changed,
                  unrelated_elements_identical=True,measurements=measurements,
                  nearby_contour=dict(index=12,decision='preserved',reason='visible protruding pointer; not a duplicated knob disk',
                                      radial_range=radial.tolist()))
    (DEST/'production-replay.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (DEST/'final-review.html').write_text('<!doctype html><meta charset="utf-8"><title>Итоговая проверка ручек</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}</style><h1>LP 1: итоговая проверка по фото</h1><p>Без ML; штатный конвейер с воспроизведением сохранённых SAM-масок. Изменены только три подтверждённые ручки. Выступ сохранён как отдельная видимая деталь.</p><p><a href="index.html">Полная галерея: LP, Fender 1/2, Ibanez, SVG и 160/800 px</a> · <a href="production-replay.json">Измерения и проверки</a></p>'+''.join(cards),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
