"""Export-only neck join audit. Saved LP replay + saved user8 SVG; no ML.

uv run python scripts/audit_editor_neck.py
The user8 artifact is historical, not assumed to be the user's latest screenshot.
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2

from audit_knob_export import replay
from validate_knobs import render, save
from gbcv import trace as tr

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'out/editor-neck-join'


def values(text):
    return np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', text)])


def cached_correction(path, board):
    """Apply ONLY the production-changed handles, preserving all other cached bytes."""
    chunks = re.findall(r'C([^CZ]+)', path)
    start = values(path.split('C')[0])[0:2]
    cubic = np.array([values(s).reshape(3,2) for s in chunks])
    nodes = np.vstack([start, cubic[:-1,2]])
    old = tr.curve_path(nodes, 0)
    new = tr.curve_path(nodes, 0, neck_join=board)
    old_c = np.array([values(s).reshape(3,2) for s in re.findall(r'C([^CZ]+)', old)])
    new_c = np.array([values(s).reshape(3,2) for s in re.findall(r'C([^CZ]+)', new)])
    np.testing.assert_array_equal(old_c[:,2], cubic[:,2])
    changed = np.any(new_c[:,:2] != old_c[:,:2], axis=2)
    replacements = list(chunks)
    for j in range(len(chunks)):
        if not changed[j].any():
            continue
        pairs = cubic[j].copy()
        for k in range(2):
            if changed[j,k]:
                pairs[k] = new_c[j,k]
        replacements[j] = ' '.join(tr._f(v) for v in pairs.ravel())
    return path.split('C')[0] + ''.join('C'+s for s in replacements) + 'Z', int(changed.sum())


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    t = replay()
    original = (t.outline.copy(), t.board.copy(), list(t.frets), t.mask.copy(), [p.copy() for p in t.parts])
    after = tr.svg(t, 'png')
    root = ET.fromstring(after)
    old_path = tr.curve_path(tr.smoothed(t.outline, tr.DEFAULT_SMOOTH_MM), tr.CURVE_TOL_MM)
    before = after.replace(root[0].attrib['d'], old_path, 1)
    assert before != after
    checks = {}
    for name, text in [('before', before), ('after', after)]:
        (DEST/f'lp_ref_1-{name}.svg').write_text(text, encoding='utf-8')
        overlay = render(ET.fromstring(text), t, tr.masked_photo(t))
        save(DEST/f'lp_ref_1-{name}-overlay.png', overlay)
        xy = t.ruler.to_px(np.array([[t.board[:,0].min()-15,-55],[t.board[:,0].min()+145,55]]))
        left, top = np.floor(xy.min(0)).astype(int)
        right, bottom = np.ceil(xy.max(0)).astype(int)
        save(DEST/f'lp_ref_1-{name}-detail.png',cv2.resize(overlay[top:bottom,left:right],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC))
    a, b = ET.fromstring(before), ET.fromstring(after)
    assert all(x.attrib == y.attrib for x,y in zip(list(a)[1:],list(b)[1:],strict=True))
    _, handles = cached_correction(old_path,t.board)
    checks['lp_ref_1'] = dict(changed_handles=handles, all_other_elements_identical=True,
                             outline_vertices_unchanged=True, mask_unchanged=True,
                             board_sides_rendered=len(tr.board_sides(t)),
                             note='Median suppression also omits board-side portions over the body; unchanged in this patch.')
    np.testing.assert_array_equal(t.outline,original[0])
    np.testing.assert_array_equal(t.board,original[1])
    assert t.frets == original[2]
    np.testing.assert_array_equal(t.mask,original[3])
    for p,q in zip(t.parts,original[4],strict=True):
        np.testing.assert_array_equal(p,q)
    # We have no source/session for this historical photo: do not reconstruct fake Trace geometry.
    source = ROOT/'out/user8_live2/png.svg'
    text = source.read_text(encoding='utf-8')
    root = ET.fromstring(text)
    board = np.vstack([values(root[1].attrib['d']).reshape(2,2),values(root[2].attrib['d']).reshape(2,2)])
    changed, count = cached_correction(root[0].attrib['d'],board)
    assert count == 4
    after = text.replace(root[0].attrib['d'],changed,1)
    (DEST/'user8-cached-before.svg').write_text(text,encoding='utf-8')
    (DEST/'user8-cached-after.svg').write_text(after,encoding='utf-8')
    for name,content in [('before',text),('after',after)]:
        detail=ET.fromstring(content)
        detail.set('viewBox','280 -60 180 120')
        (DEST/f'user8-{name}-detail.svg').write_text(ET.tostring(detail,encoding='unicode'),encoding='utf-8')
    checks['user8_cached'] = dict(source=str(source.relative_to(ROOT)),changed_handles=count,
                                 all_other_elements_identical=True,outline_vertices_unchanged=True,
                                 note='Historical SVG only; not verified as the latest screenshot. No matching clean photo/session available.')
    (DEST/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (DEST/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Стык грифа: до / после</title>
<style>body{font:16px system-ui;margin:24px}section{display:flex;gap:16px}figure{margin:0;width:48%}img{width:100%}</style>
<h1>Стык грифа: локальная коррекция касательных Bézier</h1>
<p>До / после. Не меняются маска, вершины силуэта, лады, накладка, фурнитура и остальная форма корпуса.
Меняются только четыре управляющие точки возле двух стыков длинных границ грифа с корпусом. ML не запускался.</p>
<h2>LP 1: настоящий trace() с сохранёнными SAM-масками</h2>
<section><figure>До<img src="lp_ref_1-before.svg"><img src="lp_ref_1-before-overlay.png"><img src="lp_ref_1-before-detail.png"></figure>
<figure>После<img src="lp_ref_1-after.svg"><img src="lp_ref_1-after-overlay.png"><img src="lp_ref_1-after-detail.png"></figure></section>
<h2>user8: исторический сохранённый SVG</h2><p>Не подтверждено, что это фото из последнего сообщения; скриншот не приложен.
Без соответствующего исходника показаны только реальные сохранённые векторы, не вымышленное фото-наложение.</p>
<section><figure>До<img src="user8-cached-before.svg"><img src="user8-before-detail.svg"></figure><figure>После<img src="user8-cached-after.svg"><img src="user8-after-detail.svg"></figure></section>
<p>Отдельное ограничение: board_sides подавляет всю сторону по медиане расстояний. Это объясняет пропавшие продольные
линии над корпусом, но не обратный ход Bézier. Здесь подавление не изменено: безопасная правка требует локального
отсечения и проверки точного фото. <a href="verification.json">Проверки</a></p>''',encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
