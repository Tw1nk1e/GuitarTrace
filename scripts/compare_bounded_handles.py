"""Export-only five-guitar comparison. Saved masks/polygons/SVG; no ML.

uv run --no-sync python scripts/compare_bounded_handles.py
Browser rasterization is performed by editor_browser.mjs --quality-gallery.
"""
import html
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

from audit_svg_quality import distances
from validate_headstock import sample_path
from gbcv import trace as tr
from gbcv.editor import document

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'out/quality-fixes-2026-11-12'
BASE = ROOT / 'out/quality-audit-2026-11-12'
NAMES = ('lp_ref_1', 'ref_sg_1', 'fender_1', 'fender_2', 'ibanez_rgms8')


def values(text):
    return np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', text)])


def parse(path):
    assert path.startswith('M') and path.endswith('Z') and 'C' in path and 'L' not in path
    chunks = re.findall(r'C([^CZ]+)', path)
    controls = np.array([values(s).reshape(3, 2) for s in chunks])
    nodes = np.vstack([values(path.split('C')[0]), controls[:-1, 2]])
    return chunks, controls, nodes


def cached_correction(path):
    """Only overlong handles, using production's limiter on saved rounded nodes.

    Original dense polygons do not exist for Fender/Ibanez. Never resample their
    already curved paths and pretend that is the original segmentation.
    """
    chunks, controls, nodes = parse(path)
    tangent = (np.roll(nodes, -1, axis=0) - np.roll(nodes, 1, axis=0)) / 6
    limited = tr.bounded_handles(nodes, tangent)
    changed = np.linalg.norm(tangent - limited, axis=1) > .015
    result = list(chunks)
    for i in range(len(nodes)):
        j = (i + 1) % len(nodes)
        if not (changed[i] or changed[j]):
            continue
        c = controls[i].copy()
        if changed[i]:
            c[0] = nodes[i] + limited[i]
        if changed[j]:
            c[1] = nodes[j] - limited[j]
        result[i] = ' '.join(tr._f(v) for v in c.ravel())
    return path.split('C')[0] + ''.join('C' + s for s in result) + 'Z'


def overlong(path):
    _, c, nodes = parse(path)
    lengths = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
    limits = np.minimum(lengths, np.roll(lengths, 1)) / 3
    out = np.linalg.norm(c[:, 0] - nodes, axis=1)
    inc = np.linalg.norm(nodes - np.roll(c[:, 1], 1, axis=0), axis=1)
    return int(np.count_nonzero(np.maximum(out, inc) > limits + .02))


def sampled_crossings(path):
    """Strict crossings of sampled segments, not a proof of cubic topology."""
    p = sample_path(path, step=.3)
    # The parser samples joins twice. Ignore zero-length sample edges.
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]]
    if np.linalg.norm(p[-1] - p[0]) > 1e-9:
        p = np.vstack([p, p[:1]])
    a, b = p[:-1], p[1:]
    low, high = np.minimum(a, b), np.maximum(a, b)
    total = 0
    def cross(v, w):
        return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]
    for i in range(len(a) - 2):
        j = np.arange(i + 2, len(a))
        if i == 0:
            j = j[j != len(a) - 1]
        j = j[np.all(high[j] > low[i], axis=1) & np.all(low[j] < high[i], axis=1)]
        if not len(j):
            continue
        v, w = b[i] - a[i], b[j] - a[j]
        first = cross(v, a[j] - a[i]) * cross(v, b[j] - a[i])
        second = cross(w, a[i] - a[j]) * cross(w, b[i] - a[j])
        total += int(np.count_nonzero((first < -1e-12) & (second < -1e-12)))
    return total


def photo_projection(name, photo, evaluation):
    """Reuse a saved project matrix, or recover ONLY an existing photo frame."""
    project = ROOT / 'out/baseline-editor' / name
    if (project / 'guitar.gbcv.json').exists():
        assert np.array_equal(photo, cv2.imread(str(project / 'underlay.png')))
        data = json.loads((project / 'guitar.gbcv.json').read_text(encoding='utf-8'))
        return np.array(data['document']['photo']['projection']).reshape(3, 3), 'saved project; masked photo byte-identical'
    from debug_fretboard import geometry, no_ml, load_image
    record = next(r for r in evaluation['records'] if r['id'] == name)
    source = ROOT / record['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['source_sha256']
    # Existing CPU-only frame recovery verifies the cached mask pixel-for-pixel.
    # No new body/fret hardware is used in the comparison, only its coordinate map.
    with patch('gbcv.sam.Session', no_ml), patch('gbcv.segment.probability', no_ml), patch('gbcv.knobs.recover', return_value=[]):
        t = geometry(load_image(source, mode='RGBA'), cv2.imread(str(ROOT / 'out' / name / 'mask.png'), 0))
    assert np.array_equal(cv2.cvtColor(tr.masked_photo(t), cv2.COLOR_RGB2BGR), photo)
    return np.array(document(t)['photo']['projection']).reshape(3, 3), 'CPU frame recovery; source hash + mask + masked photo verified'


def photo_overlay(root, photo, projection):
    """Sample the exported cubic, not the pre-export mask. Diagnostic 1px red."""
    image = photo.copy()
    for element in root:
        if element.tag.endswith('path'):
            points = sample_path(element.attrib['d'], step=.15)
            closed = element.attrib['d'].endswith('Z')
        elif element.tag.endswith('circle'):
            a = np.linspace(0, 2*np.pi, 160)
            points = float(element.attrib['r']) * np.c_[np.cos(a), np.sin(a)]
            points += [float(element.attrib['cx']), float(element.attrib['cy'])]
            closed = True
        else:
            continue
        q = np.c_[points, np.ones(len(points))] @ projection.T
        q = np.round(q[:, :2] / q[:, 2:] * 256).astype(np.int32)
        cv2.polylines(image, [q], closed, (30, 30, 230), 1, cv2.LINE_AA, shift=8)
    return image


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((ROOT / 'out/evaluation/manifest.json').read_text())
    records, cards = [], []
    for name in NAMES:
        folder = DEST / name
        folder.mkdir(exist_ok=True)
        before_text = (BASE / name / 'baseline.svg').read_text(encoding='utf-8')
        before, after = ET.fromstring(before_text), ET.fromstring(before_text)
        shutil.copyfile(BASE / name / 'photo.png', folder / 'photo.png')
        shutil.copyfile(BASE / name / 'overlay.png', folder / 'saved-overlay.png')
        original_polygons, candidates = {}, []
        if name in ('lp_ref_1', 'ref_sg_1'):
            source = ROOT / 'out/e2e-lp-sg' / name / 'sam'
            candidates = json.loads((source / 'candidates.json').read_text())
            with np.load(source / 'stages.npz', allow_pickle=False) as z:
                for r in candidates:
                    if r['status'] == 'kept':
                        original_polygons[r['svg_path']] = (r['id'], z[f'c{r["id"]}_export_smoothed_contour'].copy())
            count = 18 if name == 'lp_ref_1' else 14
        else:
            count = next(r['cached_body_paths'] for r in evaluation['records'] if r['id'] == name)
        offset = len(before) - count
        measurements, topology, changed, old_long, new_long = [], [], [], 0, 0
        for i, (a, b) in enumerate(zip(before, after, strict=True)):
            if i < offset or not a.tag.endswith('path'):
                continue
            old_path = a.attrib['d']
            assert old_path.endswith('Z') and 'C' in old_path
            if old_path in original_polygons:
                cid, polygon = original_polygons[old_path]
                assert tr.curve_path(polygon, tr.PART_CURVE_TOL_MM) == old_path
                new_path = tr.curve_path(polygon, tr.PART_CURVE_TOL_MM, bounded=True)
                line = np.vstack([polygon, polygon[:1]])
                old_error = float(distances(sample_path(old_path, step=.15), line).max())
                new_error = float(distances(sample_path(new_path, step=.15), line).max())
                measurements.append(dict(candidate=cid, before=round(old_error, 4), after=round(new_error, 4)))
            else:
                new_path = cached_correction(old_path)
            # Every exported node and every unrelated element is immutable.
            _, a_c, a_nodes = parse(old_path)
            _, b_c, b_nodes = parse(new_path)
            np.testing.assert_array_equal(a_nodes, b_nodes)
            np.testing.assert_array_equal(a_c[:, 2], b_c[:, 2])
            b.set('d', new_path)
            old_long += overlong(old_path)
            new_long += overlong(new_path)
            topology.append(dict(path=i, before=sampled_crossings(old_path), after=sampled_crossings(new_path)))
            if old_path != new_path:
                changed.append(i)
        assert all(ET.tostring(a) == ET.tostring(b) for a, b in zip(list(before)[:offset], list(after)[:offset]))
        (folder / 'before.svg').write_text(before_text, encoding='utf-8')
        (folder / 'after.svg').write_text(ET.tostring(after, encoding='unicode'), encoding='utf-8')
        # Same crops/weights for both versions. The whole SVG is also rasterized
        # at 160/800px in Chromium, avoiding diagnostic polyline-only judgement.
        vb = list(map(float, before.attrib['viewBox'].split()))
        windows = [('body', [vb[0], vb[1], vb[2] * .43, vb[3]])]
        if measurements:
            worst = max(measurements, key=lambda r: r['before'])
            old_path, (_, polygon) = next((p, data) for p, data in original_polygons.items() if data[0] == worst['candidate'])
            q = np.vstack([sample_path(old_path), polygon])
            lo, hi = q.min(0) - 4, q.max(0) + 4
            windows.append(('worst', [*lo, *(hi - lo)]))
        for suffix, window in windows:
            for label, root in [('before', before), ('after', after)]:
                crop = ET.fromstring(ET.tostring(root))
                crop.set('viewBox', ' '.join(map(str, window)))
                (folder / f'{label}-{suffix}.svg').write_text(ET.tostring(crop, encoding='unicode'), encoding='utf-8')
        photo = cv2.imread(str(folder / 'photo.png'))
        projection, projection_source = photo_projection(name, photo, evaluation)
        for label, root in [('before', before), ('after', after)]:
            overlay = photo_overlay(root, photo, projection)
            cv2.imwrite(str(folder / f'{label}-overlay.png'), overlay)
            x, y, w, h = windows[0][1]
            box = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]]) @ projection.T
            box = box[:, :2] / box[:, 2:]
            lo = np.maximum(0, np.floor(box.min(0))).astype(int)
            hi = np.minimum([photo.shape[1], photo.shape[0]], np.ceil(box.max(0))).astype(int)
            cv2.imwrite(str(folder / f'{label}-body-overlay.png'), overlay[lo[1]:hi[1], lo[0]:hi[0]])
        record = dict(id=name, changed_paths=changed, unchanged_nodes=True, non_hardware_identical=True,
                      source='dense saved polygons + current exporter' if measurements else 'saved rounded SVG nodes + production handle limiter',
                      overlong_nodes_before=old_long, overlong_nodes_after=new_long,
                      deviation_before=sum(r['before'] > .5 for r in measurements),
                      deviation_after=sum(r['after'] > .5 for r in measurements),
                      measured_contours=len(measurements), measurements=measurements,
                      max_before=max((r['before'] for r in measurements), default=None),
                      max_after=max((r['after'] for r in measurements), default=None), windows=[w[0] for w in windows])
        record['sampled_self_crossings'] = topology
        record['photo_projection'] = projection_source
        record['paths_with_increased_crossings'] = [r['path'] for r in topology if r['after'] > r['before']]
        assert not record['paths_with_increased_crossings'], (name, 'sampled crossing regression')
        records.append(record)
        def pair(suffix):
            return '<div class="pair">' + ''.join(f'<figure><figcaption>{label}</figcaption><img src="{name}/{side}{suffix}"></figure>' for side, label in [('before', 'До'), ('after', 'После')]) + '</div>'
        evidence = (f"Отклонение >0,5: {record['deviation_before']} → {record['deviation_after']} из {len(measurements)}; максимум {record['max_before']} → {record['max_after']}."
                    if measurements else 'Оригинальные плотные полигоны не сохранены: отклонение относительно них не измеряется. Исправлены только чрезмерные ручки сохранённых SVG; точный полный re-export из масок не заявляется.')
        cards.append(f'<section><h2>{name}</h2><p>{html.escape(evidence)}</p><p>Узлов с чрезмерными ручками: {old_long} → {new_long}. Изменено путей: {len(changed)}. Все вершины, не-фурнитура, толщины и порядок объектов сохранены.</p>'
                     f'<details><summary>Маскированное фото и историческое наложение (не after)</summary><img src="{name}/photo.png"><img src="{name}/saved-overlay.png"></details>'
                     + '<h3>Новые наложения экспортированных кривых на то же маскированное фото</h3><p>Красные линии 1 px — диагностика координат, не художественная толщина. Проекция проверена по совпадению маскированного фото.</p>' + pair('-body-overlay.png')
                     + pair('.svg') + '<h3>160 px — настоящий Chromium SVG → Canvas PNG</h3>' + pair('-160.png')
                     + '<h3>Фурнитура крупно — SVG</h3>' + pair('-body.svg')
                     + ('<h3>Наибольший исходный выброс, одинаковое окно</h3>' + pair('-worst.svg') if measurements else '')
                     + f'<details><summary>Растеризация 800 px и все измерения</summary>{pair("-800.png")}<pre>{html.escape(json.dumps(record, ensure_ascii=False, indent=2))}</pre></details></section>')
        print(name, 'deviation', record['deviation_before'], record['deviation_after'],
              'max', record['max_before'], record['max_after'], 'overlong', old_long, new_long,
              'crossing regressions', record['paths_with_increased_crossings'])
    (DEST / 'metrics.json').write_text(json.dumps(dict(ml_inference=False, records=records), ensure_ascii=False, indent=2), encoding='utf-8')
    intro = '''<h1>Bézier и автовыделение — одна ограниченная итерация</h1>
<p>До/после на тех же сохранённых результатах пяти гитар. Новых детекторов, моделей, ML-инференсов и глобального сглаживания нет.</p>
<h2>Причина и правка</h2><p>Равномерная касательная (next-prev)/6 зависит от длинного соседа и превращает короткий участок в крючок. Для фурнитуры общая касательная ограничена третью длины каждого соседнего ребра. Оба ручки узла сокращаются одинаково: вершины и направление касательной сохранены, сопряжение C1 сохраняется до округления SVG. Все обычные участки ниже ограничения остаются прежними. Силуэт, голова, колки, лады, накладка, положения деталей и толщины не менялись. Все пути остаются кубическими, не заменены ломаными.</p>
<p><b>Остаточное ограничение:</b> допуск Douglas–Peucker — не допуск итоговой кривой. Эта правка убирает чрезмерную длину ручек, но НЕ обещает отклонение всех кривых ≤0,5 и не исправляет шумную сегментацию. Числа отклонений ниже — дискретизированная нижняя оценка максимума относительно уже сглаженного полигона, не фото. Полное восстановление исходной маски по SVG не выполнялось.</p>
<h2>SAM</h2><p>Однократное явное разрешение в живом серверном Session, без записи в проект. Модель запускается только по клику; инструмент остаётся активным. Прогресс и отмена ожидания кнопкой/Esc. Сервер отвергает параллельные запросы, не ставит их в очередь. Отмена не прерывает уже идущий CPU-вызов; его поздний ответ не сливается с рисунком. Успешные контуры кэшируются по пикселю, максимум 32; смена маски сбрасывает кэш, смена сглаживания только переэкспортирует геометрию. Дубликаты и Undo/Redo остаются в браузере.</p>
<p><a href="metrics.json">Метрики</a> · <a href="browser-rasterization.json">Проверка настоящей растеризации браузером</a> · <a href="tests.txt">Результаты тестов</a>. Настоящие PNG построены browser e2e; отдельно показаны before/after-наложения самих экспортированных кривых. Историческое наложение подписано отдельно. Для трёх примеров без плотных полигонов ограничения доказательств явно подписаны.</p>'''
    style = 'body{font:16px/1.5 system-ui;max-width:1500px;margin:25px auto;padding:0 20px}.pair{display:flex;gap:16px}.pair figure{width:50%;margin:0;min-width:0}img{width:100%;background:white}section{border-top:2px solid #aaa;margin:30px 0}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}figcaption{font-weight:bold}summary{cursor:pointer}'
    (DEST / 'index.html').write_text(f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>SVG before/after</title><style>{style}</style>{intro}{"".join(cards)}</html>', encoding='utf-8')


if __name__ == '__main__':
    main()
