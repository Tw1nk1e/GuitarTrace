"""Stage A: inspect saved, guitar-masked artifacts only. No detectors or ML.

uv run --no-sync python scripts/audit_svg_quality.py
Inspection crops are gallery layout, never production geometry rules.
"""
import hashlib
import html
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from gbcv import trace as tr
from validate_headstock import sample_path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "out/quality-audit-2026-11-12"
SOURCES = [
    ("lp_ref_1", "Les Paul", "e2e-lp-sg/lp_ref_1", "final.svg", "photo-oriented-masked.png", "overlay.png", True),
    ("ref_sg_1", "SG", "e2e-lp-sg/ref_sg_1", "final.svg", "photo-oriented-masked.png", "overlay.png", True),
    ("fender_1", "Fender Telecaster 1", "knob-recovery/fender_1", "after.svg", "photo.png", "after-overlay.png", False),
    ("fender_2", "Fender Telecaster 2", "knob-recovery/fender_2", "after.svg", "photo.png", "after-overlay.png", False),
    ("ibanez_rgms8", "Ibanez RGMS8", "knob-recovery/ibanez_rgms8", "after.svg", "photo.png", "after-overlay.png", False),
]
NOTES = {
    "lp_ref_1": "Пропущен бридж и одна из четырёх ручек; рамки датчиков волнистые, местами сливаются. На стороне корпуса некоторые лады заканчиваются без продольной границы накладки. Отказ поиска накладки: неоднозначная пара краёв.",
    "ref_sg_1": "В сохранённом результате 4 ключа/втулки при шести видимых механизмах; контуры датчиков сливаются с соседними границами. Фаска корпуса реальна: считать всю внутреннюю обводку шумом нельзя. Записаны 23 лада, правильность счёта требует отдельной проверки по фото.",
    "fender_1": "Крючки и повторные линии у пластины управления, выреза и края пикгарда; неровные рамки датчика. Пропуск части управления. Силуэт в целом соответствует фото.",
    "fender_2": "Шумные выемки на бридже и пластине управления; местами две близкие линии по границе пикгарда. Слабые края накладки: прежний детектор отказался от уточнения, а не подтвердил точность.",
    "ibanez_rgms8": "Датчики разбиты на волнистые частичные контуры с выступами; бридж найден фрагментарно, верхняя ручка пропущена. Веер и восемь колков сохранены — не следует заменять их модельной раскладкой.",
}


def distances(points, line):
    """Distance to ALL finite segments; sampled maximum is only a lower bound."""
    a, v = line[:-1], np.diff(line, axis=0)
    result = []
    for chunk in np.array_split(points, max(1, len(points) // 128)):
        relative = chunk[:, None] - a
        fraction = np.clip(np.sum(relative * v, axis=2) / np.maximum(np.sum(v * v, axis=1), 1e-12), 0, 1)
        result.extend(np.linalg.norm(relative - fraction[..., None] * v, axis=2).min(1))
    return np.asarray(result)


def img(path, label):
    return f'<figure><figcaption>{html.escape(label)}</figcaption><img loading="lazy" src="{path}"></figure>'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records, sections = [], []
    for name, title, source, svg_name, photo_name, overlay_name, diagnostic in SOURCES:
        source = ROOT / "out" / source
        folder = DEST / name
        folder.mkdir(exist_ok=True)
        artifacts = {}
        for old, new in [(svg_name, "baseline.svg"), (photo_name, "photo.png"), (overlay_name, "overlay.png")]:
            path = source / old
            if new == "photo.png" and not diagnostic:
                # knob-recovery/photo.png is a rectified BODY crop, not the full
                # photograph used by after-overlay.png. Do not pair unlike frames.
                path = ROOT / "out/evaluation" / name / "photo.png"
            shutil.copyfile(path, folder / new)
            artifacts[new] = dict(source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        root = ET.parse(folder / "baseline.svg").getroot()
        vb = np.array(root.attrib["viewBox"].split(), float)
        photo = cv2.imread(str(folder / "photo.png"))
        overlay = cv2.imread(str(folder / "overlay.png"))
        assert photo is not None and overlay is not None and photo.shape == overlay.shape, name
        # These inputs are already masked. Never open an original sample here.
        ys, xs = np.nonzero(np.min(photo, axis=2) < 240)
        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        crops = []
        for region, caption, lo, hi in [("body", "Фурнитура и контуры корпуса", 0, .46),
                                        ("join", "Соединение с грифом", .28, .50),
                                        ("head", "Головка и колки", .80, 1)]:
            x0, x1 = left + int((right - left) * lo), left + int((right - left) * hi)
            for key, image in [("photo", photo), ("overlay", overlay)]:
                cv2.imwrite(str(folder / f"{region}-{key}.png"), image[top:bottom, x0:x1])
            crop = ET.fromstring(ET.tostring(root))
            crop.set("viewBox", " ".join(map(str, [vb[0] + vb[2] * lo, vb[1], vb[2] * (hi - lo), vb[3]])))
            (folder / f"{region}.svg").write_text(ET.tostring(crop, encoding="unicode"), encoding="utf-8")
            crops.append(f'<details open><summary>{caption}</summary><p>Фото и наложение: один пиксельный crop; SVG: отдельное окно viewBox той же области, не точная регистрация с фото.</p><div class="grid">'
                         + img(f"{name}/{region}-photo.png", "Фото, без немаскированного фона")
                         + img(f"{name}/{region}-overlay.png", "Сохранённое диагностическое наложение")
                         + img(f"{name}/{region}.svg", "Настоящий сохранённый SVG, увеличение") + '</div></details>')
        rows = []
        if diagnostic:
            candidates = json.loads((source / "sam/candidates.json").read_text(encoding="utf-8"))
            with np.load(source / "sam/stages.npz", allow_pickle=False) as z:
                for candidate in candidates:
                    if candidate["status"] != "kept":
                        continue
                    cid = candidate["id"]
                    polygon = z[f"c{cid}_export_smoothed_contour"]
                    current = tr.curve_path(polygon, tr.PART_CURVE_TOL_MM)
                    error = float(distances(sample_path(current, step=.15), np.vstack([polygon, polygon[:1]])).max())
                    rows.append(dict(candidate=cid, sampled_max_units=round(error, 4),
                                     matches_saved_path=current == candidate["svg_path"], tolerance=tr.PART_CURVE_TOL_MM))
                rows.sort(key=lambda r: -r["sampled_max_units"])
                worst = rows[0]
                polygon = z[f'c{worst["candidate"]}_export_smoothed_contour']
                path = tr.curve_path(polygon, tr.PART_CURVE_TOL_MM)
                q = sample_path(path, step=.15)
                low = np.vstack([polygon, q]).min(0) - 3
                size = np.vstack([polygon, q]).max(0) + 3 - low
                points = " ".join(f"{x},{y}" for x, y in polygon)
                svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{low[0]} {low[1]} {size[0]} {size[1]}">'
                       f'<polygon points="{points}" fill="none" stroke="#009ac0" stroke-width=".35"/>'
                       f'<path d="{path}" fill="none" stroke="#d42626" stroke-width=".35"/></svg>')
                (folder / "deviation.svg").write_text(svg, encoding="utf-8")
        record = dict(id=name, artifacts=artifacts, spline_audit=rows,
                      paths=sum(e.tag.endswith("path") for e in root.iter()),
                      circles=sum(e.tag.endswith("circle") for e in root.iter()))
        records.append(record)
        provenance = ("Сохранённый полный прогон с диагностикой этапов, не новый инференс."
                      if diagnostic else "Исторический SVG с сохранёнными SAM-кривыми; это НЕ полный прогон сегодняшнего кода.")
        evidence = ''
        if rows:
            evidence = '<h3>Проверка текущего curve_path() на сохранённых полигонах</h3><p>Голубой — уже сглаженный полигон, красный — экспорт текущей функции. Это не ошибка относительно фото и не before/after.</p>'
            evidence += img(f"{name}/deviation.svg", f'Деталь {rows[0]["candidate"]}: максимальное измеренное отклонение {rows[0]["sampled_max_units"]}')
            evidence += '<pre>' + html.escape(json.dumps(rows, ensure_ascii=False, indent=2)) + '</pre>'
        sections.append(f'<section><h2>{title}</h2><p>{provenance}</p><p><b>Наблюдения:</b> {NOTES[name]}</p>'
                        + img(f"{name}/photo.png", "Исходное фото в ориентации обработки, только внутри сохранённой маски")
                        + img(f"{name}/baseline.svg", "Baseline: сохранённый SVG, без изменений")
                        + img(f"{name}/overlay.png", "Сохранённое наложение (диагностика, не эталон)")
                        + ''.join(crops) + evidence + '</section>')
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [ROOT / "src/gbcv/trace.py", ROOT / "src/gbcv/contour.py", Path(__file__)]}
    (DEST / "manifest.json").write_text(json.dumps(dict(ml_inference=False, production_changed=False,
                                                       code_sha256=code_hashes, records=records), ensure_ascii=False, indent=2), encoding="utf-8")
    intro = '''<h1>GuitarTrace — этап A: аудит качества</h1>
<p><b>Только baseline. Исправления и before/after пока не выполнены.</b> 5 фото, 4 семейства; никаких новых моделей, ML-инференсов, загрузок или изменений production-кода. Оригиналы не копируются: используются уже сохранённые маскированные фото. SHA-256 и происхождение файлов: <a href="manifest.json">manifest.json</a>.</p>
<p>Важно для воспроизводимости: в knob-recovery/photo.png сохранён выпрямленный crop корпуса, а after-overlay.png — полный ориентированный кадр. Они не являются парой. Для Fender/Ibanez здесь взят полный маскированный photo.png из evaluation с проверкой одинаковых размеров. SHA фиксирует выбранные артефакты, но сам по себе не доказывает пиксельную регистрацию; соответствие дополнительно просмотрено визуально.</p>
<h2>Три приоритета по влиянию на результат</h2><ol>
<li><b>Неподтверждённый допуск итоговых Bézier.</b> Douglas–Peucker ограничивает ломаную, не Catmull–Rom; неравномерные шаги дают выбросы. Повторено текущим curve_path() на независимых LP/SG. Приоритет для первой узкой геометрической правки: ограничение касательных с проверкой отклонения и топологии. Точки, число деталей и маски не менять. Визуальная польза ещё требует before/after.</li>
<li><b>Шумные/частично совпадающие границы деталей.</b> LP, оба Fender, Ibanez: волны, лишние выемки, фрагменты датчиков/бриджа. Причины до экспорта: SAM-маски, закрытие щелей, крупнейшая компонента, фильтр повторов (разница площади &lt;15%, перекрытие &gt;85%). Реальные вложенные границы нельзя удалять по одному пересечению или вложенности. Надёжного общего исправления пока нет.</li>
<li><b>Несогласованная видимость соседних конструктивных линий.</b> board_sides() целиком скрывает сторону по медиане расстояний, а не только совпадающий участок; над корпусом граница может пропадать. Все hardware имеют одинаковую det-толщину; близкие реальные контуры сливаются в png. Особенно видно на LP/SG. Локальное отсечение требует отдельной проверки на нескольких фото; уменьшать все толщины или достраивать край без фотодоказательства нельзя.</li></ol>
<h2>Классификация причин и границы доказательств</h2><ul>
<li><b>Сегментация:</b> фрагменты бриджа/датчиков и пропуски ручек — повторяющиеся; число пропусков нельзя честно оценить одним универсальным порогом.</li>
<li><b>Геометрия/сглаживание:</b> выбросы кривых подтверждены численно; Gaussian дополнительно меняет узкие углы. Допуск .5 относится к условным единицам, не физическим мм.</li>
<li><b>Пересечения:</b> близкие и вложенные границы видны; не каждое пересечение ошибочно. Самопересечения реальных кривых отдельно не измерялись.</li>
<li><b>SVG-рендеринг:</b> три толщины уже существуют; round cap/join уже включены. Проблема не в отсутствии этих атрибутов, а в плотности и одинаковом весе неразмеченных деталей.</li>
<li><b>Обнаружение:</b> SG — 4 втулки при шести механизмах, LP — пропущенные бридж/ручка, RGMS8 — фрагментарный бридж. Конкретные пропуски зависят от входа, общий класс системный. Новый детектор не предлагается.</li>
<li><b>Уже исправленное:</b> production содержит локальное ограничение касательных у соединения грифа с корпусом; нельзя выдавать старый артефакт стыка за новую регрессию. Нет доказательства, что весь силуэт надо сглаживать сильнее.</li></ul>
<h2>Аудит автовыделения</h2>
<p>editor.js:autoSelect(): window.confirm на каждый клик → API confirm_ml:true → server.py:api_auto_select() → trace.sam_at(): новая sam.Session(crop) → set_image → together → part_from_mask → added_part/document/svg. Это локальное подтверждение приложения, не permission mode OpenCode.</p>
<ul><li>Клиент busy отсекает быстрые клики; общий threading.Lock сериализует инференсы, но сервер принимает ожидающие запросы от других вкладок. Отдельного отказа «занято», отмены и кэша кликов нет.</li>
<li>Статус «SAM выделяет деталь…» уже есть. После успеха tool='select': для следующей детали нужно снова выбрать инструмент.</li>
<li>Ошибка не изменяет серверный Trace; ответ — один объект. Проверка формата и duplicateContour предшествуют change(), Undo/Redo ведёт браузер. Это надо сохранить.</li>
<li>Серверный Session живёт 30 минут, максимум 8 сеансов. _predictor() общий и изменяемый: сохранять sam.Session между изображениями без сохранения/восстановления embeddings небезопасно.</li></ul>
<h3>Предлагаемая ограниченная реализация</h3><p>Строгое разрешение SAM в серверном Session, включаемое один раз явно; включение и смена инструмента не запускают ML. Кэш успешного результата точного пиксельного клика в памяти с ограниченным размером, привязанный к маске и сглаживанию (либо кэш сырой маски + повторный экспорт). Не возвращать соседнюю деталь по близости клика. Серверный non-blocking lock: занято → 409, без очереди. На клиенте оставить auto активным, добавить отмену ожидания и защиту от запоздавшего ответа. AbortController не прерывает уже идущий синхронный CPU-инференс: блокировка должна удерживаться до его реального завершения.</p>
<h2>Проверка следующего этапа</h2><p>Сначала unit/API-тесты на моках (разрешение, отсутствие автозапуска, кэш, занятость, отмена/поздний ответ, TTL, дубликаты, Undo/Redo). Затем одинаковые сохранённые маски before/after для геометрии. Обязательно проверять настоящий SVG на 160 px и крупно, а не только красную полилинию наложения. Проверить выбор, резинку, редактирование, .gbcv.json, SVG/PNG в браузере с запрещённым ML. Текущие тесты автовыделения закрепляют подтверждение каждого запроса: их нужно менять вместе с контрактом, а не отключать.</p>
<h2>Честная оценка</h2><p>Новых улучшений и измеренного сокращения ручной работы на этапе A нет. Ожидаемые кандидаты — меньше крючков/выступов и непрерывный режим выделения; это пока гипотеза для приёмки. Разрешение на новые ML-инференсы не требуется для проверки экспортной геометрии, но без него нельзя заявить end-to-end приёмку реального SAM. Полный новый baseline недоступен для всех пяти фото: смешанные исторические результаты явно помечены.</p>
<p>По правилу AGENTS.md «один этап за раз» следующий этап — только после одобрения. <a href="tests.txt">Свежий результат pytest</a> (при наличии файла). Клики редактора и browser e2e в этом этапе не выполнялись.</p>'''
    style = '''body{font:16px/1.5 system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#222}section{border-top:3px solid #aaa;margin-top:40px}figure{margin:12px 0;min-width:0}img{width:100%;background:white}figcaption{font-weight:600}details{border:1px solid #ccc;padding:12px;margin:16px 0}summary{cursor:pointer;font-weight:bold}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.grid img{height:400px;object-fit:contain}pre{white-space:pre-wrap;background:#f4f4f4;padding:12px}li{margin:10px 0}@media(max-width:900px){.grid{grid-template-columns:1fr}}'''
    (DEST / "index.html").write_text(f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>Качество SVG: аудит A</title><style>{style}</style>{intro}{"".join(sections)}</html>', encoding="utf-8")
    print(DEST / "index.html")
    for record in records:
        rows = record["spline_audit"]
        print(record["id"], "curves:", len(rows), "above tolerance:", sum(r["sampled_max_units"] > r["tolerance"] for r in rows),
              "max:", max((r["sampled_max_units"] for r in rows), default=None),
              "matches saved:", all(r["matches_saved_path"] for r in rows) if rows else "n/a")


if __name__ == "__main__":
    main()
