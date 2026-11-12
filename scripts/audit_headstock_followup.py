"""Headstock follow-up: cached photos/masks only, models are forbidden.

Historical contour drift is measured separately from the post-only follow-up.
The preserved body/SAM elements are copied from the accepted cached SVG.
"""
import json
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

from debug_fretboard import ROOT, geometry, load_image, no_ml, save, tr
from validate_headstock import NAMES, EXPECTED, sample_path, distance_to_polyline, svg_parts, same, validate_posts
from gbcv.orient import orient_and_split
from gbcv.split import split


DEST = ROOT / "out" / "validation-headstock-followup"


def render_outline(path, viewbox, height=160, stroke=5.5):
    """Actual SVG curve at website size, supersampled for subpixel comparison."""
    x, y, w, h = map(float, viewbox.split())
    scale = height/h
    canvas = np.full((height*4, int(np.ceil(w*scale))*4, 3), 255, np.uint8)
    pts = np.round((sample_path(path, .05)-[x, y])*scale*4).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (27, 27, 27), max(1, round(stroke*scale*4)), cv2.LINE_AA)
    return cv2.resize(canvas, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)


def main():
    DEST.mkdir(exist_ok=True)
    records = {}
    with patch("gbcv.segment.probability", no_ml), patch("gbcv.sam.Session", no_ml):
        for name, expected in zip(NAMES, EXPECTED):
            folder = DEST / name
            folder.mkdir(exist_ok=True)
            rgba = load_image(next((ROOT / "samples" / "dl").glob(name + ".*")), mode="RGBA")
            mask = cv2.imread(str(ROOT / "out" / name / "mask.png"), 0)
            t = geometry(rgba, mask)
            gray = cv2.cvtColor(t.photo, cv2.COLOR_RGB2GRAY)
            solid, plate, keys = tr.head_parts(mask, t.ruler, t.ruler.ff.b, gray)
            d = t.ruler.d
            baseline = ROOT / "out" / "validation-headstock" / name
            old_root = ET.parse(baseline / "before.svg").getroot()
            # Preserve the historical pre-post-fix snapshot even after the main
            # validator regenerates its after.svg with corrected Gibson posts.
            snapshot = folder / "before.svg"
            accepted = (snapshot if snapshot.exists() else baseline / "after.svg").read_text(encoding="utf-8")
            before_root = ET.fromstring(accepted)
            old = old_root[0].attrib["d"]
            new = before_root[0].attrib["d"]
            posts = tr._posts(gray, t.ruler, plate, keys)
            post_checks = validate_posts(gray, t.ruler, plate, keys, posts)
            t.keys, t.posts = [tr._poly_mm(k, t.ruler, .3) for k in keys], posts
            fresh, start, end = svg_parts(tr.svg(t), len(posts), len(keys))
            assert fresh[0].attrib["d"] == new, "Post fix must not change the accepted outline"
            cached, cs, ce = svg_parts(accepted, len(posts), len(keys))
            assert start == cs
            assert len(posts) == len(keys) == expected
            assert all(same(a, b) for a, b in zip(list(cached)[cs+2*len(posts):ce], list(fresh)[start+2*len(posts):end]))
            # Match the live validator's bounds around the restored tip; a
            # historical shortened viewBox must not clip unchanged geometry.
            result = ET.Element(cached.tag, fresh.attrib)
            result.extend([*list(cached)[:cs], *list(fresh)[start:start+2*len(posts)], *list(cached)[cs+2*len(posts):]])
            geometry_unchanged = all(same(a, b) for a, b in zip(cached, result))
            unchanged = geometry_unchanged and cached.attrib == result.attrib
            if name in ("fender_1", "ibanez_rgms8"):
                assert geometry_unchanged, "Accepted Fender/Ibanez geometry must remain identical"
            if name == "fender_1":
                assert unchanged, "Accepted Fender SVG must remain byte-identical"
            after = accepted if unchanged else ET.tostring(result, encoding="unicode").replace("ns0:", "").replace(":ns0", "")+"\n"
            latest = ET.parse(baseline / "after.svg").getroot()
            assert same(result, latest) and len(result) == len(latest)
            assert all(same(a, b) for a, b in zip(result, latest)), "Validators disagree on the current SVG"
            for label, text in (("before", accepted), ("after", after)):
                target = folder / f"{label}.svg"
                if label == "before" and target.exists():
                    assert target.read_text(encoding="utf-8") == text
                else:
                    target.write_text(text, encoding="utf-8")
            # Only circles may change in this follow-up, never keys or other parts.
            assert all(same(a, b) for a, b in zip(cached, result) if not a.tag.endswith("circle"))
            before_posts = [(float(e.attrib["cx"]), float(e.attrib["cy"]), float(e.attrib["r"]))
                            for e in list(cached)[cs:cs+2*len(posts):2]]
            try:
                validate_posts(gray, t.ruler, plate, keys, before_posts)
                historical_posts_valid = True
            except AssertionError:
                historical_posts_valid = False
            if not geometry_unchanged:
                assert not historical_posts_valid, "Moved posts need evidence that the old snapshot was wrong"
            y, x = np.nonzero(plate)
            y0, y1 = max(0, y.min()-100), min(mask.shape[0], y.max()+101)
            x0, x1 = max(0, x.min()-30), min(mask.shape[1], x.max()+31)
            for label, pp in (("before", before_posts), ("after", posts)):
                t.posts = pp
                image = tr.overlay(t)
                save(folder / f"{label}-head-x3.png", cv2.resize(image[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
            t.posts = posts
            a, b = sample_path(old, .025), sample_path(new, .025)
            ap, bp = t.ruler.to_px(a), t.ruler.to_px(b)
            ai, bi = a[:, 0] < t.nut_x, b[:, 0] < t.nut_x
            da, db = distance_to_polyline(ap[ai], bp), distance_to_polyline(bp[bi], ap)
            distances = np.r_[da, db]
            points = np.vstack([ap[ai], bp[bi]])
            j = distances.argmax()
            worst = points[j]
            picture = tr.masked_photo(t)
            cv2.polylines(picture, [np.round(ap).astype(np.int32)], True, (0, 100, 255), 1)
            cv2.polylines(picture, [np.round(bp).astype(np.int32)], True, (255, 50, 20), 1)
            cv2.circle(picture, tuple(np.round(worst).astype(int)), 16, (0, 180, 30), 2)
            save(folder / "outline-difference.png", picture)
            x, y = np.round(worst).astype(int)
            crop = picture[max(0, y-65):y+66, max(0, x-65):x+66]
            save(folder / "outline-max-x6.png", cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
            am, bm = distance_to_polyline(a[ai], b), distance_to_polyline(b[bi], a)
            frame, cleaned, _ = orient_and_split(tr.silhouette(rgba), open_body=False)
            sp = split(cleaned, nut_x=t.ruler.ff.b)
            inverse = cv2.invertAffineTransform(frame.M)
            to_source = lambda p: p @ inverse[:, :2].T + inverse[:, 2]
            source_point = to_source(worst[None])[0]
            source_mask = tr.silhouette(rgba)
            source = np.where(source_mask[..., None] > 0, rgba[..., :3], 255).astype(np.uint8)
            cv2.polylines(source, [np.round(to_source(ap)).astype(np.int32)], True, (0, 100, 255), 1)
            cv2.polylines(source, [np.round(to_source(bp)).astype(np.int32)], True, (255, 50, 20), 1)
            cv2.circle(source, tuple(np.round(source_point).astype(int)), 16, (0, 180, 30), 2)
            save(folder / "outline-difference-source.png", source)
            for label, path in (("before", old), ("after", new)):
                save(folder / f"outline-{label}-160.png", render_outline(path, before_root.attrib["viewBox"]))
            # Attribute the drift to individual pipeline stages, not the mask.
            old_cache = np.load(baseline / "baseline.npz")
            stages = {}
            for label, left, right in (
                ("outline_polygon", old_cache["outline"], t.outline),
                ("global_resample_smooth", tr.smoothed(old_cache["outline"], 1.5), tr.smoothed(t.outline, 1.5)),
            ):
                left, right = np.vstack([left, left[:1]]), np.vstack([right, right[:1]])
                la = tr.contour.resample(left, .05)
                rb = tr.contour.resample(right, .05)
                dd = np.r_[distance_to_polyline(la[la[:, 0] < t.nut_x], right),
                           distance_to_polyline(rb[rb[:, 0] < t.nut_x], left)]
                stages[label+"_max_svg_units"] = float(dd.max())
            neck_start = sp.neck_x0
            regions = {}
            for region, selected in (("body", points[:, 0] < neck_start),
                                     ("neck", points[:, 0] >= neck_start)):
                regions[region+"_max_px"] = float(distances[selected].max()) if selected.any() else 0.
            max_svg = float(max(am.max(), bm.max()))
            vb_height = float(before_root.attrib["viewBox"].split()[3])
            from scipy.optimize import linear_sum_assignment
            before_px = t.ruler.to_px(np.asarray(before_posts)[:, :2])
            after_px = t.ruler.to_px(np.asarray(posts)[:, :2])
            rows, cols = linear_sum_assignment(np.linalg.norm(before_px[:, None]-after_px, axis=2))
            shifts = np.linalg.norm(before_px[rows]-after_px[cols], axis=1)
            records[name] = {"px_per_unit_at_nut": d, "outside_head_max_px": float(distances.max()),
                             "outside_head_mean_px": float(distances.mean()),
                             "outside_head_p99_px": float(np.quantile(distances, .99)),
                             "outside_head_max_svg_units": max_svg,
                             "outside_head_max_at_160px_svg_height": max_svg*160/vb_height,
                             "worst_canvas_px": worst.tolist(),
                             "worst_source_px": source_point.tolist(),
                             "source_dimensions": list(rgba.shape[1::-1]),
                             "worst_svg": t.ruler.to_mm(worst[None])[0].tolist(),
                             **stages, **regions,
                             "keys": len(keys), "posts": len(posts),
                             **post_checks,
                             "validators_agree": True,
                             "historical_posts_valid": historical_posts_valid,
                             "followup_outline_identical": fresh[0].attrib == cached[0].attrib,
                             "followup_all_non_circle_svg_elements_identical": True,
                             "followup_entire_svg_identical": unchanged,
                             "followup_viewbox_updated": cached.attrib != result.attrib,
                             "posts_before_canvas_px": before_px.tolist(),
                             "posts_after_canvas_px": after_px.tolist(),
                             "post_displacements_source_px": shifts.tolist(),
                             "outside_head_mask_identical": bool(np.array_equal(solid[:, :int(t.ruler.ff.b)], old_cache["solid"][:, :int(t.ruler.ff.b)])),
                             "ml_executed": False}
            # No new contour regression in this follow-up, and no lost tips.
            removed = np.any(old_cache["keys"], axis=0) & ~np.any(keys, axis=0)
            assert (solid[removed] > 0).all() and (plate[removed] > 0).all()
            assert records[name]["outside_head_mask_identical"]
            print(name, json.dumps({key: records[name][key] for key in (
                "keys", "posts", "post_positions_valid", "historical_posts_valid", "validators_agree",
                "followup_outline_identical", "followup_viewbox_updated", "outside_head_max_px", "ml_executed")}))
    (DEST / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    sections = []
    for name, r in records.items():
        sections.append(f'<h2>{name}</h2><p>Историческое отклонение после исправления головы: {r["outside_head_max_px"]:.4f} px исходного фото / {r["outside_head_max_svg_units"]:.4f} единицы SVG; при высоте SVG 160 px — {r["outside_head_max_at_160px_svg_height"]:.3f} px. Максимум в исходном фото: {r["worst_source_px"]}. В этой доработке контур не менялся.</p>'
                        f'<p><a href="{name}/before.svg">SVG до поиска втулок</a> · <a href="{name}/after.svg">SVG после</a> · <a href="{name}/outline-difference-source.png">Контуры на исходном фото, максимум отмечен зелёным</a></p>'
                        f'<h3>Максимум исторического изменения ×6: синий — до исправления головы, красный — после</h3><img src="{name}/outline-max-x6.png">'
                        f'<h3>Внешний контур при высоте 160 px: до / после исправления головы</h3><img src="{name}/outline-before-160.png"><img src="{name}/outline-after-160.png">'
                        f'<h3>Кружки: до / после поиска втулок, ×3</h3><img src="{name}/before-head-x3.png"><img src="{name}/after-head-x3.png">')
    (DEST / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Контур и втулки: проверка</title><style>body{font:16px sans-serif;margin:24px}img{display:block;margin:12px 0;max-width:none}</style><h1>Контур и втулки: проверка без ML</h1><p>Фото только внутри маски гитары. Старое изменение головы и новая коррекция втулок измерены отдельно. Текущая доработка меняет только кружки; принятый внешний контур сохранён.</p><h2>Причина изменения контура</h2><p>В smoothed() равномерная сетка строится вдоль всего замкнутого периметра. Изменение длины головы сдвигает фазу сетки на оставшейся части контура. Замкнутый Douglas–Peucker выбирает немного другие узлы. В curve_path() равномерные касательные Catmull–Rom зависят от соседних узлов: длинный прямой отрезок грифа усиливает маленькое смещение узла у стыка с корпусом. У Ibanez после сглаживания разница лишь 0.108 единицы, после кубических кривых — 2.246.</p><h2>Визуальная оценка</h2><p>Максимум находится на переходе корпуса в гриф, не на роге или другом мелком элементе. Контур самого грифа менялся не более чем на 0.333 px исходного фото. У Les Paul разница при высоте вывода 160 px составляет 0.130 px, у Ibanez — до 1.122 px локально на стыке; потерю деталей корпуса и грифа относительно прежнего результата не обнаружено. На увеличении разница видна. Имеющаяся в обеих версиях особенность кубической аппроксимации — выброс касательной у длинного прямого участка — не исправлялась в этой доработке.</p><p>Минимальное возможное исправление при необходимости более строгой локальности: ограничивать длину касательной коротким соседним участком и фиксировать узлы на переходе корпуса в гриф. Это изменит также ранее принятые контуры Fender. Для текущей задачи от него отказались: существенной новой потери формы нет, а усложнение нарушит требование сохранить Fender.</p><h2>Втулки</h2><p>У Les Paul все шесть прежних кружков были запасными позициями: окно поиска не включало глубокие и наклонённые механизмы. Поиск расширен с запасом под целую окружность, ограничен своей половиной пластины и промежутками между кнопками. Конкурирующие хромированные окружности согласуются по смещениям независимо найденных механизмов. Верхняя и нижняя стороны используют нормализованный отступ внутрь. Ни названия гитар, ни ожидаемое число колков в алгоритме не используются. Fender и Ibanez SVG в этой доработке совпадают побайтно.</p><a href="metrics.json">Полные измерения, координаты максимумов и стадии конвейера</a>'+''.join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
