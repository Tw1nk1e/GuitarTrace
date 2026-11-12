"""Local cached-head validation. --baseline BEFORE editing head_parts; no ML.

Keeps cached body/SAM elements; replaces only outline, keys and posts after
checking their baseline against the cached full SVG. Not an ML end-to-end run.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

from debug_fretboard import ROOT, geometry, load_image, no_ml, save, tr
from gbcv.orient import orient_and_split
from gbcv.split import split

DEST = ROOT / "out" / "validation-headstock"
NAMES = ("fender_1", "lp_ref_1", "ibanez_rgms8")
EXPECTED = (6, 6, 8)  # acceptance fixtures only, never used by the classifier


def sample_path(path, step=.1):
    """Sample the actual rounded SVG coordinates, not the pre-export polygon."""
    tokens = re.findall(r"[MCLZ]|-?\d+(?:\.\d+)?", path)
    points = []
    i = 0
    while i < len(tokens):
        command = tokens[i]
        i += 1
        if command == "Z":
            break
        count = 6 if command == "C" else 2
        values = np.array(tokens[i:i+count], float).reshape(-1, 2)
        i += count
        if command == "M":
            current = values[0]
            points.append(current[None])
            continue
        controls = np.vstack([current, values])
        length = np.linalg.norm(np.diff(controls, axis=0), axis=1).sum()
        u = np.linspace(0, 1, max(2, int(np.ceil(length / step)) + 1))[:, None]
        if command == "C":
            a, b, c, d = controls
            points.append((1-u)**3*a + 3*(1-u)**2*u*b + 3*(1-u)*u*u*c + u**3*d)
        else:
            points.append((1-u)*current + u*values[0])
        current = values[-1]
    return np.vstack(points)


def distance_to_polyline(points, line):
    from scipy.spatial import cKDTree

    a, v = line[:-1], np.diff(line, axis=0)
    indices = cKDTree(a + v / 2).query(points, k=8)[1]
    aa, vv = a[indices], v[indices]
    relative = points[:, None] - aa
    fraction = np.clip(np.sum(relative * vv, axis=2) / np.maximum(np.sum(vv*vv, axis=2), 1e-12), 0, 1)
    return np.linalg.norm(relative - fraction[..., None] * vv, axis=2).min(1)


def describe(key, plate):
    y, x = np.nonzero(key)
    pts = np.c_[x, y].astype(np.float32)
    rect = cv2.minAreaRect(pts)
    a, b = sorted(rect[1])
    hull = cv2.convexHull(pts)
    perimeter = cv2.arcLength(hull, True)
    contact = (cv2.dilate(key.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (plate > 0)
    cy, cx = np.nonzero(contact)
    return {"area_px": len(x), "centre_px": [float(x.mean()), float(y.mean())],
            "rotated_aspect": float(b / max(a, 1)),
            "rotated_fill": float(len(x) / max(a * b, 1)),
            "hull_circularity": float(4 * np.pi * cv2.contourArea(hull) / max(perimeter**2, 1)),
            "bbox": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
            "contact_centre": [float(cx.mean()), float(cy.mean())] if len(cx) else None}


def svg_parts(text, posts, keys):
    root = ET.fromstring(text)
    start = next(i for i, e in enumerate(root) if e.tag.endswith("circle"))
    end = start + 2 * posts + keys
    return root, start, end


def same(a, b):
    return a.tag == b.tag and a.attrib == b.attrib


def head_overlay(t, outline, keys, posts):
    t.outline, t.keys, t.posts = outline, keys, posts
    return tr.overlay(t)


def validate_posts(gray, ruler, plate, keys, posts):
    """Independent photo/geometry checks, not equality to historical guesses.

    Match posts to keys one-to-one on their own side of the plate. Require a
    circular photo edge around each centre, sampled directly from Canny edges
    (not the production Hough result). The radius scan tolerates washers and
    posts with different visible radii. These catalogue fixtures expose them;
    this is not a requirement for all production photos or hidden mechanisms.
    """
    from scipy.optimize import linear_sum_assignment

    assert len(posts) == len(keys) > 0, "One post is required per accepted button"
    positions = np.asarray(posts, float)
    assert np.isfinite(positions).all() and (positions[:, 2] > 0).all()
    px = ruler.to_px(positions[:, :2])
    h, w = plate.shape
    assert ((px >= 0).all() and (px[:, 0] < w).all() and (px[:, 1] < h).all()), "Post outside photo"
    centres = np.round(px).astype(int)
    assert (plate[centres[:, 1], centres[:, 0]] > 0).all(), "Post outside head plate"
    locations = []
    for key in keys:
        yy, xx = np.nonzero(key)
        x, y = xx.mean(), yy.mean()
        column = np.flatnonzero(plate[:, int(x)])
        assert column.size
        side = -1 if abs(y-column.min()) < abs(y-column.max()) else 1
        edge = column.min() if side < 0 else column.max()
        locations.append((x, edge, side, .5*(column[-1]-column[0])))
    cost = np.full((len(keys), len(posts)), 1e9)
    for i, (x, edge, side, half) in enumerate(locations):
        row = [q[0] for q in locations if q[2] == side]
        left = (max((q for q in row if q < x), default=-np.inf)+x)/2
        right = (min((q for q in row if q > x), default=np.inf)+x)/2
        inward = -side*(px[:, 1]-edge)
        valid = (px[:, 0] > left) & (px[:, 0] < right) & (inward > 0) & (inward < half)
        cost[i, valid] = np.abs(px[valid, 0]-x)
    rows, cols = linear_sum_assignment(cost)
    assert (cost[rows, cols] < 1e9).all(), "Post is not opposite its own button on the inward side"
    edges = cv2.Canny(gray, 60, 120)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    scores = []
    for centre in px:
        best = 0.
        for radius in np.linspace(3*ruler.d, 8*ruler.d, 31):
            ring = np.round(centre+radius*directions).astype(int)
            if ((ring < 0).any() or (ring[:, 0] >= w).any() or (ring[:, 1] >= h).any()):
                continue
            supported = distance[ring[:, 1], ring[:, 0]] <= max(1.5, .8*ruler.d)
            best = max(best, float(supported.mean()))
        scores.append(best)
    assert min(scores) >= .6, f"Post has no circular photo evidence: {scores}"
    separation = np.linalg.norm(px[:, None]-px, axis=2)
    np.fill_diagonal(separation, np.inf)
    assert separation.min() > 3*ruler.d, "Duplicate mechanism centres"
    return {"post_key_assignment": cols.tolist(), "post_circle_edge_support": scores,
            "post_positions_valid": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    summary = {}
    with patch("gbcv.segment.probability", no_ml), patch("gbcv.sam.Session", no_ml):
        for name, expected in zip(NAMES, EXPECTED):
            folder = DEST / name
            folder.mkdir(exist_ok=True)
            rgba = load_image(next((ROOT / "samples" / "dl").glob(name + ".*")), mode="RGBA")
            mask = cv2.imread(str(ROOT / "out" / name / "mask.png"), 0)
            t = geometry(rgba, mask)
            _, cleaned, _ = orient_and_split(tr.silhouette(rgba), open_body=False)
            sp = split(cleaned, nut_x=float(t.ruler.ff.b))
            gray = cv2.cvtColor(t.photo, cv2.COLOR_RGB2GRAY)
            # gray is optional so the same baseline runner works before the fix.
            import inspect
            kwargs = {"gray": gray} if "gray" in inspect.signature(tr.head_parts).parameters else {}
            solid, plate, keys = tr.head_parts(mask, t.ruler, float(t.ruler.ff.b), **kwargs)
            outline = tr.outline_of(tr.head_closed(solid, sp), t.ruler)
            key_polys = [tr._poly_mm(k, t.ruler, .3) for k in keys]
            assert all(q is not None for q in key_polys)
            posts = tr._posts(gray, t.ruler, plate, keys)
            metrics = {"keys": len(keys), "posts": len(posts), "candidates": [describe(k, plate) for k in keys]}
            if args.baseline:
                assert not (folder / "baseline.npz").exists(), "Do not overwrite the pre-fix baseline"
                cached = (ROOT / "out" / name / "trace.svg").read_bytes()
                t.outline, t.keys, t.posts = outline, key_polys, posts
                fresh = tr.svg(t)
                a, start, end = svg_parts(cached.decode(), len(posts), len(keys))
                b, bs, be = svg_parts(fresh, len(posts), len(keys))
                assert start == bs
                assert same(a[0], b[0])
                assert all(same(x, y) for x, y in zip(list(a)[start:end], list(b)[bs:be]))
                np.savez_compressed(folder / "baseline.npz", solid=solid, plate=plate,
                                    keys=np.array(keys), outline=outline, posts=np.array(posts))
                (folder / "before.svg").write_bytes(cached)
                (folder / "baseline.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                print(name, json.dumps(metrics))
                continue
            old = np.load(folder / "baseline.npz")
            assert len(keys) == len(posts) == expected
            metrics.update(validate_posts(gray, t.ruler, plate, keys, posts))
            old_keys = list(old["keys"])
            old_polys = [tr._poly_mm(k, t.ruler, .3) for k in old_keys]
            before = (folder / "before.svg").read_text(encoding="utf-8")
            t.outline, t.keys, t.posts = outline, key_polys, posts
            fresh = tr.svg(t)
            cached, start, end = svg_parts(before, len(old["posts"]), len(old_keys))
            new, ns, ne = svg_parts(fresh, len(posts), len(keys))
            assert start == ns
            assert all(same(a, b) for a, b in zip(list(cached)[1:start], list(new)[1:ns]))
            # No cached board, fret, body part or hardware is regenerated.
            replacement = [new[0], *list(cached)[1:start], *list(new)[ns:ne], *list(cached)[end:]]
            # Use the live export's bounds: the restored tip needs its normal
            # padding, otherwise the old (shortened head) viewBox clips it.
            result = ET.Element(cached.tag, new.attrib)
            result.extend(replacement)
            after = ET.tostring(result, encoding="unicode").replace("ns0:", "").replace(":ns0", "") + "\n"
            # Keep original bytes when no head data changed (Fender acceptance).
            if same(cached[0], new[0]) and len(old_keys) == len(keys) and np.array_equal(old["posts"], posts) and all(np.array_equal(a, b) for a, b in zip(old_keys, keys)):
                after = before
            (folder / "after.svg").write_text(after, encoding="utf-8")
            for label, out, kp, pp in (("before", old["outline"], old_polys, old["posts"]),
                                      ("after", outline, key_polys, posts)):
                image = head_overlay(t, out, kp, pp)
                save(folder / f"{label}-overlay.png", image)
                head_pixels = np.argwhere((mask > 0) & (np.indices(mask.shape)[1] > t.ruler.ff.b))
                y0, x0 = head_pixels.min(0) - [20, 30]
                y1, x1 = head_pixels.max(0) + [21, 31]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
                crop = image[y0:y1, x0:x1]
                save(folder / f"{label}-head.png", cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
            photo = tr.masked_photo(t)
            save(folder / "photo-head.png", cv2.resize(photo[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
            pairs = []
            for key in keys:
                matches = [i for i, old_key in enumerate(old_keys) if np.array_equal(key, old_key)]
                assert len(matches) == 1, "Real key mask changed or new key was invented"
                pairs.extend(matches)
            removed = [i for i in range(len(old_keys)) if i not in pairs]
            rejected_pixels = np.any(old["keys"][removed], axis=0) if removed else np.zeros_like(mask, bool)
            body = np.indices(mask.shape)[1] < t.ruler.ff.b - 10 * t.ruler.d
            assert np.array_equal(solid[body], old["solid"][body])
            assert (solid[rejected_pixels] > 0).all()
            assert (plate[rejected_pixels] > 0).all()
            # Historical positions include synthetic guesses (notably Gibson).
            # Keep identity as a measurement, never as the correctness criterion.
            posts_unchanged = all(any(np.array_equal(p, q) for q in old["posts"]) for p in posts)
            old_curve = tr.curve_path(tr.smoothed(old["outline"], tr.DEFAULT_SMOOTH_MM), tr.CURVE_TOL_MM)
            new_curve = tr.curve_path(tr.smoothed(outline, tr.DEFAULT_SMOOTH_MM), tr.CURVE_TOL_MM)
            a, b = sample_path(cached[0].attrib["d"]), sample_path(new[0].attrib["d"])
            old_body, new_body = a[a[:, 0] < t.nut_x - 20], b[b[:, 0] < t.nut_x - 20]
            old_pixels, new_pixels = t.ruler.to_px(a), t.ruler.to_px(b)
            deviation = np.r_[distance_to_polyline(t.ruler.to_px(old_body), new_pixels),
                              distance_to_polyline(t.ruler.to_px(new_body), old_pixels)]
            # Global closed-curve resampling can shift the subpixel fit even
            # though the underlying non-head mask is bit-for-bit unchanged.
            # Do not hide global curve-fit drift behind the bit-identical mask
            # check. Report it separately; fixing the general SVG fitter is not
            # part of head/button classification.
            outside_elements_same = (all(same(x, y) for x, y in zip(list(cached)[1:start], replacement[1:start]))
                                     and all(same(x, y) for x, y in zip(list(cached)[end:], replacement[start+ne-ns:])))
            assert outside_elements_same
            metrics.update(old_keys=len(old_keys), old_posts=len(old["posts"]),
                           kept_key_indices=pairs, removed_key_indices=removed,
                           rejected_pixels_restored_to_solid=int(((solid > 0) & rejected_pixels).sum()),
                           rejected_pixels_total=int(rejected_pixels.sum()),
                           outside_head_mask_identical=True,
                           outside_head_svg_elements_identical=outside_elements_same,
                           outside_head_exported_outline_max_deviation_px=float(deviation.max()),
                           outside_head_exported_outline_mean_deviation_px=float(deviation.mean()),
                           outside_head_exported_outline_identical=bool(deviation.max() < 1e-9),
                           outline_svg_identical=old_curve == new_curve,
                           posts_unchanged_for_retained_keys=posts_unchanged,
                           ml_executed=False)
            summary[name] = metrics
            print(name, json.dumps({k: metrics[k] for k in (
                "old_keys", "keys", "old_posts", "posts", "rejected_pixels_total",
                "post_positions_valid", "posts_unchanged_for_retained_keys", "outside_head_svg_elements_identical",
                "outside_head_exported_outline_max_deviation_px")}, ensure_ascii=False))
    if not args.baseline:
        (DEST / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        sections = []
        for name in NAMES:
            sections.append(f'<h2>{name}</h2><p><a href="{name}/before.svg">До: полный SVG</a> · <a href="{name}/after.svg">После: полный SVG</a></p><h3>Фото / до / после, ×3</h3>' + ''.join(f'<img src="{name}/{label}-head.png">' for label in ('photo', 'before', 'after')) + f'<div class="pair"><object data="{name}/before.svg" type="image/svg+xml"></object><object data="{name}/after.svg" type="image/svg+xml"></object></div>')
        drifts = '; '.join(f'{name}: {summary[name]["outside_head_exported_outline_max_deviation_px"]:.3f} исходных px' for name in NAMES)
        (DEST / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Голова: до / после</title><style>body{font:16px sans-serif;margin:24px}img{display:block;margin:10px 0;max-width:none}.pair{display:grid;grid-template-columns:1fr 1fr}object{width:100%;height:350px}</style><h1>Классификация выступов головы</h1><p>Только существующие маски и фото. Без ML. SAM-детали корпуса сохранены из исходного SVG. Фото только внутри маски гитары.</p><p>Проверки: кнопки 6/6/8; колки находятся внутри пластины напротив своих кнопок и подтверждены круговыми кромками фото. Старые ошибочные координаты не являются эталоном. Отвергнутые выступы целиком восстановлены в пластине и силуэте. Fender SVG неизменен. Вне головы маска и отдельные SVG-элементы неизменны, но глобальная аппроксимация внешнего контура даёт максимальное отклонение: ' + drifts + '. Это остаточная проблема общей аппроксимации SVG, не новая геометрия маски. Полная неизменность внешнего SVG-контура вне головы НЕ достигнута.</p><a href="metrics.json">Измерения</a>' + ''.join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
