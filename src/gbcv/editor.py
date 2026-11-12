"""Lossless editor objects from the existing SVG renderer; no detection or inference."""

import xml.etree.ElementTree as ET
from dataclasses import replace

from . import trace as tr


def added_part(t: tr.Trace, part, smooth: float = tr.DEFAULT_SMOOTH_MM) -> dict:
    """One candidate through the existing exporter; never change the session's Trace."""
    candidate = replace(t, parts=[part], knobs=[])
    return next(obj for obj in document(candidate, smooth)['objects'] if obj['category'] == 'hardware')


def document(t: tr.Trace, smooth: float = tr.DEFAULT_SMOOTH_MM) -> dict:
    objects = []
    for style in tr.STYLES:
        root = ET.fromstring(tr.svg(t, style, smooth, editor=True))
        for order, group in enumerate(root):
            primitives = []
            for node in group:
                geometry = {k: v for k, v in node.attrib.items() if k in {"d", "cx", "cy", "r"}}
                primitives.append({
                    "type": node.tag.rsplit("}", 1)[-1],
                    "geometry": geometry,
                    "style": {k: v for k, v in node.attrib.items() if k not in geometry},
                })
            if style == next(iter(tr.STYLES)):
                objects.append({"id": group.attrib["id"], "type": "compound" if len(primitives) > 1 else primitives[0]["type"],
                                "category": group.attrib["data-category"], "order": order,
                                "primitives": [{"type": p["type"], "geometry": p["geometry"], "styles": {}} for p in primitives]})
            obj = objects[order]
            assert obj["id"] == group.attrib["id"]
            for target, primitive in zip(obj["primitives"], primitives, strict=True):
                assert target["geometry"] == primitive["geometry"]
                target["styles"][style] = primitive["style"]
    h, w = t.mask.shape
    # Exact drawing-units -> photo-pixels homography (also supports perspective).
    # Keep the SVG geometry untouched; the overlay alone projects it onto the photo.
    r = t.ruler
    a, b, c = r.ff.a, r.ff.b - r.ff.a * r.u_butt, r.ff.c
    den = 1 + c * r.u_butt
    k, m = r.centre
    return {"version": 1, "viewBox": root.attrib["viewBox"], "objects": objects,
            "manualStyles": {name: {"outer": widths["det"], "inner": widths["thin"]}
                             for name, widths in tr.STYLES.items()},
            "photo": {"width": w, "height": h,
                      "projection": [a, 0, b, k*a - m*c, r.d, k*b + m*den, -c, 0, den]}}
