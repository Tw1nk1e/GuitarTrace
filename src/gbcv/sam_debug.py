"""Opt-in, local SAM provenance. Never writes unmasked source photographs."""

import html
import json
from pathlib import Path

import cv2
import numpy as np


class SamDebug:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.part_ids = {}
        self.arrays = {}

    def start(self, proposals, points, k, crop, guitar_mask):
        self.k, self.crop = k, crop
        self.guitar_mask = guitar_mask
        self.points = points
        for cid, (mask, score, point) in enumerate(proposals):
            self.records.append(dict(id=cid, score=float(score), prompt=int(point),
                                     prompt_crop_px=points[point].tolist(), raw_area_px=int(np.count_nonzero(mask)), status="pending"))
            self.mask(cid, "raw", mask)
        # Persist the costly inference immediately, before downstream processing.
        np.savez_compressed(self.directory / "sam-cache.npz",
                            masks=np.asarray([m for m, _, _ in proposals], dtype=np.uint8),
                            scores=np.asarray([s for _, s, _ in proposals]),
                            prompts=np.asarray([i for _, _, i in proposals]),
                            points=points, k=k, crop=crop, guitar_mask=guitar_mask)

    def mask(self, cid, stage, mask):
        self.arrays[f"c{cid}_{stage}_mask"] = np.asarray(mask, np.uint8)

    def contour(self, cid, stage, points):
        self.arrays[f"c{cid}_{stage}_contour"] = points.copy()

    def reject(self, cid, reason, **metrics):
        self.records[cid].update(status="rejected", reason=reason, **metrics)

    def export(self, original, smoothed, path, width):
        cid = self.part_ids[id(original)]
        self.contour(cid, "export_smoothed", smoothed)
        self.records[cid].update(svg_path=path, svg_stroke_width=width)

    def finish(self, t, svg):
        from .trace import masked_photo

        dest = self.directory
        np.savez_compressed(dest / "stages.npz", **self.arrays)
        (dest / "candidates.json").write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        (dest / "trace.svg").write_text(svg, encoding="utf-8")
        x0, y0, x1, y1 = self.crop
        # Mask before resizing, including pixels along the antialiased edge.
        photo = masked_photo(t)[y0:y1, x0:x1]
        shape = self.arrays["c0_raw_mask"].shape[::-1] if self.records else (round(photo.shape[1]*self.k), round(photo.shape[0]*self.k))
        photo = cv2.resize(photo, shape, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest / "photo.png"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
        h, w = photo.shape[:2]
        cards, accepted = [], []
        for record in self.records:
            cid = record["id"]
            raw = self.arrays[f"c{cid}_raw_mask"]
            cv2.imwrite(str(dest / f"candidate-{cid:04d}-mask.png"), raw * 255)
            yy, xx = np.nonzero(raw)
            if len(xx):
                left, top, right, bottom = max(0, int(xx.min())-12), max(0, int(yy.min())-12), min(w, int(xx.max())+13), min(h, int(yy.max())+13)
            else:
                left, top, right, bottom = 0, 0, w, h
            image = photo.copy()
            active = (raw > 0) & (cv2.resize(self.guitar_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
            image[active] = (.65 * image[active] + .35 * np.array([40, 170, 240])).astype(np.uint8)
            cv2.putText(image, f"C{cid}", (left+3, min(h-5, top+22)), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 30, 30), 2)
            prompt = np.round(self.points[record["prompt"]] * self.k).astype(int)
            cv2.drawMarker(image, tuple(prompt), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.imwrite(str(dest / f"candidate-{cid:04d}.png"), cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_RGB2BGR))
            shapes = []
            for stage, color in (("raw", "#f90"), ("approximated", "#06f"), ("export_smoothed", "#0a0")):
                contour = self.arrays.get(f"c{cid}_{stage}_contour")
                if contour is not None:
                    points = (t.ruler.to_px(contour) - [x0, y0]) * self.k
                    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                    shapes.append(f'<polygon points="{coords}" fill="none" stroke="{color}" stroke-width="1"/>')
            # The final path is mapped from drawing units by sampled cubic points
            # in the diagnostic runner; the exact vector path is also linked below.
            metrics = html.escape(json.dumps({k: v for k, v in record.items() if k != "svg_path"}, ensure_ascii=False))
            detail = f'<svg viewBox="{left} {top} {right-left} {bottom-top}"><image href="photo.png" width="{w}" height="{h}"/>{"".join(shapes)}</svg>'
            cards.append(f'<article id="c{cid}"><h3>C{cid}: {record["status"]} {record.get("reason", "")}</h3><pre>{metrics}</pre><img loading="lazy" src="candidate-{cid:04d}.png">{detail}<a href="candidate-{cid:04d}-mask.png">raw mask</a></article>')
            if record["status"] == "kept":
                path = record["svg_path"]
                accepted.append(f'<path id="c{cid}" d="{path}" fill="none" stroke="currentColor" stroke-width="{record["svg_stroke_width"]}"/>')
        vb = " ".join(map(str, [*t.outline.min(0), *np.ptp(t.outline, axis=0)]))
        (dest / "parts.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}">{"".join(accepted)}</svg>', encoding="utf-8")
        (dest / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>SAM diagnostic</title><style>body{font:15px sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}article{border:1px solid #ccc;padding:10px}img,svg{width:100%;max-height:420px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>SAM: кандидаты и судьба каждого</h1><p>Это гипотезы, не эталон. Голубая заливка: исходная маска, крест: подсказка. Оранжевый контур: до сглаживания; синий: аппроксимация; зелёный: сглаживание экспорта. Геометрия в stages.npz; точные SVG paths в candidates.json и parts.svg.</p><p><a href="trace.svg">Итоговый SVG</a> · <a href="parts.svg">Элементы с ID</a> · <a href="candidates.json">Решения и оценки</a></p><main>'+"".join(cards)+'</main>', encoding="utf-8")
