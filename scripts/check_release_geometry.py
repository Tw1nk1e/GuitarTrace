"""Compare exported alpha silhouettes without models; write only masked previews.

uv run python scripts/check_release_geometry.py before
uv run python scripts/check_release_geometry.py after
Requires existing transparent samples; no downloads or inference.
"""
import base64
import io
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gbcv import trace as tr
from gbcv.imageio import load_image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'out/release-geometry'


def snapshot(name, stage):
    rgba = load_image(ROOT / 'samples/dl' / name, mode='RGBA')
    mask = (rgba[..., 3] >= 128).astype(np.uint8)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(cs, key=cv2.contourArea)[:, 0].astype(float)
    # Nominal coordinates, as in the exporter; no changes to segmentation.
    scale = 1000 / np.ptp(points, axis=0).max()
    points *= scale
    path = tr.curve_path(tr.smoothed(points, .8), .4)
    h, w = mask.shape
    width, height = w * scale, h * scale
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><path d="{path}" fill="none" stroke="black" stroke-width="2"/></svg>\n'
    folder = DEST / Path(name).stem
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{stage}.svg').write_text(svg, encoding='utf-8')
    # Never display pixels outside the largest guitar component.
    guitar = np.zeros_like(mask)
    cv2.drawContours(guitar, [max(cs, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    photo = np.where((guitar & mask)[..., None] > 0, rgba[..., :3], 255).astype(np.uint8)
    preview = Image.fromarray(photo)
    preview.thumbnail((1000, 1000))
    stream = io.BytesIO()
    preview.save(stream, format='PNG')
    image = base64.b64encode(stream.getvalue()).decode()
    overlay = svg.replace('<path ', f'<image width="{width}" height="{height}" href="data:image/png;base64,{image}"/><path ').replace('stroke="black"', 'stroke="red"')
    (folder / f'{stage}-overlay.svg').write_text(overlay, encoding='utf-8')
    # Raster overlay from the same exported cubics for quick visual review.
    values = np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', path)])
    anchor = values[:2]
    canvas = np.array(preview)
    resize = preview.width / width
    ts = np.linspace(0, 1, 32)[:, None]
    for controls in values[2:].reshape(-1, 3, 2):
        a, b, end = controls
        curve = (1-ts)**3*anchor + 3*(1-ts)**2*ts*a + 3*(1-ts)*ts**2*b + ts**3*end
        cv2.polylines(canvas, [np.round(curve * resize).astype(np.int32)], False, (255, 0, 0), 1, cv2.LINE_AA)
        anchor = end
    Image.fromarray(canvas).save(folder / f'{stage}-overlay.png')
    if stage == 'after':
        assert (folder / 'before.svg').read_text(encoding='utf-8') == svg, name
        assert (folder / 'before-overlay.png').read_bytes() == (folder / 'after-overlay.png').read_bytes(), name
    return f'<h2>{name}</h2><img src="{Path(name).stem}/{stage}-overlay.png">'


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage not in {'before', 'after'}:
        raise SystemExit('Expected before or after')
    names = ['fender_1.png', 'fender_2.webp', 'lp_ref_1.webp']
    cards = [snapshot(name, stage) for name in names]
    (DEST / f'{stage}.html').write_text('<!doctype html><meta charset="utf-8"><title>GuitarTrace geometry check</title>' + ''.join(cards), encoding='utf-8')
    print(f'{len(names)} existing alpha silhouettes: {stage}; ' + ('SVG and overlays byte-identical' if stage == 'after' else 'baseline saved'))
