"""Read-only diagnostic of fretboard hypotheses; no ML, no production SVG changes.

uv run python scripts/debug_fretboard.py
Uses out/<name>/mask.png and samples/dl/<name>.*. Hypotheses are NOT labels.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbcv import trace as tr
from gbcv.imageio import load_image
from gbcv.orient import orient_and_split

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "out" / "debug-fretboard"
NAMES = ("fender_1", "lp_ref_1", "ibanez_rgms8")
BLUE, ORANGE, GREEN, MAGENTA = (30, 100, 255), (255, 145, 0), (0, 200, 60), (240, 0, 190)


def save(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def no_ml(*args, **kwargs):
    raise RuntimeError("ML is disabled in this diagnostic")


def geometry(rgba, cached):
    # Recover the photo->canvas transform without a model. Retain the cached full
    # silhouette, including head/keys; orient's cleaned mask is for the split only.
    with patch("gbcv.segment.probability", no_ml):
        source_mask = tr.silhouette(rgba)
    frame, cleaned, sp = orient_and_split(source_mask, open_body=False)
    full = (frame.warp(source_mask) >= 128).astype(np.uint8) * 255
    if full.shape != cached.shape or not np.array_equal(full, cached):
        raise ValueError("Cached mask does not match recovered orientation; refusing stale geometry")
    # Execute the CURRENT fret/board branches, replacing only expensive body detail
    # extraction and reusing the cached silhouette for the full-mask warp.
    original_warp = frame.warp

    def warp(img, interp=cv2.INTER_LINEAR):
        if img is source_mask:
            return cached.copy()
        return original_warp(img, interp)

    with patch.object(tr, "silhouette", return_value=source_mask), \
         patch.object(tr, "orient_and_split", return_value=(frame, cleaned, sp)), \
         patch.object(frame, "warp", side_effect=warp), \
         patch.object(tr, "_sam_parts", return_value=[]), \
         patch.object(tr, "_real_parts", return_value=[]), \
         patch.object(tr, "_posts", return_value=[]), \
         patch.object(tr, "photo_board"):  # stage-one diagnostic always shows legacy geometry
        return tr.trace(rgba)


def hypotheses(t):
    board = t.ruler.to_px(t.board)
    lab = cv2.cvtColor(t.photo, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1)
    # Colour gradients, not just brightness: dark binding and maple/paint edges.
    gy = np.linalg.norm(cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3), axis=2)
    gx = np.linalg.norm(cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3), axis=2)
    # Longitudinal pooling suppresses short transverse fret/string intersections.
    pooled = cv2.blur(gy, (15, 1))
    x0 = int(max(board[1, 0], board[2, 0]))
    x1 = int(min(board[0, 0], board[3, 0]))
    xs = np.arange(x0 + 3, x1 - 3)
    width = float(np.mean([board[3, 1] - board[0, 1], board[2, 1] - board[1, 1]]))
    radius = max(4, int(0.23 * width))
    sides, stats = [], []
    for a, b in ((board[0], board[1]), (board[3], board[2])):
        predicted = np.interp(xs, [b[0], a[0]], [b[1], a[1]])
        ys, strengths, prominence = [], [], []
        for x, y in zip(xs, predicted):
            lo, hi = max(1, int(y) - radius), min(gy.shape[0] - 1, int(y) + radius + 1)
            v = pooled[lo:hi, x]
            j = int(v.argmax())
            ys.append(lo + j)
            strengths.append(float(v[j]))
            prominence.append(float(v[j] / (np.median(v) + 1)))
        ys = np.array(ys, float)
        keep = np.ones(len(xs), bool)
        for _ in range(3):
            fit = np.polyfit(xs[keep], ys[keep], 1)
            keep = np.abs(ys - np.polyval(fit, xs)) < max(2, 0.035 * width)
        # Draw ALL local maxima faintly as well: a fitted straight line alone can
        # hide jumps onto strings, frets or the external silhouette.
        pts = np.c_[xs, ys]
        fitted = np.c_[xs, np.polyval(fit, xs)]
        sides.append((pts, fitted))
        stats.append({
            "inlier_share": round(float(keep.mean()), 3),
            "median_gradient": round(float(np.median(strengths)), 2),
            "median_peak_to_background": round(float(np.median(prominence)), 2),
            "median_offset_from_old_px": round(float(np.median(ys - predicted)), 2),
            "search_radius_px": radius,
        })
    # Search the bodyward end as parallel translations of the last fret, rather
    # than assuming a square end. Keep competing peaks visible, not only the winner.
    xa, xb, h = t.frets[-1]
    last = t.ruler.to_px(np.array([[xa, -h], [xb, h]]))
    yy = np.linspace(last[0, 1] + .12 * width, last[1, 1] - .12 * width, 60)
    base = np.interp(yy, last[:, 1], last[:, 0])
    offsets = np.arange(-max(8, int(.65 * width)), -2)
    scores = []
    for dx in offsets:
        px, py = np.round(base + dx).astype(int), np.round(yy).astype(int)
        scores.append(float(np.median(gx[py, px])))
    order = np.argsort(scores)[::-1]
    selected = []
    for i in order:
        if all(abs(int(offsets[i]) - q[0]) >= 5 for q in selected):
            selected.append((int(offsets[i]), float(scores[i])))
        if len(selected) == 3:
            break
    ends = [(last + [dx, 0], score) for dx, score in selected]
    return board, sides, ends, gy, {"side_candidates": stats,
        "end_candidates": [{"shift_from_last_fret_px": dx, "median_gradient": round(sc, 2)} for dx, sc in selected],
        "end_search_shift_px": [int(offsets[0]), int(offsets[-1])],
        "not_ground_truth": True,
        "limitations": "Local gradient maxima near old geometry; no independent board classification or confidence calibration. End search assumes last-fret slope."}


def draw(t, board, sides, ends):
    img = tr.masked_photo(t)
    cv2.polylines(img, [np.round(board).astype(np.int32)], True, BLUE, 2)
    for xa, xb, h in t.frets:
        p = np.round(t.ruler.to_px(np.array([[xa, -h], [xb, h]]))).astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[1]), ORANGE, 1)
    for pts, fitted in sides:
        for x, y in pts[::3].astype(int):
            cv2.circle(img, (x, y), 1, (130, 235, 150), -1)
        cv2.polylines(img, [np.round(fitted).astype(np.int32)], False, GREEN, 2)
    for i, (end, _) in enumerate(ends):
        p = np.round(end).astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[1]), MAGENTA if i == 0 else (245, 155, 225), 2 if i == 0 else 1)
    return img


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in NAMES:
        source = next((ROOT / "samples" / "dl").glob(name + ".*"))
        cached_path = ROOT / "out" / name / "mask.png"
        cached = cv2.imread(str(cached_path), cv2.IMREAD_GRAYSCALE)
        if cached is None:
            raise ValueError(f"Missing cached mask: {cached_path}")
        t = geometry(load_image(source, mode="RGBA"), cached)
        board, sides, ends, gy, metrics = hypotheses(t)
        folder = DEST / name
        folder.mkdir(exist_ok=True)
        save(folder / "oriented.png", tr.masked_photo(t))
        diagnostic = draw(t, board, sides, ends)
        save(folder / "diagnostic.png", diagnostic)
        lo = np.floor(board.min(0) - [80, 45]).astype(int).clip(0)
        hi = np.ceil(board.max(0) + [35, 45]).astype(int)
        crop = diagnostic[lo[1]:hi[1], lo[0]:hi[0]]
        save(folder / "fretboard.png", crop)
        heat = cv2.applyColorMap(np.clip(gy / (np.percentile(gy[t.mask > 0], 98) + 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat[t.mask == 0] = 255
        cv2.imwrite(str(folder / "gradient.png"), heat[lo[1]:hi[1], lo[0]:hi[0]])
        metrics.update(source=str(source.relative_to(ROOT)), cached_mask=str(cached_path.relative_to(ROOT)),
                       cached_mask_verified=True, frets=len(t.frets),
                       old_board_canvas_px=np.round(board, 2).tolist())
        records[name] = metrics
        print(name, json.dumps(metrics, ensure_ascii=False))
    (DEST / "measurements.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
