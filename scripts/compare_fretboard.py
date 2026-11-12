"""Compare only fretboard geometry on cached masks, without any ML inference."""

import copy
import json
import time
from unittest.mock import patch

import cv2
import numpy as np

from debug_fretboard import NAMES, ROOT, geometry, load_image, save, tr


def main():
    dest = ROOT / "out" / "fretboard-edges"
    dest.mkdir(exist_ok=True)
    results = {}
    for name in NAMES:
        source = next((ROOT / "samples" / "dl").glob(name + ".*"))
        cached = cv2.imread(str(ROOT / "out" / name / "mask.png"), cv2.IMREAD_GRAYSCALE)
        with patch.object(tr, "photo_board"):
            old = geometry(load_image(source, mode="RGBA"), cached)
        new = copy.deepcopy(old)
        started = time.perf_counter()
        tr.photo_board(new)
        seconds = time.perf_counter() - started
        d = new.board_detection
        folder = dest / name
        folder.mkdir(exist_ok=True)
        for label, t in (("old", old), ("new", new)):
            save(folder / f"{label}-overlay.png", tr.overlay(t))
            (folder / f"{label}.svg").write_text(tr.svg(t), encoding="utf-8")
        img = tr.masked_photo(old)
        for t, colour in ((old, (35, 95, 255)), (new, (0, 185, 50))):
            px = lambda q: np.round(t.ruler.to_px(q)).astype(np.int32)
            cv2.polylines(img, [px(t.board)], True, colour, 2)
            for segment in tr.fret_segments(t):
                cv2.polylines(img, [px(segment)], False, colour, 1)
        # Rejected hypotheses stay dashed magenta, never labelled as new geometry.
        if not d.accepted and d.edges is not None:
            for edge in d.edges:
                for u in np.arange(0, 1, .035):
                    p = np.round(edge[0] + (edge[1] - edge[0]) * u).astype(int)
                    q = np.round(edge[0] + (edge[1] - edge[0]) * min(1, u + .02)).astype(int)
                    cv2.line(img, tuple(p), tuple(q), (240, 0, 190), 1)
        save(folder / "comparison.png", img)
        board = old.ruler.to_px(old.board)
        lo = np.floor(board.min(0) - [50, 35]).astype(int).clip(0)
        hi = np.ceil(board.max(0) + [25, 35]).astype(int)
        save(folder / "closeup.png", img[lo[1]:hi[1], lo[0]:hi[0]])
        panels = []
        for label, t in (("OLD", old), ("NEW", new)):
            crop = tr.overlay(t)[lo[1]:hi[1], lo[0]:hi[0]]
            panel = cv2.copyMakeBorder(crop, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text = label if label == "OLD" else f"NEW: confidence={d.confidence:.3f}, fallback={not d.accepted}"
            cv2.putText(panel, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 1, cv2.LINE_AA)
            panels.append(panel)
        save(folder / "before-after.png", np.vstack(panels))
        summary = {"confidence": round(d.confidence, 4), "accepted": d.accepted,
                   "fallback": not d.accepted, "reason": d.reason,
                   "seconds_edges_only": round(seconds, 4), "frets": len(new.frets),
                   "fret_count_unchanged": len(old.frets) == len(new.frets),
                   "svg_identical": tr.svg(old) == tr.svg(new),
                   "max_board_vertex_shift_px": round(float(np.linalg.norm(new.ruler.to_px(new.board) - board, axis=1).max()), 3),
                   "diagnostics": d.diagnostics, "not_ground_truth": True}
        if not d.accepted:
            assert summary["svg_identical"]
            for style in tr.STYLES:
                for smooth in (0, 1.5, 5):
                    assert tr.svg(old, style, smooth) == tr.svg(new, style, smooth)
            assert tr.overlay_svg(old) == tr.overlay_svg(new)
        results[name] = summary
        print(name, json.dumps(summary, ensure_ascii=False))
    (dest / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
