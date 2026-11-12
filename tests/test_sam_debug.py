from types import SimpleNamespace

import cv2
import numpy as np

from gbcv import trace as tr
from gbcv.sam_debug import SamDebug


def fixture(monkeypatch):
    from gbcv import sam

    shape = (180, 260)
    guitar = np.full(shape, 255, np.uint8)
    part = np.zeros(shape, np.uint8)
    part[40:70, 40:70] = 1
    small = np.zeros(shape, np.uint8)
    small[90:92, 50:52] = 1
    proposals = [(part, .95, 0), (part, .90, 0), (part, .2, 0),
                 (np.ones(shape, np.uint8), .99, 0), (small, .98, 0)]

    class Session:
        def __init__(self, crop):
            self.k = 1.0
            assert crop.shape[:2] == shape

        def each(self, points, min_score):
            return [(m.copy(), s, i) for m, s, i in proposals if s >= min_score]

    monkeypatch.setattr(sam, "Session", Session)
    ruler = SimpleNamespace(d=1., to_px=lambda p: np.asarray(p, float) + [0, 90],
                            to_mm=lambda p: np.asarray(p, float) - [0, 90])
    photo = np.full((*shape, 3), 50, np.uint8)
    photo[40:70, 40:70] = 230
    return photo, guitar, ruler


def test_debug_preserves_parts_filters_and_svg(monkeypatch, tmp_path):
    photo, mask, ruler = fixture(monkeypatch)
    args = (photo, mask, ruler, 190, 80, 20)
    normal = tr._sam_parts(*args)
    debug = SamDebug(tmp_path)
    traced = tr._sam_parts(*args, debug=debug)
    assert len(normal) == len(traced) == 1
    np.testing.assert_array_equal(normal[0], traced[0])
    assert [r.get("reason") for r in debug.records] == [None, "duplicate", "score", "area_max", "area_min"]
    assert debug.records[1]["duplicate_of"] == 0
    t = tr.Trace(np.array([[0, -90], [260, -90], [260, 90], [0, 90]], float),
                 220, 648, [], np.array([[220, -20], [190, -20], [190, 20], [220, 20]], float),
                 photo=photo, ruler=ruler, mask=mask)
    old = tr._real_parts(t, normal)
    new = tr._real_parts(t, traced, debug=debug)
    assert len(old) == len(new) == 1
    np.testing.assert_array_equal(old[0], new[0])
    t.parts = new
    assert tr.svg(t, "png") == tr.svg(t, "png", sam_debug=debug)
    assert debug.records[0]["status"] == "kept"
    assert "svg_path" in debug.records[0]
    for stage in ("raw", "mask_smoothed", "approximated", "export_smoothed"):
        assert f"c0_{stage}_contour" in debug.arrays
    with np.load(tmp_path / "sam-cache.npz", allow_pickle=False) as z:
        assert len(z["masks"]) == 5  # includes low-confidence candidates
    debug.finish(t, tr.svg(t, "png"))
    assert (tmp_path / "index.html").exists()
    assert 'id="c0"' in (tmp_path / "parts.svg").read_text()


def test_weak_edge_has_explicit_reason(monkeypatch, tmp_path):
    photo, mask, ruler = fixture(monkeypatch)
    debug = SamDebug(tmp_path)
    parts = tr._sam_parts(photo, mask, ruler, 190, 80, 20, debug=debug)
    t = SimpleNamespace(outline=np.array([[0, -90], [260, -90], [260, 90], [0, 90]], float),
                        photo=np.zeros_like(photo), mask=mask, ruler=ruler)
    assert not tr._real_parts(t, parts, debug=debug)
    assert debug.records[0]["reason"] == "weak_edge"
    assert debug.records[0]["edge_step"] == 0


def test_debug_photo_hides_pixels_outside_guitar(monkeypatch, tmp_path):
    photo, mask, ruler = fixture(monkeypatch)
    debug = SamDebug(tmp_path)
    parts = tr._sam_parts(photo, mask, ruler, 190, 80, 20, debug=debug)
    mask[:20] = 0
    photo[:20] = [255, 0, 100]
    t = tr.Trace(np.array([[0, -70], [260, -70], [260, 90], [0, 90]], float),
                 220, 648, [], np.array([[220, -20], [190, -20], [190, 20], [220, 20]], float),
                 photo=photo, ruler=ruler, mask=mask)
    t.parts = tr._real_parts(t, parts, debug=debug)
    debug.finish(t, tr.svg(t, "png", sam_debug=debug))
    saved = cv2.imread(str(tmp_path / "photo.png"))
    assert (saved[:20] == 255).all()
