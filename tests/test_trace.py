import numpy as np

from gbcv.trace import Trace, silhouette, svg


def test_silhouette_from_alpha_fills_holes():
    rgba = np.zeros((60, 80, 4), np.uint8)
    rgba[10:50, 10:70, 3] = 255
    rgba[25:35, 30:40, 3] = 0  # a hole (f-hole, gap between keys)
    m = silhouette(rgba)
    assert m[30, 35] == 255 and m[5, 5] == 0


def test_svg_has_no_text_and_draws_every_part():
    from gbcv.trace import board_sides

    t = Trace(
        outline=np.array([[0, -100], [900, -20], [1000, 0], [900, 20], [0, 100]], float),
        nut_x=800, scale_length=628, frets=[(700, 700, 22), (650, 650, 24)],
        board=np.array([[800, -21], [600, -26], [600, 26], [800, 21]], float),
        posts=[(850, -30, 5), (850, 30, 5)],
        keys=[np.array([[850, -60], [880, -60], [880, -40], [850, -40]], float)],
        parts=[np.array([[200, -40], [260, -40], [260, 40], [200, 40]], float)],
    )
    s = svg(t)
    # outline, the board's own long edges (only where they are not the neck's), the board's end, the nut, 2 frets,
    # the part and the key
    assert "<text" not in s and s.count("<circle") == 4
    assert s.count("<path") == 1 + len(board_sides(t)) + 1 + 1 + 2 + 1 + 1


def test_remove_takes_the_smallest_item_under_the_point():
    from gbcv.trace import remove_at

    big = np.array([[0, -50], [300, -50], [300, 50], [0, 50]], float)
    small = np.array([[100, -10], [120, -10], [120, 10], [100, 10]], float)
    t = Trace(outline=big, nut_x=800, scale_length=648, frets=[], board=big, posts=[(500, 0, 5)], parts=[big, small])
    assert remove_at(t, 110, 0) and t.parts == [big]
    assert remove_at(t, 501, 1) and t.posts == []
    assert not remove_at(t, 900, 0)


def _board_trace():
    from types import SimpleNamespace

    from gbcv import frets

    # Deliberately asymmetric board edges, including a fanned nut/end and frets.
    sp = SimpleNamespace(nut_x=570, butt_x=70, centre_fit=(0, 120))
    ruler = frets.Ruler.linear(sp, 1)
    board = np.array([[500, -30], [0, -50], [8, 50], [510, 30]], float)
    return Trace(
        outline=np.array([[-40, -100], [560, -100], [560, 100], [-40, 100]], float),
        nut_x=500, scale_length=648,
        frets=[(420, 428, 34), (210, 218, 42), (20, 28, 49)], board=board,
        photo=np.full((240, 640, 3), 120, np.uint8),
        mask=np.full((240, 640), 255, np.uint8), ruler=ruler,
    )


def test_photo_board_fallback_keeps_exact_svg_and_overlay(monkeypatch):
    from gbcv import fretboard, trace as tr

    t = _board_trace()
    board = t.board.copy()
    before = {(style, smooth): svg(t, style, smooth) for style in tr.STYLES for smooth in (0, 1.5, 5)}
    overlay = tr.overlay_svg(t)
    bitmap = tr.overlay(t)
    monkeypatch.setattr(fretboard, "detect", lambda *a: fretboard.Detection(reason="ambiguous edge"))
    tr.photo_board(t)
    assert not t.board_detection.accepted and t.fret_lines is None
    np.testing.assert_array_equal(t.board, board)
    for (style, smooth), expected in before.items():
        assert svg(t, style, smooth) == expected
    assert tr.overlay_svg(t) == overlay
    np.testing.assert_array_equal(tr.overlay(t), bitmap)


def test_accepted_pair_clips_all_lines_and_preserves_end_support(monkeypatch):
    from gbcv import fretboard, trace as tr

    t = _board_trace()
    old = t.board.copy()
    old_frets = list(t.frets)
    edges = np.array([[[70, 75], [580, 96]], [[70, 165], [580, 147]]], float)
    monkeypatch.setattr(fretboard, "detect", lambda *a: fretboard.Detection(True, .8, "pair", edges))
    tr.photo_board(t)
    assert t.board_detection.accepted
    assert t.frets == old_frets  # count and fitted sequence are untouched
    assert len(t.fret_lines) == len(t.frets)
    for before, after in ((old[[1, 2]], t.board[[1, 2]]), (old[[0, 3]], t.board[[0, 3]])):
        v = before[1] - before[0]
        for point in after:
            assert abs(np.linalg.det(np.stack((v, point - before[0])))) < 1e-7
    for (xa, xb, h), line in zip(old_frets, t.fret_lines):
        v = np.array([xb-xa, 2*h])
        for point in line:
            assert abs(np.linalg.det(np.stack((v, point - [xa, -h])))) < 1e-7
    # Both sides, polygon and endpoints originate from the SAME accepted pair.
    for points in [t.ruler.to_px(t.board[[0, 3]]), t.ruler.to_px(t.board[[1, 2]])] + [t.ruler.to_px(q) for q in t.fret_lines]:
        for point, edge in zip(points, edges):
            assert abs(np.linalg.det(np.stack((edge[1]-edge[0], point-edge[0])))) < 1e-7
    assert svg(t) != svg(_board_trace())
    assert tr.overlay_svg(t) != tr.overlay_svg(_board_trace())


def test_incompatible_intersections_reject_entire_replacement(monkeypatch):
    from gbcv import fretboard, trace as tr

    t = _board_trace()
    before = svg(t)
    # Detector acceptance is insufficient if intersections fail compatibility.
    edges = np.array([[[70, 0], [570, 0]], [[70, 239], [570, 239]]], float)
    monkeypatch.setattr(fretboard, "detect", lambda *a: fretboard.Detection(True, .9, "pair", edges))
    tr.photo_board(t)
    assert not t.board_detection.accepted
    assert "incompatible" in t.board_detection.reason
    assert t.fret_lines is None and svg(t) == before
