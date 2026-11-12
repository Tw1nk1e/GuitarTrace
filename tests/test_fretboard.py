"""Known synthetic edges are labels here; real-photo predictions are never labels."""

import cv2
import numpy as np
import pytest

from gbcv import fretboard


def scene(strings=6, fan=0.0, scale=1):
    old = np.array([[570, 90], [70, 70], [70, 170], [570, 150]], float)
    real = old + [[0, 4], [0, 6], [0, -5], [0, -3]]
    img = np.full((240, 640, 3), 230, np.uint8)
    cv2.fillPoly(img, [real.astype(np.int32)], (125, 75, 35))
    # Strings and frets intentionally have higher local contrast than the wood.
    for u in np.linspace(.12, .88, strings):
        a = real[1] * (1-u) + real[2] * u
        b = real[0] * (1-u) + real[3] * u
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 255, 255), 1)
    for x in np.linspace(90, 550, 22):
        cv2.line(img, (int(x-fan*35), 77), (int(x+fan*35), 163), (215, 215, 215), 1)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img, old * scale, real * scale


@pytest.mark.parametrize("strings,fan,scale", [(4, 0, 1), (6, 0, 1), (8, .6, 1), (10, -.6, 2), (12, .3, 1)])
def test_pair_with_strings_taper_and_fanned_frets(strings, fan, scale):
    img, prior, real = scene(strings, fan, scale)
    d = fretboard.detect(img, prior)
    assert d.accepted, (d.reason, d.diagnostics)
    assert 0 <= d.confidence <= 1
    for line, truth in zip(d.edges, (real[[1, 0]], real[[2, 3]])):
        expected = np.interp(line[:, 0], truth[:, 0], truth[:, 1])
        assert np.max(np.abs(line[:, 1] - expected)) < 3 * scale


def test_flat_image_falls_back():
    img, prior, _ = scene()
    img[:] = 120
    d = fretboard.detect(img, prior)
    assert not d.accepted
    assert "weak" in d.reason


def test_one_clear_edge_is_not_enough():
    img, prior, real = scene()
    # Keep ONLY the upper half-plane transition; lower side has no edge at all.
    img[:] = 230
    pts = np.array([real[1], real[0], [640, 240], [0, 240]], np.int32)
    cv2.fillPoly(img, [pts], (125, 75, 35))
    assert not fretboard.detect(img, prior).accepted


def test_two_similar_outer_boundaries_are_ambiguous():
    img, prior, _ = scene()
    img[:] = 240
    cv2.fillPoly(img, [prior.astype(np.int32)], (140, 110, 80))
    inner = prior + [[0, 7], [0, 11], [0, -11], [0, -7]]
    cv2.fillPoly(img, [inner.astype(np.int32)], (35, 30, 25))
    d = fretboard.detect(img, prior)
    assert not d.accepted, d.diagnostics
    assert "ambiguous" in d.reason


def test_internal_strings_alone_cannot_be_a_board_pair():
    img, prior, _ = scene()
    img[:] = 110
    for y in (88, 94, 100, 106, 112, 118, 124, 130, 136, 142, 148, 154):
        cv2.line(img, (70, y), (570, y), (255, 255, 255), 1)
    assert not fretboard.detect(img, prior).accepted


def test_broken_side_falls_back():
    img, prior, _ = scene()
    img[60:120, 200:440] = (230, 230, 230)
    assert not fretboard.detect(img, prior).accepted


def test_incompatible_taper_or_centre_falls_back():
    img, prior, _ = scene()
    img[:] = 230
    incompatible = prior + [[0, -18], [0, 25], [0, -25], [0, 18]]
    cv2.fillPoly(img, [incompatible.astype(np.int32)], (80, 50, 30))
    assert not fretboard.detect(img, prior).accepted


@pytest.mark.parametrize("board", [np.zeros((4, 2)), np.full((4, 2), np.nan), np.zeros((3, 2))])
def test_invalid_prior_is_safe(board):
    assert not fretboard.detect(np.zeros((100, 100, 3), np.uint8), board).accepted


def test_intersect_preserves_fan_supporting_line():
    edges = np.array([[[0, 20], [200, 25]], [[0, 100], [200, 85]]], float)
    fret = np.array([[70, 35], [100, 80]], float)
    result = fretboard.intersect(fret, edges)
    assert result[0, 1] < 35 and result[1, 1] > 80
    assert np.linalg.det(np.stack((fret[1]-fret[0], result[0]-fret[0]))) == pytest.approx(0, abs=1e-8)


def test_parallel_intersection_is_rejected():
    edges = np.array([[[0, 0], [10, 0]], [[0, 20], [10, 20]]], float)
    with pytest.raises(ValueError):
        fretboard.intersect(np.array([[0, 5], [10, 5]]), edges)
