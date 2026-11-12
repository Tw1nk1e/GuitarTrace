"""Fret ruler on a synthetic oriented guitar: known scale, nut and camera tilt."""

import cv2
import numpy as np
import pytest

from gbcv import frets
from gbcv.split import split

S = 648.0


def synthetic(a: float, c: float, bright_nut: bool):
    """Canvas with head to the right; frets drawn at x = (b - a*U)/(1 + c*U). a: px/mm at the nut."""
    H, W, cy, b = 700, 1800, 350, 1450.0
    x_of = lambda u: (b - a * u) / (1 + c * u)
    m = np.zeros((H, W), np.uint8)
    xb = int(x_of(S * 0.92))  # body starts a bit behind the 22nd fret
    cv2.ellipse(m, (xb - 230, cy), (260, 220), 0, 0, 360, 255, -1)
    neck_w = lambda x: 0.5 * (42 + 14 * (b - x) / (b - xb)) * a  # half width in px, 42 → 56 mm
    xs = np.arange(xb - 60, int(b) + 1)
    pts = np.r_[np.c_[xs, cy - neck_w(xs)], np.c_[xs[::-1], cy + neck_w(xs[::-1])]].astype(np.int32)
    cv2.fillPoly(m, [pts], 255)
    head = np.array([[b + 6, cy - 21 * a], [b + 40 * a, cy - 45 * a], [b + 180 * a, cy - 45 * a], [b + 180 * a, cy + 20 * a], [b + 40 * a, cy + 45 * a], [b + 6, cy + 21 * a]], np.int32)
    cv2.fillPoly(m, [head], 255)
    cv2.rectangle(m, (int(b) - 2, int(cy - 21 * a)), (int(b) + 8, int(cy + 21 * a)), 255, -1)
    gray = np.where(m > 0, 60, 20).astype(np.float32)
    gray[(m > 0) & (np.arange(W)[None] > b + 6)] = 150  # light headstock face
    for n in range(1, 22):
        x = x_of(S * (1 - 2 ** (-n / 12)))
        cv2.line(gray, (int(round(x)), int(cy - neck_w(x) + 2)), (int(round(x)), int(cy + neck_w(x) - 2)), 210, 2)
    if bright_nut:  # a bone nut looks like a fret: the self-similarity trap
        cv2.rectangle(gray, (int(b), int(cy - 20 * a)), (int(b) + 5, int(cy + 20 * a)), 230, -1)
    return gray, m


@pytest.mark.parametrize("c", [0.0, 2e-4])
@pytest.mark.parametrize("bright_nut", [False, True])
def test_recovers_nut_scale_and_tilt(c, bright_nut):
    a = 1.3
    gray, m = synthetic(a, c, bright_nut)
    sp = split(m)
    ff = frets.fit(gray, m, sp)
    assert ff is not None and len(ff.found) >= 12
    assert min(n for n, _ in ff.found) <= 2  # numbering starts at the nut, not one fret off
    mm = frets.to_mm(ff, S)
    assert mm.b == pytest.approx(1450, abs=2)
    assert mm.a == pytest.approx(a, rel=0.01)
    assert mm.c == pytest.approx(c, abs=3e-5)
    assert ff.residual < 1.0


@pytest.mark.parametrize("k", [0.0, 0.4, -0.7])
def test_best_line_recovers_a_slanted_fret(k):
    """A fanned board needs no separate branch: a fret is a line with its own slope, and 0 is the square case."""
    s = np.zeros((41, 200), np.float32)
    rows = np.arange(41)
    xs = np.round(100 + k * (rows - 20)).astype(int)
    s[rows, xs] = 1.0
    s[rows, xs + 1] = 1.0
    got_k, off, contrast = frets._best_line(s, 100, 8)
    assert got_k == pytest.approx(k, abs=0.05)
    assert off == pytest.approx(100, abs=1.5)
    assert contrast > 1.5


def recovery_case(candidate=77, mirror=False, flip_rows=False):
    score = np.zeros((41, 400), np.float32)
    rows = np.arange(41)
    for p in [100, 125, 152] + ([] if candidate is None else [candidate]):
        cols = np.round(p + 0.2 * (rows - 20)).astype(int)
        score[rows, cols] = 1
        score[rows, cols + 1] = 1
    transform = lambda p: 399 - p if mirror else p
    if mirror:
        score = score[:, ::-1].copy()
    if flip_rows:
        score = score[::-1].copy()
    slope = 0.2 * (-1 if mirror else 1) * (-1 if flip_rows else 1)
    got = [{"mid": 50 + transform(p), "off": transform(p), "k": slope, "contrast": 5.0}
           for p in (100, 125, 152)]
    got.sort(key=lambda q: q["mid"])
    return score, got, transform


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("flip_rows", [False, True])
def test_recovers_rejected_body_candidate_from_photo(mirror, flip_rows):
    score, got, transform = recovery_case(mirror=mirror, flip_rows=flip_rows)
    before = [q.copy() for q in got]
    # The coarse gaps 21, 26 fail; the actual photo gaps 23, 25, 27 agree.
    assert abs(np.log(26 / (21 * frets.RATIO))) > 0.09
    frets._recover_body_fret(score, [transform(79)], got, transform(300), 50)
    assert len(got) == 4
    assert all(q in got for q in before)  # accepted positions AND slopes are untouched
    added = next(q for q in got if q not in before)
    assert added["off"] == pytest.approx(transform(77), abs=1.5)
    assert added["contrast"] >= 1.5


@pytest.mark.parametrize("candidate, peak", [(None, 79), (83, 83), (99, 99), (52, 52)])
@pytest.mark.parametrize("mirror", [False, True])
def test_recovery_rejects_unconfirmed_false_or_duplicate_candidate(candidate, peak, mirror):
    score, got, transform = recovery_case(candidate, mirror)
    before = [q.copy() for q in got]
    frets._recover_body_fret(score, [transform(peak)], got, transform(300), 50)
    assert got == before


def test_recovery_leaves_normal_chain_unchanged():
    score, got, transform = recovery_case(candidate=None)
    before = [q.copy() for q in got]
    frets._recover_body_fret(score, [], got, transform(300), 50)
    assert got == before


@pytest.mark.parametrize("proposal", [(0.0, 77.0, 1.0), (0.0, 100.0, 5.0), (0.0, 105.0, 5.0)])
def test_recovery_checks_refined_contrast_duplicates_and_order(monkeypatch, proposal):
    score, got, transform = recovery_case()
    before = [q.copy() for q in got]
    lines = iter([proposal, (0.0, 100.0, 5.0), (0.0, 125.0, 5.0), (0.0, 152.0, 5.0)])
    monkeypatch.setattr(frets, "_best_line", lambda *args: next(lines))
    frets._recover_body_fret(score, [79], got, transform(300), 50)
    assert got == before


def test_recovery_rejects_lines_crossing_away_from_midpoints(monkeypatch):
    score = np.zeros((401, 400), np.float32)
    got = [{"mid": p, "off": p, "k": 0.0, "contrast": 5.0} for p in (100, 125, 152)]
    before = [q.copy() for q in got]
    # Compatible midpoint gaps and <10 degree slope difference, but a crossing at the edge.
    lines = iter([(0.15, 77.0, 5.0), (0.0, 100.0, 5.0), (0.0, 125.0, 5.0), (0.0, 152.0, 5.0)])
    monkeypatch.setattr(frets, "_best_line", lambda *args: next(lines))
    frets._recover_body_fret(score, [79], got, 300, 0)
    assert got == before
