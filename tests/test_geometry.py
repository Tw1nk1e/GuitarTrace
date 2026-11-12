"""Orientation and splitting on a synthetic guitar mask (no model needed)."""

import cv2
import numpy as np
import pytest

from gbcv.orient import orient_and_split

# synthetic guitar, px: body ellipse 0..340, neck 300..860 (width 50 → 42), head 860..1040 (width 80)
NUT = 860


def synthetic(angle_deg: float) -> np.ndarray:
    m = np.zeros((1400, 1400), np.uint8)
    cx, cy = 100, 700
    cv2.ellipse(m, (cx + 170, cy), (170, 160), 0, 0, 360, 255, -1)
    neck = np.array([[300, cy - 25], [NUT, cy - 21], [NUT, cy + 21], [300, cy + 25]], np.int32)
    cv2.fillPoly(m, [neck], 255)
    head = np.array([[NUT, cy - 21], [NUT + 20, cy - 40], [NUT + 180, cy - 45], [NUT + 180, cy + 10], [NUT + 20, cy + 40], [NUT, cy + 21]], np.int32)
    cv2.fillPoly(m, [head], 255)
    R = cv2.getRotationMatrix2D((620, 700), angle_deg, 1.0)
    return cv2.warpAffine(m, R, (1400, 1400), flags=cv2.INTER_NEAREST)


@pytest.mark.parametrize("angle", [0, 37, 90, 180, -120])
def test_orients_head_right_and_finds_nut(angle):
    fr, m, sp = orient_and_split(synthetic(angle))
    length = sp.tip_x - sp.butt_x
    assert length == pytest.approx(NUT + 180 - 100, rel=0.02)
    assert (sp.nut_x - sp.butt_x) == pytest.approx(NUT - 100, abs=0.03 * length)
    assert sp.neck_width(sp.nut_x) == pytest.approx(42, abs=4)
    assert abs(sp.centre_fit[0]) < 0.01  # neck is horizontal
    # arcs start on the bass side (above the centre line) and end on the treble side
    for arc in (sp.body, sp.head):
        c = sp.neck_centre(arc[[0, -1], 0])
        assert arc[0, 1] < c[0] and arc[-1, 1] > c[1]
    assert sp.body[:, 0].max() < sp.nut_x < sp.head[:, 0].min() + 2
