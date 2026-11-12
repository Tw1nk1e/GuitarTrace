from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gbcv import trace as tr


def button(axes=(14, 12), angle=0, bottom=False, scale=1):
    """A photo-sized plate, a lateral button and its narrow shaft."""
    plate = np.zeros((200, 280), np.uint8)
    plate[90:165, 20:260] = 255
    key = np.zeros_like(plate)
    cv2.ellipse(key, (130, 65), axes, angle, 0, 360, 1, -1)
    cv2.rectangle(key, (128, 70), (132, 89), 1, -1)
    if bottom:
        plate, key = plate[::-1].copy(), key[::-1].copy()
    if scale != 1:
        plate = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        key = cv2.resize(key, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return key > 0, plate, SimpleNamespace(d=scale)


@pytest.mark.parametrize("bottom", [False, True])
@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("axes,angle", [((14, 12), 0), ((9, 22), 0), ((15, 10), 25)])
def test_compact_and_edge_on_buttons_without_visible_bushing(axes, angle, bottom, scale):
    key, plate, ruler = button(axes, angle, bottom, scale)
    assert tr._key_is_button(key, plate, ruler)


def ribbon():
    plate = np.zeros((180, 260), np.uint8)
    plate[80:150, 20:240] = 255
    key = np.zeros_like(plate)
    cv2.fillPoly(key, [np.array([[135, 79], [215, 79], [235, 64], [205, 67]])], 1)
    return key > 0, plate, SimpleNamespace(d=1)


@pytest.mark.parametrize("bottom", [False, True])
def test_long_shallow_plate_edge_is_not_a_button(bottom):
    key, plate, ruler = ribbon()
    if bottom:
        key, plate = key[::-1].copy(), plate[::-1].copy()
    assert not tr._key_is_button(key, plate, ruler)


def test_detached_compact_highlight_is_not_a_button():
    key, plate, ruler = button()
    plate[:] = 0
    assert not tr._key_is_button(key, plate, ruler)


def ambiguous():
    # A genuinely elongated paddle: not accepted solely because it is short,
    # not rejected solely because it is long. Geometry needs inward evidence.
    key, plate, ruler = button((5, 25))
    return key, plate, ruler


def test_ambiguous_shape_without_photo_keeps_plate():
    key, plate, ruler = ambiguous()
    assert not tr._key_is_button(key, plate, ruler)


def test_ambiguous_shape_without_detected_bushing_keeps_plate(monkeypatch):
    key, plate, ruler = ambiguous()
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: None)
    assert not tr._key_is_button(key, plate, ruler, np.zeros_like(plate))


def test_ambiguous_shape_can_be_confirmed_by_an_inward_mechanism(monkeypatch):
    key, plate, ruler = ambiguous()
    # Search ROI: x=116..136, y=95..108; this circle is inside the plate.
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: np.array([[[12, 7, 4]]], np.float32))
    assert tr._key_is_button(key, plate, ruler, np.zeros_like(plate))


def test_circle_outside_plate_does_not_confirm_ambiguous_shape(monkeypatch):
    key, plate, ruler = ambiguous()
    plate[96:106, 120:136] = 0
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: np.array([[[12, 7, 4]]], np.float32))
    assert not tr._key_is_button(key, plate, ruler, np.zeros_like(plate))


def synthetic_head(count):
    mask = np.zeros((240, 150 + count * 50), np.uint8)
    cv2.rectangle(mask, (10, 100), (mask.shape[1] - 25, 175), 255, -1)
    for i in range(count):
        x = 90 + 50 * i
        cv2.rectangle(mask, (x - 3, 80), (x + 3, 100), 255, -1)
        cv2.ellipse(mask, (x, 72), (12, 10), 0, 0, 360, 255, -1)
    return mask


@pytest.mark.parametrize("count", [2, 5, 7, 10])
def test_head_parts_does_not_assume_six_or_eight_buttons(count):
    mask = synthetic_head(count)
    solid, plate, keys = tr.head_parts(mask, SimpleNamespace(d=1), 20)
    assert len(keys) == count
    assert np.array_equal(solid[:, :40], mask[:, :40])
    for key in keys:
        assert not (plate[key] > 0).any()


def test_rejected_candidate_is_restored_in_plate_and_original_silhouette(monkeypatch):
    mask = synthetic_head(3)
    ruler = SimpleNamespace(d=1)
    old_solid, old_plate, old_keys = tr.head_parts(mask, ruler, 20)
    assert len(old_keys) == 3
    reject = old_keys[1]
    classifier = tr._key_is_button
    monkeypatch.setattr(tr, "_key_is_button", lambda comp, *a: False if np.array_equal(comp, reject) else classifier(comp, *a))
    solid, plate, keys = tr.head_parts(mask, ruler, 20)
    assert len(keys) == 2
    assert (solid[reject] == mask[reject]).all()
    assert (plate[reject] > 0).all()
    for key in keys:
        assert any(np.array_equal(key, before) for before in old_keys)


def test_no_confident_buttons_preserves_original_mask(monkeypatch):
    mask = synthetic_head(3)
    monkeypatch.setattr(tr, "_key_is_button", lambda *a: False)
    solid, plate, keys = tr.head_parts(mask, SimpleNamespace(d=1), 20)
    np.testing.assert_array_equal(solid, mask)
    np.testing.assert_array_equal(plate, mask)
    assert keys == []
