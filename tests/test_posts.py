from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gbcv import trace as tr


def ruler(d=1):
    return SimpleNamespace(d=d, to_mm=lambda p: np.asarray(p, float)/d)


def plate_and_keys(top=(60, 140, 220), bottom=()):
    plate = np.zeros((300, 300), np.uint8)
    plate[80:221, 10:290] = 255
    keys = []
    for xs, y in ((top, 60), (bottom, 240)):
        for x in xs:
            key = np.zeros_like(plate, bool)
            key[y-5:y+6, x-5:x+6] = True
            keys.append(key)
    return plate, keys


def test_existing_supported_row_keeps_exact_detected_and_inferred_positions(monkeypatch):
    plate, keys = plate_and_keys()
    replies = iter([np.array([[[10, 8, 4]]], np.float32), None,
                    np.array([[[10, 8, 4]]], np.float32)])
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: next(replies))
    monkeypatch.setattr(tr, "_inward_bushings", lambda *a: pytest.fail("Reliable row must remain unchanged"))
    posts = tr._posts(np.zeros_like(plate), ruler(), plate, keys)
    np.testing.assert_array_equal(np.asarray(posts)[:, :2], [[56, 93], [136, 93], [216, 93]])


@pytest.mark.parametrize("top,bottom", [((60, 140, 220), ()), ((60, 140), (60, 140, 220))])
def test_unseen_mechanisms_search_both_sides_and_deeper_before_fallback(monkeypatch, top, bottom):
    plate, keys = plate_and_keys(top, bottom)
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: None)
    calls = []

    def detected(blur, plate, kx, edge, side, d, bounds):
        calls.append((kx, side, bounds))
        return [(kx+6, edge-side*30, 5)]

    monkeypatch.setattr(tr, "_inward_bushings", detected)
    posts = tr._posts(np.zeros_like(plate), ruler(), plate, keys)
    assert len(posts) == len(keys) == len(calls)
    expected = [[x+6, 110] for x in top] + [[x+6, 190] for x in bottom]
    assert sorted(map(tuple, np.asarray(posts)[:, :2])) == sorted(map(tuple, expected))
    assert all(left < x < right for x, _, (left, right) in calls)


def test_expanded_candidates_use_other_mechanisms_not_a_fixed_offset(monkeypatch):
    plate, keys = plate_and_keys()
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: None)

    def detected(blur, plate, kx, edge, side, d, bounds):
        # Competing chrome cap with a displaced centre; actual bushing is
        # consistent with the other independently detected mechanisms.
        actual = (kx+6, edge+30, 5)
        return [(kx+15, edge+30, 5), actual] if kx == 60 else [actual]

    monkeypatch.setattr(tr, "_inward_bushings", detected)
    posts = tr._posts(np.zeros_like(plate), ruler(), plate, keys)
    np.testing.assert_array_equal(np.asarray(posts)[:, :2], [[66, 110], [146, 110], [226, 110]])


def test_fallback_uses_inward_offsets_not_opposing_rows_signed_y(monkeypatch):
    plate, keys = plate_and_keys((60,), (140, 220))
    replies = iter([np.array([[[10, 8, 4]]], np.float32),
                    np.array([[[10, 5, 4]]], np.float32), None])
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: next(replies))
    monkeypatch.setattr(tr, "_inward_bushings", lambda *a: [])
    posts = tr._posts(np.zeros_like(plate), ruler(), plate, keys)
    assert posts[-1][1] == 207  # 13 inward, not a median of +13 and -13 = 0


def test_fallback_does_not_average_oppositely_leaning_rows(monkeypatch):
    plate, keys = plate_and_keys((60,), (140, 220))
    replies = iter([np.array([[[10, 8, 4]]], np.float32),
                    np.array([[[21, 5, 4]]], np.float32), None])
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: next(replies))
    monkeypatch.setattr(tr, "_inward_bushings", lambda *a: [])
    posts = tr._posts(np.zeros_like(plate), ruler(), plate, keys)
    assert posts[-1][:2] == (227, 207)


def test_concentric_chrome_cap_does_not_bias_its_own_reference(monkeypatch):
    plate, keys = plate_and_keys((60, 140, 220), (60, 140, 220))
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: None)

    def detected(blur, plate, kx, edge, side, d, bounds):
        washer = (kx+6, edge-side*30, 6)
        cap = (kx+10, edge-side*30, 3)
        return [cap, washer] if side < 0 and kx == 60 else [washer, cap]

    monkeypatch.setattr(tr, "_inward_bushings", detected)
    posts = tr._posts(np.zeros_like(plate), ruler(), plate, keys)
    expected = [(x+6, y) for x in (60, 140, 220) for y in (110, 190)]
    assert sorted(map(tuple, np.asarray(posts)[:, :2])) == sorted(expected)


@pytest.mark.parametrize("side", [-1, 1])
def test_expanded_search_accepts_only_the_corresponding_inward_plate_circle(monkeypatch, side):
    plate, _ = plate_and_keys()
    kx, edge = 140, 80 if side < 0 else 220

    def circles(region, *a, **kwargs):
        # Recover the helper's clamped ROI; proposals expressed locally.
        x0 = int(kx-27)
        y0 = int(min(edge-side*5, edge-side*40)-7)
        proposed = [(kx+6, edge-side*30, 5),  # real inward circle
                    (kx+18, edge-side*30, 5),  # neighbouring mechanism's cell
                    (kx+6, edge+side*5, 5),  # outside / wrong side
                    (kx+6, edge-side*90, 5)]  # opposite half of plate
        return np.array([[(x-x0, y-y0, r) for x, y, r in proposed]], np.float32)

    monkeypatch.setattr(cv2, "HoughCircles", circles)
    found = tr._inward_bushings(np.zeros_like(plate), plate, kx, edge, side, 1, (125, 155))
    assert found == [(146, edge-side*30, 5)]


def test_expanded_roi_contains_the_full_bushing_not_only_its_centre():
    plate, _ = plate_and_keys()
    gray = np.full_like(plate, 30)
    # Centre far deeper and on the other side of the shaft than the legacy
    # window. Real OpenCV, no mocked detector, doubled photo resolution.
    plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.circle(gray, (292, 220), 10, 235, 2)
    found = tr._inward_bushings(cv2.GaussianBlur(gray, (0, 0), 1), plate, 280, 160, -1, 2)
    assert any(np.linalg.norm(np.array([x, y])-[292, 220]) < 3 for x, y, _ in found)
