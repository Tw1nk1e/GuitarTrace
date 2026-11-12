"""The cached-photo validator must reject old guesses, not corrected centres."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.fixture
def validator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("validate_headstock").validate_posts


def fixture():
    plate = np.zeros((200, 200), np.uint8)
    plate[40:161, 10:190] = 255
    key = np.zeros_like(plate, bool)
    key[20:30, 75:86] = True
    gray = np.full_like(plate, 40)
    cv2.circle(gray, (80, 80), 6, 230, 1)
    ruler = SimpleNamespace(d=1, to_px=lambda p: np.asarray(p, float))
    return gray, ruler, plate, [key]


def test_corrected_visible_post_passes_without_a_historical_coordinate(validator):
    result = validator(*fixture(), [(80, 80, 5)])
    assert result["post_positions_valid"]
    assert result["post_circle_edge_support"][0] >= .6


def test_old_inward_fallback_without_circular_evidence_fails(validator):
    with pytest.raises(AssertionError, match="no circular photo evidence"):
        validator(*fixture(), [(80, 50, 5)])


@pytest.mark.parametrize("post", [(80, 30, 5), (80, 145, 5), (250, 80, 5)])
def test_outside_or_opposite_side_posts_fail(validator, post):
    with pytest.raises(AssertionError):
        validator(*fixture(), [post])


def test_posts_must_match_distinct_buttons(validator):
    gray, ruler, plate, keys = fixture()
    other = np.roll(keys[0], 40, axis=1)
    with pytest.raises(AssertionError, match="own button"):
        validator(gray, ruler, plate, [*keys, other], [(80, 80, 5), (80, 80, 5)])
