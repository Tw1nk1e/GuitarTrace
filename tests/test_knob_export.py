"""Confirmed controls only: SVG representation, without any ML or photo fixtures."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gbcv import knobs, trace as tr


def ring(radius=12, centre=(40, 30), noisy=False):
    theta = np.linspace(0, 2*np.pi, 180, endpoint=False)
    r = radius * (1 + .07*np.sin(19*theta)) if noisy else np.full_like(theta, radius)
    if noisy:
        r += radius*.3*np.exp(-((theta-1.5)/.16)**2)
    return np.array(centre) + np.c_[r*np.cos(theta), r*np.sin(theta)]


def drawing(parts, controls=()):
    return tr.Trace(outline=np.array([[0,0],[100,0],[100,70],[0,70]],float),
                    nut_x=90, scale_length=100, frets=[],
                    board=np.array([[90,25],[65,20],[65,40],[90,35]],float),
                    parts=parts, knobs=list(controls))


def circles(t):
    return [e.attrib for e in ET.fromstring(tr.svg(t, 'png')) if e.tag.endswith('circle')]


@pytest.mark.parametrize('noisy', [False, True])
def test_confirmed_round_and_noisy_blob_export_as_circle(noisy):
    c = ring(noisy=noisy)
    old = c.copy()
    result = circles(drawing([c], [knobs.Control(c)]))
    assert len(result) == 1
    assert float(result[0]['cx']) == pytest.approx(40, abs=.4)
    assert float(result[0]['cy']) == pytest.approx(30, abs=.4)
    assert float(result[0]['r']) == pytest.approx(12, abs=.4)
    np.testing.assert_array_equal(old, c)


def test_inner_rim_becomes_exactly_concentric():
    c, inner = ring(noisy=True), ring(7, centre=(40.4,30.2), noisy=True)
    t = drawing([c, inner], [knobs.Control(c, inner)])
    result = circles(t)
    assert len(result) == 2
    assert result[0]['cx'] == result[1]['cx']
    assert result[0]['cy'] == result[1]['cy']
    assert float(result[1]['r']) == pytest.approx(7, abs=.3)
    assert tr.svg(t).count('<path') == tr.svg(drawing([])).count('<path')


def test_bad_inner_rim_is_not_invented():
    c = ring()
    assert len(circles(drawing([c], [knobs.Control(c, ring(7, centre=(46,30)))]))) == 1


def test_non_knob_round_hardware_and_stale_labels_are_unchanged():
    c = ring(noisy=True)
    baseline = tr.svg(drawing([c]))
    assert not circles(drawing([c]))
    assert baseline == tr.svg(drawing([c], [knobs.Control(c.copy())]))
    # Other parts, head keys, posts, silhouette and frets retain identical XML.
    other = ring(4, (70,50))
    t = drawing([c,other])
    t.keys = [ring(3, (94,30))]
    t.posts = [(93,30,2)]
    before = list(ET.fromstring(tr.svg(t)))
    t.knobs = [knobs.Control(c)]
    after = list(ET.fromstring(tr.svg(t)))
    assert all(a.tag == b.tag and a.attrib == b.attrib for a,b in zip(before[:-2], after[:-2]))
    assert before[-1].attrib == after[-1].attrib


def test_unusable_confirmed_geometry_falls_back_to_original_path():
    c = np.array([[0,0],[2,0],[2,20],[0,20]],float)
    assert tr.svg(drawing([c])) == tr.svg(drawing([c], [knobs.Control(c)]))


def test_nearby_pointer_is_not_suppressed_by_proximity_or_partial_overlap():
    c = ring()
    pointer = ring(2, (50, 22))  # touches/overlaps the rim, but is not a labelled inner ring
    t = drawing([c, pointer])
    original = list(ET.fromstring(tr.svg(t)))[-1]
    t.knobs = [knobs.Control(c)]
    exported = list(ET.fromstring(tr.svg(t)))[-1]
    assert original.tag == exported.tag and original.attrib == exported.attrib
