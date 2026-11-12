"""Export-only regression: nonuniform Catmull-Rom must not lend long handles to short edges."""
import re
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gbcv import trace as tr
from editor_sample import sample


def cubics(path):
    values = np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', path)])
    return values[:2], values[2:].reshape(-1, 3, 2)


def test_short_edge_handles_bounded_without_moving_nodes_or_losing_tangency():
    pts = np.array([[0, 0], [100, 0], [101, 1], [101, 20], [0, 20]], float)
    original = pts.copy()
    old = tr.curve_path(pts, 0)
    new = tr.curve_path(pts, 0, bounded=True)
    start, a = cubics(old)
    _, b = cubics(new)
    np.testing.assert_array_equal(a[:, 2], b[:, 2])
    np.testing.assert_array_equal(pts, original)
    nodes = np.vstack([start, b[:-1, 2]])
    length = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
    incoming = nodes - np.roll(b[:, 1], 1, axis=0)
    outgoing = b[:, 0] - nodes
    np.testing.assert_allclose(incoming, outgoing, atol=.011)
    assert np.all(np.linalg.norm(outgoing, axis=1) <= np.minimum(length, np.roll(length, 1)) / 3 + .015)
    # The short span no longer curls back towards the long edge's midpoint.
    assert np.linalg.norm(a[1, 0] - nodes[1]) > 15
    assert np.linalg.norm(b[1, 0] - nodes[1]) < .5
    assert new.count('C') == old.count('C') == len(pts)  # still cubic, not a polyline


def test_regular_run_is_byte_identical_and_no_global_smoothing():
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    pts = 20 * np.c_[np.cos(angles), np.sin(angles)]
    assert tr.curve_path(pts, 0) == tr.curve_path(pts, 0, bounded=True)
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert tr.curve_path(pts, 0) == tr.curve_path(pts, 0, bounded=True)


def test_only_hardware_export_changes_all_other_elements_and_trace_are_identical(monkeypatch):
    t = sample()
    t.parts = [np.array([[0, -50], [100, -50], [101, -49], [101, -30], [0, -30]], float)]
    points = t.parts[0].copy()
    new = ET.fromstring(tr.svg(t, 'png', 0, editor=True))
    actual = tr.bounded_handles
    monkeypatch.setattr(tr, 'bounded_handles', lambda nodes, handles: handles)
    old = ET.fromstring(tr.svg(t, 'png', 0, editor=True))
    changed = []
    for a, b in zip(old, new, strict=True):
        if ET.tostring(a) != ET.tostring(b):
            changed.append(a.attrib['data-category'])
    assert changed == ['hardware']
    np.testing.assert_array_equal(points, t.parts[0])
    # A degenerate repeated node has a zero handle rather than NaN/overshoot.
    nodes = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], float)
    tangents = (np.roll(nodes, -1, axis=0) - np.roll(nodes, 1, axis=0)) / 6
    assert np.isfinite(actual(nodes, tangents)).all()


@pytest.mark.parametrize('name,expected_before,expected_after', [('lp_ref_1', 12, 6), ('ref_sg_1', 8, 6)])
def test_saved_real_contours_reduce_excursions_without_new_failed_contours(name, expected_before, expected_after, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = root / 'out/e2e-lp-sg' / name / 'sam'
    if not (source / 'stages.npz').exists():
        pytest.skip('Local ignored diagnostic polygons not available')
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    from audit_svg_quality import distances
    from validate_headstock import sample_path
    records = json.loads((source / 'candidates.json').read_text(encoding='utf-8'))
    before_errors, after_errors = [], []
    with np.load(source / 'stages.npz', allow_pickle=False) as z:
        for r in records:
            if r['status'] != 'kept':
                continue
            p = z[f'c{r["id"]}_export_smoothed_contour']
            old = tr.curve_path(p, .5)
            new = tr.curve_path(p, .5, bounded=True)
            assert old == r['svg_path']
            np.testing.assert_array_equal(cubics(old)[1][:, 2], cubics(new)[1][:, 2])
            polygon = np.vstack([p, p[:1]])
            before_errors.append(float(distances(sample_path(old, step=.15), polygon).max()))
            after_errors.append(float(distances(sample_path(new, step=.15), polygon).max()))
    a, b = np.array(before_errors), np.array(after_errors)
    assert np.count_nonzero(a > .5) == expected_before
    assert np.count_nonzero(b > .5) == expected_after
    assert np.max(b) < 1 < np.max(a)
    assert not np.any((a <= .5) & (b > .5))
