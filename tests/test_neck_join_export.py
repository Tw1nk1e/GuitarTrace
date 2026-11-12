"""Local export-only correction; detection geometry and unrelated paths are immutable."""
import re
import xml.etree.ElementTree as ET

import numpy as np

from gbcv import trace as tr
from editor_sample import sample


def cubics(path):
    values = np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', path)])
    return values[:2], values[2:].reshape(-1, 3, 2)


def test_only_neck_join_handles_change_not_vertices_or_body():
    points = np.array([[0,-80],[100,-80],[110,-22],[120,-20],[600,-20],
                       [610,-35],[650,-35],[650,35],[610,35],[600,20],
                       [120,20],[110,22],[100,80],[0,80]], float)
    board = np.array([[600,-19],[100,-23],[100,23],[600,19]], float)
    old = tr.curve_path(points, .1)
    new = tr.curve_path(points, .1, neck_join=board)
    _, a = cubics(old)
    _, b = cubics(new)
    np.testing.assert_array_equal(a[:,2], b[:,2])
    changed = np.where(np.any(a != b, axis=(1,2)))[0]
    assert len(changed) == 4  # two adjacent cubics at each of the two joins
    for j in changed:
        start = a[(j-1) % len(a),2]
        assert start[0] >= 110  # body curves themselves remain byte-identical
    # Both handles at a node retain one common tangent (C1 continuity).
    nodes = a[:,2]
    for j in range(len(a)):
        incoming = nodes[j] - b[j,1]
        outgoing = b[(j+1) % len(b),0] - nodes[j]
        np.testing.assert_allclose(incoming, outgoing, atol=.011)
    np.testing.assert_array_equal(points, np.array([[0,-80],[100,-80],[110,-22],[120,-20],[600,-20],
                       [610,-35],[650,-35],[650,35],[610,35],[600,20],
                       [120,20],[110,22],[100,80],[0,80]], float))


def test_no_unrelated_export_changes_or_mutation():
    t = sample()
    before = t.outline.copy(), t.board.copy(), list(t.frets)
    for style in tr.STYLES:
        root = ET.fromstring(tr.svg(t, style))
        expected = tr.curve_path(tr.smoothed(t.outline, tr.DEFAULT_SMOOTH_MM), tr.CURVE_TOL_MM,
                                 neck_join=t.board)
        assert root[0].attrib['d'] == expected
    np.testing.assert_array_equal(t.outline, before[0])
    np.testing.assert_array_equal(t.board, before[1])
    assert t.frets == before[2]
