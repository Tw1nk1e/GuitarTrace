import numpy as np
import pytest

from gbcv import contour


def _blob(n=2000):
    """Open arc shaped like half a body outline: bumps of different curvature."""
    t = np.linspace(0.2, 2 * np.pi - 0.2, n)
    r = 150 + 25 * np.sin(3 * t) + 10 * np.cos(7 * t)
    return np.c_[r * np.cos(t), r * np.sin(t)]


def test_resample_preserves_endpoints_and_input():
    path = np.array([[0, 0], [6, 0], [6, 4]], float)
    original = path.copy()
    points = contour.resample(path, 2)
    np.testing.assert_array_equal(points, [[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [6, 4]])
    np.testing.assert_array_equal(path, original)


def test_corner_is_kept_sharp():
    # an "ear": straight up, sharp turn, straight right
    path = contour.resample(np.array([[0, 0], [0, -60], [60, -60]], float), 1.0)
    cs = contour.find_corners(path, reach=4, min_turn_deg=70)
    assert len(cs) == 1 and np.allclose(path[cs[0]], [0, -60], atol=1.5)


def test_closed_simplification_preserves_square_corners_without_duplicate_endpoint():
    path = np.array([[0, 0], [5, 0], [10, 0], [10, 5], [10, 10], [5, 10], [0, 10], [0, 5]], float)
    original = path.copy()
    np.testing.assert_array_equal(contour.dp_closed(path, .1), [[0, 0], [10, 0], [10, 10], [0, 10]])
    np.testing.assert_array_equal(path, original)


def test_obsolete_fitter_is_not_part_of_distributed_module():
    for name in ('spline_samples', 'fit', '_fit'):
        assert not hasattr(contour, name)


@pytest.mark.parametrize('style,expected', [
    ('thin', '5000749e2abeeb30c3997819748eefe3448c545351e085dd8e46fa209199f5a7'),
    ('comic', 'fcd5799d12da0a7b9eabb1205f953be5a1457922950a9132130f11f56e680423'),
    ('png', '505350c45087e3d7afb7535926b84280584ca48b8a8ce717c043e0b9bfe26d3c'),
])
def test_svg_export_matches_pre_cleanup_geometry(style, expected):
    # Baselines captured from the current Python exporter before deleting the
    # unused fitter, not from the historical reference generator. No ML/photos.
    import hashlib
    from editor_sample import sample
    from gbcv import trace

    assert hashlib.sha256(trace.svg(sample(), style).encode()).hexdigest() == expected


@pytest.mark.parametrize("sigma", [0, 2])
def test_smooth_keeps_endpoints(sigma):
    p = _blob(300)
    s = contour.smooth(p, sigma)
    assert np.allclose(s[[0, -1]], p[[0, -1]])
