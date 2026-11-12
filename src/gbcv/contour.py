"""Outline smoothing, resampling and Douglas-Peucker node selection."""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth(path: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of an open path; endpoints stay put."""
    if sigma <= 0 or len(path) < 3:
        return path.astype(float)
    s = gaussian_filter1d(path.astype(float), sigma, axis=0, mode="nearest")
    s[0], s[-1] = path[0], path[-1]
    return s


def resample(path: np.ndarray, step: float) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    s = np.r_[s, d[-1]] if d[-1] - s[-1] > step * 0.3 else np.r_[s[:-1], d[-1]]
    return np.c_[np.interp(s, d, path[:, 0]), np.interp(s, d, path[:, 1])]


def find_corners(path: np.ndarray, reach: int, min_turn_deg: float = 55, limit: int | None = None) -> list[int]:
    """Indices where the direction turns sharply over ±reach samples (ear of a headstock, pointy horn).
    At most `limit` of the sharpest ones (a jagged raw outline can turn sharply everywhere)."""
    n = len(path)
    turn = np.zeros(n)
    for i in range(reach, n - reach):
        a, b = path[i] - path[i - reach], path[i + reach] - path[i]
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        turn[i] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    out = []
    for i in np.argsort(-turn):
        if turn[i] < min_turn_deg:
            break
        if all(abs(i - j) > 2 * reach for j in out):
            out.append(int(i))
            if len(out) == limit:
                break
    return sorted(out)


def _dp(path: np.ndarray, tol: float) -> list[int]:
    """Douglas-Peucker indices (iterative)."""
    keep, stack = {0, len(path) - 1}, [(0, len(path) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = path[a], path[b]
        d = q - p
        L = np.hypot(*d) or 1.0
        seg = path[a + 1 : b]
        dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = a + 1 + k
            keep.add(m)
            stack += [(a, m), (m, b)]
    return sorted(keep)


def dp_closed(path: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed path: cut it at the point farthest from the first, or the two ends collapse
    and the whole outline becomes one chord. Returns the kept points, in order, without repeating the first."""
    p = np.asarray(path, float)
    if len(p) < 5:
        return p
    i = int(np.argmax(np.linalg.norm(p - p[0], axis=1)))
    a = _dp(p[: i + 1], tol)
    b = _dp(np.vstack([p, p[:1]])[i:], tol)
    idx = sorted(set(a) | {i + j for j in b})
    return p[[k for k in idx if k < len(p)]]
