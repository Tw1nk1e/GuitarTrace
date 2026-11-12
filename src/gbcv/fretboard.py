"""Conservative paired photo edges. Confidence is a heuristic, NOT a probability.

Only straight, continuous sides are supported. Ambiguous binding, curved edges,
occlusion or an incompatible prior cause a complete fallback, never a one-side fit.
Coordinates throughout this module are oriented photo pixels.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    accepted: bool = False
    confidence: float = 0.0
    reason: str = "no evidence"
    edges: np.ndarray | None = None  # two lines, each two (x, y) points; top, bottom
    diagnostics: dict = field(default_factory=dict)


def _at(image, x, y):
    return cv2.remap(image, np.asarray(x, np.float32), np.asarray(y, np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def detect(photo: np.ndarray, board: np.ndarray) -> Detection:
    """Find a jointly plausible pair near the prior quadrilateral, or reject both.

    The prior constrains width/taper, not exact edge locations. Thin bright/dark
    stripes (strings, binding highlights) are suppressed by comparing colour on
    either side of a ridge over a wider band. Competing PAIRS test each side's
    ambiguity, so a strong unpaired edge can never authorize a replacement.
    """
    board = np.asarray(board, float)
    if board.shape != (4, 2) or not np.isfinite(board).all():
        return Detection(reason="invalid prior")
    left = max(board[1, 0], board[2, 0])
    right = min(board[0, 0], board[3, 0])
    if right - left < 60:
        return Detection(reason="board too short")
    # Do not let the existing (possibly wrong) end/nut dominate the side evidence.
    xs = np.linspace(left + .04 * (right - left), right - .04 * (right - left), 240)
    top = np.interp(xs, board[[1, 0], 0], board[[1, 0], 1])
    bottom = np.interp(xs, board[[2, 3], 0], board[[2, 3], 1])
    width = bottom - top
    centre = (top + bottom) / 2
    if width.min() < 12 or xs[0] < 0 or xs[-1] >= photo.shape[1]:
        return Detection(reason="board too narrow or outside photo")
    offsets = np.linspace(-.8, .8, 121)
    yy = centre[None] + offsets[:, None] * width[None]
    xx = np.broadcast_to(xs, yy.shape)
    if yy.min() < 0 or yy.max() >= photo.shape[0]:
        return Detection(reason="search window outside photo")
    # Crop before colour conversion: cost depends on neck ROI, not full photo size.
    x0, x1 = int(xs[0]), int(np.ceil(xs[-1])) + 1
    y0, y1 = int(yy.min()), int(np.ceil(yy.max())) + 1
    lab = cv2.cvtColor(photo[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), .8)
    strip = _at(lab, xx - x0, yy - y0)
    # Wider colour step rejects both sides of a narrow stripe. The local gradient
    # still pins the boundary, avoiding a broad maximum around a strong step.
    delta = np.zeros(strip.shape[:2], np.float32)
    delta[5:-5] = np.linalg.norm(
        np.mean([strip[5+d:len(strip)-5+d] for d in (2, 3, 4, 5)], axis=0)
        - np.mean([strip[5-d:len(strip)-5-d] for d in (2, 3, 4, 5)], axis=0), axis=2)
    gradient = np.linalg.norm(cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3), axis=2)
    # Do not clip the gradient to a plateau: that would let the prior win every
    # high-contrast neighbourhood without actually locating its boundary.
    response = np.minimum(delta / 35, 1) * gradient / (gradient + 80)
    response = cv2.blur(response, (7, 1))
    # Enumerate continuous lines; all candidates on BOTH sides participate in the
    # pair score. A line cannot zigzag onto successive fret/inlay intersections.
    shift = np.linspace(-.18, .18, 19)
    aa, bb = np.meshgrid(shift, shift)
    changes = np.c_[aa.ravel(), bb.ravel()]
    u = np.linspace(0, 1, len(xs))
    curves, evidence, coverage, bin_support = [], [], [], []
    for side in (-.5, .5):
        # Interpolate in PHOTO pixels, not in normalised strip coordinates:
        # changing width would otherwise turn a straight hypothesis into a curve.
        delta_y = changes[:, :1] * width[0] * (1 - u) + changes[:, 1:] * width[-1] * u
        rel = side + delta_y / width
        row = (rel + .8) / 1.6 * 120
        values = _at(response, np.broadcast_to(np.arange(len(xs)), row.shape), row)
        support = np.stack([np.mean(v > .3, axis=1) for v in np.array_split(values, 6, axis=1)], axis=1)
        coverage.append(np.mean(values > .3, axis=1))
        bin_support.append(support.min(axis=1))
        evidence.append(.65 * values.mean(axis=1) + .35 * np.quantile(values, .25, axis=1))
        curves.append(rel)
    # Joint geometry penalty: width, centre, and change of width relative to the
    # approximate taper. No assumed string count, handedness or guitar family.
    t, b = curves
    widths = b[:, None, [0, -1]] - t[None, :, [0, -1]]  # bottom x top x two ends
    centres = (b[:, None, [0, -1]] + t[None, :, [0, -1]]) / 2
    valid = ((widths >= .82) & (widths <= 1.16)).all(2)
    valid &= (np.abs(centres) <= .10).all(2)
    valid &= np.abs(widths[..., 1] - widths[..., 0]) <= .12
    taper = widths[..., 1] * width[-1] / (widths[..., 0] * width[0])
    valid &= (taper >= .60) & (taper <= 1.08)
    penalty = .35 * np.mean(np.abs(widths - 1), axis=2) + .35 * np.mean(np.abs(centres), axis=2)
    penalty += .3 * np.abs(widths[..., 1] - widths[..., 0])
    appearance = .5 * (evidence[1][:, None] + evidence[0][None])
    scores = appearance - penalty
    scores[~valid] = -np.inf
    appearance[~valid] = -np.inf
    bi, ti = np.unravel_index(np.argmax(scores), scores.shape)
    if not np.isfinite(scores[bi, ti]):
        return Detection(reason="no compatible edge pair")
    chosen = (ti, bi)
    # Distinct means displaced by >4% width on average, not adjacent samples of
    # the same antialiased edge. A competitor may change the OTHER side too.
    margins = []
    for side, idx in enumerate(chosen):
        distinct = np.mean(np.abs(curves[side] - curves[side][idx]), axis=1) > .04
        # Prior proximity must NOT turn two equally visible boundaries into an
        # "unambiguous" one. It ranks pairs, but contributes no confidence margin.
        alternative = appearance[:, distinct] if side == 0 else appearance[distinct, :]
        runner = np.max(alternative) if alternative.size else -np.inf
        margins.append(float(appearance[bi, ti] - runner))
    supports = [float(coverage[s][i]) for s, i in enumerate(chosen)]
    blocks = [float(bin_support[s][i]) for s, i in enumerate(chosen)]
    minimum = min(float(evidence[s][i]) for s, i in enumerate(chosen))
    margin = max(0.0, min(margins))
    confidence = float(min(minimum, min(supports), min(blocks), margin / (margin + .025)))
    edges = []
    for side, idx in enumerate(chosen):
        ys = centre + curves[side][idx] * width
        edges.append(np.c_[xs[[0, -1]], ys[[0, -1]]])
    diagnostics = {"support": supports, "weakest_block_support": blocks,
                   "pair_margins": margins, "pair_score": float(scores[bi, ti]),
                   "side_evidence": [float(evidence[s][i]) for s, i in enumerate(chosen)]}
    if min(supports) < .70 or min(blocks) < .45 or minimum < .40:
        reason = "weak or interrupted edge on at least one side"
    elif min(margins) < .025:
        reason = "ambiguous edge: competing compatible pair"
    else:
        reason = "continuous pair; compatible width/taper; both sides unambiguous"
        return Detection(True, confidence, reason, np.array(edges), diagnostics)
    return Detection(False, confidence, reason, np.array(edges), diagnostics)


def intersect(segment: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Intersect a fret/nut/end supporting line with BOTH accepted side lines."""
    p, q = np.asarray(segment, float)
    v = q - p
    out = []
    for a, b in edges:
        w = b - a
        matrix = np.column_stack((v, -w))
        if abs(np.linalg.det(matrix)) < 1e-8 * np.linalg.norm(v) * np.linalg.norm(w):
            raise ValueError("edge parallel to transverse line")
        s, _ = np.linalg.solve(matrix, a - p)
        out.append(p + s * v)
    return np.array(out)
