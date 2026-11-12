"""Conservative photo-supported concentric control candidates, never a knob template.

Only the body is searched. Geometry supplies hypotheses; independent radial
closure, internal rings and local colour contrast must support them. Ambiguous
hardware is left unchanged. All returned points are sampled photograph edges.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Candidate:
    centre: tuple[float, float]
    axes: tuple[float, float]
    angle: float
    source: str
    reason: str = "pending"
    metrics: dict = field(default_factory=dict)
    contour: np.ndarray | None = None


@dataclass
class Control:
    """Export-only label. Never changes the source SAM/photo contour."""

    contour: np.ndarray
    inner: np.ndarray | None = None


def fit_circle(points):
    """Uniform arc samples + robust radial fit, not an outlier-sensitive enclosing circle."""
    from scipy.optimize import least_squares
    from . import contour

    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 8 or not np.isfinite(p).all():
        return None
    if np.linalg.norm(np.ptp(p, axis=0)) < 1e-6:
        return None
    p = contour.resample(np.vstack([p, p[:1]]), max(np.linalg.norm(np.ptp(p, axis=0))/160, 1e-4))[:-1]
    centre = np.median(p, axis=0)
    radius = np.median(np.linalg.norm(p-centre, axis=1))
    if radius <= 1e-6:
        return None
    result = least_squares(lambda v: np.linalg.norm(p-v[:2], axis=1)-v[2],
                           [*centre, radius], loss="soft_l1", f_scale=.05*radius)
    x, y, r = result.x
    residual = np.abs(np.linalg.norm(p-[x,y], axis=1)-r)
    angles = np.sort(np.arctan2(p[:,1]-y, p[:,0]-x))
    gap = np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
    if (not result.success or r <= 0 or not np.isfinite(result.x).all()
            or np.quantile(residual, .9) > .3*r or gap > np.pi/3):
        return None
    return float(x), float(y), float(r)


def circle_geometry(control: Control):
    """One outer circle, optionally one reliable concentric inner rim; otherwise no override."""
    outer = fit_circle(control.contour)
    if outer is None:
        return []
    result = [outer]
    inner = fit_circle(control.inner) if control.inner is not None else None
    if inner is not None:
        x, y, r = outer
        ix, iy, ir = inner
        if np.hypot(ix-x, iy-y) <= .15*r and .35*r <= ir <= .82*r:
            result.append((x, y, ir))
    return result


def _control(c, q, origin, res):
    # Only a strongly supported measured radial-profile peak becomes an inner rim.
    rings = [v for v in q.metrics.get("inner_rings", []) if .5 <= v[0] <= .82 and v[2] >= .9]
    inner = None
    if rings:
        radius, _, _ = max(rings, key=lambda v: v[1])
        theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
        angle = np.deg2rad(q.angle)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        inner = (q.centre + np.c_[q.axes[0]*np.cos(theta), q.axes[1]*np.sin(theta)] @ rotation.T * radius)*res+origin
    return Control(c, inner)


def _sample(image, points):
    return cv2.remap(image, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def candidates(photo, mask, pixels_per_unit=2.) -> list[Candidate]:
    """Input is a body crop on a uniform drawing-unit grid, not full-frame pixels."""
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), .8)
    edges = cv2.Canny(blur, 40, 100)
    edges[mask == 0] = 0
    cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    proposals = []
    lo, hi = 6*pixels_per_unit, 18*pixels_per_unit
    for c in cs:
        if len(c) < 20:
            continue
        (x, y), (da, db), angle = cv2.fitEllipse(c)
        a, b = da/2, db/2
        if not (lo <= min(a, b) <= max(a, b) <= hi and min(a, b)/max(a, b) >= .65):
            continue
        fill = cv2.contourArea(c) / (np.pi*a*b)
        if .75 <= fill <= 1.15:
            proposals.append(Candidate((x, y), (a, b), angle, "ellipse"))
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 6*pixels_per_unit,
                               param1=100, param2=25, minRadius=round(lo), maxRadius=round(hi))
    if circles is not None:
        proposals.extend(Candidate((float(x), float(y)), (float(r), float(r)), 0., "circle") for x,y,r in circles[0])
    for q in proposals:
        verify(photo, mask, q)
        # Internal rims of one object must not become independent knobs. A
        # larger closed concentric boundary (even an unaccepted washer) wins.
        if q.reason == "supported" and any(
                p.source == "ellipse"
                and np.linalg.norm(np.array(p.centre)-q.centre) < .12*min(p.axes)
                and np.mean(q.axes) < .85*np.mean(p.axes) for p in proposals):
            q.reason = "inner_boundary"
        # A Hough accumulator also votes for rounded plate ends and groups of
        # saddles. It must agree with an independently closed Canny contour.
        if q.reason == "supported" and q.source == "circle":
            confirmed = any(p.source == "ellipse"
                            and np.linalg.norm(np.array(p.centre)-q.centre) < .15*min(q.axes)
                            and .85 < np.mean(p.axes)/np.mean(q.axes) < 1.15
                            for p in proposals)
            if not confirmed:
                q.reason = "no_closed_ellipse"
    return proposals


def verify(photo, mask, q: Candidate):
    """Check a proposed ellipse using edges at 96 angles, not a Hough score."""
    centre = np.array(q.centre)
    a, b = q.axes
    theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
    angle = np.deg2rad(q.angle)
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rays = np.c_[a*np.cos(theta), b*np.sin(theta)] @ rotate.T
    radial = np.linspace(.2, 1.25, 85)
    points = centre + radial[:, None, None] * rays[None]
    h, w = mask.shape
    if ((points < 1).any() or (points[...,0] >= w-1).any() or (points[...,1] >= h-1).any()
            or (_sample(mask, points) == 0).any()):
        q.reason = "outside_body"
        return
    lab = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), .8)
    strip = _sample(lab, points)
    # Derivative per image pixel, rather than a resolution-dependent Sobel sum.
    step = (radial[1]-radial[0]) * np.mean([a,b])
    gradient = np.linalg.norm(np.gradient(strip, axis=0), axis=2) / step
    profile = np.median(gradient, axis=1)
    outer_band = np.flatnonzero((radial >= .9) & (radial <= 1.1))
    outer = int(outer_band[np.argmax(profile[outer_band])])
    strength = float(profile[outer])
    near = np.flatnonzero(np.abs(radial-radial[outer]) <= .04)
    local = gradient[near]
    indices = near[np.argmax(local, axis=0)]
    peak = gradient[indices, np.arange(len(theta))]
    support = peak >= max(5., .4*strength)
    closure = float(support.mean())
    worst_sector = float(min(v.mean() for v in np.array_split(support, 8)))
    inner = []
    for j in range(1,len(radial)-1):
        if not (.35 <= radial[j] <= .82) or profile[j] < 5:
            continue
        if profile[j] >= profile[j-1] and profile[j] > profile[j+1]:
            share = float((gradient[max(0,j-1):j+2].max(0) >= max(5., .4*profile[j])).mean())
            if share >= .8:
                inner.append((float(radial[j]), float(profile[j]), share))
    # Neighbouring radial samples of the SAME edge are not independent rings.
    distinct = []
    for ring in sorted(inner, key=lambda r: -r[1]):
        if all(abs(ring[0]-other[0])*np.mean([a,b]) >= max(2., .1*np.mean([a,b])) for other in distinct):
            distinct.append(ring)
    inner = sorted(distinct)
    # A broad colour edge alone is a sunburst patch. An internal ring alone can
    # be a screw or washer. Require both plus an almost closed external edge.
    inside = _sample(lab, centre + .82*radial[outer]*rays)
    outside = _sample(lab, centre + 1.14*radial[outer]*rays)
    contrast = float(np.median(np.linalg.norm(inside-outside, axis=2)))
    q.metrics.update(outer_strength=strength, closure=closure, worst_sector=worst_sector,
                     contrast=contrast, inner_rings=inner, outer_scale=float(radial[outer]))
    q.contour = points[indices, np.arange(len(theta))]
    if strength < 5 or closure < .9 or worst_sector < .65:
        q.reason = "open_or_weak_outer"
    elif contrast < 12:
        q.reason = "low_contrast"
    elif not any(.5 <= radius <= .82 for radius, _, _ in inner) or len(inner) < 2:
        q.reason = "no_concentric_interior"
    else:
        q.reason = "supported"


def recover(t, body_end_x, *, diagnostics=None) -> list[np.ndarray]:
    """Additional body contours only; never replace an existing SAM detection.

    Small/blurred/edge-on knobs can intentionally remain missing. No assumed
    count, model family, layout or similarity to another detected knob is used.
    """
    from .trace import _rect

    res = .5
    x0, y0 = t.outline.min(0)-4
    x1, y1 = float(body_end_x)+40, float(t.outline[:, 1].max())+4
    photo, mask = _rect(t.photo, t.mask, t.ruler, x0, x1, y0, y1, res)
    proposals = candidates(photo, mask, pixels_per_unit=1/res)
    existing = [(c - [x0,y0])/res for c in t.parts]
    # SAM supplies a closed boundary even when Canny cannot propose one. Verify
    # it with EXACTLY the existing photo-evidence thresholds, for export only:
    # no additions, changed detection thresholds or assumptions about layout.
    controls = []
    for original, old in zip(t.parts, existing):
        if len(old) < 20:
            continue
        (cx, cy), (da, db), angle = cv2.fitEllipse(old.astype(np.float32))
        a, b = da/2, db/2
        if not (6/res <= min(a,b) <= max(a,b) <= 18/res and min(a,b)/max(a,b) >= .65):
            continue
        fill = abs(cv2.contourArea(old.astype(np.float32))) / (np.pi*a*b)
        if not .75 <= fill <= 1.15:
            continue
        q = Candidate((cx,cy), (a,b), angle, "existing")
        verify(photo, mask, q)
        if q.reason == "supported":
            controls.append(_control(original, q, [x0,y0], res))
    added = []
    # Prefer a closed ellipse over an accumulator circle and the strongest edge
    # over the two sides of the same Canny edge. Sorting is deterministic.
    order = sorted(range(len(proposals)), key=lambda i: (proposals[i].source != "ellipse",
                                                       -proposals[i].metrics.get("outer_strength", 0)))
    for i in order:
        q = proposals[i]
        if q.reason != "supported":
            continue
        c = q.contour
        area = abs(cv2.contourArea(c.astype(np.float32)))
        for old in existing:
            old_area = abs(cv2.contourArea(old.astype(np.float32)))
            if not .55*area <= old_area <= 1.8*area:
                continue
            # Rasterise only a tiny local window, not one full body per part.
            origin = np.floor(c.min(0)-3)
            w, h = np.ceil(c.max(0)-origin+4).astype(int)
            a, b = np.zeros((h,w),np.uint8), np.zeros((h,w),np.uint8)
            cv2.fillPoly(a,[np.round(c-origin).astype(np.int32)],1)
            cv2.fillPoly(b,[np.round(old-origin).astype(np.int32)],1)
            overlap = (a & b).sum() / max(a.sum(),1)
            if overlap >= .7:
                q.reason = "existing_contour"
                q.metrics["overlap"] = float(overlap)
                break
        if q.reason != "supported":
            continue
        existing.append(c)
        added.append(c*res+[x0,y0])
        controls.append(_control(added[-1], q, [x0,y0], res))
        q.reason = "added"
    t.knobs = controls
    if diagnostics is not None:
        diagnostics.update(candidates=proposals, photo=photo, mask=mask, origin=(x0,y0), res=res)
    return added
