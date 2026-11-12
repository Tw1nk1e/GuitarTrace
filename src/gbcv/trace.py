"""Trace a catalogue photo (transparent or plain background) into a line drawing, the same way for any guitar:
the exact silhouette, the fretboard with frets, tuner posts, and every part on the body as SAM 2 cuts it
out. No model-specific rules, no templates, no dimensions, no strings — the photo's own proportions.

Internal nominal units set line widths and smoothing thresholds; they are not measured dimensions.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import contour, fretboard, frets
from .orient import FOCAL, orient_and_split
from .split import split

OUTLINE_SMOOTH_PX = 3.0  # irons out the pixel staircase and edge ripple (≈ 0.5 mm on a 4000 px photo)
OUTLINE_TOL_MM = 0.1  # simplification tolerance: hundreds of points stay
BOARD_PAST_LAST_FRET = 6  # mm
STYLES = {  # line weights, mm
    "thin": {"main": 1.2, "det": 0.6, "thin": 0.35},
    "comic": {"main": 3.4, "det": 1.7, "thin": 0.9},  # bold outline, rounded ends
    "png": {"main": 5.5, "det": 3.5, "thin": 2.2},  # Keep lines visible in small PNG previews (~160 px high).
}
DEFAULT_SMOOTH_MM = 1.5  # outline and parts smoothing in the drawing (SAM masks jitter ~1–2 mm)
PART_SMOOTH_SHARE = 1.0  # parts get the same
CURVE_TOL_MM = 0.8  # how far a Bézier may run off the traced outline (a horn tip keeps its place)
PART_CURVE_TOL_MM = 0.5  # polygon simplification; NOT a bound on the final cubic's deviation
SAM_GRID_MM = 28  # prompt grid over the body (40 already loses parts)
SAM_PAST_BOARD_MM = 70  # the body crop runs this far past the fretboard end (a pickguard can run on to the cutaway)
SAM_MIN_SCORE = 0.7  # chrome knobs and washers come out less sure
SAM_MIN_MM2 = 12  # smaller is a speck
SAM_MAX_FRAC = 0.35  # of the body: bigger is the top itself
PART_EDGE_GAP_MM = 2.5  # a part's outline this close to the silhouette counts as running along it
PART_MAX_ALONG_EDGE = 0.2  # more of its outline along the body's edge: glare or a bevel highlight, not a part
PART_SAME_HUE = 12  # a, b distance (Lab): closer is the body's own colour
PART_MIN_STEP = 25  # median brightness step across its outline (Sobel, Lab L): below is a sunburst patch or soft shadow
LUMP_MIN_TAPER = 0.75  # strap buttons ≥ 0.93, horn tips (Ibanez, SG) 0.38–0.57
POST_INWARD_MM = 10  # a tuner post sits this far in from the head edge at its key, when its bushing isn't seen
HEAD_TURN_MM = 6  # the boundary is read over ±this many mm to see where it turns inward at a key's base
HEAD_CONCAVE = 0.08  # turn (radians/px normalised) below this is a concave point: a valley between keys
KEY_MAX_MM = 60  # coarse boundary-arc proposal limit; NOT sufficient evidence of a button
KEY_BULGE_MM = 4  # and it has to stand this far off its own chord to count as a key
KEY_MIN_MM2 = 20  # smaller is a speck of the head's edge or a shaving of it
KEY_GAP_MM = 12  # the bay a cut key leaves between its neighbours is closed with this kernel
BOARD_SIDE_GAP_MM = 2.0  # a board edge this close to the silhouette is the neck's own edge, not a second line


@dataclass
class Trace:
    outline: np.ndarray  # closed silhouette, mm
    nut_x: float
    scale_length: float
    frets: list[tuple[float, float, float]]  # per fret, mm: x at y = -h, x at y = +h (differ: fanned frets), h
    board: np.ndarray  # fretboard polygon, mm
    posts: list[tuple[float, float, float]] = field(default_factory=list)  # tuner posts x, y, r (mm)
    keys: list[np.ndarray] = field(default_factory=list)  # tuner keys as separate shapes, closed outlines, mm
    parts: list[np.ndarray] = field(default_factory=list)  # body parts as SAM 2 cuts them out, closed outlines, mm
    photo: np.ndarray | None = None  # rotated photo, for the overlay check
    ruler: frets.Ruler | None = None
    mask: np.ndarray | None = None
    fret_lines: list[np.ndarray] | None = None  # accepted photo-edge endpoints, drawing units
    board_detection: fretboard.Detection | None = None
    knobs: list = field(default_factory=list)  # confirmed controls; references to unchanged part contours


def silhouette(rgba: np.ndarray) -> np.ndarray:
    """Mask from the alpha channel; on an opaque photo from the background colour if the background is plain
    (catalogue shot: pixel-exact), otherwise from the BiRefNet segmentation (any photo)."""
    a = rgba[..., 3]
    if (a < 128).mean() > 0.05:
        m = a >= 128
    else:
        rgb = rgba[..., :3].astype(float)
        corners = np.vstack([rgb[:8, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3), rgb[-8:, :8].reshape(-1, 3)])
        if corners.std(0).max() < 6:  # one plain colour in all four corners
            m = np.linalg.norm(rgb - np.median(corners, 0), axis=2) > 30
        else:
            from . import segment

            img = np.ascontiguousarray(rgba[..., :3])
            k = min(1.0, 2048 / max(img.shape[:2]))
            small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
            prob = cv2.resize(segment.probability(small), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            m = prob >= 0.5
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        raise ValueError("guitar not found")
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # fill holes (f-holes, gaps between keys)
    out = np.zeros_like(m)
    cv2.drawContours(out, cs, -1, 255, cv2.FILLED)
    return out


def _outline_px(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    k = int(4 * OUTLINE_SMOOTH_PX)  # circular smoothing
    ext = np.vstack([c[-k:], c, c[:k]])
    ext = contour.smooth(ext, OUTLINE_SMOOTH_PX)
    return ext[k:-k]


def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), tol, closed=True)
    return approx[:, 0, :].astype(float)


def _inward_bushings(blur: np.ndarray, plate: np.ndarray, kx: float, edge: float,
                     side: int, d: float, bounds=(-np.inf, np.inf)) -> list[tuple[float, float, float]]:
    """Recover an unseen post without assuming the shaft leans towards the nut.

    The ROI includes the *whole circle*, not just its allowed centre positions.
    Only circles wholly on the plate and on this side's inward half qualify.
    Return Hough's ranked proposals; neighbouring mechanisms resolve concentric
    chrome edges with different apparent centres, rather than a fixed offset.
    """
    column = np.flatnonzero(plate[:, int(kx)])
    depth = min(40 * d, .5 * (column[-1] - column[0]))
    radius = 7 * d
    x0, x1 = max(0, int(kx - 20*d - radius)), min(blur.shape[1], int(kx + 20*d + radius) + 1)
    ya, yb = sorted([edge - side*5*d, edge - side*depth])
    y0, y1 = max(0, int(ya-radius)), min(blur.shape[0], int(yb+radius)+1)
    region = blur[y0:y1, x0:x1]
    circles = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, dp=1, minDist=max(1, int(3*d)),
                               param1=80, param2=20, minRadius=max(1, int(3*d)),
                               maxRadius=max(2, int(radius))) if region.size else None
    if circles is None:
        return []
    angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
    found = []
    for cx, cy, r in circles[0]:
        x, y = float(cx+x0), float(cy+y0)
        inward = -side*(y-edge)
        if not (5*d <= inward <= depth and abs(x-kx) <= 20*d and bounds[0] < x < bounds[1]):
            continue
        ring = np.round(np.c_[x+r*np.cos(angles), y+r*np.sin(angles)]).astype(int)
        if ((ring < 0).any() or (ring[:, 0] >= plate.shape[1]).any()
                or (ring[:, 1] >= plate.shape[0]).any()):
            continue
        if plate[round(y), round(x)] and (plate[ring[:, 1], ring[:, 0]] > 0).all():
            found.append((x, y, float(r)))
    return found


def _posts(gray: np.ndarray, ruler: frets.Ruler, plate: np.ndarray, keys: list[np.ndarray]) -> list[tuple[float, float, float]]:
    """Tuner posts: one per key. The post sits a little in from the plate's edge, opposite its key — where the round
    bushing is (Hough), or at the distance the other keys showed. Keys come from `head_parts`, so the plate's edge is
    the headstock's own, not an opening's guess."""
    d = ruler.d
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    seen, offsets, sides, missed = [], [], [], []
    locations = []
    for comp in keys:
        if comp.sum() < KEY_MIN_MM2 * d * d:
            continue
        ys, xs = np.nonzero(comp)
        kx, ky = float(xs.mean()), float(ys.mean())
        col = (plate[:, int(kx)] > 0).nonzero()[0]
        if not col.size:
            continue
        side = -1 if abs(ky - col.min()) < abs(ky - col.max()) else 1  # key above (bass) or below the head, in px
        edge = col.min() if side < 0 else col.max()
        locations.append((kx, side))
        # look for the bushing 5–18 mm in from the edge, 14 mm towards the nut .. 6 mm past the key (keys lean)
        x0, x1 = max(0, int(kx - 14 * d)), min(gray.shape[1], int(kx + 6 * d))
        y0, y1 = sorted([int(edge - side * 5 * d), int(edge - side * 18 * d)])
        y0, y1 = max(0, y0), min(gray.shape[0], y1)
        reg = blur[y0:y1, x0:x1]
        cs = cv2.HoughCircles(reg, cv2.HOUGH_GRADIENT, dp=1, minDist=max(1, int(8 * d)), param1=80, param2=14,
                              minRadius=max(1, int(3 * d)), maxRadius=max(2, int(7 * d))) if reg.size else None
        if cs is not None:
            cx, cy, r = cs[0][0]  # strongest
            q = ruler.to_mm(np.array([[x0 + cx, y0 + cy]]))[0]
            seen.append((float(q[0]), float(q[1]), float(min(r / d, 6))))
            offset = np.array(seen[-1][:2]) - ruler.to_mm(np.array([[kx, edge]]))[0]
            offsets.append(offset * [1, -side])  # inward distance, not a signed Y shared across opposing rows
            sides.append(side)
        else:
            missed.append((kx, edge, side))
    # Preserve well-supported rows (including a missed bushing among clear ones).
    # Where there is no trustworthy row, search wider before inventing a post.
    unresolved = []
    recovered = []
    legacy_rows = {side: [offset for offset, s in zip(offsets, sides) if s == side] for side in (-1, 1)}
    for kx, edge, side in missed:
        row = legacy_rows[side]
        supported = len(row) >= 2 and np.max(np.ptp(row, axis=0)) < 6
        neighbours = [x for x, s in locations if s == side]
        left = max((x for x in neighbours if x < kx), default=-np.inf)
        right = min((x for x in neighbours if x > kx), default=np.inf)
        bounds = ((left+kx)/2, (right+kx)/2)
        candidates = [] if supported else _inward_bushings(blur, plate, kx, edge, side, d, bounds)
        if not candidates:
            unresolved.append((kx, edge, side))
            continue
        base = ruler.to_mm(np.array([[kx, edge]]))[0]
        options = [(ruler.to_mm(np.array([[x, y]]))[0], radius) for x, y, radius in candidates]
        recovered.append((side, base, options))
    # Estimate offsets from independently detected circles, then resolve the
    # washer/post ambiguity against the other mechanisms. No assumed row count.
    proposals = offsets + [(options[0][0]-base)*[1, -side] for side, base, options in recovered]
    proposal_sides = sides + [side for side, _, _ in recovered]
    radii = [q[2] for q in seen] + [options[0][1]/d for _, _, options in recovered]
    anchor_count = len(offsets)
    for index, (side, base, options) in enumerate(recovered):
        # Leave this mechanism out: its strongest chrome reflection must not
        # pull its own reference towards the very ambiguity being resolved.
        own = anchor_count + index
        row = [offset for j, (offset, s) in enumerate(zip(proposals, proposal_sides)) if s == side and j != own]
        expected = np.median(row, axis=0) if len(row) >= 2 else None
        other_radii = [radius for j, radius in enumerate(radii) if j != own]
        expected_radius = float(np.median(other_radii)) if len(other_radii) >= 2 else None
        best = 0 if expected is None else min(range(len(options)), key=lambda i:
            np.linalg.norm((options[i][0]-base)*[1, -side]-expected)
            + (abs(options[i][1]/d-expected_radius) if expected_radius is not None else 0) + .05*i)
        q, radius = options[best]
        seen.append((float(q[0]), float(q[1]), float(min(radius/d, 6))))
        offset = q - base
        offsets.append(offset * [1, -side])
        sides.append(side)
    # New detections can now supply the geometry for still-unseen mechanisms.
    off = np.median(offsets, axis=0) if offsets else None
    out = list(seen)
    for kx, edge, side in unresolved:
        base = ruler.to_mm(np.array([[kx, edge]]))[0]
        row = [offset for offset, s in zip(offsets, sides) if s == side]
        estimate = np.median(row, axis=0) if row else off
        dx, inward = estimate if estimate is not None else (0.0, POST_INWARD_MM)
        out.append((float(base[0] + dx), float(base[1] - side * inward), 4.5))
    out = sorted(out, key=lambda q: q[0])
    r = float(np.clip(np.median([q[2] for q in out]), 4.0, 6.0)) if out else 4.5  # Hough takes the bushing or the post
    out = [(x, y, r) for x, y, _ in out]
    # one row of six (a Fender head): a post off the line (glare, a string tree) goes back onto it. A 3+3 or a
    # 4+5 head has two rows — leave it alone, or every post gets pulled onto a line that does not exist.
    if len(out) == 6 and np.ptp([q[1] for q in out]) < 25:  # drawing units
        xs, ys = np.array([q[0] for q in out]), np.array([q[1] for q in out])
        k, b = np.polyfit(xs, ys, 1)
        dev = np.abs(ys - (k * xs + b))
        if dev.max() > 4:
            keep = dev <= np.median(dev) * 3 + 1
            k, b = np.polyfit(xs[keep], ys[keep], 1)
            out = [(x, y if ok else k * x + b, r) for (x, y, r), ok in zip(out, keep)]
    return out


def _more_frets(gray: np.ndarray, sp, ff: frets.FretFit, band=(-1.0, 1.0)) -> int:
    """Fret count. The search stops where the body starts, but on a Les Paul frets 21–22 sit over it: a predicted
    fret counts if it shows on the photo and there is still wood just past it on the body side (both board edges —
    inlays are in the middle, a pickguard touches one edge only near the end, a pickup ring both)."""
    g = lambda n: 1 - 2.0 ** (-n / 12)
    x = lambda n: (ff.b - ff.a * g(n)) / (1 + ff.c * g(n))
    last = ff.frets
    strip, xs = frets.board_strip(gray, sp, max(int(x(max(last + 6, frets.MAX_FRETS + 1))), 0), int(ff.b), band)
    resp, _ = frets.response(strip, sigma=max(1.0, 0.012 * sp.neck_width(sp.nut_x)))
    col = lambda px: int(round(px - xs[0]))
    k = max(1, strip.shape[0] // 5)
    edges = [np.median(r, axis=0) for r in (strip[:k], strip[-k:])]
    wood = [np.median(t[col(x(last)) : col(x(last - 3))]) for t in edges]
    n = last
    while n < 24:
        p, spacing = x(n + 1), x(n) - x(n + 1)
        lo, hi = col(p - 0.25 * spacing), col(p + 0.25 * spacing) + 1
        a, b = col(p - 0.4 * spacing), col(p - 0.2 * spacing)
        if a < 0 or resp[lo:hi].max() <= 0.35 or any(abs(np.median(t[a:b]) - w) > 40 for t, w in zip(edges, wood)):
            break
        n += 1
    return n


def _line_mm(line, sp, ruler) -> tuple[float, float, float]:
    """One line (canvas px ends inside the board) as (x at -h, x at +h, h) in drawing units: the line is
    extended out to the fretboard's edges along its own direction."""
    (x1, y1), (x2, y2) = line
    xm = (x1 + x2) / 2
    h = sp.neck_width(xm) / 2
    c = sp.neck_centre(xm)
    slope = (x2 - x1) / (y2 - y1)
    xa, xb = x1 + (c - h - y1) * slope, x1 + (c + h - y1) * slope
    p = ruler.to_mm(np.array([[xa, c - h], [xb, c + h]]))
    return float(p[0, 0]), float(p[1, 0]), float(h / ruler.d)




def _rect(photo, mask, ruler, x0, x1, y0, y1, res):
    """Photo and mask resampled onto a grid in mm (res mm per px): rows = Y from y0, columns = X from x0."""
    X, Y = np.arange(x0, x1, res), np.arange(y0, y1, res)
    gx, gy = np.meshgrid(X, Y)
    px = ruler.to_px(np.c_[gx.ravel(), gy.ravel()]).astype(np.float32)
    mx, my = px[:, 0].reshape(gx.shape), px[:, 1].reshape(gx.shape)
    return cv2.remap(photo, mx, my, cv2.INTER_LINEAR), cv2.remap(mask, mx, my, cv2.INTER_NEAREST)


def _to_mm(pts_px, x0, y0, res):
    return np.asarray(pts_px, float) * res + [x0, y0]


def _taper(dd: np.ndarray) -> float:
    """A lump's width near its end over its width near the base (dd: its pixels' distances from the opened edge).
    A knob keeps its width (a strap button's head is as wide as its stem); a horn tip runs to a point."""
    mx = dd.max()
    width = lambda a, b: ((dd >= a * mx) & (dd < b * mx)).sum() / ((b - a) * mx)
    return width(0.7, 0.9) / max(width(0.2, 0.4), 1e-6)


def _no_strap_buttons(mask: np.ndarray, ruler, nut_px: float) -> np.ndarray:
    """Cut small knobs sticking out of the body outline (strap buttons, a jack): what an opening removes, kept only
    where it is a compact lump standing well proud of the opened edge that does not taper to a point (a horn tip)."""
    d = ruler.d
    k = int(14 * d) | 1
    body = mask.copy()
    body[:, int(nut_px):] = 0
    opened = cv2.morphologyEx(body, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    diff = ((body > 0) & (opened == 0)).astype(np.uint8)
    dist = cv2.distanceTransform((opened == 0).astype(np.uint8), cv2.DIST_L2, 5)
    core = (diff > 0) & (dist > 1.0 * d)  # drops the hairline the opening shaves off along every edge
    n, lab, st, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
    out = mask.copy()
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 * d) | 1,) * 2)
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        x, _, w, _, area = st[i]
        if nut_px - (x + w) < k:  # the corners the cut at the nut leaves, not a knob
            continue
        dd = dist[comp > 0]
        if area < 400 * d * d and dd.max() > 2.5 * d and _taper(dd) > LUMP_MIN_TAPER:
            out[(cv2.dilate(comp, grow) > 0) & (diff > 0)] = 0
    return out


def _sam_parts(photo, mask, ruler, body_end_x, half, board_half, debug=None) -> list[np.ndarray]:
    """Every part on the body as SAM 2 cuts it out from a grid of prompt points. Masks are cut off at the neck band
    (a pickguard runs on under the strings). Kept: masks SAM is sure of, inside the guitar, not the whole top, not a
    repeat. Outlines in mm."""
    from . import sam

    d = ruler.d
    xe = min(mask.shape[1], int(ruler.to_px([[body_end_x + SAM_PAST_BOARD_MM, 0]])[0, 0]))  # a Tele's pickguard runs on past the board end
    cols = np.flatnonzero(mask[:, :xe].any(0))
    rows = np.flatnonzero(mask[:, :xe].any(1))
    x0, x1, y0, y1 = int(cols[0]), xe, int(rows[0]), int(rows[-1]) + 1
    crop, cmask = photo[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 0
    to_crop = lambda pts: ruler.to_px(np.asarray(pts, float)) - [x0, y0]
    ok = lambda u, v: 0 <= u < cmask.shape[1] and 0 <= v < cmask.shape[0] and cmask[int(v), int(u)]
    grid_mm = np.array([(x, y) for x in np.arange(10, body_end_x - 5, SAM_GRID_MM) for y in np.arange(-half, half, SAM_GRID_MM)])
    pts = to_crop(grid_mm)
    pts = pts[[ok(u, v) for u, v in pts]]
    sess = sam.Session(crop)
    k = sess.k
    body = cv2.resize(cmask.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST) > 0
    lab = cv2.cvtColor(cv2.resize(crop, body.shape[::-1], interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(float)
    lab = cv2.GaussianBlur(lab, (0, 0), 2)
    proposals = sess.each(pts, min_score=float("-inf") if debug is not None else SAM_MIN_SCORE)
    if debug is not None:
        debug.start(proposals, pts, k, (x0, y0, x1, y1), cmask)
    cands = sorted(enumerate(proposals), key=lambda c: -c[1][1])
    mm2 = 1 / (k * d) ** 2  # mm² per mask pixel
    # the neck band past the board end: masks run on along it under the strings (a Tele pickguard) — cut it off them
    band = np.array([[body_end_x, -board_half - 2], [body_end_x + 2 * SAM_PAST_BOARD_MM, -board_half - 2],
                     [body_end_x + 2 * SAM_PAST_BOARD_MM, board_half + 2], [body_end_x, board_half + 2]])
    off_neck = np.ones(body.shape, np.uint8)
    cv2.fillPoly(off_neck, [np.round(to_crop(band) * k).astype(np.int32)], 0)
    cands = [(cid, m & off_neck, sc, i) for cid, (m, sc, i) in cands]
    fits = lambda m, a: SAM_MIN_MM2 < a * mm2 < SAM_MAX_FRAC * body.sum() * mm2 and (m & body).sum() >= 0.97 * a

    kept = []  # (mask, bbox, area, candidate id)
    for cid, m, sc, _ in cands:
        if debug is not None:
            debug.mask(cid, "off_neck", m)
            debug.records[cid].update(area_mm2=float(m.astype(bool).sum() * mm2),
                                      neck_removed_px=int(np.count_nonzero(proposals[cid][0]) - np.count_nonzero(m)))
        if sc < SAM_MIN_SCORE:
            if debug is not None:
                debug.reject(cid, "score", score=float(sc))
            continue
        mb = m.astype(bool)
        a = mb.sum()
        if not fits(mb, a):
            if debug is not None:
                reason = "area_min" if a * mm2 <= SAM_MIN_MM2 else "area_max" if a >= SAM_MAX_FRAC * body.sum() else "outside_body"
                debug.reject(cid, reason, area_mm2=float(a * mm2), inside_share=float((mb & body).sum() / max(a, 1)))
            continue
        ys, xs = np.nonzero(mb)
        bb = (xs.min(), ys.min(), xs.max(), ys.max())
        rep = False
        for q, qb, qa, qid in kept:
            if bb[0] > qb[2] or qb[0] > bb[2] or bb[1] > qb[3] or qb[1] > bb[3]:
                continue
            if abs(a - qa) < 0.15 * max(a, qa) and (mb & q).sum() > 0.85 * min(a, qa):
                rep = True
                if debug is not None:
                    debug.reject(cid, "duplicate", duplicate_of=qid)
                break
        if not rep:
            kept.append((mb, bb, a, cid))
    out = []
    kk = int(3 * k * d) | 1  # closes the gaps strings leave along a mask (a string is ~1 mm)
    for mb, _, _, cid in kept:
        mb = cv2.morphologyEx(mb.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        if debug is not None:
            debug.mask(cid, "closed", mb)
        cs, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        if debug is not None:
            debug.contour(cid, "raw", ruler.to_mm(c / k + [x0, y0]))
            debug.records[cid]["component_areas_px"] = [float(cv2.contourArea(q)) for q in cs]
        if len(c) < 8:
            if debug is not None:
                debug.reject(cid, "contour_points")
            continue
        c = contour.smooth(np.vstack([c, c[:1]]), 1.0) / k + [x0, y0]
        c = ruler.to_mm(c)
        if debug is not None:
            debug.contour(cid, "mask_smoothed", c)
        c = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), 0.2, True)[:, 0, :].astype(float)
        if len(c) >= 4:
            out.append(c)
            if debug is not None:
                debug.part_ids[id(c)] = cid
                debug.contour(cid, "approximated", c)
                debug.records[cid]["status"] = "sam_kept"
        elif debug is not None:
            debug.reject(cid, "approximation_points")
    return out


def _real_parts(t: "Trace", parts: list[np.ndarray], debug=None) -> list[np.ndarray]:
    """Drop what SAM cuts out of light, not of objects: glare and bevel highlights run along the body's edge and are
    the body's own colour, only lighter (a pickguard along the edge is another colour; a knob or a pickup never runs
    along it); a sunburst patch or a soft shadow has no edge at all."""
    from scipy.spatial import cKDTree

    edge_tree = cKDTree(contour.resample(np.vstack([t.outline, t.outline[:1]]), 1.0))
    lab = cv2.cvtColor(t.photo, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), max(1.0, 0.5 * t.ruler.d))
    k = int(4 * t.ruler.d) | 1
    grad = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    keep = []
    for c in parts:
        cid = debug.part_ids[id(c)] if debug is not None else None
        ring = contour.resample(np.vstack([c, c[:1]]), 1.0)
        along_edge = float(np.mean(edge_tree.query(ring)[0] < PART_EDGE_GAP_MM))
        px = np.round(t.ruler.to_px(ring)).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, grad.shape[1] - 1)
        px[:, 1] = np.clip(px[:, 1], 0, grad.shape[0] - 1)
        step = float(np.median(grad[px[:, 1], px[:, 0]]))
        if debug is not None:
            debug.records[cid].update(edge_step=step, along_edge=along_edge)
        if step < PART_MIN_STEP:
            if debug is not None:
                debug.reject(cid, "weak_edge")
            continue
        if along_edge > PART_MAX_ALONG_EDGE:  # glare or a part along the edge: compare its hue with the body round it
            m = np.zeros(t.mask.shape, np.uint8)
            cv2.fillPoly(m, [np.round(t.ruler.to_px(c)).astype(np.int32)], 1)
            ring_out = (cv2.dilate(m, np.ones((k, k), np.uint8)) > 0) & (m == 0) & (t.mask > 0)
            inside = cv2.erode(m, np.ones((k, k), np.uint8)) > 0
            if ring_out.sum() > 20 and inside.sum() > 20:
                hue = np.linalg.norm(np.median(lab[..., 1:][inside], 0) - np.median(lab[..., 1:][ring_out], 0))
                if hue < PART_SAME_HUE:
                    if debug is not None:
                        debug.reject(cid, "body_hue", hue_distance=float(hue))
                    continue  # the body's own colour, lighter or darker: light on the lacquer
        keep.append(c)
        if debug is not None:
            debug.records[cid]["status"] = "kept"
    return keep


def outline_of(mask: np.ndarray, ruler) -> np.ndarray:
    """The silhouette's outline in mm: strap buttons cut, pixel steps smoothed, simplified."""
    return _simplify(ruler.to_mm(_outline_px(_no_strap_buttons(mask, ruler, float(ruler.ff.b)))), OUTLINE_TOL_MM)


def head_closed(mask: np.ndarray, sp) -> np.ndarray:
    """The head is a comb: keys, string ends and shadows cut deep narrow notches into the mask. Close them for
    the outline only — the tuner search needs the keys separate from the head they sit on."""
    out = mask.copy()
    hx = int(sp.nut_x)
    kc = max(3, int(0.12 * sp.neck_width(sp.nut_x))) | 1
    out[:, hx:] = cv2.morphologyEx(out[:, hx:], cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    return out


def _ellipse(k: float) -> np.ndarray:
    k = max(3, int(round(k)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _key_is_button(comp: np.ndarray, plate: np.ndarray, ruler, gray: np.ndarray | None = None) -> bool:
    """A compact lateral button, not a shallow ribbon cut off the head's tip.

    Test the candidate *and its attachment*, not an expected key count or an
    absolute length. Slender edge-on buttons remain valid. Unusual/ambiguous
    shapes need a visible bushing on the inward side; no evidence means keeping
    the original plate, not inventing a key and a fallback post.
    """
    ys, xs = np.nonzero(comp)
    if len(xs) < 3:
        return False
    pts = np.c_[xs, ys].astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    axes = np.diff(box[:3], axis=0)
    sizes = np.linalg.norm(axes, axis=1)
    long = int(np.argmax(sizes))
    length, width = sizes[long], sizes[1 - long]
    if width < 1:
        return False
    hull = cv2.convexHull(pts)
    perimeter = cv2.arcLength(hull, True)
    compactness = 4 * np.pi * cv2.contourArea(hull) / max(perimeter**2, 1)
    fill = len(xs) / (length * width)
    contact = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (plate > 0)
    cy, cx = np.nonzero(contact)
    if not len(cx):
        return False
    attachment = np.c_[cx, cy]
    outward = pts.mean(0) - attachment.mean(0)
    distance = np.linalg.norm(outward)
    # A terminal tip projects along the guitar axis; lateral buttons project
    # across it. Broad contact along most of a thin shape is a cut plate edge.
    lateral = abs(outward[1]) / max(distance, 1)
    contact_share = np.ptp(attachment @ (axes[long] / length)) / length
    ordinary = (length / width <= 3.5 and compactness >= .52 and fill >= .58
                and contact_share <= .75 and lateral >= .5)
    if ordinary:
        return True
    if gray is None or lateral < .35 or contact_share > .9:
        return False
    # Reuse the post search, but a shape cannot authorize its own synthetic
    # fallback. Only an actual, inward bushing can resolve an ambiguous shape.
    kx, ky = pts.mean(0)
    col = np.flatnonzero(plate[:, int(kx)])
    if not col.size:
        return False
    side = -1 if abs(ky - col.min()) < abs(ky - col.max()) else 1
    edge = col.min() if side < 0 else col.max()
    d = ruler.d
    x0, x1 = max(0, int(kx - 14 * d)), min(gray.shape[1], int(kx + 6 * d))
    y0, y1 = sorted([int(edge - side * 5 * d), int(edge - side * 18 * d)])
    y0, y1 = max(0, y0), min(gray.shape[0], y1)
    reg = cv2.GaussianBlur(gray, (0, 0), 1.0)[y0:y1, x0:x1]
    circles = cv2.HoughCircles(reg, cv2.HOUGH_GRADIENT, dp=1, minDist=max(1, int(8 * d)),
                              param1=80, param2=14, minRadius=max(1, int(3 * d)),
                              maxRadius=max(2, int(7 * d))) if reg.size else None
    if circles is None:
        return False
    for x, y, radius in circles[0]:
        x, y = x + x0, y + y0
        # The complete bushing must be on the plate and opposite the button,
        # not an unrelated circular highlight on the outside of a tip.
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ring = np.round(np.c_[x + radius * np.cos(angles), y + radius * np.sin(angles)]).astype(int)
        if ((ring < 0).any() or (ring[:, 0] >= plate.shape[1]).any()
                or (ring[:, 1] >= plate.shape[0]).any()):
            continue
        if (side * (y - edge) < 0 and plate[int(round(y)), int(round(x))] > 0
                and (plate[ring[:, 1], ring[:, 0]] > 0).all()):
            return True
    return False


def head_parts(mask: np.ndarray, ruler, nut_px: float, gray: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """The head split into its plate and the keys standing out of it.

    The keys are part of the silhouette, so an outline of the whole head runs round every key — a comb instead of a
    headstock. Morphology can't help: a key sits on the plate over a base as wide as the key itself, so an opening
    that cuts it also eats the plate's tip. What does separate them is the shape: at a key's base the boundary turns
    *inward* (concave). The stretch of boundary between two such points is either a key (short, bulging off the
    chord) or a real edge of the plate. Chords only propose candidates: compactness, lateral attachment and,
    for ambiguous shapes, a visible inward bushing decide which cuts survive. Rejected cuts are restored.
    """
    d = ruler.d
    head = mask.copy()
    head[:, : int(nut_px + 20 * d)] = 0
    if not (head > 0).any():
        return mask, mask, []
    cs, _ = cv2.findContours(head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return mask, mask, []
    from scipy.ndimage import gaussian_filter1d

    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    ring = contour.resample(np.vstack([c, c[:1]]), 1.0)[:-1]
    if len(ring) < 20:
        return mask, mask, []
    ring = gaussian_filter1d(ring, 1.5, axis=0, mode="wrap")
    n = len(ring)
    i = np.arange(n)
    k = max(3, int(HEAD_TURN_MM * d))
    u, v = ring[(i + k) % n] - ring, ring - ring[(i - k) % n]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    turn = cross / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    concave = turn < -HEAD_CONCAVE
    runs, cur = [], []
    for j in range(n):
        if not concave[j]:
            cur.append(j)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:  # a ring: join the two halves of one stretch
        runs[0] = [x - n for x in runs[-1]] + runs[0]
        runs.pop()
    plate_ring = ring.copy()
    for r in runs:
        if len(r) < 3 or len(r) / d > KEY_MAX_MM:
            continue
        p, q = ring[r[0]], ring[r[-1]]
        ab = q - p
        L = np.linalg.norm(ab) + 1e-9
        seg = ring[r] - p
        bulge = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        if bulge.max() < KEY_BULGE_MM * d:
            continue
        for j, idx in enumerate(r):
            plate_ring[idx % n] = p + ab * ((j + 1) / (len(r) + 1))
    plate = np.zeros_like(head)
    cv2.fillPoly(plate, [np.round(plate_ring).astype(np.int32)], 255)
    plate &= head
    keys = []
    rest = ((head > 0) & (plate == 0)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(rest)
    cut_mask = np.zeros_like(head)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < KEY_MIN_MM2 * d * d:
            continue
        cx, cy = cen[c]
        if plate[int(cy), int(cx)] > 0:  # its middle is on the plate: a bay or a notch, not a key standing out
            continue
        comp = lab == c
        if not _key_is_button(comp, plate, ruler, gray):
            continue
        keys.append(comp)
        cut_mask[comp] = 1
    if not keys:  # nothing cut: the head answered with its own outline
        return mask, mask, []
    # The provisional chords also cut rejected tips. Restore those in the
    # plate used by the post search, as well as in the final silhouette.
    plate[(rest > 0) & (cut_mask == 0)] = 255
    solid = mask.copy()
    solid[cut_mask > 0] = 0  # only the keys go; the head stays joined to the neck it sits on
    # the cut leaves a bay between two neighbouring keys; close it, or the head is drawn as a comb with shallow
    # notches. The closing runs only next to the cuts — elsewhere the head's own concave shapes must survive.
    kc = _ellipse(KEY_GAP_MM * d)
    closed = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kc)
    near = cv2.dilate(cut_mask.astype(np.uint8), kc) > 0
    solid[near & (closed > 0)] = 255
    return solid, plate, keys


def _poly_mm(m: np.ndarray, ruler, tol: float) -> np.ndarray | None:
    """A small closed shape (px mask) as an outline in drawing units."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    if len(c) < 8:
        return None
    c = contour.smooth(np.vstack([c, c[:1]]), 1.0)
    c = cv2.approxPolyDP(ruler.to_mm(c).astype(np.float32).reshape(-1, 1, 2), tol, True)[:, 0, :].astype(float)
    return c if len(c) >= 4 else None


def board_sides(t: "Trace") -> list[np.ndarray]:
    """The fretboard's long edges that are separate from the silhouette. On an electric the board is flush with the
    neck, so those sides would draw the same line twice, a hair inside the neck's own edge; the board's ends are
    drawn regardless (the nut is a real bar, the body end is where the board stops)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(contour.resample(np.vstack([t.outline, t.outline[:1]]), 1.0))
    out = []
    for a, b in ((t.board[0], t.board[1]), (t.board[2], t.board[3])):
        seg = contour.resample(np.array([a, b], float), 1.0)
        if float(np.median(tree.query(seg)[0])) >= BOARD_SIDE_GAP_MM:
            out.append(np.array([a, b], float))
    return out


# ---------- manual edits (the web service) ----------
CLICK_CROP_MM = 200  # SAM sees this square round a click: a part fills it well, a knob still gets enough pixels


def sam_at(t: Trace, x_px: float, y_px: float) -> list[tuple[np.ndarray, float]]:
    """SAM's masks for one click (canvas px), small to large, as (canvas-size boolean mask, score)."""
    from . import sam

    h, w = t.mask.shape
    r = int(CLICK_CROP_MM * t.ruler.d / 2)
    x0, y0, x1, y1 = max(0, int(x_px) - r), max(0, int(y_px) - r), min(w, int(x_px) + r), min(h, int(y_px) + r)
    sess = sam.Session(t.photo[y0:y1, x0:x1])
    out = []
    for m, sc in sess.together(np.array([[x_px - x0, y_px - y0]])):
        if sc < 0.3:
            continue
        big = np.zeros((h, w), bool)
        big[y0:y1, x0:x1] = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        if big.sum() > 20:
            out.append((big, float(sc)))
    return sorted(out, key=lambda c: c[0].sum())


def part_from_mask(t: Trace, m: np.ndarray) -> np.ndarray | None:
    """A boolean canvas mask as a closed outline in mm (largest piece), within the guitar."""
    m = (m & (t.mask > 0)).astype(np.uint8)
    k = int(1.5 * t.ruler.d) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    if len(c) < 8:
        return None
    c = contour.smooth(np.vstack([c, c[:1]]), 2.0)
    c = cv2.approxPolyDP(t.ruler.to_mm(c).astype(np.float32).reshape(-1, 1, 2), 0.15, True)[:, 0, :].astype(float)
    return c if len(c) >= 4 else None


def cut_silhouette(t: Trace, m: np.ndarray) -> None:
    """Take a region (a stand, glare outside the body) out of the silhouette; keep the largest piece."""
    full = ((t.mask > 0) & ~m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(full)
    if n < 2:
        return
    t.mask = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    t.outline = outline_of(t.mask, t.ruler)

    def inside(c: np.ndarray) -> bool:  # most of it still on the guitar
        r = np.zeros_like(t.mask)
        cv2.fillPoly(r, [np.round(t.ruler.to_px(c)).astype(np.int32)], 1)
        return (r & (t.mask > 0)).sum() >= 0.5 * max(r.sum(), 1)

    t.parts = [c for c in t.parts if inside(c)]
    on = lambda x, y: t.mask[int(np.clip(y, 0, t.mask.shape[0] - 1)), int(np.clip(x, 0, t.mask.shape[1] - 1))] > 0
    t.posts = [q for q in t.posts if on(*t.ruler.to_px(np.array([q[:2]]))[0])]


def remove_at(t: Trace, x_mm: float, y_mm: float) -> bool:
    """Delete the smallest drawn item under a point: a part or a tuner post."""
    hits = []
    for kind, items in (("parts", t.parts),):
        for i, c in enumerate(items):
            if cv2.pointPolygonTest(c.astype(np.float32).reshape(-1, 1, 2), (float(x_mm), float(y_mm)), False) >= 0:
                hits.append((cv2.contourArea(c.astype(np.float32)), kind, i))
    for i, (x, y, r) in enumerate(t.posts):
        if np.hypot(x - x_mm, y - y_mm) <= r + 2:
            hits.append((np.pi * r * r, "posts", i))
    if not hits:
        return False
    _, kind, i = min(hits)
    getattr(t, kind).pop(i)
    return True


def fret_segments(t: Trace):
    """Actual endpoints when both photo edges passed, otherwise exact legacy endpoints."""
    if t.fret_lines is not None:
        return t.fret_lines
    return [np.array([[xa, -h], [xb, h]]) for xa, xb, h in t.frets]


def photo_board(t: Trace) -> None:
    """Atomic replacement; old nut/end supporting lines and fret count stay unchanged."""
    old = t.ruler.to_px(t.board)
    result = fretboard.detect(t.photo, old)
    t.board_detection = result
    if not result.accepted:
        return
    try:
        nut = fretboard.intersect(old[[0, 3]], result.edges)
        end = fretboard.intersect(old[[1, 2]], result.edges)
        segments = [fretboard.intersect(t.ruler.to_px(q), result.edges) for q in fret_segments(t)]
        polygon = np.array([nut[0], end[0], end[1], nut[1]])
        all_points = np.vstack([polygon, *segments])
        if not np.isfinite(all_points).all():
            raise ValueError("nonfinite intersections")
        # Reject extrapolated intersections outside the image or too far from the
        # prior. Fan frets may move in X, but may not escape their board's ends.
        h, w = t.mask.shape
        if ((all_points < 0).any() or (all_points[:, 0] >= w).any()
                or (all_points[:, 1] >= h).any()):
            raise ValueError("intersections outside photo")
        if np.max(np.linalg.norm(polygon - old, axis=1)) > .3 * np.mean(old[[3, 2], 1] - old[[0, 1], 1]):
            raise ValueError("incompatible nut/end intersections")
        # Each fret endpoint must be on/inside the new board. No changes at all
        # until the full polygon AND every transverse line have been validated.
        poly = polygon.astype(np.float32)
        if any(cv2.pointPolygonTest(poly, tuple(map(float, p)), True) < -1.0 for p in np.vstack(segments)):
            raise ValueError("fret intersections outside board")
        new_board = t.ruler.to_mm(polygon)
        new_lines = [t.ruler.to_mm(q) for q in segments]
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.accepted = False
        result.confidence = 0.0
        result.reason = f"incompatible geometry: {exc}"
        return
    t.board, t.fret_lines = new_board, new_lines


def trace(rgba: np.ndarray, scale_length: float = 648, *, sam_debug=None) -> Trace:
    mask = silhouette(rgba)
    fr, m, sp = orient_and_split(mask, open_body=False)
    photo = fr.warp(np.ascontiguousarray(rgba[..., :3]))
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    full = (fr.warp(mask, cv2.INTER_LINEAR) >= 128).astype(np.uint8) * 255
    ff = frets.fit(gray.astype(np.float32), m, sp)
    focal = FOCAL * max(rgba.shape[:2])
    det = None
    try:  # the photo's own lines first: a clean run of them beats a sequence that is then dragged onto the photo
        det = frets.detect(gray.astype(np.float32), sp, mask=full)
    except ValueError:
        det = None
    if det is not None and len(det[1]) >= 18:
        nut_ends, fret_ends, seq = det
        sp = split(m, nut_x=float((nut_ends[0][0] + nut_ends[1][0]) / 2))
        ruler = frets.Ruler.linear(sp, seq.a / scale_length)
        px_lines = [{"ends": nut_ends}] + [{"ends": q} for q in fret_ends]
        ff = None
    elif ff is not None:
        sp = split(m, nut_x=ff.b)
        ruler = frets.Ruler.from_frets(frets.to_mm(ff, scale_length), sp, focal)
    else:
        # no sequence fit: a heavy fan on a wide board smears the row median. Find the lines on their own.
        nut_ends, fret_ends, seq = det if det is not None else frets.detect(gray.astype(np.float32), sp, mask=full)
        sp = split(m, nut_x=float((nut_ends[0][0] + nut_ends[1][0]) / 2))
        ruler = frets.Ruler.linear(sp, seq.a / scale_length)
        px_lines = [{"ends": nut_ends}] + [{"ends": q} for q in fret_ends]
    outline = outline_of(head_closed(full, sp), ruler)
    solid, plate_px, key_px = head_parts(full, ruler, float(ruler.ff.b), gray)
    if key_px:  # the keys keep their own shapes; the head's outline runs round the plate, not round every key
        outline = outline_of(head_closed(solid, sp), ruler)

    nut_x = ruler.u_butt
    if ff is not None:
        # the fitted sequence decides which fret and roughly where; the photo gives each line its position and angle
        n = _more_frets(gray.astype(np.float32), sp, ff)
        if n < 19:  # the sequence could not see the high frets (a fanned board smears them): take the usual 24
            n = 24
        px_lines = frets.refine(gray.astype(np.float32), sp, ff, n, mask=full)
    na, nb, h0 = _line_mm(px_lines[0]["ends"], sp, ruler)
    lines = [_line_mm(q["ends"], sp, ruler) for q in px_lines[1:]]
    ea, eb, he = lines[-1]  # the board's end sits a little past the last fret, along its direction
    ea, eb = ea - BOARD_PAST_LAST_FRET, eb - BOARD_PAST_LAST_FRET
    end = (ea + eb) / 2
    board = np.array([[na, -h0], [ea, -he], [eb, he], [nb, h0]])

    t = Trace(outline, nut_x, scale_length, lines, board, photo=photo, ruler=ruler, mask=full)
    t.keys = [k for k in (_poly_mm(q, ruler, 0.3) for q in key_px) if k is not None]
    t.posts = _posts(gray, ruler, plate_px, key_px)
    debug_kw = {"debug": sam_debug} if sam_debug is not None else {}
    t.parts = _sam_parts(photo, full, ruler, end, float(np.abs(outline[:, 1]).max()) + 5, he, **debug_kw)
    t.parts = _real_parts(t, t.parts, **debug_kw)
    from . import knobs

    t.parts.extend(knobs.recover(t, end))
    photo_board(t)  # after parts: the side fit must not change SAM prompts/filters
    return t


def masked_photo(t: Trace) -> np.ndarray:
    """Photo pixels only inside the guitar's silhouette, white around."""
    return np.where(t.mask[..., None] > 0, t.photo, 255).astype(np.uint8)


def overlay_svg(t: Trace) -> str:
    """The trace as SVG in photo pixels (viewBox = the photo): drawn over it in the browser, hairline at any zoom."""
    px = lambda pts: t.ruler.to_px(np.asarray(pts, float))
    d = lambda pts: "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + "Z"
    s = f'<path d="{d(px(t.outline))}"/><path class="det" d="{d(px(t.board))}"/>'
    for segment in fret_segments(t):
        (x0, y0), (x1, y1) = px(segment)
        s += f'<path class="det" d="M{x0:.1f} {y0:.1f} L{x1:.1f} {y1:.1f}"/>'
    for x, y, r in t.posts:
        c = px([[x, y]])[0]
        s += f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r * t.ruler.d:.1f}"/>'
    s += "".join(f'<path d="{d(px(c))}"/>' for c in (*t.keys, *t.parts))
    h, w = t.mask.shape[:2]
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="none">{s}</svg>'


def overlay(t: Trace) -> np.ndarray:
    """The trace in red over the photo (photo pixels only inside the guitar's silhouette)."""
    img = masked_photo(t)
    px = lambda pts: np.round(t.ruler.to_px(np.asarray(pts, float))).astype(np.int32)
    red = (230, 30, 30)
    cv2.polylines(img, [px(t.outline)], True, red, 2)
    cv2.polylines(img, [px(t.board)], True, red, 1)
    for segment in fret_segments(t):
        cv2.line(img, *map(tuple, px(segment)), red, 1)
    for x, y, r in t.posts:
        c = px([[x, y]])[0]
        cv2.circle(img, (int(c[0]), int(c[1])), int(r * t.ruler.d), red, 2)
    cv2.polylines(img, [px(c) for c in (*t.keys, *t.parts)], True, red, 2)
    return img


# ---------- SVG ----------
def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _st(w: float) -> str:
    return f'fill="none" stroke="currentColor" stroke-width="{w}"'


def _path(pts: np.ndarray, closed=True) -> str:
    d = "M" + " L".join(f"{_f(x)} {_f(y)}" for x, y in pts)
    return d + ("Z" if closed else "")


def smoothed(pts: np.ndarray, sigma_mm: float) -> np.ndarray:
    """A closed outline with bumps smaller than ~sigma ironed out (Gaussian along it, wrapping round), simplified."""
    if sigma_mm <= 0 or len(pts) < 8:
        return pts
    step = 0.5
    r = contour.resample(np.vstack([pts, pts[:1]]), step)[:-1]
    if len(r) < 8:
        return pts
    from scipy.ndimage import gaussian_filter1d

    r = gaussian_filter1d(r, sigma_mm / step, axis=0, mode="wrap")
    return _simplify(r, OUTLINE_TOL_MM)


def bounded_handles(nodes: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """Limit a shared tangent to a third of EACH adjacent chord.

    Uniform Catmull-Rom otherwise lends a short edge a handle proportional to
    its long neighbour. Shorten BOTH incident handles by the same factor:
    vertices, tangent direction and C1 continuity are preserved. Ordinary
    equally spaced runs are unchanged. This is not a photo-error guarantee.
    """
    lengths = np.linalg.norm(np.roll(nodes, -1, axis=0) - nodes, axis=1)
    limit = np.minimum(lengths, np.roll(lengths, 1)) / 3
    size = np.linalg.norm(tangents, axis=1)
    return tangents * np.minimum(1, limit / np.maximum(size, 1e-12))[:, None]


def curve_path(pts: np.ndarray, tol: float, *, neck_join: np.ndarray | None = None, bounded: bool = False) -> str:
    """A closed outline as one run of cubic Béziers through the Douglas-Peucker nodes (Catmull-Rom tangents).

    A polyline of 400–600 points draws the same shape, but a renderer keeps every corner between them; the curves
    draw the shape once and stay smooth at any zoom. The nodes themselves lie on the traced outline, so a horn tip
    or a key's edge keeps its place.
    """
    c = contour.dp_closed(pts, tol)
    n = len(c)
    if n < 4:
        return _path(pts)
    tangents = (np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)) / 6
    if bounded:
        tangents = bounded_handles(c, tangents)
    if neck_join is not None:
        # Only the body-side end of a long, straight neck edge. Uniform Catmull-Rom
        # gives its short neighbour a handle ~neck_length/6, which turns backwards
        # into a hook. Limit BOTH handles at this node, retaining the tangent and
        # every contour vertex; body/head/parts away from this join stay untouched.
        xmin, xmax = float(neck_join[:, 0].min()), float(neck_join[:, 0].max())
        xmax = xmin + .25 * (xmax - xmin)  # body end only, never mid-neck or nut/head
        half = float(np.abs(neck_join[:, 1]).max()) + 8
        for j, p in enumerate(c):
            if not (xmin <= p[0] <= xmax and abs(p[1]) <= half):
                continue
            neighbours = [c[(j-1) % n] - p, c[(j+1) % n] - p]
            lengths = [np.linalg.norm(v) for v in neighbours]
            k = int(np.argmax(lengths))
            v, long, short = neighbours[k], lengths[k], lengths[1-k]
            if v[0] > 40 and abs(v[1]) < .1*v[0] and long > 4*short:
                size = np.linalg.norm(tangents[j])
                if size > short/3:
                    tangents[j] *= (short/3)/size
    d = f"M{_f(c[0][0])} {_f(c[0][1])}"
    for i in range(n):
        p1, p2 = c[i], c[(i + 1) % n]
        c1, c2 = p1 + tangents[i], p2 - tangents[(i+1) % n]
        d += f"C{_f(c1[0])} {_f(c1[1])} {_f(c2[0])} {_f(c2[1])} {_f(p2[0])} {_f(p2[1])}"
    return d + "Z"


def svg(t: Trace, style: str = "comic", smooth_mm: float = DEFAULT_SMOOTH_MM, *, sam_debug=None, editor=False) -> str:
    """smooth_mm: how much the outline and the parts are ironed out (0 = as traced)."""
    from .knobs import circle_geometry

    W = STYLES[style]
    st = lambda k: _st(W[k]) + ' stroke-linejoin="round" stroke-linecap="round"'
    counts = {}
    def obj(category, content):
        if not editor:
            return content
        index = counts.get(category, 0)
        counts[category] = index + 1
        return f'<g id="{category}-{index}" data-category="{category}">{content}</g>'

    s = obj("silhouette", f'<path d="{curve_path(smoothed(t.outline, smooth_mm), CURVE_TOL_MM, neck_join=t.board)}" {st("main")}/>')
    for side in board_sides(t):  # the board's long edges only where they are not the neck's own edge
        s += obj("board", f'<path d="{_path(side, closed=False)}" {st("det")}/>')
    s += obj("board", f'<path d="{_path(t.board[1:3], closed=False)}" {st("det")}/>')  # where the board stops on the body
    (na, ya), (nb, yb) = t.board[0], t.board[3]  # the nut
    s += obj("nut", f'<path d="M{_f(na)} {_f(ya)} L{_f(nb)} {_f(yb)}" {st("main")}/>')
    for (xa, ya), (xb, yb) in fret_segments(t):
        s += obj("fret", f'<path d="M{_f(xa)} {_f(ya)} L{_f(xb)} {_f(yb)}" {st("thin")}/>')
    for x, y, r in t.posts:
        s += obj("post", f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" {st("det")}/><circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r * 0.5)}" {st("thin")}/>')
    for c in t.keys:
        s += obj("key", f'<path d="{curve_path(smoothed(c, smooth_mm), PART_CURVE_TOL_MM)}" {st("det")}/>')
    controls = {id(q.contour): circle_geometry(q) for q in t.knobs
                if any(c is q.contour for c in t.parts)}
    inner_ids = {id(q.inner) for q in t.knobs if len(controls.get(id(q.contour), [])) == 2}
    for c in t.parts:
        if id(c) in inner_ids:
            continue  # the confirmed inner contour is already drawn concentrically
        circles = controls.get(id(c), [])
        if circles:
            content = ""
            for i, (x, y, r) in enumerate(circles):
                content += f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" {st("det" if i == 0 else "thin")}/>'
            s += obj("control", content)
            continue
        smoothed_part = smoothed(c, smooth_mm * PART_SMOOTH_SHARE)
        path = curve_path(smoothed_part, PART_CURVE_TOL_MM, bounded=True)
        s += obj("hardware", f'<path d="{path}" {st("det")}/>')
        if sam_debug is not None and id(c) in sam_debug.part_ids:
            sam_debug.export(c, smoothed_part, path, W["det"])
    (x0, y0), (x1, y1) = t.outline.min(0) - 10, t.outline.max(0) + 10
    vb = f"{_f(x0)} {_f(y0)} {_f(x1 - x0)} {_f(y1 - y0)}"
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" style="color:#1b1b1b;background:#fff">{s}</svg>\n'
