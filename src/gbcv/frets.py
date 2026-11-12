"""Frets as a built-in ruler: pixel scale and camera tilt along the guitar axis.

Fret n sits at U_n = S * (1 - 2^(-n/12)) mm from the nut (S = scale length). A plane tilted about an axis
across the guitar projects U to canvas x as a 1D homography:  x = (b - a*U) / (1 + c*U).
We find frets on a straightened strip of the fretboard, match them to n = 1.., and fit (a, b, c).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d

from .split import Split

ROWS = 41  # samples across the fretboard
MAX_FRETS = 24
BOARD_SHARE = 0.75  # of the neck width the strip spans
SLOPE_JUMP_DEG = 10  # a line this far off its neighbours is a false ridge, not that fret


@dataclass
class FretFit:
    a: float  # px per mm at the nut
    b: float  # canvas x of the nut
    c: float  # perspective, 1/mm
    found: list[tuple[int, float]]  # (fret number, canvas x) of matched frets
    residual: float  # rms px of matched frets vs the model
    frets: int  # last fret number seen on the board

    def u(self, x):
        """Canvas x → distance from the nut towards the bridge, mm."""
        x = np.asarray(x, float)
        return (self.b - x) / (self.a + self.c * x)

    def x(self, u):
        u = np.asarray(u, float)
        return (self.b - self.a * u) / (1 + self.c * u)


def board_strip(gray: np.ndarray, sp: Split, x0: int, x1: int, band=(-1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Straightened fretboard strip (ROWS × len) between canvas x0..x1, 75 % of the neck width.
    band: the rows' span across, -1..1 of that (one edge only: fanned frets)."""
    xs = np.arange(x0, x1, dtype=np.float32)
    hw = sp.neck_width(xs) / 2 * BOARD_SHARE
    rows = np.linspace(*band, ROWS, dtype=np.float32)[:, None]
    mx = np.repeat(xs[None], ROWS, 0)
    my = (sp.neck_centre(xs)[None] + rows * hw[None]).astype(np.float32)
    return cv2.remap(gray, mx, my, cv2.INTER_LINEAR), xs


def response(strip: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-column scores across the whole board: (thin bright line = fret, any edge = nut / board end).

    Fret: bright ridge (-d2/dx2) minus edge strength, so the sides of block inlays (steps) don't count.
    Median over rows so dots, strings and partial inlays don't count either.
    """
    s = gaussian_filter(strip.astype(np.float32), (1.0, sigma))
    gx = np.gradient(s, axis=1)
    gxx = np.gradient(gx, axis=1)
    ridge = np.median(np.clip(-gxx * sigma**2 - 0.5 * np.abs(gx) * sigma, 0, None), axis=0)
    edge = np.median(np.abs(gx) * sigma, axis=0)
    ridge /= np.percentile(ridge, 99) + 1e-6
    return ridge, edge / (np.percentile(edge, 99) + 1e-6)


def fit(gray: np.ndarray, mask: np.ndarray, sp: Split, band=(-1.0, 1.0)) -> FretFit | None:
    neck_px = sp.nut_x - sp.neck_x0
    x0 = max(sp.butt_x, int(sp.nut_x - 1.15 * neck_px))  # the board runs past the horns
    x1 = int(sp.nut_x + 0.15 * neck_px)  # a bit of the head: needed to check the nut
    strip, xs = board_strip(gray, sp, x0, x1, band)
    resp, edge = response(strip, sigma=max(1.0, 0.012 * sp.neck_width(sp.nut_x)))
    n = np.arange(1, MAX_FRETS + 1)
    g = 1 - 2.0 ** (-n / 12)

    # coarse search without perspective: scale in px (K) and nut position (b)
    length = sp.tip_x - sp.butt_x
    tol = max(1, int(0.0025 * length))
    rmax = np.minimum(maximum_filter1d(resp, 2 * tol + 1), 1.5)
    bs = np.arange(sp.nut_x - 0.04 * neck_px, sp.nut_x + 0.04 * neck_px, 1.0)
    best = (-1.0, 0.0, 0.0)
    for K in np.linspace(0.50 * length, 0.75 * length, 400):
        idx = np.round(bs[:, None] - K * g[None, :16] - xs[0]).astype(int)
        ok = (idx >= 0) & (idx < len(rmax))
        v = np.where(ok, rmax[np.clip(idx, 0, len(rmax) - 1)], 0).sum(1)
        i = int(np.argmax(v))
        if v[i] > best[0]:
            best = (float(v[i]), K, bs[i])
    _, K, b = best

    # The fret sequence is self-similar: shifting the anchor by one fret (K scaled by 2^(±1/12)) fits the other
    # frets just as well, and a bone nut looks like a fret. Break the tie with the mask: the head is wider
    # than the neck taper, the board between nut and fret 1 is not.
    width = (mask > 0).sum(0).astype(float)
    is_head = lambda xa, xb: np.median(width[int(min(xa, xb)) : int(max(xa, xb)) + 1]) > sp.neck_width((xa + xb) / 2) * 1.12 + 2
    nut_line = np.maximum(maximum_filter1d(resp, 2 * tol + 1), maximum_filter1d(edge, 2 * tol + 1))
    r = 2 ** (1 / 12)
    for _ in range(3):
        x1_ = b - K * (1 - 1 / r)  # fret 1
        xm1 = b + K * (r - 1)  # "fret -1"
        km1 = int(round(xm1 - xs[0]))
        if is_head(b - 0.15 * (b - x1_), b):  # just before the anchor it's already head: fret 1 is the nut
            b, K = x1_, K / r
        elif 0 <= km1 < len(nut_line) and nut_line[km1] > 0.5 and not is_head(xm1 - 0.15 * (xm1 - b), xm1):  # board goes on: nut is further
            b, K = xm1, K * r
        else:
            break

    # refine with perspective: match peaks near predictions, least squares on x(1 + cU) = b - aU
    a, c = K, 0.0  # in 'scale units' (U = g): a = px per scale length
    found = []
    for _ in range(3):
        pred = (b - a * g) / (1 + c * g)
        spacing = np.abs(np.gradient(pred))
        found = []
        for ni, p, sp_ in zip(n, pred, spacing):
            lo, hi = int(p - 0.25 * sp_ - xs[0]), int(p + 0.25 * sp_ - xs[0]) + 1
            if lo < 0 or hi > len(resp):
                continue
            k = lo + int(np.argmax(resp[lo:hi]))
            if resp[k] > 0.35:
                found.append((int(ni), float(xs[k])))
        if len(found) < 6:
            return None
        ff = from_points(found)
        b, a, c = ff.b, ff.a, ff.c
    return ff


def from_points(found: list[tuple[int, float]], anchor: float | None = None) -> FretFit:
    """Least squares on x(1 + cU) = b - aU through (fret number, canvas x) pairs, U in scale units.

    `anchor`: the nut's canvas x, when it is known from the split. Pinning it leaves only scale and perspective to
    the frets — with three free parameters on a short fuzzy run they trade against each other and the whole line
    lands beside the real ones.
    """
    U = 1 - 2.0 ** (-np.array([f for f, _ in found], float) / 12)
    X = np.array([x for _, x in found])
    if anchor is None:
        A = np.c_[np.ones_like(U), -U, -U * X]  # b, a, c
        (b, a, c), *_ = np.linalg.lstsq(A, X, rcond=None)
    else:
        A = np.c_[U, U * X]  # a, c with b = anchor
        (a, c), *_ = np.linalg.lstsq(A, anchor - X, rcond=None)
        b = float(anchor)
    pred = (b - a * U) / (1 + c * U)
    residual = float(np.sqrt(np.mean((pred - X) ** 2)))
    return FretFit(a=a, b=b, c=c, found=found, residual=residual, frets=max(f for f, _ in found))


def to_mm(ff: FretFit, scale_mm: float) -> FretFit:
    """Fit was done with U in scale units; convert to mm for a chosen scale length."""
    return FretFit(a=ff.a / scale_mm, b=ff.b, c=ff.c / scale_mm, found=ff.found, residual=ff.residual, frets=ff.frets)


def ridge(strip: np.ndarray) -> np.ndarray:
    """Score of 'thin line across the board', 0..~1: a top-hat/black-hat along x picks a line narrower than the
    kernel, brighter OR darker than the wood (dark frets on maple are common). A step — the board's end, a
    pickup edge, a pickguard — is wider than the kernel and scores ~0."""
    s = gaussian_filter(strip.astype(np.float32), (1.0, 1.0))
    u8 = np.clip(s, 0, 255).astype(np.uint8)
    w = max(5, int(round(0.004 * strip.shape[1])) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 1))
    r = np.maximum(cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, se), cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, se))
    return r.astype(np.float32) / (np.percentile(r, 99.5) + 1e-6)


def _best_line(score: np.ndarray, p: float, half: int, slopes=np.arange(-1.2, 1.21, 0.15)):
    """The line through the ridge nearest column p: its slope and offset at the middle row.

    A sum along the line over all rows, tried over a few slopes, picks the offset; then the line is re-fitted
    to the row-wise ridge maxima, which is what gives a slanted fret its true angle. Returns
    (slope, column at mid row, contrast)."""
    rows, w = score.shape
    mid = (rows - 1) / 2
    offs = np.arange(-half, half + 1)
    best = (-1.0, 0.0, p, 1.0)
    rr = np.arange(rows)
    for k in slopes:
        base = np.round(p + k * (rr - mid)).astype(int)
        acc = np.zeros(offs.size, np.float32)
        for i, s in enumerate(offs):
            idx = base + s
            ok = (idx >= 0) & (idx < w)
            acc[i] = score[rr[ok], idx[ok]].sum()
        med = float(np.median(acc)) + 1e-6
        i = int(acc.argmax())
        if acc[i] > best[0]:
            best = (float(acc[i]), float(k), float(p + offs[i]), float(acc[i] / med))
    k, off, contrast = best[1], best[2], best[3]
    # follow the ridge row by row: the sum can prefer a flatter shared line, the maxima do not
    for _ in range(2):
        xs = np.full(rows, np.nan)
        for r in range(rows):
            c = off + k * (r - mid)
            lo, hi = int(c - 0.5 * half), int(c + 0.5 * half) + 1
            if lo < 0 or hi > w:
                continue
            seg = score[r, lo:hi]
            j = int(seg.argmax())
            if seg[j] > 0.05:
                xs[r] = lo + j
        ok = ~np.isnan(xs)
        if ok.sum() < 0.5 * rows:
            break
        k2, b2 = np.polyfit(rr[ok], xs[ok], 1)
        if abs(k2) > 2.0:
            break
        k, off = float(k2), float(b2 + k2 * mid)
    return k, off, contrast



RATIO = 2 ** (1 / 12)  # each fret towards the body shortens the remaining string by this much


def _project(score: np.ndarray, k: float) -> np.ndarray:
    """Column profile of the ridge along lines of slope k: a slanted fret peaks sharply at its own k."""
    rows, w = score.shape
    acc = np.zeros(w, np.float32)
    mid = (rows - 1) / 2
    for r in range(rows):
        s = int(round(k * (r - mid)))
        if s >= 0:
            acc[: w - s] += score[r, s:]
        else:
            acc[-s:] += score[r, : w + s]
    return acc / rows


def _peaks(score: np.ndarray) -> list[tuple[float, int, float]]:
    """Candidate lines without any model: sharp peaks of the ridge projected along a slope, over a few slopes.
    A fan gives each fret its own slope, so no single projection sees them all."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    rows, w = score.shape
    found = []
    for k in np.arange(-1.0, 1.01, 0.1):
        p = gaussian_filter1d(_project(score, k), max(2.0, 0.004 * w))
        if p.max() <= 0:
            continue
        idx, _ = find_peaks(p, distance=max(4, int(0.006 * w)), height=0.15 * p.max(), prominence=0.1 * p.max())
        for x in idx:
            found.append((float(k), int(x), float(p[x])))
    # one physical fret gives a peak at every slope: keep the strongest per x, not per (slope, x)
    found.sort(key=lambda q: -q[2])
    kept: list[tuple[float, int, float]] = []
    tol = max(4, int(0.004 * w))
    for q in found:
        if all(abs(q[1] - k[1]) > tol for k in kept):
            kept.append(q)
    return sorted(kept, key=lambda q: q[1])


def _chain(xs, tol: float = 0.09) -> list[int]:
    """The longest run of candidates whose gaps grow by RATIO per fret towards the head (a missing fret counts
    as two steps). This is the frets' own physics, not a guitar model, so it throws away pickups and knobs."""
    xs = np.asarray(xs, float)
    n = len(xs)
    best: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] - xs[i] < 3:
                continue
            idx, prev = [i, j], xs[j] - xs[i]
            while idx[-1] + 1 < n:
                rest = np.arange(idx[-1] + 1, n)
                gaps = xs[rest] - xs[idx[-1]]
                pick = (np.inf, 0)
                for s in (1, 2):  # one fret may be missed
                    err = np.abs(np.log(gaps / (prev * RATIO**s)))
                    k = int(err.argmin())
                    if err[k] < pick[0]:
                        pick = (float(err[k]), int(rest[k]))
                if pick[0] > tol:
                    break
                prev = xs[pick[1]] - xs[idx[-1]]
                idx.append(pick[1])
            if len(idx) > len(best):
                best = idx
    return best


def _recover_body_fret(score: np.ndarray, rejected: list[float], got: list[dict], nut: float, x0: int) -> None:
    """Retry one real candidate beyond the body end, using refined rather than coarse peak gaps.

    `nut` and offsets are strip coordinates. The end farthest from the nut is bodyward,
    irrespective of the X direction. Existing accepted lines are never moved.
    """
    if len(got) < 3:
        return
    ordered = sorted(got, key=lambda q: q["off"])
    if abs(ordered[0]["off"] - nut) > abs(ordered[-1]["off"] - nut):
        ordered = ordered[:3]
    else:
        ordered = ordered[::-1][:3]
    direction = np.sign(ordered[0]["off"] - ordered[1]["off"])
    outside = [p for p in rejected if direction * (p - ordered[0]["off"]) > 0]
    if not outside:
        return
    p = min(outside, key=lambda p: abs(p - ordered[0]["off"]))
    spacing = abs(ordered[1]["off"] - ordered[0]["off"])
    expected = spacing / RATIO
    # Do not search across a missing interval or retry another peak of the same fret.
    if not 0.5 * expected < abs(p - ordered[0]["off"]) < 1.5 * expected:
        return
    half = max(5, int(0.35 * expected))
    positions = [p] + [q["off"] for q in ordered]
    refined = [_best_line(score, q, half) for q in positions]
    if any(con < 1.5 for _, _, con in refined):
        return
    ks, offsets, _ = np.array(refined).T
    gaps = -direction * np.diff(offsets)
    if (gaps <= 0).any() or np.any(np.abs(np.log(gaps[1:] / gaps[:-1] / RATIO)) > 0.09):
        return
    kept_gaps = -direction * np.diff([offsets[0]] + [q["off"] for q in ordered])
    if (kept_gaps <= 0).any() or np.any(np.abs(np.log(kept_gaps[1:] / kept_gaps[:-1] / RATIO)) > 0.09):
        return
    if abs(np.degrees(np.arctan(ks[0])) - np.median(np.degrees(np.arctan(ks[1:])))) > SLOPE_JUMP_DEG:
        return
    # Distinct at every sampled row, not just at the midpoints (slanted duplicates/crossings).
    rr = np.arange(score.shape[0]) - (score.shape[0] - 1) / 2
    line = offsets[0] + ks[0] * rr
    if (line < 0).any() or (line >= score.shape[1]).any():
        return
    if any(np.min(direction * (line - q["off"] - q["k"] * rr)) <= 0.35 * expected for q in got):
        return
    k, off, con = refined[0]
    got.append({"mid": x0 + off, "k": k, "off": off, "contrast": con})
    got.sort(key=lambda q: q["mid"])


def detect(gray: np.ndarray, sp: Split, mask: np.ndarray | None = None) -> tuple[tuple, list, FretFit]:
    """Fret lines with no sequence fit at all — for a board the fit cannot read (a heavy fan on a wide board
    smears the row median). Candidates over a few slopes, then the run whose gaps follow the frets' ratio.
    Returns (nut line, fret lines, a sequence fitted to them for the drawing's units)."""
    neck_px = sp.nut_x - sp.neck_x0
    x0 = sp.butt_x
    strip, _ = board_strip(gray.astype(np.float32), sp, x0, int(sp.nut_x + 0.15 * neck_px))
    score = ridge(strip)
    rows = strip.shape[0]
    cand = _peaks(score)
    xs = [q[1] for q in cand]
    idx = _chain(xs)
    if len(idx) < 5:
        raise ValueError("frets not found")
    mids = [xs[i] for i in idx]
    got = []
    for j, p in enumerate(mids):
        near = [abs(p - mids[i]) for i in (j - 1, j + 1) if 0 <= i < len(mids)]
        half = max(5, int(0.35 * min(near))) if near else 8
        k, off, con = _best_line(score, p, half)
        if con < 1.5:
            continue
        got.append({"mid": x0 + off, "k": k, "off": off, "contrast": con})
    got.sort(key=lambda q: q["mid"])
    _recover_body_fret(score, [xs[i] for i in range(len(xs)) if i not in idx], got, sp.nut_x - x0, x0)
    chain = got
    if len(chain) < 4:
        raise ValueError("frets not found")
    for q in chain:
        ends = []
        for r in (0, rows - 1):
            x = x0 + q["off"] + q["k"] * (r - (rows - 1) / 2)
            t = -1.0 if r == 0 else 1.0
            ends.append((float(x), float(sp.neck_centre(x) + t * sp.neck_width(x) / 2 * BOARD_SHARE)))
        q["ends"] = ends
    spacing = float(np.median(np.diff([q["mid"] for q in chain])))
    i = int(np.argmin([abs(q["mid"] - sp.nut_x) for q in chain]))
    if abs(chain[i]["mid"] - sp.nut_x) <= 0.5 * spacing:
        nut, frets = chain[i], chain[:i]
    else:
        k = min(chain, key=lambda q: abs(q["mid"] - sp.nut_x))["k"]
        ends = []
        for r in (0, rows - 1):
            x = sp.nut_x + k * (r - (rows - 1) / 2)
            t = -1.0 if r == 0 else 1.0
            ends.append((float(x), float(sp.neck_centre(x) + t * sp.neck_width(x) / 2 * BOARD_SHARE)))
        nut = {"ends": tuple(ends)}
        frets = [q for q in chain if q["mid"] < sp.nut_x]
    frets = frets[::-1]
    ff = from_points([(m, q["mid"]) for m, q in enumerate(frets, start=1)])
    return nut["ends"], [q["ends"] for q in frets], ff


def _ends(sp: Split, x0: int, off: float, k: float, rows: int) -> list[tuple[float, float]]:
    """The two ends of a line given as (offset at the strip's middle row, slope) in canvas px."""
    out = []
    for r in (0, rows - 1):
        x = x0 + off + k * (r - (rows - 1) / 2)
        t = -1.0 if r == 0 else 1.0
        out.append((float(x), float(sp.neck_centre(x) + t * sp.neck_width(x) / 2 * BOARD_SHARE)))
    return out


def _settle_slopes(got: list[dict], sp: Split, x0: int, rows: int, jump_deg: float = SLOPE_JUMP_DEG) -> None:
    """Give every line the slope of its neighbours when it has none or when it jumps away from them.

    On a board the slope changes gradually from fret to fret (a fan is a smooth progression), so a line standing
    10° off its neighbours is a false ridge — the nut's side face, a pickup's edge — not that fret. Its own x stays:
    the photo was right about where the line is, wrong about what it saw.
    """
    if not got:
        return
    for _ in range(2):  # twice: a bad line must not pull its neighbours off in the same pass
        deg = np.array([np.degrees(np.arctan(q["k"])) for q in got])
        blind = [q["contrast"] <= 0 for q in got]
        fixed = list(deg)
        for i, q in enumerate(got):
            near = [j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(got) and not blind[j]]
            if not near:
                continue
            m = float(np.median([deg[j] for j in near]))
            if blind[i] or abs(deg[i] - m) > jump_deg:
                fixed[i] = m
        if all(a == b for a, b in zip(fixed, deg)):
            break
        for q, d in zip(got, fixed):
            q["k"] = float(np.tan(np.radians(d)))
            q["ends"] = _ends(sp, x0, q["off"], q["k"], rows)


def align(score: np.ndarray, xs: np.ndarray, ff: FretFit, last: int, anchor: float | None = None) -> FretFit:
    """Move the whole fret sequence onto the photo's ridges (scale and perspective, or the anchor too).

    Each line searched on its own picks whatever ridge is strongest near it — an inlay, a string, the board's
    binding — and the run comes out with gaps that jump. The frets are one sequence, so the sequence is what gets
    matched: maximise the mean ridge under all `last` predicted lines, with a penalty for uneven strength.
    With `anchor` (the nut from the split) the anchor stays put: it is known, and free it trades against the scale.
    """
    from scipy.optimize import minimize

    g = 1 - 2 ** (-np.arange(0, last + 1) / 12)
    x0 = float(xs[0])
    w = score.shape[1]

    def loss(p):
        a, c = p[0], p[1]
        b = p[2] if anchor is None else anchor
        if a <= 0 or abs(c) > 1.5:
            return 1e6
        x = (b - a * g) / (1 + c * g)
        col = np.round(x - x0).astype(int)
        ok = (col >= 0) & (col < w)
        if ok.sum() < 0.9 * len(g):
            return 1e6
        v = np.array([score[:, j].max() for j in col[ok]])
        return -float(v.mean() - 0.3 * v.std())

    p0 = [ff.a, ff.c] if anchor is not None else [ff.a, ff.c, ff.b]
    base = loss(p0)
    res = minimize(loss, p0, method="Nelder-Mead", options={"xatol": 0.2, "fatol": 1e-5, "maxiter": 300})
    a, c = res.x[0], res.x[1]
    b = float(anchor) if anchor is not None else float(res.x[2])
    if res.fun >= base:
        a, b, c = ff.a, ff.b, ff.c
    return FretFit(a=float(a), b=float(b), c=float(c), found=ff.found, residual=ff.residual, frets=last)


def _search(score, ff, x0: int, strip, sp: Split, last: int, inside=None, share: float = 0.45):
    """Lines for fret indices 0..last: each snapped to the photo where the photo shows it, otherwise put at the
    sequence's own position. `last` decides the count; it comes from the sequence fit, which is reliable for a
    straight board, and is capped by the caller for a fanned one. `share` is how far a line may slide off the
    sequence's place, in parts of the fret's own spacing (small: slope only)."""
    g = lambda m: 1 - 2 ** (-m / 12)
    rows = strip.shape[0]
    got = []
    for m in range(0, last + 1):
        p = float(ff.x(g(m))) - x0
        step = np.mean([abs(ff.x(g(max(0, m - 1))) - ff.x(g(m))), abs(ff.x(g(m + 1)) - ff.x(g(m)))])
        half = max(1, int(share * step))
        k, off, contrast = _best_line(score, p, half)
        if inside is not None:  # the line has to lie on the guitar, not on the background beside it
            col = (off + k * (np.arange(rows) - (rows - 1) / 2)).astype(int)
            ok = (col >= 0) & (col < inside.shape[1])
            on = inside[np.arange(rows)[ok], col[ok]].mean() if ok.any() else 0.0
            if on < 0.9:
                contrast = 0.0
        if contrast < 1.5:
            k, off = 0.0, p  # nothing on the photo this close: the sequence's position, square to the neck
            contrast = 0.0
        got.append({"m": m, "ends": _ends(sp, x0, off, k, rows), "contrast": contrast, "mid": x0 + off,
                    "k": k, "off": off})
    _settle_slopes(got, sp, x0, rows)
    return got


def refine(gray: np.ndarray, sp: Split, ff: FretFit, last: int, mask: np.ndarray | None = None) -> list[dict]:
    """Every fret's line in the photo from the nut (index 0) to fret `last`.

    The fitted sequence only says which fret and roughly where; the photo decides the exact position and the
    angle, so a fanned board needs no special case: each line is searched over a few slopes inside that fret's
    own spacing. What the photo shows (the strong lines) re-fits the sequence once, because a fan distorts the
    first fit — it trades scale against perspective — and the better sequence puts the lines where they belong."""
    neck_px = sp.nut_x - sp.neck_x0
    x0 = sp.butt_x  # the board runs over the body: the last frets are far left of the neck pocket
    x1 = int(sp.nut_x + 0.15 * neck_px)
    strip, xs = board_strip(gray.astype(np.float32), sp, x0, x1)
    inside = board_strip((mask > 0).astype(np.float32), sp, x0, x1)[0] if mask is not None else None
    score = ridge(strip)
    got = _search(score, ff, x0, strip, sp, last, inside)
    strong = [(q["m"], q["mid"]) for q in got if q["m"] >= 1 and q["contrast"] > 0]
    if len(strong) >= 5:
        ff = from_points(strong, anchor=float(sp.nut_x))
        got = _search(score, ff, x0, strip, sp, last, inside)
    # the run as a whole goes onto the photo's ridges, then each line only takes its slope: a line may not jump
    # onto a neighbour's ridge or an inlay, which is what made the gaps 48, 17, 26, 41 px instead of a smooth run
    ff = align(score, xs, ff, last, anchor=float(sp.nut_x))
    got = _search(score, ff, x0, strip, sp, last, inside, share=0.06)
    _keep_order(got, ff, x0, sp, strip.shape[0])
    return got


def _keep_order(got: list[dict], ff: FretFit, x0: int, sp: Split, rows: int) -> None:
    """Frets run one way and their spacing shrinks: 2^(−1/12) per fret.

    A line that snapped onto its neighbour's ridge (two frets on one x) or that overtook the line before it is not
    that fret — the high frets sit close together and their windows overlap. Such a line goes back to the sequence's
    own place, which also keeps the board's end (last fret + a few mm) off the pickups.
    """
    pred = [float(ff.x(1 - 2 ** (-q["m"] / 12))) for q in got]
    for i in range(1, len(got)):
        step = abs(pred[i] - pred[i - 1])
        if step <= 0:
            continue
        if got[i]["mid"] > got[i - 1]["mid"] - 0.45 * step:
            off = pred[i] - x0
            got[i].update(mid=pred[i], off=off, ends=_ends(sp, x0, off, got[i]["k"], rows))


@dataclass
class Ruler:
    """Canvas px ↔ blueprint mm (X from the butt towards the head, Y from the neck centre line).

    Along the axis: the fret homography (or a plain scale). Across: px per mm at the nut (d), shrinking with
    distance like the axis does (1 + c*U). d = a / cos(tilt), tilt from c and an assumed focal length.
    """

    ff: FretFit  # in mm (a: px/mm, c: 1/mm); a plain scale is a FretFit with c = 0
    centre: tuple[float, float]  # neck centre line y = k*x + m, px
    u_butt: float  # U of the butt, mm
    d: float  # px per mm across, at the nut
    tilt_deg: float = 0.0

    @classmethod
    def linear(cls, sp: Split, px_per_mm: float) -> "Ruler":
        ff = FretFit(a=px_per_mm, b=float(sp.nut_x), c=0.0, found=[], residual=0.0, frets=0)
        return cls(ff, sp.centre_fit, float(ff.u(sp.butt_x)), px_per_mm)

    @classmethod
    def from_frets(cls, ff: FretFit, sp: Split, focal_px: float) -> "Ruler":
        tilt = np.arctan(ff.c * focal_px / ff.a)
        return cls(ff, sp.centre_fit, float(ff.u(sp.butt_x)), ff.a / np.cos(tilt), float(np.degrees(tilt)))

    def _yc(self, x):
        return self.centre[0] * x + self.centre[1]

    def to_mm(self, pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 0], pts[:, 1]
        u = self.ff.u(x)
        return np.c_[self.u_butt - u, (y - self._yc(x)) * (1 + self.ff.c * u) / self.d]

    def to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_mm, float)
        u = self.u_butt - p[:, 0]
        x = self.ff.x(u)
        return np.c_[x, self._yc(x) + p[:, 1] * self.d / (1 + self.ff.c * u)]

    def width_mm(self, x: float, width_px: float) -> float:
        return float(width_px * (1 + self.ff.c * self.ff.u(x)) / self.d)
