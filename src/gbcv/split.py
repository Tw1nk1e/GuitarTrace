"""Split the oriented mask into body / neck / head using the width profile along the axis.

Works on the rotated canvas (head to the right) in pixels. The neck is the long narrow run right of the body;
its width taper is fitted with a line, the nut is where the profile leaves that line (head flare).
The outline is then cut into arcs: points inside the neck strip are neck edges, right of the nut is the head.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Split:
    butt_x: int
    tip_x: int
    widest_x: int
    neck_x0: int  # left end of the clean neck run (horn tips / body edge)
    nut_x: float
    width_fit: tuple[float, float]  # neck width(x) = a*x + b, px
    centre_fit: tuple[float, float]  # neck centre y(x) = a*x + b, px
    body: np.ndarray  # open arc (N, 2), bass neck exit → butt → treble neck exit
    head: np.ndarray  # open arc (N, 2), bass side of nut → tip → treble side
    outline: np.ndarray  # full closed outline (N, 2)
    warnings: list[str] = field(default_factory=list)

    def neck_width(self, x):
        return self.width_fit[0] * x + self.width_fit[1]

    def neck_centre(self, x):
        return self.centre_fit[0] * x + self.centre_fit[1]


def profile(mask: np.ndarray):
    m = mask > 0
    count = m.sum(0)
    any_ = count > 0
    top = np.where(any_, m.argmax(0), 0)
    bottom = np.where(any_, m.shape[0] - 1 - m[::-1].argmax(0), 0)
    return count, top, bottom


def _robust_line(x, y, rel=0.12, iters=3):
    keep = np.ones(len(x), bool)
    for _ in range(iters):
        a, b = np.polyfit(x[keep], y[keep], 1)
        r = np.abs(y - (a * x + b))
        keep = r <= max(rel * np.median(np.abs(y[keep])), 1.5)
        if keep.sum() < 10:
            break
    return float(a), float(b)


def find_neck(mask: np.ndarray):
    """(neck_x0, run_end, width_fit, centre_fit) on an oriented mask."""
    count, top, bottom = profile(mask)
    xs = np.nonzero(count)[0]
    x0, x1 = int(xs[0]), int(xs[-1])
    narrow = (count > 0) & (count < 0.33 * count.max())
    # longest narrow run = neck + head
    best, start = (0, 0), None
    for x in range(x0, x1 + 2):
        if x <= x1 and narrow[x]:
            start = x if start is None else start
        elif start is not None:
            if x - start > best[1] - best[0]:
                best = (start, x - 1)
            start = None
    r0, r1 = best
    if r1 - r0 < 0.25 * (x1 - x0):
        raise ValueError("neck not found")
    L = r1 - r0
    fx = np.arange(r0 + int(0.05 * L), r0 + int(0.55 * L))
    wfit = _robust_line(fx.astype(float), count[fx].astype(float))
    solid = count[fx] >= 0.9 * (bottom[fx] - top[fx] + 1)  # only the neck in this column
    cx = fx[solid]
    cfit = _robust_line(cx.astype(float), (top[cx] + bottom[cx]) / 2.0, rel=0.02)
    # walk left from the fit region while the column still matches the neck
    w = lambda x: wfit[0] * x + wfit[1]
    nx0 = fx[0]
    while nx0 > x0 and count[nx0 - 1] <= w(nx0 - 1) * 1.15 + 2:
        nx0 -= 1
    return nx0, r1, wfit, cfit, count


def find_nut(count: np.ndarray, wfit, x_from: int, x_to: int) -> float:
    """First x where the profile stays above the neck taper line (head flare starts)."""
    w = lambda x: wfit[0] * x + wfit[1]
    run = max(3, int(0.15 * w(x_from)))  # flare must persist ~ a few mm
    above = [count[x] > w(x) * 1.12 + 2 for x in range(x_from, x_to + 1)]
    for i in range(len(above) - run):
        if all(above[i : i + run]):
            return float(x_from + i)
    return float(x_to)


def _runs(labels: np.ndarray, value: int):
    """Contiguous runs of `value` in a circular label array → list of index arrays (longest first)."""
    n = len(labels)
    on = labels == value
    if on.all():
        return [np.arange(n)]
    shift = int(np.argmin(on))  # start at an 'off' point so runs don't wrap
    out, cur = [], []
    for k in range(n):
        i = (shift + k) % n
        if on[i]:
            cur.append(i)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return sorted(out, key=len, reverse=True)


def split(mask: np.ndarray, nut_x: float | None = None) -> Split:
    """nut_x: known nut position (from frets); otherwise estimated from the width profile."""
    nx0, run_end, wfit, cfit, count = find_neck(mask)
    xs = np.nonzero(count)[0]
    butt_x, tip_x = int(xs[0]), int(xs[-1])
    widest_x = int(np.argmax(count))
    mid = (nx0 + run_end) // 2
    if nut_x is None:
        nut_x = find_nut(count, wfit, mid, run_end)

    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    x, y = outline[:, 0], outline[:, 1]
    half = (wfit[0] * x + wfit[1]) / 2
    tol = np.maximum(3, 0.12 * half * 2)
    in_strip = (np.abs(y - (cfit[0] * x + cfit[1])) <= half + tol) & (x > widest_x)
    labels = np.zeros(len(outline), int)  # 0 body, 1 neck, 2 head
    labels[in_strip] = 1
    labels[x > nut_x] = 2

    warnings = []
    body_runs, head_runs = _runs(labels, 0), _runs(labels, 2)
    if not body_runs or not head_runs:
        raise ValueError("body or head not found (guitar cropped?)")
    if len(body_runs) > 1 and len(body_runs[1]) > 0.1 * len(body_runs[0]):
        warnings.append("body_outline_fragmented")
    body, head = outline[body_runs[0]], outline[head_runs[0]]
    centre = lambda p: cfit[0] * p[0] + cfit[1]
    if body[0][1] > centre(body[0]):  # must start on the bass side (y above the centre line)
        body = body[::-1]
    if head[0][1] > centre(head[0]):
        head = head[::-1]
    return Split(butt_x, tip_x, widest_x, int(nx0), nut_x, wfit, cfit, body, head, outline, warnings)
