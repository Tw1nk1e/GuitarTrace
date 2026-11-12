"""Main axis of the guitar mask and the rotation that lays it horizontally, head to the right.

Only rotations are used (no mirroring), so a front-view photo of a right-handed guitar keeps the bass side
on top (Y < 0) as in the renderer. Left-handed / back views need detail cues (stage 5–6).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .split import Split, split

FOCAL = 1.0  # assumed focal length, in long sides of the photo (phones ~0.8–1.2)


@dataclass
class Frame:
    """Affine map image px → rotated canvas px (head to the right)."""

    M: np.ndarray  # 2x3
    size: tuple[int, int]  # (w, h) of the canvas

    def warp(self, img: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.warpAffine(img, self.M, self.size, flags=interp, borderValue=0)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.M[:, :2].T + self.M[:, 2]


def pca_angle(mask: np.ndarray) -> float:
    """Angle (degrees, image coords) of the mask's major axis."""
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(float)
    pts -= pts.mean(0)
    w, v = np.linalg.eigh(np.cov(pts.T))
    vx, vy = v[:, np.argmax(w)]
    return float(np.degrees(np.arctan2(vy, vx)))


def frame_for(mask: np.ndarray, angle: float) -> Frame:
    """Rotation by -angle about the mask centroid, canvas expanded to hold the whole image."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    c = (float(xs.mean()), float(ys.mean()))
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) @ M[:, :2].T + M[:, 2]
    lo, hi = corners.min(0), corners.max(0)
    M[:, 2] -= lo
    return Frame(M, (int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1]))))


def head_right(mask: np.ndarray, angle: float) -> float:
    """Return angle or angle+180 so that the wide end (body) lands on the left."""
    m = frame_for(mask, angle).warp(mask, cv2.INTER_NEAREST)
    cols = (m > 0).sum(0)
    xs = np.nonzero(cols)[0]
    x0, x1 = xs[0], xs[-1]
    third = (x1 - x0) // 3
    left, right = cols[x0 : x0 + third].sum(), cols[x1 - third : x1 + 1].sum()
    return angle if left >= right else angle + 180


def remove_thin_bits(mask: np.ndarray, sp: Split, open_body: bool = True) -> np.ndarray:
    """Up to the nut: open away stand legs, cables, strap buttons (the kernel is narrower than the neck);
    it also rounds sharp horns, so a clean-background photo skips it (open_body=False).
    Head: cut tuner shafts so the keys fall off, then grow back a little within the original mask
    to restore the headstock's corners (leaves short shaft stubs)."""
    ellipse = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    nw = sp.neck_width(sp.neck_x0)
    hx = int(sp.nut_x)
    out = mask.copy()
    if open_body:
        out[:, :hx] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(max(3, int(0.45 * nw))))[:, :hx]
    kh = max(3, int(0.5 * nw))
    out[:, hx:] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(kh))[:, hx:]
    n, lab, st, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8))
    keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    out = (lab == keep).astype(np.uint8) * 255
    grown = cv2.dilate(out, ellipse(kh // 3 * 2 + 1)) & mask
    out[:, hx:] = grown[:, hx:]
    # a strap button on the butt is too big for the opening; cut a short run of end columns far narrower than the body
    w = (out > 0).sum(0)
    x0 = int(np.flatnonzero(w)[0])
    k = x0
    while k < sp.widest_x and w[k] < 0.1 * w.max():
        k += 1
    if k - x0 <= nw:
        out[:, :k] = 0
    return out


def orient_and_split(mask: np.ndarray, open_body: bool = True):
    """Rotate the guitar horizontal (head right), split it, clean the mask; returns (frame, mask, split)."""
    angle = head_right(mask, pca_angle(mask))
    fr = frame_for(mask, angle)
    sp = split(fr.warp(mask, cv2.INTER_NEAREST))
    # refine: the strings run along the neck centre line, not the mask's principal axis
    angle += float(np.degrees(np.arctan(sp.centre_fit[0])))
    fr = frame_for(mask, angle)
    m = fr.warp(mask, cv2.INTER_NEAREST)
    sp = split(m)
    m = remove_thin_bits(m, sp, open_body)
    return fr, m, split(m)
