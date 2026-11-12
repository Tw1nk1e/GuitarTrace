"""SAM 2 (Apache 2.0) on CPU: masks of whatever object sits under prompt points.

One encoder pass per photo (~1 s for 1600 px), then ~10 ms per prompt. Weights: models/sam2.1_hiera_tiny.pt
(scripts/fetch_models.py).
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MAX_SIDE = 1600  # SAM works at 1024 inside; more only costs time
BATCH = 24  # prompts per decoder call (3 masks each)


@lru_cache(maxsize=1)
def _predictor():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS.name} missing: uv run python scripts/fetch_models.py")
    return SAM2ImagePredictor(build_sam2(CONFIG, str(WEIGHTS), device="cpu"))


class Session:
    """One image encoded once; prompts in image px. Masks come at image size × k (image px = mask px / k)."""

    def __init__(self, image: np.ndarray):
        import torch

        h, w = image.shape[:2]
        self.k = min(1.0, MAX_SIDE / max(h, w))
        small = cv2.resize(image, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_AREA) if self.k < 1 else image
        self.pred = _predictor()
        with torch.inference_mode():
            self.pred.set_image(np.ascontiguousarray(small))

    def each(self, points: np.ndarray, min_score: float = 0.8) -> list[tuple[np.ndarray, float, int]]:
        """One prompt per point: the candidate masks SAM is sure of, as (mask, score, point index)."""
        import torch

        out = []
        pts = np.asarray(points, float) * self.k
        with torch.inference_mode():
            for i0 in range(0, len(pts), BATCH):
                chunk = pts[i0 : i0 + BATCH]
                m, sc, _ = self.pred.predict(point_coords=chunk[:, None, :], point_labels=np.ones((len(chunk), 1)), multimask_output=True)
                m, sc = m.reshape(len(chunk), 3, *m.shape[-2:]), sc.reshape(len(chunk), 3)
                for j in range(len(chunk)):
                    for c in range(3):
                        if sc[j, c] >= min_score:
                            out.append((m[j, c].astype(np.uint8), float(sc[j, c]), i0 + j))
        return out

    def together(self, points: np.ndarray, negative: np.ndarray | None = None) -> list[tuple[np.ndarray, float]]:
        """One prompt of several points on one object (and points known to be off it): its candidate masks."""
        import torch

        pos = np.asarray(points, float) * self.k
        neg = np.zeros((0, 2)) if negative is None else np.asarray(negative, float) * self.k
        with torch.inference_mode():
            m, sc, _ = self.pred.predict(point_coords=np.vstack([pos, neg]), point_labels=np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                                         multimask_output=True)
        return [(m[c].astype(np.uint8), float(sc[c])) for c in range(len(sc))]
