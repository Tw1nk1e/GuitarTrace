"""Guitar mask from a photo: BiRefNet (MIT) via ONNX Runtime on CPU.

Model file: models/birefnet-lite.onnx (scripts/fetch_models.py). Override the folder with GBCV_MODEL_DIR.
"""

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(os.environ.get("GBCV_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
MODEL = "birefnet-lite.onnx"
SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    path = MODEL_DIR / MODEL
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: uv run python scripts/fetch_models.py")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def probability(image: np.ndarray) -> np.ndarray:
    """Foreground probability (H, W) float32 in 0..1 at the image's resolution."""
    s = _session()
    x = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    logits = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    p = 1 / (1 + np.exp(-logits))
    return cv2.resize(p, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def clean(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Binary mask (uint8 0/255): largest component, holes filled."""
    m = (prob > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == keep).astype(np.uint8) * 255
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cs, -1, 255, cv2.FILLED)
    return filled


def mask_quality(prob: np.ndarray, mask: np.ndarray) -> float:
    """Share of confident pixels on the mask's edge band: blurry/uncertain edges → low score."""
    band = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)) > 0
    if not band.any():
        return 0.0
    p = prob[band]
    return float(np.mean((p < 0.1) | (p > 0.9)))
