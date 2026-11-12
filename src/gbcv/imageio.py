"""Loading untrusted photos: format sniffed from content, size and pixel limits, EXIF dropped."""

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "AVIF"}
MAX_FILE_BYTES = int(float(os.environ.get("GBCV_MAX_FILE_MB", 25)) * 1024 * 1024)
MAX_PIXELS = int(float(os.environ.get("GBCV_MAX_MEGAPIXELS", 40)) * 1_000_000)
MIN_SIDE = 64

# Pillow's own bomb guard as a backstop (it raises at 2x this value)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class BadImage(ValueError):
    """Input rejected; the message is safe to show to the uploader."""


def load_image(src: bytes | str | Path, mode: str = "RGB") -> np.ndarray:
    """Return an RGB (or mode="RGBA") uint8 array (H, W, C), upright per EXIF orientation, with no metadata attached."""
    data = Path(src).read_bytes() if isinstance(src, (str, Path)) else src
    if len(data) > MAX_FILE_BYTES:
        raise BadImage(f"file too large (> {MAX_FILE_BYTES // (1024 * 1024)} MB)")
    try:
        im = Image.open(io.BytesIO(data))
    except Exception as e:
        raise BadImage("not an image") from e
    with im:
        if im.format not in ALLOWED_FORMATS:
            raise BadImage(f"unsupported format {im.format}; allowed: {', '.join(sorted(ALLOWED_FORMATS))}")
        w, h = im.size  # header only, nothing decoded yet
        if w * h > MAX_PIXELS:
            raise BadImage(f"image too large ({w}x{h}, limit {MAX_PIXELS / 1e6:.0f} MP)")
        if min(w, h) < MIN_SIDE:
            raise BadImage(f"image too small ({w}x{h})")
        try:
            im = ImageOps.exif_transpose(im)
            return np.asarray(im.convert(mode)).copy()
        except Image.DecompressionBombError as e:
            raise BadImage("image too large") from e
        except Exception as e:
            raise BadImage("corrupt image") from e
