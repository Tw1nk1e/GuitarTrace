import io

import numpy as np
import pytest
from PIL import Image

from gbcv import imageio
from gbcv.imageio import BadImage, load_image


def encode(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def test_loads_rgb_and_applies_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    exif[0x010F] = "SecretCam"
    data = encode(Image.new("RGB", (200, 100), (10, 20, 30)), "JPEG", exif=exif)
    a = load_image(data)
    assert isinstance(a, np.ndarray) and a.dtype == np.uint8
    assert a.shape == (200, 100, 3)


def test_rejects_non_image():
    with pytest.raises(BadImage, match="not an image"):
        load_image(b"<svg xmlns='http://www.w3.org/2000/svg'/>")


def test_rejects_disallowed_format():
    with pytest.raises(BadImage, match="unsupported format"):
        load_image(encode(Image.new("RGB", (100, 100)), "BMP"))


def test_rejects_too_many_pixels(monkeypatch):
    monkeypatch.setattr(imageio, "MAX_PIXELS", 100 * 100)
    with pytest.raises(BadImage, match="too large"):
        load_image(encode(Image.new("L", (101, 100)), "PNG"))


def test_rejects_big_file(monkeypatch):
    monkeypatch.setattr(imageio, "MAX_FILE_BYTES", 10)
    with pytest.raises(BadImage, match="file too large"):
        load_image(encode(Image.new("RGB", (100, 100)), "PNG"))


def test_rejects_truncated():
    data = encode(Image.effect_noise((300, 300), 50).convert("RGB"), "JPEG")
    with pytest.raises(BadImage):
        load_image(data[:200])


def test_loads_avif():
    a = load_image(encode(Image.new("RGB", (120, 80), (200, 100, 50)), "AVIF"))
    assert a.shape == (80, 120, 3)
