"""Download model weights into models/ (not in git) and verify checksums.

uv run python scripts/fetch_models.py
"""

import hashlib
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"

# BiRefNet (MIT, github.com/ZhengPeng7/BiRefNet), swin-tiny "general" weights exported to ONNX by rembg (MIT)
FILES = {
    "birefnet-lite.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
    ),
    # SAM 2.1 hiera-tiny (Apache 2.0, github.com/facebookresearch/sam2): parts of catalogue photos for trace.py
    "sam2.1_hiera_tiny.pt": (
        "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt",
        "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for name, (url, digest) in FILES.items():
        dst = MODELS / name
        if not dst.exists():
            print(f"downloading {name} …")
            urllib.request.urlretrieve(url, dst)
        if sha256(dst) != digest:
            dst.unlink()
            raise SystemExit(f"{name}: checksum mismatch, file removed")
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
