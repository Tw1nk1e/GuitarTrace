"""One SAM inference, or exact cached replay. No downloads, no new dependencies.

uv run --no-sync python scripts/debug_sam.py samples/dl/lp_ref_1.webp
uv run --no-sync python scripts/debug_sam.py samples/dl/lp_ref_1.webp --replay out/debug-sam/lp_ref_1/before/sam-cache.npz --label after
"""

import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gbcv import sam, trace as tr
from gbcv.imageio import load_image
from gbcv.sam_debug import SamDebug


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--label", default="before")
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.image.read_bytes()).hexdigest()
    dest = Path("out/debug-sam") / args.image.stem / args.label
    if (dest / "sam-cache.npz").exists():
        raise SystemExit("Refusing to overwrite an existing diagnostic; choose another --label")
    if not args.replay and not sam.WEIGHTS.exists():
        raise SystemExit("Installed SAM weights missing; downloading is forbidden")

    def forbidden(*a, **kw):
        raise RuntimeError("BiRefNet inference is forbidden in this SAM diagnostic")

    with ExitStack() as stack:
        stack.enter_context(patch("gbcv.segment.probability", forbidden))
        if args.replay:
            cache = stack.enter_context(np.load(args.replay, allow_pickle=False))
            provenance = args.replay.parent / "source.sha256"
            if not provenance.exists() or provenance.read_text().strip() != source_hash:
                raise SystemExit("Replay source hash missing or mismatched")

            class Replay:
                def __init__(self, image):
                    self.k = float(cache["k"])
                    x0, y0, x1, y1 = cache["crop"]
                    assert image.shape[:2] == (y1-y0, x1-x0), "Stale crop"

                def each(self, points, min_score):
                    np.testing.assert_allclose(points, cache["points"], atol=1e-8)
                    return [(m, float(s), int(i)) for m, s, i in zip(cache["masks"], cache["scores"], cache["prompts"]) if s >= min_score]

            stack.enter_context(patch.object(sam, "Session", Replay))
        debug = SamDebug(dest)
        (dest / "source.sha256").write_text(source_hash + "\n")
        t = tr.trace(load_image(args.image, mode="RGBA"), sam_debug=debug)
        text = tr.svg(t, "png", sam_debug=debug)
        debug.finish(t, text)
        print(f"{dest / 'index.html'}: {len(debug.records)} candidates, {len(t.parts)} exported parts", flush=True)


if __name__ == "__main__":
    main()
