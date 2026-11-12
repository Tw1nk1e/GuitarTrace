"""Photo → line drawing traced from the photo itself: uv run python trace.py samples/dl/*.webp

For catalogue shots (transparent or plain background). Writes out/<name>/{trace.svg, trace.png} and out/trace.html
(all traces in out/, earlier runs too).
"""

import argparse
import html
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gbcv import trace as tr  # noqa: E402
from gbcv.imageio import BadImage, load_image  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--style", choices=sorted(tr.STYLES), default="comic", help="line weights: comic (bold) or thin")
    args = ap.parse_args(argv)
    rows, failed = [], 0
    for path in args.images:
        t0 = time.perf_counter()
        try:
            t = tr.trace(load_image(path, mode="RGBA"))
        except (BadImage, OSError, ValueError) as e:
            print(f"{path.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        out = args.out / path.stem
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace.svg").write_text(tr.svg(t, args.style), encoding="utf-8")
        cv2.imwrite(str(out / "trace.png"), cv2.cvtColor(tr.overlay(t), cv2.COLOR_RGB2BGR))
        if t.mask is not None:
            cv2.imwrite(str(out / "mask.png"), t.mask)  # the silhouette everything is built from, for a quick look
        rows.append(path.stem)
        print(f"{path.name} -> {out}  ({time.perf_counter() - t0:.1f} s; outline {len(t.outline)} pts, "
              f"frets {len(t.frets)}, posts {len(t.posts)}, parts {len(t.parts)})")
    rows = sorted(d.name for d in args.out.iterdir() if (d / "trace.svg").exists())  # this run and earlier ones
    if rows:
        page = args.out / "trace.html"
        page.write_text(
            "<!doctype html><meta charset=utf-8><title>trace</title><style>body{font:14px sans-serif;margin:16px}"
            ".row{display:flex;gap:12px}.row img{width:49%;border:1px solid #ccc;background:#fff}</style>"
            + "".join(f'<h3>{html.escape(n)}</h3><div class="row"><img src="{n}/trace.png"><img src="{n}/trace.svg"></div>' for n in rows),
            encoding="utf-8",
        )
        print(page)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
