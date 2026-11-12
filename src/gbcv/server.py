"""Local web service: upload a guitar photo, see its trace at once.

uv run python -m gbcv.server  →  http://127.0.0.1:8000

The photo lives only in memory, in a session kept for later edits (30 min, or until the page is closed); it is never
written to disk or logged. One trace at a time: SAM and BiRefNet are shared and CPU-bound.
"""

import base64
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool

from . import trace as tr
from .editor import added_part, document
from .imageio import MAX_FILE_BYTES, BadImage, load_image

WEB = Path(__file__).resolve().parent / "web"
SESSION_TTL_S = 30 * 60
MAX_SESSIONS = 8
OVERLAY_MAX_SIDE = 3200  # zoomed in for clicking: needs pixels


HISTORY = 30  # undo steps
AUTO_CACHE_SIZE = 32  # small contours only; never store full-frame masks or photos


@dataclass
class Session:
    trace: tr.Trace
    touched: float
    history: list = None  # snapshots for undo: (outline, mask or None, parts, posts)
    variants: list = None  # SAM's masks for the last added part, and which one is shown
    shown: int = 0
    smooth: float = tr.DEFAULT_SMOOTH_MM
    sam_enabled: bool = False
    auto_cache: dict = field(default_factory=dict)  # (pixel x, pixel y) -> unsmoothed part
    auto_mask: np.ndarray | None = None  # identity guard, not a second photo/mask copy
    auto_lock: object = field(default_factory=threading.Lock, repr=False)

    def snapshot(self, with_mask: bool = False) -> None:
        t = self.trace
        self.history = (self.history or [])[-HISTORY + 1 :]
        self.history.append((t.outline, t.mask.copy() if with_mask else None, list(t.parts), list(t.posts)))


class Click(BaseModel):
    mode: str  # add | remove | cut
    x: float  # 0..1 across the overlay image
    y: float


sessions: dict[str, Session] = {}
lock = threading.Lock()  # one trace at a time
app = FastAPI(title="GuitarTrace", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")


def _expire() -> None:
    now = time.time()
    for sid in [k for k, v in sessions.items() if now - v.touched > SESSION_TTL_S]:
        del sessions[sid]
    while len(sessions) > MAX_SESSIONS:
        del sessions[min(sessions, key=lambda k: sessions[k].touched)]


def _jpeg_data_url(rgb: np.ndarray) -> str:
    k = min(1.0, OVERLAY_MAX_SIDE / max(rgb.shape[:2]))
    if k < 1:
        rgb = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _svgs(t: tr.Trace, smooth: float) -> dict:
    return {name: tr.svg(t, name, smooth) for name in tr.STYLES}


def _result(sid: str, t: tr.Trace, seconds: float) -> dict:
    s = sessions.get(sid)
    return {
        "id": sid,
        "sam_enabled": bool(s and s.sam_enabled),
        "svg": _svgs(t, s.smooth if s else tr.DEFAULT_SMOOTH_MM),
        "photo": _jpeg_data_url(tr.masked_photo(t)),
        "lines": tr.overlay_svg(t),
        "editor": document(t, s.smooth if s else tr.DEFAULT_SMOOTH_MM),
        "stats": {"frets": len(t.frets), "posts": len(t.posts), "parts": len(t.parts), "seconds": round(seconds, 1)},
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.post("/api/trace")
def api_trace(file: UploadFile = File(...)) -> dict:
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB")
    try:
        rgba = load_image(data, mode="RGBA")
    except BadImage as e:
        raise HTTPException(400, str(e)) from None
    t0 = time.perf_counter()
    with lock:
        try:
            t = tr.trace(rgba)  # the drawing keeps the photo's proportions; a standard scale only sets mm
        except ValueError as e:
            raise HTTPException(422, f"Processing failed: {e}") from None
    del rgba, data
    sid = secrets.token_urlsafe(12)
    sessions[sid] = Session(t, time.time())
    _expire()
    return _result(sid, t, time.perf_counter() - t0)


class Smooth(BaseModel):
    mm: float


@app.post("/api/s/{sid}/smooth")
def api_smooth(sid: str, b: Smooth) -> dict:
    """Re-draw with another smoothing (no re-trace): only the SVG changes."""
    s = _session(sid)
    s.smooth = float(np.clip(b.mm, 0, 5))
    return {"svg": _svgs(s.trace, s.smooth)}


def _session(sid: str) -> Session:
    _expire()
    s = sessions.get(sid)
    if s is None:
        raise HTTPException(404, "Session expired. Upload the image again.")
    s.touched = time.time()
    return s


@app.get("/api/s/{sid}")
def api_result(sid: str) -> dict:
    """Open an in-memory trace in the editor without running inference again."""
    s = _session(sid)
    return _result(sid, s.trace, 0)


@app.post("/api/s/{sid}/click")
def api_click(sid: str, c: Click) -> dict:
    s = _session(sid)
    t = s.trace
    h, w = t.mask.shape
    x_px, y_px = c.x * w, c.y * h
    t0 = time.perf_counter()
    if c.mode == "remove":
        x_mm, y_mm = t.ruler.to_mm(np.array([[x_px, y_px]]))[0]
        s.snapshot()
        if not tr.remove_at(t, x_mm, y_mm):
            s.history.pop()
            raise HTTPException(422, "Nothing to remove here")
        s.variants = None
    elif c.mode in ("add", "cut"):
        with lock:
            cands = tr.sam_at(t, x_px, y_px)
        if not cands:
            raise HTTPException(422, "No selection found here. Try a more precise click.")
        best = max(range(len(cands)), key=lambda i: cands[i][1])
        if c.mode == "cut":
            s.snapshot(with_mask=True)
            tr.cut_silhouette(t, cands[best][0])
            s.variants = None
        else:
            part = tr.part_from_mask(t, cands[best][0])
            if part is None:
                raise HTTPException(422, "No selection found here. Try a more precise click.")
            s.snapshot()
            t.parts.append(part)
            s.variants, s.shown = [m for m, _ in cands], best
    else:
        raise HTTPException(400, "Unknown mode")
    return _result(sid, t, time.perf_counter() - t0) | {"variants": len(s.variants or [])}


class AutoSelect(BaseModel):
    x: float = Field(ge=0, lt=1, allow_inf_nan=False)
    y: float = Field(ge=0, lt=1, allow_inf_nan=False)


class SAMPermission(BaseModel):
    enabled: StrictBool


@app.post("/api/s/{sid}/sam-permission")
def api_sam_permission(sid: str, permission: SAMPermission) -> dict:
    """Explicit application-only opt-in for this live photo session. No ML."""
    s = _session(sid)
    s.sam_enabled = permission.enabled
    return {"sam_enabled": s.sam_enabled}


@app.post("/api/s/{sid}/auto-select")
def api_auto_select(sid: str, click: AutoSelect) -> dict:
    """Existing click-to-SAM, but return ONE object, not an obsolete full document.

    The browser owns edits/history. Inference is opt-in per live session. No trace(),
    BiRefNet, downloads, server-side additions or stale document merges here.
    """
    s = _session(sid)
    if not s.sam_enabled:
        raise HTTPException(428, "Enable SAM for this editor session first.")
    if not s.auto_lock.acquire(blocking=False):
        raise HTTPException(409, "Selection is still processing. Wait for it to finish and try again.")
    try:
        t = s.trace
        h, w = t.mask.shape
        # A prompt is a photo pixel: subpixel mouse jitter is the same input.
        x, y = int(click.x * w), int(click.y * h)
        if not t.mask[y, x]:
            raise HTTPException(422, "Click a part inside the silhouette, not the background.")
        if s.auto_mask is not t.mask:
            s.auto_cache.clear()
            s.auto_mask = t.mask
        key = (x, y)
        part = s.auto_cache.get(key)
        if part is None:
            if not lock.acquire(blocking=False):
                raise HTTPException(409, "The model is busy. Wait for it to finish and try again.")
            try:
                try:
                    candidates = tr.sam_at(t, x, y)
                except (FileNotFoundError, RuntimeError, ImportError):
                    raise HTTPException(503, "SAM is unavailable or disabled. Models are not downloaded automatically; use manual tools.") from None
                if not candidates:
                    raise HTTPException(422, "No part selected. Try a more precise click or draw it manually.")
                part = tr.part_from_mask(t, max(candidates, key=lambda c: c[1])[0])
                if part is None:
                    raise HTTPException(422, "Could not extract an outline. Try another point or a manual tool.")
            finally:
                lock.release()
            s.auto_cache[key] = part
            if len(s.auto_cache) > AUTO_CACHE_SIZE:
                del s.auto_cache[next(iter(s.auto_cache))]
        if not s.sam_enabled:
            raise HTTPException(428, "SAM permission is disabled. The drawing is unchanged.")
        # Re-export from the cached polygon: smoothing changes never repeat ML.
        return {"object": added_part(t, part, s.smooth)}
    finally:
        s.auto_lock.release()


@app.post("/api/s/{sid}/variant")
def api_variant(sid: str) -> dict:
    """The last added part: SAM's next mask for the same click (smaller / larger)."""
    s = _session(sid)
    if not s.variants:
        raise HTTPException(422, "No alternatives available. Add a part first.")
    t = s.trace
    for _ in range(len(s.variants)):
        s.shown = (s.shown + 1) % len(s.variants)
        part = tr.part_from_mask(t, s.variants[s.shown])
        if part is not None:
            t.parts[-1] = part
            break
    return _result(sid, t, 0) | {"variants": len(s.variants)}


@app.post("/api/s/{sid}/undo")
def api_undo(sid: str) -> dict:
    s = _session(sid)
    if not s.history:
        raise HTTPException(422, "Nothing to undo")
    t = s.trace
    outline, mask, parts, posts = s.history.pop()
    t.outline, t.parts, t.posts = outline, parts, posts
    if mask is not None:
        t.mask = mask
    s.variants = None
    return _result(sid, t, 0) | {"variants": 0}


@app.post("/api/s/{sid}/close")
def api_close(sid: str) -> dict:
    sessions.pop(sid, None)
    return {"ok": True}


def _open_when_ready(url: str) -> None:
    """Open the page in the default browser once the server answers."""
    import urllib.request
    import webbrowser

    for _ in range(120):
        try:
            urllib.request.urlopen(url, timeout=1)
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.5)


def _free_port(host: str, start: int, tries: int = 20) -> int:
    """First port from `start` that nobody listens on (8000 is often taken, e.g. by Docker)."""
    import socket

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port in {start}–{start + tries - 1}; set GBCV_PORT")


def main() -> None:
    """--open: open the page in the browser when the server is up (Запуск.cmd)."""
    import sys

    import uvicorn

    host = os.environ.get("GBCV_HOST", "127.0.0.1")
    port = int(os.environ["GBCV_PORT"]) if "GBCV_PORT" in os.environ else _free_port(host, 8000)
    print(f"Server: http://{host}:{port}", flush=True)
    if "--open" in sys.argv:
        threading.Thread(target=_open_when_ready, args=(f"http://{host}:{port}/",), daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
