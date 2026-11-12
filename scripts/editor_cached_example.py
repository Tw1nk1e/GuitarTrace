"""Open the real application with LP's saved SAM results, without ML.

uv run python scripts/editor_cached_example.py
uv run python scripts/editor_cached_example.py --open --allow-sam
Requires the existing samples/dl/lp_ref_1.webp and out/debug-sam cache.
No photographs or new caches are written to disk.
Default: all inference disabled. --allow-sam enables the existing SAM for explicit
auto-selection requests only (one application permission per session); startup stays ML-free.
"""
from contextlib import ExitStack
import secrets
import sys
import threading
import time
from unittest.mock import patch

import uvicorn

from audit_knob_export import replay
from debug_fretboard import no_ml
from gbcv import server


if __name__ == '__main__':
    # Startup always uses saved masks. Even --allow-sam must not infer on startup.
    with patch('gbcv.segment.probability', no_ml), patch('gbcv.sam.Session', no_ml):
        t = replay()
    with ExitStack() as guards:
        guards.enter_context(patch('gbcv.segment.probability', no_ml))
        guards.enter_context(patch.object(server.tr, 'trace', no_ml))
        if '--allow-sam' not in sys.argv:
            guards.enter_context(patch('gbcv.sam.Session', no_ml))
        sid = secrets.token_urlsafe(12)
        server.sessions[sid] = server.Session(t, time.time())
        port = server._free_port('127.0.0.1', 8766)
        mode = 'SAM после разрешения на сеанс' if '--allow-sam' in sys.argv else 'без ML'
        print(f'Редактор реальной LP ({mode}): http://127.0.0.1:{port}/?session={sid}', flush=True)
        if '--open' in sys.argv:
            threading.Thread(target=server._open_when_ready,
                             args=(f'http://127.0.0.1:{port}/?session={sid}',), daemon=True).start()
        uvicorn.run(server.app, host='127.0.0.1', port=port, log_level='warning')
