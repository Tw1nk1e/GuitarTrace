"""Manual/headless browser fixture. The upload endpoint uses a synthetic Trace, never ML.

Run: uv run python tests/editor_fixture_server.py
"""
from unittest.mock import patch

import uvicorn

from editor_sample import sample
from gbcv.server import app

if __name__ == '__main__':
    with patch('gbcv.server.tr.trace', side_effect=lambda rgba: sample(.0005)):
        uvicorn.run(app, host='127.0.0.1', port=8765, log_level='warning')
