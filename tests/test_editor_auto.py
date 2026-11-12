"""Existing mask->part->SVG adapter and API, with SAM always replaced (no inference)."""
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from editor_sample import sample
from gbcv import server, trace as tr
from gbcv.editor import added_part, document


@pytest.fixture
def scene(monkeypatch):
    t = sample(.0005)
    monkeypatch.setattr(server, 'sessions', {'live': server.Session(t, time.time(), sam_enabled=True)})
    def forbidden(*args, **kwargs):
        pytest.fail('ML/trace must never run in this test')
    monkeypatch.setattr(tr, 'trace', forbidden)
    monkeypatch.setattr('gbcv.sam.Session', forbidden)
    monkeypatch.setattr('gbcv.segment.probability', forbidden)
    mask = np.zeros(t.mask.shape, np.uint8)
    cv2.ellipse(mask, (300, 150), (25, 12), 15, 0, 360, 1, -1)
    return t, mask.astype(bool)


def test_one_object_reuses_mask_conversion_and_export_without_mutating_trace(scene, monkeypatch):
    t, mask = scene
    before = document(t)
    parts = [p.copy() for p in t.parts]
    calls = []
    monkeypatch.setattr(tr, 'sam_at', lambda trace, x, y: calls.append((trace, x, y)) or [(np.zeros_like(mask), .4), (mask, .9)])
    reply = server.api_auto_select('live', server.AutoSelect(x=300/640, y=150/320))
    expected = added_part(t, tr.part_from_mask(t, mask))
    assert reply == {'object': expected}
    assert expected['category'] == 'hardware' and expected['type'] == 'path'
    assert calls == [(t, 300, 150)]
    assert document(t) == before
    for a, b in zip(t.parts, parts, strict=True):
        np.testing.assert_array_equal(a, b)
    assert server.sessions['live'].history is None
    # Repeated requests never append server-side copies or invalidate local Undo.
    assert server.api_auto_select('live', server.AutoSelect(x=300/640, y=150/320)) == reply
    assert calls == [(t, 300, 150)]  # repeated pixel reuses the result, no inference
    assert document(t) == before


def test_part_adapter_matches_existing_renderer_for_all_styles(scene):
    t, mask = scene
    part = tr.part_from_mask(t, mask)
    expected = next(o for o in document(replace(t, parts=[part], knobs=[]))['objects'] if o['category'] == 'hardware')
    assert added_part(t, part) == expected


def test_session_consent_does_not_infer_and_is_strict_revocable_and_not_persisted(scene, monkeypatch):
    monkeypatch.setattr(tr, 'sam_at', lambda *a: pytest.fail('unconfirmed inference'))
    s = server.sessions['live']
    s.sam_enabled = False
    with pytest.raises(HTTPException) as error:
        server.api_auto_select('live', server.AutoSelect(x=.5, y=.5))
    assert error.value.status_code == 428
    assert server.api_sam_permission('live', server.SAMPermission(enabled=True)) == {'sam_enabled': True}
    assert s.sam_enabled
    assert server.api_result('live')['sam_enabled']  # reload the live session without re-consenting or inferring
    assert 'sam_enabled' not in document(s.trace)
    assert not server.Session(s.trace, time.time()).sam_enabled
    server.api_sam_permission('live', server.SAMPermission(enabled=False))
    with pytest.raises(HTTPException) as error:
        server.api_auto_select('live', server.AutoSelect(x=.5, y=.5))
    assert error.value.status_code == 428
    with pytest.raises(ValidationError):
        server.SAMPermission(enabled='true')


@pytest.mark.parametrize('field,value', [('x',-1),('x',1),('y',float('nan')),('y',float('inf')),('x',1.1)])
def test_click_coordinates_are_bounded(field, value):
    args = dict(x=.5, y=.5)
    args[field] = value
    with pytest.raises(ValidationError):
        server.AutoSelect(**args)


def test_background_never_runs_sam(scene, monkeypatch):
    t, _ = scene
    t.mask[:20] = 0
    monkeypatch.setattr(tr, 'sam_at', lambda *a: pytest.fail('background must not run SAM'))
    with pytest.raises(HTTPException) as error:
        server.api_auto_select('live', server.AutoSelect(x=.1, y=.01))
    assert error.value.status_code == 422


@pytest.mark.parametrize('candidates', [[], [(np.zeros((320,640), bool), .9)]])
def test_failed_selection_changes_nothing(scene, monkeypatch, candidates):
    t, _ = scene
    before = document(t)
    monkeypatch.setattr(tr, 'sam_at', lambda *a: candidates)
    with pytest.raises(HTTPException) as error:
        server.api_auto_select('live', server.AutoSelect(x=.5, y=.5))
    assert error.value.status_code == 422 and document(t) == before


@pytest.mark.parametrize('exception', [FileNotFoundError, RuntimeError, ImportError])
def test_missing_or_disabled_sam_is_reported_without_downloading(scene, monkeypatch, exception):
    def unavailable(*args):
        raise exception('disabled')
    monkeypatch.setattr(tr, 'sam_at', unavailable)
    with pytest.raises(HTTPException) as error:
        server.api_auto_select('live', server.AutoSelect(x=.5, y=.5))
    assert error.value.status_code == 503


def test_expired_session_does_not_try_retrace(scene, monkeypatch):
    monkeypatch.setattr(tr, 'sam_at', lambda *a: pytest.fail('no source session'))
    with pytest.raises(HTTPException) as error:
        server.api_auto_select('expired', server.AutoSelect(x=.5, y=.5))
    assert error.value.status_code == 404


def test_cache_uses_photo_pixels_reexports_smoothing_and_invalidates_mask(scene, monkeypatch):
    t, mask = scene
    calls = []
    monkeypatch.setattr(tr, 'sam_at', lambda *a: calls.append(a) or [(mask, .9)])
    click = server.AutoSelect(x=300/640, y=150/320)
    before = server.api_auto_select('live', click)
    s = server.sessions['live']
    s.smooth = 0
    after = server.api_auto_select('live', server.AutoSelect(x=300.9/640, y=150.9/320))
    assert len(calls) == 1
    assert before != after
    assert after == {'object': added_part(t, tr.part_from_mask(t, mask), 0)}
    t.mask = t.mask.copy()
    server.api_auto_select('live', click)
    assert len(calls) == 2
    server.api_auto_select('live', server.AutoSelect(x=301/640, y=150/320))
    assert len(calls) == 3  # a different pixel is NOT assumed to be the same detail


def test_cache_is_bounded_and_cached_result_does_not_need_model_lock(scene, monkeypatch):
    _, mask = scene
    monkeypatch.setattr(server, 'AUTO_CACHE_SIZE', 2)
    calls = []
    monkeypatch.setattr(tr, 'sam_at', lambda *a: calls.append(a) or [(mask, .9)])
    for x in (300, 301, 302):
        server.api_auto_select('live', server.AutoSelect(x=x/640, y=150/320))
    assert len(server.sessions['live'].auto_cache) == 2
    with server.lock:
        server.api_auto_select('live', server.AutoSelect(x=302/640, y=150/320))
        with pytest.raises(HTTPException) as error:
            server.api_auto_select('live', server.AutoSelect(x=300/640, y=150/320))
        assert error.value.status_code == 409
    assert len(calls) == 3


def test_concurrent_requests_reject_instead_of_queue_and_release_after_failure(scene, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    _, mask = scene
    started, release = Event(), Event()
    def slow(*args):
        started.set()
        assert release.wait(5)
        return [(mask, .9)]
    monkeypatch.setattr(tr, 'sam_at', slow)
    click = server.AutoSelect(x=300/640, y=150/320)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(server.api_auto_select, 'live', click)
        try:
            assert started.wait(5)
            with pytest.raises(HTTPException) as error:
                server.api_auto_select('live', click)
            assert error.value.status_code == 409
            s = server.sessions['live']
            server.sessions['other'] = server.Session(s.trace, time.time(), sam_enabled=True)
            with pytest.raises(HTTPException) as error:
                server.api_auto_select('other', click)
            assert error.value.status_code == 409
        finally:
            release.set()
        first.result(5)
    # No lock is leaked by either rejection or failed segmentation.
    monkeypatch.setattr(tr, 'sam_at', lambda *a: [])
    for _ in range(2):
        with pytest.raises(HTTPException) as error:
            server.api_auto_select('live', server.AutoSelect(x=301/640, y=150/320))
        assert error.value.status_code == 422
    assert (301, 150) not in s.auto_cache


def test_expiry_removes_permission_and_cache(scene, monkeypatch):
    s = server.sessions['live']
    s.touched = time.time() - server.SESSION_TTL_S - 1
    monkeypatch.setattr(tr, 'sam_at', lambda *a: pytest.fail('expired inference'))
    with pytest.raises(HTTPException) as error:
        server.api_auto_select('live', server.AutoSelect(x=.5, y=.5))
    assert error.value.status_code == 404
    assert 'live' not in server.sessions
