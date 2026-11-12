"""Editor conversion preserves the existing renderer exactly, without inference."""
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from editor_sample import sample
from gbcv import trace as tr
from gbcv.editor import document
from gbcv import server
from fastapi import HTTPException


@pytest.mark.parametrize('style', tr.STYLES)
def test_lossless_primitives_and_styles(style):
    t = sample()
    before = tr.svg(t, style)
    doc = document(t)
    original = ET.fromstring(before)
    primitives = [p for obj in doc['objects'] for p in obj['primitives']]
    assert len(primitives) == len(original)
    for primitive, node in zip(primitives, original, strict=True):
        assert primitive['type'] == node.tag.rsplit('}', 1)[-1]
        assert primitive['geometry'] | primitive['styles'][style] == node.attrib
    assert doc['viewBox'] == original.attrib['viewBox']
    assert tr.svg(t, style) == before
    assert json.loads(json.dumps(doc)) == doc


def test_ids_order_categories_and_compound_rules():
    t = sample()
    doc = document(t)
    assert doc == document(t)
    objects = doc['objects']
    assert len({o['id'] for o in objects}) == len(objects)
    assert [o['order'] for o in objects] == list(range(len(objects)))
    assert {'silhouette','board','nut','fret','post','key','hardware','control'} <= {o['category'] for o in objects}
    assert len([o for o in objects if o['category'] == 'fret']) == 3
    for category in ('post', 'control'):
        obj = next(o for o in objects if o['category'] == category)
        assert obj['type'] == 'compound'
        assert [p['type'] for p in obj['primitives']] == ['circle','circle']
    remaining = [o for o in objects if o['id'] != 'fret-1']
    assert [o['id'] for o in remaining if o['category'] == 'fret'] == ['fret-0', 'fret-2']


def test_manual_style_widths_match_automatic_posts():
    doc = document(sample())
    post = next(o for o in doc['objects'] if o['category'] == 'post')
    assert all(o['category'] != 'manual' for o in doc['objects'])
    for name, widths in doc['manualStyles'].items():
        assert widths['outer'] == float(post['primitives'][0]['styles'][name]['stroke-width'])
        assert widths['inner'] == float(post['primitives'][1]['styles'][name]['stroke-width'])


@pytest.mark.parametrize('perspective', [0, .0005, -.0003])
def test_overlay_projection_matches_ruler(perspective):
    t = sample(perspective)
    projection = np.array(document(t)['photo']['projection']).reshape(3,3)
    pts = np.vstack([t.outline, t.board, [[160,0],[30,15]]])
    projected = np.c_[pts, np.ones(len(pts))] @ projection.T
    np.testing.assert_allclose(projected[:,:2] / projected[:,2,None], t.ruler.to_px(pts), atol=1e-10)


def test_open_existing_session_without_inference(monkeypatch):
    t = sample()
    monkeypatch.setattr(server, 'sessions', {'saved': server.Session(t, time.time())})
    def no_inference(*args, **kwargs):
        pytest.fail('opening a saved session must not call trace()')
    monkeypatch.setattr(tr, 'trace', no_inference)
    payload = server.api_result('saved')
    assert payload['id'] == 'saved'
    assert payload['editor'] == document(t)
    assert payload['photo'].startswith('data:image/jpeg;base64,')
    assert payload['stats']['seconds'] == 0


@pytest.mark.parametrize('expired', [False, True])
def test_missing_or_expired_editor_session(monkeypatch, expired):
    monkeypatch.setattr(server, 'sessions', {'saved': server.Session(sample(), 0)} if expired else {})
    with pytest.raises(HTTPException) as error:
        server.api_result('saved')
    assert error.value.status_code == 404
