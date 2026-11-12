"""Photo evidence and adversarial primitives; no models or output caches needed."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from gbcv import knobs, trace as tr
from gbcv.imageio import load_image


def concentric(axes=(28, 28), angle=0):
    image=np.full((180,200,3),[50,40,30],np.uint8)
    for scale,color in ((1.,(220,190,100)),(.78,(70,45,20)),(.57,(230,200,130)),(.36,(80,50,25))):
        cv2.ellipse(image,(90,90),tuple(round(a*scale) for a in axes),angle,0,360,color,-1)
    return image,np.full(image.shape[:2],255,np.uint8)


@pytest.mark.parametrize('axes,angle',[((28,28),0),((30,24),28)])
def test_concentric_photo_edges_support_circle_and_tilted_ellipse(axes,angle):
    photo,mask=concentric(axes,angle)
    accepted=[q for q in knobs.candidates(photo,mask) if q.reason=='supported']
    assert accepted
    assert all(np.linalg.norm(np.array(q.centre)-[90,90])<2 for q in accepted)
    assert all(max(q.axes)>24 for q in accepted)  # not a little inner ring
    assert all(q.metrics['closure']>=.9 and len(q.metrics['inner_rings'])>=2 for q in accepted)
    # Detection retains sampled photo boundaries; export may regularise controls.
    assert all(q.contour is not None and len(q.contour)==96 for q in accepted)


@pytest.mark.parametrize('kind',['disk','washer','open_arc','stripes','small_screw','weak_rings'])
def test_round_or_repeated_hardware_is_not_enough(kind):
    image=np.full((180,200,3),40,np.uint8)
    mask=np.full(image.shape[:2],255,np.uint8)
    if kind=='disk':
        cv2.circle(image,(90,90),28,(210,210,210),-1)
    elif kind=='washer':
        cv2.circle(image,(90,90),30,(230,230,230),-1)
        cv2.circle(image,(90,90),10,(40,40,40),-1)
    elif kind=='open_arc':
        for r in (28,20,12):
            cv2.ellipse(image,(90,90),(r,r),0,10,285,(220,220,220),2)
    elif kind=='stripes':
        for y in range(55,130,12):
            cv2.line(image,(50,y),(130,y),(220,220,220),3)
    elif kind=='small_screw':
        for r in (5,3):
            cv2.circle(image,(90,90),r,(230,230,230),1)
    else:
        for r in (28,20,12):
            cv2.circle(image,(90,90),r,(45,45,45),1)
    assert not [q for q in knobs.candidates(image,mask) if q.reason=='supported']


def test_boundary_outside_guitar_is_not_added():
    photo,mask=concentric()
    mask[:,105:]=0
    assert not [q for q in knobs.candidates(photo,mask) if q.reason=='supported']


def test_recovery_deduplicates_and_never_mutates_existing_parts(monkeypatch):
    photo,mask=concentric()
    monkeypatch.setattr(tr,'_rect',lambda *a:(photo.copy(),mask.copy()))
    t=SimpleNamespace(photo=photo,mask=mask,ruler=None,
                      outline=np.array([[4,4],[104,4],[104,94],[4,94]],float),parts=[])
    added=knobs.recover(t,80)
    assert len(added)==1  # inner/outer edges and Hough aren't extra knobs
    original=added[0].copy()
    t.parts=[original]
    assert not knobs.recover(t,80)
    assert len(t.parts)==1
    np.testing.assert_array_equal(original,t.parts[0])


@pytest.mark.parametrize('name',['lp_ref_1','fender_1','fender_2','ibanez_rgms8'])
def test_actual_photos_recover_lower_right_and_reject_other_round_hardware(name,monkeypatch):
    # Optional local integration tests: catalogue photos are not distributed in
    # git. Fixture inspection windows are tests, never detector coordinates.
    root=Path(__file__).resolve().parents[1]
    source=next((root/'samples/dl').glob(name+'.*'),None)
    if source is None:
        pytest.skip('Local catalogue fixture not installed')
    def forbidden(*a,**kw):
        pytest.fail('No ML inference is allowed in knob tests')
    monkeypatch.setattr('gbcv.segment.probability',forbidden)
    monkeypatch.setattr('gbcv.sam.Session',forbidden)
    monkeypatch.setattr(tr,'_sam_parts',lambda *a,**kw:[])
    with patch.object(knobs,'recover',return_value=[]):
        t=tr.trace(load_image(source,mode='RGBA'))
    debug={}
    added=knobs.recover(t,float(t.board[1:3,0].max()),diagnostics=debug)
    accepted=[q for q in debug['candidates'] if q.reason=='added']
    if name=='lp_ref_1':
        assert len(added)==2  # one already-found knob + the missing lower-right
        assert all(200<q.centre[0]<365 and 545<q.centre[1]<605 for q in accepted)
        assert any(320<q.centre[0]<365 for q in accepted)
        # Toggle washer, bridge posts, pickup screws and neck have no additions.
        assert not any(q.centre[0]>400 or q.centre[1]<400 for q in accepted)
    else:
        assert not added


def test_standard_trace_appends_recovery_after_sam_without_debug(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = root / 'samples/dl/lp_ref_1.webp'
    if not source.exists():
        pytest.skip('Local catalogue fixture not installed')

    def forbidden(*a, **kw):
        pytest.fail('No ML inference is allowed in knob tests')

    monkeypatch.setattr('gbcv.segment.probability', forbidden)
    monkeypatch.setattr('gbcv.sam.Session', forbidden)
    monkeypatch.setattr(tr, '_sam_parts', lambda *a, **kw: [])
    rgba = load_image(source, mode='RGBA')
    with patch.object(knobs, 'recover', return_value=[]):
        baseline = tr.trace(rgba)
    detected = knobs.recover(baseline, float(baseline.board[1:3, 0].max()))
    assert len(detected) == 2
    existing = detected[0].copy()
    monkeypatch.setattr(tr, '_sam_parts', lambda *a, **kw: [existing])
    monkeypatch.setattr(tr, '_real_parts', lambda t, parts: parts)
    with patch.object(knobs, 'recover', wraps=knobs.recover) as recover:
        result = tr.trace(rgba)  # no sam_debug or diagnostic runner
    recover.assert_called_once()
    assert recover.call_args.args[0] is result
    assert result.parts[0] is existing
    np.testing.assert_array_equal(existing, detected[0])
    assert len(result.parts) == 2  # existing SAM knob + one recovered knob
    complete = tr.svg(result, 'png')
    result.parts = [existing]
    assert complete.count('<path') == tr.svg(result, 'png').count('<path')
    assert complete.count('<circle') > tr.svg(result, 'png').count('<circle')
