"""Read-only experiment checks. No bridge algorithm is enabled in production."""
import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('investigate_bridge')


def fixture_photo(metal=True):
    image = np.full((400,500,3),[160,80,20],np.uint8)
    if metal:
        cv2.rectangle(image,(200,130),(224,270),(150,150,150),-1)
        cv2.ellipse(image,(212,130),(12,12),0,180,360,(150,150,150),-1)
        cv2.ellipse(image,(212,270),(12,12),0,0,180,(150,150,150),-1)
    for y in (155,175,195,215,235):
        cv2.line(image,(30,y),(470,y),(220,220,220),2)
    return image,np.full(image.shape[:2],255,np.uint8)


def test_opening_removes_longitudinal_strings_not_curved_endcaps(probe):
    photo,mask = fixture_photo()
    shapes,_ = probe.metal_contours(photo,mask,25)
    assert len(shapes) == 1
    c = shapes[0]
    assert np.ptp(c[:,0]) < 15
    assert np.ptp(c[:,1]) > 75
    assert len(np.unique(c[:,0])) > 10  # actual rounded boundary, not a box
    crossing = probe.string_crossings(photo,c)
    assert len(crossing['matched']) == 5


def test_strings_alone_do_not_propose_a_bridge(probe):
    photo,mask = fixture_photo(metal=False)
    assert not probe.metal_contours(photo,mask,25)[0]


def test_missing_or_merged_boundary_is_not_reported_stable(probe):
    c = np.array([[0,0],[10,0],[10,80],[0,80]],float)
    assert probe.stability95(c,([],))[0] is None
    assert probe.stability95(c,([c],))[0] == 0
    assert probe.stability95(c,([c+[50,0]],))[0] > 1.5


@pytest.mark.parametrize('failure',['unstable','few_strings','weak_edges','irregular_sides'])
def test_confidence_rejects_each_missing_piece_of_evidence(probe,failure):
    stable = [0.,0.]
    crossing = dict(matched=[-25,-15,-5,5,15,25],span_share=.6)
    boundary = dict(edge_support=1.,weakest_boundary_eighth=1.,side_line_residual90=[0.,0.])
    assert not probe.rejection_reasons(stable,crossing,boundary)
    if failure == 'unstable':
        stable[0] = 4.
    elif failure == 'few_strings':
        crossing['matched'] = [-5,5]
    elif failure == 'weak_edges':
        boundary['weakest_boundary_eighth'] = .2
    else:
        boundary['side_line_residual90'] = [0.,4.]
    assert probe.rejection_reasons(stable,crossing,boundary)
