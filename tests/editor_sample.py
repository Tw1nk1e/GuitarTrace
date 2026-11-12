"""Small synthetic drawing: no models and no private photographs."""
import numpy as np

from gbcv import knobs, trace as tr
from gbcv.frets import FretFit, Ruler


def sample(perspective=0.0):
    theta = np.linspace(0, 2*np.pi, 40, endpoint=False)
    control = np.c_[30 + 6*np.cos(theta), 15 + 6*np.sin(theta)]
    inner = np.c_[30 + 3*np.cos(theta), 15 + 3*np.sin(theta)]
    t = tr.Trace(
        outline=np.array([[0,-35],[50,-35],[60,-12],[140,-12],[150,-20],[180,-20],
                          [180,20],[150,20],[140,12],[60,12],[50,35],[0,35]], float),
        nut_x=140, scale_length=100,
        frets=[(80,81,10),(100,101,10),(120,121,10)],
        board=np.array([[140,-10],[60,-10],[60,10],[140,10]], float),
        posts=[(160,0,4)],
        keys=[np.array([[164,-20],[164,-28],[174,-28],[174,-20]], float)],
        parts=[np.array([[20,-20],[40,-20],[40,-10],[20,-10]], float), control, inner],
        knobs=[knobs.Control(control, inner)],
        ruler=Ruler(FretFit(3,500,perspective,[],0,3), (.03,150), 150, 3),
        mask=np.ones((320,640), np.uint8),
        photo=np.full((320,640,3), 235, np.uint8),
    )
    return t
