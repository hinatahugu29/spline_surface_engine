"""Starting shapes for car body panels.

Each preset returns named 6 x 6 Bezier nets in a car-like frame: x is across
the car (the centre line at x = 0), y runs front to back, z is up. The shapes
are plain starting points to push and pull into the design, not finished
panels.
"""
import numpy as np

KINDS = ('HOOD', 'ROOF', 'DOOR', 'HOOD_FENDER')
POINTS = 6


def _grid():
    g = np.linspace(0.0, 1.0, POINTS)
    return g[:, None] + 0 * g, 0 * g[:, None] + g          # U (along y), V (across x)


def _crowned(length, width, crown, half, long_crown):
    """A panel crowned across (and a little along): u runs along y, v across x."""
    U, V = _grid()
    along = 1.0 - (2.0 * U - 1.0) ** 2
    if half:                                               # centre line at x = 0, v = 0
        x = 0.5 * width * V
        across = 1.0 - V ** 2
    else:
        x = width * (V - 0.5)
        across = 1.0 - (2.0 * V - 1.0) ** 2
    z = crown * across + long_crown * crown * along
    return np.stack([x, length * U, z], -1)


def hood(length=1.4, width=1.6, crown=0.12, half=False):
    return [("Hood", _crowned(length, width, crown, half, 0.3))]


def roof(length=1.2, width=1.3, crown=0.08, half=False):
    return [("Roof", _crowned(length, width, crown, half, 0.5))]


def door(length=1.1, height=0.8, bulge=0.06, tumblehome=0.05):
    """An upright side panel: u runs along y, v up z, bulging outward (+x) and leaning in at the top."""
    U, V = _grid()
    x = bulge * np.sin(np.pi * V) * (1.0 - (2.0 * U - 1.0) ** 2) - tumblehome * V * V
    return [("Door", np.stack([x, length * U, height * V], -1))]


def hood_fender(length=1.4, width=1.6, crown=0.12, depth=0.45, bulge=0.04):
    """Half a hood with a fender wall hanging off its side edge (join them with Fillet)."""
    (_, H), = hood(length, width, crown, half=True)
    U, V = _grid()
    side_z = H[:, -1, 2]                                   # the hood's side edge height
    x = H[0, -1, 0] + bulge * np.sin(np.pi * V) * (1.0 - (2.0 * U - 1.0) ** 2)
    z = side_z[:, None] - depth * V
    return [("Hood", H), ("Fender", np.stack([x, length * U, z], -1))]


def build(kind, **sizes):
    """Named nets for preset `kind`; sizes override the defaults (unknown names are ignored)."""
    fn = {'HOOD': hood, 'ROOF': roof, 'DOOR': door, 'HOOD_FENDER': hood_fender}[kind]
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    return fn(**{k: v for k, v in sizes.items() if k in names})
