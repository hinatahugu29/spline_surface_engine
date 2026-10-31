"""Flange: a wall folded off an edge of a patch.

The wall is a ruled strip along the edge. It leaves each control point of the
edge in a direction turned `angle` degrees away from the patch's own
continuation, towards the back of the patch (against its normal). Rounding the
fold is left to Fillet; this only makes the sharp wall.
"""
import numpy as np

from .extend import bezier_orders, edge_derivatives, edge_frame, frame_orders

# sign of cross(along, outward) relative to the patch normal cross(du, dv)
_NORMAL_SIGN = {'V1': 1.0, 'V0': -1.0, 'U1': -1.0, 'U0': 1.0}


def flange_net(P, edge, length=0.3, angle=90.0, flip=False, orders=None, offset=0.0):
    """Wall off `edge` of net P. Returns (net, orders); the edge is the wall's v = 0 row.

    angle: 0 continues the patch straight on, 90 folds square to it, 180 folds
    back over it. length: wall depth in world units (along the fold direction).
    flip: fold towards the front (along the normal) instead of the back.
    offset: radius of a fold rounded *outside* the edge. The wall then starts where
    that rounding ends, so the patch itself is not cut back (see Flange's Outside).
    """
    if not 0.0 < angle < 180.0:
        raise ValueError("angle must be between 0 and 180 degrees")
    P = np.asarray(P, dtype=float)
    orders = orders or bezier_orders(P)
    C, _ = edge_frame(P, edge)
    _, k_across = frame_orders(orders, edge)
    k_along = frame_orders(orders, edge)[0]
    T, _ = edge_derivatives(C, k_across)          # outward across-edge derivative per control point
    E = C[:, -1]                                  # the edge's control points
    A = np.gradient(E, axis=0)
    if np.linalg.norm(A, axis=1).max() < 1e-12:
        raise ValueError("the edge has no length")
    n = np.cross(A, T) * _NORMAL_SIGN[edge]       # patch normal at each control point
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    t = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    a = np.radians(angle)
    back = n if flip else -n
    d = np.cos(a) * t + np.sin(a) * back
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    start = E
    if offset > 0.0:                               # where an arc of that radius, leaving E along t, ends
        start = E + offset * (np.sin(a) * t + (1.0 - np.cos(a)) * back)
    net = np.stack([start, start + length * d], axis=1)   # (along, 2, 3): u along the edge, v out
    return net, (k_along, 2)
