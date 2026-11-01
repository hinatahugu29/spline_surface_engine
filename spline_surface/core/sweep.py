"""Sweep: carry the profile of one patch edge along the edge of another patch.

The new surface is S(u, v) = profile(u) + rail(v) - rail(start). Both curves are
clamped B-splines whose basis functions sum to one, so the control net is simply
profile point i + rail offset j: the sweep is exact, no fitting involved.
The profile stays where it is at the rail's end nearest to it and follows the
rail from there (the profile is moved, not turned).

With `straight`, the profile's other end does not follow the rail but runs
straight out of the source patch; the profile is stretched between the two.
Writing the straight run as another offset curve A(v) (a line, parametrised like
the rail), S = profile(u) + A(v) + s(u) (rail(v) - rail(0) - A(v)) with s running from 0
at the straight end to 1 at the rail end. s is linear in u, so it is exact too:
a clamped B-spline reproduces a linear function with its Greville abscissae.
"""
import numpy as np

from .extend import edge_derivatives, edge_frame, frame_orders
from .linked import edge_order, edge_row
from .nurbs import clamped_knots


def greville(n, k):
    """Greville abscissae of a clamped uniform B-spline with n points and order k."""
    t = clamped_knots(n, k)
    return np.array([t[j + 1:j + k].mean() for j in range(n)])


def straight_run(origin, out, target):
    """How far the ray origin + t*out (t > 0) must run to reach `target`.

    target is ('plane', axis, value): the plane coordinate[axis] == value, or
    ('line', points): the point of the polyline the ray passes closest to.
    """
    origin, out = np.asarray(origin, float), np.asarray(out, float)
    if target[0] == 'plane':
        axis, value = int(target[1]), float(target[2])
        if abs(out[axis]) < 1e-9:
            raise ValueError("the straight end runs parallel to that plane")
        t = (value - origin[axis]) / out[axis]
        if t <= 1e-9:
            raise ValueError("that plane is behind the straight end")
        return float(t)
    pts = np.asarray(target[1], float)
    best = None
    for a, b in zip(pts[:-1], pts[1:]):
        e = b - a
        w = origin - a
        de, ee = float(np.dot(out, e)), float(np.dot(e, e))
        if ee < 1e-18:
            continue
        det = de * de - ee
        s = 0.0 if abs(det) < 1e-12 else (de * float(np.dot(out, w)) - float(np.dot(e, w))) / det
        s = min(1.0, max(0.0, s))
        q = a + s * e
        t = float(np.dot(out, q - origin))
        if t <= 1e-9:
            continue
        d = float(np.linalg.norm(origin + t * out - q))
        if best is None or d < best[0]:
            best = (d, t)
    if best is None:
        raise ValueError("the target is behind the straight end")
    return best[1]


def sweep_net(P, orders_p, edge_p, R, orders_r, edge_r, flip=False, straight=False, length=0.0,
              target=None):
    """Net and orders of the profile `edge_p` of net P swept along `edge_r` of net R.

    The rail is walked from the end nearest to the profile; flip walks it from the other end.
    straight: the profile's other end runs straight out of P instead, for `length`
    (0: as far as the rail's end has gone in that direction, so the two ends line up).
    target ('plane', axis, value) or ('line', points) instead stops it where it reaches that.
    Returns (net, orders): u runs along the profile, v along the rail.
    """
    P, R = np.asarray(P, float), np.asarray(R, float)
    prof = P[edge_row(P, edge_p)]
    rail = R[edge_row(R, edge_r)]
    ends = [(np.linalg.norm(prof[i] - rail[j]), i, j) for i in (0, -1) for j in (0, -1)]
    _, near_i, near_j = min(ends)
    if (near_j == -1) != bool(flip):
        rail = rail[::-1]
    chord = float(np.linalg.norm(rail[-1] - rail[0]))
    if chord < 1e-12:
        raise ValueError("the rail edge has no length")
    offsets = rail - rail[0]
    orders = (edge_order(orders_p, edge_p), edge_order(orders_r, edge_r))
    if not straight:
        return prof[:, None, :] + offsets[None, :, :], orders
    k_along, k_across = frame_orders(orders_p, edge_p)
    C, _ = edge_frame(P, edge_p)
    T, _ = edge_derivatives(C, k_across)             # outward across-edge derivative, per profile point
    far = -1 if near_i == 0 else 0
    out = T[far]
    if np.linalg.norm(out) < 1e-12:
        raise ValueError("the profile edge has no outward direction here")
    out = out / np.linalg.norm(out)
    run = length
    if target is not None:
        run = straight_run(prof[far], out, target)
    elif run <= 0.0:                                   # line the straight end up with the rail's end
        run = float(np.dot(rail[-1] - rail[0], out))
        if run < 1e-9 * chord:                       # the rail runs sideways or back: use its chord
            run = chord
    line = run * greville(len(rail), orders[1])[:, None] * out[None, :]      # A(v) - A(0)
    s = greville(len(prof), orders[0])
    if near_i == 0:                                  # s = 1 at the rail end
        s = 1.0 - s
    net = (prof[:, None, :] + line[None, :, :]
           + s[:, None, None] * (offsets - line)[None, :, :])
    return net, orders
