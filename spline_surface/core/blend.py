"""Bridge the gap between two separate patch edges with a blend patch.

Across the gap the blend is a degree-5 Bezier: the first rows follow edge A's
Taylor expansion and the last rows edge B's, so both seams are G1 or G2 over
their full length (no collapsed corner, unlike Corner Fill).
"""
import numpy as np

from .corner import _refit, _side
from .fill import _structure
from .nurbs import basis_matrix, fit_polyline as _fit_polyline

DEGREE = 5


def blend_net(Pa, orders_a, edge_a, Pb, orders_b, edge_b,
              continuity='G2', tension=1.0, flip=False):
    """Net of the patch bridging edge_a of Pa and edge_b of Pb.

    Edge ends are paired so the bridge does not twist; `flip` pairs them the
    other way. Returns (net, orders).
    """
    if continuity not in ('G0', 'G1', 'G2'):
        raise ValueError("continuity must be G0, G1 or G2")
    sa, ka, sign_a = _side(np.asarray(Pa, float), orders_a, edge_a)
    sb, kb, _ = _side(np.asarray(Pb, float), orders_b, edge_b)
    Ea, Eb = sa[0], sb[0]
    straight = np.linalg.norm(Ea[0] - Eb[0]) + np.linalg.norm(Ea[-1] - Eb[-1])
    crossed = np.linalg.norm(Ea[0] - Eb[-1]) + np.linalg.norm(Ea[-1] - Eb[0])
    if (crossed < straight) != bool(flip):
        sb = [R[::-1] for R in sb]

    na, nb = len(sa[0]), len(sb[0])
    n, k = _structure(na, ka, nb, kb, min_points=2)
    (Ea, Ta, Aa), (Eb, Tb, Ab) = _refit(sa, na, ka, n, k), _refit(sb, nb, kb, n, k)

    width = np.linalg.norm(Ea - Eb, axis=1).mean()
    if width < 1e-9:
        raise ValueError("the edges coincide; use Match Edge instead")
    d = DEGREE
    rows = [None] * (d + 1)
    rows[0], rows[d] = Ea, Eb
    if continuity == 'G0':
        for i in range(1, d):
            rows[i] = Ea + (Eb - Ea) * (i / d)
    else:
        # Constant reach per side keeps the Taylor rows exact (pure rescaling).
        ha = tension * width / max(np.linalg.norm(Ta, axis=1).mean(), 1e-12)
        hb = tension * width / max(np.linalg.norm(Tb, axis=1).mean(), 1e-12)
        rows[1] = Ea + ha * Ta / d
        rows[d - 1] = Eb + hb * Tb / d
        if continuity == 'G2':
            rows[2] = 2 * rows[1] - Ea + ha * ha * Aa / (d * (d - 1))
            rows[d - 2] = 2 * rows[d - 1] - Eb + hb * hb * Ab / (d * (d - 1))
        else:
            rows[2] = rows[1] + (rows[d - 1] - rows[1]) / 3.0
            rows[d - 2] = rows[1] + 2.0 * (rows[d - 1] - rows[1]) / 3.0
    net = np.stack(rows, axis=1)
    # At edge A the bridge's normal is cross(along, +T_a); see corner_net.
    if sign_a < 0:
        net = net[:, ::-1]
    return net, (k, d + 1)


def blend_curve_net(Pa, orders_a, edge_a, curve_pts, continuity='G2', tension=1.0,
                    flip=False, curve_tangent='SMOOTH', min_spans=8):
    """Net of a patch from edge_a of Pa to a free curve (a polyline).

    The patch side joins with G1/G2; the curve side only touches the curve
    (G0). curve_tangent: 'SMOOTH' lets the patch arrive naturally, 'STRAIGHT'
    leaves the curve heading straight for the patch edge.
    Returns (net, orders, curve fit error).
    """
    if continuity not in ('G0', 'G1', 'G2'):
        raise ValueError("continuity must be G0, G1 or G2")
    sa, ka, sign_a = _side(np.asarray(Pa, float), orders_a, edge_a)
    C = np.asarray(curve_pts, float)
    Ea = sa[0]
    straight = np.linalg.norm(Ea[0] - C[0]) + np.linalg.norm(Ea[-1] - C[-1])
    crossed = np.linalg.norm(Ea[0] - C[-1]) + np.linalg.norm(Ea[-1] - C[0])
    if (crossed < straight) != bool(flip):
        C = C[::-1]

    # Refine edge A's structure (nested, so the edge stays exact) to give the
    # curve enough freedom to be followed closely.
    na = len(Ea)
    spans = na - ka + 1
    spans *= max(1, -(-min_spans // spans))
    n = spans + ka - 1
    Ea, Ta, Aa = _refit(sa, na, ka, n, ka)
    Ec, err = _fit_polyline(C, n, ka)

    width = np.linalg.norm(Ea - Ec, axis=1).mean()
    if width < 1e-9:
        raise ValueError("the curve lies on the edge")
    d = DEGREE
    rows = [None] * (d + 1)
    rows[0], rows[d] = Ea, Ec
    if continuity == 'G0':
        start = 1
    else:
        ha = tension * width / max(np.linalg.norm(Ta, axis=1).mean(), 1e-12)
        rows[1] = Ea + ha * Ta / d
        start = 2
        if continuity == 'G2':
            rows[2] = 2 * rows[1] - Ea + ha * ha * Aa / (d * (d - 1))
            start = 3
    if curve_tangent == 'STRAIGHT':
        # Leave the curve along the straight line towards the patch edge.
        for i in range(max(start, 3), d):
            rows[i] = Ec + (Ea - Ec) * ((d - i) / d)
    anchor = start - 1
    free = [i for i in range(1, d) if rows[i] is None]
    for i in free:
        # Remaining rows: even spacing between the last fixed row and the curve side.
        hi = min([j for j in range(i + 1, d + 1) if rows[j] is not None])
        f = (i - anchor) / (hi - anchor)
        rows[i] = rows[anchor] + (rows[hi] - rows[anchor]) * f
    net = np.stack(rows, axis=1)
    if sign_a < 0:
        net = net[:, ::-1]
    return net, (ka, d + 1), err
