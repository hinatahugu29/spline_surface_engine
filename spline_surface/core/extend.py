"""Grow a new patch off an edge of an existing one with G1/G2 continuity.

Patches are a control net plus `orders` = (order_u, order_v) of a clamped
uniform B-spline (Blender NURBS with Endpoint on). `orders=None` means a
single-span Bezier patch (order = point count in both directions).
"""
import numpy as np

from .bezier import elevate_curve
from .nurbs import basis_matrix

# edge -> (transform to canonical frame, keeps orientation, swaps axes)
# Canonical frame: axis 0 runs along the edge, the edge is the last row of axis 1.
_EDGES = {
    'V1': (lambda P: P, True, False),
    'V0': (lambda P: P[:, ::-1], False, False),
    'U1': (lambda P: P.transpose(1, 0, 2), False, True),
    'U0': (lambda P: P.transpose(1, 0, 2)[:, ::-1], True, True),
}
EDGES = tuple(_EDGES)


def bezier_orders(P):
    return (P.shape[0], P.shape[1])


def edge_frame(P, edge):
    """Canonical view of net P for `edge` plus whether orientation is kept."""
    fn, keeps, _ = _EDGES[edge]
    return fn(np.asarray(P, dtype=float)), keeps


def frame_orders(orders, edge):
    """(along-edge order, across order) in the canonical frame of `edge`."""
    return (orders[1], orders[0]) if _EDGES[edge][2] else tuple(orders)


def edge_derivatives(C, k_across, at_end=True):
    """Rows of the first and second across-derivative at the edge row of C."""
    n = C.shape[1]
    t = 1.0 if at_end else 0.0
    d1 = basis_matrix(n, k_across, t, 1)[0]
    d2 = basis_matrix(n, k_across, t, 2)[0]
    return np.einsum("j,ijd->id", d1, C), np.einsum("j,ijd->id", d2, C)


def length_handle(P, edge, orders=None):
    """Where a drag handle for extend_net's `length` sits on the source edge.

    Returns (base, direction, scale): the middle of `edge`, the unit
    direction the extension grows in there, and world distance per unit of
    `length` (the tip moves by length * scale; exact for G1, G2 adds a
    curvature term).
    """
    P = np.asarray(P, dtype=float)
    orders = orders or bezier_orders(P)
    C, _ = edge_frame(P, edge)
    k_along, k_across = frame_orders(orders, edge)
    T, _ = edge_derivatives(C, k_across)
    mid = basis_matrix(C.shape[0], k_along, 0.5)[0]
    base, t = mid @ C[:, -1], mid @ T
    scale = float(np.linalg.norm(t))
    return base, t / max(scale, 1e-12), scale


def extend_net(P, edge, length=0.5, continuity='G2', degree=3, orders=None):
    """Patch continuing `P` across `edge`. Returns (net, orders).

    Along every edge column the extension follows the Taylor expansion of the
    source at the edge, scaled to `length` (fraction of the source's parameter
    span): linear for G1 (tangent match), quadratic for G2 (tangent and
    curvature match). The new patch keeps the source's along-edge structure,
    shares the edge exactly and its normals point the same way.
    """
    if continuity not in ('G1', 'G2'):
        raise ValueError("continuity must be G1 or G2")
    P = np.asarray(P, dtype=float)
    orders = orders or bezier_orders(P)
    C, keeps = edge_frame(P, edge)
    k_along, k_across = frame_orders(orders, edge)
    if continuity == 'G2' and k_across < 3:
        raise ValueError("G2 needs a source of degree >= 2 across the edge")
    h = float(length)
    Q0 = C[:, -1]
    T, A = edge_derivatives(C, k_across)
    rows = [Q0, Q0 + h * T / 2.0, Q0 + h * T]
    if continuity == 'G2':
        rows[2] = rows[2] + 0.5 * h * h * A
    quad = np.stack(rows, axis=1)  # (n_along, 3, 3): quadratic across the edge
    k = max(int(degree), 2)
    Q = np.stack([elevate_curve(col, k) for col in quad], axis=0)
    return (Q if keeps else Q[::-1]), (k_along, k + 1)
