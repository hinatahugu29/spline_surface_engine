"""Split a patch into two along an iso-line.

Blender NURBS only have uniform clamped knots, so a piece is exact when the
cut is on a span boundary (or anywhere on a single-span Bezier patch): its
knots are then uniform again. Elsewhere the pieces are least-squares fits;
the cut row is pinned on both pieces, so they always share the seam exactly.
"""
import numpy as np

from .nurbs import basis_matrix


def _end_rows(B0, D1, D2, P0, n_pin):
    """Rows next to a clamped end that reproduce derivatives D1, D2 there.

    B0 = (first-derivative, second-derivative) basis rows at that end, ordered
    from the end inwards.
    """
    d1, d2 = B0
    rows = [P0]
    if n_pin > 1:
        rows.append((D1 - d1[0] * P0) / d1[1])
    if n_pin > 2:
        rows.append((D2 - d2[0] * P0 - d2[1] * rows[1]) / d2[2])
    return rows


def _fit_piece(P, k, t0, t1, n_new):
    """Rows (n_new, ...) of the part of curve-rows P (n, ...) over [t0, t1].

    Position and first two derivatives are pinned at both ends (so a cut is
    G2 by construction); the rows in between are a least-squares fit.
    """
    n = P.shape[0]
    flat = P.reshape(n, -1)
    m = max(8 * max(n, n_new), 64)
    s = np.linspace(0.0, 1.0, m)
    C = basis_matrix(n, k, t0 + s * (t1 - t0)) @ flat
    h = t1 - t0
    ends = []
    for t, s_end in ((t0, 0.0), (t1, 1.0)):
        D1 = h * (basis_matrix(n, k, t, 1)[0] @ flat)
        D2 = h * h * (basis_matrix(n, k, t, 2)[0] @ flat)
        Bd1 = basis_matrix(n_new, k, s_end, 1)[0]
        Bd2 = basis_matrix(n_new, k, s_end, 2)[0]
        ends.append((D1, D2, Bd1, Bd2))
    n_pin = min(3, k - 1, n_new // 2)
    (a1, a2, ba1, ba2), (b1, b2, bb1, bb2) = ends
    head = _end_rows((ba1, ba2), a1, a2, C[0], n_pin)
    tail = _end_rows((bb1[::-1], bb2[::-1]), b1, b2, C[-1], n_pin)[::-1]
    B = basis_matrix(n_new, k, s)
    Q = np.zeros((n_new, flat.shape[1]))
    Q[:n_pin], Q[n_new - n_pin:] = head, tail
    free = slice(n_pin, n_new - n_pin)
    if n_new > 2 * n_pin:
        rhs = C - B[:, :n_pin] @ Q[:n_pin] - B[:, n_new - n_pin:] @ Q[n_new - n_pin:]
        Q[free] = np.linalg.lstsq(B[:, free], rhs, rcond=None)[0]
    err = float(np.linalg.norm((B @ Q - C).reshape(m, -1, 3), axis=2).max())
    return Q.reshape((n_new,) + P.shape[1:]), err


def split_net(net, orders, direction, t, snap=True):
    """Split at u = t ('U') or v = t ('V').

    Returns (net_a, net_b, orders_a, orders_b, t_used, max_error); piece a
    holds the parameter range [0, t], piece b [t, 1].
    """
    P = np.asarray(net, dtype=float)
    if direction == 'V':
        P = P.transpose(1, 0, 2)
        orders = (orders[1], orders[0])
    n, k = P.shape[0], orders[0]
    spans = n - k + 1
    if spans > 1 and snap:
        t = round(t * spans) / spans
    if not 0.0 < t < 1.0:
        raise ValueError("the cut must lie inside the patch")
    j = t * spans
    if spans == 1:
        na = nb = n
    elif abs(j - round(j)) < 1e-9:
        na, nb = int(round(j)) + k - 1, spans - int(round(j)) + k - 1
    else:
        # Off a span boundary: one extra span per piece keeps the fit close.
        na = int(np.ceil(j)) + 1 + k - 1
        nb = int(np.ceil(spans - j)) + 1 + k - 1
    A, ea = _fit_piece(P, k, 0.0, t, na)
    B, eb = _fit_piece(P, k, t, 1.0, nb)
    if direction == 'V':
        A, B = A.transpose(1, 0, 2), B.transpose(1, 0, 2)
        oa = ob = (orders[1], k)
    else:
        oa = ob = (k, orders[1])
    return A, B, oa, ob, t, max(ea, eb)


def iso_line(net, orders, direction, t, samples=48):
    """World points of the iso-line u = t ('U') or v = t ('V')."""
    from .nurbs import eval_points
    s = np.linspace(0.0, 1.0, samples)
    c = np.full(samples, t)
    return eval_points(net, orders, c, s) if direction == 'U' else eval_points(net, orders, s, c)
