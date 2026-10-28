"""Fill the V-shaped notch between two patch edges that meet at a corner.

The fill is a three-sided patch: one side lies on each picked edge, the third
side is the open mouth of the notch, and the shared corner is a collapsed
edge. Across each picked edge the fill follows the neighbour's Taylor
expansion, so it joins with G1 or G2; the collapsed corner itself is G0.
"""
import numpy as np

from .extend import edge_derivatives, edge_frame, frame_orders
from .nurbs import basis_matrix

DEGREE = 5  # across the notch; 6 rows hold both sides' G2 conditions


def _side(P, orders, edge):
    """Edge row, outward first and second derivative rows, along order, normal sign."""
    C, keeps = edge_frame(P, edge)
    k_along, k_across = frame_orders(orders, edge)
    T, A = edge_derivatives(C, k_across)
    return [C[:, -1], T, A], k_along, (1.0 if keeps else -1.0)


def _refit(rows, n, k, n_new, k_new, samples=200):
    """Re-express along-edge rows in another clamped structure (least squares)."""
    if (n, k) == (n_new, k_new):
        return rows
    t = np.linspace(0.0, 1.0, samples)
    B, B_new = basis_matrix(n, k, t), basis_matrix(n_new, k_new, t)
    out = []
    for R in rows:
        F = np.linalg.lstsq(B_new, B @ R, rcond=None)[0]
        F[0], F[-1] = R[0], R[-1]
        out.append(F)
    return out


def common_structure(na, ka, nb, kb, max_spans=64):
    """Along-edge (points, order) both edges fit into.

    Uniform knots with S spans nest inside those with any multiple of S, so for
    equal orders the lcm of the span counts holds both edges exactly. Otherwise
    take the richer structure and accept a small least-squares error.
    """
    if ka == kb:
        spans = np.lcm(na - ka + 1, nb - kb + 1)
        if spans <= max_spans:
            # Extra spans (still nested) give the tapered rows room to fit.
            spans *= max(1, min(8, max_spans // spans, 60 // spans))
            return int(spans + ka - 1), ka
    return max((na, ka), (nb, kb))


def corner_net(Pa, orders_a, edge_a, Pb, orders_b, edge_b,
               continuity='G2', tension=1.0, tol=1e-3):
    """Net of the patch filling between edge_a of Pa and edge_b of Pb.

    Returns (net, orders). The edges must share an endpoint within `tol`.
    """
    if continuity not in ('G0', 'G1', 'G2'):
        raise ValueError("continuity must be G0, G1 or G2")
    sa, ka, sign_a = _side(np.asarray(Pa, float), orders_a, edge_a)
    sb, kb, sign_b = _side(np.asarray(Pb, float), orders_b, edge_b)

    # Orient both edges to start at the shared corner.
    ends = [(np.linalg.norm(sa[0][i] - sb[0][j]), i, j) for i in (0, -1) for j in (0, -1)]
    gap, ia, ib = min(ends)
    if gap > tol:
        raise ValueError("the edges do not share a corner (closest ends %.4g apart)" % gap)
    if ia == -1:
        sa, sign_a = [R[::-1] for R in sa], -sign_a
    if ib == -1:
        sb, sign_b = [R[::-1] for R in sb], -sign_b
    if np.linalg.norm(sa[0][-1] - sb[0][-1]) < tol:
        raise ValueError("the edges meet at both ends; this is a closed hole, not a corner")

    na, nb = len(sa[0]), len(sb[0])
    n, k = common_structure(na, ka, nb, kb)
    sa, sb = _refit(sa, na, ka, n, k), _refit(sb, nb, kb, n, k)
    (Ea, Ta, Aa), (Eb, Tb, Ab) = sa, sb

    d = DEGREE
    rows = [None] * (d + 1)
    rows[0], rows[d] = Ea, Eb
    if continuity == 'G0':
        for i in range(1, d):
            rows[i] = Ea + (Eb - Ea) * (i / d)
    else:
        # Reach into the fill in proportion to the local notch width so the
        # rows taper towards the corner instead of overshooting it. The fill
        # is F(t, alpha(t) s) of each side's Taylor expansion: an exact
        # reparametrisation, evaluated densely and fitted back to the net.
        t = np.linspace(0.0, 1.0, 8 * n)
        B = basis_matrix(n, k, t)
        ea, ta, aa, eb, tb, ab = (B @ R for R in (Ea, Ta, Aa, Eb, Tb, Ab))
        w = np.linalg.norm(ea - eb, axis=1)[:, None]
        alpha_a = tension * w / np.maximum(np.linalg.norm(ta, axis=1), 1e-12)[:, None]
        alpha_b = tension * w / np.maximum(np.linalg.norm(tb, axis=1), 1e-12)[:, None]
        r1a, r1b = ea + alpha_a * ta / d, eb + alpha_b * tb / d
        if continuity == 'G2':
            r2a = 2 * r1a - ea + alpha_a ** 2 * aa / (d * (d - 1))
            r2b = 2 * r1b - eb + alpha_b ** 2 * ab / (d * (d - 1))
        else:
            r2a = r1a + (r1b - r1a) / 3.0
            r2b = r1a + 2.0 * (r1b - r1a) / 3.0
        # Weight by 1/width: near the corner the reach is tiny, so the fit
        # must be relatively tight there or the seam normal tilts.
        wt = 1.0 / np.maximum(w, 0.01 * w.max())
        fit = lambda y: np.linalg.lstsq(B * wt, y * wt, rcond=None)[0]
        rows[1], rows[2], rows[d - 2], rows[d - 1] = fit(r1a), fit(r2a), fit(r2b), fit(r1b)
    net = np.stack(rows, axis=1)  # (n along, d + 1 across)
    net[0, :] = 0.5 * (Ea[0] + Eb[0])  # collapse the corner
    # Normals: at the a side the fill's normal is cross(along, +T_a) and the
    # source's is sign_a times that, so flip the fill when sign_a is negative.
    if sign_a < 0:
        net = net[:, ::-1]
    return net, (k, d + 1)
