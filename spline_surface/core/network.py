"""Curve network: skin a grid of crossing curves with smoothly joined patches.

The curves fall into two families (every curve of one family crosses every
curve of the other). Each grid cell becomes one patch whose sides are the
curve segments between crossings. Across an inner curve both neighbouring
cells take their cross tangent (and curvature) from the *other* family's
curves at the crossings, blended along the segment: the two cells see the
same direction and curvature, so they meet with G1/G2. One twist per
crossing, shared by all sides meeting there, keeps the corners consistent.
Curves that miss each other slightly are first bent through the crossings.
The outer border only fixes position.
"""
import numpy as np

from .fill import STAR_POINTS, _orient, quad_net
from .split import _fit_piece
from .nurbs import basis_matrix, fit_polyline

CURVE_SPANS = 12   # fitted representation of each input curve
SEG_POINTS = 9     # quintic B-spline (4 spans) per side segment
SEG_ORDER = 6


class Curve:
    """A fitted B-spline through a polyline, parametrised by chord length."""

    def __init__(self, pts, spans=CURVE_SPANS, k=6):
        self.k = k
        self.P, self.err = fit_polyline(pts, spans + k - 1, k)

    def __call__(self, t, d=0):
        return basis_matrix(len(self.P), self.k, np.atleast_1d(t), d) @ self.P

    def reverse(self):
        self.P = self.P[::-1].copy()

    def pass_through(self, ts, targets, samples=400):
        """Bend the curve smoothly so that curve(ts[k]) == targets[k].

        The offsets are spread with Gaussian bumps (smooth, so tangents and
        curvature stay consistent) and the result is refitted in the same
        parameter, so crossing parameters stay valid.
        """
        ts = np.asarray(ts, float)
        delta = np.asarray(targets, float) - self(ts)
        if np.abs(delta).max() < 1e-12:
            return
        sigma = max(np.diff(np.sort(ts)).min() if len(ts) > 1 else 1.0, 0.05)
        K = np.exp(-((ts[:, None] - ts[None]) / sigma) ** 2)
        w = np.linalg.solve(K + 1e-9 * np.eye(len(ts)), delta)
        t = np.linspace(0.0, 1.0, samples)
        bent = self(t) + np.exp(-((t[:, None] - ts[None]) / sigma) ** 2) @ w
        B = basis_matrix(len(self.P), self.k, t)
        self.P = np.linalg.lstsq(B, bent, rcond=None)[0]


def _closest(a, b, samples=400):
    """(t_a, t_b, distance) of the closest approach between two curves."""
    t = np.linspace(0.0, 1.0, samples)
    A, B = a(t), b(t)
    D = np.linalg.norm(A[:, None] - B[None], axis=2)
    i, j = np.unravel_index(np.argmin(D), D.shape)
    ta, tb = t[i], t[j]
    for _ in range(20):  # Gauss-Newton on |a(ta) - b(tb)|^2
        r = a(ta)[0] - b(tb)[0]
        J = np.stack([a(ta, 1)[0], -b(tb, 1)[0]], axis=1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        ta, tb = np.clip(ta + step[0], 0, 1), np.clip(tb + step[1], 0, 1)
        if np.abs(step).max() < 1e-12:
            break
    return float(ta), float(tb), float(np.linalg.norm(a(ta)[0] - b(tb)[0]))


def _families(n, crosses):
    """2-colour the crossing graph. Returns the colour of each curve."""
    colour = [None] * n
    for start in range(n):
        if colour[start] is not None:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            c = stack.pop()
            for d in range(n):
                if (c, d) in crosses:
                    if colour[d] is None:
                        colour[d] = 1 - colour[c]
                        stack.append(d)
                    elif colour[d] == colour[c]:
                        raise ValueError("curves of the same direction cross each other")
    return colour


def build_network(polylines, tol, continuity='G2'):
    """Patches skinning a curve network.

    Returns ([(i, j, net, orders)], info) with cells indexed along the two
    families; info holds the curve fit error and the largest crossing gap.
    """
    if continuity not in ('G0', 'G1', 'G2'):
        raise ValueError("continuity must be G0, G1 or G2")
    curves = [Curve(p) for p in polylines]
    n = len(curves)
    crosses = {}
    for a in range(n):
        for b in range(a + 1, n):
            ta, tb, d = _closest(curves[a], curves[b])
            if d <= tol:
                crosses[(a, b)] = (ta, tb, d)
                crosses[(b, a)] = (tb, ta, d)
    colour = _families(n, crosses)
    U = [c for c in range(n) if colour[c] == 0]
    V = [c for c in range(n) if colour[c] == 1]
    if len(U) < 2 or len(V) < 2:
        raise ValueError("need at least two curves in each direction that cross each other")
    missing = [(u, v) for u in U for v in V if (u, v) not in crosses]
    if missing:
        raise ValueError("every curve must cross every curve of the other direction "
                         "(%d crossing(s) missing)" % len(missing))

    # Order each family along the other, and orient every curve to match.
    U.sort(key=lambda u: crosses[(V[0], u)][0])
    V.sort(key=lambda v: crosses[(U[0], v)][0])
    for fam, other in ((U, V), (V, U)):
        for c in fam:
            ts = [crosses[(c, o)][0] for o in other]
            if ts[0] > ts[-1]:
                curves[c].reverse()
                for o in other:
                    ta, tb, d = crosses[(c, o)]
                    crosses[(c, o)] = (1.0 - ta, tb, d)
                    crosses[(o, c)] = (tb, 1.0 - ta, d)
                ts = [crosses[(c, o)][0] for o in other]
            if np.any(np.diff(ts) <= 0):
                raise ValueError("the curves cross in a tangled order")

    nu, nv = len(U), len(V)
    # node[i][j]: crossing of v-curve V[i] (varying along u-index i) and u-curve U[j]
    node = [[0.5 * (curves[U[j]](crosses[(U[j], V[i])][0])[0]
                    + curves[V[i]](crosses[(V[i], U[j])][0])[0]) for j in range(nu)]
            for i in range(nv)]
    t_on_u = [[crosses[(U[j], V[i])][0] for i in range(nv)] for j in range(nu)]  # along U[j]
    t_on_v = [[crosses[(V[i], U[j])][0] for j in range(nu)] for i in range(nv)]  # along V[i]
    # Curves drawn by hand rarely meet exactly: bend each one through its
    # crossing nodes so both families agree there (tangents included).
    for j in range(nu):
        curves[U[j]].pass_through(t_on_u[j], [node[i][j] for i in range(nv)])
    for i in range(nv):
        curves[V[i]].pass_through(t_on_v[i], [node[i][j] for j in range(nu)])

    def segment(curve, t0, t1, p0, p1):
        """The curve between two crossings as a quintic B-spline.

        Position, tangent and curvature are pinned at both ends (so the pieces
        of one curve join smoothly at the nodes and match the cross tangents
        the transverse sides use there); ends snap onto the nodes.
        """
        E, _ = _fit_piece(curve.P, curve.k, t0, t1, SEG_POINTS)
        E[0], E[-1] = p0, p1
        return E

    # Tangents and curvature of both families at every node.
    du1 = [[curves[U[j]](t_on_u[j][i], 1)[0] for j in range(nu)] for i in range(nv)]
    du2 = [[curves[U[j]](t_on_u[j][i], 2)[0] for j in range(nu)] for i in range(nv)]
    dv1 = [[curves[V[i]](t_on_v[i][j], 1)[0] for j in range(nu)] for i in range(nv)]
    dv2 = [[curves[V[i]](t_on_v[i][j], 2)[0] for j in range(nu)] for i in range(nv)]

    def _diff(seq, ts, k):
        """d(seq)/dt at index k by (one-sided at the ends) finite differences."""
        a, b = max(k - 1, 0), min(k + 1, len(seq) - 1)
        return (seq[b] - seq[a]) / (ts[b] - ts[a])

    # One twist (mixed derivative, in curve parameters) per node, estimated from
    # both families and shared by every side meeting there: the cells around a
    # node then agree on how the surface turns, which keeps the corners G1.
    twist = [[0.5 * (_diff([dv1[a][j] for a in range(nv)], t_on_u[j], i)
                     + _diff([du1[i][b] for b in range(nu)], t_on_v[i], j))
              for j in range(nu)] for i in range(nv)]

    s = np.linspace(0.0, 1.0, 32)[:, None]
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    Bq = basis_matrix(SEG_POINTS, SEG_ORDER, s[:, 0])

    def cross_rows(d1a, d1b, xa, xb, d2a, d2b, span_along, reach_a, reach_b, sign):
        """T and A rows along a side.

        The direction field G(s) is a Hermite blend of the transverse tangents
        at the two end nodes with the node twists as slopes; it depends only on
        the side, so the cells on both sides of it share it. Each cell scales
        it by its own reach (its parameter span across), which keeps the two
        cross tangents parallel: G1 along the whole curve.
        """
        G = h00 * d1a + h10 * span_along * xa + h01 * d1b + h11 * span_along * xb
        lam = (1 - s) * reach_a + s * reach_b
        T = sign * lam * G
        A = lam * lam * ((1 - s) * d2a + s * d2b)
        # Exact fits: degree 4 and 3 polynomials in a quintic B-spline.
        return (np.linalg.lstsq(Bq, T, rcond=None)[0],
                np.linalg.lstsq(Bq, A, rcond=None)[0] if continuity == 'G2'
                else np.zeros((SEG_POINTS, 3)))

    cells = []
    gaps = [crosses[(u, v)][2] for u in U for v in V]
    for i in range(nv - 1):           # between v-curves V[i] and V[i+1]  (u direction)
        for j in range(nu - 1):       # between u-curves U[j] and U[j+1]  (v direction)
            cu0, cu1 = curves[U[j]], curves[U[j + 1]]
            cv0, cv1 = curves[V[i]], curves[V[i + 1]]
            c00, c10, c01, c11 = node[i][j], node[i + 1][j], node[i][j + 1], node[i + 1][j + 1]
            # Sides as curve segments (u-curves run along u, v-curves along v).
            bottom = segment(cu0, t_on_u[j][i], t_on_u[j][i + 1], c00, c10)
            top = segment(cu1, t_on_u[j + 1][i], t_on_u[j + 1][i + 1], c01, c11)
            left = segment(cv0, t_on_v[i][j], t_on_v[i][j + 1], c00, c01)
            right = segment(cv1, t_on_v[i + 1][j], t_on_v[i + 1][j + 1], c10, c11)
            # This cell's parameter spans on its four curves.
            hu0 = t_on_u[j][i + 1] - t_on_u[j][i]          # along bottom
            hu1 = t_on_u[j + 1][i + 1] - t_on_u[j + 1][i]  # along top
            hv0 = t_on_v[i][j + 1] - t_on_v[i][j]          # along left
            hv1 = t_on_v[i + 1][j + 1] - t_on_v[i + 1][j]  # along right

            rows = {}
            if continuity != 'G0':
                if j > 0:          # bottom lies on an inner u-curve
                    rows['b'] = cross_rows(dv1[i][j], dv1[i + 1][j], twist[i][j], twist[i + 1][j],
                                           dv2[i][j], dv2[i + 1][j], hu0, hv0, hv1, 1.0)
                if j + 1 < nu - 1:  # top
                    rows['t'] = cross_rows(dv1[i][j + 1], dv1[i + 1][j + 1], twist[i][j + 1],
                                           twist[i + 1][j + 1], dv2[i][j + 1], dv2[i + 1][j + 1],
                                           hu1, hv0, hv1, -1.0)
                if i > 0:          # left lies on an inner v-curve
                    rows['l'] = cross_rows(du1[i][j], du1[i][j + 1], twist[i][j], twist[i][j + 1],
                                           du2[i][j], du2[i][j + 1], hv0, hu0, hu1, 1.0)
                if i + 1 < nv - 1:  # right
                    rows['r'] = cross_rows(du1[i + 1][j], du1[i + 1][j + 1], twist[i + 1][j],
                                           twist[i + 1][j + 1], du2[i + 1][j], du2[i + 1][j + 1],
                                           hv1, hu0, hu1, -1.0)

            def side(E, key):
                if key in rows:
                    T, A = rows[key]
                    return ([E, T, A, 1.0], SEG_ORDER, 0.0)
                return ([E, None, None], SEG_ORDER, 0.0)

            def rev(sd):
                rws, k, sg = sd
                return ([R[::-1] if isinstance(R, np.ndarray) else R for R in rws], k, -sg)

            loop = [side(bottom, 'b'), side(right, 'r'), rev(side(top, 't')), rev(side(left, 'l'))]
            net, orders = quad_net(loop, continuity, min_points=STAR_POINTS)
            # Normal reference: the crossing curves' tangents at the cell's first corner.
            ref = np.cross(du1[i][j], dv1[i][j])
            cells.append((i, j, _orient(net, orders, ref), orders))
    info = {"fit_error": max(c.err for c in curves), "max_gap": max(gaps),
            "size": (nv - 1, nu - 1)}
    return cells, info
