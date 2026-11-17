"""Finding shared edges between patches, enforcing and measuring G0/G1/G2.

All work happens in the canonical frame of core.extend.edge_frame: axis 0 runs
along the edge, axis 1 across it. The reference patch has its edge as the
last row; the target patch is flipped so its edge is row 0.

Patches are (net, orders) as in core.extend; orders=None means Bezier.
"""
import numpy as np

from .bezier import elevate_curve
from .extend import EDGES, bezier_orders, edge_derivatives, edge_frame, frame_orders
from .nurbs import basis_matrix, eval_curve, eval_points, project_on_curve

MAX_ORDER = 6  # Blender NURBS order limit


def _framed(P, orders, edge):
    """Canonical net and (along, across) orders for `edge`."""
    P = np.asarray(P, dtype=float)
    return edge_frame(P, edge)[0], frame_orders(orders or bezier_orders(P), edge)


def _edge_curve(P, orders, e, t):
    C, (ka, _) = _framed(P, orders, e)
    return eval_curve(C[:, -1], ka, t)


def edge_pair_gap(Pa, Pb, edge_a, edge_b, reversed_, orders_a=None, orders_b=None, samples=16):
    """Mean distance between edge_a of Pa and edge_b of Pb (b walked backwards if reversed_)."""
    t = np.linspace(0.0, 1.0, samples)
    ca, cb = _edge_curve(Pa, orders_a, edge_a, t), _edge_curve(Pb, orders_b, edge_b, t)
    return float(np.linalg.norm(ca - (cb[::-1] if reversed_ else cb), axis=1).mean())


def find_shared_edge(Pa, Pb, tol, orders_a=None, orders_b=None, samples=16):
    """Best matching edge pair between two patches.

    Returns (edge_a, edge_b, reversed, mean_gap) or None if no edges lie
    within `tol` of each other. `reversed` means the edges run in opposite
    directions.
    """
    t = np.linspace(0.0, 1.0, samples)

    def collapsed(c):  # a patch edge squeezed to a point (pole of a cap, corner fill)
        return np.linalg.norm(np.diff(c, axis=0), axis=1).sum() <= tol

    best = None
    for ea in EDGES:
        ca = _edge_curve(Pa, orders_a, ea, t)
        if collapsed(ca):
            continue
        for eb in EDGES:
            cb = _edge_curve(Pb, orders_b, eb, t)
            if collapsed(cb):
                continue
            for rev in (False, True):
                gap = np.linalg.norm(ca - (cb[::-1] if rev else cb), axis=1).mean()
                if best is None or gap < best[3]:
                    best = (ea, eb, rev, gap)
    return best if best and best[3] <= tol else None


def _framed_indices(shape, edge):
    """Index grid of a net of `shape`, viewed in the canonical frame of `edge`."""
    idx = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), -1)
    return edge_frame(idx, edge)[0].astype(int)


def _elevate_rows(D, axis, count):
    """Degree-elevate every Bezier curve of a net along `axis` to `count` points."""
    if D.shape[axis] >= count:
        return D
    if count > MAX_ORDER:
        raise ValueError("needs degree %d, above Blender's limit of 5" % (count - 1))
    D = np.moveaxis(D, axis, 1)
    D = np.stack([elevate_curve(row, count - 1) for row in D], axis=0)
    return np.moveaxis(D, 1, axis)


def _describe(n, k):
    spans = n - k + 1
    return "%d points, order %d (%d span%s)" % (n, k, spans, "" if spans == 1 else "s")


def _spread_weights(m):
    """1 on the seam row, easing to 0 on the opposite edge row (smoothstep)."""
    s = np.linspace(0.0, 1.0, m)
    return 1.0 - s * s * (3.0 - 2.0 * s)


def match_net(target, ref, edge_t, edge_r, reversed_, continuity,
              orders_t=None, orders_r=None, spread=False):
    """Make `target`'s `edge_t` meet `ref` on `edge_r` with G0/G1/G2.

    Returns (net, orders) of the modified target. Only the first one to three
    control rows next to the edge are changed and the target's tangent scale
    is preserved. With `spread`, the move of the seam row is first eased into
    the whole patch (fading out towards the opposite edge, which stays put),
    so a seam pulled far away bends the patch smoothly instead of leaving a
    ridge three rows in. Along the edge both patches must share the same structure;
    two Bezier edges of different degree are reconciled by degree elevation.
    Across the edge a Bezier target is elevated if it has too few rows to keep
    the opposite edge untouched.
    """
    level = {'G0': 0, 'G1': 1, 'G2': 2}[continuity]
    D, (ka_t, kc_t) = _framed(target, orders_t, edge_t)
    D = D[:, ::-1]
    C, (ka_r, kc_r) = _framed(ref, orders_r, edge_r)
    if reversed_:
        C = C[::-1]

    bez_t, bez_r = ka_t == D.shape[0], ka_r == C.shape[0]
    if bez_t and bez_r:
        n = max(D.shape[0], C.shape[0])
        D, C = _elevate_rows(D, 0, n), _elevate_rows(C, 0, n)
        ka_t = ka_r = n
    elif (D.shape[0], ka_t) != (C.shape[0], ka_r):
        raise ValueError("edges differ along their length: target has %s, reference has %s; "
                         "give both the same point count and order along the seam"
                         % (_describe(D.shape[0], ka_t), _describe(C.shape[0], ka_r)))

    need = level + 2  # keeps the opposite edge row untouched
    if D.shape[1] < need:
        if D.shape[1] != kc_t:
            raise ValueError("target has too few rows across the seam for %s" % continuity)
        D = _elevate_rows(D, 1, need)
        kc_t = need
    D = D.copy()
    m_t = D.shape[1]
    e1 = basis_matrix(m_t, kc_t, 0.0, 1)[0]
    e2 = basis_matrix(m_t, kc_t, 0.0, 2)[0]

    if spread:
        D += _spread_weights(m_t)[None, :, None] * (C[:, -1] - D[:, 0])[:, None, :]
    # The target's own tangent scale, read before its seam row moves: measured
    # after, a seam pulled far away would blow the tangents up.
    cur = np.einsum("j,ijd->id", e1, D)
    D[:, 0] = C[:, -1]
    if level >= 1:
        Tr, Ar = edge_derivatives(C, kc_r)
        lam = np.linalg.norm(cur, axis=1).mean() / max(np.linalg.norm(Tr, axis=1).mean(), 1e-12)
        D[:, 1] = (lam * Tr - e1[0] * D[:, 0]) / e1[1]
    if level >= 2:
        if abs(e2[2]) < 1e-12:
            raise ValueError("target is only degree 1 across the seam; G2 is impossible")
        D[:, 2] = (lam * lam * Ar - e2[0] * D[:, 0] - e2[1] * D[:, 1]) / e2[2]

    # Back to the target's own layout (its shape may have grown).
    framed = D[:, ::-1]
    along_is_u = edge_t[0] == 'V'
    shape = framed.shape[:2] if along_is_u else framed.shape[1::-1]
    out = np.zeros(shape + (3,))
    idx = _framed_indices(shape, edge_t)
    out[idx[..., 0], idx[..., 1]] = framed
    orders = (ka_t, kc_t) if along_is_u else (kc_t, ka_t)
    return out, orders


def _cross_curvature(P, orders, a, b):
    """Point, unit normal and normal curvature across the edge (perpendicular to a)."""
    def ev(da, db):
        return eval_points(P, orders, a, b, da, db)

    S, Sa, Sb = ev(0, 0), ev(1, 0), ev(0, 1)
    Saa, Sab, Sbb = ev(2, 0), ev(1, 1), ev(0, 2)
    N = np.cross(Sa, Sb)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    d = np.cross(N, Sa)
    kappa = np.zeros(len(a))
    for i in range(len(a)):
        (x, y), *_ = np.linalg.lstsq(np.stack([Sa[i], Sb[i]], 1), d[i], rcond=None)
        dd = x * Sa[i] + y * Sb[i]
        II = x * x * Saa[i] @ N[i] + 2 * x * y * Sab[i] @ N[i] + y * y * Sbb[i] @ N[i]
        kappa[i] = II / (dd @ dd)
    return S, N, kappa


def measure(target, ref, edge_t, edge_r, reversed_, orders_t=None, orders_r=None, samples=33):
    """Measure the seam: max position gap, max normal angle (deg), curvature jump.

    Curvature jump is relative: |dk| / (max |k| + 1/size), so flat seams do not
    blow up the ratio.
    """
    C, oc = _framed(ref, orders_r, edge_r)
    if reversed_:
        C = C[::-1]
    D, od = _framed(target, orders_t, edge_t)
    D = D[:, ::-1]
    s = np.linspace(0.0, 1.0, samples)
    Sr, Nr, kr = _cross_curvature(C, oc, s, np.ones(samples))
    # Pair every reference sample with the closest point on the target edge.
    dense = np.linspace(0.0, 1.0, 1001)
    pts = eval_curve(D[:, 0], od[0], dense)
    t = dense[np.argmin(np.linalg.norm(Sr[:, None] - pts[None], axis=2), axis=1)]
    t = project_on_curve(D[:, 0], od[0], Sr, t)
    St, Nt, kt = _cross_curvature(D, od, t, np.zeros(samples))
    sign = np.sign(np.nanmean((Nr * Nt).sum(1))) or 1.0
    Nt, kt = Nt * sign, kt * sign
    size = np.ptp(np.concatenate([C.reshape(-1, 3), D.reshape(-1, 3)]), axis=0).max()
    gap = np.linalg.norm(Sr - St, axis=1).max()
    # nan samples sit on a collapsed (degenerate) corner and have no normal
    angle = np.nanmax(np.degrees(np.arccos(np.clip((Nr * Nt).sum(1), -1.0, 1.0))))
    rel = np.nanmax(np.abs(kr - kt) / (np.maximum(np.abs(kr), np.abs(kt)) + 1.0 / size))
    return {"gap": float(gap), "angle": float(angle), "curvature": float(rel), "size": float(size)}


def grade(m, gap_rel=1e-4, angle_deg=0.1, curv_rel=0.02):
    """Classify a measure() result as 'G2', 'G1', 'G0' or 'open'."""
    if m["gap"] > gap_rel * m["size"]:
        return "open"
    if m["angle"] > angle_deg:
        return "G0"
    if m["curvature"] > curv_rel:
        return "G1"
    return "G2"


_EDGE_UV = {'U0': lambda t: (0 * t, t), 'U1': lambda t: (0 * t + 1, t),
            'V0': lambda t: (t, 0 * t), 'V1': lambda t: (t, 0 * t + 1)}


def _edge_samples(P, orders, edge, t):
    """Points and unit normals along an edge.

    The normal is taken right on the edge. Only where the surface is degenerate
    there (a collapsed corner, no normal) is it taken a hair inside: always
    stepping inside moved it along the edge too, and on a steep patch the two
    sides of a seam then read their normals at different points (0.2-0.5 deg)."""
    u, v = _EDGE_UV[edge](t)
    pts = eval_points(P, orders, u, v)
    n = np.cross(eval_points(P, orders, u, v, 1, 0), eval_points(P, orders, u, v, 0, 1))
    size = np.linalg.norm(n, axis=1)
    bad = size < 1e-9 * max(size.max(), 1e-30)
    if bad.any():
        ui, vi = np.clip(u[bad], 1e-4, 1 - 1e-4), np.clip(v[bad], 1e-4, 1 - 1e-4)
        n[bad] = np.cross(eval_points(P, orders, ui, vi, 1, 0), eval_points(P, orders, ui, vi, 0, 1))
    return pts, n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-24)


def partial_contact(Pa, Pb, tol, orders_a=None, orders_b=None, samples=41, dense=241):
    """A seam where part of an edge of one patch lies on an edge of the other.

    Whole-edge seams are found by find_shared_edge; this one finds the rest: a
    piece of a split fill along part of its neighbour's edge, a short patch on a
    long one. For every pair of edges, the samples of the shorter stretch that lie
    within `tol` of the other edge form the contact (at least three samples and a
    twentieth of the larger patch). Returns dict(edge_a, edge_b, gap, angle,
    length) for the longest contact - angle is the largest normal angle along it,
    in degrees - or None.
    """
    ca, cb = np.reshape(Pa, (-1, 3)), np.reshape(Pb, (-1, 3))
    # control hulls contain the surfaces: far apart boxes cannot touch
    if np.any(ca.max(0) + tol < cb.min(0)) or np.any(cb.max(0) + tol < ca.min(0)):
        return None
    orders_a = orders_a or bezier_orders(np.asarray(Pa))
    orders_b = orders_b or bezier_orders(np.asarray(Pb))
    size = np.ptp(np.concatenate([ca, cb]), 0).max()
    t, td = np.linspace(0.0, 1.0, samples), np.linspace(0.0, 1.0, dense)
    edges_b = {eb: _edge_samples(Pb, orders_b, eb, td) for eb in EDGES}
    best = None
    for ea in EDGES:
        pa, na = _edge_samples(Pa, orders_a, ea, t)
        lo, hi = pa.min(0) - tol, pa.max(0) + tol
        for eb in EDGES:
            pb, nb = edges_b[eb]
            if np.any(pb.max(0) < lo) or np.any(pb.min(0) > hi):
                continue
            # distance to the edge itself (its fine polyline's segments), not to its samples
            a0, ab = pb[:-1], pb[1:] - pb[:-1]
            f = np.clip(((pa[:, None] - a0[None]) * ab[None]).sum(2)
                        / np.maximum((ab * ab).sum(1), 1e-30)[None], 0.0, 1.0)
            dist = np.linalg.norm(a0[None] + f[..., None] * ab[None] - pa[:, None], axis=2)
            seg = dist.argmin(1)
            gap = dist[np.arange(len(t)), seg]
            on = gap <= tol
            if on.sum() < 3:
                continue
            length = np.linalg.norm(np.diff(pa[on], axis=0), axis=1).sum()
            if length < 0.05 * size:
                continue
            # the other patch's normal at the exact foot point, not at its nearest sample:
            # on a tight fillet the normal turns about a degree between samples
            idx = np.arange(len(t))[on]
            foot = td[seg[idx]] + f[idx, seg[idx]] * (td[1] - td[0])
            _, nb_on = _edge_samples(Pb, orders_b, eb, foot)
            ang = np.degrees(np.arccos(np.clip(np.abs((na[on] * nb_on).sum(1)), 0.0, 1.0)))
            cand = {"edge_a": ea, "edge_b": eb, "gap": float(gap[on].max()),
                    "angle": float(ang.max()), "length": float(length)}
            if best is None or length > best["length"]:
                best = cand
    return best
