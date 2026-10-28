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


def find_shared_edge(Pa, Pb, tol, orders_a=None, orders_b=None, samples=16):
    """Best matching edge pair between two patches.

    Returns (edge_a, edge_b, reversed, mean_gap) or None if no edges lie
    within `tol` of each other. `reversed` means the edges run in opposite
    directions.
    """
    t = np.linspace(0.0, 1.0, samples)

    def curve(P, orders, e):
        C, (ka, _) = _framed(P, orders, e)
        return eval_curve(C[:, -1], ka, t)

    best = None
    for ea in EDGES:
        ca = curve(Pa, orders_a, ea)
        for eb in EDGES:
            cb = curve(Pb, orders_b, eb)
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


def match_net(target, ref, edge_t, edge_r, reversed_, continuity,
              orders_t=None, orders_r=None):
    """Make `target`'s `edge_t` meet `ref` on `edge_r` with G0/G1/G2.

    Returns (net, orders) of the modified target. Only the first one to three
    control rows next to the edge are changed and the target's tangent scale
    is preserved. Along the edge both patches must share the same structure;
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

    D[:, 0] = C[:, -1]
    if level >= 1:
        Tr, Ar = edge_derivatives(C, kc_r)
        cur = np.einsum("j,ijd->id", e1, D)
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
