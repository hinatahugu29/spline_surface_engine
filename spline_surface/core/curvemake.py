"""Making curves from curves: two views into 3D, a fillet at a corner, an
offset, an extension. Chains as in core.curvepoint (A, L, R, open or cyclic).
"""
import numpy as np

from .curvepoint import EPS, _unit, arc_table, insert_point, point_at, segments
from .curvesmooth import point_state, refit


# ---------------------------------------------------------------- dense points -> chain
def chain_through(points, cyclic=False):
    """A G1 chain through every point (Catmull-Rom handles); for dense points it
    follows them closely and Refit can thin it out."""
    P = np.asarray(points, float)
    n = len(P)
    if cyclic:
        prev, nxt = np.roll(P, 1, axis=0), np.roll(P, -1, axis=0)
    else:
        prev = np.concatenate([P[:1], P[:-1]])
        nxt = np.concatenate([P[1:], P[-1:]])
    D = (nxt - prev) / 6.0
    if not cyclic:
        D[0] = (P[1] - P[0]) / 3.0
        D[-1] = (P[-1] - P[-2]) / 3.0
    return P.copy(), P - D, P + D


def fit_points(points, cyclic=False, tolerance=None):
    """A chain with few points through dense `points` (within `tolerance`)."""
    A, L, R = chain_through(points, cyclic)
    if tolerance is None:
        size = float(np.ptp(np.asarray(points), axis=0).max()) or 1.0
        tolerance = 1e-3 * size
    return refit(A, L, R, cyclic, tolerance=tolerance)


# ---------------------------------------------------------------- two views
def flat_axis(points, tol=1e-3):
    """The world axis (0, 1, 2) a polyline is flat in, or None."""
    P = np.asarray(points, float)
    ext = np.ptp(P, axis=0)
    k = int(np.argmin(ext))
    return k if ext[k] <= tol * max(ext.max(), 1e-12) else None


def two_view(a, b, count=200):
    """3D points from two flat polylines drawn in two views (side and top, say).

    Each is flat in a different world axis; the third axis, which both run
    along, is the common one: at each position along it, the result takes from
    each view the coordinate the other view cannot see. Both must keep going one
    way along the common axis. Returns (points, common axis).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = flat_axis(a), flat_axis(b)
    if na is None or nb is None:
        raise ValueError("each curve must be flat, drawn in a front / side / top view")
    if na == nb:
        raise ValueError("the two curves are drawn in the same view; draw them in two views")
    c = 3 - na - nb
    out = []
    for P in (a, b):
        d = np.diff(P[:, c])
        if not (np.all(d >= -1e-9) or np.all(d <= 1e-9)):
            raise ValueError("a curve turns back along the %s axis" % "XYZ"[c])
    lo = max(a[:, c].min(), b[:, c].min())
    hi = min(a[:, c].max(), b[:, c].max())
    if hi - lo < 1e-9:
        raise ValueError("the two curves do not overlap along the %s axis" % "XYZ"[c])
    cs = np.linspace(lo, hi, count)
    srt = lambda P: P[np.argsort(P[:, c])]
    A, B = srt(a), srt(b)
    pts = np.zeros((count, 3))
    pts[:, c] = cs
    pts[:, nb] = np.interp(cs, A[:, c], A[:, nb])     # a's view shows the axis b is flat in
    pts[:, na] = np.interp(cs, B[:, c], B[:, na])
    return pts, c


# ---------------------------------------------------------------- fillet a corner
def fillet_corner(A, L, R, i, radius, cyclic=False):
    """Round the corner at anchor i: the curve is cut `radius` (along the curve)
    before and after it, exactly, and joined by an arc-like span that meets both
    sides G1. Returns (A, L, R, index of the first new point) or raises ValueError."""
    A, L, R = (np.array(x, float) for x in (A, L, R))
    n = len(A)
    if point_state(A, L, R, i, cyclic) != 'CORNER':
        raise ValueError("not a corner")
    m = segments(n, cyclic)
    tt, ss = arc_table(A, L, R, cyclic)
    total = ss[-1]
    arc = lambda t: float(np.interp(t, tt, ss))
    s0 = arc(i)
    s_prev = arc(i - 1) if i > 0 else arc(m - 1) - total        # a loop: across point 0
    s_next = arc(i + 1)
    r = min(float(radius), 0.9 * min(s0 - s_prev, s_next - s0))
    if r <= 0:
        raise ValueError("no room")
    inv = lambda s: float(np.interp(s % total if cyclic else s, ss, tt))
    ta, tb = inv(s0 - r), inv(s0 + r)          # before and after the corner
    pa, pb, pc = (point_at(A, L, R, [t], cyclic)[0] for t in (ta, tb, float(i)))
    for t in sorted((ta, tb), reverse=True):    # the later parameter first: the earlier stays valid
        A, L, R, _ = insert_point(A, L, R, t, cyclic)
    find = lambda p: int(np.argmin(np.linalg.norm(A - p, axis=1)))
    j1, j2, corner = find(pa), find(pb), find(pc)
    n = len(A)
    P1, P2 = A[j1], A[j2]
    T1 = _unit(A[j1] - L[j1])          # arriving at P1
    T2 = _unit(R[j2] - A[j2])          # leaving P2
    c = np.linalg.norm(P2 - P1)
    th = np.arccos(np.clip(T1 @ T2, -1.0, 1.0))
    h = c / 3.0 if th < 1e-6 else (2.0 / 3.0) * c * np.tan(th / 4.0) / np.sin(th / 2.0)
    R[j1] = P1 + h * T1
    L[j2] = P2 - h * T2
    keep = [x for x in range(n) if x != corner]
    return A[keep], L[keep], R[keep], keep.index(j1)


# ---------------------------------------------------------------- offset
def offset_points(A, L, R, distance, normal, cyclic=False, per_segment=48):
    """Points of the curve offset by `distance` within the plane of `normal`
    (positive: to the left looking along the curve with the normal up).
    Corners get round joins."""
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    nrm = _unit(np.asarray(normal, float))
    m = segments(len(A), cyclic)
    out = []
    for k in range(m):
        u = np.linspace(k, k + 1, per_segment + 1)
        u[0], u[-1] = k + 1e-7, k + 1 - 1e-7
        P = point_at(A, L, R, u, cyclic)
        T = point_at(A, L, R, u, cyclic, 1)
        T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), EPS)
        side = np.cross(nrm, T)
        Q = P + distance * side
        if out:
            prev = out[-1][-1]
            corner = point_at(A, L, R, [float(k)], cyclic)[0]
            a, b = prev - corner, Q[0] - corner
            ang = np.arctan2(np.cross(a, b) @ nrm, a @ b)
            if np.linalg.norm(prev - Q[0]) > 1e-6 * max(abs(distance), 1e-9):
                steps = max(int(abs(ang) / np.radians(10)), 1)
                arc = [corner + _rot(a, nrm, ang * j / steps) for j in range(1, steps)]
                out.append(np.array(arc).reshape(-1, 3))
            Q = Q[1:]
        out.append(Q)
    pts = np.concatenate([o for o in out if len(o)])
    return pts


def _rot(v, axis, ang):
    axis = _unit(axis)
    return (v * np.cos(ang) + np.cross(axis, v) * np.sin(ang)
            + axis * (axis @ v) * (1 - np.cos(ang)))


# ---------------------------------------------------------------- extend
def _blossom(P, a, b, c):
    """The blossom of the cubic with control points P at (a, b, c)."""
    Q = np.asarray(P, float)
    for x in (a, b, c):
        Q = (1 - x) * Q[:-1] + x * Q[1:]
    return Q[0]


def _span(P, a, b):
    """Control points of the same cubic over its parameters a .. b."""
    return np.array([_blossom(P, a, a, a), _blossom(P, a, a, b),
                     _blossom(P, a, b, b), _blossom(P, b, b, b)])


def extend_end(A, L, R, end, length, mode='NATURAL'):
    """A new segment at `end` ('START' / 'END') about `length` long.

    NATURAL continues the end segment's own cubic past its end (the join is as
    smooth as can be); STRAIGHT goes straight on along the end tangent.
    Returns (A, L, R)."""
    A, L, R = (np.array(x, float) for x in (A, L, R))
    n = len(A)
    if end == 'START':
        P = np.array([A[0], R[0], L[1], A[1]])
        out_dir = _unit(A[0] - R[0])
    else:
        P = np.array([A[-2], R[-2], L[-1], A[-1]])
        out_dir = _unit(A[-1] - L[-1])
    if np.linalg.norm(out_dir) == 0:
        out_dir = _unit(P[0] - P[3]) if end == 'START' else _unit(P[3] - P[0])
    length = float(length)
    if mode == 'STRAIGHT' or length <= 0:
        tip = (A[0] if end == 'START' else A[-1]) + out_dir * length
        base = A[0] if end == 'START' else A[-1]
        h = length / 3.0
        seg = np.array([tip, tip - out_dir * h, base + out_dir * h, base]) if end == 'START' \
            else np.array([base, base + out_dir * h, tip - out_dir * h, tip])
    else:
        def piece(e):
            return _span(P, -e, 0.0) if end == 'START' else _span(P, 1.0, 1.0 + e)

        def arc(Q):
            u = np.linspace(0, 1, 65)
            pts = ((1 - u) ** 3)[:, None] * Q[0] + (3 * (1 - u) ** 2 * u)[:, None] * Q[1] \
                + (3 * (1 - u) * u ** 2)[:, None] * Q[2] + (u ** 3)[:, None] * Q[3]
            return np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        lo, hi = 0.0, 1.0
        while arc(piece(hi)) < length and hi < 64:
            hi *= 2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if arc(piece(mid)) < length else (lo, mid)
        seg = piece(hi)
    if end == 'START':                 # seg runs from the new tip to the old start
        L[0] = seg[2]
        A = np.insert(A, 0, seg[0], axis=0)
        R = np.insert(R, 0, seg[1], axis=0)
        L = np.insert(L, 0, seg[0] - (seg[1] - seg[0]), axis=0)
    else:                              # from the old end to the new tip
        R[-1] = seg[1]
        A = np.append(A, seg[3][None], axis=0)
        L = np.append(L, seg[2][None], axis=0)
        R = np.append(R, (seg[3] + (seg[3] - seg[2]))[None], axis=0)
    return A, L, R
