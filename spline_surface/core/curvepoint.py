"""Editing the points of a Bezier curve while keeping its shape.

A spline is a chain of cubic Bezier segments: anchors A, left handles L and
right handles R, each (n, 3), as Blender stores them, plus whether it is
cyclic. Its parameter t runs over 0 .. segments (segment k covers k .. k+1);
a cyclic chain has n segments, the last one from A[n-1] back to A[0].

Insert splits a segment exactly (de Casteljau). Dissolve and Slide rebuild the
one or two segments next to a point by `fit_span`: the end points and the end
tangent directions are kept, only the two handle lengths are fitted to the old
shape, so the neighbours stay G1 exactly and the shape moves very little.
Knows nothing about Blender.
"""
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- evaluation
def segments(n, cyclic):
    return n if cyclic else n - 1


def segment(A, L, R, k, cyclic=False):
    """The four control points of segment k."""
    n = len(A)
    j = (k + 1) % n if cyclic else k + 1
    return np.array([A[k], R[k], L[j], A[j]], float)


def _bez(P, u, deriv=0):
    u = np.asarray(u, float)[..., None]
    P0, P1, P2, P3 = P
    if deriv == 0:
        return ((1 - u) ** 3 * P0 + 3 * (1 - u) ** 2 * u * P1
                + 3 * (1 - u) * u ** 2 * P2 + u ** 3 * P3)
    if deriv == 1:
        return 3 * ((1 - u) ** 2 * (P1 - P0) + 2 * (1 - u) * u * (P2 - P1) + u ** 2 * (P3 - P2))
    return 6 * ((1 - u) * (P2 - 2 * P1 + P0) + u * (P3 - 2 * P2 + P1))


def _locate(n, t, cyclic):
    """(segment index, local u) of chain parameters t."""
    m = segments(n, cyclic)
    t = np.clip(np.atleast_1d(np.asarray(t, float)), 0.0, m)
    k = np.minimum(np.floor(t).astype(int), m - 1)
    return k, t - k


def point_at(A, L, R, t, cyclic=False, deriv=0):
    """Points (or derivatives) of the chain at parameters t, shape (len(t), 3)."""
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    k, u = _locate(len(A), t, cyclic)
    out = np.empty((len(k), 3))
    for s in np.unique(k):
        sel = k == s
        out[sel] = _bez(segment(A, L, R, s, cyclic), u[sel], deriv)
    return out


def sample(A, L, R, t0, t1, count, cyclic=False):
    """count points from parameter t0 to t1 (t1 may pass the end of a cyclic chain)."""
    m = segments(len(A), cyclic)
    t = np.linspace(t0, t1, count)
    if cyclic:
        t = np.where(t > m, t - m, t)
        t = np.where(t < 0, t + m, t)
    return point_at(A, L, R, t, cyclic)


def arc_table(A, L, R, cyclic=False, per_segment=64):
    """(parameters, arc lengths from the start) on a fine grid."""
    m = segments(len(A), cyclic)
    t = np.linspace(0.0, m, m * per_segment + 1)
    p = point_at(A, L, R, t, cyclic)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return t, s


def nearest(A, L, R, point, cyclic=False, per_segment=48):
    """(parameter, distance) of the chain point closest to `point`."""
    m = segments(len(A), cyclic)
    t = np.linspace(0.0, m, m * per_segment + 1)
    q = np.asarray(point, float)
    d = np.linalg.norm(point_at(A, L, R, t, cyclic) - q, axis=1)
    i = int(np.argmin(d))
    lo, hi = t[max(i - 1, 0)], t[min(i + 1, len(t) - 1)]
    fine = np.linspace(lo, hi, 65)
    d2 = np.linalg.norm(point_at(A, L, R, fine, cyclic) - q, axis=1)
    x = float(fine[int(np.argmin(d2))])
    for _ in range(6):                     # Newton: foot of the perpendicular
        c = point_at(A, L, R, [x], cyclic)[0] - q
        d1 = point_at(A, L, R, [x], cyclic, 1)[0]
        dd = point_at(A, L, R, [x], cyclic, 2)[0]
        den = d1 @ d1 + c @ dd
        if abs(den) < EPS:
            break
        x = float(np.clip(x - (c @ d1) / den, lo, hi))
    return x, float(np.linalg.norm(point_at(A, L, R, [x], cyclic)[0] - q))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


# ---------------------------------------------------------------- insert
def split(P, u):
    """De Casteljau: the two halves (4 points each) of a cubic at u."""
    P = np.asarray(P, float)
    a = (1 - u) * P[:-1] + u * P[1:]
    b = (1 - u) * a[:-1] + u * a[1:]
    c = (1 - u) * b[0] + u * b[1]
    return np.array([P[0], a[0], b[0], c]), np.array([c, b[1], a[2], P[3]])


def insert_point(A, L, R, t, cyclic=False, snap=1e-6):
    """A new anchor at parameter t, the shape unchanged.

    Returns (A, L, R, index of the anchor at t); when t already sits on an
    anchor, that anchor's index and the chain unchanged.
    """
    A, L, R = (np.array(x, float) for x in (A, L, R))
    n = len(A)
    k, u = _locate(n, t, cyclic)
    k, u = int(k[0]), float(u[0])
    if u < snap:
        return A, L, R, k
    if u > 1 - snap:
        return A, L, R, (k + 1) % n if cyclic else k + 1
    left, right = split(segment(A, L, R, k, cyclic), u)
    j = (k + 1) % n if cyclic else k + 1
    R[k], L[j] = left[1], right[2]
    A = np.insert(A, k + 1, left[3], axis=0)
    L = np.insert(L, k + 1, left[2], axis=0)
    R = np.insert(R, k + 1, right[1], axis=0)
    return A, L, R, k + 1


# ---------------------------------------------------------------- span fitting
def fit_span(Q, d0, d3, iterations=30):
    """Handles (P1, P2) of the cubic from Q[0] to Q[-1] closest to the points Q,
    leaving Q[0] along d0 and arriving at Q[-1] along d3 (directions; only the
    two handle lengths are fitted). Returns (P1, P2, largest distance).

    The two lengths and the parameter of every point are solved together
    (Levenberg-Marquardt): alternating between them crawls.
    """
    Q = np.asarray(Q, float)
    P0, P3 = Q[0], Q[-1]
    t0, t3 = _unit(np.asarray(d0, float)), _unit(np.asarray(d3, float))
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    total = seg.sum()
    if total < EPS:
        return P0.copy(), P3.copy(), 0.0
    m = len(Q)
    inner = slice(1, m - 1)                       # the end parameters stay 0 and 1
    u = np.concatenate([[0.0], np.cumsum(seg)]) / total

    def cubic(a, b):
        return np.array([P0, P0 + a * t0, P3 - b * t3, P3])

    def lengths(u):
        b1, b2 = 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2
        base = ((1 - u) ** 3 + b1)[:, None] * P0 + (b2 + u ** 3)[:, None] * P3
        M = np.stack([(b1[:, None] * t0).ravel(), (-b2[:, None] * t3).ravel()], 1)
        return np.linalg.lstsq(M, (Q - base).ravel(), rcond=None)[0]

    a, b = lengths(u)
    if not (a > 0 and b > 0):
        a = b = total / 3.0

    def cost(a, b, u):
        return float(((_bez(cubic(a, b), u) - Q) ** 2).sum())

    lam, f = 1e-3, cost(a, b, u)
    for _ in range(iterations):
        P = cubic(a, b)
        r = (_bez(P, u) - Q)[inner]                                   # (k, 3)
        ui = u[inner]
        k = len(ui)
        J = np.zeros((3 * k, 2 + k))
        J[:, 0] = (3 * (1 - ui) ** 2 * ui)[:, None].repeat(3, 1).ravel() * np.tile(t0, k)
        J[:, 1] = -(3 * (1 - ui) * ui ** 2)[:, None].repeat(3, 1).ravel() * np.tile(t3, k)
        d1 = _bez(P, ui, 1)
        for i in range(k):
            J[3 * i:3 * i + 3, 2 + i] = d1[i]
        g = J.T @ r.ravel()
        H = J.T @ J
        while True:
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
            a2, b2 = a + step[0], b + step[1]
            u2 = u.copy()
            u2[inner] = np.clip(ui + step[2:], 0.0, 1.0)
            f2 = cost(a2, b2, u2) if a2 > 0 and b2 > 0 else np.inf
            if f2 < f:
                a, b, u, f, lam = a2, b2, u2, f2, max(lam / 10, 1e-9)
                break
            lam *= 10
            if lam > 1e8:
                break
        if lam > 1e8 or f < 1e-24:
            break
    P = cubic(a, b)
    err = np.linalg.norm(_bez(P, u) - Q, axis=1).max()
    return P[1], P[2], float(err)


def _out_dir(A, L, R, i, cyclic, side):
    """Unit tangent at anchor i leaving to the right ('R') or arriving from the left ('L')."""
    if side == 'R':
        v = R[i] - A[i]
        if np.linalg.norm(v) < EPS:
            t = float(i)
            v = point_at(A, L, R, [min(t + 1e-3, segments(len(A), cyclic))], cyclic)[0] - A[i]
    else:
        v = A[i] - L[i]
        if np.linalg.norm(v) < EPS:
            m = segments(len(A), cyclic)
            t = float(i) if i > 0 else (m if cyclic else 0.0)
            v = A[i] - point_at(A, L, R, [max(t - 1e-3, 0.0)], cyclic)[0]
    return _unit(v)


# ---------------------------------------------------------------- dissolve
def dissolve_point(A, L, R, i, cyclic=False, samples=48):
    """Remove anchor i and refit the joined span to the old shape.

    The neighbours keep their position and tangent direction. An end anchor of
    an open chain just goes (its end segment with it). Returns (A, L, R, error).
    """
    A, L, R = (np.array(x, float) for x in (A, L, R))
    n = len(A)
    if n <= 2 or (cyclic and n <= 3):
        raise ValueError("too few points to remove one")
    if not cyclic and i in (0, n - 1):
        keep = [j for j in range(n) if j != i]
        return A[keep], L[keep], R[keep], 0.0
    j, k = (i - 1) % n, (i + 1) % n
    tj = float(j)
    Q = sample(A, L, R, tj, tj + 2.0, 2 * samples + 1, cyclic)
    P1, P2, err = fit_span(Q, _out_dir(A, L, R, j, cyclic, 'R'), _out_dir(A, L, R, k, cyclic, 'L'))
    R[j], L[k] = P1, P2
    keep = [x for x in range(n) if x != i]
    return A[keep], L[keep], R[keep], err


# ---------------------------------------------------------------- slide
def slide_point(A, L, R, i, t, cyclic=False, samples=48):
    """Move anchor i to the chain's own point at parameter t (of the chain as
    given), keeping the shape: the curve is the same, its point is elsewhere.

    t is kept between the neighbouring anchors. An end anchor of an open chain
    can only slide inwards (it trims the curve exactly). Returns (A, L, R, error).
    """
    A, L, R = (np.array(x, float) for x in (A, L, R))
    n = len(A)
    m = segments(n, cyclic)
    gap = 1e-3
    if not cyclic and i == 0:
        t = float(np.clip(t, 0.0, 1.0 - gap))
        _, right = split(segment(A, L, R, 0), t)
        A[0], R[0], L[1] = right[0], right[1], right[2]
        L[0] = A[0] - (R[0] - A[0])
        return A, L, R, 0.0
    if not cyclic and i == n - 1:
        t = float(np.clip(t, n - 2 + gap, n - 1.0))
        left, _ = split(segment(A, L, R, n - 2), t - (n - 2))
        R[n - 2], L[n - 1], A[n - 1] = left[1], left[2], left[3]
        R[n - 1] = A[n - 1] + (A[n - 1] - L[n - 1])
        return A, L, R, 0.0
    lo, hi = i - 1.0 + gap, i + 1.0 - gap
    if cyclic and i == 0:          # parameters around anchor 0 wrap: work on the far side
        if t > m / 2:
            t -= m
    t = float(np.clip(t, lo, hi))
    if abs(t - i) < 1e-9:
        return A, L, R, 0.0
    j, k = (i - 1) % n, (i + 1) % n
    tw = t + m if t < 0 else t
    p = point_at(A, L, R, [tw], cyclic)[0]
    tan = _unit(point_at(A, L, R, [tw], cyclic, 1)[0])
    if np.linalg.norm(tan) == 0:
        return A, L, R, 0.0
    tj = float(j) if j < i or not cyclic else float(j) - m
    Q1 = sample(A, L, R, tj, t, samples + 1, cyclic)
    Q2 = sample(A, L, R, t, float(i) + 1.0, samples + 1, cyclic)
    Q1[-1] = Q2[0] = p
    r1, l1, e1 = fit_span(Q1, _out_dir(A, L, R, j, cyclic, 'R'), tan)
    r2, l2, e2 = fit_span(Q2, tan, _out_dir(A, L, R, k, cyclic, 'L'))
    R[j], L[i], A[i], R[i], L[k] = r1, l1, p, r2, l2
    return A, L, R, max(e1, e2)


def arc_param(A, L, R, t, ds, cyclic=False):
    """Parameter reached by walking the arc length ds (signed) from parameter t."""
    tt, s = arc_table(A, L, R, cyclic)
    total = s[-1]
    s0 = float(np.interp(t, tt, s)) + ds
    if cyclic:
        s0 = s0 % total
    else:
        s0 = float(np.clip(s0, 0.0, total))
    return float(np.interp(s0, s, tt))
