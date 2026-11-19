"""Smoothness at the points of a Bezier chain: measure it, make it G2, refit.

Chains as in core.curvepoint (anchors A, handles L / R, open or cyclic).

At an anchor the curve's curvature on each side depends only on the handle on
that side and the next handle beyond it: for the segment leaving A along R,
    kappa = 2/3 * h / a^2,   a = |R - A|,  h = distance of the next handle from
                             the tangent line.
So with the tangent kept, the anchor's own two handle lengths set the two
curvatures: Make G2 scales them so both sides have the same curvature; no
other control point moves (the neighbours' curvature shifts a little, so a
whole curve is solved as one system, make_all_g2). That works when
both sides bend the same way; at an inflection, or next to a straight side,
lengths alone cannot do it and the point is reported instead.
"""
import numpy as np

from .curvepoint import EPS, _unit, fit_span, point_at, segments
from .curvepull import smooth_points

G1_DEG = 0.1          # handles within this angle count as lined up (G1)
G2_REL = 0.01         # curvatures within 1% (and within G1_DEG of direction) count as G2


def _neighbours(n, i, cyclic):
    """(previous anchor, next anchor), None past an open end."""
    prev = (i - 1) % n if (cyclic or i > 0) else None
    nxt = (i + 1) % n if (cyclic or i < n - 1) else None
    return prev, nxt


def side_curvature(A, L, R, i, cyclic=False):
    """Curvature vectors (left, right) at anchor i; None for a side with no segment.

    Each is kappa times the unit direction it bends towards.
    """
    n = len(A)
    prev, nxt = _neighbours(n, i, cyclic)
    out = []
    for side, other in (('L', prev), ('R', nxt)):
        if other is None:
            out.append(None)
            continue
        if side == 'R':
            hv, far = R[i] - A[i], L[other]
        else:
            hv, far = L[i] - A[i], R[other]
        a = np.linalg.norm(hv)
        if a < EPS:
            out.append(None)
            continue
        t = hv / a
        off = (far - (A[i] + hv))
        perp = off - (off @ t) * t
        out.append(2.0 / 3.0 * perp / (a * a))
    return out[0], out[1]


def point_state(A, L, R, i, cyclic=False):
    """'END', 'CORNER' (G0), 'G1' or 'G2' at anchor i."""
    n = len(A)
    prev, nxt = _neighbours(n, i, cyclic)
    if prev is None or nxt is None:
        return 'END'
    a, b = A[i] - L[i], R[i] - A[i]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS or a @ b < np.cos(np.radians(G1_DEG)) * na * nb:
        return 'CORNER'
    kl, kr = side_curvature(A, L, R, i, cyclic)
    if kl is None or kr is None:
        return 'G1'
    ml, mr = np.linalg.norm(kl), np.linalg.norm(kr)
    scale = max(ml, mr)
    if scale < 1e-9:
        return 'G2'                                   # straight through
    if abs(ml - mr) > G2_REL * scale:
        return 'G1'
    if min(ml, mr) > 1e-9 * scale and kl @ kr < np.cos(np.radians(G1_DEG)) * ml * mr:
        return 'G1'
    return 'G2'


def make_g2(A, L, R, i, cyclic=False, max_factor=4.0):
    """Scale anchor i's two handle lengths so both sides have the same curvature.

    Returns (A, L, R, why): why is None when done, else the reason it was left
    alone ('end', 'corner', 'straight', 'inflection', 'twisted', 'too far').
    """
    A, L, R = (np.array(x, float) for x in (A, L, R))
    state = point_state(A, L, R, i, cyclic)
    if state == 'END':
        return A, L, R, 'end'
    if state == 'CORNER':
        return A, L, R, 'corner'
    if state == 'G2':
        return A, L, R, None
    kl, kr = side_curvature(A, L, R, i, cyclic)
    ml, mr = np.linalg.norm(kl), np.linalg.norm(kr)
    big = max(ml, mr)
    if min(ml, mr) < 1e-6 * big:
        return A, L, R, 'straight'
    cos = kl @ kr / (ml * mr)
    if cos < -0.5:
        return A, L, R, 'inflection'
    if cos < np.cos(np.radians(5.0)):
        return A, L, R, 'twisted'
    target = np.sqrt(ml * mr)                  # least change of the two lengths (in log)
    al, ar = np.linalg.norm(L[i] - A[i]), np.linalg.norm(R[i] - A[i])
    # kappa = 2/3 h / a^2 with h fixed: a' = a * sqrt(kappa / target)
    fl, fr = np.sqrt(ml / target), np.sqrt(mr / target)
    if max(fl, fr, 1 / fl, 1 / fr) > max_factor:
        return A, L, R, 'too far'
    L[i] = A[i] + (L[i] - A[i]) * fl
    R[i] = A[i] + (R[i] - A[i]) * fr
    del al, ar
    return A, L, R, None


def _why_not(A, L, R, i, cyclic):
    """Why lengths alone cannot make anchor i G2 (None: they can, or it already is)."""
    state = point_state(A, L, R, i, cyclic)
    if state == 'END':
        return 'end'
    if state == 'CORNER':
        return 'corner'
    kl, kr = side_curvature(A, L, R, i, cyclic)
    if kl is None or kr is None:
        return 'end'
    ml, mr = np.linalg.norm(kl), np.linalg.norm(kr)
    if max(ml, mr) < 1e-12:
        return None                                   # straight through: already G2
    if min(ml, mr) < 1e-6 * max(ml, mr):
        return 'straight'
    cos = kl @ kr / (ml * mr)
    if cos < -0.5:
        return 'inflection'
    if cos < np.cos(np.radians(5.0)):
        return 'twisted'
    return None


def make_all_g2(A, L, R, points, cyclic=False, max_factor=4.0, iterations=60):
    """Make G2 at all of `points` together, by their handle lengths only.

    A point's lengths also change its neighbours' curvature, so they are solved
    as one system (Levenberg-Marquardt on the log lengths): the curvature on
    both sides of every point made equal, with the least change of the lengths.
    A point that cannot be done is left as it was, with the reason.
    Returns (A, L, R, {point: why it was left}).
    """
    A, L, R = (np.array(x, float) for x in (A, L, R))
    left, work = {}, []
    for i in points:
        why = _why_not(A, L, R, i, cyclic)
        if why == 'end':
            continue
        if why is not None:
            left[i] = why
        else:
            work.append(i)              # G2 ones too: their neighbours' lengths will change
    if not work or all(point_state(A, L, R, i, cyclic) == 'G2' for i in work):
        return A, L, R, left
    uL = {i: _unit(L[i] - A[i]) for i in work}
    uR = {i: _unit(R[i] - A[i]) for i in work}
    x0 = np.log(np.array([[np.linalg.norm(L[i] - A[i]), np.linalg.norm(R[i] - A[i])]
                          for i in work]).ravel())
    keep = 0.02                                   # how strongly the lengths stay put

    def put(x):
        for k, i in enumerate(work):
            L[i] = A[i] + np.exp(x[2 * k]) * uL[i]
            R[i] = A[i] + np.exp(x[2 * k + 1]) * uR[i]

    def resid(x):
        put(x)
        r = []
        for i in work:
            kl, kr = side_curvature(A, L, R, i, cyclic)
            ml, mr = np.linalg.norm(kl), np.linalg.norm(kr)
            if ml < 1e-15 or mr < 1e-15 or kl @ kr <= 0:
                r.append(10.0)                    # a side went straight or flipped: far off
            else:
                r.append(np.log(ml) - np.log(mr))
        return np.concatenate([r, keep * (x - x0)])

    x = x0.copy()
    f = resid(x)
    cost, lam = f @ f, 1e-3
    for _ in range(iterations):
        J = np.empty((len(f), len(x)))
        for c in range(len(x)):
            dx = np.zeros_like(x)
            dx[c] = 1e-6
            J[:, c] = (resid(x + dx) - f) / 1e-6
        g, H = J.T @ f, J.T @ J
        while True:
            x2 = x - np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            f2 = resid(x2)
            if f2 @ f2 < cost:
                x, f, cost, lam = x2, f2, f2 @ f2, max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e10:
                break
        if lam > 1e10 or np.abs(f[:len(work)]).max() < 1e-6:
            break
    put(x)
    # Points still off, or pushed too far from where they were: back as they were.
    factor = np.exp(np.abs(x - x0)).reshape(-1, 2).max(axis=1)
    for k, i in enumerate(work):
        if factor[k] > max_factor:
            left[i] = 'too far'
    for k, i in enumerate(work):
        if i in left:
            L[i] = A[i] + np.exp(x0[2 * k]) * uL[i]
            R[i] = A[i] + np.exp(x0[2 * k + 1]) * uR[i]
    for i in work:
        if i not in left and point_state(A, L, R, i, cyclic) != 'G2':
            left[i] = 'too far'
    return A, L, R, left


def frame(A, L, R, i, cyclic=False):
    """(tangent, normal, binormal) unit vectors at anchor i.

    The normal points to where the curve bends (average of the two sides); on a
    straight stretch it is any perpendicular, kept in the curve's plane if it
    has one.
    """
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    n = len(A)
    t = _unit(R[i] - L[i])
    if np.linalg.norm(t) == 0:
        t = _unit(R[i] - A[i]) if np.linalg.norm(R[i] - A[i]) > EPS else _unit(A[i] - L[i])
    if np.linalg.norm(t) == 0:
        prev, nxt = _neighbours(n, i, cyclic)
        t = _unit(A[nxt if nxt is not None else i] - A[prev if prev is not None else i])
    kl, kr = side_curvature(A, L, R, i, cyclic)
    k = sum(v for v in (kl, kr) if v is not None) if (kl is not None or kr is not None) else np.zeros(3)
    k = k - (k @ t) * t
    if np.linalg.norm(k) < 1e-9:
        P = np.concatenate([A, L, R])
        _, sv, vt = np.linalg.svd(P - P.mean(axis=0))
        plane_n = vt[2]
        k = np.cross(plane_n, t)
        if np.linalg.norm(k) < 1e-9:
            k = np.cross(t, (1.0, 0.0, 0.0) if abs(t[0]) < 0.9 else (0.0, 1.0, 0.0))
    nrm = _unit(k)
    return t, nrm, np.cross(t, nrm)


# ---------------------------------------------------------------- refit
def _corners(A, L, R, cyclic):
    n = len(A)
    ok = smooth_points(A, L, R, tol=1 - np.cos(np.radians(G1_DEG)))
    out = [i for i in range(n) if not ok[i]]
    if not cyclic:
        out = sorted(set(out) | {0, n - 1})
    return out


def refit(A, L, R, cyclic=False, count=None, tolerance=None, samples=24, max_points=200):
    """A chain with fewer (or more) points along the same shape.

    count: the number of anchors wanted (corners and the ends are always kept,
    the rest spread by arc length between them). tolerance: instead, the fewest
    anchors that keep every point within this distance. Returns (A, L, R, error).
    """
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    m = segments(len(A), cyclic)
    keep = [float(c) for c in _corners(A, L, R, cyclic)] or [0.0]
    tt = np.linspace(0.0, m, m * 64 + 1)
    pts = point_at(A, L, R, tt, cyclic)
    ss = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total_len = ss[-1]

    def arcu(x):                                 # arc length, past the end of a loop too
        return float(np.interp(x, tt, ss)) if x <= m else total_len + float(np.interp(x - m, tt, ss))

    def inv(s):
        return float(np.interp(s - total_len if s > total_len else s, ss, tt))

    bounds = keep + ([keep[0] + m] if cyclic else [])
    spans = list(zip(bounds[:-1], bounds[1:]))
    lens = [arcu(b) - arcu(a) for a, b in spans]
    least = max(len(keep), 3 if cyclic else 2)

    def params_for(total):
        extra = max(total - len(keep), 0)
        alloc = [0] * len(spans)
        for _ in range(extra):                   # each extra point to the longest piece
            k = int(np.argmax([ln / (c + 1) for ln, c in zip(lens, alloc)]))
            alloc[k] += 1
        out = []
        for (a, b), ln, c in zip(spans, lens, alloc):
            out.append(a)
            out += [inv(arcu(a) + ln * j / (c + 1)) for j in range(1, c + 1)]
        if not cyclic:
            out.append(float(m))
        return out

    def make(params):
        k = len(params)
        P = point_at(A, L, R, params, cyclic)
        nA, nL, nR = P.copy(), P.copy(), P.copy()
        err = 0.0
        eps = 1e-6
        for j in range(k if cyclic else k - 1):
            t0, t1 = params[j], params[(j + 1) % k]
            if cyclic and t1 <= t0:
                t1 += m
            ts = np.linspace(t0, t1, samples + 1)
            ts = np.where(ts > m, ts - m, ts)
            tb = t1 - m if t1 > m else t1
            d0 = point_at(A, L, R, [t0 + eps], cyclic, 1)[0]          # leaving t0
            d3 = point_at(A, L, R, [tb - eps if tb > eps else m - eps], cyclic, 1)[0]   # arriving
            p1, p2, e = fit_span(point_at(A, L, R, ts, cyclic), d0, d3)
            nR[j], nL[(j + 1) % k] = p1, p2
            err = max(err, e)
        if not cyclic:
            nL[0] = nA[0] - (nR[0] - nA[0])
            nR[-1] = nA[-1] + (nA[-1] - nL[-1])
        return nA, nL, nR, err

    if count is not None:
        return make(params_for(max(int(count), least)))
    best = None
    for total in range(least, max_points + 1):
        best = make(params_for(total))
        if tolerance is None or best[3] <= tolerance:
            break
    return best
