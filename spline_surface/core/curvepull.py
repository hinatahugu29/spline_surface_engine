"""Curve Pull: grab any point of a Bezier curve and pull, the rest follows softly.

The curve is a chain (anchors A, handles L / R, open or cyclic; see
core.curvepoint). Every control point stands for a place on the curve: an
anchor for its own point, a handle for the point a third of a segment towards
it. A control point moves by kernel weights of how far, along the curve, that
place is from where the curve was grabbed (falloff shapes as in Soft Pull),
so the pull spreads by arc length, not by distance in space.

Constraints are met exactly: the grabbed point moves by `delta`, the points at
the `fixed` parameters stay put (a kernel of the same width around each one
takes up the difference), pinned ends do not move (their handles neither, so
the end tangent is kept). Smooth points (aligned handles) are kept smooth: the
handles are lined up again and the constraints solved again, a few rounds;
a last correction moves each anchor with its handles as one piece, which
meets the constraints exactly without bending any handle.
"""
import numpy as np

from .curvepoint import arc_table, point_at, segments
from .softpull import falloff

MIN_GAIN = 0.05        # below this the grabbed point is (almost) pinned: leave the curve


def _ctrl_params(n, cyclic):
    """Curve parameter each control point stands for: (anchors, left, right)."""
    m = segments(n, cyclic)
    i = np.arange(n, dtype=float)
    left = i - 1.0 / 3.0
    right = i + 1.0 / 3.0
    if cyclic:
        left = np.where(left < 0, left + m, left)
    else:
        left = np.maximum(left, 0.0)
        right = np.minimum(right, m)
    return i, left, right


def _basis_rows(n, t, cyclic):
    """How the curve point at t is made of the control points: (wA, wL, wR), each (n,)."""
    m = segments(n, cyclic)
    t = float(np.clip(t, 0.0, m))
    k = min(int(np.floor(t)), m - 1)
    u = t - k
    j = (k + 1) % n if cyclic else k + 1
    wA, wL, wR = np.zeros(n), np.zeros(n), np.zeros(n)
    wA[k] += (1 - u) ** 3
    wR[k] += 3 * (1 - u) ** 2 * u
    wL[j] += 3 * (1 - u) * u ** 2
    wA[j] += u ** 3
    return wA, wL, wR


def smooth_points(A, L, R, tol=1e-4):
    """Anchors whose two handles are lined up (G1 there)."""
    a, b = A - L, R - A
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 1e-9) & (nb > 1e-9)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-18)
    return ok & (cos > 1 - tol)


def pull_curve(A, L, R, t, delta, radius, cyclic=False, profile='SMOOTH', pin_ends=True,
               fixed=(), keep_smooth=None, iterations=4):
    """The chain pulled so its point at t moves by delta. Returns (A, L, R)."""
    A, L, R = (np.array(x, float) for x in (A, L, R))
    n = len(A)
    delta = np.asarray(delta, float)
    if keep_smooth is None:
        keep_smooth = smooth_points(A, L, R)
    tt, ss = arc_table(A, L, R, cyclic)
    total = ss[-1]
    arc = lambda p: np.interp(p, tt, ss)
    pa, pl, pr = _ctrl_params(n, cyclic)
    sa, sl, sr = arc(pa), arc(pl), arc(pr)
    radius = max(float(radius), 1e-9)
    mask = np.ones(n)
    if pin_ends and not cyclic:
        mask[0] = mask[-1] = 0.0

    def kernel(at):
        s0 = arc(at)
        out = []
        for s in (sa, sl, sr):
            d = np.abs(s - s0)
            if cyclic:
                d = np.minimum(d, total - d)
            out.append(falloff(d / radius, profile) * mask)
        kA, kL, kR = out
        # a point's handles never lag behind it (with a small range they would kink the curve)
        return kA, np.maximum(kL, kA), np.maximum(kR, kA)

    points = [float(t)] + [float(f) for f in fixed]
    targets = np.zeros((len(points), 3))
    kernels = [kernel(p) for p in points]
    A0, L0, R0 = A.copy(), L.copy(), R.copy()
    goal = point_at(A0, L0, R0, [t], cyclic)[0] + delta
    stay = point_at(A0, L0, R0, [float(f) for f in fixed], cyclic) if len(fixed) else np.zeros((0, 3))
    for it in range(iterations):
        # Residual of every constraint with the chain as it is now.
        now = point_at(A, L, R, points, cyclic)
        targets[0] = goal - now[0]
        if len(fixed):
            targets[1:] = stay - now[1:]
        # How much each kernel moves each constraint point.
        M = np.zeros((len(points), len(points)))
        for r, p in enumerate(points):
            bA, bL, bR = _basis_rows(n, p, cyclic)
            for c, (kA, kL, kR) in enumerate(kernels):
                M[r, c] = bA @ kA + bL @ kL + bR @ kR
        if M[0, 0] < MIN_GAIN:                  # nothing free holds the grabbed point
            return A0, L0, R0
        x = np.linalg.lstsq(M, targets, rcond=None)[0]          # (k, 3)
        for (kA, kL, kR), xc in zip(kernels, x):
            A += kA[:, None] * xc
            L += kL[:, None] * xc
            R += kR[:, None] * xc
        # Smooth points stay smooth: line the handles up through the anchor again.
        for i in np.nonzero(keep_smooth)[0]:
            a, b = L[i] - A[i], R[i] - A[i]
            axis = b - a
            ln = np.linalg.norm(axis)
            if ln < 1e-12:
                continue
            axis /= ln
            L[i] = A[i] - np.linalg.norm(a) * axis
            R[i] = A[i] + np.linalg.norm(b) * axis
    # Last, meet the constraints exactly: each anchor moves with its two handles
    # (a pure translation keeps every point as smooth as it is).
    rigid = [(kA, kA, kA) for kA, _, _ in kernels]
    now = point_at(A, L, R, points, cyclic)
    targets[0] = goal - now[0]
    if len(fixed):
        targets[1:] = stay - now[1:]
    M = np.zeros((len(points), len(points)))
    for r, p in enumerate(points):
        bA, bL, bR = _basis_rows(n, p, cyclic)
        for c, (kA, kL, kR) in enumerate(rigid):
            M[r, c] = bA @ kA + bL @ kL + bR @ kR
    x = np.linalg.lstsq(M, targets, rcond=None)[0]
    for (kA, _, _), xc in zip(rigid, x):
        A += kA[:, None] * xc
        L += kA[:, None] * xc
        R += kA[:, None] * xc
    return A, L, R


def chain_plane(A, L, R, tol=1e-4):
    """(point, unit normal) of the plane the chain lies in, or None if it is not flat."""
    P = np.concatenate([A, L, R])
    c = P.mean(axis=0)
    size = np.ptp(P, axis=0).max()
    if size < 1e-12:
        return None
    _, sv, vt = np.linalg.svd(P - c)
    if sv[0] < 1e-12:
        return None
    normal = vt[2]
    if np.abs((P - c) @ normal).max() > tol * size:
        return None
    if sv[1] < 1e-9 * sv[0]:             # a straight line: any plane through it; none chosen
        return None
    return c, normal


def pull_gain(A, L, R, t, radius, cyclic=False, profile='SMOOTH', pin_ends=True):
    """How strongly the free control points in range hold the curve point at t
    (1: it is a free point itself; 0: every control point near it is pinned or
    out of range). Below about a third, add a point at t before pulling."""
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    n = len(A)
    tt, ss = arc_table(A, L, R, cyclic)
    total = ss[-1]
    pa, pl, pr = _ctrl_params(n, cyclic)
    mask = np.ones(n)
    if pin_ends and not cyclic:
        mask[0] = mask[-1] = 0.0
    s0 = np.interp(t, tt, ss)
    ks = []
    for p in (pa, pl, pr):
        d = np.abs(np.interp(p, tt, ss) - s0)
        if cyclic:
            d = np.minimum(d, total - d)
        ks.append(falloff(d / max(float(radius), 1e-9), profile) * mask)
    kA, kL, kR = ks[0], np.maximum(ks[1], ks[0]), np.maximum(ks[2], ks[0])
    bA, bL, bR = _basis_rows(n, t, cyclic)
    return float(bA @ kA + bL @ kL + bR @ kR)
