"""Bezier curve / surface math. Pure numpy, no Blender dependency.

Conventions
-----------
- Curve control points: array of shape (n+1, 3), degree n.
- Surface control net: array of shape (n+1, m+1, 3); first index runs along U
  (degree n), second along V (degree m).
"""
from math import comb

import numpy as np


def bernstein(n, t):
    """Bernstein basis matrix, shape (len(t), n+1)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
    i = np.arange(n + 1)[None, :]
    c = np.array([comb(n, k) for k in range(n + 1)], dtype=float)[None, :]
    return c * t ** i * (1.0 - t) ** (n - i)


def eval_curve(P, t):
    P = np.asarray(P, dtype=float)
    return bernstein(len(P) - 1, t) @ P


def derivative(P):
    """Control points of the hodograph (degree n-1)."""
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    return n * (P[1:] - P[:-1])


def eval_surface(P, u, v):
    """Evaluate a surface on the grid u x v, shape (len(u), len(v), 3)."""
    P = np.asarray(P, dtype=float)
    Bu = bernstein(P.shape[0] - 1, u)
    Bv = bernstein(P.shape[1] - 1, v)
    return np.einsum("ai,ijd,bj->abd", Bu, P, Bv)


def elevate_curve(P, target_degree):
    """Raise the degree of a curve without changing its shape."""
    P = np.asarray(P, dtype=float)
    while len(P) - 1 < target_degree:
        n = len(P) - 1
        a = (np.arange(1, n + 1) / (n + 1))[:, None]
        mid = a * P[:-1] + (1.0 - a) * P[1:]
        P = np.vstack([P[0], mid, P[-1]])
    return P


def chord_params(pts):
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)])
    if t[-1] <= 0.0:
        raise ValueError("degenerate polyline (zero length)")
    return t / t[-1]


def _project_params(P, pts, t, steps=4):
    """Refine curve parameters so each eval_curve(P, t[k]) is closest to pts[k]."""
    D1 = derivative(P)
    D2 = derivative(D1)
    for _ in range(steps):
        diff = eval_curve(P, t) - pts
        c1 = eval_curve(D1, t)
        c2 = eval_curve(D2, t)
        num = (diff * c1).sum(axis=1)
        den = (c1 * c1).sum(axis=1) + (diff * c2).sum(axis=1)
        ok = den > 1e-12
        step = np.divide(num, den, out=np.zeros_like(num), where=ok)
        t = np.clip(t - step, 0.0, 1.0)
    t[0], t[-1] = 0.0, 1.0
    return np.maximum.accumulate(t)


def fit_curve(pts, degree, iterations=30):
    """Least-squares Bezier fit through a polyline.

    The end points are interpolated exactly so that patch corners stay put.
    Parameters start from chord length and are refined by alternating a
    least-squares solve with projecting the samples onto the curve.
    Returns (control_points, max_error).
    """
    if degree < 2:
        raise ValueError("degree must be >= 2")
    pts = np.asarray(pts, dtype=float)
    if len(pts) < degree + 1:
        # Too few samples: densify linearly so the system is well posed.
        s = chord_params(pts)
        dense = np.linspace(0.0, 1.0, (degree + 1) * 4)
        pts = np.stack([np.interp(dense, s, pts[:, k]) for k in range(3)], axis=1)
    t = chord_params(pts)
    P0, Pn = pts[0], pts[-1]

    def solve_points(t):
        B = bernstein(degree, t)
        rhs = pts - np.outer(B[:, 0], P0) - np.outer(B[:, -1], Pn)
        inner, *_ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
        return np.vstack([P0, inner, Pn])

    # A few cheap alternating passes give a good starting point ...
    for _ in range(5):
        P = solve_points(t)
        t = _project_params(P, pts, t)
    P = solve_points(t)

    # ... then Levenberg-Marquardt on control points and parameters together,
    # which converges much faster than alternating alone.
    k, ni = len(pts), degree - 1
    cost = lambda P, t: float(((eval_curve(P, t) - pts) ** 2).sum())  # noqa: E731
    c, lam = cost(P, t), 1e-3
    for _ in range(iterations):
        r = (eval_curve(P, t) - pts).reshape(-1)
        B = bernstein(degree, t)[:, 1:-1]
        dC = eval_curve(derivative(P), t)
        J = np.zeros((3 * k, 3 * ni + k - 2))
        for d in range(3):
            J[d::3, d:3 * ni:3] = B
        J[np.arange(3, 3 * k - 3).reshape(-1, 3), 3 * ni + np.arange(k - 2)[:, None]] = dC[1:-1]
        JTJ, g = J.T @ J, J.T @ r
        while True:
            delta = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), -g)
            P_new = P.copy()
            P_new[1:-1] += delta[:3 * ni].reshape(ni, 3)
            t_new = t.copy()
            t_new[1:-1] = np.clip(t[1:-1] + delta[3 * ni:], 0.0, 1.0)
            c_new = cost(P_new, t_new)
            if c_new < c:
                P, t, c, lam = P_new, t_new, c_new, lam * 0.3
                break
            lam *= 10.0
            if lam > 1e8:
                break
        if lam > 1e8 or c < 1e-20:
            break
    t = _project_params(P, pts, t)
    err = float(np.linalg.norm(eval_curve(P, t) - pts, axis=1).max())
    return P, err
