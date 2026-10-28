"""Clamped uniform B-spline basis, matching Blender's NURBS with Endpoint on.

A NURBS direction with `n` points and order `k` (degree k-1) uses the knot
vector [0]*k, uniform interior knots, [1]*k. With n == k this is exactly the
Bernstein (Bezier) basis.
"""
import numpy as np


def clamped_knots(n, k):
    if not 2 <= k <= n:
        raise ValueError("order %d needs 2 <= order <= points (%d)" % (k, n))
    interior = np.arange(1, n - k + 1) / (n - k + 1)
    return np.concatenate([np.zeros(k), interior, np.ones(k)])


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def basis_matrix(n, k, t, deriv=0):
    """Basis functions (or their `deriv`-th derivative), shape (len(t), n)."""
    U = clamped_knots(n, k)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    P = k - 1
    if deriv > P:
        return np.zeros((len(t), n))
    m = len(U) - 1
    N = ((U[None, :-1] <= t[:, None]) & (t[:, None] < U[None, 1:])).astype(float)
    last = max(i for i in range(m) if U[i] < U[i + 1])
    at_end = t >= U[-1]
    N[at_end] = 0.0
    N[at_end, last] = 1.0
    # Values up to degree P - deriv ...
    for q in range(1, P - deriv + 1):
        i = np.arange(m - q)
        left = _div(t[:, None] - U[i], np.broadcast_to(U[i + q] - U[i], (len(t), len(i))))
        right = _div(U[i + q + 1] - t[:, None],
                     np.broadcast_to(U[i + q + 1] - U[i + 1], (len(t), len(i))))
        N = left * N[:, :-1] + right * N[:, 1:]
    # ... then one derivative per remaining degree step.
    for q in range(P - deriv + 1, P + 1):
        i = np.arange(m - q)
        a = np.where(U[i + q] - U[i] != 0, q / np.where(U[i + q] - U[i] != 0, U[i + q] - U[i], 1), 0)
        b = np.where(U[i + q + 1] - U[i + 1] != 0,
                     q / np.where(U[i + q + 1] - U[i + 1] != 0, U[i + q + 1] - U[i + 1], 1), 0)
        N = a * N[:, :-1] - b * N[:, 1:]
    return N


def eval_curve(P, k, t, deriv=0):
    P = np.asarray(P, dtype=float)
    return basis_matrix(len(P), k, t, deriv) @ P


def eval_surface(P, orders, u, v, du=0, dv=0):
    """Evaluate a surface (or a partial derivative) on the grid u x v."""
    P = np.asarray(P, dtype=float)
    Bu = basis_matrix(P.shape[0], orders[0], u, du)
    Bv = basis_matrix(P.shape[1], orders[1], v, dv)
    return np.einsum("ai,ijd,bj->abd", Bu, P, Bv)


def eval_points(P, orders, a, b, da=0, db=0):
    """Evaluate at paired parameters (a[k], b[k]), shape (len(a), 3)."""
    P = np.asarray(P, dtype=float)
    Ba = basis_matrix(P.shape[0], orders[0], a, da)
    Bb = basis_matrix(P.shape[1], orders[1], b, db)
    return np.einsum("ki,ijd,kj->kd", Ba, P, Bb)


def project_on_curve(P, k, pts, t, steps=8):
    """Newton-refine parameters t so eval_curve(P, k, t) is closest to pts."""
    for _ in range(steps):
        diff = eval_curve(P, k, t) - pts
        c1 = eval_curve(P, k, t, 1)
        c2 = eval_curve(P, k, t, 2)
        num = (diff * c1).sum(axis=1)
        den = (c1 * c1).sum(axis=1) + (diff * c2).sum(axis=1)
        step = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        t = np.clip(t - step, 0.0, 1.0)
    return t


def fit_polyline(pts, n, k):
    """Control points (n, order k) of a curve through a polyline, ends fixed.

    Returns (points, max deviation from the polyline).
    """
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-12)
    B = basis_matrix(n, k, t)
    inner = B[:, 1:-1]
    rhs = pts - np.outer(B[:, 0], pts[0]) - np.outer(B[:, -1], pts[-1])
    mid = np.linalg.lstsq(inner, rhs, rcond=None)[0]
    P = np.vstack([pts[0], mid, pts[-1]])
    return P, float(np.linalg.norm(B @ P - pts, axis=1).max())
