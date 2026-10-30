"""Editing Bezier curves of a curve network without disturbing its crossings.

A curve is a chain of cubic Bezier segments: anchors A, left handles L and
right handles R, each (n, 3), as Blender stores them. Its parameter t runs
0 .. n-1 (segment k covers k .. k+1).
"""
import numpy as np


def chain_point(A, L, R, t):
    """Points of the chain at parameters t, shape (len(t), 3)."""
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    t = np.atleast_1d(np.asarray(t, float))
    k = np.clip(np.floor(t).astype(int), 0, len(A) - 2)
    u = (t - k)[:, None]
    return ((1 - u) ** 3 * A[k] + 3 * (1 - u) ** 2 * u * R[k]
            + 3 * (1 - u) * u ** 2 * L[k + 1] + u ** 3 * A[k + 1])


def nearest_param(A, L, R, point, samples=48):
    """Chain parameter of the point closest to `point`."""
    n = len(A)
    t = np.linspace(0.0, n - 1.0, samples * (n - 1) + 1)
    d = np.linalg.norm(chain_point(A, L, R, t) - np.asarray(point, float), axis=1)
    i = int(np.argmin(d))
    lo, hi = t[max(i - 1, 0)], t[min(i + 1, len(t) - 1)]
    fine = np.linspace(lo, hi, 64)
    return float(fine[np.argmin(np.linalg.norm(chain_point(A, L, R, fine) - point, axis=1))])


def _rigid_row(n, t):
    """How the point at t moves when each Bezier point (anchor + both handles)
    is translated rigidly: the point's displacement is row @ d."""
    k = min(int(np.floor(t)), n - 2)
    u = t - k
    row = np.zeros(n)
    row[k] = (1 - u) ** 3 + 3 * (1 - u) ** 2 * u
    row[k + 1] = 3 * (1 - u) * u ** 2 + u ** 3
    return row


def pull_chain(A, L, R, t, delta, fixed=()):
    """Move the chain so its point at t moves by exactly `delta` while its
    points at the `fixed` parameters stay put.

    Each Bezier point moves rigidly with its handles (so the tangent
    directions, and G1 at the anchors, are kept); among all such moves the
    one with the least weighted motion is taken, weights growing with the
    distance from t so the edit stays local. Returns (A, L, R).
    """
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    n = len(A)
    rows = np.array([_rigid_row(n, t)] + [_rigid_row(n, f) for f in fixed])
    targets = np.zeros((len(rows), 3))
    targets[0] = delta
    # Reach: as far as the farther neighbouring fixed point (or end) plus a segment.
    left = max([f for f in fixed if f < t], default=-1.0)
    right = min([f for f in fixed if f > t], default=float(n))
    reach = max(t - left, right - t) + 1.0
    s = np.abs(np.arange(n) - t) / reach
    w = np.clip(1.0 - s * s, 0.0, 1.0) ** 3 + 1e-6
    lam = np.linalg.pinv((rows * w) @ rows.T) @ targets
    d = (w[:, None] * rows.T) @ lam
    return A + d, L + d, R + d


def fair_chain(A, L, R, fixed=(), strength=1.0):
    """Smooth the chain's control polygon (A0 R0 L1 A1 R1 ... An) by its
    second differences, keeping the end points, the end tangents and the
    points at the `fixed` parameters. strength 0..1 blends from the original
    to the fairest polygon. Handles are then lined up through their anchor
    so the chain stays G1. Returns (A, L, R).
    """
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    n = len(A)
    m = 3 * n - 2
    P0 = np.empty((m, 3))
    P0[0::3], P0[1::3], P0[2::3] = A, R[:-1], L[1:]
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    s = float(np.clip(strength, 0.0, 1.0))
    H = s * D.T @ D + (1.0 - s) * np.eye(m) + 1e-9 * np.eye(m)
    rows, targets = [], []
    for j in (0, 1, m - 2, m - 1):  # end points and end tangents
        r = np.zeros(m)
        r[j] = 1.0
        rows.append(r)
        targets.append(P0[j])
    for t in fixed:
        k = min(int(np.floor(t)), n - 2)
        u = t - k
        r = np.zeros(m)
        r[3 * k:3 * k + 4] = ((1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2, u ** 3)
        rows.append(r)
        targets.append(chain_point(A, L, R, [t])[0])
    C, b = np.array(rows), np.array(targets)
    kkt = np.block([[H, C.T], [C, np.zeros((len(C), len(C)))]])
    rhs = np.vstack([(1.0 - s) * P0, b])
    P = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:m]
    A2, R2, L2 = P[0::3].copy(), P[1::3], P[2::3]
    Ln, Rn = L.copy(), R.copy()
    Rn[:-1], Ln[1:] = R2, L2
    for i in range(1, n - 1):  # line the handles up through the anchor (G1)
        a, b_ = Ln[i] - A2[i], Rn[i] - A2[i]
        axis = b_ - a
        axis /= max(np.linalg.norm(axis), 1e-12)
        Ln[i] = A2[i] - np.linalg.norm(a) * axis
        Rn[i] = A2[i] + np.linalg.norm(b_) * axis
    return A2, Ln, Rn
