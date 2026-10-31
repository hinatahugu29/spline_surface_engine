"""Fair (smooth out the interior) and Refine (more control points, same shape).

Both work on control nets (numpy arrays) and know nothing about Blender.
"""
import numpy as np

from .bezier import elevate_curve
from .nurbs import basis_matrix, clamped_knots


# ---------------------------------------------------------------- refine

def _insert_knot(P, U, p, t):
    """Boehm knot insertion of `t` into a curve of degree `p`; returns (P, U)."""
    n = len(P)
    s = int(np.searchsorted(U, t, side="right")) - 1
    s = min(max(s, p), n - 1)
    Q = np.empty((n + 1,) + P.shape[1:])
    for i in range(n + 1):
        if i <= s - p:
            Q[i] = P[i]
        elif i > s:
            Q[i] = P[i - 1]
        else:
            a = (t - U[i]) / (U[i + p] - U[i])
            Q[i] = a * P[i] + (1.0 - a) * P[i - 1]
    return Q, np.insert(U, s + 1, t)


def refine_curve(P, k):
    """Same curve with every span split in two: n points -> n + (n - k + 1).

    The result is again a clamped uniform B-spline of order k, so Blender can
    store it, and its shape is unchanged.
    """
    P = np.asarray(P, float)
    U = clamped_knots(len(P), k)
    breaks = np.unique(U)
    for a, b in zip(breaks[:-1], breaks[1:]):
        P, U = _insert_knot(P, U, k - 1, 0.5 * (a + b))
    return P


def refine_net(net, orders, direction='BOTH'):
    """Split every span along u, v or both. Returns (net, orders); shape unchanged."""
    net = np.asarray(net, float)
    if direction in ('U', 'BOTH'):
        net = np.stack([refine_curve(net[:, j], orders[0]) for j in range(net.shape[1])], axis=1)
    if direction in ('V', 'BOTH'):
        net = np.stack([refine_curve(net[i], orders[1]) for i in range(net.shape[0])], axis=0)
    return net, tuple(orders)


def _insert_line_curves(lines, k):
    """One more point on every curve in `lines` (m, n, 3): (new lines, exact)."""
    n = lines.shape[1]
    if n == k and k < 6:                       # a single span is a Bezier: raise its degree
        return np.stack([elevate_curve(c, n) for c in lines]), True
    t = np.linspace(0.0, 1.0, max(8 * (n + 1), 32))
    old = basis_matrix(n, k, t)
    new = basis_matrix(n + 1, k, t)
    target = np.einsum("ti,mid->mtd", old, lines)
    first, last = lines[:, 0], lines[:, -1]
    rhs = target - new[None, :, 0, None] * first[:, None] - new[None, :, -1, None] * last[:, None]
    inner = np.linalg.lstsq(new[:, 1:-1], rhs.transpose(1, 0, 2).reshape(len(t), -1), rcond=None)[0]
    inner = inner.reshape(n - 1, len(lines), 3).transpose(1, 0, 2)
    return np.concatenate([first[:, None], inner, last[:, None]], axis=1), False


def insert_line(net, orders, axis):
    """One more row (axis 0, U) or column (axis 1, V) of control points.

    A single-span direction (points == order) is exact: its degree is raised.
    Otherwise the knots of Blender's clamped uniform B-spline cannot take a
    single new knot, so the net is refitted with one more point (ends fixed)
    and the shape moves slightly. Returns (net, orders, exact).
    """
    net = np.asarray(net, float)
    k = orders[axis]
    if axis == 0:
        new, exact = _insert_line_curves(net.transpose(1, 0, 2), k)
        net = new.transpose(1, 0, 2)
    else:
        new, exact = _insert_line_curves(net, k)
        net = new
    out = list(orders)
    if exact and net.shape[axis] - 1 == k:
        out[axis] = k + 1
    return net, tuple(out), exact


# ---------------------------------------------------------------- fair

def _fairing_rows(nu, nv):
    """Sparse rows (as dense index/coefficient lists) of the second differences."""
    rows = []
    for i in range(nu):
        for j in range(nv):
            if 0 < i < nu - 1:
                rows.append([((i - 1, j), 1.0), ((i, j), -2.0), ((i + 1, j), 1.0)])
            if 0 < j < nv - 1:
                rows.append([((i, j - 1), 1.0), ((i, j), -2.0), ((i, j + 1), 1.0)])
            if i < nu - 1 and j < nv - 1:
                w = np.sqrt(2.0)  # the twist counts twice in the thin-plate energy
                rows.append([((i, j), w), ((i + 1, j), -w), ((i, j + 1), -w), ((i + 1, j + 1), w)])
    return rows


def fair_net(net, fixed, strength=1.0):
    """Smooth the free control points, leaving `fixed` ones (bool mask) as they are.

    Minimises the squared second differences of the net (a thin-plate-like
    energy) over the free points; `strength` blends from the original (0) to
    the fully faired net (1). Returns the new net.
    """
    net = np.asarray(net, float)
    nu, nv = net.shape[:2]
    fixed = np.asarray(fixed, bool)
    free = ~fixed
    if not free.any() or strength <= 0.0:
        return net.copy()
    index = -np.ones((nu, nv), int)
    index[free] = np.arange(int(free.sum()))
    rows = _fairing_rows(nu, nv)
    A = np.zeros((len(rows), int(free.sum())))
    b = np.zeros((len(rows), 3))
    for r, row in enumerate(rows):
        for (i, j), c in row:
            if free[i, j]:
                A[r, index[i, j]] = c
            else:
                b[r] -= c * net[i, j]
    # A faint pull towards the original keeps the problem determined when little is fixed.
    ridge = 1e-3
    n_free = int(free.sum())
    A = np.vstack([A, ridge * np.eye(n_free)])
    b = np.vstack([b, ridge * net[free]])
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    out = net.copy()
    out[free] = net[free] + float(strength) * (x - net[free])
    return out
