"""Curvature of a patch: combs along iso-curves and Gaussian / mean maps."""
import numpy as np

from .nurbs import eval_points


def _grid(a, b):
    A, B = np.meshgrid(a, b, indexing="ij")
    return A.ravel(), B.ravel()


def comb(net, orders, direction, count=7, samples=40):
    """Curvature along `count` iso-curves running in `direction` ('U' or 'V').

    Returns (points, kvec), both shaped (count, samples, 3): kvec is the
    curvature vector (towards the centre of curvature, length 1 / radius).
    """
    t = np.linspace(0.0, 1.0, samples)
    w = np.linspace(0.0, 1.0, count)
    if direction == 'U':
        W, T = _grid(w, t)
        u, v, d1, d2 = T, W, (1, 0), (2, 0)
    else:
        W, T = _grid(w, t)
        u, v, d1, d2 = W, T, (0, 1), (0, 2)
    P = eval_points(net, orders, u, v)
    C1 = eval_points(net, orders, u, v, *d1)
    C2 = eval_points(net, orders, u, v, *d2)
    speed2 = np.einsum("ij,ij->i", C1, C1)[:, None]
    tangential = np.einsum("ij,ij->i", C2, C1)[:, None] * C1 / np.maximum(speed2, 1e-24)
    kvec = (C2 - tangential) / np.maximum(speed2, 1e-24)
    return P.reshape(count, samples, 3), kvec.reshape(count, samples, 3)


def surface_curvature(net, orders, samples=24):
    """Points, unit normals, Gaussian and mean curvature on a samples x samples grid.

    Returns (points (s, s, 3), normals (s, s, 3), K (s, s), H (s, s)). The
    sign of H follows the patch normal cross(S_u, S_v): positive when the
    surface bends towards it.
    """
    s = np.linspace(0.0, 1.0, samples)
    u, v = _grid(s, s)
    # Clamp off collapsed corners/edges, where S_u x S_v vanishes.
    u, v = np.clip(u, 1e-4, 1 - 1e-4), np.clip(v, 1e-4, 1 - 1e-4)
    P = eval_points(net, orders, u, v)
    Su, Sv = eval_points(net, orders, u, v, 1, 0), eval_points(net, orders, u, v, 0, 1)
    Suu = eval_points(net, orders, u, v, 2, 0)
    Suv = eval_points(net, orders, u, v, 1, 1)
    Svv = eval_points(net, orders, u, v, 0, 2)
    N = np.cross(Su, Sv)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-24)
    dot = lambda a, b: np.einsum("ij,ij->i", a, b)
    E, F, G = dot(Su, Su), dot(Su, Sv), dot(Sv, Sv)
    L, M, Nn = dot(Suu, N), dot(Suv, N), dot(Svv, N)
    det = np.maximum(E * G - F * F, 1e-24)
    K = (L * Nn - M * M) / det
    H = (E * Nn - 2 * F * M + G * L) / (2 * det)
    shape = (samples, samples)
    return P.reshape(shape + (3,)), N.reshape(shape + (3,)), K.reshape(shape), H.reshape(shape)


def color_ramp(values, limit):
    """RGBA for curvature values: blue (negative) - green (zero) - red (positive),
    saturating at +-limit."""
    x = np.clip(np.asarray(values, float) / max(limit, 1e-24), -1.0, 1.0)
    r = np.clip(x, 0.0, 1.0)
    b = np.clip(-x, 0.0, 1.0)
    g = 1.0 - np.abs(x)
    return np.stack([0.15 + 0.85 * r, 0.25 + 0.6 * g, 0.15 + 0.85 * b,
                     np.full_like(x, 0.55)], axis=-1)
