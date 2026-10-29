"""Soft Pull: drag one point of a patch and let nearby free control points follow.

Control points pinned by seams (core.lock) never move, so every seam keeps its
G0/G1/G2. Works on world-space control nets and knows nothing about Blender.
"""
import numpy as np

from .nurbs import basis_matrix, clamped_knots, eval_points, eval_surface

# Below this the anchor sits so deep in the pinned rows that pulling it is pointless.
MIN_GAIN = 1e-3
# Normalising by the anchor's gain makes it follow the cursor; cap it so a point
# only weakly influenced by free control points cannot fling them far away.
GAIN_FLOOR = 0.3


def greville(n, k):
    """Parameter at which each control point of a clamped uniform B-spline peaks."""
    U = clamped_knots(n, k)
    return np.array([U[i + 1:i + k].mean() for i in range(n)])


def pull_weights(net, orders, mask, anchor, radius):
    """Falloff weight (nu, nv) of every control point; 0 for pinned ones.

    Distance is measured from the anchor to the surface point each control
    point stands for (its Greville image), so a net bulging far off the
    surface still pulls in a sensible neighbourhood.
    """
    net = np.asarray(net, float)
    pos = eval_surface(net, orders, greville(net.shape[0], orders[0]),
                       greville(net.shape[1], orders[1]))
    s = np.clip(np.linalg.norm(pos - anchor, axis=-1) / max(radius, 1e-9), 0.0, 1.0)
    w = (1.0 - s * s) ** 3
    w[np.asarray(mask, bool)] = 0.0
    return w


def soft_pull(net, orders, mask, uv, delta, radius):
    """Move the surface point at `uv` by `delta`. Returns (new net, gain).

    `gain` is how much of the pull the anchor can follow (0: it is pinned and
    nothing moved). Pinned control points are left exactly as they were.
    """
    net = np.asarray(net, float)
    u, v = uv
    anchor = eval_points(net, orders, [u], [v])[0]
    w = pull_weights(net, orders, mask, anchor, radius)
    gain = float(basis_matrix(net.shape[0], orders[0], [u])[0]
                 @ w @ basis_matrix(net.shape[1], orders[1], [v])[0])
    if gain < MIN_GAIN:
        return net.copy(), 0.0
    return net + (w / max(gain, GAIN_FLOOR))[..., None] * np.asarray(delta, float), gain
