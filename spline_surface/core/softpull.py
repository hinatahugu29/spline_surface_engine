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


PROFILES = ('SMOOTH', 'SPHERE', 'LINEAR', 'SHARP', 'CONSTANT')


def falloff(s, profile='SMOOTH'):
    """Weight for normalised distance s in [0, 1]: 1 at the origin, 0 at the rim.

    Same shapes as Blender's proportional editing; CONSTANT moves everything
    inside the rim as one piece.
    """
    s = np.clip(s, 0.0, 1.0)
    if profile == 'SPHERE':
        return np.sqrt(1.0 - s * s)
    if profile == 'LINEAR':
        return 1.0 - s
    if profile == 'SHARP':
        return (1.0 - s) ** 2
    if profile == 'CONSTANT':
        return (s < 1.0).astype(float)
    return (1.0 - s * s) ** 3


def greville(n, k):
    """Parameter at which each control point of a clamped uniform B-spline peaks."""
    U = clamped_knots(n, k)
    return np.array([U[i + 1:i + k].mean() for i in range(n)])


def pull_weights(net, orders, mask, anchor, radius, profile='SMOOTH'):
    """Falloff weight (nu, nv) of every control point; 0 for pinned ones.

    Distance is measured from the anchor to the surface point each control
    point stands for (its Greville image), so a net bulging far off the
    surface still pulls in a sensible neighbourhood.
    """
    net = np.asarray(net, float)
    pos = eval_surface(net, orders, greville(net.shape[0], orders[0]),
                       greville(net.shape[1], orders[1]))
    w = falloff(np.linalg.norm(pos - anchor, axis=-1) / max(radius, 1e-9), profile)
    w[np.asarray(mask, bool)] = 0.0
    return w


def line_weights(net, orders, mask, direction, t0, width, taper=False, profile='SMOOTH'):
    """Falloff weight (nu, nv) of every control point around an iso-line.

    The line is u = t0 ('U', running along v) or v = t0 ('V', along u). Each
    control point's Greville image is measured against the line point at the
    same along-parameter, so the region is a band of half-width `width`.
    `taper` fades the band to nothing at both ends of the line.
    """
    net = np.asarray(net, float)
    gu, gv = greville(net.shape[0], orders[0]), greville(net.shape[1], orders[1])
    pos = eval_surface(net, orders, gu, gv)
    if direction == 'U':
        line = eval_surface(net, orders, [t0], gv)[0][None, :, :]
        along = gv[None, :]
    else:
        line = eval_surface(net, orders, gu, [t0])[:, 0][:, None, :]
        along = gu[:, None]
    w = falloff(np.linalg.norm(pos - line, axis=-1) / max(width, 1e-9), profile)
    if taper:
        w = w * (4.0 * along * (1.0 - along))
    w[np.asarray(mask, bool)] = 0.0
    return w


def line_pull(net, orders, mask, direction, t0, along, delta, width, taper=False,
              profile='SMOOTH'):
    """Move the iso-line (see line_weights) so its point at `along` moves by
    `delta`. Returns (new net, gain) like soft_pull."""
    net = np.asarray(net, float)
    w = line_weights(net, orders, mask, direction, t0, width, taper, profile)
    u, v = (t0, along) if direction == 'U' else (along, t0)
    gain = float(basis_matrix(net.shape[0], orders[0], [u])[0]
                 @ w @ basis_matrix(net.shape[1], orders[1], [v])[0])
    if gain < MIN_GAIN:
        return net.copy(), 0.0
    return net + (w / max(gain, GAIN_FLOOR))[..., None] * np.asarray(delta, float), gain


def grid_weights(shape, mask, i0, j0, rows_u, rows_v, profile='SMOOTH'):
    """Falloff weight (nu, nv) around control point (i0, j0), measured in rows:
    the region reaches `rows_u` rows along u and `rows_v` along v (an ellipse
    in the control grid)."""
    i = np.arange(shape[0])[:, None]
    j = np.arange(shape[1])[None, :]
    s = np.hypot((i - i0) / max(rows_u, 1e-9), (j - j0) / max(rows_v, 1e-9))
    w = falloff(s, profile)
    w[np.asarray(mask, bool)] = 0.0
    return w


def point_pull(net, mask, i0, j0, delta, rows_u, rows_v, profile='SMOOTH'):
    """Move control point (i0, j0) by exactly `delta`, its neighbours by their
    grid falloff. Returns (new net, gain); gain is 0 if the point is pinned."""
    net = np.asarray(net, float)
    if np.asarray(mask, bool)[i0, j0]:
        return net.copy(), 0.0
    w = grid_weights(net.shape[:2], mask, i0, j0, rows_u, rows_v, profile)
    return net + w[..., None] * np.asarray(delta, float), 1.0


def soft_pull(net, orders, mask, uv, delta, radius, profile='SMOOTH'):
    """Move the surface point at `uv` by `delta`. Returns (new net, gain).

    `gain` is how much of the pull the anchor can follow (0: it is pinned and
    nothing moved). Pinned control points are left exactly as they were.
    """
    net = np.asarray(net, float)
    u, v = uv
    anchor = eval_points(net, orders, [u], [v])[0]
    w = pull_weights(net, orders, mask, anchor, radius, profile)
    gain = float(basis_matrix(net.shape[0], orders[0], [u])[0]
                 @ w @ basis_matrix(net.shape[1], orders[1], [v])[0])
    if gain < MIN_GAIN:
        return net.copy(), 0.0
    return net + (w / max(gain, GAIN_FLOOR))[..., None] * np.asarray(delta, float), gain
