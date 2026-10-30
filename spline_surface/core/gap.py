"""Panel gaps: the clearance between facing edges of neighbouring patches.

Two panels of a car sit a few millimetres apart; the gap should be even along
the whole joint and the two surfaces flush. For two edge polylines this
measures, at every sample of the first, the distance to the closest point of
the second, split into the gap proper (across the surface) and the step
(along the surface normal, i.e. how far one panel sits proud of the other).
"""
import numpy as np

from .nurbs import eval_points


def closest_on_polyline(points, line):
    """Closest point of polyline `line` to every row of `points`: (points, distances)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-18)
    # (points, segments, 3) offsets; segment counts are small (tens)
    t = np.clip(((points[:, None] - a[None]) * ab[None]).sum(2) / denom[None], 0.0, 1.0)
    cand = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(cand - points[:, None], axis=2)
    k = np.argmin(d, axis=1)
    idx = np.arange(len(points))
    return cand[idx, k], d[idx, k]


def edge_normals(net, orders, edge, samples):
    """Unit surface normals at `samples` points along an edge, matching edge_polyline's order."""
    t = np.linspace(0.0, 1.0, samples)
    zero, one = np.zeros(samples), np.ones(samples)
    u, v = {'U0': (zero, t), 'U1': (one, t), 'V0': (t, zero), 'V1': (t, one)}[edge]
    du = eval_points(net, orders, u, v, 1, 0)
    dv = eval_points(net, orders, u, v, 0, 1)
    n = np.cross(du, dv)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def measure_gap(line_a, line_b, normals_a=None):
    """Gap and step from every sample of `line_a` to `line_b`.

    Returns dict(a, b, dist, gap, step): the sample points, their closest
    points on b, the straight distance, and its split across / along the
    normal (step is zeros when no normals are given).
    """
    line_a, line_b = np.asarray(line_a, float), np.asarray(line_b, float)
    near, dist = closest_on_polyline(line_a, line_b)
    vec = near - line_a
    if normals_a is None:
        step = np.zeros(len(line_a))
        gap = dist
    else:
        step = np.abs((vec * normals_a).sum(1))
        gap = np.sqrt(np.maximum(dist ** 2 - step ** 2, 0.0))
    return {"a": line_a, "b": near, "dist": dist, "gap": gap, "step": step}


def find_gap_pairs(edges, search, touch=1e-6, coverage=0.7):
    """Pairs of facing edges from different patches.

    edges: {(patch, edge): polyline}. Two edges face each other when at least
    `coverage` of each one's samples lie within `search` of the other and they
    are not touching (mean distance above `touch`). Returns [(key_a, key_b)].
    """
    keys = list(edges)
    out = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka[0] == kb[0]:
                continue
            la, lb = np.asarray(edges[ka], float), np.asarray(edges[kb], float)
            pad = search
            if np.any(la.max(0) + pad < lb.min(0)) or np.any(lb.max(0) + pad < la.min(0)):
                continue
            _, da = closest_on_polyline(la, lb)
            _, db = closest_on_polyline(lb, la)
            if (da <= search).mean() >= coverage and (db <= search).mean() >= coverage \
                    and da.mean() > touch:
                out.append((ka, kb))
    return out
