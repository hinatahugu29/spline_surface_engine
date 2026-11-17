"""Panel gaps: the clearance between facing edges of neighbouring patches.

Two panels of a car sit a few millimetres apart; the gap should be even along
the whole joint and the two surfaces flush. For two edge polylines this
measures, at every sample of the first, the distance to the closest point of
the second, split into the gap proper (across the surface) and the step
(along the surface normal, i.e. how far one panel sits proud of the other).
"""
import numpy as np

from .extend import bezier_orders, edge_frame, frame_orders
from .nurbs import basis_matrix, eval_curve, eval_points


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


def find_gap_pairs(edges, search, touch=1e-6, coverage=0.7, partial=False):
    """Pairs of facing edges from different patches.

    edges: {(patch, edge): polyline}. Two edges face each other when at least
    `coverage` of each one's samples lie within `search` of the other and they
    are not touching (mean distance above `touch`). With `partial`, joints that
    overlap only in part count too: each edge covered for a sixth of its
    length and one of them for a quarter (a door's front edge running past the
    end of the fender's rear edge; a side window over the rear of a door).
    Returns [(key_a, key_b)].
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
            ca, cb = (da <= search).mean(), (db <= search).mean()
            if partial:
                covered = max(ca, cb) >= 0.25 and min(ca, cb) >= 1.0 / 6.0
            else:
                covered = ca >= coverage and cb >= coverage
            if covered and da.mean() > touch:
                out.append((ka, kb))
    return out


def _framed_indices(shape, edge):
    idx = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), -1)
    return edge_frame(idx, edge)[0].astype(int)


def even_gap_net(net, orders, edge, line_b, target, flush=True, iterations=4, samples=None,
                 reach=None):
    """Move `edge` of a patch so it runs `target` away from polyline `line_b`.

    The gap is measured across the surface, as measure_gap does: at every
    sample of the edge the closest point of `line_b` is found, and the edge
    point is put `target` from it, in the patch's tangent plane, on the side it
    already sits. With `flush` the step along the normal is removed too (the
    two panels end level); without it the step is kept.

    With `reach`, only the part of the edge that faces `line_b` is evened: the
    samples further than `reach` from it, or past its ends, keep their place
    (a door edge longer than the fender edge it meets).

    Only the edge row and the one or two rows inside it move, all by the same
    amount, so the slope and curvature across the edge are kept and the
    opposite edge is not touched. Returns (net, orders, (worst gap error before,
    after)).
    """
    net = np.asarray(net, dtype=float)
    orders = tuple(orders or bezier_orders(net))
    line_b = np.asarray(line_b, dtype=float)
    C = edge_frame(net, edge)[0].copy()           # axis 0 along the edge, the edge is C[:, -1]
    ka, kc = frame_orders(orders, edge)
    n, m = C.shape[:2]
    rows = max(1, min(3, m - 1))                  # the first row (opposite edge) stays put
    S = samples or max(24, 4 * n)
    t = np.linspace(0.0, 1.0, S)
    B = basis_matrix(n, ka, t)                    # (S, n): edge samples from edge control points
    d_across = basis_matrix(m, kc, 1.0, 1)[0]

    def errors(C):
        pts = B @ C[:, -1]
        along = eval_curve(C[:, -1], ka, t, 1)
        across = B @ np.einsum("j,ijd->id", d_across, C)
        nrm = np.cross(along, across)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        across_u = across / np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-12)
        near, _ = closest_on_polyline(pts, line_b)
        vec = pts - near
        h = (vec * nrm).sum(1)
        w = vec - h[:, None] * nrm
        wl = np.linalg.norm(w, axis=1)
        # with no in-plane offset yet (touching), push outwards along the patch
        dirn = np.where(wl[:, None] > 1e-9, w / np.maximum(wl, 1e-12)[:, None], across_u)
        want = near + target * dirn + (0.0 if flush else h[:, None] * nrm)
        want = np.where(facing[:, None], want, pts)
        return pts, want, np.abs(wl - target)[facing].max()

    facing = np.ones(S, bool)
    if reach is not None:
        pts0 = B @ C[:, -1]
        near0, d0 = closest_on_polyline(pts0, line_b)
        at_end = np.minimum(np.linalg.norm(near0 - line_b[0], axis=1),
                            np.linalg.norm(near0 - line_b[-1], axis=1)) < 1e-9 + 1e-6 * reach
        facing = (d0 <= reach) & ~at_end
        if not facing.any():
            raise ValueError("no part of the edge faces the other panel")
    _, _, before = errors(C)
    for _ in range(iterations):
        pts, want, _ = errors(C)
        # The part not facing the other edge only loosely keeps its place, so a
        # single-span edge can bend into the gap without rippling.
        w = np.where(facing, 1.0, 0.05)[:, None]
        dP, *_ = np.linalg.lstsq(w * B, w * (want - pts), rcond=None)
        C[:, m - rows:] += dP[:, None, :]
    _, _, after = errors(C)
    out = net.copy()
    idx = _framed_indices(net.shape[:2], edge)
    out[idx[..., 0], idx[..., 1]] = C
    return out, orders, (float(before), float(after))
