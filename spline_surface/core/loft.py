"""Loft a single Bezier patch through a sequence of section curves."""
import numpy as np

from .bezier import bernstein, elevate_curve


def order_sections(polys, keep_order=False):
    """Order sections into a chain and orient them consistently.

    Sections are chained by nearest centroid, starting from one end of the
    chain (the farthest-apart pair), unless `keep_order` is set. Each section
    is flipped if needed so its start lies on the same side as the previous
    section's start. Returns the list of (index, flipped).
    """
    cents = np.array([p.mean(axis=0) for p in polys])
    if len(polys) > 2 and not keep_order:
        d = np.linalg.norm(cents[:, None] - cents[None], axis=2)
        start = int(np.unravel_index(np.argmax(d), d.shape)[0])
        order, left = [start], set(range(len(polys))) - {start}
        while left:
            nxt = min(left, key=lambda j: d[order[-1], j])
            order.append(nxt)
            left.remove(nxt)
    else:
        order = list(range(len(polys)))
    result, prev = [], None
    for i in order:
        p = polys[i]
        flip = False
        if prev is not None:
            same = np.linalg.norm(p[0] - prev[0]) + np.linalg.norm(p[-1] - prev[-1])
            cross = np.linalg.norm(p[-1] - prev[0]) + np.linalg.norm(p[0] - prev[-1])
            flip = cross < same
        result.append((i, flip))
        prev = p[::-1] if flip else p
    return result


def loft_net(rows, max_degree=5):
    """Control net through section control polygons.

    `rows` has shape (N, n+1, 3): N fitted sections of equal degree n. Across
    the sections a Bezier of degree min(N-1, max_degree) is used; with
    N <= max_degree+1 it interpolates every section exactly, otherwise it is a
    least-squares fit that still passes through the first and last section.
    Section parameters come from the chord length between section control
    polygons. Returns (n+1, dv+1, 3).
    """
    rows = np.asarray(rows, dtype=float)
    N = len(rows)
    if N < 2:
        raise ValueError("need at least 2 sections")
    dv = min(N - 1, max_degree)
    step = np.linalg.norm(np.diff(rows, axis=0), axis=2).mean(axis=1)
    v = np.concatenate([[0.0], np.cumsum(step)])
    if v[-1] <= 0.0:
        raise ValueError("sections coincide")
    v /= v[-1]
    flat = rows.reshape(N, -1)
    if dv == 1:
        ctrl = np.stack([flat[0], flat[-1]])
    else:
        B = bernstein(dv, v)
        rhs = flat - np.outer(B[:, 0], flat[0]) - np.outer(B[:, -1], flat[-1])
        inner, *_ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
        ctrl = np.vstack([flat[0], inner, flat[-1]])
    return ctrl.reshape(dv + 1, -1, 3).transpose(1, 0, 2), v


def add_rows(net, rows, max_degree=5):
    """Extra control rows between the sections, without changing the shape.

    The across-section Bezier is raised in degree by `rows` (at most to
    `max_degree`), so a two-section loft gets evenly spaced points on its
    straight lines. Returns (n+1, dv'+1, 3).
    """
    net = np.asarray(net, dtype=float)
    target = min(net.shape[1] - 1 + max(int(rows), 0), max_degree)
    if target <= net.shape[1] - 1:
        return net
    return np.stack([elevate_curve(row, target) for row in net])
