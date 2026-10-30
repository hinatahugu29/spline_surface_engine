"""Trim (prototype): a patch's mesh with a hole cut along a closed loop in (u, v).

The surface is meshed on a regular (u, v) grid. Cells touching the inside of
the loop are removed and the ring of vertices around the gap is snapped onto
the loop, so the hole's outline follows the loop exactly and the rest of the
mesh stays a plain quad grid. Fine for loops much larger than a cell; it does
not clip cells in the middle (that needs a proper constrained triangulation).
"""
import numpy as np

from .nurbs import eval_points


def inside_polygon(points, loop):
    """Even-odd test of `points` (n, 2) against the closed polygon `loop` (m, 2)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    a, b = loop, np.roll(loop, -1, axis=0)
    ax, ay, bx, by = a[:, 0][None], a[:, 1][None], b[:, 0][None], b[:, 1][None]
    crosses = (ay > y) != (by > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = ax + (y - ay) * (bx - ax) / (by - ay)
    return ((crosses & (x < xi)).sum(axis=1) % 2) == 1


def nearest_on_polygon(points, loop):
    """Closest point of the closed polygon `loop` to every row of `points`."""
    a, b = loop, np.roll(loop, -1, axis=0)
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-18)
    t = np.clip(((points[:, None] - a[None]) * ab[None]).sum(2) / denom[None], 0.0, 1.0)
    cand = a[None] + t[..., None] * ab[None]
    k = np.argmin(np.linalg.norm(cand - points[:, None], axis=2), axis=1)
    return cand[np.arange(len(points)), k]


def trim_grid(net, orders, loops, nu=48, nv=48):
    """Mesh of the patch with holes. Returns (vertices (n, 3), quads (m, 4), uv (n, 2)).

    loops: list of closed polygons in (u, v), each (k, 2) without repeating the first point.
    """
    gu = np.linspace(0.0, 1.0, nu + 1)
    gv = np.linspace(0.0, 1.0, nv + 1)
    uv = np.stack(np.meshgrid(gu, gv, indexing="ij"), -1).reshape(-1, 2)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    inside = np.zeros(len(uv), bool)
    for loop in loops:
        inside |= inside_polygon(uv, np.asarray(loop, float))
    ins = inside.reshape(nu + 1, nv + 1)
    # a cell goes if any of its corners is inside a loop
    gone = ins[:-1, :-1] | ins[1:, :-1] | ins[1:, 1:] | ins[:-1, 1:]
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1)
    kept = quads[~gone]
    ring = np.zeros(len(uv), bool)
    ring[quads[gone].ravel()] = True
    ring &= ~inside
    used = np.zeros(len(uv), bool)
    used[kept.ravel()] = True
    ring &= used
    uv = uv.copy()
    for loop in loops:
        loop = np.asarray(loop, float)
        sel = np.where(ring)[0]
        if len(sel):
            snapped = nearest_on_polygon(uv[sel], loop)
            # only the ring vertices nearest to this loop (not to another one) move
            near = np.linalg.norm(snapped - uv[sel], axis=1)
            for other in loops:
                if other is loop:
                    continue
                other_d = np.linalg.norm(nearest_on_polygon(uv[sel], np.asarray(other, float)) - uv[sel], axis=1)
                near = np.where(other_d < near, np.inf, near)
            move = np.isfinite(near)
            uv[sel[move]] = snapped[move]
    keep_ids = np.unique(kept.ravel())
    remap = -np.ones(len(uv), int)
    remap[keep_ids] = np.arange(len(keep_ids))
    verts = eval_points(net, orders, uv[keep_ids, 0], uv[keep_ids, 1])
    return verts, remap[kept], uv[keep_ids]
