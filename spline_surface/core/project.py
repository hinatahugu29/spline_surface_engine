"""Project points onto a patch: nearest point, or along a direction.

Both start from the best of a coarse grid of surface samples and refine with
Newton steps in the patch's (u, v). Points whose ray misses the patch (or
whose nearest point lies on its boundary far from the target) are flagged.
"""
import numpy as np

from .nurbs import eval_points


def _grid(net, orders, n):
    g = np.linspace(0.0, 1.0, n)
    u, v = np.meshgrid(g, g, indexing="ij")
    u, v = u.ravel(), v.ravel()
    return u, v, eval_points(net, orders, u, v)


def project_points(net, orders, points, direction=None, grid=24, iterations=30, tol=1e-9):
    """Project `points` onto the patch.

    direction: None for the nearest point of the patch, else a vector; every
    point is then moved along +-direction until it lies on the patch.
    Returns (projected (n, 3), uv (n, 2), ok (n,) bool). ok is False where a
    ray found no hit inside the patch (the point is then the nearest end of
    its best try).
    """
    points = np.asarray(points, float)
    gu, gv, gp = _grid(net, orders, grid)
    size = float(np.ptp(gp, axis=0).max()) or 1.0
    out = np.zeros_like(points)
    uv = np.zeros((len(points), 2))
    ok = np.ones(len(points), bool)
    d = None
    if direction is not None:
        d = np.asarray(direction, float)
        d = d / max(np.linalg.norm(d), 1e-12)
    for i, p in enumerate(points):
        if d is None:
            k = int(np.argmin(np.linalg.norm(gp - p, axis=1)))
        else:
            rel = gp - p
            off = rel - (rel @ d)[:, None] * d              # distance from the ray's line
            k = int(np.argmin(np.linalg.norm(off, axis=1)))
        u, v = float(gu[k]), float(gv[k])
        t = float((gp[k] - p) @ d) if d is not None else 0.0
        for _ in range(iterations):
            S = eval_points(net, orders, [u], [v])[0]
            Su = eval_points(net, orders, [u], [v], 1, 0)[0]
            Sv = eval_points(net, orders, [u], [v], 0, 1)[0]
            if d is None:
                r = S - p
                J = np.stack([Su, Sv], 1)
                Suu = eval_points(net, orders, [u], [v], 2, 0)[0]
                Suv = eval_points(net, orders, [u], [v], 1, 1)[0]
                Svv = eval_points(net, orders, [u], [v], 0, 2)[0]
                H = J.T @ J + np.array([[r @ Suu, r @ Suv], [r @ Suv, r @ Svv]])
                g = J.T @ r
                try:
                    if np.linalg.eigvalsh(H).min() <= 1e-12:
                        raise np.linalg.LinAlgError
                    du, dv = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    du, dv = np.linalg.lstsq(J.T @ J, -g, rcond=None)[0]
                # halve the step until the point really gets closer
                f0 = r @ r
                for _ in range(8):
                    un, vn = np.clip(u + du, 0.0, 1.0), np.clip(v + dv, 0.0, 1.0)
                    rn = eval_points(net, orders, [un], [vn])[0] - p
                    if rn @ rn <= f0 + 1e-15:
                        break
                    du, dv = 0.5 * du, 0.5 * dv
            else:
                r = S - p - t * d
                J = np.stack([Su, Sv, -d], 1)
                step = np.linalg.lstsq(J, -r, rcond=None)[0]
                du, dv, dt = step
                t += dt
            u = float(np.clip(u + du, 0.0, 1.0))
            v = float(np.clip(v + dv, 0.0, 1.0))
            if abs(du) + abs(dv) < tol:
                break
        S = eval_points(net, orders, [u], [v])[0]
        out[i], uv[i] = S, (u, v)
        if d is not None:
            off = (S - p) - ((S - p) @ d) * d
            ok[i] = np.linalg.norm(off) < 1e-4 * size
    return out, uv, ok
