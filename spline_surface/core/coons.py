"""Boundary-curve loop ordering and Coons patch construction."""
import numpy as np

from .bezier import elevate_curve


def orient_loop(polylines, tol):
    """Chain 3 or 4 polylines head-to-tail into a closed loop.

    Returns the polylines reordered and flipped so that each one ends where the
    next one starts. Raises ValueError if the ends do not meet within `tol`.
    """
    if len(polylines) not in (3, 4):
        raise ValueError("need 3 or 4 boundary curves, got %d" % len(polylines))
    remaining = [np.asarray(p, dtype=float) for p in polylines]
    loop = [remaining.pop(0)]
    while remaining:
        end = loop[-1][-1]
        best = None
        for idx, p in enumerate(remaining):
            for flip in (False, True):
                head = p[-1] if flip else p[0]
                d = np.linalg.norm(head - end)
                if best is None or d < best[0]:
                    best = (d, idx, flip)
        d, idx, flip = best
        if d > tol:
            raise ValueError("curve ends do not meet (gap %.4g > tolerance %.4g)" % (d, tol))
        p = remaining.pop(idx)
        loop.append(p[::-1] if flip else p)
    gap = np.linalg.norm(loop[-1][-1] - loop[0][0])
    if gap > tol:
        raise ValueError("boundary loop is not closed (gap %.4g > tolerance %.4g)" % (gap, tol))
    return loop


def close_loop(loop):
    """Pull the polylines of an oriented loop so neighbouring ends meet.

    Each corner moves to the midpoint of the two ends; the offset is blended
    linearly along each curve (by chord length) so a large gap bends the
    whole curve gently instead of kinking its last segment.
    """
    k = len(loop)
    corners = [(loop[i - 1][-1] + loop[i][0]) / 2.0 for i in range(k)]
    out = []
    for i, p in enumerate(loop):
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        s = (s / s[-1])[:, None]
        start_off = corners[i] - p[0]
        end_off = corners[(i + 1) % k] - p[-1]
        out.append(p + (1.0 - s) * start_off + s * end_off)
    return out


def loop_to_sides(loop):
    """Map a head-to-tail loop to (bottom, right, top, left) Coons sides.

    bottom/top run along U (left -> right), left/right run along V
    (bottom -> top). With 3 curves the left side collapses to a point.
    """
    if len(loop) == 4:
        c0, c1, c2, c3 = loop
        return c0, c1, c2[::-1], c3[::-1]
    c0, c1, c2 = loop
    return c0, c1, c2[::-1], None


def snap_corners(bottom, right, top, left):
    """Make the shared end points of fitted control polygons coincide exactly."""
    bottom, right, top = bottom.copy(), right.copy(), top.copy()
    if left is None:
        c00 = (bottom[0] + top[0]) / 2.0
        left = np.repeat(c00[None, :], len(right), axis=0)
    else:
        left = left.copy()
        c00 = (bottom[0] + left[0]) / 2.0
        c01 = (top[0] + left[-1]) / 2.0
        top[0], left[-1] = c01, c01
    c10 = (bottom[-1] + right[0]) / 2.0
    c11 = (top[-1] + right[-1]) / 2.0
    bottom[0], left[0] = c00, c00
    bottom[-1], right[0] = c10, c10
    top[-1], right[-1] = c11, c11
    if left is not None and np.allclose(left, left[0]):
        top[0] = c00
    return bottom, right, top, left


def coons_net(bottom, right, top, left):
    """Bilinearly blended Coons patch in Bezier control-net form.

    Linear blending has linear precision, so applying it to control points
    gives exactly the Bezier form of the Coons surface. Returns (n+1, m+1, 3).
    """
    n = max(len(bottom), len(top)) - 1
    m = max(len(left), len(right)) - 1
    B, T = elevate_curve(bottom, n), elevate_curve(top, n)
    L, R = elevate_curve(left, m), elevate_curve(right, m)
    u = (np.arange(n + 1) / n)[:, None, None]
    v = (np.arange(m + 1) / m)[None, :, None]
    ruled_v = (1 - v) * B[:, None, :] + v * T[:, None, :]
    ruled_u = (1 - u) * L[None, :, :] + u * R[None, :, :]
    bilinear = ((1 - u) * (1 - v) * B[0] + u * (1 - v) * B[-1]
                + (1 - u) * v * T[0] + u * v * T[-1])
    return ruled_v + ruled_u - bilinear
