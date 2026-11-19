"""Attaching the end of a Bezier chain to something: a point, a direction, a curvature.

Chains as in core.curvepoint (anchors A, handles L / R). The start of an open
chain leaves A[0] along R[0]; its end arrives at A[-1] from L[-1]. Here every
direction is the one *into* the curve from its end: R[0] - A[0] at the start,
L[-1] - A[-1] at the end.
"""
import numpy as np

from .curvepoint import EPS, _unit


def _idx(n, end):
    """(end anchor, its inner handle array name, next anchor, next anchor's handle
    facing the end, next anchor's other handle)."""
    if end == 'START':
        return 0, 'R', 1, 'L', 'R'
    return n - 1, 'L', n - 2, 'R', 'L'


def end_frame(A, L, R, end):
    """(end point, unit direction into the curve, curvature vector there)."""
    A, L, R = (np.asarray(x, float) for x in (A, L, R))
    H = {'L': L, 'R': R}
    i, h, j, hj, _ = _idx(len(A), end)
    p = A[i]
    hv = H[h][i] - p
    a = np.linalg.norm(hv)
    if a < EPS:
        return p, _unit(A[j] - p), np.zeros(3)
    t = hv / a
    off = H[hj][j] - H[h][i]
    perp = off - (off @ t) * t
    return p, t, 2.0 / 3.0 * perp / (a * a)


def attach_end(A, L, R, end, point, direction=None, curvature=None):
    """The chain with its `end` ('START' or 'END') moved to `point`.

    direction: the unit direction the curve should leave the end in (into the
    curve); the end handle keeps its length. curvature: the curvature vector the
    curve should have there (G2): the next handle is moved off the tangent line
    to match, and the next anchor's other handle is turned with it so the curve
    stays smooth there. Returns (A, L, R).
    """
    A, L, R = (np.array(x, float) for x in (A, L, R))
    H = {'L': L, 'R': R}
    n = len(A)
    i, h, j, hj, hk = _idx(n, end)
    shift = np.asarray(point, float) - A[i]
    A[i] += shift
    L[i] += shift
    R[i] += shift
    if direction is not None:
        d = _unit(np.asarray(direction, float))
        if np.linalg.norm(d) > 0:
            a = np.linalg.norm(H[h][i] - A[i])
            if a < EPS:
                a = np.linalg.norm(A[j] - A[i]) / 3.0
            other = 'L' if h == 'R' else 'R'
            H[h][i] = A[i] + a * d
            H[other][i] = A[i] - a * d            # the unused outer handle, lined up
    if curvature is not None and n >= 2:
        t = _unit(H[h][i] - A[i])
        a = np.linalg.norm(H[h][i] - A[i])
        k = np.asarray(curvature, float)
        k = k - (k @ t) * t
        far = H[hj][j]
        along = (far - H[h][i]) @ t
        new_far = H[h][i] + along * t + 1.5 * a * a * k
        if n > 2:            # keep the next anchor smooth: its other handle turns with this one
            old_dir, new_dir = _unit(A[j] - far), _unit(A[j] - new_far)
            b = np.linalg.norm(H[hk][j] - A[j])
            if (np.linalg.norm(old_dir) > 0 and np.linalg.norm(new_dir) > 0
                    and (H[hk][j] - A[j]) @ old_dir > (1 - 1e-6) * b):
                H[hk][j] = A[j] + b * new_dir
        H[hj][j] = new_far
    return A, L, R
