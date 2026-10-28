"""Mirror symmetry: reflected copies and smooth joins on the symmetry plane.

A patch edge lying on the plane joins its own mirror image smoothly when the
surface crosses the plane at a right angle: the second control row must sit
straight out from the edge row along the plane normal, so the
across-derivative is parallel to the normal. That gives G1, and for a mirror
image G1 already implies G2 (reflection and the reversed direction flip the
curvature sign twice), so nothing more is needed.
"""
import numpy as np

AXES = {'X': 0, 'Y': 1, 'Z': 2}

# edge -> (index of the edge row, index of the next row inside) on that axis of the net
_ROWS = {'U0': (0, 1, 0), 'U1': (-1, -2, 0), 'V0': (0, 1, 1), 'V1': (-1, -2, 1)}


def mirror_net(net, axis='X', offset=0.0):
    """Reflected net. The u direction is reversed so normals stay outward."""
    M = np.array(net, dtype=float, copy=True)
    a = AXES[axis]
    M[..., a] = 2.0 * offset - M[..., a]
    return M[::-1]


def _rows(net, edge):
    i0, i1, ax = _ROWS[edge]
    return (net[i0], net[i1]) if ax == 0 else (net[:, i0], net[:, i1])


def edges_on_plane(net, axis='X', offset=0.0, tol=1e-3):
    """Edges of the net whose control points all lie within tol of the plane."""
    a = AXES[axis]
    return [e for e in _ROWS if np.abs(_rows(net, e)[0][:, a] - offset).max() <= tol]


def symmetrize(net, axis='X', offset=0.0, tol=1e-3):
    """Make every on-plane edge cross the plane at a right angle.

    The edge row is snapped onto the plane and the next row keeps only its
    distance from the plane (its in-plane coordinates copy the edge row's).

    Returns (net, list of edges fixed).
    """
    P = np.array(net, dtype=float, copy=True)
    a = AXES[axis]
    fixed = edges_on_plane(P, axis, offset, tol)
    for e in fixed:
        i0, i1, ax = _ROWS[e]
        other = [k for k in range(3) if k != a]
        if ax == 0:
            P[i0, :, a] = offset
            P[i1][:, other] = P[i0][:, other]
        else:
            P[:, i0, a] = offset
            P[:, i1][:, other] = P[:, i0][:, other]
    return P, fixed
