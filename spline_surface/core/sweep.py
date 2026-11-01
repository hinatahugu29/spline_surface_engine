"""Sweep: carry the profile of one patch edge along the edge of another patch.

The new surface is S(u, v) = profile(u) + rail(v) - rail(start). Both curves are
clamped B-splines whose basis functions sum to one, so the control net is simply
profile point i + rail offset j: the sweep is exact, no fitting involved.
The profile stays where it is at the rail's end nearest to it and follows the
rail from there (the profile is moved, not turned).
"""
import numpy as np

from .linked import edge_order, edge_row


def sweep_net(P, orders_p, edge_p, R, orders_r, edge_r, flip=False):
    """Net and orders of the profile `edge_p` of net P swept along `edge_r` of net R.

    The rail is walked from the end nearest to the profile; flip walks it from the other end.
    Returns (net, orders): u runs along the profile, v along the rail.
    """
    P, R = np.asarray(P, float), np.asarray(R, float)
    prof = P[edge_row(P, edge_p)]
    rail = R[edge_row(R, edge_r)]
    ends = [(np.linalg.norm(prof[i] - rail[j]), j) for i in (0, -1) for j in (0, -1)]
    nearest = min(ends)[1]
    if (nearest == -1) != bool(flip):
        rail = rail[::-1]
    if np.linalg.norm(rail[-1] - rail[0]) < 1e-12:
        raise ValueError("the rail edge has no length")
    offsets = rail - rail[0]
    net = prof[:, None, :] + offsets[None, :, :]
    return net, (edge_order(orders_p, edge_p), edge_order(orders_r, edge_r))
