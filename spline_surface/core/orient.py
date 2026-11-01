"""Facing of patches: which way a patch's normal (du x dv) points, relative to its neighbours.

Walking round a patch counter-clockwise (seen from the front, along its normal) runs
along the edge V0 in +u, U1 in +v, V1 in -u and U0 in -v. Two patches that share an
edge face the same way exactly when they walk it in opposite directions. Reversing a
patch's u direction turns it over without changing its shape.
"""
import numpy as np

from .continuity import find_shared_edge

_WALK = {'V0': 1, 'U1': 1, 'V1': -1, 'U0': -1}      # counter-clockwise walk along the canonical edge direction


def same_facing(edge_a, edge_b, reversed_):
    """Whether two patches sharing an edge (as find_shared_edge reports it) face the same way."""
    return _WALK[edge_a] * _WALK[edge_b] * (-1 if reversed_ else 1) == -1


def flip_net(net):
    """The same surface turned over (u reversed)."""
    return np.asarray(net)[::-1]


def facing_flips(patches, tol_rel=0.01, root=0, pinned=()):
    """Which patches must be turned over so that all that touch face the same way.

    patches: [(net, orders)]. Returns a list of bools, one per patch. Each group of touching
    patches is made to face like its first member (`root` for the group it is in). Patches
    in `pinned` cannot be turned over: the group takes whichever facing turns the fewest of
    them (ties: the fewest patches overall, then the root's facing).
    """
    n = len(patches)
    size = [float(np.ptp(np.asarray(P).reshape(-1, 3), axis=0).max()) for P, _ in patches]
    links = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            tol = max(max(size[i], size[j]) * tol_rel, 1e-6)
            shared = find_shared_edge(patches[i][0], patches[j][0], tol, patches[i][1], patches[j][1])
            if shared is not None:
                flipped_pair = not same_facing(*shared[:3])
                links[i].append((j, flipped_pair))
                links[j].append((i, flipped_pair))
    flips = [None] * n
    pinned = set(pinned)
    for start in [root] + list(range(n)):
        if flips[start] is not None:
            continue
        flips[start] = False
        group, stack = [start], [start]
        while stack:
            i = stack.pop()
            for j, differ in links[i]:
                if flips[j] is None:
                    flips[j] = flips[i] != differ
                    group.append(j)
                    stack.append(j)
        stuck = sum(1 for k in group if flips[k] and k in pinned)
        stuck_other = sum(1 for k in group if not flips[k] and k in pinned)
        count = sum(1 for k in group if flips[k])
        if (stuck_other, len(group) - count) < (stuck, count):
            for k in group:
                flips[k] = not flips[k]
    return flips
