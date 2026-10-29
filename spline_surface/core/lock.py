"""Which control points of a patch are pinned by its seams.

A seam with G0 continuity fixes the edge row of control points, G1 also the
second row, G2 also the third (see core.continuity.match_net). Moving any of
those breaks the seam, so interior edits are limited to the points left over.
Open edges (no neighbour) pin nothing.
"""
import numpy as np

from .continuity import _framed_indices

LOCK_ROWS = {'G0': 1, 'G1': 2, 'G2': 3}


def lock_mask(shape, seams):
    """Boolean (nu, nv) mask of pinned control points.

    `shape` is the net's (nu, nv); `seams` is [(edge, grade)] with edge one of
    'U0' 'U1' 'V0' 'V1' and grade 'G0'/'G1'/'G2' ('open' pins nothing).
    """
    mask = np.zeros(tuple(shape), dtype=bool)
    for edge, grade in seams:
        rows = LOCK_ROWS.get(grade, 0)
        if rows:
            idx = _framed_indices(shape, edge)[:, -rows:]
            mask[idx[..., 0], idx[..., 1]] = True
    return mask
