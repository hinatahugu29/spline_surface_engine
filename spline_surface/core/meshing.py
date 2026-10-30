"""Mesh clean-up for To Mesh: find T-junctions along patch seams.

A long patch edge can meet two shorter patches (a Fill Hole around a centre,
a split neighbour), so after equal sampling the short side has vertices in
the middle of the long side's edges. Those vertices are found here so the
long side's edges can be split there and the seam welded shut.
"""
import numpy as np


def t_junctions(points, edges, verts, abs_tol, rel_tol=0.05):
    """Vertices lying on the inside of an edge they don't belong to.

    points: (n, 3) vertex positions; edges: (m, 2) vertex indices of the
    (boundary) edges to test; verts: candidate vertex indices. A vertex hits
    an edge when it projects strictly inside it and lies within
    max(abs_tol, rel_tol * edge length) of it - the relative part lets a
    curved seam's vertex sit off the coarse side's straight chord.
    Returns {edge row: [(fraction along the edge, vertex index)]}, sorted.
    """
    P = np.asarray(points, float)
    E = np.asarray(edges, int).reshape(-1, 2)
    A, B = P[E[:, 0]], P[E[:, 1]]
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    tol = np.maximum(abs_tol, rel_tol * np.sqrt(L2))
    hits = {}
    for v in verts:
        p = P[v]
        f = np.einsum("ij,ij->i", p - A, AB) / np.maximum(L2, 1e-30)
        d = np.linalg.norm(A + f[:, None] * AB - p, axis=1)
        clear = (np.linalg.norm(p - A, axis=1) > abs_tol) & (np.linalg.norm(p - B, axis=1) > abs_tol)
        ok = (E != v).all(axis=1) & (f > 1e-3) & (f < 1 - 1e-3) & (d < tol) & clear
        for e in np.nonzero(ok)[0]:
            hits.setdefault(int(e), []).append((float(f[e]), int(v)))
    for e in hits:
        hits[e].sort()
    return hits
