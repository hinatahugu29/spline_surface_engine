"""Mesh clean-up for To Mesh: find T-junctions along patch seams.

A long patch edge can meet two shorter patches (a Fill Hole around a centre,
a split neighbour), so after equal sampling the short side has vertices in
the middle of the long side's edges. Those vertices are found here so the
long side's edges can be split there and the seam welded shut.
"""
import numpy as np

from .continuity import find_shared_edge
from .nurbs import eval_points

# Which direction's segment count an edge's sampling follows: V-edges run
# along u, U-edges along v.
_EDGE_DIR = {'V0': 0, 'V1': 0, 'U0': 1, 'U1': 1}


def _side_lengths(net, orders, samples=24):
    """Mean length of the patch's iso-curves along u and along v."""
    t = np.linspace(0.0, 1.0, samples)
    out = []
    for d in (0, 1):
        total = 0.0
        for c in (0.0, 0.5, 1.0):
            cc = np.full(samples, c)
            pts = eval_points(net, orders, t, cc) if d == 0 else eval_points(net, orders, cc, t)
            total += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        out.append(total / 3.0)
    return out


def segment_counts(patches, resolution, rel_tol=1e-3):
    """(segments along u, along v) per patch for an even mesh density.

    The longest side among all patches gets `resolution` segments and every
    other side about the same segment length. Patches sharing a whole edge
    get equal counts along it (largest wins), so their seam vertices meet.
    patches: [(net, orders)].
    """
    lengths = [_side_lengths(n, o) for n, o in patches]
    longest = max((max(l) for l in lengths), default=1.0)
    step = max(longest, 1e-12) / max(int(resolution), 1)
    counts = [[max(2, int(round(l / step))) for l in ls] for ls in lengths]

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(patches)):
        Pa, oa = patches[a]
        for b in range(a + 1, len(patches)):
            Pb, ob = patches[b]
            pts = np.concatenate([Pa.reshape(-1, 3), Pb.reshape(-1, 3)])
            tol = max(rel_tol * np.ptp(pts, axis=0).max(), 1e-9)
            shared = find_shared_edge(Pa, Pb, tol, oa, ob)
            if shared is not None:
                parent[find((a, _EDGE_DIR[shared[0]]))] = find((b, _EDGE_DIR[shared[1]]))
    best = {}
    for i, c in enumerate(counts):
        for d in (0, 1):
            r = find((i, d))
            best[r] = max(best.get(r, 0), c[d])
    return [(best[find((i, 0))], best[find((i, 1))]) for i in range(len(patches))]


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
