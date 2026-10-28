"""Edits that spread: a smooth displacement field, edge chains and patch merging.

All functions work on world-space control nets (numpy arrays) and know
nothing about Blender.
"""
import numpy as np

from .nurbs import basis_matrix, eval_points

MAX_ORDER = 6
MAX_POINTS = 64


# ---------------------------------------------------------------- falloff field

def falloff_field(src, disp, radius):
    """Smooth displacement field through the samples (src[i] moves by disp[i]).

    Inside the samples it is a Shepard average (exact at the samples, so two
    patches touching at a point both get the same move: shared edges stay
    closed). It fades to zero `radius` away from the nearest sample with a C2
    falloff, so far patches are untouched.
    """
    src, disp = np.asarray(src, float), np.asarray(disp, float)
    radius = max(float(radius), 1e-9)
    eps = (1e-4 * radius) ** 2

    def field(x):
        x = np.asarray(x, float)
        shape = x.shape
        x = x.reshape(-1, 3)
        d2 = ((x[:, None] - src[None]) ** 2).sum(-1)
        w = 1.0 / (d2 + eps) ** 2
        avg = (w @ disp) / w.sum(1, keepdims=True)
        s = np.clip(np.sqrt(d2.min(1)) / radius, 0.0, 1.0)
        return (avg * ((1.0 - s * s) ** 3)[:, None]).reshape(shape)

    return field


def deform(net, field):
    net = np.asarray(net, float)
    return net + field(net)


# ---------------------------------------------------------------- edge rows

def edge_row(net, edge):
    """Index of the control row lying on `edge` (U0/U1: a u row, V0/V1: a v column)."""
    if edge == 'U0':
        return np.s_[0, :]
    if edge == 'U1':
        return np.s_[-1, :]
    if edge == 'V0':
        return np.s_[:, 0]
    return np.s_[:, -1]


def edge_order(orders, edge):
    """Order along `edge`."""
    return orders[1] if edge[0] == 'U' else orders[0]


def order_chain(lines, tol, max_turn=45.0):
    """Chain of edges joined end to end, starting from the first key.

    `lines` maps key -> polyline. Returns [(key, reversed)], walking both ways
    from the first edge. At each end it continues with the edge that goes on
    straightest (turning less than `max_turn` degrees), so a chain runs along
    a border and does not turn round a patch corner.
    """
    keys = list(lines)
    chain = [(keys[0], False)]
    used = {keys[0]}

    def oriented(key, rev):
        return lines[key][::-1] if rev else lines[key]

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-12)

    def walk(forward):
        while True:
            pl = oriented(*(chain[-1] if forward else chain[0]))
            if not forward:
                pl = pl[::-1]
            tip, heading = pl[-1], unit(pl[-1] - pl[-2])
            best = None
            for k in keys:
                if k in used:
                    continue
                for rev in (False, True):
                    q = oriented(k, rev)
                    if np.linalg.norm(q[0] - tip) >= tol:
                        continue
                    turn = np.degrees(np.arccos(np.clip(heading @ unit(q[1] - q[0]), -1, 1)))
                    if turn < max_turn and (best is None or turn < best[0]):
                        best = (turn, k, rev)
            if best is None:
                return
            _, k, rev = best
            used.add(k)
            if forward:
                chain.append((k, rev))
            else:
                chain.insert(0, (k, not rev))

    walk(True)
    walk(False)
    return chain


def chain_polyline(lines, chain):
    """Joined polyline and the parameter range (chord length) of each piece."""
    pts, spans = [], []
    for key, rev in chain:
        pl = lines[key][::-1] if rev else lines[key]
        start = sum(len(p) for p in pts) - (1 if pts else 0)
        pts.append(pl if not pts else pl[1:])
        spans.append((start, start + len(pl) - 1))
    pts = np.concatenate(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-12)
    return pts, [(float(t[a]), float(t[b])) for a, b in spans]


# ---------------------------------------------------------------- merge

def _orientations(net, orders):
    """All 8 ways to lay a patch on a grid: (net, orders, tag)."""
    out = []
    for tr in (False, True):
        P = net.transpose(1, 0, 2) if tr else net
        o = (orders[1], orders[0]) if tr else tuple(orders)
        for fu in (False, True):
            for fv in (False, True):
                Q = P[::-1] if fu else P
                Q = Q[:, ::-1] if fv else Q
                out.append((np.ascontiguousarray(Q), o, (tr, fu, fv)))
    return out


def _edge_curve(net, orders, edge, samples=12):
    t = np.linspace(0.0, 1.0, samples)
    c = np.full(samples, 1.0 if edge[1] == '1' else 0.0)
    return eval_points(net, orders, c, t) if edge[0] == 'U' else eval_points(net, orders, t, c)


_STEP = {'U1': (1, 0, 'U0'), 'U0': (-1, 0, 'U1'), 'V1': (0, 1, 'V0'), 'V0': (0, -1, 'V1')}


def grid_layout(patches, tol):
    """Arrange patches [(net, orders)] edge to edge on a rectangular grid.

    Returns {(i, j): (oriented net, orders, index)} with i along u, j along v.
    Raises ValueError if they do not tile a full rectangle.
    """
    placed = {(0, 0): (np.asarray(patches[0][0], float), tuple(patches[0][1]), 0)}
    where = {0: (0, 0)}
    stack = [0]
    while stack:
        idx = stack.pop()
        cell = where[idx]
        Q, oq, _ = placed[cell]
        for side, (di, dj, opp) in _STEP.items():
            target = (cell[0] + di, cell[1] + dj)
            if target in placed:
                continue
            cq = _edge_curve(Q, oq, side)
            best = None
            for n, (net, orders) in enumerate(patches):
                if n in where:
                    continue
                for P, o, _ in _orientations(np.asarray(net, float), orders):
                    gap = np.linalg.norm(_edge_curve(P, o, opp) - cq, axis=1).max()
                    if best is None or gap < best[0]:
                        best = (gap, n, P, o)
            if best is not None and best[0] < tol:
                _, n, P, o = best
                placed[target] = (P, o, n)
                where[n] = target
                stack.append(n)
    if len(where) != len(patches):
        raise ValueError("%d of %d patches are not joined edge to edge with the rest"
                         % (len(patches) - len(where), len(patches)))
    i0 = min(c[0] for c in placed)
    j0 = min(c[1] for c in placed)
    grid = {(i - i0, j - j0): v for (i, j), v in placed.items()}
    ni = max(c[0] for c in grid) + 1
    nj = max(c[1] for c in grid) + 1
    if len(grid) != ni * nj:
        raise ValueError("the patches do not form a full rectangular grid (%d of %dx%d cells)"
                         % (len(grid), ni, nj))
    return grid


def _chord_params(G, axis):
    d = np.linalg.norm(np.diff(G, axis=axis), axis=-1)
    d = d.mean(axis=1 - axis)
    t = np.concatenate([[0.0], np.cumsum(d)])
    return t / max(t[-1], 1e-12)


def merge_grid(grid, samples=24, detail=1.0):
    """One NURBS net fitted through a grid of patches. Returns (net, orders, max error)."""
    ni = max(c[0] for c in grid) + 1
    nj = max(c[1] for c in grid) + 1
    s = np.linspace(0.0, 1.0, samples + 1)
    U, V = np.meshgrid(s, s, indexing="ij")
    G = np.zeros((ni * samples + 1, nj * samples + 1, 3))
    for (i, j), (P, o, _) in grid.items():
        pts = eval_points(P, o, U.ravel(), V.ravel()).reshape(samples + 1, samples + 1, 3)
        G[i * samples:(i + 1) * samples + 1, j * samples:(j + 1) * samples + 1] = pts
    k = min(MAX_ORDER, max(max(o) for _, o, _ in grid.values()))
    ku = kv = max(k, 4)

    def count(axis):
        # As many points as the originals have along that way (seams counted once).
        total = 1
        for a in range(ni if axis == 0 else nj):
            cells = [v for c, v in grid.items() if c[axis] == a]
            total += max(v[0].shape[axis] for v in cells) - 1
        n = int(round(total * detail))
        # Stay well below the sample count so the least-squares fit is determined.
        return int(np.clip(n, ku, min(MAX_POINTS, G.shape[axis] // 2)))

    nu, nv = count(0), count(1)
    tu, tv = _chord_params(G, 0), _chord_params(G, 1)
    Bu, Bv = basis_matrix(nu, ku, tu), basis_matrix(nv, kv, tv)
    Pu, Pv = np.linalg.pinv(Bu), np.linalg.pinv(Bv)
    net = np.einsum("ia,abd,jb->ijd", Pu, G, Pv)
    fit = np.einsum("ai,ijd,bj->abd", Bu, net, Bv)
    return net, (ku, kv), float(np.linalg.norm(fit - G, axis=-1).max())
