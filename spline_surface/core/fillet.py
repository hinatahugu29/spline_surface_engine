"""Round the crease between two coincident patch edges with a given radius.

Unlike Bridge Edges (`blend.py`), the two edges here already share the same
position - only their surface tangents differ, so there is a sharp G0 crease
(e.g. two patches joined with Match Edge, or any edge picked twice). The
fillet is a rolling-ball blend: at each point along the shared edge, the
circle of `radius` sitting between the two surfaces, tangent to both, is
found, and the patch is fit (least squares) to that circle's arc. The result
is G1 (tangent-plane continuous) with each surface, not G2 - a true radius
fillet generally has different normal curvature than the surfaces it joins.

The fillet's own two edges are the circle's tangent points, set back from the
original crease by the "reach" the radius demands. `fillet_net` leaves the
source patches untouched; `multi_fillet` cuts them back by that reach and
bridges the new edges, so the result is one smooth chain of patches. The
bridge's handles are sized so a G1 bridge between flat surfaces
approximates the arc.
"""
import numpy as np

from .blend import blend_net
from .corner import _refit, _side, common_structure
from .nurbs import basis_matrix, eval_curve, eval_points
from .split import split_net

DEGREE = 5  # across the fillet; enough rows to fit an arc with good accuracy


def _unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def fillet_net(Pa, orders_a, edge_a, Pb, orders_b, edge_b, radius, tol=1e-3, flip=False):
    """Net of the fillet patch rounding edge_a of Pa into edge_b of Pb.

    The edges must be coincident (shared position within `tol`); only their
    outward derivatives may differ. Returns (net, orders, info) where info
    has `reach` (array, how far into each surface the fillet reaches),
    `max_angle` (largest dihedral angle along the edge, degrees - near 180
    means the surfaces are nearly flush and the fillet barely bites) and
    `fit_error` (max deviation of the fitted patch from the true arc).
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    sa, ka, sign_a = _side(np.asarray(Pa, float), orders_a, edge_a)
    sb, kb, _ = _side(np.asarray(Pb, float), orders_b, edge_b)
    Ea, Ta, Aa = sa
    Eb, Tb, Ab = sb

    # Compare the curves as point sets (sampled), not their control points:
    # two differently-built nets can trace the same curve with control points
    # that never coincide index-for-index (different point counts, spans,
    # or fitting history) - this is exactly what Check Continuity already
    # validated, so match its method here instead of a raw control-net diff.
    t = np.linspace(0.0, 1.0, 32)
    ca, cb = eval_curve(Ea, ka, t), eval_curve(Eb, kb, t)
    straight = np.linalg.norm(ca[0] - cb[0]) + np.linalg.norm(ca[-1] - cb[-1])
    crossed = np.linalg.norm(ca[0] - cb[-1]) + np.linalg.norm(ca[-1] - cb[0])
    if (crossed < straight) != bool(flip):
        Eb, Tb, Ab, cb = Eb[::-1], Tb[::-1], Ab[::-1], cb[::-1]

    gap = np.linalg.norm(ca - cb, axis=1).max()
    if gap > tol:
        raise ValueError("the edges are not coincident (max gap %.4g); use Bridge Edges instead" % gap)

    na, nb = len(Ea), len(Eb)
    n, k = common_structure(na, ka, nb, kb)
    (Ea, Ta, Aa) = _refit((Ea, Ta, Aa), na, ka, n, k)
    (Eb, Tb, Ab) = _refit((Eb, Tb, Ab), nb, kb, n, k)
    C = 0.5 * (Ea + Eb)

    # T points past the edge, away from the patch; the ball sits between the
    # two surfaces themselves, so roll along the inward directions -T. Then
    # the tangent points lie on the source surfaces (where trimming cuts).
    ta, tb = -_unit(Ta), -_unit(Tb)
    cos_phi = np.clip(np.einsum("ij,ij->i", ta, tb), -1.0, 1.0)
    phi = np.arccos(cos_phi)
    phi = np.clip(phi, np.radians(1.0), np.radians(179.0))  # guard flush / folded-back edges
    half = phi / 2.0
    reach = (radius / np.tan(half))[:, None]
    bis = _unit(ta + tb)
    centre = C + (radius / np.sin(half))[:, None] * bis

    PA, PB = C + reach * ta, C + reach * tb
    e1 = _unit(PA - centre)
    v2 = (PB - centre) - np.einsum("ij,ij->i", PB - centre, e1)[:, None] * e1
    e2 = _unit(v2)
    alpha = np.pi - phi  # sweep angle of the arc

    m = 4 * DEGREE  # dense across-direction sampling of the true arc
    u = np.linspace(0.0, 1.0, m)
    ang = u[None, :] * alpha[:, None]
    pts = (centre[:, None, :] + radius * np.cos(ang)[:, :, None] * e1[:, None, :]
           + radius * np.sin(ang)[:, :, None] * e2[:, None, :])

    d = DEGREE
    Bv = basis_matrix(d + 1, min(d + 1, 6), u)
    Pv = np.linalg.pinv(Bv)
    net = np.einsum("ia,naD->niD", Pv, pts)
    fit = np.einsum("ai,niD->naD", Bv, net)
    fit_error = float(np.linalg.norm(fit - pts, axis=2).max())
    net[:, 0], net[:, -1] = PA, PB

    if sign_a < 0:
        net = net[:, ::-1]
    info = {
        "reach": reach[:, 0],
        "max_angle": float(np.degrees(phi.max())),
        "mean_angle": float(np.degrees(phi.mean())),
        "fit_error": fit_error,
    }
    return net, (k, d + 1), info


def trim_net(net, orders, edge, distance, samples=200):
    """Cut `distance` off the patch at `edge`, measured along the surface
    (averaged over the edge). The cut boundary keeps the key `edge`.

    Returns (net, orders, t_used, max_fit_error).
    """
    P = np.asarray(net, float)
    direction, at_end = edge[0], edge[1] == '1'
    s, w = np.linspace(0.0, 1.0, samples), np.linspace(0.0, 1.0, 9)
    S, W = (a.ravel() for a in np.meshgrid(s, w, indexing="ij"))
    u, v = (S, W) if direction == 'U' else (W, S)
    pts = eval_points(P, orders, u, v).reshape(samples, len(w), 3)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=2).mean(axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    total = cum[-1]
    if distance >= 0.95 * total:
        raise ValueError("the radius is too large: needs %.3g of a patch only %.3g across"
                         % (distance, total))
    t = float(np.interp(total - distance if at_end else distance, cum, s))
    A, B, oa, ob, t, err = split_net(P, orders, direction, t, snap=False)
    return (A, oa, t, err) if at_end else (B, ob, t, err)


def _miter_ends(bridges, tol=1e-6):
    """Give two bridges that meet at an unrounded edge one shared end curve.

    Rounding two edges of a corner but not the third leaves each bridge ending
    on a skew arc; the two arcs share their end points but not their shape, so
    the corner is cracked. Both ends are replaced by the curve whose control
    points add the two arcs' offsets from the shared end point (the miter of
    two cylinders). The points where the bridges already agree stay put.
    bridges: [(net, orders, tension)]; returns the same list.
    """
    nets = [np.array(n, float) for n, _, _ in bridges]
    for i in range(len(nets)):
        for j in range(i + 1, len(nets)):
            for ei in (0, -1):
                for ej in (0, -1):
                    a, b = nets[i][ei], nets[j][ej]
                    if np.allclose(a[0], b[0], atol=tol) and np.allclose(a[-1], b[-1], atol=tol):
                        flip = False
                    elif np.allclose(a[0], b[-1], atol=tol) and np.allclose(a[-1], b[0], atol=tol):
                        flip = True
                    else:
                        continue
                    b = b[::-1] if flip else b
                    differ = np.where(np.linalg.norm(a - b, axis=1) > tol)[0]
                    if not len(differ):
                        continue
                    anchor = a[0] if differ.mean() < (len(a) - 1) / 2 else a[-1]
                    new = a.copy()
                    new[differ] = anchor + (a[differ] - anchor) + (b[differ] - anchor)
                    nets[i][ei] = new
                    nets[j][ej] = new[::-1] if flip else new
    return [(n, o, t) for n, (_, o, t) in zip(nets, bridges)]


def multi_fillet(patches, seams, radius, continuity='G2', tol=1e-3):
    """Fillet several seams at once.

    patches: {name: (net, orders)}; seams: [(name_a, edge_a, name_b, edge_b)].
    Every reach is measured on the original patches, then each patch is cut
    back on all of its filleted edges, then each seam's two cut edges are
    bridged. (Doing one seam at a time would fail: cutting a patch on one
    edge shortens its neighbouring edges, which then no longer match the
    patch across the next seam.) Where fillets meet at a corner a small
    hole is left for Fill Hole; where only two meet (the third edge stays
    sharp) their ends are mitred into one shared curve.

    Returns (trimmed {name: (net, orders)}, [(net, orders, tension)] per
    seam, [info] per seam).
    """
    infos, cuts = [], {}
    for a, ea, b, eb in seams:
        (Pa, oa), (Pb, ob) = patches[a], patches[b]
        _, _, info = fillet_net(Pa, oa, ea, Pb, ob, eb, radius, tol)
        info["trim"] = reach = float(info["reach"].mean())
        infos.append(info)
        for name, edge in ((a, ea), (b, eb)):
            if any(e == edge for e, _ in cuts.get(name, [])):
                raise ValueError("%s %s is picked twice" % (name, edge))
            cuts.setdefault(name, []).append((edge, reach))
    trimmed = {}
    for name, edges in cuts.items():
        net, orders = patches[name]
        err = 0.0
        for edge, reach in edges:
            net, orders, _, e = trim_net(net, orders, edge, reach)
            err = max(err, e)
        trimmed[name] = (net, orders)
        for info, (a, _, b, _) in zip(infos, seams):
            if name in (a, b):
                info["trim_error"] = max(info.get("trim_error", 0.0), err)
    bridges = []
    for info, (a, ea, b, eb) in zip(infos, seams):
        # Degree-5 handle of a circular arc of sweep alpha is 3/5 of the cubic's
        # (4/3) tan(alpha/4) r; over the chord 2 r sin(alpha/2) that is this tension.
        alpha = np.pi - np.radians(info["mean_angle"])
        info["tension"] = tension = float(2.0 * np.tan(alpha / 4.0) / np.sin(alpha / 2.0))
        net, orders = blend_net(*trimmed[a], ea, *trimmed[b], eb, continuity, tension)
        bridges.append((net, orders, tension))
    return trimmed, _miter_ends(bridges), infos


def corner_loops(bridges, rel_tol=1e-3):
    """Holes left where fillets meet: closed loops of bridge end edges.

    A fillet bridge runs along its seam (edges V0, V1 on the trimmed
    patches); its ends U0 and U1 cross from one patch to the other. Where
    fillets meet at a corner their ends join end to end around a hole (a
    triangle at a box corner where all three edges are filleted). Ends that
    don't close a loop (open boundaries) are skipped.
    bridges: [(net, orders)]. Returns [[(bridge index, 'U0' | 'U1'), ...]].
    """
    pts = np.concatenate([np.asarray(n).reshape(-1, 3) for n, _ in bridges])
    tol = max(rel_tol * float(np.ptp(pts, axis=0).max()), 1e-9)
    nodes = []

    def node(p):
        for i, q in enumerate(nodes):
            if np.linalg.norm(p - q) < tol:
                return i
        nodes.append(p)
        return len(nodes) - 1

    ends = []
    for i, (net, _) in enumerate(bridges):
        net = np.asarray(net, float)
        for edge, row in (('U0', net[0]), ('U1', net[-1])):
            a, b = node(row[0]), node(row[-1])
            if a != b:
                ends.append(((i, edge), a, b))
    links = {}
    for k, (_, a, b) in enumerate(ends):
        links.setdefault(a, []).append(k)
        links.setdefault(b, []).append(k)
    loops, used = [], set()
    for start in range(len(ends)):
        if start in used:
            continue
        # Walk end to end; a loop closes back on the starting node.
        chain, k, (_, first, at) = [start], start, ends[start]
        while at != first:
            nxt = [j for j in links[at] if j != k and j not in chain]
            if len(links[at]) != 2 or not nxt:
                chain = None
                break
            k = nxt[0]
            chain.append(k)
            _, a, b = ends[k]
            at = b if a == at else a
        if chain and len(links[first]) == 2:
            used.update(chain)
            loops.append([ends[j][0] for j in chain])
    return loops
