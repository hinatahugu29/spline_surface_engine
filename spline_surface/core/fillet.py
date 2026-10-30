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
source patches untouched; `trimmed_fillet` cuts them back by that reach and
bridges the new edges, so the result is one smooth chain of patches.
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


def trimmed_fillet(Pa, orders_a, edge_a, Pb, orders_b, edge_b, radius,
                   continuity='G2', tol=1e-3):
    """Trim both patches back from their shared edge and bridge the gap.

    The trim distance is the rolling ball's reach, so the new edges sit
    where a `radius` fillet would touch the surfaces; the bridge's handles
    are sized so a G1 bridge between flat surfaces approximates that arc.
    Returns ((net_a, orders_a), (net_b, orders_b), (net, orders), info).
    """
    _, _, info = fillet_net(Pa, orders_a, edge_a, Pb, orders_b, edge_b, radius, tol)
    reach = float(info["reach"].mean())
    A, oa, _, err_a = trim_net(Pa, orders_a, edge_a, reach)
    B, ob, _, err_b = trim_net(Pb, orders_b, edge_b, reach)
    # Degree-5 handle of a circular arc of sweep alpha is 3/5 of the cubic's
    # (4/3) tan(alpha/4) r; over the chord 2 r sin(alpha/2) that is this tension.
    alpha = np.pi - np.radians(info["mean_angle"])
    tension = float(2.0 * np.tan(alpha / 4.0) / np.sin(alpha / 2.0))
    net, orders = blend_net(A, oa, edge_a, B, ob, edge_b, continuity, tension)
    info.update(trim=reach, tension=tension, trim_error=max(err_a, err_b))
    return (A, oa), (B, ob), (net, orders), info
