"""Fill a hole bounded by patch edges and/or curves (any number of sides).

The boundary is taken exactly from the picked edges (G0). Across every patch
edge the first one or two control rows follow that neighbour's Taylor
expansion (G1/G2), as in Extend; curve sides only fix the position. Near the hole's corners two sides want the same
control points; there they are averaged, so continuity is exact along each
edge and approximate only in the corner zones. The rest is a Coons blend.

Four sides give one patch; other counts become one quad per corner around a
centre point, sharing "spokes" (and their cross-tangent field) for G1.
"""
import numpy as np

from .corner import _refit, _side
from .bezier import elevate_curve
from .nurbs import basis_matrix, clamped_knots, fit_polyline

MIN_POINTS = 6  # three rows from each side for G2
STAR_POINTS = 13  # corner quads of N-sided fills: finer, so corner averaging stays local


def _structure(na, ka, nb, kb, min_points=MIN_POINTS, max_spans=64):
    """(points, order) both opposite edges nest into, with room for G2 rows."""
    if ka == kb:
        spans = int(np.lcm(na - ka + 1, nb - kb + 1))
        if spans <= max_spans:
            while spans + ka - 1 < min_points:
                spans *= 2
            return spans + ka - 1, ka
    n, k = max((na, ka), (nb, kb))
    return max(n, min_points), k


def greville(n, k):
    U = clamped_knots(n, k)
    return np.array([U[i + 1:i + k].mean() for i in range(n)])


def side_from_patch(P, orders, edge):
    """A hole side on a patch edge: ([edge row, T, A], order, normal sign)."""
    return _side(np.asarray(P, float), orders, edge)


def side_from_curve(pts, spans=4, k=6):
    """A hole side on a free curve (position only). Returns (side, fit error)."""
    E, err = fit_polyline(pts, spans + k - 1, k)
    return ([E, None, None], k, 0.0), err


def _is_rows(R):
    return isinstance(R, np.ndarray)


def _rev(side):
    """Same side run backwards. rows may carry a 4th entry: a fixed reach (float)."""
    rows, k, sign = side
    return [R[::-1] if _is_rows(R) else R for R in rows], k, -sign


def _refit_rows(rows, n, k, n_new, k_new):
    have = [R for R in rows if _is_rows(R)]
    out = iter(_refit(have, n, k, n_new, k_new))
    return [next(out) if _is_rows(R) else R for R in rows]


def order_loop(sides, tol):
    """Chain sides head to tail. sides: [(rows, k, sign)], rows[0] = edge points.

    Returns the sides reordered and reversed as needed, each running from its
    start corner to the next side's start corner.
    """
    rest = list(sides)
    loop = [rest.pop(0)]
    while rest:
        end = loop[-1][0][0][-1]
        best = None
        for i, (rows, k, sign) in enumerate(rest):
            for rev in (False, True):
                start = rows[0][-1] if rev else rows[0][0]
                d = np.linalg.norm(start - end)
                if best is None or d < best[0]:
                    best = (d, i, rev)
        d, i, rev = best
        if d > tol:
            raise ValueError("the edges do not form a closed loop (gap %.4g)" % d)
        side = rest.pop(i)
        loop.append(_rev(side) if rev else side)
    d = np.linalg.norm(loop[-1][0][0][-1] - loop[0][0][0][0])
    if d > tol:
        raise ValueError("the edges do not close the loop (gap %.4g)" % d)
    return loop


def _taylor_rows(P0, T, A, n, k, alpha, continuity):
    """Second and third control rows from a clamped start: S' = alpha T, S'' = alpha^2 A."""
    d1 = basis_matrix(n, k, 0.0, 1)[0]
    d2 = basis_matrix(n, k, 0.0, 2)[0]
    P1 = P0 + alpha * T / d1[1]
    if continuity != 'G2':
        return [P1]
    P2 = (alpha * alpha * A - d2[0] * P0 - d2[1] * P1) / d2[2]
    return [P1, P2]


def _side_normal(side, t=0.5):
    """Source surface normal at parameter t of a loop-oriented patch side."""
    (E, T, _), k, sign = side
    n = len(E)
    dE = basis_matrix(n, k, t, 1)[0] @ E
    return sign * np.cross(dE, basis_matrix(n, k, t)[0] @ T)


def quad_net(loop, continuity='G2', tension=1.0, min_points=MIN_POINTS, g1_first=False):
    """One patch from four loop-oriented sides (c0->c1, c1->c2, c2->c3, c3->c0).

    More points (min_points) keep the averaged corner zones small. With
    g1_first, tangent rows win over curvature rows where sides overlap (for
    sides whose data disagree at the corners, e.g. radial caps).
    """
    s0, s1, s2, s3 = loop
    # Coons sides: bottom c0->c1, right c1->c2, top c3->c2, left c0->c3.
    bottom, right, top, left = s0, s1, _rev(s2), _rev(s3)
    nu, ku = _structure(len(bottom[0][0]), bottom[1], len(top[0][0]), top[1], min_points)
    nv, kv = _structure(len(left[0][0]), left[1], len(right[0][0]), right[1], min_points)
    fit = lambda s, n, k: _refit_rows(s[0], len(s[0][0]), s[1], n, k)
    B, Tp = fit(bottom, nu, ku), fit(top, nu, ku)
    L, R = fit(left, nv, kv), fit(right, nv, kv)
    corners = [0.5 * (B[0][0] + L[0][0]), 0.5 * (B[0][-1] + R[0][0]),
               0.5 * (Tp[0][0] + L[0][-1]), 0.5 * (Tp[0][-1] + R[0][-1])]
    B[0][0], L[0][0] = corners[0], corners[0]
    B[0][-1], R[0][0] = corners[1], corners[1]
    Tp[0][0], L[0][-1] = corners[2], corners[2]
    Tp[0][-1], R[0][-1] = corners[3], corners[3]

    # Coons blend with Greville weights (linear precision for B-spline nets).
    u = greville(nu, ku)[:, None, None]
    v = greville(nv, kv)[None, :, None]
    Eb, Et, El, Er = B[0][:, None], Tp[0][:, None], L[0][None], R[0][None]
    net = ((1 - v) * Eb + v * Et + (1 - u) * El + u * Er
           - ((1 - u) * (1 - v) * corners[0] + u * (1 - v) * corners[1]
              + (1 - u) * v * corners[2] + u * v * corners[3]))

    if continuity != 'G0':
        width_v = np.linalg.norm(B[0] - Tp[0], axis=1).mean()
        width_u = np.linalg.norm(L[0] - R[0], axis=1).mean()
        acc = np.zeros_like(net)
        cnt = np.zeros(net.shape[:2] + (1,))
        for rows_, n, k, width, inner in (
                (B, nv, kv, width_v, lambda j: (slice(1, -1), j)),
                (Tp, nv, kv, width_v, lambda j: (slice(1, -1), -1 - j)),
                (L, nu, ku, width_u, lambda j: (j, slice(1, -1))),
                (R, nu, ku, width_u, lambda j: (-1 - j, slice(1, -1)))):
            E, T, A = rows_[:3]
            if T is None:
                continue  # a curve side: position only
            if len(rows_) > 3:
                alpha = rows_[3]  # reach fixed by the caller (N-sided quads)
            else:
                alpha = tension * width / max(np.linalg.norm(T, axis=1).mean(), 1e-12)
            # A side without curvature rows (A None) is joined with G1 only.
            rows = _taylor_rows(E, T, A, n, k, alpha, 'G1' if A is None else continuity)
            for j, r in enumerate(rows, 1):
                # g1_first: where sides overlap near a corner, tangent rows (G1)
                # win over curvature rows (G2): a kink shows more than a jump.
                w = 1e-3 if (g1_first and j > 1) else 1.0
                acc[inner(j)] += w * r[1:-1]
                cnt[inner(j)] += w
        mask = cnt[..., 0] > 0
        net[mask] = acc[mask] / cnt[mask]
    return net, (ku, kv)


def _orient(net, orders, ref_normal):
    """Flip the patch if its normal disagrees with ref_normal."""
    from .nurbs import eval_points
    du = eval_points(net, orders, [0.5], [0.5], 1, 0)[0]
    dv = eval_points(net, orders, [0.5], [0.5], 0, 1)[0]
    if np.dot(np.cross(du, dv), ref_normal) < 0:
        return net[:, ::-1]
    return net


def _halves(side):
    """Split a loop-oriented side at its middle into two sides."""
    from .split import _fit_piece
    rows, k, sign = side
    n = len(rows[0])
    spans = n - k + 1
    if spans == 1:
        n_half = n
    elif spans % 2 == 0:
        n_half = spans // 2 + k - 1
    else:
        n_half = (spans + 1) // 2 + 1 + k - 1
    have = [R for R in rows if _is_rows(R)]
    stack = np.stack(have, axis=1)  # (n, rows, 3)
    halves = []
    for t0, t1 in ((0.0, 0.5), (0.5, 1.0)):
        piece, _ = _fit_piece(stack, k, t0, t1, n_half)
        it = iter(piece[:, i] for i in range(piece.shape[1]))
        halves.append(([next(it) if _is_rows(R) else R for R in rows], k, sign))
    return halves


def _star_quads(loop, ref, continuity, tension):
    """One quad per corner of an N-sided loop, around a shared centre point."""
    N = len(loop)
    nc = ref / max(np.linalg.norm(ref), 1e-12)
    halves = [_halves(s) for s in loop]
    mids, along, into = [], [], []
    for side, (first, _) in zip(loop, halves):
        (E, T, _), k, _ = side
        n = len(E)
        mids.append(first[0][0][-1])
        along.append(basis_matrix(n, k, 0.5, 1)[0] @ E)
        into.append(None if T is None else basis_matrix(n, k, 0.5)[0] @ T)
    c0 = np.mean(mids, axis=0)
    for i in range(N):
        if into[i] is None:  # curve side: head for the middle, flat to the centre normal
            d = c0 - mids[i]
            into[i] = d - np.dot(d, nc) * nc
    into = [d / max(np.linalg.norm(d), 1e-12) for d in into]
    # Each middle "aims" at the centre along its cross tangent; average the aims.
    centre = np.mean([m + d * np.linalg.norm(c0 - m) for m, d in zip(mids, into)], axis=0)

    # Corner compatibility, so neighbouring quads see the same tangents:
    # - at a side's middle the spoke leaves along the side's cross tangent
    #   alpha_i T_i (and the half-sides use that same reach alpha_i);
    # - the spoke's cross tangent there is half the side's own tangent (each
    #   half-side covers half of the side's parameter range).
    # At the centre N quads meet; for N != 4 no regular quad layout is G1
    # there, so the spoke's cross tangent splits the two neighbours' wishes.
    T_mid, alpha, dirs, lengths = [], [], [], []
    for i, side in enumerate(loop):
        (E, T, _), k, _ = side[0][:3], side[1], side[2]
        L = np.linalg.norm(centre - mids[i])
        t_vec = into[i] * L if T is None else basis_matrix(len(E), k, 0.5)[0] @ T
        T_mid.append(t_vec)
        alpha.append(L / max(np.linalg.norm(t_vec), 1e-12) * tension)
        dc = mids[i] - centre
        dc = dc - np.dot(dc, nc) * nc
        dirs.append(dc / max(np.linalg.norm(dc), 1e-12))
        lengths.append(L)

    spokes = []
    for i in range(N):
        m, L = mids[i], lengths[i]
        P1 = m + alpha[i] * T_mid[i] / 5.0
        P4 = centre + dirs[i] * L / 5.0
        E = np.array([m, P1, m + 2 * (P1 - m), centre + 2 * (P4 - centre), P4, centre])
        D0 = -0.5 * along[i]
        D1 = 0.5 * (dirs[i - 1] * lengths[i - 1] - dirs[(i + 1) % N] * lengths[(i + 1) % N])
        D = D0 + (D1 - D0) * np.linspace(0.0, 1.0, 6)[:, None]
        spokes.append((E, D))

    quads = []
    for j in range(N):
        Ej, Dj = spokes[j]
        Ep, Dp = spokes[j - 1]
        zero = np.zeros_like(Dj)
        first = (halves[j][0][0][:3] + [alpha[j]], halves[j][0][1], halves[j][0][2])
        second = (halves[j - 1][1][0][:3] + [alpha[j - 1]], halves[j - 1][1][1],
                  halves[j - 1][1][2])
        spoke_j = ([Ej, Dj, zero, 1.0], 6, 0.0)      # m_j -> centre, pointing into quad j
        spoke_p = ([Ep, -Dp, zero, 1.0], 6, 0.0)     # m_{j-1} -> centre, into quad j
        loop_j = [first, spoke_j, _rev(spoke_p), second]
        quads.append(quad_net(loop_j, continuity, tension, STAR_POINTS))
    return quads


def _corner_turn(loop):
    """Turning angle (deg) of the boundary at every corner c_j (between side j-1 and j)."""
    out = []
    for j in range(len(loop)):
        (Ep, _, _), kp = loop[j - 1][0][:3], loop[j - 1][1]
        (Ej, _, _), kj = loop[j][0][:3], loop[j][1]
        a = basis_matrix(len(Ep), kp, 1.0, 1)[0] @ Ep
        b = basis_matrix(len(Ej), kj, 0.0, 1)[0] @ Ej
        cosang = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
        out.append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    return out


def _side_pieces(side, count):
    """Cut a loop-oriented side into `count` equal parameter pieces."""
    if count <= 1:
        return [side]
    from .split import _fit_piece
    rows, k, sign = side
    n = len(rows[0])
    spans = n - k + 1
    n_piece = max(spans // count, 1) + 1 + k - 1 if spans > 1 else n
    have = [R for R in rows if _is_rows(R)]
    stack = np.stack(have, axis=1)
    out = []
    for i in range(count):
        piece, _ = _fit_piece(stack, k, i / count, (i + 1) / count, n_piece)
        it = iter(piece[:, c] for c in range(piece.shape[1]))
        out.append(([next(it) if _is_rows(R) else R for R in rows], k, sign))
    return out


def _balanced(loop, sectors=6):
    """Split long sides so the radial patches cover similar parts of the boundary."""
    lengths = []
    for (rows, k, _) in loop:
        pts = basis_matrix(len(rows[0]), k, np.linspace(0, 1, 50)) @ rows[0]
        lengths.append(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    total = sum(lengths)
    target = max(sectors, len(loop))
    out = []
    for side, length in zip(loop, lengths):
        out += _side_pieces(side, max(1, int(round(length / total * target))))
    return out


def _radial_patches(loop, continuity, tension):
    """One patch per side, all meeting at a centre pole (a cap / dome).

    Used when the boundary has smooth "corners" (e.g. closing the open end
    of a tube), where quads would be crushed. Patch j: its side, two spokes
    from the side's ends to the centre, and a collapsed top at the centre.
    Neighbouring patches share each spoke with a common cross direction (the
    boundary tangent at the spoke's foot) fading to zero at the pole.
    """
    loop = _balanced(loop)
    N = len(loop)
    ends = []
    for side in loop:
        (E, T, A), k = side[0][:3], side[1]
        n = len(E)
        d0, d1 = basis_matrix(n, k, 0.0, 1)[0] @ E, basis_matrix(n, k, 1.0, 1)[0] @ E
        t0 = None if T is None else basis_matrix(n, k, 0.0)[0] @ T
        t1 = None if T is None else basis_matrix(n, k, 1.0)[0] @ T
        ends.append((E[0], d0, d1, t0, t1))
    corners = [e[0] for e in ends]
    c0 = np.mean(corners, axis=0)
    # Cross tangent at each corner: mean of the two sides' (curve sides: head inwards).
    t_corner = []
    for j in range(N):
        ts = [v for v in (ends[j][3], ends[j - 1][4]) if v is not None]
        t_corner.append(np.mean(ts, axis=0) if ts else c0 - corners[j])
    unit = [t / max(np.linalg.norm(t), 1e-12) for t in t_corner]
    axis = np.mean(unit, axis=0)
    if np.linalg.norm(axis) < 1e-6:  # flat-ish ring: use the plane normal
        axis = np.cross(corners[1] - corners[0], corners[2 % N] - corners[0])
    axis /= max(np.linalg.norm(axis), 1e-12)
    # Pole: over the centroid of the whole boundary, as far out along the cap's
    # axis as the boundary is wide (a hemisphere-like cap).
    samples = np.concatenate([basis_matrix(len(sd[0][0]), sd[1], np.linspace(0, 1, 40))
                              @ sd[0][0] for sd in loop])
    c0 = samples.mean(axis=0)
    radius = np.linalg.norm(samples - c0, axis=1).mean()
    centre = c0 + axis * radius * tension
    L = np.mean([np.linalg.norm(centre - c) for c in corners])
    alpha = tension * L / max(np.mean([np.linalg.norm(t) for t in t_corner]), 1e-12)

    spokes = []
    s = np.linspace(0.0, 1.0, 6)[:, None]
    for j in range(N):
        c = corners[j]
        dc = c - centre
        dc = dc - np.dot(dc, axis) * axis
        dc /= max(np.linalg.norm(dc), 1e-12)
        Lj = np.linalg.norm(centre - c)
        cubic = np.array([c, c + alpha * t_corner[j] / 3.0, centre + dc * Lj / 3.0, centre])
        # Shared cross direction: the boundary's tangent at the corner.
        a, b = ends[j - 1][2], ends[j][1]
        D = a / max(np.linalg.norm(a), 1e-12) + b / max(np.linalg.norm(b), 1e-12)
        D /= max(np.linalg.norm(D), 1e-12)
        spokes.append((elevate_curve(cubic, 5), D * (1.0 - s)))

    patches = []
    for j in range(N):
        rows, k, sign = loop[j]
        n = len(rows[0])
        side = (list(rows[:3]) + [alpha] if rows[1] is not None else list(rows[:3]), k, sign)
        Ej, Dj = spokes[j]
        En, Dn = spokes[(j + 1) % N]
        lam_l = np.linalg.norm(ends[j][1])          # side j's speed at its start
        lam_r = np.linalg.norm(ends[j][2])          # and at its end
        # Spokes join the neighbouring cap patches with G1 (A = None): a
        # curvature condition there would fight the boundary's own near the
        # corners. The boundary itself keeps the requested continuity.
        left = ([Ej, lam_l * Dj, None, 1.0], 6, 0.0)       # c_j -> centre, into patch j
        right = ([En, -lam_r * Dn, None, 1.0], 6, 0.0)     # c_j+1 -> centre, into patch j
        pole = ([np.repeat(centre[None], n, axis=0), None, None], k, 0.0)
        quad = [side, right, _rev(pole), _rev(left)]
        net, orders = quad_net(quad, continuity, tension, STAR_POINTS, g1_first=True)
        patches.append((net, orders, _side_normal(loop[j]) if rows[1] is not None else axis))
    return patches


SMOOTH_CORNER = 30.0  # deg: boundary turning less than this is not a real corner


def fill_patches(sides, continuity='G2', tension=1.0, tol=1e-3, layout='AUTO'):
    """Nets filling the hole bounded by `sides` (from side_from_patch/curve).

    layout: 'GRID' = one patch for four sides, else one quad per corner around
    a centre; 'RADIAL' = one patch per side meeting at a centre pole (caps,
    e.g. closing the end of a tube); 'AUTO' picks RADIAL when some boundary
    "corner" is smooth (quads would be crushed there), GRID otherwise.
    Returns [(net, orders)].
    """
    if continuity not in ('G0', 'G1', 'G2'):
        raise ValueError("continuity must be G0, G1 or G2")
    if len(sides) < 2:
        raise ValueError("a hole needs at least two sides")
    patch_first = sorted(sides, key=lambda s: s[0][1] is None)
    if patch_first[0][0][1] is None:
        raise ValueError("pick at least one patch edge (only curves: use Coons)")
    loop = order_loop(patch_first, tol)
    if layout == 'AUTO':
        layout = 'RADIAL' if (len(loop) < 3 or min(_corner_turn(loop)) < SMOOTH_CORNER)             else 'GRID'
    if layout == 'RADIAL':
        return [(_orient(net, orders, ref), orders)
                for net, orders, ref in _radial_patches(loop, continuity, tension)]
    if len(loop) < 3:
        raise ValueError("two sides can only be filled radially")
    normals = [_side_normal(s) for s in loop if s[0][1] is not None]
    ref = np.sum([v / max(np.linalg.norm(v), 1e-12) for v in normals], axis=0)
    if len(loop) == 4:
        nets = [quad_net(loop, continuity, tension, STAR_POINTS)]
    else:
        nets = _star_quads(loop, ref, continuity, tension)
    return [(_orient(net, orders, ref), orders) for net, orders in nets]


def fill_net(sides, continuity='G2', tension=1.0, tol=1e-3):
    """Single net filling a four-sided hole of patch edges [(P, orders, edge)]."""
    if len(sides) != 4:
        raise ValueError("pick exactly four edges")
    raw = [side_from_patch(P, orders, edge) for P, orders, edge in sides]
    return fill_patches(raw, continuity, tension, tol)[0]
