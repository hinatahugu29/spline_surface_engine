"""Core math tests. Run with:  py tests/test_core.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spline_surface"))
from core.bezier import elevate_curve, eval_curve, eval_surface, fit_curve  # noqa: E402
from core.extend import EDGES, extend_net  # noqa: E402
from core.loft import loft_net, order_sections  # noqa: E402
from core import nurbs  # noqa: E402
from core.continuity import find_shared_edge, grade, match_net, measure  # noqa: E402
from core.coons import close_loop, coons_net, loop_to_sides, orient_loop, snap_corners  # noqa: E402

rng = np.random.default_rng(0)
t = np.linspace(0, 1, 50)


def test_fit_recovers_bezier():
    P = rng.normal(size=(6, 3))
    Q, err = fit_curve(eval_curve(P, t), 5)
    assert err < 1e-6, err
    assert np.allclose(eval_curve(Q, t), eval_curve(P, t), atol=1e-5)


def test_elevate_keeps_shape():
    P = rng.normal(size=(4, 3))
    assert np.allclose(eval_curve(elevate_curve(P, 5), t), eval_curve(P, t))


def test_coons_reproduces_boundaries():
    corners = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], float)

    def side(a, b, bump):
        P = np.linspace(a, b, 6)
        P[1:-1, 2] += bump
        return P

    bottom = side(corners[0], corners[1], 1.0)
    right = side(corners[1], corners[2], -0.5)
    top = side(corners[3], corners[2], 0.7)
    left = side(corners[0], corners[3], 0.2)
    # Feed them shuffled and reversed, as a user selection would be.
    polys = [eval_curve(c, t) for c in (top[::-1], bottom, left, right[::-1])]
    loop = orient_loop(polys, tol=1e-6)
    fitted = [fit_curve(p, 5)[0] for p in loop]
    b, r, tp, lf = snap_corners(*loop_to_sides(fitted))
    S = eval_surface(coons_net(b, r, tp, lf), t, t)
    ref = {"bottom": (S[:, 0], bottom), "top": (S[:, -1], top),
           "left": (S[0, :], left), "right": (S[-1, :], right)}
    for name, (got, ctrl) in ref.items():
        exp = eval_curve(ctrl, t)
        # The loop may start on any side; compare against either orientation
        # of the surface boundary set.
        ok = any(np.allclose(g, e, atol=1e-5) for g in
                 (S[:, 0], S[:, -1], S[0, :], S[-1, :], S[::-1, 0], S[::-1, -1],
                  S[0, ::-1], S[-1, ::-1]) for e in (exp,))
        assert ok, name


def test_three_sided():
    a, b, c = np.array([0, 0, 0.]), np.array([2, 0, 0.]), np.array([1, 2, 0.])
    polys = [np.linspace(a, b, 10), np.linspace(b, c, 10), np.linspace(c, a, 10)]
    loop = orient_loop(polys, tol=1e-6)
    sides = snap_corners(*loop_to_sides([fit_curve(p, 3)[0] for p in loop]))
    S = eval_surface(coons_net(*sides), t, t)
    assert np.allclose(S[0, :], a)  # collapsed side


def test_open_loop_rejected():
    polys = [np.linspace([0, 0, 0], [1, 0, 0], 5), np.linspace([1, 0, 0], [1, 1, 0], 5),
             np.linspace([1, 1, 0], [0, 1, 0], 5), np.linspace([0, 1, 0], [0, 0.5, 0], 5)]
    try:
        orient_loop(polys, tol=1e-3)
    except ValueError:
        return
    raise AssertionError("open loop accepted")


def _derivs(S, axis, at_end):
    """1st/2nd finite differences across `axis` at one boundary of a sampled grid."""
    S = np.moveaxis(S, axis, 1)
    if at_end:
        S = S[:, ::-1]
    return S[:, 0], S[:, 0] - S[:, 1], S[:, 0] - 2 * S[:, 1] + S[:, 2]


def test_extend_continuity():
    n_s = 2001
    P = rng.normal(scale=0.3, size=(6, 5, 3))
    P[..., 0] += np.arange(6)[:, None]
    P[..., 1] += np.arange(5)[None, :]
    u = np.linspace(0, 1, 7)
    s = np.linspace(0, 1, n_s)
    h = 0.4
    for edge in EDGES:
        Q, _ = extend_net(P, edge, length=h, continuity="G2", degree=4)
        axis = 1 if edge[0] == "V" else 0
        src = eval_surface(P, u, s) if axis == 1 else eval_surface(P, s, u)
        ext = eval_surface(Q, u, s)
        a0, a1, a2 = _derivs(src, axis, at_end=edge[1] == "1")
        # Extension: edge at v=0, flip along-edge if needed to line up.
        e0, e1, e2 = _derivs(ext, 1, at_end=False)
        if not np.allclose(e0, a0, atol=1e-9):
            e0, e1, e2 = e0[::-1], e1[::-1], e2[::-1]
        assert np.allclose(e0, a0, atol=1e-9), edge
        # Same parameter speed ratio h: d_ext = -h * d_src (pointing outward).
        assert np.allclose(e1, -h * a1, atol=1e-5), edge
        assert np.allclose(e2, h * h * a2, atol=1e-5), edge
        # Normals agree in direction.
        nu = lambda S: np.cross(S[1:, :-1] - S[:-1, :-1], S[:-1, 1:] - S[:-1, :-1]).sum(axis=(0, 1))  # noqa: E731
        assert np.dot(nu(eval_surface(P, s[::50], s[::50])), nu(eval_surface(Q, s[::50], s[::50]))) > 0, edge


def _bumpy(nu, nv, scale=0.3):
    P = rng.normal(scale=scale, size=(nu, nv, 3))
    P[..., 0] += np.linspace(0, 4, nu)[:, None]
    P[..., 1] += np.linspace(0, 3, nv)[None, :]
    return P


def test_measure_and_match():
    A = _bumpy(6, 5)
    B, _ = extend_net(A, "V1", 0.5, "G2", degree=4)
    shared = find_shared_edge(B, A, tol=1e-6)
    assert shared is not None
    assert grade(measure(B, A, *shared[:3])) == "G2"

    # Break it, then repair to each level. Target has a different along-edge
    # degree, runs the other way and is only degree 2 across.
    for cont in ("G0", "G1", "G2"):
        T = _bumpy(4, 3, scale=0.2)
        T[..., 1] += 3.0  # sits above A's V1 edge
        T = T[::-1]       # edge runs opposite to A's
        T[:, 0] = np.linspace(A[-1, -1], A[0, -1], 4) + 0.05
        shared = find_shared_edge(T, A, tol=0.5)
        assert shared is not None and shared[2], shared
        M, _ = match_net(T, A, *shared[:3], cont)
        assert M.shape[0] == 6 and M.shape[1] >= {"G0": 3, "G1": 3, "G2": 4}[cont]
        g = grade(measure(M, A, *shared[:3]))
        assert g == cont or (cont == "G0" and g in ("G0", "G1")), (cont, g)
        # Edge opposite to the seam is unchanged in shape.
        far = eval_curve(M[:, -1], t)
        far_ref = eval_curve(T[:, -1], t)
        assert np.allclose(far, far_ref, atol=1e-9), cont


def _section(x, bump, n=40):
    y = np.linspace(-1, 1, n)
    return np.stack([np.full(n, x), y, bump * (1 - y * y)], axis=1)


def test_loft_interpolates_sections():
    secs = [_section(x, b) for x, b in [(0, 0.2), (1, 0.6), (2.2, 0.5), (3, 0.1)]]
    shuffled = [secs[2], secs[0][::-1], secs[3], secs[1][::-1]]
    order = order_sections(shuffled)
    assert [i for i, _ in order] in ([1, 3, 0, 2], [2, 0, 3, 1]), order
    rows = []
    for i, flip in order:
        p = shuffled[i][::-1] if flip else shuffled[i]
        rows.append(fit_curve(p, 4)[0])
    # All sections end up running the same way.
    assert all(np.sign(r[-1, 1] - r[0, 1]) == np.sign(rows[0][-1, 1] - rows[0][0, 1]) for r in rows)
    net, v = loft_net(rows)
    assert net.shape == (5, 4, 3)
    S = eval_surface(net, t, v)
    for k, r in enumerate(rows):
        assert np.allclose(S[:, k], eval_curve(r, t), atol=1e-9), k


def test_loft_many_sections_and_two():
    secs = [fit_curve(_section(x, 0.3 * np.sin(x)), 3)[0] for x in np.linspace(0, 6, 9)]
    net, v = loft_net(secs, max_degree=5)
    assert net.shape == (4, 6, 3)
    assert np.allclose(net[:, 0], secs[0]) and np.allclose(net[:, -1], secs[-1])
    net2, _ = loft_net(secs[:2])
    assert net2.shape == (4, 2, 3)


def test_multispan_extend_match_measure():
    # Source: 7 x 6 points, order 6 x 4 -> 2 spans in U, 3 spans in V.
    A = _bumpy(7, 6)
    oa = (6, 4)
    for edge in EDGES:
        Q, oq = extend_net(A, edge, 0.4, "G2", degree=4, orders=oa)
        shared = find_shared_edge(Q, A, 1e-6, oq, oa)
        assert shared is not None, edge
        assert grade(measure(Q, A, *shared[:3], oq, oa)) == "G2", edge
    # Multi-span target matched to the multi-span source along V1 (U structure shared).
    T = _bumpy(7, 5, scale=0.2)
    T[..., 1] += 3.05
    shared = find_shared_edge(T, A, 0.5, (6, 4), oa)
    for cont in ("G0", "G1", "G2"):
        M, om = match_net(T, A, *shared[:3], cont, (6, 4), oa)
        assert om == (6, 4) and M.shape == T.shape
        g = grade(measure(M, A, *shared[:3], om, oa))
        assert g == cont or (cont == "G0" and g in ("G0", "G1")), (cont, g)
        far = nurbs.eval_curve(M[:, -1], 6, t)
        assert np.allclose(far, nurbs.eval_curve(T[:, -1], 6, t)), cont
    # Mismatched along-edge structure is refused with an explanation.
    try:
        match_net(_bumpy(6, 5), A, *shared[:3], "G1", (5, 4), oa)
    except ValueError as e:
        assert "differ" in str(e)
    else:
        raise AssertionError("mismatch accepted")


def _poly_patch(x0, y0, n=6):
    """Bezier net of z = f(x, y) (cubic) on [x0, x0+1] x [y0, y0+1]."""
    from core.bezier import bernstein
    t = np.linspace(0, 1, 15)
    Bt = bernstein(n - 1, t)
    X, Y = x0 + t[:, None] + 0 * t, y0 + 0 * t[:, None] + t
    f = lambda x, y: 0.1 * x * x - 0.05 * x * y + 0.08 * y * y + 0.01 * x ** 3
    pinv = np.linalg.pinv(Bt)
    net = np.zeros((n, n, 3))
    for c, val in enumerate((X, Y, f(X, Y))):
        net[..., c] = pinv @ val @ pinv.T
    return net


def test_fill_hole_g2():
    from core.fill import fill_net
    patches = {(i, j): _poly_patch(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)}
    sides = [(patches[(1, 0)], None, 'V1'), (patches[(2, 1)], None, 'U0'),
             (patches[(1, 2)], None, 'V0'), (patches[(0, 1)], None, 'U1')]
    sides = [(P, (6, 6), e) for P, _, e in sides]
    for order in ([0, 1, 2, 3], [2, 0, 3, 1]):
        net, orders = fill_net([sides[i] for i in order], 'G2')
        for P, o, e in sides:
            sh = find_shared_edge(P, net, 1e-6, o, orders)
            assert sh is not None, e
            m = measure(P, net, sh[0], sh[1], sh[2], o, orders)
            assert m["gap"] < 1e-9 and m["angle"] < 0.1 and m["curvature"] < 0.05, (e, m)
        # The exact polynomial surface should be reproduced in the middle.
        mid = nurbs.eval_surface(net, orders, [0.5], [0.5])[0, 0]
        assert abs(mid[2] - (0.1 * 2.25 - 0.05 * 2.25 + 0.08 * 2.25 + 0.01 * 3.375)) < 0.02, mid


def test_blend_bridge_g2():
    from core.blend import blend_net
    A, B = _poly_patch(0, 1), _poly_patch(2, 1)
    for flip_b in (False, True):
        Pb = B[:, ::-1] if flip_b else B  # reversed parametrisation must not twist
        net, orders = blend_net(A, (6, 6), 'U1', Pb, (6, 6), 'U0', 'G2')
        for P, e in ((A, 'U1'), (Pb, 'U0')):
            sh = find_shared_edge(P, net, 1e-9, (6, 6), orders)
            assert sh is not None, e
            m = measure(P, net, sh[0], sh[1], sh[2], (6, 6), orders)
            assert grade(m) == 'G2', (e, m)
        mid = nurbs.eval_surface(net, orders, [0.5], [0.5])[0, 0]
        assert np.linalg.norm(mid - [1.5, 1.5, 0.1 * 2.25 - 0.05 * 2.25 + 0.08 * 2.25
                                     + 0.01 * 3.375]) < 0.05, mid


def test_mirror_center_g2():
    from core.mirror import edges_on_plane, mirror_net, symmetrize
    P = _poly_patch(0, 0)  # u runs along x from the plane x = 0
    P[0, :, 0] = 0.0
    P[1:, :, 2] += 0.3 * np.linspace(1, 0, 6)[:, None][1:] * 1.0  # tilt near the centre
    assert edges_on_plane(P) == ['U0']
    S, fixed = symmetrize(P)
    assert fixed == ['U0']
    M = mirror_net(S)
    sh = find_shared_edge(S, M, 1e-9, (6, 6), (6, 6))
    m = measure(S, M, sh[0], sh[1], sh[2], (6, 6), (6, 6))
    assert grade(m) == 'G2', m
    # before the fix the centre has a crease
    sh = find_shared_edge(P, mirror_net(P), 1e-9, (6, 6), (6, 6))
    assert grade(measure(P, mirror_net(P), *sh[:3], (6, 6), (6, 6))) != 'G2'
    # normals of the mirror point the same way (seen from +z both face up)
    n = lambda N: np.cross(N[1, 2] - N[0, 2], N[0, 3] - N[0, 2])[2]
    assert np.sign(n(S)) == np.sign(n(M))


def test_split_exact_and_fitted():
    from core.split import split_net
    P = _poly_patch(0, 0)
    for direction in ('U', 'V'):
        A, B, oa, ob, t, err = split_net(P, (6, 6), direction, 0.37)
        assert t == 0.37 and err < 1e-9, err  # Bezier: exact anywhere
        g = np.linspace(0, 1, 7)
        if direction == 'U':
            ref = nurbs.eval_surface(P, (6, 6), 0.37 * g, g)
        else:
            ref = nurbs.eval_surface(P, (6, 6), g, 0.37 * g)
        assert np.abs(nurbs.eval_surface(A, oa, g, g) - ref).max() < 1e-9
        sh = find_shared_edge(A, B, 1e-9, oa, ob)
        assert grade(measure(A, B, *sh[:3], oa, ob)) == 'G2'
    # three spans: snapping lands on a span boundary and stays exact
    M = _bumpy(8, 6)
    A, B, oa, ob, t, err = split_net(M, (6, 6), 'U', 0.4)
    assert abs(t - 1 / 3) < 1e-12 and err < 1e-9 and A.shape[0] == 6 and B.shape[0] == 7
    # off a boundary: fitted, still sharing the seam exactly and close to G2
    A, B, oa, ob, t, err = split_net(M, (6, 6), 'U', 0.4, snap=False)
    assert t == 0.4 and err < 0.05, err
    sh = find_shared_edge(A, B, 1e-9, oa, ob)
    m = measure(A, B, *sh[:3], oa, ob)
    assert grade(m) == "G2", m


def _ring_patches(n_sides, radius=1.0, width=0.6):
    """Patches around an n-gon hole in z = f(x, y); edge V0 of each lies on the hole."""
    from core.bezier import bernstein
    f = lambda x, y: 0.15 * x * x - 0.1 * x * y + 0.12 * y * y + 0.03 * x ** 3
    ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + 0.3
    c = np.stack([np.cos(ang), np.sin(ang)], axis=1) * radius
    t = np.linspace(0, 1, 15)
    pinv = np.linalg.pinv(bernstein(5, t))
    out = []
    for i in range(n_sides):
        a, b = c[i], c[(i + 1) % n_sides]
        o = np.array([b[1] - a[1], a[0] - b[0]])
        o = o / np.linalg.norm(o) * width  # outward for a counter-clockwise loop
        U, V = t[:, None], t[None, :]
        X = (1 - V) * ((1 - U) * a[0] + U * b[0]) + V * ((1 - U) * (a[0] + o[0]) + U * (b[0] + o[0]))
        Y = (1 - V) * ((1 - U) * a[1] + U * b[1]) + V * ((1 - U) * (a[1] + o[1]) + U * (b[1] + o[1]))
        net = np.stack([pinv @ C @ pinv.T for C in (X, Y, f(X, Y))], axis=-1)
        out.append(net)
    return out


def _boundary_angle(P, orders, edge, Q, oq):
    """Max normal angle (deg) between patch P along `edge` and patch Q where they touch."""
    g = np.linspace(0, 1, 400)
    c = np.full_like(g, 0.0 if edge[1] == '0' else 1.0)
    uv = (c, g) if edge[0] == 'U' else (g, c)
    pts = nurbs.eval_points(P, orders, *uv)
    nP = np.cross(nurbs.eval_points(P, orders, *uv, 1, 0), nurbs.eval_points(P, orders, *uv, 0, 1))
    worst, hits = 0.0, 0
    s = np.linspace(0, 1, 41)
    for e in ('U0', 'U1', 'V0', 'V1'):
        cc = np.full_like(s, 0.0 if e[1] == '0' else 1.0)
        quv = (cc, s) if e[0] == 'U' else (s, cc)
        q = nurbs.eval_points(Q, oq, *quv)
        nQ = np.cross(nurbs.eval_points(Q, oq, *quv, 1, 0), nurbs.eval_points(Q, oq, *quv, 0, 1))
        for k in range(2, 39):  # stay clear of the corner zones
            d = np.linalg.norm(pts - q[k], axis=1)
            j = np.argmin(d)
            if d[j] < 1e-3:
                hits += 1
                a, b = nP[j], nQ[k]
                cosang = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
                worst = max(worst, np.degrees(np.arccos(min(1.0, cosang))))
    return worst, hits


def test_fill_nsided():
    from core.fill import fill_patches, side_from_patch
    for n_sides in (3, 5):
        ring = _ring_patches(n_sides)
        sides = [side_from_patch(P, (6, 6), 'V0') for P in ring]
        quads = fill_patches(sides, 'G2', tol=1e-6)
        assert len(quads) == n_sides
        for P in ring:
            touched = 0
            for Q, oq in quads:
                ang, hits = _boundary_angle(P, (6, 6), 'V0', Q, oq)
                touched += hits
                assert ang < 1.0, (n_sides, ang)
            assert touched > 20
        # neighbouring quads share a spoke and meet smoothly
        for j in range(n_sides):
            (A, oa), (B, ob_) = quads[j], quads[(j + 1) % n_sides]
            sh = find_shared_edge(A, B, 1e-6, oa, ob_)
            assert sh is not None
            m = measure(A, B, *sh[:3], oa, ob_)
            assert m["gap"] < 1e-9 and m["angle"] < 1.0, (n_sides, j, m)


def test_offset_distance_and_seams():
    from core.offset import offset_net
    A, B = _poly_patch(0, 0), _poly_patch(1, 0)  # neighbours sharing U1 / U0
    oA, _, eA = offset_net(A, (6, 6), 0.1)
    oB, _, eB = offset_net(B, (6, 6), 0.1)
    assert max(eA, eB) < 1e-3, (eA, eB)
    # every offset point is ~0.1 from the source surface
    g = np.linspace(0, 1, 25)
    src = nurbs.eval_surface(A, (6, 6), g, g).reshape(-1, 3)
    off = nurbs.eval_surface(oA, (6, 6), g[2:-2], g[2:-2]).reshape(-1, 3)
    d = np.linalg.norm(off[:, None] - src[None], axis=2).min(axis=1)
    fine = nurbs.eval_surface(A, (6, 6), np.linspace(0, 1, 120), np.linspace(0, 1, 120)).reshape(-1, 3)
    d = np.linalg.norm(off[:, None] - fine[None], axis=2).min(axis=1)
    assert np.abs(d - 0.1).max() < 3e-3, np.abs(d - 0.1).max()
    # offsets of G2 neighbours still meet
    sh = find_shared_edge(oA, oB, 1e-3, (6, 6), (6, 6))
    assert sh is not None
    m = measure(oA, oB, *sh[:3], (6, 6), (6, 6))
    assert m["gap"] < 1e-3 and m["angle"] < 0.5, m


def _network_curves():
    f = lambda x, y: 0.15 * x * x - 0.1 * x * y + 0.12 * y * y + 0.03 * x ** 3
    s = np.linspace(0, 1, 120)
    curves = []
    for y in (0.0, 1.0, 2.0):  # u-curves, overhanging the grid a little
        x = -0.3 + 3.6 * s
        curves.append(np.stack([x, np.full_like(x, y), f(x, y)], axis=1))
    for x in (0.0, 1.0, 1.7, 3.0):  # v-curves
        y = -0.2 + 2.4 * s
        curves.append(np.stack([np.full_like(y, x), y, f(x, y)], axis=1))
    return curves, f


def test_curve_network():
    from core.network import build_network
    curves, f = _network_curves()
    order = [4, 0, 6, 2, 5, 1, 3]  # shuffled, some reversed
    polys = [curves[i][::-1] if i % 2 else curves[i] for i in order]
    cells, info = build_network(polys, tol=1e-3)
    assert sorted(info["size"]) == [2, 3] and len(cells) == 6, info
    nets = {(i, j): (net, o) for i, j, net, o in cells}
    # every cell hugs the surface the curves were drawn on
    g = np.linspace(0, 1, 9)
    for (net, o) in nets.values():
        pts = nurbs.eval_surface(net, o, g, g).reshape(-1, 3)
        assert np.abs(pts[:, 2] - f(pts[:, 0], pts[:, 1])).max() < 0.02
    # neighbouring cells share their curve exactly and join smoothly
    worst = 0.0
    for (i, j), (A, oa) in nets.items():
        for (di, dj) in ((1, 0), (0, 1)):
            if (i + di, j + dj) in nets:
                B, ob_ = nets[(i + di, j + dj)]
                sh = find_shared_edge(A, B, 1e-9, oa, ob_)
                assert sh is not None, (i, j, di, dj)
                m = measure(A, B, *sh[:3], oa, ob_)
                worst = max(worst, m["angle"])
                assert m["gap"] < 1e-9 and m["angle"] < 0.5, (i, j, di, dj, m)
    # normals all face the same way
    ns = [np.cross(*(nurbs.eval_surface(n, o, [0.5], [0.5], 1, 0)[0, 0],
                     nurbs.eval_surface(n, o, [0.5], [0.5], 0, 1)[0, 0])) for n, o in nets.values()]
    assert all(np.dot(ns[0], v) > 0 for v in ns)


def test_curve_network_top_view():
    """Curves crossing only in top view (different heights) still make a G2 network."""
    from core.network import build_network
    curves, f = _network_curves()
    for k in range(3, 7):
        curves[k] = curves[k] + [0, 0, 0.4 + 0.05 * k]
    cells, info = build_network(curves, tol=0.06)
    assert len(cells) == 6 and info["max_gap"] > 0.4
    nets = {(i, j): (net, o) for i, j, net, o in cells}
    for (i, j), (A, oa) in nets.items():
        for (di, dj) in ((1, 0), (0, 1)):
            if (i + di, j + dj) in nets:
                B, ob_ = nets[(i + di, j + dj)]
                sh = find_shared_edge(A, B, 1e-9, oa, ob_)
                assert grade(measure(A, B, *sh[:3], oa, ob_)) == 'G2'


def test_fill_tube_end_cap():
    """Closing the open end of a tube (smooth outline) goes radial and stays G1."""
    from core.bezier import bernstein
    from core.fill import fill_patches, side_from_patch
    t = np.linspace(0, 1, 15)
    pinv = np.linalg.pinv(bernstein(5, t))
    ring = []
    for q in range(4):  # an oval cross-section in the yz plane, extruded along +x
        a = (q + t) * np.pi / 2
        y, z = 1.5 * np.cos(a), np.sin(a)
        X = 0.0 + 2.0 * t[None, :] + 0 * y[:, None]
        Y, Z = y[:, None] + 0 * X, z[:, None] + 0 * X
        ring.append(np.stack([pinv @ C @ pinv.T for C in (X, Y, Z)], axis=-1))
    sides = [side_from_patch(P, (6, 6), 'V0') for P in ring]  # the x = 0 end
    parts = fill_patches(sides, 'G2', tol=1e-3)
    assert len(parts) >= 4
    g = np.linspace(0.02, 0.98, 30)
    for net, o in parts:  # tangent plane at the outline must contain the x axis
        du = nurbs.eval_points(net, o, g, np.zeros_like(g), 1, 0)
        dv = nurbs.eval_points(net, o, g, np.zeros_like(g), 0, 1)
        n = np.cross(du, dv)
        n /= np.linalg.norm(n, axis=1)[:, None]
        assert np.degrees(np.arcsin(np.abs(n[:, 0]).max())) < 0.1
        assert nurbs.eval_points(net, o, [0.5], [0.5])[0][0] < -0.3  # bulges outwards
    for j in range(len(parts)):  # neighbouring cap patches meet with G1
        (A, oa), (B, ob_) = parts[j], parts[(j + 1) % len(parts)]
        v = np.linspace(0.02, 0.95, 20)
        na = np.cross(nurbs.eval_points(A, oa, np.ones_like(v), v, 1, 0),
                      nurbs.eval_points(A, oa, np.ones_like(v), v, 0, 1))
        nb = np.cross(nurbs.eval_points(B, ob_, np.zeros_like(v), v, 1, 0),
                      nurbs.eval_points(B, ob_, np.zeros_like(v), v, 0, 1))
        cosang = np.abs((na * nb).sum(1)) / np.linalg.norm(na, axis=1) / np.linalg.norm(nb, axis=1)
        assert np.degrees(np.arccos(np.clip(cosang, 0, 1))).max() < 0.1


def test_falloff_field():
    from core.linked import falloff_field
    src = np.array([[0, 0, 0], [1, 0, 0]], float)
    disp = np.array([[0, 0, 1], [0, 0, 0]], float)
    f = falloff_field(src, disp, 2.0)
    out = f(np.array([[0, 0, 0], [1, 0, 0], [0, 5, 0], [0, 0.5, 0]], float))
    assert np.allclose(out[0], [0, 0, 1], atol=1e-6) and np.allclose(out[1], 0, atol=1e-6)
    assert np.allclose(out[2], 0) and 0 < out[3, 2] < 1


def test_order_chain():
    from core.linked import chain_polyline, order_chain
    a = np.array([[0, 0, 0], [1, 0, 0]], float)
    b = np.array([[2, 0, 0], [1, 0, 0]], float)   # reversed
    c = np.array([[2, 0, 0], [3, 0, 0]], float)
    d = np.array([[9, 9, 9], [8, 9, 9]], float)   # not joined
    chain = order_chain({"b": b, "a": a, "c": c, "d": d}, 1e-6)
    assert [k for k, _ in chain] in (["a", "b", "c"], ["c", "b", "a"]), chain
    pts, ranges = chain_polyline({"a": a, "b": b, "c": c}, chain)
    assert len(pts) == 4 and np.isclose(ranges[0][0], 0) and np.isclose(ranges[-1][1], 1)
    assert all(np.isclose(r0[1], r1[0]) for r0, r1 in zip(ranges, ranges[1:]))


def test_merge_grid():
    from core.linked import grid_layout, merge_grid
    from core.network import build_network
    curves, f = _network_curves()
    cells, _ = build_network(curves, tol=1e-3)
    patches = []
    for n, (i, j, net, o) in enumerate(cells):
        if n % 2:  # lay some cells the other way round
            net, o = net.transpose(1, 0, 2)[::-1], (o[1], o[0])
        patches.append((net, o))
    grid = grid_layout(patches, 1e-4)
    assert len(grid) == 6
    net, orders, err = merge_grid(grid)
    assert err < 0.01, err
    g = np.linspace(0, 1, 9)
    pts = nurbs.eval_surface(net, orders, g, g).reshape(-1, 3)
    assert np.abs(pts[:, 2] - f(pts[:, 0], pts[:, 1])).max() < 0.02
    try:
        grid_layout(patches[:3] + patches[4:5], 1e-4)
    except ValueError:
        pass
    else:
        raise AssertionError("an L shape is not a grid")


def test_lock_mask():
    from core.lock import lock_mask
    m = lock_mask((7, 6), [('U1', 'G2')])
    assert m[-3:].all() and not m[:-3].any()  # U1 = last u row, three deep
    m = lock_mask((7, 6), [('V0', 'G1'), ('U0', 'G0'), ('V1', 'open')])
    assert m[:, :2].all() and m[0].all()
    assert m.sum() == 7 * 2 + 6 - 2  # V0 two columns plus the U0 row minus the overlap
    assert not m[1:, -1].any()
    m = lock_mask((4, 4), [('U0', 'G2'), ('U1', 'G2')])
    assert m.all()  # opposite G2 seams on a 4-row patch pin everything
    assert not lock_mask((5, 5), []).any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
