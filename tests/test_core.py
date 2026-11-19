"""Core math tests. Run with:  py tests/test_core.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spline_surface"))
from core.bezier import elevate_curve, eval_curve, eval_surface, fit_curve  # noqa: E402
from core.extend import EDGES, extend_net  # noqa: E402
from core.loft import add_rows, loft_net, order_sections  # noqa: E402
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



def test_match_far_and_spread():
    """A seam pulled far away: tangents keep their scale; Spread bends the patch
    smoothly and keeps the opposite edge."""
    from core.extend import edge_frame
    A = _bumpy(6, 5)
    T, _ = extend_net(A, "V1", 0.6, "G2", degree=4)
    shared = find_shared_edge(T, A, tol=1e-6)
    A2 = A + np.array([0.0, 0.0, 1.5])     # the reference moves far, beyond any tolerance
    assert find_shared_edge(T, A2, tol=0.1) is None

    def framed(M):                          # seam row first, opposite edge last
        return edge_frame(np.asarray(M), shared[0])[0][:, ::-1]

    def first_step(M):
        F = framed(M)
        return np.linalg.norm(F[:, 1] - F[:, 0], axis=1).mean()

    F0 = framed(T)
    for cont in ("G1", "G2"):
        plain, _ = match_net(T, A2, *shared[:3], cont)
        soft, _ = match_net(T, A2, *shared[:3], cont, spread=True)
        for M in (plain, soft):
            g = grade(measure(M, A2, *shared[:3]))
            assert g == cont or (cont == "G1" and g == "G2"), (cont, g)
            assert np.allclose(framed(M)[:, -1], F0[:, -1])      # opposite edge kept
        # The target's own tangent scale: read after the seam row moved, it shot off.
        assert first_step(plain) < 3 * first_step(T), (first_step(plain), first_step(T))
        # Spread: the rows inside follow the seam part of the way (no ridge).
        mid = F0.shape[1] // 2
        assert np.abs(framed(soft)[:, mid] - F0[:, mid]).max() > 0.2

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


def test_loft_extra_rows_keep_shape():
    secs = [fit_curve(_section(x, 0.3 * np.sin(x)), 3)[0] for x in (0.0, 2.0)]
    net, _ = loft_net(secs)
    more = add_rows(net, 2)
    assert more.shape == (4, 4, 3)
    assert np.allclose(more[:, 0], net[:, 0]) and np.allclose(more[:, -1], net[:, -1])
    t = np.linspace(0, 1, 7)
    assert np.allclose(eval_surface(more, t, t), eval_surface(net, t, t), atol=1e-12)
    # evenly spaced on the straight lines between the sections
    assert np.allclose(more[:, 1], net[:, 0] + (net[:, -1] - net[:, 0]) / 3)
    assert add_rows(net, 9, max_degree=5).shape[1] == 6 and add_rows(net, 0).shape == net.shape


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


def test_iso_bezier_exact_and_close():
    from core.split import iso_bezier, iso_line
    rng = np.random.default_rng(5)

    def bezier_pts(segs, m=20):
        out = []
        for (a, _, ar), (b, bl, _) in zip(segs[:-1], segs[1:]):
            x = np.linspace(0, 1, m)[:, None]
            out.append((1 - x) ** 3 * a + 3 * (1 - x) ** 2 * x * ar + 3 * (1 - x) * x ** 2 * bl
                       + x ** 3 * b)
        return np.concatenate(out)
    for shape, orders, exact in (((7, 6), (4, 4), True), ((5, 4), (3, 2), True),
                                 ((8, 8), (6, 6), False)):
        P = rng.normal(size=shape + (3,))
        for direction in "UV":
            for t in (0.0, 0.37, 1.0):
                segs = iso_bezier(P, orders, direction, t)
                axis = 1 if direction == 'U' else 0
                spans = shape[axis] - orders[axis] + 1
                assert len(segs) == (spans if exact else spans * 4) + 1
                got = bezier_pts(segs)
                m = (len(segs) - 1) * 20
                want = iso_line(P, orders, direction, t, samples=len(segs))   # segment ends
                assert np.abs(np.array([g[0] for g in segs]) - want).max() < 1e-12
                # same parameter on both sides: segment j, local x <-> line parameter
                x = np.concatenate([(j + np.linspace(0, 1, 20)) / (len(segs) - 1)
                                    for j in range(len(segs) - 1)])
                from core.nurbs import eval_points
                cst = np.full(len(x), t)
                ref = (eval_points(P, orders, cst, x) if direction == 'U'
                       else eval_points(P, orders, x, cst))
                err = np.abs(got - ref).max() / np.ptp(P.reshape(-1, 3), 0).max()
                assert err < (1e-12 if exact else 2e-3), (shape, orders, direction, t, err)
                assert len(got) == m


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


def _ring_patches(n_sides, radius=1.0, width=0.6, tilt=None):
    """Patches around an n-gon hole in z = f(x, y); edge V0 of each lies on the hole.

    With `tilt`, the hole lies in z = 0 and each patch rises tilt * width away from it.
    """
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
        Z = f(X, Y) if tilt is None else tilt * width * (V + 0 * U)
        net = np.stack([pinv @ C @ pinv.T for C in (X, Y, Z)], axis=-1)
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


def test_fill_g0_flat_lid():
    from core.fill import fill_patches, side_from_patch
    # A planar hole whose neighbours tilt up away from it (a bowl rim): G1/G2
    # follow the tilt, G0 lays a flat lid across the boundary.
    for n_sides, layout in ((3, 'AUTO'), (4, 'AUTO'), (5, 'AUTO'), (6, 'RADIAL')):
        ring = _ring_patches(n_sides, tilt=0.5)
        sides = [side_from_patch(P, (6, 6), 'V0') for P in ring]
        assert max(abs(s[0][0][:, 2]).max() for s in sides) < 1e-6  # the hole is planar
        lid = fill_patches(sides, 'G0', tol=1e-6, layout=layout)
        g = np.linspace(0, 1, 9)
        z = max(np.abs(nurbs.eval_surface(n, o, g, g)[..., 2]).max() for n, o in lid)
        assert z < 1e-6, (n_sides, z)
        dome = fill_patches(sides, 'G2', tol=1e-6, layout=layout)
        assert max(np.abs(nurbs.eval_surface(n, o, g, g)[..., 2]).max() for n, o in dome) > 0.01
        # the lid still faces the same way as its neighbours
        normal = lambda n, o: np.cross(nurbs.eval_points(n, o, [0.5], [0.3], 1, 0)[0],
                                       nurbs.eval_points(n, o, [0.5], [0.3], 0, 1)[0])
        side = np.sign(normal(ring[0], (6, 6))[2])
        assert all(np.sign(normal(n, o)[2]) == side for n, o in lid), n_sides


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


def test_diagnose_network():
    from core.network import diagnose_network
    curves, f = _network_curves()   # 3 curves along x (0-2), 4 along y (3-6)
    d = diagnose_network(curves, 1e-3)
    assert d["ok"] and sorted(d["counts"]) == [3, 4] and len(d["crossings"]) == 12
    assert d["family"][0] == d["family"][1] == d["family"][2] != d["family"][3]
    assert all(np.linalg.norm(p[:2] - [x, y]) < 1e-3   # crossings land on the grid
               for (_, _, p, _), (x, y) in zip(
                   sorted(d["crossings"], key=lambda c: tuple(np.round(c[2][:2], 3))),
                   sorted((x, y) for x in (0.0, 1.0, 1.7, 3.0) for y in (0.0, 1.0, 2.0))))
    # Curve 6 (x = 3) cut short so it stops before reaching y = 2: one crossing
    # missing, reported with how far apart the two curves come.
    short = [c.copy() for c in curves]
    short[6] = short[6][short[6][:, 1] < 1.6]
    d = diagnose_network(short, 1e-3)
    assert not d["ok"] and [(u, v) for u, v, *_ in d["missing"]] in ([(2, 6)], [(6, 2)])
    gap = d["missing"][0][4]
    assert 0.3 < gap < 0.5, gap   # it ends at y = 1.6, the y = 2 curve is 0.4 away
    # A diagonal crosses curves of both directions: same-direction crossings.
    diag = curves + [np.stack([np.linspace(-0.3, 3.3, 120), np.linspace(-0.2, 2.2, 120),
                               np.zeros(120)], 1)]
    d = diagnose_network(diag, 1e-3)
    assert not d["ok"] and d["conflicts"]
    # A stray curve far away crosses nothing.
    stray = curves + [curves[0] + [0.0, 10.0, 0.0]]
    d = diagnose_network(stray, 1e-3)
    assert not d["ok"] and d["alone"] == [7] and d["family"][7] is None


def _wavy_chain(n=7, amp=0.0):
    """A Bezier chain along x with handles a third of the way, z = amp * wave."""
    x = np.arange(n, dtype=float)
    A = np.stack([x, 0 * x, amp * np.sin(2.6 * x)], 1)
    tangent = np.gradient(A, axis=0)
    return A, A - tangent / 3.0, A + tangent / 3.0


def test_pull_chain():
    from core.curveedit import chain_point, nearest_param, pull_chain
    A, L, R = _wavy_chain(7, 0.2)
    t, fixed = 3.2, (1.5, 5.0)
    delta = np.array([0.0, 0.3, 0.5])
    before = chain_point(A, L, R, [t] + list(fixed))
    A2, L2, R2 = pull_chain(A, L, R, t, delta, fixed)
    after = chain_point(A2, L2, R2, [t] + list(fixed))
    assert np.allclose(after[0] - before[0], delta, atol=1e-9)  # grabbed point follows exactly
    assert np.allclose(after[1:], before[1:], atol=1e-9)         # other crossings stay
    assert np.allclose(R2 - A2, R - A) and np.allclose(L2 - A2, L - A)  # handles ride along (G1 kept)
    assert np.allclose(A2[0], A[0], atol=1e-3) and np.allclose(A2[-1], A[-1], atol=1e-3)  # local
    # nearest_param finds a point's parameter back
    p = chain_point(A, L, R, [2.37])[0]
    assert abs(nearest_param(A, L, R, p) - 2.37) < 1e-3


def test_fair_chain():
    from core.curveedit import chain_point, fair_chain
    A, L, R = _wavy_chain(9, 0.15)
    fixed = (4.0,)
    keep = chain_point(A, L, R, list(fixed))
    A2, L2, R2 = fair_chain(A, L, R, fixed, strength=1.0)
    t = np.linspace(0, 8, 400)
    z0, z1 = chain_point(A, L, R, t)[:, 2], chain_point(A2, L2, R2, t)[:, 2]
    bend = lambda z: np.abs(np.diff(z, 2)).sum()
    print("fair chain: wiggle", bend(z0), "->", bend(z1))
    assert bend(z1) < 0.3 * bend(z0)                               # the waves are ironed out
    assert np.allclose(chain_point(A2, L2, R2, list(fixed)), keep, atol=1e-6)  # crossing kept
    assert np.allclose(A2[[0, -1]], A[[0, -1]]) and np.allclose(R2[0], R[0]) and np.allclose(L2[-1], L[-1])
    for i in range(1, 8):  # handles line up through their anchor (G1)
        c = np.cross(L2[i] - A2[i], R2[i] - A2[i])
        assert np.linalg.norm(c) < 1e-9
    A3, L3, R3 = fair_chain(A, L, R, fixed, strength=0.0)
    assert np.allclose(A3, A, atol=1e-6)


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


def test_soft_pull():
    from core.lock import lock_mask
    from core.softpull import soft_pull
    g = np.linspace(0, 1, 7)
    net = np.stack(list(np.meshgrid(g, g, indexing="ij")) + [np.zeros((7, 7))], -1)
    orders = (7, 7)
    mask = lock_mask((7, 7), [('U1', 'G2')])
    delta = np.array([0.0, 0.0, 0.4])
    new, gain = soft_pull(net, orders, mask, (0.4, 0.5), delta, 0.5)
    assert gain > 0.3
    assert np.allclose(new[mask], net[mask])  # pinned rows untouched
    p0 = nurbs.eval_points(net, orders, [0.4], [0.5])[0]
    p1 = nurbs.eval_points(new, orders, [0.4], [0.5])[0]
    assert np.allclose(p1 - p0, delta, atol=1e-9), p1 - p0  # the anchor follows the cursor
    far = np.linalg.norm(net[..., :2] - p0[:2], axis=-1) > 0.9
    assert np.allclose(new[far], net[far])  # outside the radius nothing moves
    # an anchor on the pinned edge cannot be pulled
    same, gain = soft_pull(net, orders, mask, (1.0, 0.5), delta, 0.5)
    assert gain == 0.0 and np.allclose(same, net)
    # multispan NURBS: same contract
    net2 = np.stack(list(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9), indexing="ij"))
                    + [np.zeros((9, 9))], -1)
    new2, _ = soft_pull(net2, (4, 4), np.zeros((9, 9), bool), (0.5, 0.5), delta, 0.4)
    q0 = nurbs.eval_points(net2, (4, 4), [0.5], [0.5])[0]
    q1 = nurbs.eval_points(new2, (4, 4), [0.5], [0.5])[0]
    assert np.allclose(q1 - q0, delta, atol=1e-9)


def test_line_pull():
    from core.lock import lock_mask
    from core.softpull import line_pull
    g = np.linspace(0, 1, 9)
    net = np.stack(list(np.meshgrid(g, g, indexing="ij")) + [np.zeros((9, 9))], -1)
    o = (4, 4)
    mask = lock_mask((9, 9), [('U1', 'G2')])
    delta = np.array([0.0, 0.0, 0.3])
    along = np.linspace(0, 1, 11)
    line0 = nurbs.eval_points(net, o, np.full(11, 0.5), along)
    # The whole iso-line u = 0.5 moves with the cursor, not just one point.
    new, gain = line_pull(net, o, mask, 'U', 0.5, 0.3, delta, 0.3)
    assert gain > 0.3
    line1 = nurbs.eval_points(new, o, np.full(11, 0.5), along)
    assert np.allclose(line1 - line0, delta, atol=1e-9)
    assert np.allclose(new[mask], net[mask])
    far = np.abs(net[..., 0] - 0.5) > 0.35
    assert np.allclose(new[far], net[far])  # outside the band nothing moves
    # Taper: the middle follows the cursor, the line's ends stay put.
    new, _ = line_pull(net, o, mask, 'U', 0.5, 0.5, delta, 0.3, taper=True)
    line1 = nurbs.eval_points(new, o, np.full(11, 0.5), along)
    assert np.allclose(line1[5] - line0[5], delta, atol=1e-9)
    assert np.allclose(line1[[0, -1]], line0[[0, -1]], atol=1e-9)
    # An open boundary edge can be pulled; the pinned seam edge cannot.
    new, gain = line_pull(net, o, mask, 'U', 0.0, 0.5, delta, 0.3)
    e0 = nurbs.eval_points(net, o, np.zeros(11), along)
    e1 = nurbs.eval_points(new, o, np.zeros(11), along)
    assert gain > 0.3 and np.allclose(e1 - e0, delta, atol=1e-9)
    same, gain = line_pull(net, o, mask, 'U', 1.0, 0.5, delta, 0.3)
    assert gain == 0.0 and np.allclose(same, net)
    # 'V' lines run along u.
    new, _ = line_pull(net, o, np.zeros((9, 9), bool), 'V', 0.25, 0.5, delta, 0.3)
    l0 = nurbs.eval_points(net, o, along, np.full(11, 0.25))
    l1 = nurbs.eval_points(new, o, along, np.full(11, 0.25))
    assert np.allclose(l1 - l0, delta, atol=1e-9)


def test_pull_profiles():
    from core.softpull import PROFILES, falloff, soft_pull
    s = np.array([0.0, 0.5, 1.0])
    for p in PROFILES:
        w = falloff(s, p)
        assert w[0] == 1.0 and w[2] == 0.0, p
    mid = {p: falloff(np.array([0.5]), p)[0] for p in PROFILES}
    expect = {'CONSTANT': 1.0, 'SPHERE': 0.75 ** 0.5, 'LINEAR': 0.5,
              'SMOOTH': 0.75 ** 3, 'SHARP': 0.25}
    assert all(abs(mid[p] - expect[p]) < 1e-12 for p in PROFILES), mid
    g = np.linspace(0, 1, 9)
    net = np.stack(list(np.meshgrid(g, g, indexing="ij")) + [np.zeros((9, 9))], -1)
    delta = np.array([0.0, 0.0, 0.2])
    free = np.zeros((9, 9), bool)
    # Constant lifts everything inside the radius by exactly delta, nothing else.
    new, gain = soft_pull(net, (4, 4), free, (0.5, 0.5), delta, 0.3, 'CONSTANT')
    inside = np.linalg.norm(net[..., :2] - 0.5, axis=-1) < 0.25
    outside = np.linalg.norm(net[..., :2] - 0.5, axis=-1) > 0.35
    assert gain > 0.99
    assert np.allclose(new[inside] - net[inside], delta)
    assert np.allclose(new[outside], net[outside])
    # Every profile still makes the anchor follow the cursor.
    for p in PROFILES:
        new, _ = soft_pull(net, (4, 4), free, (0.5, 0.5), delta, 0.3, p)
        a0 = nurbs.eval_points(net, (4, 4), [0.5], [0.5])[0]
        a1 = nurbs.eval_points(new, (4, 4), [0.5], [0.5])[0]
        assert np.allclose(a1 - a0, delta, atol=1e-9), p


def test_point_pull():
    from core.lock import lock_mask
    from core.softpull import grid_weights, point_pull
    g = np.linspace(0, 1, 9)
    net = np.stack(list(np.meshgrid(g, g, indexing="ij")) + [np.zeros((9, 9))], -1)
    mask = lock_mask((9, 9), [('U1', 'G2')])
    delta = np.array([0.0, 0.0, 0.3])
    new, gain = point_pull(net, mask, 3, 4, delta, 2.0, 1.0)
    assert gain == 1.0
    assert np.allclose(new[3, 4] - net[3, 4], delta)  # the grabbed point follows exactly
    moved = np.any(new != net, axis=-1)
    # 2 rows along u, 1 along v: an ellipse in the grid, not a circle
    assert moved[2, 4] and moved[4, 4] and not moved[1, 4] and not moved[5, 4]
    assert not moved[3, 3] and not moved[3, 5]
    assert np.allclose(new[mask], net[mask])
    # rows below 1 move just the point itself
    new, _ = point_pull(net, mask, 3, 4, delta, 0.5, 0.5)
    assert np.argwhere(np.any(new != net, axis=-1)).tolist() == [[3, 4]]
    # a pinned point cannot be grabbed
    same, gain = point_pull(net, mask, 8, 4, delta, 2.0, 2.0)
    assert gain == 0.0 and np.allclose(same, net)
    # Constant moves the whole ellipse rigidly
    w = grid_weights((9, 9), np.zeros((9, 9), bool), 4, 4, 2.0, 2.0, 'CONSTANT')
    assert w[4, 4] == w[3, 4] == w[4, 3] == w[3, 3] == 1.0 and w[2, 4] == 0.0


def test_extend_length_handle():
    from core.extend import length_handle
    P = _poly_patch(0, 0)
    for edge in EDGES:
        base, direction, scale = length_handle(P, edge, (6, 6))
        h = 0.4
        Q, o = extend_net(P, edge, h, 'G1', orders=(6, 6))
        # The handle sits mid-edge on the source, and the extension's far
        # edge middle is length * scale along the handle direction (G1).
        mids = [nurbs.eval_points(Q, o, [a], [b])[0]
                for a, b in ((0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5))]
        tip = base + h * scale * direction
        assert any(np.allclose(m, tip, atol=1e-9) for m in mids), edge
        assert any(np.allclose(m, base, atol=1e-9) for m in mids), edge
        assert abs(np.linalg.norm(direction) - 1) < 1e-12


def test_offset_distance_handle():
    from core.offset import distance_handle, offset_net
    P = _poly_patch(0, 0)
    base, normal = distance_handle(P, (6, 6))
    assert abs(np.linalg.norm(normal) - 1) < 1e-12 and normal[2] > 0
    for d in (0.15, -0.1):
        Q, o, err = offset_net(P, (6, 6), d)
        mid = nurbs.eval_points(Q, o, [0.5], [0.5])[0]
        assert np.linalg.norm(mid - (base + d * normal)) < max(2 * err, 1e-6), (d, err)


def test_t_junctions():
    from core.meshing import t_junctions
    # A coarse edge 0-1 (along x, on a slight arc) meets a fine side whose
    # middle vertex 2 sits on the curve, a little off the coarse chord.
    P = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.02, 0],   # 0, 1: coarse; 2: fine middle
                  [0.25, 0.0, 0], [0.5, 0.3, 0]], float)  # 3: endpoint-ish; 4: far away
    E = np.array([[0, 1]])
    hits = t_junctions(P, E, [0, 1, 2, 3, 4], abs_tol=1e-4)
    assert [v for _, v in hits[0]] == [3, 2]  # sorted along the edge, far vertex ignored
    assert abs(hits[0][1][0] - 0.5) < 1e-12
    # Endpoints never hit their own edge; tighter rel_tol drops the off-chord one.
    hits = t_junctions(P, E, [0, 1, 2], abs_tol=1e-4, rel_tol=0.01)
    assert hits == {}


def test_curve_detail():
    from core.meshing import segment_counts
    # a long flat patch and a small quarter-round strip: by length alone the strip gets 2 facets
    g = np.linspace(0, 1, 6)
    flat = (np.stack(np.meshgrid(4 * g, g, [0.0], indexing="ij"), -1)[..., 0, :], (6, 6))
    ang = np.linspace(0, np.pi / 2, 7)
    arc = np.stack([0.08 * np.sin(ang), 0.08 * (1 - np.cos(ang))], -1)           # r = 0.08, 90 degrees
    strip = np.stack([np.stack([[x + 10.0, 0.0, 0.0 + 0.0] for x in arc[:, 0]], 0)] * 2, 1)
    strip[:, 1, 1] = 0.5
    strip[:, :, 2] = arc[:, 1][:, None]
    plain = segment_counts([flat, (strip, (6, 2))], 24)
    assert plain[1][0] == 2
    fine = segment_counts([flat, (strip, (6, 2))], 24, curve_detail=6)
    assert 5 <= fine[1][0] <= 8, fine
    assert fine[0] == plain[0]                                  # flat patches are untouched


def test_segment_counts():
    from core.meshing import segment_counts

    def rect(x0, x1, y0, y1):
        g = np.linspace(0, 1, 3)
        X = x0 + (x1 - x0) * g[:, None] + 0 * g
        Y = y0 + 0 * g[:, None] + (y1 - y0) * g
        return np.stack([X, Y, np.zeros((3, 3))], -1), (3, 3)

    A = rect(0, 2, 0, 1)       # big: 2 x 1
    B = rect(2, 3, 0, 1)       # shares A's x = 2 edge (runs along v in both)
    C = rect(5, 5.5, 0, 0.5)   # on its own, small
    (au, av), (bu, bv), (cu, cv) = segment_counts([A, B, C], 8)
    assert (au, av) == (8, 4)  # longest side gets the resolution, the rest the same density
    assert bu == 4 and bv == av  # equal counts along the shared edge
    assert (cu, cv) == (2, 2)
    # A shared edge takes the finer side's count: the trapezoid T is 3 tall
    # on its far side but meets the square S along a 1-long edge at x = 1.
    g = np.linspace(0, 1, 3)
    X = g[:, None] + 0 * g
    Y = (-1 + g[:, None]) + (3 - 2 * g[:, None]) * g     # v-lines from 3 long to 1 long
    T = (np.stack([X, Y, np.zeros((3, 3))], -1), (3, 3))
    S = rect(1, 2, 0, 1)
    (tu, tv), (su, sv) = segment_counts([T, S], 6)
    assert tv > 3 and sv == tv  # the square follows the trapezoid's finer count


def test_refine_keeps_shape():
    from core.fair import refine_curve, refine_net
    for n, k in ((4, 4), (6, 4), (8, 3), (9, 4)):
        P = rng.normal(size=(n, 3))
        Q = refine_curve(P, k)
        spans = n - k + 1
        assert len(Q) == n + spans
        t = np.linspace(0, 1, 60)
        assert np.abs(nurbs.eval_curve(P, k, t) - nurbs.eval_curve(Q, k, t)).max() < 1e-9
    net = rng.normal(size=(6, 5, 3))
    new, o = refine_net(net, (4, 5), 'BOTH')
    assert new.shape == (6 + 3, 5 + 1, 3)
    g = np.linspace(0, 1, 11)
    assert np.abs(nurbs.eval_surface(net, (4, 5), g, g) - nurbs.eval_surface(new, o, g, g)).max() < 1e-9


def test_insert_line():
    from core.fair import insert_line
    g = np.linspace(0, 1, 11)
    # single span (Bezier) along U: exact, order goes up
    net = rng.normal(size=(4, 6, 3))
    new, o, exact = insert_line(net, (4, 4), 0)
    assert exact and new.shape == (5, 6, 3) and o == (5, 4)
    assert np.abs(nurbs.eval_surface(net, (4, 4), g, g) - nurbs.eval_surface(new, o, g, g)).max() < 1e-9
    # multi-span along V: one more point, ends fixed, shape moves only a little on a smooth net
    xs = np.linspace(0, 1, 7)
    smooth = np.stack(list(np.meshgrid(xs, xs, indexing="ij")) + [0.3 * np.sin(2 * np.meshgrid(xs, xs, indexing="ij")[1])], -1)
    new, o, exact = insert_line(smooth, (4, 4), 1)
    assert not exact and new.shape == (7, 8, 3) and o == (4, 4)
    assert np.allclose(new[:, 0], smooth[:, 0]) and np.allclose(new[:, -1], smooth[:, -1])
    dev = np.abs(nurbs.eval_surface(smooth, (4, 4), g, g) - nurbs.eval_surface(new, o, g, g)).max()
    assert dev < 0.02, dev


def test_fair_net():
    from core.fair import fair_net
    g = np.linspace(0, 1, 8)
    flat = np.stack(list(np.meshgrid(g, g, indexing="ij")) + [np.zeros((8, 8))], -1)
    bumpy = flat.copy()
    bumpy[3, 4, 2] += 0.5
    bumpy[4, 2, 2] -= 0.3
    fixed = np.zeros((8, 8), bool)
    fixed[[0, -1]] = fixed[:, [0, -1]] = True
    out = fair_net(bumpy, fixed)
    assert np.allclose(out[fixed], bumpy[fixed])
    assert np.abs(out[..., 2]).max() < 0.02  # the bumps are ironed out
    half = fair_net(bumpy, fixed, 0.5)
    assert np.abs(half[..., 2]).max() < np.abs(bumpy[..., 2]).max()
    assert np.allclose(fair_net(bumpy, fixed, 0.0), bumpy)
    # nothing fixed: still well defined, and a flat net stays flat
    assert np.abs(fair_net(flat, np.zeros((8, 8), bool)) - flat).max() < 1e-6


def test_fillet_rounds_right_angle_corner():
    from core.fillet import fillet_net
    n = 6
    g = np.linspace(0.0, 1.0, n)
    # Patch A: the y = 0 wall, x in [0, 1], z in [0, 1]; edge U1 is x = 1.
    A = np.zeros((n, n, 3))
    A[..., 0] = g[:, None]
    A[..., 2] = g[None, :]
    # Patch B: the x = 1 wall, y in [0, 1], z in [0, 1]; edge U0 is y = 0,
    # coincident with A's x = 1 edge.
    B = np.zeros((n, n, 3))
    B[..., 0] = 1.0
    B[..., 1] = g[:, None]
    B[..., 2] = g[None, :]

    radius = 0.2
    net, orders, info = fillet_net(A, (6, 6), 'U1', B, (6, 6), 'U0', radius)
    assert info["fit_error"] < 1e-5  # degree 5 cannot reproduce a circular arc exactly
    assert abs(info["max_angle"] - 90.0) < 1e-6
    assert np.allclose(info["reach"], radius, atol=1e-9)  # tan(45 deg) == 1

    gg = np.linspace(0.0, 1.0, 25)
    surf = nurbs.eval_surface(net, orders, gg, gg)
    # The crease runs along z at x = 1, y = 0; A lies at x < 1, B at y > 0.
    # The ball sits between the walls, so every fillet point is `radius` from
    # the line (1 - radius, radius, z) - the circle's axis.
    axis = np.array([1.0 - radius, radius, 0.0])
    d = surf.copy()
    d[..., 2] = 0.0
    dist = np.linalg.norm(d - axis[None, None, :], axis=2)
    assert np.abs(dist - radius).max() < 1e-5, np.abs(dist - radius).max()
    # Each fillet edge lies ON its source wall, inset `radius` from the crease
    # (that is where the walls get trimmed back to).
    ends = [surf[:, 0], surf[:, -1]]
    on_a = [e for e in ends if np.abs(e[:, 1]).max() < 1e-9]
    on_b = [e for e in ends if np.abs(e[:, 0] - 1.0).max() < 1e-9]
    assert len(on_a) == 1 and len(on_b) == 1
    assert np.abs(on_a[0][:, 0] - (1.0 - radius)).max() < 1e-9
    assert np.abs(on_b[0][:, 1] - radius).max() < 1e-9
    # The arc bows toward the crease, not away from it.
    mid = surf[:, len(gg) // 2]
    assert np.all(mid[:, 0] > 1.0 - radius) and np.all(mid[:, 1] < radius)


def test_trimmed_fillet():
    from core.fillet import multi_fillet, trim_net

    def trimmed_fillet(Pa, oa, ea, Pb, ob, eb, r):
        cut, bridges, infos = multi_fillet({"a": (Pa, oa), "b": (Pb, ob)},
                                           [("a", ea, "b", eb)], r)
        return cut["a"], cut["b"], bridges[0][:2], infos[0]

    n = 6
    g = np.linspace(0.0, 1.0, n)
    A = np.zeros((n, n, 3))
    A[..., 0] = g[:, None]
    A[..., 2] = g[None, :]
    B = np.zeros((n, n, 3))
    B[..., 0] = 1.0
    B[..., 1] = g[:, None]
    B[..., 2] = g[None, :]

    # trim_net measures along the surface and keeps the far side
    T, ot, _, err = trim_net(A, (6, 6), 'U1', 0.3)
    assert err < 1e-9 and np.abs(T[..., 0].max() - 0.7) < 1e-6
    T, ot, _, err = trim_net(B, (6, 6), 'U0', 0.3)
    assert np.abs(T[..., 1].min() - 0.3) < 1e-6

    radius = 0.2
    (TA, oa), (TB, ob), (net, orders), info = trimmed_fillet(
        A, (6, 6), 'U1', B, (6, 6), 'U0', radius)
    assert abs(info["trim"] - radius) < 1e-9  # right angle: reach == radius
    assert np.abs(TA[..., 0].max() - (1 - radius)) < 1e-6
    assert np.abs(TB[..., 1].min() - radius) < 1e-6
    for P, o in ((TA, oa), (TB, ob)):
        sh = find_shared_edge(P, net, 1e-6, o, orders)
        assert sh is not None
        assert grade(measure(P, net, *sh[:3], o, orders)) == 'G2'
    # The bridge rounds the corner: its middle is near the r = 0.2 arc
    # around (0.8, 0.2), well inside the old sharp corner.
    mid = nurbs.eval_surface(net, orders, np.linspace(0, 1, 7), [0.5])[:, 0]
    r = np.linalg.norm(mid[:, :2] - [1 - radius, radius], axis=1)
    assert np.abs(r - radius).max() < 0.15 * radius, r
    try:
        trimmed_fillet(A, (6, 6), 'U1', B, (6, 6), 'U0', 2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a radius larger than the patch must be refused")


def test_multi_fillet_corner():
    from core.fillet import multi_fillet
    g = np.linspace(0, 1, 3)
    G0, G1 = g[:, None] + 0 * g, 0 * g[:, None] + g
    F = np.stack([G0, G1, 0 * G0], -1)            # floor z = 0; U1 is x = 1, V1 is y = 1
    W1 = np.stack([1 + 0 * G0, G1, G0], -1)       # wall x = 1; U0 is z = 0
    W2 = np.stack([G1, 1 + 0 * G0, G0], -1)       # wall y = 1; U0 is z = 0
    o = (3, 3)
    patches = {"F": (F, o), "W1": (W1, o), "W2": (W2, o)}
    r = 0.2
    trimmed, bridges, infos = multi_fillet(
        patches, [("F", "U1", "W1", "U0"), ("F", "V1", "W2", "U0")], r)
    TF = trimmed["F"][0]
    # The floor is cut back on both edges; each wall on its bottom edge.
    assert abs(TF[..., 0].max() - (1 - r)) < 1e-6 and abs(TF[..., 1].max() - (1 - r)) < 1e-6
    assert abs(trimmed["W1"][0][..., 2].min() - r) < 1e-6
    assert abs(trimmed["W2"][0][..., 2].min() - r) < 1e-6
    assert all(abs(i["trim"] - r) < 1e-9 for i in infos)
    # Each bridge joins its two trimmed edges smoothly.
    for (net, bo, _), (a, _, b, _) in zip(bridges, [("F", 0, "W1", 0), ("F", 0, "W2", 0)]):
        for name in (a, b):
            P, po = trimmed[name]
            sh = find_shared_edge(P, net, 1e-6, po, bo)
            assert sh is not None, name
            assert grade(measure(P, net, *sh[:3], po, bo)) == 'G2', name
    try:
        multi_fillet(patches, [("F", "U1", "W1", "U0"), ("F", "U1", "W2", "U0")], r)
    except ValueError:
        pass
    else:
        raise AssertionError("the same edge picked twice must be refused")


def test_fillet_corner_patch():
    from core.fill import fill_patches, side_from_patch
    from core.fillet import corner_loops, multi_fillet
    g = np.linspace(0, 1, 3)
    G0, G1 = g[:, None] + 0 * g, 0 * g[:, None] + g
    F = np.stack([G0, G1, 0 * G0], -1)            # floor z = 0 (u = x, v = y)
    W1 = np.stack([1 + 0 * G0, G1, G0], -1)       # wall x = 1 (u = z, v = y)
    W2 = np.stack([G1, 1 + 0 * G0, G0], -1)       # wall y = 1 (u = z, v = x)
    o = (3, 3)
    patches = {"F": (F, o), "W1": (W1, o), "W2": (W2, o)}
    # All three edges of the box corner (1, 1, 0) rounded.
    seams = [("F", "U1", "W1", "U0"), ("F", "V1", "W2", "U0"), ("W1", "V1", "W2", "V1")]
    _, bridges, _ = multi_fillet(patches, seams, 0.2)
    nets = [(n, bo) for n, bo, _ in bridges]
    loops = corner_loops(nets)
    assert len(loops) == 1 and len(loops[0]) == 3  # one triangular hole
    assert sorted(i for i, _ in loops[0]) == [0, 1, 2]
    sides = [side_from_patch(*nets[i], e) for i, e in loops[0]]
    corner = fill_patches(sides, 'G1', tol=1e-6)
    for i, e in loops[0]:  # the corner patch closes the hole against every bridge
        touched = 0
        for Q, oq in corner:
            ang, hits = _boundary_angle(nets[i][0], nets[i][1], e, Q, oq)
            touched += hits
            assert ang < 2.0, (i, e, ang)
        assert touched > 20, (i, e)
    # One fillet alone ends on open boundaries: nothing to fill.
    _, one, _ = multi_fillet(patches, seams[:1], 0.2)
    assert corner_loops([(n, bo) for n, bo, _ in one]) == []


def test_curve_comb():
    from core.curvature import bezier_comb, polyline_comb
    R, c = 2.0, 0.5522847498  # the usual cubic quarter circle
    quarter = [(R, 0, 0), (R, R * c, 0), (R * c, R, 0), (0, R, 0)]
    (pts, k), = bezier_comb([quarter])
    assert np.abs(np.linalg.norm(k, axis=1) * R - 1).max() < 0.03   # ~1/R, a cubic is not exact
    # it points at the centre
    assert all(np.dot(kv, -p) > 0 for kv, p in zip(k, pts))
    # a straight segment has none; a jump at a join shows as two teeth there
    line = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    (p0, k0), (p1, k1) = bezier_comb([line, [(3, 0, 0), (4, 0, 0), (5, 1, 0), (6, 3, 0)]])
    assert np.abs(k0).max() < 1e-12 and np.linalg.norm(k1[0]) > 0.1
    assert np.allclose(p0[-1], p1[0])
    # polyline: points on a circle give 1 / R towards the centre
    a = np.linspace(0, np.pi, 50)
    circle = np.stack([R * np.cos(a), R * np.sin(a), 0 * a], 1)
    P, kp = polyline_comb(circle)
    assert np.abs(np.linalg.norm(kp, axis=1) * R - 1).max() < 1e-9
    assert np.abs(polyline_comb(np.stack([a, 0 * a, 0 * a], 1))[1]).max() == 0.0


def test_curvature_paraboloid():
    from core.bezier import bernstein
    from core.curvature import color_ramp, comb, surface_curvature
    a, b = 0.4, -0.25  # z = a x^2 + b y^2 on [-1, 1]^2: a saddle
    t = np.linspace(0, 1, 15)
    pinv = np.linalg.pinv(bernstein(3, t))
    X, Y = -1 + 2 * t[:, None] + 0 * t, -1 + 0 * t[:, None] + 2 * t
    net = np.stack([pinv @ C @ pinv.T for C in (X, Y, a * X * X + b * Y * Y)], -1)
    o = (4, 4)
    # Through the middle, the x-curve is z = a x^2: curvature 2a at x = 0.
    pts, k = comb(net, o, 'U', count=3, samples=21)
    centre = k[1, 10]
    assert np.allclose(pts[1, 10], [0, 0, 0], atol=1e-9)
    assert np.allclose(centre, [0, 0, 2 * a], atol=1e-6), centre
    pts, k = comb(net, o, 'V', count=3, samples=21)
    assert np.allclose(k[1, 10], [0, 0, 2 * b], atol=1e-6), k[1, 10]
    # At the origin K = 4ab and H = a + b (normal +z here).
    P, N, K, H = surface_curvature(net, o, samples=21)
    assert np.allclose(P[10, 10], 0, atol=1e-9) and N[10, 10, 2] > 0.99
    assert abs(K[10, 10] - 4 * a * b) < 1e-6 and abs(H[10, 10] - (a + b)) < 1e-6
    c = color_ramp([-1.0, 0.0, 1.0], 1.0)
    assert c[0, 2] > c[0, 0] and c[1, 1] > c[1, 0] and c[2, 0] > c[2, 2]


def test_fit_polyline_keeps_end_tangents():
    # An S-shaped curve: a plain least-squares fit lets the ends tilt; the
    # tangent-keeping fit leaves each end along the polyline's own direction.
    t = np.linspace(0, 1, 48)
    pts = np.stack([t, 0.25 * np.sin(2 * np.pi * t), 0.4 * t * t], -1)
    P, err = nurbs.fit_polyline_tangent(pts, 9, 6)
    assert err < 0.02, err
    d0 = -3 * pts[0] + 4 * pts[1] - pts[2]
    d1 = 3 * pts[-1] - 4 * pts[-2] + pts[-3]
    for want, got in ((d0, P[1] - P[0]), (d1, P[-1] - P[-2])):
        cos = np.dot(want, got) / (np.linalg.norm(want) * np.linalg.norm(got))
        assert cos > 0.9999, cos


def test_two_fillets_meet_in_a_mitre():
    from core.fillet import multi_fillet
    g = np.linspace(0, 1, 3)
    G0, G1 = g[:, None] + 0 * g, 0 * g[:, None] + g
    F = np.stack([G0, G1, 0 * G0], -1)
    W1 = np.stack([1 + 0 * G0, G1, G0], -1)
    W2 = np.stack([G1, 1 + 0 * G0, G0], -1)
    o = (3, 3)
    patches = {"F": (F, o), "W1": (W1, o), "W2": (W2, o)}
    trimmed, bridges, _ = multi_fillet(
        patches, [("F", "U1", "W1", "U0"), ("F", "V1", "W2", "U0")], 0.2)
    ends = [net[-1] for net, _, _ in bridges]
    # the two bridge ends are the same curve, lying in the plane x = y
    assert np.allclose(ends[0], ends[1][::-1], atol=1e-9)
    assert np.allclose(ends[0][:, 0], ends[0][:, 1], atol=1e-9)
    # and the bridges are still G2 with the patches they join
    for (net, bo, _), names in zip(bridges, (("F", "W1"), ("F", "W2"))):
        for name in names:
            P, po = trimmed[name]
            sh = find_shared_edge(P, net, 1e-6, po, bo)
            assert grade(measure(P, net, *sh[:3], po, bo)) == 'G2', name


def test_flange_wall():
    from core.flange import flange_net
    g = np.linspace(0, 1, 4)
    U, V = g[:, None] + 0 * g, 0 * g[:, None] + g
    P = np.stack([U, V, 0 * U], -1)                    # flat patch, normal +z
    for edge, at in (('U0', ('x', 0.0)), ('U1', ('x', 1.0)), ('V0', ('y', 0.0)), ('V1', ('y', 1.0))):
        net, orders = flange_net(P, edge, 0.5, 90.0)
        assert orders == (4, 2) and net.shape == (4, 2, 3)
        # the wall starts on the edge and drops 0.5 along -z (the back of the patch)
        assert np.allclose(net[:, 1] - net[:, 0], [0, 0, -0.5], atol=1e-9), edge
        axis = 0 if at[0] == 'x' else 1
        assert np.allclose(net[..., axis], at[1]), edge
    net, _ = flange_net(P, 'V1', 0.5, 90.0, flip=True)
    assert np.allclose(net[:, 1] - net[:, 0], [0, 0, 0.5], atol=1e-9)
    net, _ = flange_net(P, 'V1', 1.0, 45.0)            # leans halfway: out and down
    assert np.allclose(net[0, 1] - net[0, 0], [0, np.sqrt(0.5), -np.sqrt(0.5)], atol=1e-9)
    for bad in (0.0, 180.0, -10.0):
        try:
            flange_net(P, 'V1', 0.5, bad)
        except ValueError:
            continue
        raise AssertionError("angle %s must be refused" % bad)


def test_sweep_net():
    from core.sweep import sweep_net
    g = np.linspace(0, 1, 5)
    # profile: the V1 edge of a patch, a bump along x at y = 0 (u runs along x)
    prof_patch = np.stack([np.stack([g, 0 * g, 0.3 * np.sin(np.pi * g)], -1)] * 2, 1)   # (5, 2, 3)
    prof_patch[:, 1, 1] = -1.0                       # second column: the patch lies at y = -1
    # rail: a V0 edge starting at the profile's end (1, 0, 0) and bending up
    rail_patch = np.stack([np.stack([1 + 0.4 * g, g, 0.5 * g ** 2], -1)] * 2, 0)         # (2, 5, 3): U edges
    rail_patch[1] += [0, 0, 1.0]
    net, orders = sweep_net(prof_patch, (5, 2), 'V0', rail_patch, (2, 5), 'U0')
    assert orders == (5, 5) and net.shape == (5, 5, 3)
    prof = prof_patch[:, 0]
    rail = rail_patch[0]
    # exact: S(u, v) = profile(u) + rail(v) - rail(0), here with the rail starting at x = 1
    for i in range(5):
        for j in range(5):
            assert np.allclose(net[i, j], prof[i] + rail[j] - rail[0]), (i, j)
    # the surface really is the sum of the two curves
    u, v = np.array([0.3, 0.7, 0.5]), np.array([0.2, 0.9, 0.5])
    S = nurbs.eval_points(net, orders, u, v)
    pc = nurbs.eval_points(np.stack([prof, prof], 1), (5, 2), u, 0 * u)
    rc = nurbs.eval_points(np.stack([rail, rail]), (2, 5), 0 * v, v)
    r0 = nurbs.eval_points(np.stack([rail, rail]), (2, 5), np.zeros(1), np.zeros(1))[0]
    assert np.allclose(S, pc + rc - r0, atol=1e-9)
    # flip walks the rail from its other end
    flipped, _ = sweep_net(prof_patch, (5, 2), 'V0', rail_patch, (2, 5), 'U0', flip=True)
    assert np.allclose(flipped[0], prof[0] + (rail[::-1] - rail[-1]))
    # straight: the profile's far end runs straight out of the patch (+y), the near end follows the rail
    for fl in (False, True):
        sn, so = sweep_net(prof_patch, (5, 2), 'V0', rail_patch, (2, 5), 'U0', flip=fl, straight=True)
        assert so == (5, 5)
        assert np.allclose(sn[:, 0], prof)                                     # the profile edge is kept
        rl = rail if not fl else rail[::-1]
        near = sn[0, :]                                                        # profile end at x = 0 is the far end
        chord = np.linalg.norm(rl[-1] - rl[0])
        vs = np.linspace(0, 1, 9)
        far_pts = nurbs.eval_points(np.stack([sn[0], sn[0]], 0), (2, 5), 0 * vs, vs)
        assert np.allclose(far_pts[:, [0, 2]], prof[0][[0, 2]], atol=1e-9)    # straight along y only
        assert np.all(np.diff(far_pts[:, 1]) > 0)
        reach = (rl[-1] - rl[0])[1]                                            # the rail's progress along +y; backwards: its chord
        assert abs(sn[0, -1, 1] - prof[0][1] - (reach if reach > 0 else chord)) < 1e-9
        assert np.allclose(sn[-1], prof[-1] + rl - rl[0])                      # the rail end follows the rail
    ln, _ = sweep_net(prof_patch, (5, 2), 'V0', rail_patch, (2, 5), 'U0', straight=True, length=0.7)
    assert abs(ln[0, -1, 1] - prof[0][1] - 0.7) < 1e-9
    # target: a plane square to y, or a polyline, stops the straight end where it arrives
    tn, _ = sweep_net(prof_patch, (5, 2), 'V0', rail_patch, (2, 5), 'U0', straight=True,
                      target=('plane', 1, 0.8))
    assert abs(tn[0, -1, 1] - 0.8) < 1e-9
    wall = np.array([[-1.0, 0.45, -1.0], [-1.0, 0.45, 1.0], [2.0, 0.45, 1.0]])     # y = 0.45
    tn, _ = sweep_net(prof_patch, (5, 2), 'V0', rail_patch, (2, 5), 'U0', straight=True,
                      target=('line', wall))
    assert abs(tn[0, -1, 1] - 0.45) < 1e-9
    from core.sweep import straight_run
    o, d = np.zeros(3), np.array([0.0, 1.0, 0.0])
    assert abs(straight_run(o, d, ('line', np.array([[1.0, 2.0, 0.5], [1.0, 3.0, 0.5]]))) - 2.0) < 1e-9
    for bad in (('plane', 0, 1.0), ('plane', 1, -1.0), ('line', np.array([[0.0, -1.0, 0.0], [1.0, -2.0, 0.0]]))):
        try:
            straight_run(o, d, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("an unreachable target must be refused: %s" % (bad,))
    # a rail of zero length is refused
    try:
        sweep_net(prof_patch, (5, 2), 'V0', np.zeros((2, 5, 3)), (2, 5), 'U0')
    except ValueError:
        pass
    else:
        raise AssertionError("a degenerate rail must be refused")


def test_facing_flips():
    from core.orient import facing_flips, flip_net
    g = np.linspace(0, 1, 4)
    X, Y = np.meshgrid(g, g, indexing="ij")

    def flat(x0, y0, turned=False):
        net = np.stack([X + x0, Y + y0, 0 * X], -1)         # normal +z
        return flip_net(net) if turned else net

    def normal_z(net):
        du, dv = net[-1, 0] - net[0, 0], net[0, -1] - net[0, 0]
        return np.cross(du, dv)[2]

    # a row of three patches, the middle one turned over; and one above them, turned too
    patches = [(flat(0, 0), (4, 4)), (flat(1, 0, True), (4, 4)), (flat(2, 0), (4, 4)),
               (flat(1, 1, True), (4, 4))]
    assert normal_z(patches[1][0]) < 0
    assert facing_flips(patches) == [False, True, False, True]
    # with the root turned instead, everything follows it
    assert facing_flips(patches, root=1) == [True, False, True, False]
    # patches that cannot be turned: the group takes the facing that turns none of them
    assert facing_flips(patches, pinned=[0, 2]) == [False, True, False, True]
    assert facing_flips(patches, pinned=[1], root=0) == [True, False, True, False]
    # a lone far-away patch is its own group and is left alone
    far = patches + [(flat(50, 50, True), (4, 4))]
    assert facing_flips(far)[-1] is False


def test_part_flips():
    from core.orient import flip_net, part_flips
    g = np.linspace(0, 1, 4)
    X, Y = np.meshgrid(g, g, indexing="ij")
    ref = (np.stack([X, Y, 0 * X], -1), (4, 4))
    beside = np.stack([X + 1, Y, 0 * X], -1)
    assert part_flips([ref], [(beside, (4, 4)), (flip_net(beside), (4, 4))]) == [False, True]
    # the parts of one hole face like each other when they touch no ref
    far = np.stack([X + 9, Y, 0 * X], -1)
    far2 = flip_net(np.stack([X + 10, Y, 0 * X], -1))
    assert part_flips([], [(far, (4, 4)), (far2, (4, 4))]) == [False, True]
    assert part_flips([ref], []) == []


def test_panel_gap():
    from core.gap import edge_normals, find_gap_pairs, measure_gap
    t = np.linspace(0, 1, 20)
    A = np.stack([t, 0 * t, 0 * t], -1)
    B = np.stack([t, 0.02 + 0 * t, 0.005 + 0 * t], -1)      # 20 mm across, 5 mm proud
    m = measure_gap(A, B, np.tile([0.0, 0.0, 1.0], (20, 1)))
    assert np.allclose(m["gap"], 0.02) and np.allclose(m["step"], 0.005)
    assert np.allclose(m["dist"], np.hypot(0.02, 0.005))
    # a tapering joint: the gap grows from 10 to 30 mm
    C = np.stack([t, 0.01 + 0.02 * t, 0 * t], -1)
    g = measure_gap(A, C)["gap"]
    assert abs(g.min() - 0.01) < 5e-3 and abs(g.max() - 0.03) < 5e-3 and g[0] < g[-1]
    pairs = find_gap_pairs({("a", "V1"): A, ("b", "V0"): B, ("c", "U0"): A + [0, 5, 0]}, 0.05)
    assert pairs == [(("a", "V1"), ("b", "V0"))], pairs      # the far-away edge is skipped
    # normals of a flat patch point along z
    g4 = np.linspace(0, 1, 4)
    U, V = g4[:, None] + 0 * g4, 0 * g4[:, None] + g4
    N = edge_normals(np.stack([U, V, 0 * U], -1), (4, 4), 'V1', 8)
    assert np.allclose(np.abs(N[:, 2]), 1.0)


def test_even_gap():
    from core.gap import even_gap_net, measure_gap, edge_normals
    g = np.linspace(0, 1, 4)
    U, V = np.meshgrid(g, g, indexing="ij")
    A = np.stack([U, V, 0.1 * np.sin(np.pi * U)], -1)            # a gently curved panel, y in 0..1
    t = np.linspace(0, 1, 40)
    # the neighbour's edge: A's own edge moved out 20-50 mm (tapering) and 4 mm proud
    neighbour = edge_line(A, 'V1', 40) + np.stack([0 * t, 0.02 + 0.03 * t, 0 * t + 0.004], -1)
    for edge in ['V1']:
        net, orders, (before, after) = even_gap_net(A, (4, 4), edge, neighbour, 0.01)
        assert before > 0.02 and after < 1e-3, (before, after)
        m = measure_gap(edge_line(net, 'V1'), neighbour, edge_normals(net, orders, 'V1', 30))
        assert np.abs(m["gap"] - 0.01).max() < 1.5e-3 and m["step"].max() < 1.5e-3, m
        assert np.allclose(net[:, 0], A[:, 0])                   # the opposite edge stays put
    # not flush: the step is kept
    net2, _, _ = even_gap_net(A, (4, 4), 'V1', neighbour, 0.01, flush=False)
    m2 = measure_gap(edge_line(net2, 'V1'), neighbour, edge_normals(net2, (4, 4), 'V1', 30))
    assert m2["step"].mean() > 0.003


def test_even_gap_partial():
    """A door edge twice as long as the fender edge it meets: only the facing half moves."""
    from core.gap import even_gap_net, find_gap_pairs
    g = np.linspace(0, 1, 6)
    U, V = np.meshgrid(g, g, indexing="ij")
    door = np.stack([0 * U + 1.0, -0.02 - 0.02 * V - 1.0 * U, -0.5 + 1.0 * V], -1)   # front edge u=0
    t = np.linspace(0, 1, 30)
    fender_edge = np.stack([0 * t + 1.0, 0 * t, -0.5 + 0.5 * t], -1)               # lower half only
    door_front = edge_line(door, 'U0', 30)
    assert find_gap_pairs({("d", "U0"): door_front, ("f", "U0"): fender_edge}, 0.07, coverage=0.6) == []
    assert find_gap_pairs({("d", "U0"): door_front, ("f", "U0"): fender_edge}, 0.07, coverage=0.6,
                          partial=True) == [(("d", "U0"), ("f", "U0"))]
    net, orders, (before, after) = even_gap_net(door, (6, 6), 'U0', fender_edge, 0.004, reach=0.07)
    front = edge_line(net, 'U0', 30)
    low = front[:, 2] < -0.05
    assert np.abs(-front[low, 1] - 0.004).max() < 2e-3, front[low, 1]      # evened where they face
    assert front[-1, 1] < -0.02                                           # the top stays clear
    assert after < before


def edge_line(net, edge, n=30):
    from core.nurbs import eval_points
    s = np.linspace(0, 1, n)
    o = {'V1': (s, np.ones(n)), 'V0': (s, np.zeros(n)), 'U1': (np.ones(n), s), 'U0': (np.zeros(n), s)}[edge]
    return eval_points(net, (net.shape[0], net.shape[1]), *o)


def test_sweep_rotate():
    from core.sweep import sweep_net
    # rail: a quarter turn in the xy plane (cubic arc), starting along +x at the origin
    k = 0.5523
    rail_pts = np.array([[0, 0, 0], [k, 0, 0], [1, 1 - k, 0], [1, 1, 0]], float)
    R = np.stack([rail_pts, rail_pts + [0, 0, -0.5]], 1)          # rail on its V... edge
    R = np.transpose(R, (0, 1, 2))                                  # (4, 2, 3): u along the rail
    # profile: a straight upright edge at the rail start, 0.3 tall, leaning out along -y
    prof = np.array([[0, 0, 0], [0, -0.1, 0.1], [0, -0.2, 0.2], [0, -0.3, 0.3]], float)
    P = np.stack([prof, prof + [-0.5, 0, 0]], 1)                     # (4, 2, 3): profile is the V0 edge
    net, orders = sweep_net(P, (4, 2), 'V0', R, (4, 2), 'V0', rotate=True)
    assert net.shape == (4, 4, 3)
    assert np.allclose(net[0], rail_pts)                            # the rail edge is exact
    end = net[:, -1] - rail_pts[-1]                                 # the profile at the rail end
    assert np.allclose(end[-1], [0.3, 0, 0.3], atol=1e-6), end[-1]   # turned a quarter: -y -> +x
    plain, _ = sweep_net(P, (4, 2), 'V0', R, (4, 2), 'V0')
    assert np.allclose(plain[-1, -1] - rail_pts[-1], [0, -0.3, 0.3])  # without: moved unturned


def test_diagnose():
    from core.diagnose import diagnose, expected_seams, waviness
    from core.presets import build
    (_, hood), = build('HOOD')
    assert waviness(hood, (6, 6)) == {"inflections": 0, "bumps": 0, "line": None}
    flat = hood.copy()
    flat[..., 2] = 0.0
    assert waviness(flat, (6, 6))["inflections"] == 0                # flat: nothing to report
    wavy = hood.copy()
    wavy[2:4, :, 2] -= 0.15                                           # a hollow across the middle
    w = waviness(wavy, (6, 6))
    assert w["inflections"] >= 2, w
    metas = {"A": {"method": "raw", "sources": [], "matches": [{"ref": "B", "continuity": "G2"}]},
             "B": None, "C": {"method": "extend", "sources": ["B"], "continuity": "G1"}}
    assert expected_seams(metas["A"]) == {"B": "G2"} and expected_seams(metas["C"]) == {"B": "G1"}
    seams = [{"a": "A", "b": "B", "grade": "G1", "angle": 0.0, "curvature": 0.3},
             {"a": "B", "b": "C", "grade": "G1", "angle": 0.0, "curvature": 0.3},
             {"a": "B", "b": "D", "grade": "G0", "angle": 4.0, "curvature": 1.0}]
    issues = diagnose({"W": (wavy, (6, 6)), "H": (hood, (6, 6))}, metas, seams)
    kinds = [(i["kind"], i["a"], i["b"]) for i in issues]
    assert kinds == [("seam_below", "A", "B"), ("seam_kink", "B", "D"), ("inflection", "W", "W")], kinds
    assert "Regenerate A" in issues[0]["hint"]
    trimmed = {"method": "fillet_trim", "sources": [], "fillet": {"id": "f1"}}
    apart = [{"a": "P", "b": "Q", "grade": "open", "angle": 0.0, "curvature": 0.0}]
    assert diagnose({}, {"P": trimmed, "Q": trimmed}, apart) == []          # the fillet joins them
    assert [i["kind"] for i in diagnose({}, {}, apart)] == ["seam_open"]
    near = [dict(apart[0], gap=0.005, size=1.0)]
    far = [dict(apart[0], gap=0.1, size=1.0)]                  # a pillar strip between them
    assert [i["kind"] for i in diagnose({}, {}, near)] == ["seam_open"]
    assert diagnose({}, {}, far) == []
    # v0.82: Seam Colors drops the same seams as Diagnose (one rule, drop_intended)
    from core.diagnose import drop_intended
    merged = {"method": "fill", "sources": ["Q"], "continuity": "G2", "merge": True}
    rows = [dict(apart[0], gap=0.005, size=1.0)]
    assert drop_intended(rows, {"P": trimmed, "Q": trimmed}) == []      # their fillet joins them
    assert drop_intended(rows, {"P": merged}) == []                      # a piece of a split fill
    assert drop_intended(rows, {}) == rows                               # a near miss stays
    assert drop_intended(far, {}) == []                                  # well apart, nothing asked
    # a Mirror Group copy of a fillet's patch meets its original on the mirror plane: kept
    mirror = {"method": "mirror", "sources": ["P"]}
    centre = [{"a": "P", "b": "P_mirror", "grade": "G2", "gap": 0.0, "size": 1.0}]
    assert drop_intended(centre, {"P": trimmed, "P_mirror": mirror}) == centre
    pair = [{"a": "P_mirror", "b": "Q_mirror", "grade": "open", "gap": 0.005, "size": 1.0}]
    assert drop_intended(pair, {"P": trimmed, "Q": trimmed, "P_mirror": mirror,
                                "Q_mirror": {"method": "mirror", "sources": ["Q"]}}) == []
    asked_far = [dict(far[0])]
    assert drop_intended(asked_far, {"P": {"method": "raw", "sources": [],
                                           "matches": [{"ref": "Q", "continuity": "G1"}]}}) == asked_far
    from core.diagnose import drop_accepted, issue_keys
    kink = {"kind": "seam_kink", "a": "B", "b": "D"}
    wave = {"kind": "inflection", "a": "W", "b": "W"}
    assert issue_keys(kink) == {"B": "seam_kink:D", "D": "seam_kink:B"}
    assert issue_keys(wave) == {"W": "inflection"}
    shown, hidden = drop_accepted([kink, wave], {"D": {"seam_kink:B"}})
    assert shown == [wave] and hidden == 1                   # marked on either patch is enough


def test_fill_merges_straight_corner():
    """A triangle whose bottom side is two patch edges meeting straight on: with merge
    it is filled by its three real corners (three star pieces, flat, no fold)."""
    from core.fill import fill_patches, side_from_patch
    from core.diagnose import waviness
    from core.nurbs import eval_points
    g = np.linspace(0, 1, 4)

    def patch(o, a, b):            # bilinear 4x4 net: o + u a + v b, the hole on v = 1
        o, a, b = (np.array(x, float) for x in (o, a, b))
        return o + g[:, None, None] * a + g[None, :, None] * b

    P1 = patch((0, -1, 0), (1, 0, 0), (0, 1, 0))      # below A(0,0)-M(1,0)
    P2 = patch((1, -1, 0), (1, 0, 0), (0, 1, 0))      # below M(1,0)-B(2,0)
    P3 = patch((3, 1, 0), (-2, 2, 0), (-1, -1, 0))    # beyond B(2,0)-C(0,2)
    P4 = patch((-1, 2, 0), (0, -2, 0), (1, 0, 0))     # left of C(0,2)-A(0,0)
    sides = [side_from_patch(P, (4, 4), 'V1') for P in (P1, P2, P3, P4)]
    old = fill_patches(sides, 'G1', layout='GRID')
    new = fill_patches(sides, 'G1', layout='GRID', merge=True)
    assert len(old) == 1 and len(new) == 3, (len(old), len(new))
    for net, orders in new:
        assert np.abs(net[..., 2]).max() < 1e-6                     # flat stays flat
        assert waviness(net, orders)["inflections"] == 0
    # together the pieces cover the triangle: area 2
    area = 0.0
    for net, orders in new:
        t = np.linspace(0, 1, 41)
        U, V = np.meshgrid(t, t, indexing="ij")
        du = eval_points(net, orders, U.ravel(), V.ravel(), 1, 0)
        dv = eval_points(net, orders, U.ravel(), V.ravel(), 0, 1)
        area += np.abs(np.cross(du, dv)[:, 2]).mean()
    assert abs(area - 2.0) < 0.05, area
    # the joined side runs exactly through the corner its two edges share (M), so
    # To Mesh can weld the pieces to both neighbours there
    edge_pts = np.concatenate([eval_points(net, o, np.linspace(0, 1, 401), np.zeros(401))
                               for net, o in new] + [eval_points(net, o, np.zeros(401),
                               np.linspace(0, 1, 401)) for net, o in new])
    assert np.linalg.norm(edge_pts - [1.0, 0.0, 0.0], axis=1).min() < 1e-9


def test_chord_allowance():
    from core.meshing import chord_allowance, t_junctions
    # a straight edge next to a real corner keeps the strict 5%
    P = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]], float)
    E = np.array([[0, 1], [1, 2], [2, 3]])
    assert chord_allowance(P, E)[1] == 0.05
    # a coarse chord across a quarter arc (r = 1), its neighbours leaving along the
    # arc's tangents: the arc's middle sags 21% of the chord off it and still welds
    Q = np.array([[1, 0, 0], [2, 0, 0], [3, 1, 0], [3, 2, 0]], float)
    F = np.array([[0, 1], [1, 2], [2, 3]])
    allow = chord_allowance(Q, F)
    assert allow[0] == 0.05 and allow[1] > 0.21, allow
    mid = [2 + np.sin(np.pi / 4), 1 - np.cos(np.pi / 4), 0.0]
    pts = np.vstack([Q, mid])
    assert t_junctions(pts, F[1:2], [4], 1e-6, allow[1:2])
    assert not t_junctions(pts, F[1:2], [4], 1e-6, 0.05)        # the old fixed 5% missed it


def test_partial_contact():
    from core.continuity import find_shared_edge, partial_contact
    g = np.linspace(0, 1, 4)

    def patch(o, a, b):
        o, a, b = (np.array(x, float) for x in (o, a, b))
        return o + g[:, None, None] * a + g[None, :, None] * b

    long_ = patch((0, -1, 0), (2, 0, 0), (0, 1, 0))           # its v=1 edge: x 0..2 at y=0
    short = patch((0.5, 0, 0), (1, 0, 0), (0, 1, 0))           # its v=0 edge: x 0.5..1.5 at y=0
    assert find_shared_edge(long_, short, 1e-3) is None       # not a whole-edge seam
    c = partial_contact(long_, short, 1e-3)
    assert c and {c["edge_a"], c["edge_b"]} == {"V1", "V0"} and c["angle"] < 1e-6, c
    assert abs(c["length"] - 1.0) < 0.06, c
    tilt = short.copy()                                         # kinked 5 degrees along the seam
    tilt[..., 2] = tilt[..., 1] * np.tan(np.radians(5))
    assert abs(partial_contact(long_, tilt, 1e-3)["angle"] - 5.0) < 0.1
    apart = short + [0, 0.004, 0]                               # a 4 mm panel gap: not a seam
    assert partial_contact(long_, apart, 1e-3 * 2.0) is None


def test_spoke_fields_g1():
    """The two quads' cross fields along a star spoke stay coplanar with it (G1) and
    end on the half-side tangent and on each quad's other spoke at the centre."""
    from core.fill import _spoke_fields
    from core.nurbs import basis_matrix
    rng_ = np.random.default_rng(3)
    for _ in range(5):
        c = np.zeros(3)
        dirs = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.radians([90, 215, 330])]
        dirs = [d + 0.15 * rng_.normal(size=3) * [1, 1, 0] for d in dirs]
        dirs = [d / np.linalg.norm(d) for d in dirs]
        m = c + dirs[0] * 1.0 + [0, 0, 0.2]
        E = np.array([m, m + [0.05, -0.2, 0.0], m + [0.08, -0.4, -0.05], c + dirs[0] * 0.4,
                      c + dirs[0] * 0.2, c])
        D0 = np.array([-0.5, 0.05, 0.1])
        X1, Y1 = dirs[2] * 0.9, dirs[1] * 1.1
        Xc, Yc = _spoke_fields(E, D0, X1, Y1)
        t = np.linspace(0, 1, 40)
        X, Y = basis_matrix(6, 6, t) @ Xc, basis_matrix(6, 6, t) @ Yc
        dE = basis_matrix(6, 6, t, 1) @ E
        vol = np.abs(np.einsum("ij,ij->i", np.cross(X, Y), dE))
        scale = np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) * np.linalg.norm(dE, axis=1)
        assert (vol / scale).max() < 1e-9                      # coplanar all along: G1
        side = np.einsum("ij,ij->i", np.cross(dE, X), np.cross(dE, Y))
        assert (side < 0).all()                                # on opposite sides of the spoke
        assert np.allclose(X[0], D0) and np.allclose(Y[0], -D0)
        assert np.allclose(X[-1], X1) and np.allclose(Y[-1], Y1)


def test_partial_contact_tight_bend():
    """On a tight bend the normal turns quickly: both normals must be read at the same
    point, right on the edge (stepping a hair inside moved them apart along the edge
    and showed ~0.5 deg on a perfect seam)."""
    from core.continuity import partial_contact

    def parabolic(x0, x1, y0, y1, c=100.0):       # z = c x^2: steep (radius 5 mm at x = 0)
        xs = np.array([x0, 0.5 * (x0 + x1), x1])
        zs = np.array([c * x0 * x0, c * x0 * x1, c * x1 * x1])   # exact quadratic Bezier
        return np.stack([np.stack([xs, 0 * xs + y, zs], -1) for y in (y0, y1)], 1)

    A = parabolic(0.0, 0.8, -1.0, 0.0)            # its edge at y = 0 runs x 0..0.8
    B = parabolic(0.0, 0.4, 0.0, 1.0)             # the same surface beyond, x 0..0.4 only
    c = partial_contact(A, B, 1e-3, (3, 2), (3, 2))
    assert c is not None and c["gap"] < 1e-9, c
    assert c["angle"] < 1e-3, c                   # one smooth surface: no kink (read 0.46 before)


def test_presets():
    from core.presets import KINDS, POINTS, build
    for kind in KINDS:
        for name, net in build(kind):
            assert net.shape == (POINTS, POINTS, 3) and np.isfinite(net).all(), (kind, name)
    (_, hood), = build('HOOD', length=2.0, width=1.0, crown=0.1)
    assert abs(np.ptp(hood[..., 1]) - 2.0) < 1e-9 and abs(np.ptp(hood[..., 0]) - 1.0) < 1e-9
    assert 0.10 < hood[..., 2].max() <= 0.13 and hood[:, 0, 2].max() <= 0.031   # crowned: edges lower
    from core.presets import SEAMS
    for kind, seams in SEAMS.items():                    # the joined panels share their seam edge
        nets = [n for _, n in build(kind)]
        for a, ea, b, eb in seams:
            ka = {'U0': nets[a][0], 'U1': nets[a][-1], 'V0': nets[a][:, 0], 'V1': nets[a][:, -1]}[ea]
            kb = {'U0': nets[b][0], 'U1': nets[b][-1], 'V0': nets[b][:, 0], 'V1': nets[b][:, -1]}[eb]
            assert np.allclose(ka, kb), kind
    (_, bump), = build('BUMPER', width=2.0, wrap=0.3)
    assert abs(np.ptp(bump[..., 0]) - 2.0) < 1e-9 and bump[0, 0, 1] > bump[0, 2, 1]   # ends swept back
    (_, half), = build('HOOD', width=1.0, half=True)
    assert abs(half[..., 0].min()) < 1e-9 and abs(half[..., 0].max() - 0.5) < 1e-9
    (_, d), = build('DOOR', length=1.0, height=0.8)
    assert abs(np.ptp(d[..., 2]) - 0.8) < 1e-9
    (_, h), (_, f) = build('HOOD_FENDER', depth=0.4, bulge=0.0)
    # the fender wall starts exactly on the hood's side edge and drops by `depth`
    assert np.allclose(f[:, 0], h[:, -1]) and abs((f[:, -1, 2] - f[:, 0, 2]).mean() + 0.4) < 1e-9


def test_project_points():
    from core.project import project_points
    g = np.linspace(0, 1, 6)
    U, V = g[:, None] + 0 * g, 0 * g[:, None] + g
    net = np.stack([U, V, 0.3 * (1 - (2 * U - 1) ** 2) * (1 - (2 * V - 1) ** 2)], -1)
    o = (6, 6)
    top = project_points(net, o, [[0.5, 0.5, 1.0]])[0][0]            # nearest point lies straight below
    assert np.allclose(top, [0.5, 0.5, 0.192], atol=1e-6), top
    side = project_points(net, o, [[0.2, 0.7, -0.5]])[0][0]          # nearer to the patch's edge u = 0
    assert abs(side[0]) < 1e-9 and abs(side[2]) < 1e-9
    p, uv, ok = project_points(net, o, [[0.5, 0.5, 1.0], [0.2, 0.7, -0.5], [1.5, 0.5, 0.2]],
                               direction=(0, 0, -1))
    assert ok.tolist() == [True, True, False]                         # the last ray misses (x > 1)
    assert np.allclose(p[0], [0.5, 0.5, 0.192], atol=1e-6)
    assert np.allclose(p[1][:2], [0.2, 0.7]) and abs(p[1][2] - nurbs.eval_points(net, o, [0.2], [0.7])[0][2]) < 1e-6
    # a slanted ray lands on the patch along its own line
    q = project_points(net, o, [[0.0, 0.5, 0.6]], direction=(1, 0, -1))[0][0]
    assert abs(q[0] - (0.6 - q[2])) < 1e-6 and 0.0 < q[0] < 1.0


def test_trim_grid():
    from core.trim import trim_grid
    g = np.linspace(0, 1, 6)
    U, V = g[:, None] + 0 * g, 0 * g[:, None] + g
    net = np.stack([U, V, 0.3 * (1 - (2 * U - 1) ** 2) * (1 - (2 * V - 1) ** 2)], -1)
    o = (6, 6)

    def area(q, uv):
        a, b, c, d = (uv[q[:, i]] for i in range(4))
        cross = lambda p, r: p[:, 0] * r[:, 1] - r[:, 0] * p[:, 1]
        return 0.5 * (cross(a, b) + cross(b, c) + cross(c, d) + cross(d, a))

    rect = np.array([[0.3, 0.4], [0.7, 0.4], [0.7, 0.6], [0.3, 0.6]])
    verts, quads, uv = trim_grid(net, o, [rect], 48, 48)
    A = area(quads, uv)
    assert abs(A.sum() - (1 - 0.4 * 0.2)) < 1e-9 and A.min() > 0          # exact hole, no flipped cells
    on_edge = (np.abs(uv[:, 0] - 0.3) < 1e-9) | (np.abs(uv[:, 0] - 0.7) < 1e-9)         | (np.abs(uv[:, 1] - 0.4) < 1e-9) | (np.abs(uv[:, 1] - 0.6) < 1e-9)
    assert on_edge.sum() > 40                                              # the ring sits on the outline
    assert not np.any((uv[:, 0] > 0.3 + 1e-9) & (uv[:, 0] < 0.7 - 1e-9)
                      & (uv[:, 1] > 0.4 + 1e-9) & (uv[:, 1] < 0.6 - 1e-9))
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ellipse = np.stack([0.5 + 0.2 * np.cos(t), 0.5 + 0.15 * np.sin(t)], -1)
    verts, quads, uv = trim_grid(net, o, [ellipse], 48, 48)
    A = area(quads, uv)
    assert A.min() > 0
    assert abs(A.sum() - (1 - np.pi * 0.2 * 0.15)) < 0.01                  # the 40-gon is a touch smaller
    two = trim_grid(net, o, [rect, ellipse + np.array([0.3, 0.3])], 48, 48)   # two holes
    assert len(two[1]) > 0 and not np.any(np.isnan(two[0]))


# ---------------------------------------------------------------- curve editing (core.curvepoint)
def _smooth_chain(n=6, cyclic=False, seed=1):
    """A G1 Bezier chain through a wavy 3D path (handles from the path's derivative)."""
    r = np.random.default_rng(seed)
    if cyclic:
        s = np.linspace(0, 2 * np.pi, n, endpoint=False)
        f = lambda s: np.stack([np.cos(s) * (2 + 0.3 * np.sin(3 * s)), np.sin(s) * 2,
                                0.4 * np.sin(2 * s)], -1)
        df = lambda s: (f(s + 1e-5) - f(s - 1e-5)) / 2e-5
        h = 2 * np.pi / n / 3
    else:
        s = np.linspace(0, 3, n)
        f = lambda s: np.stack([s * 2, np.sin(s * 1.7), 0.3 * np.cos(s * 2.3)], -1)
        df = lambda s: (f(s + 1e-5) - f(s - 1e-5)) / 2e-5
        h = 3 / (n - 1) / 3
    A = f(s)
    D = df(s) * h * (1 + 0.2 * r.random((n, 1)))   # uneven handle lengths, still G1
    return A, A - D, A + D


def _off(A, L, R, cyclic, A2, L2, R2, cyclic2=None):
    """Largest distance of the second chain's shape from the first's."""
    from core.curvepoint import nearest, point_at, segments
    c2 = cyclic if cyclic2 is None else cyclic2
    m = segments(len(A2), c2)
    pts = point_at(A2, L2, R2, np.linspace(0, m, 40 * m + 1), c2)
    return max(nearest(A, L, R, p, cyclic)[1] for p in pts)


def test_curve_insert_and_dissolve():
    from core.curvepoint import dissolve_point, insert_point, point_at
    for cyclic in (False, True):
        A, L, R = _smooth_chain(cyclic=cyclic)
        A2, L2, R2, i = insert_point(A, L, R, 2.37, cyclic)
        assert len(A2) == len(A) + 1 and i == 3
        assert np.allclose(A2[i], point_at(A, L, R, [2.37], cyclic)[0])
        assert _off(A, L, R, cyclic, A2, L2, R2) < 1e-9            # the shape is untouched
        # back out the inserted point: the original span is one cubic, so it comes back
        A3, L3, R3, err = dissolve_point(A2, L2, R2, i, cyclic)
        assert len(A3) == len(A) and err < 1e-4
        assert np.abs(R3 - R).max() < 1e-3 and np.abs(L3 - L).max() < 1e-3
        # on an anchor: nothing changes
        A4, _, _, j = insert_point(A, L, R, 2.0, cyclic)
        assert j == 2 and len(A4) == len(A)


def test_curve_dissolve_keeps_neighbours():
    from core.curvepoint import _out_dir, dissolve_point
    A, L, R = _smooth_chain(n=8)
    A2, L2, R2, err = dissolve_point(A, L, R, 4)
    assert len(A2) == 7
    size = np.ptp(A, 0).max()
    assert err < 0.02 * size                                         # close to the old shape
    assert _off(A, L, R, False, A2, L2, R2) < 0.02 * size
    for j, j2 in ((3, 3), (5, 4)):                                   # neighbours: same tangent
        assert np.allclose(_out_dir(A, L, R, j, False, 'R'), _out_dir(A2, L2, R2, j2, False, 'R'))
        assert np.allclose(_out_dir(A, L, R, j, False, 'L'), _out_dir(A2, L2, R2, j2, False, 'L'))
    A3, _, _, _ = dissolve_point(A, L, R, 0)                         # an end just goes
    assert len(A3) == 7 and np.allclose(A3[0], A[1])


def test_curve_slide_keeps_shape():
    from core.curvepoint import point_at, slide_point
    for cyclic in (False, True):
        # 12 points round the wavy loop; with fewer, one span cannot follow a lobe
        A, L, R = _smooth_chain(n=12 if cyclic else 7, cyclic=cyclic)
        size = np.ptp(A, 0).max()
        for i, t in ((3, 3.4), (3, 2.55), (1, 0.4)):
            A2, L2, R2, err = slide_point(A, L, R, i, t, cyclic)
            assert np.allclose(A2[i], point_at(A, L, R, [t], cyclic)[0])     # on the old curve
            assert err < 5e-3 * size, (cyclic, i, t, err)
            assert _off(A, L, R, cyclic, A2, L2, R2) < 5e-3 * size
            assert len(A2) == len(A)
        if cyclic:                                                   # across the seam at point 0
            for t in (0.4, len(A) - 0.4):
                A2, L2, R2, err = slide_point(A, L, R, 0, t, cyclic)
                assert np.allclose(A2[0], point_at(A, L, R, [t], cyclic)[0])
                assert _off(A, L, R, cyclic, A2, L2, R2) < 5e-3 * size, t
        else:                                                        # ends trim exactly
            A2, L2, R2, err = slide_point(A, L, R, 0, 0.3, cyclic)
            assert err == 0 and _off(A, L, R, False, A2, L2, R2) < 1e-9
            assert np.allclose(A2[0], point_at(A, L, R, [0.3])[0])
            A2, L2, R2, _ = slide_point(A, L, R, len(A) - 1, len(A) - 1.5, cyclic)
            assert _off(A, L, R, False, A2, L2, R2) < 1e-9
        # held between its neighbours
        A2, _, _, _ = slide_point(A, L, R, 3, 9.0, cyclic)
        assert np.linalg.norm(A2[3] - A[4]) < 0.05 * size


def test_curve_arc_param():
    from core.curvepoint import arc_param, arc_table
    A, L, R = _smooth_chain()
    t, s = arc_table(A, L, R)
    t1 = arc_param(A, L, R, 1.0, 0.5)
    assert abs(np.interp(t1, t, s) - np.interp(1.0, t, s) - 0.5) < 1e-6
    assert arc_param(A, L, R, 1.0, -1e9) == 0.0


def test_curve_pull():
    from core.curvepoint import arc_table, point_at
    from core.curvepull import chain_plane, pull_curve, smooth_points
    A, L, R = _smooth_chain(n=9)
    size = np.ptp(A, 0).max()
    delta = np.array([0.0, 0.0, 0.4])
    for profile in ('SMOOTH', 'SPHERE', 'LINEAR', 'SHARP', 'CONSTANT'):
        A2, L2, R2 = pull_curve(A, L, R, 3.4, delta, radius=0.35 * size, profile=profile)
        hit = point_at(A2, L2, R2, [3.4])[0] - point_at(A, L, R, [3.4])[0]
        assert np.linalg.norm(hit - delta) < 1e-9, (profile, hit)                # exact
        assert np.allclose(A2[0], A[0]) and np.allclose(R2[0], R[0])      # pinned end + tangent
        assert np.allclose(A2[-1], A[-1]) and np.allclose(L2[-1], L[-1])
        assert smooth_points(A2, L2, R2).all()                             # still G1 everywhere
    # outside the radius nothing moves
    t, s = arc_table(A, L, R)
    A2, L2, R2 = pull_curve(A, L, R, 4.0, delta, radius=0.1 * size)
    far = np.abs(np.interp(np.arange(9.0), t, s) - np.interp(4.0, t, s)) > 0.15 * size
    assert np.allclose(A2[far], A[far])
    # a fixed point stays put while a wide pull goes past it
    A2, L2, R2 = pull_curve(A, L, R, 3.0, delta, radius=0.6 * size, fixed=[5.0])
    assert np.linalg.norm(point_at(A2, L2, R2, [5.0])[0] - point_at(A, L, R, [5.0])[0]) < 1e-9
    assert np.linalg.norm(point_at(A2, L2, R2, [3.0])[0] - point_at(A, L, R, [3.0])[0] - delta) < 1e-9
    # ends free: the end itself follows
    A2, _, _ = pull_curve(A, L, R, 0.0, delta, radius=0.3 * size, pin_ends=False)
    assert np.linalg.norm(A2[0] - A[0] - delta) < 0.02 * 0.4
    # cyclic
    A, L, R = _smooth_chain(n=12, cyclic=True)
    A2, L2, R2 = pull_curve(A, L, R, 0.2, delta, radius=0.3 * np.ptp(A, 0).max(), cyclic=True)
    hit = point_at(A2, L2, R2, [0.2], True)[0] - point_at(A, L, R, [0.2], True)[0]
    assert np.linalg.norm(hit - delta) < 0.02 * 0.4 and smooth_points(A2, L2, R2).all()
    # a flat curve pulled within its plane stays flat
    xs = np.linspace(0, 6, 5)
    A = np.stack([xs, np.sin(xs), np.zeros(5)], -1)
    D = np.stack([np.ones(5), np.cos(xs), np.zeros(5)], -1) * 0.5
    L, R = A - D, A + D
    c, nrm = chain_plane(A, L, R)
    assert abs(abs(nrm[2]) - 1) < 1e-9
    A2, L2, R2 = pull_curve(A, L, R, 2.2, np.array([0.3, 0.5, 0.0]), radius=3.0)
    assert np.abs(np.concatenate([A2, L2, R2])[:, 2]).max() < 1e-12
    assert chain_plane(A + np.array([[0, 0, 0.0]] * 4 + [[0, 0, 1.0]]), L, R) is None


def _flat_chain(n=7, seed=2, cyclic=False):
    """A flat (z = 0) G1 chain bending one way (no inflection), uneven handle lengths."""
    r = np.random.default_rng(seed)
    s = np.linspace(0, 2 * np.pi, n, endpoint=False) if cyclic else np.linspace(0, 2.4, n)
    rad = 3.0
    A = np.stack([rad * np.cos(s), rad * np.sin(s), 0 * s], -1)
    T = np.stack([-np.sin(s), np.cos(s), 0 * s], -1)
    step = (s[1] - s[0]) * rad / 3
    a = step * (0.6 + 0.8 * r.random(n))
    b = step * (0.6 + 0.8 * r.random(n))
    return A, A - a[:, None] * T, A + b[:, None] * T


def test_curve_make_g2():
    from core.curvesmooth import make_all_g2, make_g2, point_state, side_curvature
    A, L, R = _flat_chain()
    i = 3
    assert point_state(A, L, R, i) == 'G1'
    A2, L2, R2, why = make_g2(A, L, R, i)
    assert why is None and point_state(A2, L2, R2, i) == 'G2'
    kl, kr = side_curvature(A2, L2, R2, i)
    assert np.linalg.norm(kl - kr) < 1e-9 * np.linalg.norm(kl) + 1e-12
    moved = [j for j in range(len(A)) if not (np.allclose(L2[j], L[j]) and np.allclose(R2[j], R[j]))]
    assert moved == [i] and np.allclose(A2, A)                        # only this point's handles
    assert np.allclose(np.cross(R2[i] - A[i], R[i] - A[i]), 0)        # along the same tangent
    # the whole curve, a few passes
    A3, L3, R3, left = make_all_g2(A, L, R, range(1, len(A) - 1))
    assert not left and all(point_state(A3, L3, R3, j) == 'G2' for j in range(1, len(A) - 1))
    # a loop
    A, L, R = _flat_chain(n=8, cyclic=True)
    A3, L3, R3, left = make_all_g2(A, L, R, range(len(A)), cyclic=True)
    assert not left and all(point_state(A3, L3, R3, j, True) == 'G2' for j in range(len(A)))
    # an S bend: bending up on the left, down on the right; lengths cannot make it G2
    A = np.array([(-2.0, 1, 0), (0, 0, 0), (2, -1, 0)])
    L = np.array([(-2.7, 1.3, 0), (-0.7, 0, 0), (1.3, -1, 0)])
    R = np.array([(-1.3, 1.0, 0), (0.9, 0, 0), (2.7, -1.3, 0)])
    assert make_g2(A, L, R, 1)[3] == 'inflection'
    xs = np.linspace(0, 6, 5)
    A = np.stack([xs, np.sin(xs * np.pi / 3), 0 * xs], -1)
    D = np.stack([np.ones(5), np.pi / 3 * np.cos(xs * np.pi / 3), 0 * xs], -1) * 0.5
    assert make_g2(A, A - D, A + D, 0)[3] == 'end'
    L = A - D
    L[2] = A[2] + np.array([0.0, -0.5, 0.0])                   # a corner
    assert make_g2(A, L, A + D, 2)[3] == 'corner'


def test_curve_frame():
    from core.curvesmooth import frame
    A, L, R = _flat_chain()
    t, nrm, b = frame(A, L, R, 3)
    assert abs(t @ nrm) < 1e-9 and abs(np.linalg.norm(nrm) - 1) < 1e-9
    assert nrm @ (-A[3]) > 0                     # bends towards the circle's centre
    assert abs(abs(b[2]) - 1) < 1e-9             # out of the flat curve's plane
    xs = np.linspace(0, 3, 4)                    # straight: any perpendicular, kept in its plane
    A = np.stack([xs, 0 * xs, 0 * xs], -1)
    D = np.array([0.3, 0, 0])
    t, nrm, b = frame(A, A - D, A + D, 1)
    assert abs(t[0] - 1) < 1e-9 and abs(nrm @ t) < 1e-9


def test_curve_refit():
    from core.curvepoint import nearest, point_at, segments
    from core.curvesmooth import point_state, refit
    from core.curvepull import smooth_points
    A, L, R = _smooth_chain(n=25)
    size = np.ptp(A, 0).max()
    A2, L2, R2, err = refit(A, L, R, count=7)
    assert len(A2) == 7 and err < 0.01 * size, err
    assert _off(A, L, R, False, A2, L2, R2) < 0.01 * size
    assert np.allclose(A2[0], A[0]) and np.allclose(A2[-1], A[-1])
    assert smooth_points(A2, L2, R2)[1:-1].all()
    A3, L3, R3, err3 = refit(A, L, R, tolerance=1e-3 * size)
    assert err3 <= 1e-3 * size and len(A3) < 25, (len(A3), err3)
    # a corner is kept as a corner
    Lc = L.copy()
    Lc[12] = A[12] + np.array([0.0, 0.0, 0.4])
    A4, L4, R4, _ = refit(A, Lc, R, count=6)
    k = int(np.argmin(np.linalg.norm(A4 - A[12], axis=1)))
    assert np.linalg.norm(A4[k] - A[12]) < 1e-9 and point_state(A4, L4, R4, k) == 'CORNER'
    # a loop
    A, L, R = _smooth_chain(n=24, cyclic=True)
    A5, L5, R5, err5 = refit(A, L, R, cyclic=True, count=10)
    assert len(A5) == 10 and err5 < 0.01 * np.ptp(A, 0).max()
    assert _off(A, L, R, True, A5, L5, R5) < 0.01 * np.ptp(A, 0).max()


def test_curve_attach_end():
    from core.curveends import attach_end, end_frame
    from core.curvesmooth import point_state
    A, L, R = _flat_chain(n=5)
    # G0: the end moves, with its handle
    p = A[0] + np.array([0.1, -0.2, 0.3])
    A2, L2, R2 = attach_end(A, L, R, 'START', p)
    assert np.allclose(A2[0], p) and np.allclose(R2[0] - A2[0], R[0] - A[0])
    assert np.allclose(A2[1:], A[1:])
    # G1: leaves along the given direction, handle length kept
    d = np.array([0.0, 0.6, 0.8])
    A2, L2, R2 = attach_end(A, L, R, 'END', p, direction=d)
    q, t, k = end_frame(A2, L2, R2, 'END')
    assert np.allclose(q, p) and np.allclose(t, d)
    assert abs(np.linalg.norm(L2[-1] - A2[-1]) - np.linalg.norm(L[-1] - A[-1])) < 1e-12
    # G2: and bends as asked; the next point stays smooth
    want = np.array([0.4, 0.0, 0.0]) - (np.array([0.4, 0, 0]) @ d) * d
    A2, L2, R2 = attach_end(A, L, R, 'START', A[0], direction=d, curvature=want)
    q, t, k = end_frame(A2, L2, R2, 'START')
    assert np.allclose(t, d) and np.allclose(k, want, atol=1e-9), (k, want)
    assert point_state(A2, L2, R2, 1) in ('G1', 'G2')
    # continuing another curve's end: the same point, the opposite direction, the same bend
    B, M, N = A + np.array([0.0, 0.0, 1.0]), L + np.array([0.0, 0.0, 1.0]), R + np.array([0.0, 0.0, 1.0])
    p0, t0, k0 = end_frame(A, L, R, 'END')
    B2, M2, N2 = attach_end(B, M, N, 'START', p0, direction=-t0, curvature=k0)
    q, t, k = end_frame(B2, M2, N2, 'START')
    assert np.allclose(q, p0) and np.allclose(t, -t0) and np.allclose(k, k0, atol=1e-9)


def test_curve_two_view():
    from core.curvemake import fit_points, two_view
    from core.curvepoint import point_at, segments
    y = np.linspace(0, 4, 60)
    side = np.stack([np.full_like(y, 5.0), y, 0.5 * np.sin(y)], -1)        # x = 5: seen from the side
    top = np.stack([1 + 0.3 * y ** 1.5, y, np.full_like(y, -2.0)], -1)[::-1]   # z = -2: from the top (drawn backwards)
    pts, c = two_view(side, top)
    assert c == 1
    assert np.allclose(pts[:, 2], 0.5 * np.sin(pts[:, 1]), atol=2e-3)       # height from the side view
    assert np.allclose(pts[:, 0], 1 + 0.3 * pts[:, 1] ** 1.5, atol=2e-3)   # width from the top view
    A, L, R, err = fit_points(pts)
    assert len(A) < 15 and err < 5e-3
    for bad, msg in ((side, "same view"), (np.stack([y, y, y], -1), "flat")):
        try:
            two_view(side, bad)
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError("expected an error")


def test_curve_fillet_corner():
    from core.curvemake import fillet_corner
    from core.curvepoint import arc_table
    from core.curvesmooth import point_state
    # an L of two straight pieces: the round must be a quarter circle-like G1 span
    A = np.array([(0.0, 0, 0), (2, 0, 0), (2, 2, 0)])
    L = np.array([(-0.5, 0, 0), (1.5, 0, 0), (2, 1.5, 0)])
    R = np.array([(0.5, 0, 0), (2, 0.5, 0), (2, 2.5, 0)])
    A2, L2, R2, j = fillet_corner(A, L, R, 1, 0.5)
    assert len(A2) == 4 and np.allclose(A2[j], (1.5, 0, 0), atol=1e-3) \
        and np.allclose(A2[j + 1], (2, 0.5, 0), atol=1e-3)                  # cut 0.5 along each side
    assert point_state(A2, L2, R2, j) != 'CORNER' and point_state(A2, L2, R2, j + 1) != 'CORNER'
    mid = 0.125 * A2[j] + 0.375 * R2[j] + 0.375 * L2[j + 1] + 0.125 * A2[j + 1]
    assert abs(np.linalg.norm(mid - (1.5, 0.5, 0)) - 0.5) < 2e-3          # on the circle of radius 0.5
    # a loop with the corner at point 0
    A = np.array([(0.0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
    L = A + np.array([(0, 0.4, 0), (-0.4, 0, 0), (0, -0.4, 0), (0.4, 0, 0)])
    R = A + np.array([(0.4, 0, 0), (0, 0.4, 0), (-0.4, 0, 0), (0, -0.4, 0)])
    A2, L2, R2, j = fillet_corner(A, L, R, 0, 0.3, cyclic=True)
    assert len(A2) == 5 and all(point_state(A2, L2, R2, k, True) != 'CORNER'
                                for k in (j, (j + 1) % 5))
    try:
        fillet_corner(A2, L2, R2, j, 0.1, cyclic=True)
    except ValueError:
        pass
    else:
        raise AssertionError("a smooth point is not a corner")


def test_curve_offset_and_extend():
    from core.curvemake import extend_end, fit_points, offset_points
    from core.curvepoint import point_at
    from core.curvesmooth import point_state
    s = np.linspace(0, np.pi / 2, 4)                                         # a quarter circle, r = 3
    A = np.stack([3 * np.cos(s), 3 * np.sin(s), 0 * s], -1)
    T = np.stack([-np.sin(s), np.cos(s), 0 * s], -1) * (3 * 4 / 3 * np.tan(np.pi / 24))   # circle handles
    L, R = A - T, A + T
    pts = offset_points(A, L, R, 0.5, (0, 0, 1))
    r = np.linalg.norm(pts[:, :2], axis=1)
    assert np.abs(r - 2.5).max() < 2e-3                                     # to the left: inwards
    pts = offset_points(A, L, R, -0.5, (0, 0, 1))
    assert np.abs(np.linalg.norm(pts[:, :2], axis=1) - 3.5).max() < 2e-3
    B, M, N, err = fit_points(pts)
    assert err < 5e-3
    # extend naturally: continues the circle-like cubic, smooth at the old end
    A2, L2, R2 = extend_end(A, L, R, 'END', 0.8, 'NATURAL')
    assert len(A2) == 5 and point_state(A2, L2, R2, 3) in ('G1', 'G2')
    seg = np.linspace(0, 1, 30)
    tail = point_at(A2, L2, R2, 3 + seg)
    assert np.linalg.norm(np.diff(tail, axis=0), axis=1).sum() - 0.8 < 1e-3
    assert np.abs(np.linalg.norm(tail[:, :2], axis=1) - 3.0).max() < 0.02   # still near the circle
    A3, L3, R3 = extend_end(A, L, R, 'START', 1.0, 'STRAIGHT')
    assert len(A3) == 5 and np.allclose(A3[0], (3, -1, 0))
    assert point_state(A3, L3, R3, 1) == 'G1'          # straight meets the circle: tangent, not bend


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
