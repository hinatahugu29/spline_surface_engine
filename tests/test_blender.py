"""Headless Blender test.
blender -b --factory-startup --python tests/test_blender.py
"""
import os
import sys
import json
import traceback

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import spline_surface  # noqa: E402
from spline_surface.bridge.write_surface import read_surface_net  # noqa: E402
from spline_surface.core.bezier import eval_surface  # noqa: E402
from spline_surface.operators import world_net  # noqa: E402


def bezier_curve(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for bp, co in zip(sp.bezier_points, pts):
        bp.co = co
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(ob)
    return ob


def select(objs):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def evaluated_verts(ob):
    ob_eval = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    m = ob_eval.to_mesh()
    v = np.array([x.co[:] for x in m.vertices])
    ob_eval.to_mesh_clear()
    return v


def nurbs_eval(net, orders, g):
    from spline_surface.core.nurbs import eval_surface
    return eval_surface(net, orders, g, g).reshape(-1, 3)


def run():
    spline_surface.register()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    # A bulged 4x3 "hood" panel; curve directions deliberately mixed and
    # corners slightly off to exercise orientation and snapping.
    c = [
        bezier_curve("bottom", [(0, 0, 0), (2, -0.2, 0.6), (4, 0, 0)]),
        bezier_curve("right", [(4, 3, 0), (4.3, 1.5, 0.4), (4, 0.01, 0)]),
        bezier_curve("top", [(0, 3, 0), (2, 3.3, 0.8), (4, 3, 0)]),
        bezier_curve("left", [(0, 0, 0.005), (-0.2, 1.5, 0.3), (0, 3, 0)]),
    ]
    c[1].location = (0, 0, 0)
    bpy.context.scene.spline_surface.resolution = 64
    select(c)
    assert bpy.ops.spline_surface.coons() == {'FINISHED'}
    patch = bpy.context.active_object
    sp = patch.data.splines[0]
    print("grid", sp.point_count_u, sp.point_count_v, "order", sp.order_u, sp.order_v)
    assert (sp.point_count_u, sp.point_count_v) == (6, 6)

    # Blender's own evaluation must match our math (checks point ordering).
    net = read_surface_net(patch.data)
    ours = eval_surface(net, np.linspace(0, 1, 7), np.linspace(0, 1, 7)).reshape(-1, 3)
    verts = evaluated_verts(patch)
    d = np.array([np.linalg.norm(verts - p, axis=1).min() for p in ours])
    print("blender-vs-core max dist", d.max())
    assert d.max() < 0.01, d.max()  # tessellation distance only

    # Corners land on the curve ends.
    for corner in [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)]:
        assert np.linalg.norm(verts - np.array(corner), axis=1).min() < 0.01, corner

    # Regenerate after editing a source curve and changing degree.
    bpy.context.scene.spline_surface.degree = 3
    c[2].data.splines[0].bezier_points[1].co.z = 1.5
    select([patch])
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    sp = patch.data.splines[0]
    assert (sp.point_count_u, sp.point_count_v) == (4, 4)
    assert evaluated_verts(patch)[:, 2].max() > 1.0

    # Zebra toggles on and back off.
    assert bpy.ops.spline_surface.zebra_toggle() == {'FINISHED'}
    assert patch.data.materials[0].name == "SS_Zebra"
    bpy.context.scene.spline_surface.zebra_axis = 'X'
    # stripe width and softness drive the map-range thresholds: white where sin(x) > cos(pi * width)
    import math
    ss = bpy.context.scene.spline_surface
    ss.zebra_width, ss.zebra_softness = 0.25, 0.1
    rng = bpy.data.materials["SS_Zebra"].node_tree.nodes["range"].inputs
    assert abs(rng["From Min"].default_value - (math.cos(math.pi * 0.25) - 0.1)) < 1e-6
    assert abs(rng["From Max"].default_value - (math.cos(math.pi * 0.25) + 0.1)) < 1e-4
    ss.zebra_width, ss.zebra_softness = 0.5, 0.05
    assert bpy.ops.spline_surface.zebra_toggle() == {'FINISHED'}
    assert len(patch.data.materials) == 0

    # Three-sided patch.
    tri = [bezier_curve("t0", [(0, 0, 0), (1, -0.3, 0.2), (2, 0, 0)]),
           bezier_curve("t1", [(2, 0, 0), (1.6, 1, 0.3), (1, 2, 0)]),
           bezier_curve("t2", [(1, 2, 0), (0.4, 1, 0.3), (0, 0, 0)])]
    select(tri)
    assert bpy.ops.spline_surface.coons() == {'FINISHED'}

    # Mesh output.
    select([patch])
    assert bpy.ops.spline_surface.to_mesh() == {'FINISHED'}
    mob = bpy.context.active_object
    print("mesh verts", len(mob.data.vertices), "faces", len(mob.data.polygons))
    assert mob.type == 'MESH' and len(mob.data.polygons) > 100

    # Extend the hood patch off its V max edge, then follow a source edit.
    select([patch])
    patch.location = (0.5, 0, 0)  # transforms must be honoured
    assert bpy.ops.spline_surface.extend_edge(edge='V1', length=0.4, continuity='G2') == {'FINISHED'}
    ext = bpy.context.active_object
    src_net = world_net(patch)[0]
    ext_net = read_surface_net(ext.data)
    assert np.allclose(ext_net[:, 0], src_net[:, -1], atol=1e-5)
    # G1: first row direction continues the source's last row direction.
    d_src = src_net[:, -1] - src_net[:, -2]
    d_ext = ext_net[:, 1] - ext_net[:, 0]
    cos = (d_src * d_ext).sum(1) / np.linalg.norm(d_src, axis=1) / np.linalg.norm(d_ext, axis=1)
    assert cos.min() > 0.9999, cos
    patch.location = (0, 0, 0)
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    assert np.allclose(read_surface_net(ext.data)[:, 0], world_net(patch)[0][:, -1], atol=1e-5)
    for e in ('V0', 'U0', 'U1'):
        select([patch])
        assert bpy.ops.spline_surface.extend_edge(edge=e, continuity='G1') == {'FINISHED'}

    # Extend Edge's "click a second edge to reach it" length solver. Use a
    # flat, axis-aligned patch of its own so the extension direction (+x off
    # the U1 edge) is known, independent of the hood patch's own curvature.
    from spline_surface.operators import (_solve_extend_length, _extend_far_edge,
                                          edge_polyline)
    from spline_surface.core.extend import extend_net
    flat = [bezier_curve("fb", [(10, 0, 0), (11, 0, 0), (12, 0, 0)]),
           bezier_curve("ft", [(10, 2, 0), (11, 2, 0), (12, 2, 0)]),
           bezier_curve("fl", [(10, 0, 0), (10, 1, 0), (10, 2, 0)]),
           bezier_curve("fr", [(12, 0, 0), (12, 1, 0), (12, 2, 0)])]
    select(flat)
    assert bpy.ops.spline_surface.coons() == {'FINISHED'}
    flat_patch = bpy.context.active_object
    net0, orders0 = world_net(flat_patch)
    # Straight target directly ahead of the U1 edge: solved length should
    # land the far edge flush on it.
    far0 = edge_polyline(net0, orders0, 'U1', 16)
    straight_target = far0 + np.array([2.0, 0.0, 0.0])
    length = _solve_extend_length(net0, orders0, 'U1', 'G1', 5, orders0, straight_target)
    def nearest_gap(pts, target):
        # Polylines can be sampled in opposite parameter order, so match
        # each point to its own nearest target point rather than by index.
        return np.linalg.norm(pts[:, None, :] - target[None, :, :], axis=2).min(axis=1)

    net, orders = extend_net(net0, 'U1', length, 'G1', 5, orders0)
    far = _extend_far_edge(net0, orders0, 'U1', net, orders)
    far_pts = edge_polyline(net, orders, far, 16)
    gap = nearest_gap(far_pts, straight_target).max()
    print("extend-to-target straight gap", gap)
    assert gap < 0.05, gap
    # Bowed target: solved length should settle inside the target's own
    # extent (a balanced compromise), not snap to one end of it.
    bow = far0 + np.array([2.0, 0.0, 0.0]) + 0.5 * np.sin(
        np.linspace(0, np.pi, len(far0)))[:, None] * np.array([1.0, 0.0, 0.0])
    length_bow = _solve_extend_length(net0, orders0, 'U1', 'G1', 5, orders0, bow)
    net_bow, orders_bow = extend_net(net0, 'U1', length_bow, 'G1', 5, orders0)
    far_bow = _extend_far_edge(net0, orders0, 'U1', net_bow, orders_bow)
    far_bow_pts = edge_polyline(net_bow, orders_bow, far_bow, 16)
    lo_x, hi_x = bow[:, 0].min(), bow[:, 0].max()
    print("extend-to-target bowed far-edge x mean", far_bow_pts[:, 0].mean(),
         "target x range", lo_x, hi_x)
    assert lo_x - 0.1 <= far_bow_pts[:, 0].mean() <= hi_x + 0.1
    # Target further than the length upper bound: solved length clamps to hi
    # instead of raising or returning something out of range.
    far_target = far0 + np.array([50.0, 0.0, 0.0])
    length_far = _solve_extend_length(net0, orders0, 'U1', 'G1', 5, orders0, far_target,
                                      lo=0.01, hi=5.0)
    assert 0.01 <= length_far <= 5.0
    # Target behind the source edge (wrong direction): solver should still
    # return something within [lo, hi], never negative or unbounded.
    behind_target = far0 - np.array([2.0, 0.0, 0.0])
    length_behind = _solve_extend_length(net0, orders0, 'U1', 'G1', 5, orders0, behind_target,
                                         lo=0.01, hi=5.0)
    assert 0.01 <= length_behind <= 5.0
    print("extend-to-target edge cases OK (far", length_far, "behind", length_behind, ")")

    # The modal path: click source edge then a target edge in one call, as
    # the "click again / click a different edge" confirm would do.
    select([flat_patch])
    assert bpy.ops.spline_surface.extend_edge(
        source=flat_patch.name, edge='U1', length=length, continuity='G1') == {'FINISHED'}

    # Neighbouring Coons patch sharing the hood's top curve: only G0 at first.
    roof = [c[2],
            bezier_curve("r_right", [(4, 3, 0), (4.2, 4.5, 0.2), (4, 6, 0)]),
            bezier_curve("r_top", [(0, 6, 0), (2, 6.2, 0.3), (4, 6, 0)]),
            bezier_curve("r_left", [(0, 3, 0), (-0.2, 4.5, 0.2), (0, 6, 0)])]
    select(roof)
    assert bpy.ops.spline_surface.coons() == {'FINISHED'}
    roof_patch = bpy.context.active_object
    settings = bpy.context.scene.spline_surface

    def check():
        select([roof_patch, patch])
        assert bpy.ops.spline_surface.check_continuity() == {'FINISHED'}
        rows = json.loads(settings.continuity_report)
        return {(r["a"], r["b"]): r["grade"] for r in rows}

    before = check()
    print("before", before)
    assert set(before.values()) <= {"G0", "G1"}, before
    select([patch, roof_patch])
    bpy.context.view_layer.objects.active = roof_patch
    assert bpy.ops.spline_surface.match_edge(continuity='G2') == {'FINISHED'}
    after = check()
    print("after", after)
    assert set(after.values()) == {"G2"}, after
    # Regenerate keeps the match.
    select([roof_patch])
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    assert set(check().values()) == {"G2"}

    # Icons used by the panel exist.
    icons = bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys()
    from spline_surface.ui import GRADE_ICONS
    for name in list(GRADE_ICONS.values()) + ['VIEWZOOM', 'AUTOMERGE_ON', 'MOD_SOLIDIFY', 'SNAP_ON']:
        assert name in icons, name

    # Loft through 4 shuffled, mixed-direction sections.
    bpy.context.scene.spline_surface.degree = 4
    xs = [(30, 0.2), (31, 0.6), (32.2, 0.5), (33, 0.1)]
    secs = []
    for k, (x, bmp) in enumerate(xs):
        pts = [(x, -1, 0), (x, 0, 2 * bmp), (x, 1, 0)]
        secs.append(bezier_curve("sec%d" % k, pts[::-1] if k % 2 else pts))
    select([secs[2], secs[0], secs[3], secs[1]])
    assert bpy.ops.spline_surface.loft() == {'FINISHED'}
    loft = bpy.context.active_object
    assert json.loads(loft[spline_surface.operators.META_KEY])["sources"] in (
        ["sec0", "sec1", "sec2", "sec3"], ["sec3", "sec2", "sec1", "sec0"])
    sp = loft.data.splines[0]
    assert (sp.point_count_u, sp.point_count_v) == (5, 4), (sp.point_count_u, sp.point_count_v)
    verts = evaluated_verts(loft)
    for s_ob in secs:
        mid = s_ob.data.splines[0].bezier_points[1].co
        d = np.linalg.norm(verts - np.array(mid), axis=1).min()
        assert d < 0.05, (s_ob.name, d)  # bounded by the section fit error
    secs[1].data.splines[0].bezier_points[1].co.z = 2.0
    select([loft])
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    assert evaluated_verts(loft)[:, 2].max() > 1.5

    # Hand extrusion in edit mode turns a Bezier patch into a 2-span NURBS.
    sel = [c[0], c[1], c[2], c[3]]
    select(sel)
    bpy.context.scene.spline_surface.degree = 5
    assert bpy.ops.spline_surface.coons() == {'FINISHED'}
    hand = bpy.context.active_object
    sp = hand.data.splines[0]
    nu, nv = sp.point_count_u, sp.point_count_v
    for idx, pt in enumerate(sp.points):
        pt.select = idx // nu == nv - 1
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.extrude(mode='TRANSLATION')
    bpy.ops.transform.translate(value=(0.0, 1.0, 0.3))
    bpy.ops.object.mode_set(mode='OBJECT')
    sp = hand.data.splines[0]
    print("extruded", sp.point_count_u, sp.point_count_v, sp.order_u, sp.order_v)
    assert (sp.point_count_u, sp.point_count_v) == (6, 7)

    # Our B-spline evaluation must agree with Blender's (same knot vector).
    from spline_surface.core import nurbs
    hnet, horders = world_net(hand)
    assert horders == (6, 6)
    ours = nurbs.eval_surface(hnet, horders, np.linspace(0, 1, 9), np.linspace(0, 1, 9)).reshape(-1, 3)
    verts = evaluated_verts(hand)
    d = np.array([np.linalg.norm(verts - p, axis=1).min() for p in ours])
    print("multispan blender-vs-core", d.max())
    assert d.max() < 0.01, d.max()

    # Extend from the new edge, check G2, then regenerate warns about hand edits.
    select([hand])
    assert bpy.ops.spline_surface.extend_edge(edge='V1', length=0.3, continuity='G2') == {'FINISHED'}
    hext = bpy.context.active_object
    select([hand, hext])
    assert bpy.ops.spline_surface.check_continuity() == {'FINISHED'}
    rows = json.loads(bpy.context.scene.spline_surface.continuity_report)
    print("multispan seam", rows[0]["grade"], rows[0]["angle"], rows[0]["curvature"])
    assert rows[0]["grade"] == "G2"
    # report with nothing selected scans all visible patches, worst grade first, and jumps to a seam
    [o.select_set(False) for o in bpy.context.scene.objects]
    assert bpy.ops.spline_surface.check_continuity() == {'FINISHED'}
    allrows = json.loads(bpy.context.scene.spline_surface.continuity_report)
    order = {'open': -1, 'G0': 0, 'G1': 1, 'G2': 2}
    assert len(allrows) >= 1
    assert [order[r["grade"]] for r in allrows] == sorted(order[r["grade"]] for r in allrows)
    assert bpy.ops.spline_surface.seam_jump(a=allrows[0]["a"], b=allrows[0]["b"]) == {'FINISHED'}
    assert {o.name for o in bpy.context.selected_objects} == {allrows[0]["a"], allrows[0]["b"]}
    print("continuity report all", len(allrows), allrows[0]["grade"])
    meta = json.loads(hand[spline_surface.operators.META_KEY])
    assert spline_surface.operators.hand_edited(hand, meta)
    # Matching a hand-edited patch keeps the hand shape as its recipe.
    select([hext, hand])
    bpy.context.view_layer.objects.active = hand
    assert bpy.ops.spline_surface.match_edge(continuity='G1') == {'FINISHED'}
    assert json.loads(hand[spline_surface.operators.META_KEY])["method"] == "raw"
    select([hand])
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    assert hand.data.splines[0].point_count_v == 7

    # Hand-drawn style: every corner open by ~8% of the size.
    gap = [bezier_curve("g0", [(20.3, 0, 0), (22, -0.3, 0.3), (23.7, 0, 0)]),
           bezier_curve("g1", [(24, 0.3, 0), (24.2, 1.5, 0.2), (24, 2.7, 0)]),
           bezier_curve("g2", [(23.7, 3, 0), (22, 3.2, 0.4), (20.3, 3, 0)]),
           bezier_curve("g3", [(20, 2.7, 0), (19.8, 1.5, 0.2), (20, 0.3, 0)])]
    select(gap)
    assert bpy.ops.spline_surface.coons() == {'FINISHED'}
    select(gap)
    assert bpy.ops.spline_surface.snap_ends() == {'FINISHED'}
    a = gap[0].matrix_world @ gap[0].data.splines[0].bezier_points[-1].co
    b = gap[1].matrix_world @ gap[1].data.splines[0].bezier_points[0].co
    assert (a - b).length < 1e-6, (a - b).length

    # Open loop is rejected.
    bad = [bezier_curve("b0", [(10, 0, 0), (11, 0, 0)]),
           bezier_curve("b1", [(11, 0, 0), (11, 1, 0)]),
           bezier_curve("b2", [(11, 1, 0), (10, 1, 0)]),
           bezier_curve("b3", [(10, 1, 0), (10, 0.5, 0)])]
    select(bad)
    try:
        r = bpy.ops.spline_surface.coons()
    except RuntimeError:
        r = {'CANCELLED'}
    assert r == {'CANCELLED'}
    # starter primitives: build, extend G2, regenerate keeps the shape
    from spline_surface.core.continuity import find_shared_edge, grade, measure
    for shape, nu in (('PLANE', 4), ('ARC', 6), ('DOME', 8)):
        bpy.context.scene.cursor.location = (10, 0, 0)
        assert bpy.ops.spline_surface.add_patch(shape=shape, points_u=nu, points_v=5) == {'FINISHED'}
        base = bpy.context.active_object
        net, orders = world_net(base)
        assert net.shape == (nu, 5, 3) and orders == (min(6, nu), 5), (net.shape, orders)
        assert abs(net[..., 0].mean() - 10) < 1e-6
        assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
        assert np.allclose(world_net(base)[0], net, atol=1e-5)
        assert bpy.ops.spline_surface.extend_edge(edge='U1', continuity='G2') == {'FINISHED'}
        ext = bpy.context.active_object
        na, oa = world_net(base)
        nb, ob_ = world_net(ext)
        ea, eb, rev, gap = find_shared_edge(na, nb, 1e-4, oa, ob_)
        g = grade(measure(na, nb, ea, eb, rev, oa, ob_))
        print("primitive", shape, g)
        assert g == 'G2', g
    # Add Boundary Curve: a true straight line and a true circular arc,
    # usable directly as patch boundaries.
    from spline_surface.bridge.read_curves import read_polyline
    bpy.context.scene.cursor.location = (10, 20, 0)
    assert bpy.ops.spline_surface.add_boundary_curve(shape='LINE', length=4.0) == {'FINISHED'}
    line = bpy.context.active_object
    dg = bpy.context.evaluated_depsgraph_get()
    pl = read_polyline(line, dg, 32)
    d = pl[-1] - pl[0]
    straight = np.linalg.norm(np.cross(pl - pl[0], d / np.linalg.norm(d)), axis=1).max()
    print("boundary line length", np.linalg.norm(d), "straightness", straight)
    assert abs(np.linalg.norm(d) - 4.0) < 1e-5 and straight < 1e-6
    assert bpy.ops.spline_surface.add_boundary_curve(
        shape='ARC', radius=2.0, angle=180.0, segments=2) == {'FINISHED'}
    arc = bpy.context.active_object
    pl = read_polyline(arc, dg, 64)
    radii = np.linalg.norm(pl - np.array(bpy.context.scene.cursor.location), axis=1)
    print("boundary arc radius range", radii.min(), radii.max())
    assert abs(radii.max() - 2.0) < 0.01 and radii.min() > 2.0 - 0.02
    # corner fill: two extensions off adjacent edges leave a notch; fill it G2
    bpy.context.scene.cursor.location = (0, 30, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME', points_u=6, points_v=10)
    base = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(edge='U1', continuity='G2')
    e1 = bpy.context.active_object
    _make = bpy.context.view_layer.objects
    _make.active = base
    bpy.ops.spline_surface.extend_edge(edge='V1', continuity='G2')
    e2 = bpy.context.active_object
    from spline_surface.operators import surface_edges
    from spline_surface.core.corner import corner_net
    na, oa = world_net(e1)
    nb, ob_ = world_net(e2)
    # find the pair of open edges sharing the base's corner
    pairs = [(x, y) for x in ('U0', 'U1', 'V0', 'V1') for y in ('U0', 'U1', 'V0', 'V1')]
    ok = None
    for x, y in pairs:
        try:
            corner_net(na, oa, x, nb, ob_, y, tol=1e-4)
        except ValueError:
            continue
        free = surface_edges(bpy.context)
        if (e1.name, x) in free and (e2.name, y) in free:
            ok = (x, y)
            break
    assert ok, "no corner pair found"
    r = bpy.ops.spline_surface.corner_fill(source_a=e1.name, edge_a=ok[0],
                                           source_b=e2.name, edge_b=ok[1], continuity='G2')
    assert r == {'FINISHED'}
    fill = bpy.context.active_object
    nf, of = world_net(fill)
    for src in (e1, e2):
        ns, os_ = world_net(src)
        sh = find_shared_edge(ns, nf, 1e-4, os_, of)
        assert sh is not None
        m = measure(ns, nf, sh[0], sh[1], sh[2], os_, of)
        print("corner fill", src.name, grade(m), m)
        # G2 along the seam; only the sample next to the collapsed corner may drift
        assert m["gap"] < 1e-6 and m["angle"] < 0.05 and m["curvature"] < 0.05, m
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    assert np.allclose(world_net(fill)[0], nf, atol=1e-5)
    # fill hole: ring of 8 patches sampled from a wavy surface, fill the middle
    from spline_surface.bridge.write_surface import build_surface_data
    from spline_surface.operators import store_meta
    from spline_surface.core.bezier import bernstein
    t = np.linspace(0, 1, 15)
    pinv = np.linalg.pinv(bernstein(5, t))
    ring = {}
    for i in range(3):
        for j in range(3):
            if (i, j) == (1, 1):
                continue
            X = 40 + i + t[:, None] + 0 * t
            Y = j + 0 * t[:, None] + t
            Z = 0.4 * np.sin(1.3 * X) * np.cos(0.9 * Y)
            net = np.stack([pinv @ c @ pinv.T for c in (X, Y, Z)], axis=-1)
            cu = build_surface_data("Ring%d%d" % (i, j), net, 12, (6, 6))
            ob = bpy.data.objects.new(cu.name, cu)
            bpy.context.collection.objects.link(ob)
            store_meta(ob, {"method": "raw", "sources": [], "net": net.tolist(), "orders": [6, 6]})
            ring[(i, j)] = ob
    picks = "%s:V1;%s:V0;%s:U0;%s:U1" % (ring[(1, 0)].name, ring[(1, 2)].name,
                                         ring[(2, 1)].name, ring[(0, 1)].name)
    assert bpy.ops.spline_surface.fill_hole(picks=picks, continuity='G2') == {'FINISHED'}
    fill = bpy.context.active_object
    nf, of = world_net(fill)
    for key in ((1, 0), (1, 2), (2, 1), (0, 1)):
        ns, os_ = world_net(ring[key])
        sh = find_shared_edge(ns, nf, 1e-4, os_, of)
        m = measure(ns, nf, sh[0], sh[1], sh[2], os_, of)
        print("fill hole", key, grade(m), m)
        assert m["gap"] < 1e-5 and m["angle"] < 0.5, m
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    # bridge: two separate starter patches (different point counts) joined G2
    bpy.context.scene.cursor.location = (0, 60, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME', points_u=6, points_v=6)
    pa = bpy.context.active_object
    bpy.context.scene.cursor.location = (0, 64, 0.5)
    bpy.ops.spline_surface.add_patch(shape='ARC', points_u=8, points_v=6, bulge=-0.3)
    pb = bpy.context.active_object
    for flip in (False, True):
        r = bpy.ops.spline_surface.bridge_edges(source_a=pa.name, edge_a='V1', source_b=pb.name,
                                                edge_b='V0', continuity='G2', flip=flip)
        assert r == {'FINISHED'}
        br = bpy.context.active_object
        nf, of = world_net(br)
        for src in (pa, pb):
            ns, os_ = world_net(src)
            sh = find_shared_edge(ns, nf, 1e-4, os_, of)
            m = measure(ns, nf, sh[0], sh[1], sh[2], os_, of)
            print("bridge", flip, src.name, grade(m), of, nf.shape)
            assert grade(m) == 'G2', m
        assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
        assert np.allclose(world_net(br)[0], nf, atol=1e-5)
    # live update: move the base, the extension chain follows and stays G2
    bpy.context.scene.cursor.location = (0, 90, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME')
    base = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(edge='U1', continuity='G2')
    e1 = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(edge='U1', continuity='G2')
    e2 = bpy.context.active_object
    vl = bpy.context.view_layer
    vl.update()

    def seam(a, b):
        na, oa = world_net(a)
        nb, ob_ = world_net(b)
        sh = find_shared_edge(na, nb, 1e-4, oa, ob_)
        return None if sh is None else grade(measure(na, nb, *sh[:3], oa, ob_))

    base.location.z += 1.0
    base.rotation_euler.x += 0.3
    vl.update()
    print("live", seam(base, e1), seam(e1, e2))
    assert seam(base, e1) == 'G2' and seam(e1, e2) == 'G2'
    # hand-edit e1: it and everything after it is left alone
    e1.location.y += 0.5
    vl.update()
    before = world_net(e2)[0].copy()
    base.location.z += 0.5
    vl.update()
    assert seam(base, e1) is None
    assert np.allclose(world_net(e2)[0], before)
    assert "hand edited" in bpy.context.scene.spline_surface.live_note
    # select group + forced regenerate puts everything back on the base
    for o in bpy.context.selected_objects:
        o.select_set(False)
    e2.select_set(True)
    bpy.ops.spline_surface.select_group()
    assert {base, e1, e2} <= set(bpy.context.selected_objects)
    assert bpy.ops.spline_surface.regenerate_all(force=True) == {'FINISHED'}
    print("forced", seam(base, e1), seam(e1, e2))
    assert seam(base, e1) == 'G2' and seam(e1, e2) == 'G2'
    # live update from curves: editing a boundary curve reshapes its Coons patch
    o = (0, 120, 0)
    cs = [bezier_curve("lb", [(0, 0, 0), (2, 0, 0.3), (4, 0, 0)]),
          bezier_curve("lr", [(4, 0, 0), (4, 1.5, 0.3), (4, 3, 0)]),
          bezier_curve("lt", [(0, 3, 0), (2, 3, 0.3), (4, 3, 0)]),
          bezier_curve("ll", [(0, 0, 0), (0, 1.5, 0.3), (0, 3, 0)])]
    for cobj in cs:
        cobj.location = o
    select(cs)
    assert bpy.ops.spline_surface.coons() == {'FINISHED'}
    cp = bpy.context.active_object
    vl.update()
    before = world_net(cp)[0].copy()
    cs[0].data.splines[0].bezier_points[1].co.z = 1.2
    vl.update()
    after = world_net(cp)[0]
    print("curve live", float(np.abs(after - before).max()))
    assert np.abs(after - before).max() > 0.3
    # cage: deform a group with a lattice, bake, seams come back G2
    bpy.context.scene.cursor.location = (0, 150, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME')
    cbase = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(edge='U1', continuity='G2')
    c1 = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(edge='U1', continuity='G2')
    c2 = bpy.context.active_object
    vl.update()
    for o in bpy.context.selected_objects:
        o.select_set(False)
    cbase.select_set(True)
    # Cancel: the cage and its modifiers go away, geometry is untouched.
    assert bpy.ops.spline_surface.cage_add() == {'FINISHED'}
    cancel_cage_name = bpy.context.active_object.name
    before_cancel = world_net(cbase)[0].copy()
    assert bpy.ops.spline_surface.cage_cancel() == {'FINISHED'}
    assert cancel_cage_name not in bpy.data.objects
    assert not any(m.name == "SS_Cage" for m in cbase.modifiers)
    assert np.allclose(world_net(cbase)[0], before_cancel)
    print("cage cancel OK")

    for o in bpy.context.selected_objects:
        o.select_set(False)
    cbase.select_set(True)
    assert bpy.ops.spline_surface.cage_add() == {'FINISHED'}
    cage = bpy.context.active_object
    assert cage.type == 'LATTICE' and all(
        any(m.name == "SS_Cage" for m in o.modifiers) for o in (cbase, c1, c2))
    for p in cage.data.points:
        if p.co[0] > 0.1:
            p.co_deform.z += 0.8  # lift one end of the cage
    before = world_net(cbase)[0].copy()
    assert bpy.ops.spline_surface.cage_apply() == {'FINISHED'}
    assert "SS_Cage" not in bpy.data.objects
    assert np.abs(world_net(cbase)[0] - before).max() > 0.05
    print("cage", seam(cbase, c1), seam(c1, c2))
    assert seam(cbase, c1) == 'G2' and seam(c1, c2) == 'G2'
    assert bpy.ops.spline_surface.regenerate_all() == {'FINISHED'}
    assert seam(cbase, c1) == 'G2'
    # weld: the three patches become one mesh with merged seams
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in (cbase, c1, c2):
        o.select_set(True)
    bpy.context.scene.spline_surface.mesh_resolution = 24
    assert bpy.ops.spline_surface.to_mesh(weld=True) == {'FINISHED'}
    wm = bpy.context.active_object
    print("weld verts", len(wm.data.vertices), len(wm.data.polygons))
    def expected_mesh(objs, res=24):
        """(verts, faces) To Mesh should give: each patch cut in proportion to
        its size (the longest side gets `res`), coincident points welded."""
        from spline_surface.core.meshing import segment_counts
        nets = [world_net(o) for o in objs]
        counts = segment_counts(nets, res)
        assert max(max(c) for c in counts) == res
        pts = []
        for (net, o), (nu, nv) in zip(nets, counts):
            U, V = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
            pts.append(nurbs.eval_points(net, o, U.ravel(), V.ravel()))
        pts = np.concatenate(pts)
        unique = np.unique(np.round(pts / np.ptp(pts, axis=0).max(), 4), axis=0)
        return len(unique), sum(nu * nv for nu, nv in counts)

    verts, faces = expected_mesh((cbase, c1, c2))
    print("weld expected", verts, faces)
    assert len(wm.data.vertices) == verts and len(wm.data.polygons) == faces
    # extend by name, independent of selection / active object
    for o in bpy.context.selected_objects:
        o.select_set(False)
    vl.objects.active = None
    assert bpy.ops.spline_surface.extend_edge(source=cbase.name, edge='V1') == {'FINISHED'}
    ext = bpy.context.active_object
    assert ext.name.startswith(cbase.name + "_ext") and seam(cbase, ext) == 'G2'
    # bridge to a curve: G2 at the patch, on the curve at the other side
    from spline_surface.operators import edge_polyline
    from spline_surface.bridge.read_curves import read_polyline
    bpy.context.scene.cursor.location = (0, 200, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME')
    bp = bpy.context.active_object
    guide = bezier_curve("guide", [(1, 4, 1.0), (0, 4.5, 1.6), (-1, 4, 1.0)])  # reversed on purpose
    guide.location = (0, 200, 0)
    vl.update()
    for mode in ('SMOOTH', 'STRAIGHT'):
        r = bpy.ops.spline_surface.bridge_edges(source_a=bp.name, edge_a='V1', source_b=guide.name,
                                                edge_b='CURVE', curve_tangent=mode)
        assert r == {'FINISHED'}, r
        bc = bpy.context.active_object
        print("curve bridge", mode, seam(bp, bc))
        assert seam(bp, bc) == 'G2'
        net, orders = world_net(bc)
        far = min((edge_polyline(net, orders, e, 64) for e in ('U0', 'U1', 'V0', 'V1')),
                  key=lambda pl: np.linalg.norm(pl.mean(0) - np.array([0, 204.3, 1.3])))
        curve = read_polyline(guide, bpy.context.evaluated_depsgraph_get(), 200)
        dev = np.linalg.norm(far[:, None] - curve[None], axis=2).min(axis=1).max()
        print("curve deviation", dev)
        assert dev < 0.01, dev
    # moving the curve drags the (last) bridge along through live update
    before = world_net(bc)[0].copy()
    guide.location.z += 0.5
    vl.update()
    assert np.abs(world_net(bc)[0] - before).max() > 0.3
    assert seam(bp, bc) == 'G2'
    # mirror: half body on x >= 0, mirrored across x = 0 with a G2 centre
    bpy.context.scene.cursor.location = (1, 300, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME')  # spans x 0..2, U0 edge on x = 0
    half = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(edge='V1', continuity='G2')
    hext = bpy.context.active_object
    vl.update()
    print("mirror before", seam(half, hext))
    for o in bpy.context.selected_objects:
        o.select_set(False)
    half.select_set(True)
    assert bpy.ops.spline_surface.mirror_group() == {'FINISHED'}
    mh = bpy.data.objects[half.name + "_mirror"]
    me_ = bpy.data.objects[hext.name + "_mirror"]
    print("mirror seams", seam(half, mh), seam(hext, me_), seam(half, hext), seam(mh, me_))
    assert seam(half, mh) == 'G2' and seam(hext, me_) == 'G2'
    assert seam(half, hext) in ('G2', 'G1') and seam(mh, me_) in ('G2', 'G1')
    # running it again does not duplicate
    half.select_set(True)
    assert bpy.ops.spline_surface.mirror_group() == {'CANCELLED'}
    # the mirror follows edits of the original
    half.scale.z = 1.5
    vl.update()
    assert np.allclose(world_net(mh)[0][..., 2], world_net(half)[0][::-1][..., 2])
    assert seam(half, mh) == 'G2'
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in (half, hext, mh, me_):
        o.select_set(True)
    assert bpy.ops.spline_surface.to_mesh(weld=True) == {'FINISHED'}
    print("mirror weld", len(bpy.context.active_object.data.vertices))
    assert len(bpy.context.active_object.data.vertices) == expected_mesh((half, hext, mh, me_))[0]
    # split: cut a mirrored patch with extensions; links follow the pieces
    bpy.context.scene.cursor.location = (1, 400, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME')
    sp_base = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(source=sp_base.name, edge='U1')
    far = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(source=sp_base.name, edge='V1')
    side = bpy.context.active_object
    for o in bpy.context.selected_objects:
        o.select_set(False)
    sp_base.select_set(True)
    assert bpy.ops.spline_surface.mirror_group() == {'FINISHED'}
    vl.update()
    r = bpy.ops.spline_surface.split_patch(source=sp_base.name, direction='U', t=0.4)
    assert r == {'FINISHED'}
    piece = bpy.data.objects[sp_base.name + "_split"]
    meta_far = json.loads(far["spline_surface"])
    meta_side = json.loads(side["spline_surface"])
    print("split", seam(sp_base, piece), seam(piece, far), meta_far["sources"], meta_side["method"])
    assert seam(sp_base, piece) == 'G2' and seam(piece, far) == 'G2'
    assert meta_far["sources"] == [piece.name] and meta_side["method"] == "raw"
    pm = bpy.data.objects[piece.name + "_mirror"]
    bm_ = bpy.data.objects[sp_base.name + "_mirror"]
    print("split mirror", seam(sp_base, bm_), seam(piece, pm), seam(bm_, pm))
    assert seam(sp_base, bm_) == 'G2' and seam(bm_, pm) == 'G2'
    # the far extension still follows the new piece
    piece.location.z += 0.3
    vl.update()
    assert seam(piece, far) == 'G2'
    # Split Patch: direction='V' and snap=False (fitted at an arbitrary t,
    # not a span boundary), both untested by the direction='U'/snap=True
    # case above.
    bpy.context.scene.cursor.location = (1, 600, 0)
    bpy.ops.spline_surface.add_patch(shape='PLANE', points_u=6, points_v=9, order=5)
    multi = bpy.context.active_object
    net_multi, orders_multi = world_net(multi)
    print("split source spans", orders_multi, net_multi.shape)
    before_v = world_net(multi)[0].copy()
    rv = bpy.ops.spline_surface.split_patch(
        source=multi.name, direction='V', t=0.37, snap=False)
    assert rv == {'FINISHED'}
    piece_v = bpy.data.objects[multi.name + "_split"]
    print("split V snap=False", seam(multi, piece_v))
    assert seam(multi, piece_v) == 'G2'
    # The source keeps its own control net span structure (not fitted away);
    # only its V extent shrinks to the cut.
    after_a, after_orders = world_net(multi)
    assert after_orders[1] <= orders_multi[1]
    assert not np.allclose(after_a[..., 1].max(), before_v[..., 1].max())

    # N-sided holes: 5 patch edges, and 2 patch edges + 1 curve (triangle)
    def ring(n_sides, at):
        f = lambda x, y: 0.15 * x * x - 0.1 * x * y + 0.12 * y * y
        ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + 0.3
        c = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        tt = np.linspace(0, 1, 15)
        pinv = np.linalg.pinv(bernstein(5, tt))
        obs = []
        for i in range(n_sides):
            a, b = c[i], c[(i + 1) % n_sides]
            o = np.array([b[1] - a[1], a[0] - b[0]])
            o = o / np.linalg.norm(o) * 0.6
            U, V = tt[:, None], tt[None, :]
            X = (1 - V) * ((1 - U) * a[0] + U * b[0]) + V * ((1 - U) * (a[0] + o[0]) + U * (b[0] + o[0]))
            Y = (1 - V) * ((1 - U) * a[1] + U * b[1]) + V * ((1 - U) * (a[1] + o[1]) + U * (b[1] + o[1]))
            net = np.stack([pinv @ C @ pinv.T for C in (X, Y, f(X, Y))], axis=-1) + np.array(at)
            cu = build_surface_data("Ring", net, 12, (6, 6))
            ob = bpy.data.objects.new(cu.name, cu)
            bpy.context.collection.objects.link(ob)
            store_meta(ob, {"method": "raw", "sources": [], "net": net.tolist(), "orders": [6, 6]})
            obs.append(ob)
        return obs

    five = ring(5, (0, 500, 0))
    picks = ";".join("%s:V0" % o.name for o in five)
    assert bpy.ops.spline_surface.fill_hole(picks=picks) == {'FINISHED'}
    parts5 = list(bpy.context.selected_objects)
    print("five parts", len(parts5))
    assert len(parts5) == 5
    tri = ring(3, (0, 520, 0))
    edge3 = edge_polyline(*world_net(tri[2]), 'V0', 9)
    cv = bezier_curve("tri_side", [tuple(p) for p in edge3[::4]])
    bpy.data.objects.remove(tri[2])
    picks = "%s:V0;%s:V0;%s:CURVE" % (tri[0].name, tri[1].name, cv.name)
    assert bpy.ops.spline_surface.fill_hole(picks=picks) == {'FINISHED'}
    parts3 = list(bpy.context.selected_objects)
    assert len(parts3) == 3
    vl.update()
    before = [world_net(o)[0].copy() for o in parts3]
    cv.data.splines[0].bezier_points[1].co.z += 0.3  # bend the middle, keep the ends
    vl.update()
    moved = [np.abs(world_net(o)[0] - b).max() for o, b in zip(parts3, before)]
    print("triangle parts follow curve", np.round(moved, 3))
    assert max(moved) > 0.1
    vl.objects.active = parts3[0]
    assert bpy.ops.spline_surface.regenerate() == {'FINISHED'}
    # seam colours: every seam of the scene graded; known pairs come out right
    from spline_surface import seams as seam_mod
    found = {tuple(sorted(pair)): g for g, _, _, pair in seam_mod.compute_seams(bpy.context)}
    print("seam colours", len(found), sum(1 for g in found.values() if g == 'G2'))
    assert found[tuple(sorted((half.name, mh.name)))] == 'G2'
    assert found[tuple(sorted((piece.name, far.name)))] == 'G2'
    assert bpy.ops.spline_surface.seam_colors() == {'FINISHED'}
    assert seam_mod.counts().get('G2', 0) >= 2
    assert bpy.ops.spline_surface.seam_colors() == {'FINISHED'}
    assert seam_mod._state["handle"] is None
    # offset: linked copy at a constant distance, follows the source
    for o in bpy.context.selected_objects:
        o.select_set(False)
    half.select_set(True)
    assert bpy.ops.spline_surface.offset_surface(distance=-0.1) == {'FINISHED'}
    off = bpy.context.active_object
    on, oo = world_net(off)
    hn, ho = world_net(half)
    g = np.linspace(0.1, 0.9, 9)
    pts = nurbs_eval(on, oo, g)
    dense = nurbs_eval(hn, ho, np.linspace(0, 1, 150))
    d = np.linalg.norm(pts[:, None] - dense[None], axis=2).min(axis=1)
    print("offset distance", d.min(), d.max())
    assert np.abs(d - 0.1).max() < 5e-3
    before = on.copy()
    half.location.z += 0.4
    vl.update()
    assert np.abs(world_net(off)[0] - before).max() > 0.3
    # curve network: 3 x 4 crossing curves -> 6 smoothly joined cells, live on edits
    fz = lambda x, y: 0.15 * x * x - 0.1 * x * y + 0.12 * y * y
    net_curves = []
    for y in (0.0, 1.0, 2.0):
        xs = np.linspace(-0.3, 3.3, 5)
        net_curves.append(bezier_curve("nu", [(x, y, fz(x, y) - 40) for x in xs]))
    for x in (0.0, 1.0, 1.7, 3.0):
        ys = np.linspace(-0.2, 2.2, 5)
        net_curves.append(bezier_curve("nv", [(x, y, fz(x, y) - 40) for y in ys]))
    for o in bpy.context.selected_objects:
        o.select_set(False)
    select(net_curves)
    assert bpy.ops.spline_surface.curve_network() == {'FINISHED'}
    cells = list(bpy.context.selected_objects)
    assert len(cells) == 6
    grades = []
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            g = seam(cells[a], cells[b])
            if g is not None:
                grades.append(g)
    print("network seams", grades)
    assert len(grades) == 7 and all(g == 'G2' for g in grades), grades
    vl.update()
    before = [world_net(o)[0].copy() for o in cells]
    bp = net_curves[0].data.splines[0].bezier_points[2]
    bp.co.z += 0.02  # small: the curves must still cross
    vl.update()
    moved = max(np.abs(world_net(o)[0] - b).max() for o, b in zip(cells, before))
    print("network follows curve", moved)
    assert moved > 0.005
    # curve grid starter -> curve network, all inner seams G2
    from spline_surface import zebra as zebra_mod
    bpy.context.scene.cursor.location = (0, -60, 0)
    assert bpy.ops.spline_surface.add_curve_grid() == {'FINISHED'}
    assert len(bpy.context.selected_objects) == 7
    assert bpy.ops.spline_surface.curve_network() == {'FINISHED'}
    grid_cells = list(bpy.context.selected_objects)
    gs = [seam(a, b) for k, a in enumerate(grid_cells) for b in grid_cells[k + 1:]]
    gs = [g for g in gs if g is not None]
    print("curve grid seams", gs)
    assert len(gs) == 7 and all(g == 'G2' for g in gs)
    # zebra: one click stripes every visible patch, new patches too, and back off
    sc = bpy.context.scene.spline_surface
    if sc.zebra_on:
        bpy.ops.spline_surface.zebra_toggle()
    for o in bpy.context.selected_objects:
        o.select_set(False)
    assert bpy.ops.spline_surface.zebra_toggle() == {'FINISHED'} and sc.zebra_on
    surfs = [o for o in bpy.context.visible_objects if o.type == 'SURFACE']
    assert surfs and all(zebra_mod.is_on(o) for o in surfs)
    bpy.ops.spline_surface.extend_edge(source=grid_cells[0].name, edge='U0')
    newp = bpy.context.active_object
    vl.update()
    assert zebra_mod.is_on(newp)
    assert bpy.ops.spline_surface.zebra_toggle() == {'FINISHED'} and not sc.zebra_on
    assert not any(zebra_mod.is_on(o) for o in bpy.data.objects if o.type == 'SURFACE')
    # a one-row "surface" (e.g. Add > Surface > NURBS Curve) is rejected, not a crash
    bpy.ops.surface.primitive_nurbs_surface_curve_add(location=(0, -90, 0))
    curve_surf = bpy.context.active_object
    try:
        world_net(curve_surf)
    except ValueError as e:
        print("one-row surface:", e)
    else:
        raise AssertionError("one-row surface accepted")
    from spline_surface.operators import surface_edges
    assert not any(k[0] == curve_surf.name for k in surface_edges(bpy.context))
    # ---- v0.21: Linked Edit, Edge Chain, Merge Patches
    from spline_surface import linked
    from spline_surface.core.continuity import find_shared_edge as fse, grade as grd, measure as msr
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    sc.live_update = False
    bpy.context.scene.cursor.location = (0, 0, 0)
    assert bpy.ops.spline_surface.add_patch(shape='DOME', points_u=6, points_v=6) == {'FINISHED'}
    A = bpy.context.active_object
    assert bpy.ops.spline_surface.extend_edge(source=A.name, edge='V1', continuity='G2') == {'FINISHED'}
    B = bpy.context.active_object

    def seam():
        a, oa = world_net(A)
        b, ob_ = world_net(B)
        sh = fse(b, a, 1e-4, ob_, oa)
        assert sh is not None, "seam opened"
        return grd(msr(b, a, *sh[:3], ob_, oa))

    assert seam() == 'G2'
    sc.linked_edit = True
    linked.reset()
    vl.update()
    linked.remember_all(bpy.context)
    select([A])
    sp = A.data.splines[0]
    nu = sp.point_count_u
    before_b = world_net(B)[0].copy()
    for i, p in enumerate(sp.points):   # lift the row next to the shared edge (v = max)
        if i // nu == sp.point_count_v - 2:
            p.co.z += 0.3
    A.data.update_tag()
    vl.update()
    linked.on_update({A.name}, bpy.context)  # the depsgraph handler may already have run it
    assert seam() == 'G2', seam()
    assert np.abs(world_net(B)[0] - before_b).max() > 0.01
    # Edge chain along the side of A and B
    from spline_surface.operators import surface_edges
    edges = surface_edges(bpy.context)
    side = next(e for e in ('U0', 'U1') if (A.name, e) in edges)
    assert bpy.ops.spline_surface.edge_chain(source=A.name, edge=side) == {'FINISHED'}
    cob = bpy.context.active_object
    segs = json.loads(cob[linked.CHAIN_KEY])["segments"]
    assert {s_["ob"] for s_ in segs} == {A.name, B.name}, segs
    for p in cob.data.splines[0].points:
        p.co.z += 0.2
    cob.data.update_tag()
    vl.update()
    linked.on_update({cob.name}, bpy.context)
    for o in (A, B):
        e = next(s_["edge"] for s_ in segs if s_["ob"] == o.name)
        pl = surface_edges(bpy.context, free_only=False)[(o.name, e)]
        curve = linked.chain_samples(cob)
        d = np.linalg.norm(pl[:, None] - curve[None], axis=2).min(1).max()
        assert d < 0.03, (o.name, d)  # seam re-matching may pull the corner a little
    assert seam() in ('G1', 'G2'), seam()
    sc.linked_edit = False
    # Merge
    select([A, B])
    assert bpy.ops.spline_surface.merge_patches() == {'FINISHED'}
    M = bpy.context.active_object
    assert M.name.startswith("Merged") and A.hide_get() and B.hide_get()
    mnet, mo = world_net(M)
    print("merged", mnet.shape, mo)
    # control point overlay: G2 seam pins three rows on each side, the rest is free
    from spline_surface import ctrlpoints
    bpy.context.scene.cursor.location = (0, 200, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME')
    cp_base = bpy.context.active_object
    bpy.ops.spline_surface.extend_edge(edge='U1', continuity='G2')
    cp_ext = bpy.context.active_object
    bpy.ops.object.select_all(action='DESELECT')
    cp_base.select_set(True)
    cp_ext.select_set(True)
    bpy.context.view_layer.update()
    ctrlpoints.show(bpy.context, True)
    pats = ctrlpoints.compute_patches(bpy.context)
    assert len(pats) == 2
    for net, mask in pats:
        assert mask.shape == net.shape[:2] and 0 < mask.sum() < mask.size
    free, locked = ctrlpoints.counts()
    print("control points free/pinned", free, locked)
    assert free > 0 and locked > 0
    ctrlpoints.show(bpy.context, False)
    assert ctrlpoints.counts() is None
    # soft pull (the maths the modal runs): the pinned rows stay, so the seam stays G2
    from spline_surface.core.softpull import soft_pull
    from spline_surface.live import _write_in_place
    masks = ctrlpoints.pin_masks(bpy.context, [cp_base, cp_ext])
    net0, o0, mask0 = masks[cp_ext.name]
    assert mask0.any() and not mask0.all()
    new, gain = soft_pull(net0, o0, mask0, (0.5, 0.7), (0.0, 0.0, 0.3), 0.3 * np.ptp(net0.reshape(-1, 3), axis=0).max())
    assert gain > 0 and np.abs(new - net0).max() > 0.05 and np.allclose(new[mask0], net0[mask0])
    assert _write_in_place(cp_ext, new, o0)
    bpy.context.view_layer.update()
    na, oa = world_net(cp_base)
    nb, ob2 = world_net(cp_ext)
    sh = find_shared_edge(na, nb, 1e-4, oa, ob2)
    g = grade(measure(na, nb, *sh[:3], oa, ob2))
    print("soft pull seam", g)
    assert g == 'G2'
    # control mesh: free vertices move the patch, pinned ones snap back, the seam stays G2
    from spline_surface import controlmesh
    cp = controlmesh.make_control_mesh(bpy.context, cp_ext)
    n_u, n_v = nb.shape[:2]
    mask = controlmesh._masks[cp.name]
    assert len(cp.data.vertices) == n_u * n_v and cp.vertex_groups["Pinned"]
    free = int(np.flatnonzero(~mask.ravel())[0])
    pin = int(np.flatnonzero(mask.ravel())[0])
    before = world_net(cp_ext)[0].copy()
    pin_co = tuple(cp.data.vertices[pin].co)
    cp.data.vertices[free].co.z += 0.4
    cp.data.vertices[pin].co.z += 0.4
    assert controlmesh.sync(bpy.context, cp) == "ok"
    after = world_net(cp_ext)[0]
    assert abs(after.reshape(-1, 3)[free, 2] - before.reshape(-1, 3)[free, 2] - 0.4) < 1e-5
    assert np.allclose(after[mask], before[mask], atol=1e-6)
    assert np.allclose(tuple(cp.data.vertices[pin].co), pin_co, atol=1e-6)  # snapped back
    # the same through Edit Mode (what the user does)
    bpy.ops.object.select_all(action='DESELECT')
    cp.select_set(True)
    bpy.context.view_layer.objects.active = cp
    bpy.ops.object.mode_set(mode='EDIT')
    import bmesh
    bm = bmesh.from_edit_mesh(cp.data)
    bm.verts.ensure_lookup_table()
    bm.verts[free].co.z += 0.3
    bm.verts[pin].co.z += 0.3
    bmesh.update_edit_mesh(cp.data)
    assert controlmesh.sync(bpy.context, cp) == "ok"
    bm = bmesh.from_edit_mesh(cp.data)
    bm.verts.ensure_lookup_table()
    assert abs(bm.verts[pin].co.z - pin_co[2]) < 1e-5
    bpy.ops.object.mode_set(mode='OBJECT')
    assert abs(world_net(cp_ext)[0].reshape(-1, 3)[free, 2] - before.reshape(-1, 3)[free, 2] - 0.7) < 1e-5
    na, oa = world_net(cp_base)
    nb, ob2 = world_net(cp_ext)
    sh = find_shared_edge(na, nb, 1e-4, oa, ob2)
    g2 = grade(measure(na, nb, *sh[:3], oa, ob2))
    print("control mesh seam", g2)
    assert g2 == 'G2'
    assert bpy.ops.spline_surface.control_mesh_done() == {'FINISHED'}
    assert controlmesh._control_mesh_of(cp_ext) is None
    # fair keeps the seam G2; refine keeps the shape
    bpy.ops.object.select_all(action='DESELECT')
    cp_ext.select_set(True)
    bpy.context.view_layer.objects.active = cp_ext
    before = world_net(cp_ext)[0]
    assert bpy.ops.spline_surface.fair() == {'FINISHED'}
    bpy.context.view_layer.update()
    na, oa = world_net(cp_base)
    nb, ob2 = world_net(cp_ext)
    sh = find_shared_edge(na, nb, 1e-4, oa, ob2)
    gf = grade(measure(na, nb, *sh[:3], oa, ob2))
    print("fair seam", gf)
    assert gf == 'G2'
    pts_before = nurbs_eval(*world_net(cp_ext), np.linspace(0, 1, 15))
    assert bpy.ops.spline_surface.refine(direction='BOTH') == {'FINISHED'}
    after = world_net(cp_ext)
    assert after[0].shape[0] > before.shape[0] and after[0].shape[1] > before.shape[1]
    print("refine", before.shape, "->", after[0].shape, "max dist",
          float(np.abs(nurbs_eval(*after, np.linspace(0, 1, 15)) - pts_before).max()))
    assert np.abs(nurbs_eval(*after, np.linspace(0, 1, 15)) - pts_before).max() < 1e-5

    def graded(a, b):
        """Continuity grade of the seam between two patch objects (None if apart)."""
        na, oa = world_net(a)
        nb, ob_ = world_net(b)
        sh = find_shared_edge(na, nb, 1e-4, oa, ob_)
        return None if sh is None else grade(measure(na, nb, *sh[:3], oa, ob_))

    # ---- v0.30-v0.32: Soft Pull from a line and from a control point keeps the seam G2
    from spline_surface.core.softpull import line_pull, point_pull
    masks = ctrlpoints.pin_masks(bpy.context, [cp_base, cp_ext])
    net0, o0, mask0 = masks[cp_ext.name]
    reach = 0.3 * np.ptp(net0.reshape(-1, 3), axis=0).max()
    i, j = np.argwhere(~mask0)[len(np.argwhere(~mask0)) // 2]
    for label, (new, gain) in (
            ("line", line_pull(net0, o0, mask0, 'U', 0.7, 0.5, (0.0, 0.0, 0.2), reach)),
            ("point", point_pull(net0, mask0, i, j, (0.0, 0.0, 0.2), 2.0, 2.0))):
        assert gain > 0 and np.allclose(new[mask0], net0[mask0]), label
        assert _write_in_place(cp_ext, new, o0)
        vl.update()
        print("soft pull", label, "seam", graded(cp_base, cp_ext))
        assert graded(cp_base, cp_ext) == 'G2', label
    assert _write_in_place(cp_ext, net0, o0)
    vl.update()

    # ---- v0.29: curvature comb and map switch on and off
    from spline_surface import curvature as curv
    bpy.ops.object.select_all(action='DESELECT')
    cp_ext.select_set(True)
    assert bpy.ops.spline_surface.curvature_comb() == {'FINISHED'}
    assert bpy.ops.spline_surface.curvature_map() == {'FINISHED'}
    assert curv._state["teeth"] and curv._state["tris"] is not None and curv.limit() > 0
    assert bpy.ops.spline_surface.curvature_comb() == {'FINISHED'}
    assert bpy.ops.spline_surface.curvature_map() == {'FINISHED'}
    assert curv._state["handle"] is None
    print("curvature overlays OK")

    # ---- v0.33: what the Extend arrow does - rewrite the length, rebuild in place
    from spline_surface.core.extend import length_handle
    from spline_surface.recipe import META_KEY
    from spline_surface import live as live_mod
    bpy.context.scene.cursor.location = (0, 320, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME')
    gb = bpy.context.active_object
    assert bpy.ops.spline_surface.extend_edge(edge='U1', length=0.3, continuity='G1') == {'FINISHED'}
    ge = bpy.context.active_object
    meta = json.loads(ge[META_KEY])
    src_net, src_orders = world_net(gb)
    base, direction, scale = length_handle(src_net, meta["edge"], src_orders)
    meta["length"] = 0.6
    ge[META_KEY] = json.dumps(meta)
    assert live_mod.rebuild(ge, bpy.context)[0] == "ok"
    gn, go = world_net(ge)
    mids = [nurbs.eval_points(gn, go, [a], [b])[0]
            for a, b in ((0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5))]
    tip = base + 0.6 * scale * direction
    print("extend arrow tip error", min(np.linalg.norm(m - tip) for m in mids))
    assert min(np.linalg.norm(m - tip) for m in mids) < 1e-4
    assert graded(gb, ge) in ('G1', 'G2')

    # ---- v0.33.2-3: To Mesh closes the seams around an N-sided fill (T-junctions)
    five = ring(5, (0, 540, 0))
    assert bpy.ops.spline_surface.fill_hole(picks=";".join("%s:V0" % o.name for o in five)) \
        == {'FINISHED'}
    parts5 = list(bpy.context.selected_objects)
    bpy.ops.object.select_all(action='DESELECT')
    for o in five + parts5:
        o.select_set(True)
    vl.objects.active = five[0]
    assert bpy.ops.spline_surface.to_mesh(weld=True) == {'FINISHED'}
    tm = bpy.context.active_object
    bm = bmesh.new()
    bm.from_mesh(tm.data)
    # Along the hole's rim (each ring patch's V0, corners left out) every mesh
    # vertex must be welded inside; a boundary vertex there is a crack.
    rim = np.concatenate([edge_polyline(*world_net(o), 'V0', 400)[8:-8] for o in five])
    cracked = sum(1 for v in bm.verts
                  if v.is_boundary and np.linalg.norm(rim - np.array(v.co), axis=1).min() < 1e-3)
    print("to mesh five-sided fill: open vertices on the seams", cracked, "faces", len(bm.faces))
    bm.free()
    assert cracked == 0

    # ---- v0.28-v0.35: Fillet three seams of a box corner, with a corner patch
    from spline_surface.filleting import find_fillet_seams
    sc.live_update = True  # the v0.21 section switched it off
    g6 = np.linspace(0, 1, 6)
    U6, V6 = g6[:, None] + 0 * g6, 0 * g6[:, None] + g6
    box = {}
    for name, net in (("BoxF", np.stack([U6, V6, 0 * U6], -1)),     # floor z = 0
                      ("BoxW1", np.stack([1 + 0 * U6, V6, U6], -1)),  # wall x = 1
                      ("BoxW2", np.stack([V6, 1 + 0 * U6, U6], -1))):  # wall y = 1
        net = net + np.array([0.0, 0.0, 300.0])  # clear of the rest of the scene
        cu = build_surface_data(name, net, 12, (6, 6))
        ob = bpy.data.objects.new(cu.name, cu)
        bpy.context.collection.objects.link(ob)
        store_meta(ob, {"method": "raw", "sources": [], "net": net.tolist(), "orders": [6, 6]})
        box[name] = ob
    vl.update()
    names = {o.name for o in box.values()}
    pairs = {k: v for k, v in find_fillet_seams(bpy.context, bpy.context.scene.spline_surface).items()
             if set(k) <= names}
    print("box corner seams", sorted(pairs))
    assert len(pairs) == 3
    picked = json.dumps([[a, v[0], b, v[1]] for (a, b), v in pairs.items()])
    assert bpy.ops.spline_surface.fillet(seams=picked, radius=0.2, continuity='G2',
                                         corners=True) == {'FINISHED'}
    made = list(bpy.context.selected_objects)
    bridges = [o for o in made if json.loads(o[META_KEY])["method"] == "blend"]
    corner_parts = [o for o in made if json.loads(o[META_KEY])["method"] == "fill"]
    print("fillet bridges", len(bridges), "corner parts", len(corner_parts))
    assert len(bridges) == 3 and corner_parts
    for br in bridges:
        a, b = json.loads(br[META_KEY])["sources"]
        assert graded(bpy.data.objects[a], br) == 'G2' and graded(bpy.data.objects[b], br) == 'G2'
    # Each bridge's end at the corner is closed off by the corner patch's rim
    # (a corner part covers part of an end, so this checks coverage, not a
    # whole shared edge; the angle there is covered by test_core).
    rims = np.concatenate([edge_polyline(*world_net(p), e, 400)
                           for p in corner_parts for e in ('U0', 'U1', 'V0', 'V1')])
    for br in bridges:
        bn, bo = world_net(br)
        covered = max(np.mean([np.linalg.norm(rims - q, axis=1).min() < 1e-3
                               for q in edge_polyline(bn, bo, e, 21)]) for e in ('U0', 'U1'))
        assert covered > 0.9, (br.name, covered)
    # v0.38: the trimmed patches keep their untrimmed recipe, so the fillet can be redone
    from spline_surface.filleting import fillet_members, set_fillet_radius
    floor_meta = json.loads(box["BoxF"][META_KEY])
    assert floor_meta["method"] == "fillet_trim" and floor_meta["base"]["method"] == "raw"
    fid = floor_meta["fillet"]["id"]
    assert sorted(ob.name for ob, _ in fillet_members(fid)) == sorted(names)
    assert all(json.loads(o[META_KEY]).get("fillet") == fid for o in bridges + corner_parts)
    vl.update()
    set_fillet_radius(bpy.context, fid, 0.3)
    vl.update()
    fx = world_net(box["BoxF"])[0]
    print("radius 0.3: floor reaches", fx[..., 0].max(), fx[..., 1].max())
    assert abs(fx[..., 0].max() - 0.7) < 1e-4 and abs(fx[..., 1].max() - 0.7) < 1e-4
    for br in bridges:
        a, b = json.loads(br[META_KEY])["sources"]
        assert graded(bpy.data.objects[a], br) == 'G2' and graded(bpy.data.objects[b], br) == 'G2'
    rims = np.concatenate([edge_polyline(*world_net(p), e, 400)
                           for p in corner_parts for e in ('U0', 'U1', 'V0', 'V1')])
    for br in bridges:
        bn, bo = world_net(br)
        assert max(np.mean([np.linalg.norm(rims - q, axis=1).min() < 1e-3
                            for q in edge_polyline(bn, bo, e, 21)]) for e in ('U0', 'U1')) > 0.9
    # Regenerate gives the same trimmed shape back
    assert live_mod.rebuild(box["BoxF"], bpy.context)[0] == "ok"
    assert np.abs(world_net(box["BoxF"])[0] - fx).max() < 1e-9
    # a radius that doesn't fit is refused and changes nothing
    try:
        set_fillet_radius(bpy.context, fid, 5.0)
    except ValueError as e:
        print("radius 5.0 refused:", e)
    else:
        raise AssertionError("an oversized radius must be refused")
    assert np.abs(world_net(box["BoxF"])[0] - fx).max() < 1e-12
    assert json.loads(box["BoxF"][META_KEY])["fillet"]["radius"] == 0.3

    # the trimmed floor moves: its bridges follow and stay G2. One evaluation
    # first, as the UI does after an operator, so Live Update has seen the floor.
    vl.update()
    before = [world_net(o)[0].copy() for o in bridges]
    box["BoxF"].location.z += 0.05
    vl.update()
    moved = [float(np.abs(world_net(o)[0] - b).max()) for o, b in zip(bridges, before)]
    print("fillet bridges follow the floor", np.round(moved, 3))
    assert max(moved) > 0.01
    for br in bridges:
        a, b = json.loads(br[META_KEY])["sources"]
        assert graded(bpy.data.objects[a], br) == 'G2' and graded(bpy.data.objects[b], br) == 'G2'

    # ---- v0.37: Network Check and curvature combs on curves
    from spline_surface import netcheck
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.scene.cursor.location = (0, 700, 0)
    assert bpy.ops.spline_surface.add_curve_grid() == {'FINISHED'}  # 3 + 4 curves
    grid_curves = list(bpy.context.selected_objects)
    assert bpy.ops.spline_surface.network_check() == {'FINISHED'}
    print("network check", netcheck.summary())
    assert netcheck.summary()[0].startswith("4 x 3") or netcheck.summary()[0].startswith("3 x 4")
    assert netcheck.summary()[0].endswith("ready")
    grid_curves[0].location.y += 50.0  # out of the grid: crosses nothing, crossings missing
    vl.update()
    netcheck.refresh(bpy.context)
    print("network check, one curve moved away", netcheck.summary())
    # the rest is still a clean grid; only the stray curve is reported
    assert not netcheck.summary()[0].endswith("ready")
    assert netcheck.summary()[1:] == ["1 curve(s) cross nothing (grey)"]
    grid_curves[0].location.y -= 50.0
    vl.update()
    assert bpy.ops.spline_surface.network_check() == {'FINISHED'}
    assert netcheck.summary() == [] and netcheck._state["handle"] is None
    assert bpy.ops.spline_surface.curvature_comb() == {'FINISHED'}
    print("curve combs", len(curv._state["envelopes"]))
    assert len(curv._state["envelopes"]) == 7 * 4  # one per Bezier segment
    assert bpy.ops.spline_surface.curvature_comb() == {'FINISHED'}

    # ---- v0.39: Move Crossing (what each drag does) and Fair Curves
    from spline_surface import curvetools as ct
    from spline_surface.core.curveedit import nearest_param, pull_chain
    bpy.ops.object.select_all(action='DESELECT')
    for o in grid_curves:
        o.select_set(True)
    vl.update()
    found = ct.crossings(bpy.context, grid_curves)
    assert len(found) == 12
    a, b, p = found[0]
    delta = np.array([0.0, 0.0, 0.3])
    for ob in (a, b):
        chain = ct.read_chain(ob)
        ct.write_chain(ob, *pull_chain(*chain, nearest_param(*chain, p), delta,
                                       ct._fixed_params(ob, chain, found, skip=p)))
    vl.update()
    after = ct.crossings(bpy.context, grid_curves)
    reached = min(np.linalg.norm(q - (p + delta)) for _, _, q in after)
    others = max(min(np.linalg.norm(q - r) for _, _, q in after) for _, _, r in found[1:])
    print("move crossing: off target", reached, "other crossings moved", others)
    assert len(after) == 12 and reached < 1e-2 and others < 1e-2
    assert all(pt.handle_left_type == 'ALIGNED' for pt in a.data.splines[0].bezier_points)
    netcheck.show(bpy.context, True)
    assert netcheck.summary()[0].endswith("ready")
    netcheck.show(bpy.context, False)
    # a bump between crossings, then Fair Curves: the crossings stay
    cv = grid_curves[1]
    mid = cv.data.splines[0].bezier_points[2]
    for attr in ("co", "handle_left", "handle_right"):
        v = getattr(mid, attr)
        v.z += 0.3
        setattr(mid, attr, v)
    vl.update()
    bumped = [q for x, y, q in ct.crossings(bpy.context, grid_curves) if cv in (x, y)]
    shape = ct.read_chain(cv)[0].copy()
    bpy.ops.object.select_all(action='DESELECT')
    cv.select_set(True)
    vl.objects.active = cv
    assert bpy.ops.spline_surface.fair_curves(strength=1.0) == {'FINISHED'}
    vl.update()
    faired = [q for x, y, q in ct.crossings(bpy.context, grid_curves) if cv in (x, y)]
    drift = max(min(np.linalg.norm(q - r) for q in faired) for r in bumped)
    changed = float(np.abs(ct.read_chain(cv)[0] - shape).max())
    print("fair curves: shape changed", changed, "crossings drifted", drift)
    assert changed > 1e-3 and drift < 2e-2 and len(faired) == len(bumped)

    # ---- v0.41: Flange folds a rounded wall off a patch edge
    from spline_surface.operators import compute_flange  # noqa: F401
    bpy.context.scene.cursor.location = (0, 700, 0)
    bpy.ops.spline_surface.add_patch(shape='PLANE', points_u=5, points_v=5)
    panel = bpy.context.active_object
    vl.update()
    edge_y = world_net(panel)[0][..., 1].max()
    before_names = {o.name for o in bpy.data.objects}
    assert bpy.ops.spline_surface.flange(source=panel.name, edge='V1', length=0.4, angle=90.0,
                                         radius=0.08) == {'FINISHED'}
    made = [o for o in bpy.data.objects if o.name not in before_names]
    methods = sorted(json.loads(o[spline_surface.operators.META_KEY])["method"] for o in made)
    print("flange made", [(o.name, json.loads(o[spline_surface.operators.META_KEY])["method"]) for o in made])
    assert methods == ["blend", "fillet_trim"], methods
    wall = next(o for o in made if json.loads(o[spline_surface.operators.META_KEY])["method"]
                == "fillet_trim" and "_flange" in o.name)
    wnet = world_net(wall)[0]
    drop = wnet[..., 2].min()
    print("flange wall bottom z", drop)
    assert abs(drop - (-0.4)) < 1e-4, drop
    # the wall hangs below the panel's edge, square to it
    assert np.allclose(wnet[..., 1], edge_y, atol=1e-4)
    # Return: a second flange off the free edge of the first wall
    assert bpy.ops.spline_surface.flange(source=wall.name, edge='V1', length=0.2, angle=90.0,
                                         radius=0.04) == {'FINISHED'}
    ret = bpy.context.active_object
    print("return flange", ret.name, world_net(ret)[0][..., 2].min())
    # without rounding it is a single sharp wall that follows its source
    bpy.context.scene.cursor.location = (0, 720, 0)
    bpy.ops.spline_surface.add_patch(shape='PLANE', points_u=5, points_v=5)
    sharp = bpy.context.active_object
    assert bpy.ops.spline_surface.flange(source=sharp.name, edge='U0', length=0.3, angle=60.0,
                                         radius=0.0) == {'FINISHED'}
    w2 = bpy.context.active_object
    before = world_net(w2)[0].copy()
    sharp.location.x += 0.5
    vl.update()
    assert np.abs(world_net(w2)[0] - before).max() > 0.4

    # ---- v0.42: Panel Gap measures the clearance between facing panels
    from spline_surface import panelgap
    bpy.context.scene.cursor.location = (0, 800, 0)
    bpy.ops.spline_surface.add_patch(shape='PLANE', points_u=5, points_v=5, size_x=1.0, size_y=1.0)
    left = bpy.context.active_object
    bpy.context.scene.cursor.location = (1.02, 800, 0.004)   # 20 mm to the right, 4 mm proud
    bpy.ops.spline_surface.add_patch(shape='PLANE', points_u=5, points_v=5, size_x=1.0, size_y=1.0)
    right = bpy.context.active_object
    vl.update()
    hidden = [o for o in bpy.data.objects if o.type == 'SURFACE' and o not in (left, right)]
    for o in hidden:
        o.hide_set(True)
    assert bpy.ops.spline_surface.panel_gap() == {'FINISHED'}
    joints = [j for j in panelgap.joints() if {j["a"], j["b"]} == {left.name, right.name}]
    print("panel gap joints", [(j["a"], j["b"], round(j["min"], 4), round(j["max"], 4),
                                round(j["step"], 4)) for j in joints])
    assert len(joints) == 1
    j = joints[0]
    assert abs(j["min"] - 0.02) < 1e-3 and abs(j["max"] - 0.02) < 1e-3 and abs(j["step"] - 0.004) < 1e-3
    right.location.x += 0.01
    vl.update()
    panelgap.refresh(bpy.context)
    j = [j for j in panelgap.joints() if {j["a"], j["b"]} == {left.name, right.name}][0]
    assert abs(j["mean"] - 0.03) < 1e-3, j["mean"]
    assert bpy.ops.spline_surface.panel_gap() == {'FINISHED'} and not panelgap.is_on()
    for o in hidden:
        o.hide_set(False)

    # ---- v0.43: Presets drop in starting panels
    for kind in ('HOOD', 'ROOF', 'DOOR'):
        bpy.context.scene.cursor.location = (0, 900, 0)
        before = {o.name for o in bpy.data.objects}
        assert bpy.ops.spline_surface.add_preset(kind=kind) == {'FINISHED'}
        new = [o for o in bpy.data.objects if o.name not in before]
        assert len(new) == 1 and new[0].type == 'SURFACE', (kind, new)
        assert world_net(new[0])[0].shape == (6, 6, 3)
    bpy.context.scene.cursor.location = (0, 920, 0)
    before = {o.name for o in bpy.data.objects}
    assert bpy.ops.spline_surface.add_preset(kind='HOOD_FENDER', radius=0.08) == {'FINISHED'}
    new = [o for o in bpy.data.objects if o.name not in before]
    methods = sorted(json.loads(o[spline_surface.operators.META_KEY])["method"] for o in new)
    print("preset hood+fender", methods)
    assert methods == ["blend", "fillet_trim", "fillet_trim"], methods
    bpy.context.scene.cursor.location = (0, 940, 0)
    before = {o.name for o in bpy.data.objects}
    assert bpy.ops.spline_surface.add_preset(kind='HOOD_FENDER', radius=0.0) == {'FINISHED'}
    assert len([o for o in bpy.data.objects if o.name not in before]) == 2

    # ---- v0.44: Blueprints stand a reference drawing at its real size
    import tempfile
    img = bpy.data.images.new("bp_test", 200, 100)
    path = os.path.join(tempfile.gettempdir(), "ss_blueprint_test.png")
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    for view, check in (
            ('SIDE', lambda c: abs(np.ptp(c[:, 1]) - 4.0) < 1e-4 and abs(c[:, 2].min()) < 1e-4
             and abs(c[:, 2].max() - 2.0) < 1e-4 and abs(np.ptp(c[:, 0])) < 1e-4
             and abs((c[:, 1].min() + c[:, 1].max()) / 2) < 1e-4),
            ('FRONT', lambda c: abs(np.ptp(c[:, 0]) - 4.0) < 1e-4 and abs(c[:, 2].min()) < 1e-4
             and abs(c[:, 2].max() - 2.0) < 1e-4 and abs(np.ptp(c[:, 1])) < 1e-4),
            ('TOP', lambda c: abs(np.ptp(c[:, 0]) - 4.0) < 1e-4 and abs(np.ptp(c[:, 1]) - 2.0) < 1e-4
             and abs(c[:, 1].min() + 1.0) < 1e-4 and abs(np.ptp(c[:, 2])) < 1e-4)):
        assert bpy.ops.spline_surface.add_blueprint(filepath=path, view=view, length=4.0) == {'FINISHED'}
        bp = bpy.context.active_object
        assert bp.type == 'EMPTY' and bp.empty_display_type == 'IMAGE'
        vl.update()
        sz = bp.empty_display_size
        ox, oy = bp.empty_image_offset
        asp = 0.5   # 200 x 100 image: the offset is in units of the displayed width and height
        local = [(ox * sz, oy * sz * asp), ((ox + 1) * sz, oy * sz * asp),
                 ((ox + 1) * sz, (oy + 1) * sz * asp), (ox * sz, (oy + 1) * sz * asp)]
        corners = np.array([np.array(bp.matrix_world @ __import__("mathutils").Vector((x, y, 0.0)))
                            for x, y in local])
        print("blueprint", view, corners.min(0).round(3), corners.max(0).round(3))
        assert check(corners), (view, corners)

    # ---- v0.45: Project Curve lays a curve onto a patch
    bpy.context.scene.cursor.location = (0, 1000, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME', points_u=6, points_v=6, size_x=2.0, size_y=2.0)
    dome = bpy.context.active_object
    proj_src = bezier_curve("proj_line", [(-0.6, 1000.0, 3.0), (0.0, 1000.3, 3.0), (0.6, 1000.0, 3.0)])
    select([proj_src, dome])
    vl.objects.active = dome
    assert bpy.ops.spline_surface.project_curve(curve=proj_src.name, patch=dome.name, mode='Z',
                                                samples=10) == {'FINISHED'}
    proj = bpy.context.active_object
    pl = read_polyline(proj, bpy.context.evaluated_depsgraph_get(), 40)
    dnet, dord = world_net(dome)
    # every point of the new curve lies on the dome
    from spline_surface.core.project import project_points as _pp
    back, _, _ = _pp(dnet, dord, pl)
    print("project curve: max distance from the dome", np.linalg.norm(back - pl, axis=1).max())
    assert np.linalg.norm(back - pl, axis=1).max() < 5e-3
    assert abs(pl[0][0] + 0.6) < 1e-3 and abs(pl[-1][0] - 0.6) < 1e-3   # ends keep their x (along Z)
    assert pl[:, 2].max() < 3.0 and pl[:, 2].min() > -1.0

    # ---- v0.46: Trim to Mesh (prototype) cuts a hole along a closed curve
    bpy.context.scene.cursor.location = (0, 1100, 0)
    bpy.ops.spline_surface.add_patch(shape='DOME', points_u=6, points_v=6, size_x=2.0, size_y=2.0)
    cap = bpy.context.active_object
    circ = bezier_curve("hole_loop", [(0.4, 1100.0, 3.0), (0.0, 1100.4, 3.0), (-0.4, 1100.0, 3.0),
                                      (0.0, 1099.6, 3.0)])
    circ.data.splines[0].use_cyclic_u = True
    select([circ, cap])
    assert bpy.ops.spline_surface.trim_mesh(mode='Z', resolution=40, points=48) == {'FINISHED'}
    cut = bpy.context.active_object
    full_faces = 40 * 40
    xy = np.array([(v.co.x, v.co.y - 1100.0) for v in cut.data.vertices])
    r = np.hypot(xy[:, 0], xy[:, 1])
    print("trim: faces", len(cut.data.polygons), "of", full_faces, "closest vertex to the centre", r.min())
    assert len(cut.data.polygons) < full_faces - 20 and r.min() > 0.3     # the hole is open
    assert r.min() < 0.42                                                  # the ring sits on the outline
    # ---- a window opening built from four Coons strips around the opening (the manual's recipe)
    from spline_surface.seams import shared_seams
    z0 = 1200.0
    P = lambda x, y: (x, z0 + y, 0.0)
    outer = [P(0, 0), P(4, 0), P(4, 3), P(0, 3)]
    inner = [P(1, 0.8), P(3, 0.8), P(3, 2.2), P(1, 2.2)]
    cv = lambda a, b, n: bezier_curve(n, [a, tuple((np.array(a) + np.array(b)) / 2), b])
    outer_c = [cv(outer[i], outer[(i + 1) % 4], "o%d" % i) for i in range(4)]
    inner_c = [cv(inner[i], inner[(i + 1) % 4], "i%d" % i) for i in range(4)]
    conn_c = [cv(outer[i], inner[i], "c%d" % i) for i in range(4)]
    strips = []
    for i in range(4):
        j = (i + 1) % 4
        select([outer_c[i], inner_c[i], conn_c[i], conn_c[j]])
        assert bpy.ops.spline_surface.coons() == {'FINISHED'}, i
        strips.append(bpy.context.active_object)
    vl.update()
    for o in bpy.data.objects:
        if o.type == 'SURFACE' and o not in strips:
            o.hide_set(True)
    ring_seams = [s for s in shared_seams(bpy.context)
                  if s["a"] in {x.name for x in strips} and s["b"] in {x.name for x in strips}]
    print("window frame: strips", len(strips), "seams", len(ring_seams), [s["grade"] for s in ring_seams])
    assert len(ring_seams) == 4 and all(s["grade"] != 'open' for s in ring_seams)
    for o in bpy.data.objects:
        if o.type == 'SURFACE':
            o.hide_set(False)
    # ---- v0.47: To Mesh cuts the pentagons left by welded T-junctions into quads and triangles
    five_q = ring(5, (0, 1300, 0))
    assert bpy.ops.spline_surface.fill_hole(picks=";".join("%s:V0" % o.name for o in five_q))         == {'FINISHED'}
    group_q = list(five_q) + [o for o in bpy.context.selected_objects]
    for flag in (False, True):
        select(group_q)
        assert bpy.ops.spline_surface.to_mesh(quads=flag) == {'FINISHED'}
        me_ = bpy.context.active_object.data
        big = sum(1 for p in me_.polygons if len(p.vertices) > 4)
        print("to mesh quads", flag, "faces", len(me_.polygons), "n-gons", big)
        if flag:
            assert big == 0
    spline_surface.unregister()
    print("ALL OK")


try:
    run()
except Exception:
    traceback.print_exc()
    sys.exit(1)
