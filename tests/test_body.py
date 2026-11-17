"""The larger sample (front of a body with its cabin), end to end.

    blender -b --factory-startup samples/body_front_cabin.blend --python tests/test_body.py

Opens the sample built by tools/build_sample_body.py, rebuilds every patch from
its recipe, and checks what the build set up still holds: the Even Gap joints
(door to fender; front window to door, handed on by Split Patch) at 4 mm and
flush, also after the door is raised, the frozen fillet partners
after Split Patch, Diagnose without false "open" seams, and a clean To Mesh.
Prints "ALL OK".
"""
import json
import os
import sys

import bmesh
import bpy
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import spline_surface  # noqa: E402
from spline_surface import live  # noqa: E402
from spline_surface.core.gap import edge_normals, measure_gap  # noqa: E402
from spline_surface.recipe import META_KEY, edge_polyline, world_net  # noqa: E402


def meta(name):
    return json.loads(bpy.data.objects[name][META_KEY])


def joint(name, g=None):
    """Gap and step along the part of `name`'s evened edge that faces its partner
    (g: the joint as Even Gap stores it; default: `name`'s own recipe)."""
    g = g or meta(name)["gap"]
    net, orders = world_net(bpy.data.objects[name])
    rnet, rorders = world_net(bpy.data.objects[g["ref"]])
    line = edge_polyline(rnet, rorders, g["edge_ref"], 80)
    m = measure_gap(edge_polyline(net, orders, g["edge"], 40), line,
                    edge_normals(net, orders, g["edge"], 40))
    past_end = np.minimum(np.linalg.norm(m["b"] - line[0], axis=1),
                          np.linalg.norm(m["b"] - line[-1], axis=1)) < 1e-6
    facing = (m["dist"] <= g["reach"]) & ~past_end
    return m["gap"][facing], m["step"][facing]


def run():
    spline_surface.register()
    ctx = bpy.context
    patches = [o for o in ctx.scene.objects if o.type == 'SURFACE' and o.visible_get()]
    assert len(patches) == 42, len(patches)          # 21 per side (the Lip is hidden)
    for o in ctx.scene.objects:
        o.select_set(o in patches)
    ctx.view_layer.objects.active = patches[0]
    assert bpy.ops.spline_surface.regenerate_all() == {'FINISHED'}
    assert not live.FAILED, live.FAILED

    # The side was evened to the door before it was split; Split Patch hands the gap to
    # the piece that faces the door (the front window).
    assert "gap" in meta("Side_split_split") and "gap" not in meta("Side")
    for name in ("Door", "Side_split_split"):
        gap, step = joint(name)
        print("%s joint: gap %.4f-%.4f, step <= %.4f over %d samples"
              % (name, gap.min(), gap.max(), step.max(), len(gap)))
        assert len(gap) >= 8 and np.abs(gap - 0.004).max() < 1e-3 and step.max() < 1e-3, name
    # raise the door 5 mm: the front window's bottom follows, still joined to the B-pillar
    door = bpy.data.objects["Door"]
    door.location.z += 0.005
    ctx.view_layer.update()
    res = live.update_chain(["Door"], ctx)
    assert "Side_split_split" in res["ok"], res
    gap, step = joint("Side_split_split")
    print("after raising the door: belt gap %.4f-%.4f" % (gap.min(), gap.max()))
    assert np.abs(gap - 0.004).max() < 1e-3 and step.max() < 1e-3
    from spline_surface.core.continuity import find_shared_edge
    (Pw, ow), (Pp, op) = world_net(bpy.data.objects["Side_split_split"]), world_net(bpy.data.objects["Side_split"])
    shared = find_shared_edge(Pw, Pp, 1e-3, ow, op)
    assert shared is not None and shared[3] < 5e-4, shared      # the window and the pillar stay joined
    door.location.z -= 0.005
    ctx.view_layer.update()
    live.update_chain(["Door"], ctx)
    # splitting the side froze its fillet partners instead of leaving them unbuildable
    assert meta("Roof")["method"] == "raw" and meta("Side_fillet")["method"] == "raw"
    assert meta("Lip_sweep").get("rotate") is True

    for o in ctx.scene.objects:
        o.select_set(False)
    assert bpy.ops.spline_surface.diagnose() == {'FINISHED'}
    issues = json.loads(ctx.scene.spline_surface.diagnose_report)
    print("diagnose:", len(issues), sorted({i["kind"] for i in issues}))
    assert not [i for i in issues if i["kind"] == "seam_open"], issues

    for o in ctx.scene.objects:
        o.select_set(o in patches)
    assert bpy.ops.spline_surface.to_mesh() == {'FINISHED'}
    bm = bmesh.new()
    bm.from_mesh(ctx.active_object.data)
    wrong_way = 0
    for e in bm.edges:
        if len(e.link_faces) == 2:
            a, b = (next(lp for lp in f.loops if lp.edge == e) for f in e.link_faces)
            wrong_way += a.vert == b.vert
    nonmanifold = sum(len(e.link_faces) > 2 for e in bm.edges)
    tiny = sum(f.calc_area() < 1e-10 for f in bm.faces)
    print("body mesh: faces %d, wrong-way edges %d, non-manifold %d, collapsed %d"
          % (len(bm.faces), wrong_way, nonmanifold, tiny))
    assert tiny == 0
    # the moulding lip meets the bumper and the nose bridge on one edge: To Mesh marks it
    group = ctx.active_object.vertex_groups.get("non_manifold")
    assert nonmanifold > 0 and group is not None
    marked = sum(1 for v in ctx.active_object.data.vertices if any(g.group == group.index for g in v.groups))
    print("marked %d vertices in 'non_manifold'" % marked)
    assert marked >= nonmanifold
    bm.free()
    spline_surface.unregister()
    print("ALL OK")


try:
    run()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
