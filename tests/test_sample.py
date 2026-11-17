"""The bundled sample, end to end: open it, rebuild everything, weld it to a mesh.

    blender -b --factory-startup samples/hood_fender.blend --python tests/test_sample.py

Checks that every patch rebuilds from its recipe, and that To Mesh gives one
clean mesh: no holes along the seams (the only open border is the outline of
the whole body), one consistent winding, no collapsed faces. Prints "ALL OK".

It also compares every rebuilt patch with tests/sample_snapshot.json (the control
points after the rebuild), so a refactoring that changes a shape is caught. After a
change that is meant to move the shapes, write a new snapshot:

    blender -b --factory-startup samples/hood_fender.blend --python tests/test_sample.py -- --update
"""
import json
import os
import sys

import bmesh
import bpy

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import spline_surface  # noqa: E402
from spline_surface import live  # noqa: E402
from spline_surface.recipe import world_net  # noqa: E402

SNAPSHOT = os.path.join(ROOT, "tests", "sample_snapshot.json")
SNAPSHOT_TOL = 1e-5


def check_snapshot(patches):
    """Compare the rebuilt control points with the stored ones (or store them)."""
    now = {}
    for o in patches:
        net, orders = world_net(o)
        now[o.name] = {"shape": list(net.shape), "orders": list(orders),
                       "points": [round(float(x), 7) for x in net.ravel()]}
    if "--update" in sys.argv or not os.path.exists(SNAPSHOT):
        with open(SNAPSHOT, "w", encoding="utf-8") as f:
            json.dump(now, f, indent=0, sort_keys=True)
        print("snapshot written:", len(now), "patches")
        return
    with open(SNAPSHOT, encoding="utf-8") as f:
        old = json.load(f)
    assert sorted(old) == sorted(now), (sorted(old), sorted(now))
    worst = 0.0
    for name, a in now.items():
        b = old[name]
        assert a["shape"] == b["shape"] and a["orders"] == b["orders"], name
        d = max(abs(x - y) for x, y in zip(a["points"], b["points"]))
        assert d < SNAPSHOT_TOL, "%s moved by %g" % (name, d)
        worst = max(worst, d)
    print("snapshot: %d patches match (largest difference %.2g)" % (len(now), worst))


def boundary_loops(bm):
    """Number of closed loops of open (one-face) edges."""
    open_edges = {e for e in bm.edges if len(e.link_faces) == 1}
    loops = 0
    while open_edges:
        loops += 1
        stack = [open_edges.pop()]
        while stack:
            e = stack.pop()
            for v in e.verts:
                for f in v.link_edges:
                    if f in open_edges:
                        open_edges.remove(f)
                        stack.append(f)
    return loops


def run():
    spline_surface.register()
    ctx = bpy.context
    patches = [o for o in ctx.scene.objects if o.type == 'SURFACE']
    assert len(patches) == 14, len(patches)
    for o in ctx.scene.objects:
        o.select_set(o.type == 'SURFACE')
    ctx.view_layer.objects.active = patches[0]
    assert bpy.ops.spline_surface.regenerate_all() == {'FINISHED'}
    assert not live.FAILED, live.FAILED
    check_snapshot(patches)
    for o in ctx.scene.objects:
        o.select_set(False)
    assert bpy.ops.spline_surface.diagnose() == {'FINISHED'}
    issues = json.loads(ctx.scene.spline_surface.diagnose_report)
    kinds = sorted({i["kind"] for i in issues})
    print("diagnose:", len(issues), "issue(s)", kinds)
    # Fender and Hood sit apart on purpose (their fillet joins them): never "open"
    assert not [i for i in issues if i["kind"] == "seam_open"], issues
    for o in ctx.scene.objects:
        o.select_set(o.type == 'SURFACE')
    assert bpy.ops.spline_surface.to_mesh() == {'FINISHED'}
    mesh_ob = ctx.active_object
    assert mesh_ob.type == 'MESH'
    bm = bmesh.new()
    bm.from_mesh(mesh_ob.data)
    bm.edges.ensure_lookup_table()
    wrong_way = 0
    for e in bm.edges:
        if len(e.link_faces) == 2:
            # Neighbouring faces wind the same way when they walk the shared edge in
            # opposite directions.
            a, b = (next(lp for lp in f.loops if lp.edge == e) for f in e.link_faces)
            wrong_way += a.vert == b.vert
    nonmanifold = sum(len(e.link_faces) > 2 for e in bm.edges)
    tiny = sum(f.calc_area() < 1e-10 for f in bm.faces)
    loops = boundary_loops(bm)
    print("sample mesh: faces %d, border loops %d, wrong-way edges %d, non-manifold %d, "
          "collapsed faces %d" % (len(bm.faces), loops, wrong_way, nonmanifold, tiny))
    assert wrong_way == 0 and nonmanifold == 0 and tiny == 0
    assert loops == 1, loops          # the outline of the body only: no holes at the seams
    bm.free()
    spline_surface.unregister()
    print("ALL OK")


try:
    run()
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
