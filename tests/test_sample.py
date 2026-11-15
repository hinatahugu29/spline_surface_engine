"""The bundled sample, end to end: open it, rebuild everything, weld it to a mesh.

    blender -b --factory-startup samples/hood_fender.blend --python tests/test_sample.py

Checks that every patch rebuilds from its recipe, and that To Mesh gives one
clean mesh: no holes along the seams (the only open border is the outline of
the whole body), one consistent winding, no collapsed faces. Prints "ALL OK".
"""
import os
import sys

import bmesh
import bpy

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import spline_surface  # noqa: E402
from spline_surface import live  # noqa: E402


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
