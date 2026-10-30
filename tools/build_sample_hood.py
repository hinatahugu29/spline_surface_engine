"""Build the sample hood + fender model (a reference for exercising the add-on).

blender -b --factory-startup --python tools/build_sample_hood.py -- [out.blend]

Half a car front: a crowned hood (centre line on x = 0), a fender wall hanging
off its side edge, a rounded shoulder between them (Fillet), a nose extended
off the hood's front edge (Extend, G2), a cowl off its rear edge, both corners
between extension and fender closed with Fill Hole, then the whole group mirrored across
x = 0 and put through To Mesh. Prints each step and the seam grades, so pain
points show up as lines in the log.
"""
import json
import os
import sys

import bpy
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import spline_surface  # noqa: E402
from spline_surface.bridge.write_surface import build_surface_data  # noqa: E402
from spline_surface.core.continuity import find_shared_edge, grade, measure  # noqa: E402
from spline_surface.core.nurbs import eval_points  # noqa: E402
from spline_surface.filleting import find_fillet_seams  # noqa: E402
from spline_surface.operators import edge_polyline, store_meta, world_net  # noqa: E402
from spline_surface.recipe import META_KEY  # noqa: E402

extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = extra[0] if extra else os.path.join(ROOT, "samples", "hood_fender.blend")
spline_surface.register()
sc = bpy.context.scene.spline_surface
vl = bpy.context.view_layer


def raw_patch(name, net):
    cu = build_surface_data(name, net, 12, (6, 6))
    ob = bpy.data.objects.new(cu.name, cu)
    bpy.context.collection.objects.link(ob)
    store_meta(ob, {"method": "raw", "sources": [], "net": net.tolist(), "orders": [6, 6]})
    return ob


def step(label, result):
    ok = result == {'FINISHED'}
    print(("ok   " if ok else "PAIN ") + label, "" if ok else result)
    return ok


def seam_grade(a, b):
    na, oa = world_net(a)
    nb, ob_ = world_net(b)
    try:
        ea, eb, rev, gap = find_shared_edge(na, nb, 1e-3, oa, ob_)
    except Exception:
        return None
    if ea is None:
        return None
    return grade(measure(na, nb, ea, eb, rev, oa, ob_)), gap


def report_seams(tag, objs):
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            s = seam_grade(a, b)
            if s:
                print("seam [%s] %s | %s : %s (gap %.2g)" % (tag, a.name, b.name, s[0], s[1]))


g = np.linspace(0, 1, 6)
U, V = g[:, None] + 0 * g, 0 * g[:, None] + g
# Hood: width x = V (0 is the centre line), length y = 1.4 U, crowned both ways.
hz = 0.18 * (1 - V ** 2) + 0.05 * (1 - (2 * U - 1) ** 2)
hood = raw_patch("Hood", np.stack([V, 1.4 * U, hz], -1))
# Fender: a wall dropping from the hood's x = 1 edge.
edge_z = 0.05 * (1 - (2 * g - 1) ** 2)
fx = 1.0 + 0.04 * np.sin(np.pi * V) * (1 - (2 * U - 1) ** 2)
fender = raw_patch("Fender", np.stack([fx, 1.4 * U, edge_z[:, None] - 0.5 * V], -1))
vl.update()

pairs = find_fillet_seams(bpy.context, sc)
print("seams", sorted(pairs))
if pairs:
    picked = json.dumps([[a, v[0], b, v[1]] for (a, b), v in pairs.items()])
    step("fillet shoulder", bpy.ops.spline_surface.fillet(
        seams=picked, radius=0.08, continuity='G2', corners=True))
else:
    print("PAIN no fillet seam between hood and fender")

# Nose: extend the hood's front edge (y = 1.4) forward, tangent- and curvature-continuous.
for o in bpy.context.selected_objects:
    o.select_set(False)
vl.objects.active = bpy.data.objects["Hood"]
bpy.data.objects["Hood"].select_set(True)
step("extend nose", bpy.ops.spline_surface.extend_edge(edge='U1', length=0.4, continuity='G2'))
vl.update()

def _end(name, edge, last):
    net, orders = world_net(bpy.data.objects[name])
    pl = edge_polyline(net, orders, edge, 8)
    return tuple(pl[-1] if last else pl[0])


def _normal(name, u, v):
    net, orders = world_net(bpy.data.objects[name])
    du = eval_points(net, orders, [u], [v], 1, 0)[0]
    dv = eval_points(net, orders, [u], [v], 0, 1)[0]
    n = np.cross(du, dv)
    return n / np.linalg.norm(n)


def _in_plane(d, n):
    d = d - np.dot(d, n) * n
    return d / np.linalg.norm(d)


def _edge_touching(name, point):
    """(edge, u, v) of the patch's edge that ends at `point`, with the corner's (u, v)."""
    for edge in ('U0', 'U1', 'V0', 'V1'):
        for last in (False, True):
            if np.linalg.norm(np.array(_end(name, edge, last)) - point) < 1e-4:
                u = 0.0 if edge[0] == 'U' and edge[1] == '0' else 1.0 if edge[0] == 'U' else (1.0 if last else 0.0)
                v = (1.0 if last else 0.0) if edge[0] == 'U' else (0.0 if edge[1] == '0' else 1.0)
                return edge, u, v, last
    raise ValueError("no edge of %s ends at %s" % (name, point))


def close_corner(tag, ext, ext_point, wall, wall_edge, fillet, fillet_edge, mid):
    """Fill the hole between an extension's side, the fender's end edge and a closing curve.

    The closing curve runs from the bottom of the fender's end edge to the extension's
    tip `ext_point` and leaves each end inside the neighbouring patch's tangent plane
    (otherwise the fill cannot be tangent there). Returns the fill_hole result.
    """
    F = np.array(_end(wall, wall_edge, True))
    B = np.array(ext_point)
    e_edge, eu, ev, _ = _edge_touching(ext, B)
    M = np.array(mid)
    reach = 0.35 * np.linalg.norm(B - F)
    wu, wv = (1.0 if wall_edge == 'U1' else 0.0), 1.0
    dF = _in_plane(M - F, _normal(wall, wu, wv))
    dB = _in_plane(M - B, _normal(ext, eu, ev))
    dM = (B - F) / np.linalg.norm(B - F)
    cu = bpy.data.curves.new("Corner" + tag, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(2)
    pts = [(F, F - dF * reach, F + dF * reach), (M, M - dM * reach, M + dM * reach),
           (B, B + dB * reach, B - dB * reach)]
    for bp, (co, hl, hr) in zip(sp.bezier_points, pts):
        bp.handle_left_type = bp.handle_right_type = 'FREE'
        bp.co, bp.handle_left, bp.handle_right = co, hl, hr
    ob = bpy.data.objects.new("Corner" + tag, cu)
    bpy.context.collection.objects.link(ob)
    vl.update()
    picks = "%s:%s;%s:%s;%s:%s;%s:CURVE" % (ext, e_edge, fillet, fillet_edge, wall, wall_edge, ob.name)
    r = bpy.ops.spline_surface.fill_hole(picks=picks, continuity='G2', layout='GRID')
    vl.update()
    return r


# Front corner: the nose side, the fender's front edge (fillet + wall) and a
# closing curve from the fender's bottom to the nose tip bound a four-sided hole.
step("fill front corner", close_corner(
    "Front", "Hood_ext", _end("Hood_ext", 'U0', True), "Fender", 'U1', "Fender_fillet", 'U1',
    (1.02, 1.75, -0.32)))

# Cowl: extend the hood's rear edge backwards, then close the rear corner the same way.
for o in bpy.context.selected_objects:
    o.select_set(False)
vl.objects.active = bpy.data.objects["Hood"]
bpy.data.objects["Hood"].select_set(True)
step("extend cowl", bpy.ops.spline_surface.extend_edge(edge='U0', length=0.3, continuity='G2'))
vl.update()
cowl = [o for o in bpy.data.objects if o.name.startswith("Hood_ext") and o.name != "Hood_ext"
        and "fill" not in o.name and "mirror" not in o.name][-1]
print("cowl is", cowl.name)
def _side_tip(name, x):
    """End point (smallest y) of the patch's edge lying along x = `x`."""
    for edge in ('U0', 'U1', 'V0', 'V1'):
        a, b = np.array(_end(name, edge, False)), np.array(_end(name, edge, True))
        if abs(a[0] - x) < 1e-3 and abs(b[0] - x) < 1e-3:
            return tuple(a if a[1] < b[1] else b)
    raise ValueError("%s has no edge along x = %s" % (name, x))


step("fill rear corner", close_corner(
    "Rear", cowl.name, _side_tip(cowl.name, 0.95), "Fender", 'U0', "Fender_fillet", 'U0',
    (1.02, -0.2, -0.32)))

group = [o for o in bpy.data.objects if META_KEY in o]
print("half-body patches", sorted(o.name for o in group))
report_seams("half", group)

# Mirror across x = 0.
for o in group:
    o.select_set(True)
step("mirror group", bpy.ops.spline_surface.mirror_group())
vl.update()
full = [o for o in bpy.data.objects if META_KEY in o]
print("full-body patches", len(full))
report_seams("full", [o for o in full if o.name in ("Hood", "Hood_mirror") or "nose" in o.name.lower()
                      or o.name.startswith("Hood")])

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
print("saved", os.path.abspath(out))
for o in full:
    o.select_set(True)
step("to mesh", bpy.ops.spline_surface.to_mesh())
