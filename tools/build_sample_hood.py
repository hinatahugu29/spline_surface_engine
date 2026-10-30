"""Build the sample hood + fender model (a reference for exercising the add-on).

blender -b --factory-startup --python tools/build_sample_hood.py -- [out.blend]

Half a car front: a crowned hood (centre line on x = 0), a fender wall hanging
off its side edge, a rounded shoulder between them (Fillet), a nose extended
off the hood's front edge (Extend, G2), then the whole group mirrored across
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

# Front corner: the nose side, the fender's front edge (fillet + wall) and a
# closing curve from the fender's bottom to the nose tip bound a four-sided hole.
cu = bpy.data.curves.new("CornerEdge", 'CURVE')
cu.dimensions = '3D'
sp = cu.splines.new('BEZIER')
sp.bezier_points.add(2)
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


# The closing curve has to leave each end inside the neighbouring patch's
# tangent plane, otherwise the fill cannot be tangent to that patch at the corner.
F, B = np.array(_end("Fender", 'U1', True)), np.array(_end("Hood_ext", 'U0', True))
M = np.array((1.02, 1.75, -0.32))
reach = 0.35 * np.linalg.norm(B - F)
dF = _in_plane(M - F, _normal("Fender", 1.0, 1.0))
dB = _in_plane(M - B, _normal("Hood_ext", 0.0, 1.0))
dM = (B - F) / np.linalg.norm(B - F)
pts = [(F, F - dF * reach, F + dF * reach), (M, M - dM * reach, M + dM * reach),
       (B, B + dB * reach, B - dB * reach)]
for bp, (co, hl, hr) in zip(sp.bezier_points, pts):
    bp.handle_left_type = bp.handle_right_type = 'FREE'
    bp.co, bp.handle_left, bp.handle_right = co, hl, hr
corner_edge = bpy.data.objects.new("CornerEdge", cu)
bpy.context.collection.objects.link(corner_edge)
vl.update()
picks = "Hood_ext:U0;Fender_fillet:U1;Fender:U1;CornerEdge:CURVE"
step("fill front corner", bpy.ops.spline_surface.fill_hole(picks=picks, continuity=os.environ.get('FC', 'G2'), layout='GRID', tension=float(os.environ.get('FT', '1.0'))))
vl.update()

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
