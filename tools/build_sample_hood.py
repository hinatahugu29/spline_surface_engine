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
from spline_surface.filleting import find_fillet_seams  # noqa: E402
from spline_surface.operators import store_meta, world_net  # noqa: E402
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
