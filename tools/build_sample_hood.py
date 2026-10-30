"""Build the sample hood + fender model (a reference for exercising the add-on).

blender -b --factory-startup --python tools/build_sample_hood.py -- [out.blend]

A crowned hood, a fender wall hanging off its side edge, and a rounded
shoulder (Fillet) between them. Prints what it did and anything that failed,
so pain points show up as lines in the log.
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
from spline_surface.operators import store_meta  # noqa: E402
from spline_surface.filleting import find_fillet_seams  # noqa: E402
from spline_surface.recipe import META_KEY  # noqa: E402

extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = extra[0] if extra else os.path.join(ROOT, "samples", "hood_fender.blend")
spline_surface.register()
sc = bpy.context.scene.spline_surface


def raw_patch(name, net):
    cu = build_surface_data(name, net, 12, (6, 6))
    ob = bpy.data.objects.new(cu.name, cu)
    bpy.context.collection.objects.link(ob)
    store_meta(ob, {"method": "raw", "sources": [], "net": net.tolist(), "orders": [6, 6]})
    return ob


g = np.linspace(0, 1, 6)
U, V = g[:, None] + 0 * g, 0 * g[:, None] + g
# Hood: 1.4 long (U, x) by 1.0 wide (V, y), crowned across and along.
hx, hy = 1.4 * U, 1.0 * V
hz = 0.18 * (1 - (2 * V - 1) ** 2) + 0.05 * (1 - (2 * U - 1) ** 2)
hood = raw_patch("Hood", np.stack([hx, hy, hz], -1))
# Fender: a wall dropping from the hood's y = 1 edge, with the same bulge there.
edge_z = 0.18 * 0.0 + 0.05 * (1 - (2 * g - 1) ** 2)
fx = 1.4 * U
fy = 1.0 + 0.04 * np.sin(np.pi * V) * (1 - (2 * U - 1) ** 2)
fz = edge_z[:, None] - 0.5 * V
fender = raw_patch("Fender", np.stack([fx, fy, fz], -1))
bpy.context.view_layer.update()

pairs = find_fillet_seams(bpy.context, sc)
print("seams", sorted(pairs))
if not pairs:
    print("PAIN: no fillet seam found between hood and fender")
else:
    picked = json.dumps([[a, v[0], b, v[1]] for (a, b), v in pairs.items()])
    r = bpy.ops.spline_surface.fillet(seams=picked, radius=0.08, continuity='G2', corners=True)
    print("fillet", r)
    for o in bpy.context.selected_objects:
        print(" made", o.name, json.loads(o[META_KEY])["method"])

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
print("saved", os.path.abspath(out))

# ---- report: seam grades and mesh output
from spline_surface.core.continuity import find_shared_edge, grade, measure  # noqa: E402
from spline_surface.operators import world_net  # noqa: E402
objs = [o for o in bpy.data.objects if META_KEY in o]
for i, a in enumerate(objs):
    for b in objs[i + 1:]:
        na, oa = world_net(a)
        nb, ob_ = world_net(b)
        try:
            ea, eb, rev, gap = find_shared_edge(na, nb, 1e-3, oa, ob_)
        except Exception:
            continue
        if ea is None:
            continue
        print("seam", a.name, b.name, grade(measure(na, nb, ea, eb, rev, oa, ob_)), "gap", gap)
for o in objs:
    o.select_set(True)
print("to_mesh", bpy.ops.spline_surface.to_mesh())
