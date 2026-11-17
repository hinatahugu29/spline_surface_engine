"""Build the larger sample: the front half of a car body with its cabin.

blender -b --factory-startup --python tools/build_sample_body.py -- [out.blend] [--render DIR]

Half a body (x >= 0, centre line on x = 0, y forward, z up), then mirrored:

- front: crowned hood, fender wall, rounded shoulder (Fillet), nose and cowl
  extended off the hood (Extend G2), both corners closed with Fill Hole
- bumper: an upright fascia below the nose, joined to it with Bridge Edges
- cabin: Roof + Side preset, windshield bridged from the cowl to the roof
- door: below the side, behind the fender; Even Gap evens its front edge to the
  fender and the side's bottom edge to the door top (4 mm, flush; both joints
  only partly face each other)
- windows: the side split twice (Split Patch) into rear window, B-pillar strip
  and front window, openings made of separate patches
- moulding: a lip swept along the bumper's top edge with Sweep Rotate

Every step prints "ok" or "PAIN" (a step that failed or gave a poor result),
then the seam grades and Diagnose's findings, so the log is the report.
With --render DIR, JPEG views of the result are written there.
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
from spline_surface.core.nurbs import eval_points  # noqa: E402
from spline_surface.filleting import find_fillet_seams  # noqa: E402
from spline_surface.matching import measure_seams  # noqa: E402
from spline_surface.recipe import EDGE_KEYS, META_KEY, edge_polyline, store_meta, world_net  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
render_dir = None
if "--render" in args:
    i = args.index("--render")
    render_dir = os.path.abspath(args[i + 1])
    args = args[:i] + args[i + 2:]
out = args[0] if args else os.path.join(ROOT, "samples", "body_front_cabin.blend")
spline_surface.register()
for _ob in [o for o in bpy.data.objects if o.type == 'MESH']:   # the factory startup cube
    bpy.data.objects.remove(_ob)
ctx = bpy.context
sc = ctx.scene.spline_surface
vl = ctx.view_layer
PAINS = []


# ---------------------------------------------------------------- helpers

def step(label, result, note=""):
    ok = result == {'FINISHED'}
    print(("ok   " if ok else "PAIN ") + label, note if ok else "%s %s" % (result, note))
    if not ok:
        PAINS.append(label)
    return ok


def pain(label, why):
    print("PAIN %s: %s" % (label, why))
    PAINS.append(label)


def raw_patch(name, net, orders=None):
    net = np.asarray(net, float)
    orders = orders or (net.shape[0], net.shape[1])
    cu = build_surface_data(name, net, 12, orders)
    ob = bpy.data.objects.new(cu.name, cu)
    ctx.collection.objects.link(ob)
    store_meta(ob, {"method": "raw", "sources": [], "net": net.tolist(), "orders": list(orders)})
    return ob


def only(*obs, active=None):
    for o in ctx.selected_objects:
        o.select_set(False)
    for o in obs:
        o.select_set(True)
    vl.objects.active = active or obs[0]


def ob(name):
    return bpy.data.objects[name]


def new_objects(before):
    return [o for o in bpy.data.objects if o.name not in before]


def edge_by(name, key):
    """The edge of patch `name` whose polyline minimises key(polyline)."""
    net, orders = world_net(ob(name))
    return min(EDGE_KEYS, key=lambda e: key(edge_polyline(net, orders, e, 16)))


def edge_line(name, edge, n=16):
    net, orders = world_net(ob(name))
    return edge_polyline(net, orders, edge, n)


def normal(name, u, v):
    net, orders = world_net(ob(name))
    du = eval_points(net, orders, [u], [v], 1, 0)[0]
    dv = eval_points(net, orders, [u], [v], 0, 1)[0]
    n = np.cross(du, dv)
    return n / np.linalg.norm(n)


def in_plane(d, n):
    d = d - np.dot(d, n) * n
    return d / np.linalg.norm(d)


def corner_uv(name, point):
    """(u, v) of the patch corner nearest `point`."""
    net, orders = world_net(ob(name))
    best = min(((u, v) for u in (0.0, 1.0) for v in (0.0, 1.0)),
               key=lambda uv: np.linalg.norm(eval_points(net, orders, [uv[0]], [uv[1]])[0] - point))
    return best


def bezier(name, pts):
    """A 3D Bezier curve through [(co, handle_left, handle_right)]."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for bp, (co, hl, hr) in zip(sp.bezier_points, pts):
        bp.handle_left_type = bp.handle_right_type = 'FREE'
        bp.co, bp.handle_left, bp.handle_right = co, hl, hr
    o = bpy.data.objects.new(name, cu)
    ctx.collection.objects.link(o)
    return o


def closing_curve(name, A, a_patch, B, b_patch, mid):
    """A curve from A to B through `mid`, leaving each end in its patch's tangent plane."""
    A, B, M = np.array(A), np.array(B), np.array(mid)
    reach = 0.35 * np.linalg.norm(B - A)
    dA = in_plane(M - A, normal(a_patch, *corner_uv(a_patch, A)))
    dB = in_plane(M - B, normal(b_patch, *corner_uv(b_patch, B)))
    dM = (B - A) / np.linalg.norm(B - A)
    o = bezier(name, [(A, A - dA * reach, A + dA * reach), (M, M - dM * reach, M + dM * reach),
                      (B, B + dB * reach, B - dB * reach)])
    vl.update()
    return o


def end_nearest(name, edge, point):
    pl = edge_line(name, edge)
    return pl[0] if np.linalg.norm(pl[0] - point) < np.linalg.norm(pl[-1] - point) else pl[-1]


g = np.linspace(0, 1, 6)
U, V = g[:, None] + 0 * g, 0 * g[:, None] + g

# ---------------------------------------------------------------- 1. front

# Hood: x = V (0 on the centre line), y = 1.4 U forward, crowned both ways.
hz = 0.18 * (1 - V ** 2) + 0.05 * (1 - (2 * U - 1) ** 2)
raw_patch("Hood", np.stack([V, 1.4 * U, hz], -1))
edge_z = 0.05 * (1 - (2 * g - 1) ** 2)
fx = 1.0 + 0.04 * np.sin(np.pi * V) * (1 - (2 * U - 1) ** 2)
raw_patch("Fender", np.stack([fx, 1.4 * U, edge_z[:, None] - 0.5 * V], -1))
vl.update()
pairs = find_fillet_seams(ctx, sc)
step("fillet shoulder", bpy.ops.spline_surface.fillet(
    seams=json.dumps([[a, v[0], b, v[1]] for (a, b), v in pairs.items()]),
    radius=0.08, continuity='G2', corners=True) if pairs else {'CANCELLED'})

only(ob("Hood"))
step("extend nose", bpy.ops.spline_surface.extend_edge(edge='U1', length=0.4, continuity='G2'))
only(ob("Hood"))
before = {o.name for o in bpy.data.objects}
step("extend cowl", bpy.ops.spline_surface.extend_edge(edge='U0', length=0.3, continuity='G2'))
vl.update()
NOSE = "Hood_ext"
COWL = new_objects(before)[0].name


def close_corner(tag, ext, front):
    """Fill between an extension's side, the fender's end (fillet + wall) and a closing curve."""
    fy = 1.4 if front else 0.0
    wall_edge = edge_by("Fender", lambda pl: np.abs(pl[:, 1] - fy).max())
    fil_edge = edge_by("Fender_fillet", lambda pl: np.abs(pl[:, 1] - fy).max())
    ext_side = edge_by(ext, lambda pl: -pl[:, 0].mean())                  # its outer side
    bottom = min(edge_line("Fender", wall_edge), key=lambda p: p[2])
    tip = min(edge_line(ext, ext_side), key=lambda p: -p[1] if front else p[1])
    mid = (1.02, 1.75 if front else -0.2, -0.32)
    curve = closing_curve("Corner" + tag, bottom, "Fender", tip, ext, mid)
    picks = "%s:%s;%s:%s;%s:%s;%s:CURVE" % (ext, ext_side, "Fender_fillet", fil_edge,
                                            "Fender", wall_edge, curve.name)
    r = bpy.ops.spline_surface.fill_hole(picks=picks, continuity='G2', layout='GRID')
    vl.update()
    return r


step("fill front corner", close_corner("Front", NOSE, True))
step("fill rear corner", close_corner("Rear", COWL, False))

# ---------------------------------------------------------------- 2. bumper

nose_tip = edge_by(NOSE, lambda pl: -pl[:, 1].mean())
tip = edge_line(NOSE, nose_tip)
y0 = tip[:, 1].max() + 0.12
bx = 0.98 * V                                         # u up, v across (centre at v = 0)
by = y0 - 0.30 * V ** 4 + 0.04 * np.sin(np.pi * U) * (1 - V ** 2)
bz = -0.45 + 0.38 * U
raw_patch("Bumper", np.stack([bx, by, bz], -1))
vl.update()
bump_top = edge_by("Bumper", lambda pl: -pl[:, 2].mean())
before = {o.name for o in bpy.data.objects}
step("bridge nose to bumper", bpy.ops.spline_surface.bridge_edges(
    source_a=NOSE, edge_a=nose_tip, source_b="Bumper", edge_b=bump_top, continuity='G1'))
made = new_objects(before)
NOSE_BRIDGE = made[0].name if made else None

# ---------------------------------------------------------------- 3. cabin

ctx.scene.cursor.location = (0.0, -2.0, 0.85)
before = {o.name for o in bpy.data.objects}
step("preset roof + side", bpy.ops.spline_surface.add_preset(
    kind='ROOF_SIDE', length=1.2, width=1.6, crown=0.06, depth=0.35, radius=0.05))
cabin = new_objects(before)
print("cabin patches", sorted(o.name for o in cabin))
ROOF = next(o.name for o in cabin if o.name.startswith("Roof"))
SIDE = next(o.name for o in cabin if o.name.startswith("Side") and "fillet" not in o.name)
vl.update()
cowl_rear = edge_by(COWL, lambda pl: pl[:, 1].mean())
roof_front = edge_by(ROOF, lambda pl: -pl[:, 1].mean())
before = {o.name for o in bpy.data.objects}
step("bridge cowl to roof (windshield)", bpy.ops.spline_surface.bridge_edges(
    source_a=COWL, edge_a=cowl_rear, source_b=ROOF, edge_b=roof_front, continuity='G1'))
made = new_objects(before)
SCREEN = made[0].name if made else None

# ---------------------------------------------------------------- 4. door

# Below the side, behind the fender; its front edge 10-30 mm off the fender (tapering).
dU, dV = U, V                                          # u back along y, v up
door_front = -0.010 - 0.020 * dV
dy = door_front - 1.2 * dU
dx = 0.99 + 0.035 * np.sin(np.pi * dV) * (1 - (2 * dU - 1) ** 2) - 0.04 * dV ** 2
dz = -0.48 + 0.95 * dV
raw_patch("Door", np.stack([dx, dy, dz], -1))
vl.update()
only(ob("Door"), ob("Fender"), active=ob("Door"))
step("even gap door to fender", bpy.ops.spline_surface.even_gap(target=0.004, flush=True))
gap = json.loads(ob("Door")[META_KEY]).get("gap")
print("door gap recipe", gap)
# Beltline: bring the side's bottom edge level with the door top, 4 mm above it.
# (Before splitting: evening one piece afterwards would open its seams with the others.)
only(ob(SIDE), ob("Door"), active=ob(SIDE))
step("even gap side to door (beltline)", bpy.ops.spline_surface.even_gap(target=0.004, flush=True))
print("side gap recipe", json.loads(ob(SIDE)[META_KEY]).get("gap"))

# Window openings as separate patches: split the side twice, leaving a B-pillar strip
# between the rear and the front window.
side_net, _ = world_net(ob(SIDE))
along_y = 'U' if np.ptp(side_net[:, 0, 1]) > np.ptp(side_net[0, :, 1]) else 'V'
before = {o.name for o in bpy.data.objects}
only(ob(SIDE))
step("split side (rear window | rest)", bpy.ops.spline_surface.split_patch(
    source=SIDE, direction=along_y, t=0.45))
rest = new_objects(before)[0].name
before = {o.name for o in bpy.data.objects}
only(ob(rest))
step("split side (B-pillar | front window)", bpy.ops.spline_surface.split_patch(
    source=rest, direction=along_y, t=0.15))
FRONT_WINDOW = new_objects(before)[0].name
pieces = sorted([SIDE, rest, FRONT_WINDOW], key=lambda n: world_net(ob(n))[0][..., 1].mean())
print("side pieces back to front", pieces)

# ---------------------------------------------------------------- 5. moulding

# A small lip hanging off the bumper's top edge at the centre line, swept along it.
top = edge_line("Bumper", bump_top)
start = top[np.argmin(top[:, 0])]
s = np.linspace(0, 1, 4)
lip_edge = np.stack([0 * s + start[0], start[1] + 0.03 * s, start[2] - 0.05 * s + 0.0 * s], -1)
lip = np.stack([lip_edge, lip_edge + [-0.05, 0, 0]], 1)       # (4, 2): the edge is v = 0
raw_patch("Lip", lip, (4, 2))
vl.update()
lip_edge_key = edge_by("Lip", lambda pl: np.abs(pl[:, 0] - start[0]).max() + 10 * (np.ptp(pl[:, 2]) < 1e-6))
before = {o.name for o in bpy.data.objects}
step("sweep lip along the bumper (Rotate)", bpy.ops.spline_surface.sweep(
    source_a="Lip", edge_a=lip_edge_key, source_b="Bumper", edge_b=bump_top, rotate=True))
made = new_objects(before)
MOULDING = made[0].name if made else None

# ---------------------------------------------------------------- 6. mirror + mesh

vl.update()
half = [o for o in bpy.data.objects if META_KEY in o and o.name != "Lip"]
print("half-body patches", len(half), sorted(o.name for o in half))
only(*half)
step("mirror group", bpy.ops.spline_surface.mirror_group(scope='SELECTED'))
vl.update()


# ---------------------------------------------------------------- render

def render_views(directory):
    """Workbench JPEGs of the visible patches from a few angles."""
    os.makedirs(directory, exist_ok=True)
    scene = ctx.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'MATCAP'
    scene.display.shading.color_type = 'SINGLE'
    scene.render.resolution_x, scene.render.resolution_y = 1200, 800
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 85
    cam_data = bpy.data.cameras.new("SampleCam")
    cam_data.lens = 40
    cam = bpy.data.objects.new("SampleCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    pts = np.concatenate([world_net(o)[0].reshape(-1, 3) for o in bpy.data.objects
                          if o.type == 'SURFACE' and o.visible_get()])
    centre = 0.5 * (pts.min(0) + pts.max(0))
    size = np.ptp(pts, axis=0).max()
    views = {"front34": (1.0, 1.1, 0.6), "side": (1.0, 0.0, 0.15), "top": (0.02, 0.0, 1.0),
             "rear34": (0.9, -1.0, 0.7)}
    for label, d in views.items():
        d = np.array(d) / np.linalg.norm(d)
        cam.location = centre + d * size * 1.6
        direction = centre - np.array(cam.location)
        from mathutils import Vector
        cam.rotation_euler = Vector(direction).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = os.path.join(directory, label + ".jpg")
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)
    bpy.data.objects.remove(cam)


def report():
    nets = {o.name: world_net(o) for o in bpy.data.objects if o.type == 'SURFACE'}
    for r in sorted(measure_seams(nets, sc), key=lambda r: (r["a"], r["b"])):
        print("seam %s | %s : %s (angle %.3g deg, curvature %.0f%%)"
              % (r["a"], r["b"], r["grade"], r["angle"], 100 * r["curvature"]))
    only(*[o for o in bpy.data.objects if o.type == 'SURFACE'][:1])
    for o in ctx.selected_objects:
        o.select_set(False)
    bpy.ops.spline_surface.diagnose()
    for i in json.loads(sc.diagnose_report):
        print("diagnose %s: %s -> %s" % (i["kind"], i["text"], i["hint"]))


if __name__ == "__main__":
    report()
    ob("Lip").hide_set(True)                       # a construction patch for the sweep
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
    print("saved", os.path.abspath(out))
    surfaces = [o for o in bpy.data.objects if o.type == 'SURFACE' and o.visible_get()]
    only(*surfaces)
    step("to mesh", bpy.ops.spline_surface.to_mesh())
    mesh = ctx.active_object
    if mesh and mesh.type == 'MESH':
        print("mesh: %d faces" % len(mesh.data.polygons))
        mesh.hide_set(True)
    if render_dir:
        render_views(render_dir)
    print("PAINS:", PAINS or "none")
