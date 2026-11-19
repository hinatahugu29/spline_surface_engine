"""GUI check of the click tools with real (simulated) mouse and key events.

    blender --factory-startup --enable-event-simulate --python tests/test_gui_modal.py -- LOG

Blender only lets a script send events when started with
--enable-event-simulate. The script hovers a seam and an edge on screen,
clicks, presses Enter / C / the mouse wheel / Esc the way a user would, and
checks what the Fillet, Flange, Fill Hole, Duplicate Edge, Bridge (E: edges /
curves) and Loft (click order) tools made. LOG (default:
tests/gui_modal.log) ends with "SUMMARY ALL OK" or the failed checks.
"""
import json
import os
import sys
import traceback

import bpy
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Euler, Vector

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import spline_surface  # noqa: E402
from spline_surface.bridge.write_surface import build_surface_data  # noqa: E402
from spline_surface.core.split import iso_line  # noqa: E402
from spline_surface.recipe import META_KEY, edge_polyline, raw_meta, store_meta, world_net  # noqa: E402

extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
LOG = open(extra[0] if extra else os.path.join(ROOT, "tests", "gui_modal.log"), "w",
           encoding="utf-8")
RESULTS = []


def log(*a):
    LOG.write(" ".join(str(x) for x in a) + "\n")
    LOG.flush()


def check(tag, ok, detail=""):
    RESULTS.append((tag, bool(ok)))
    log(("OK  " if ok else "BAD ") + tag, detail)


def view():
    win = bpy.context.window_manager.windows[0]
    area = max((a for a in win.screen.areas if a.type == 'VIEW_3D'),
               key=lambda a: a.width * a.height)
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return win, area, region


def to_window(p):
    """Window pixel position of the world point p."""
    win, area, region = view()
    rv3d = area.spaces[0].region_3d
    x, y = location_3d_to_region_2d(region, rv3d, Vector(p))
    return int(region.x + x), int(region.y + y)


def send(kind, value='PRESS', at=None, shift=False, ctrl=False):
    win = view()[0]
    x, y = at if at is not None else (win.width // 2, win.height // 2)
    win.event_simulate(type=kind, value=value, x=x, y=y, shift=shift, ctrl=ctrl)


def press(kind, at=None, shift=False, ctrl=False):
    send(kind, 'PRESS', at, shift, ctrl)
    send(kind, 'RELEASE', at, shift, ctrl)


def free(name, net):
    ob = bpy.data.objects.new(name, build_surface_data(name, net, 12))
    bpy.context.collection.objects.link(ob)
    store_meta(ob, raw_meta(net, world_net(ob)[1]))
    return ob


S = {}


def s_setup():
    bpy.context.preferences.view.smooth_view = 0     # views jump at once (factory settings, not saved)
    spline_surface.register()
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    xs, ys = np.meshgrid(np.linspace(0, 2, 4), np.linspace(0, 2, 4), indexing="ij")
    top = np.stack([xs, ys, np.zeros_like(xs)], -1)              # flat, at z = 0
    zs = np.linspace(0, -1.5, 4)
    wall = np.stack(np.broadcast_arrays(xs[:, :1] * 0 + np.linspace(0, 2, 4)[:, None],
                                        np.zeros((4, 4)), zs[None, :]), -1)
    free("Top", top)
    free("Wall", wall)                                           # meets Top along y = 0 at 90 deg
    win, area, region = view()
    sp = area.spaces[0]
    sp.region_3d.view_perspective = 'PERSP'
    sp.region_3d.view_rotation = Euler((1.0, 0.0, 0.5)).to_quaternion()
    for o in bpy.context.scene.objects:
        o.select_set(o.type == 'SURFACE')
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.view3d.view_selected()
    for o in bpy.context.scene.objects:
        o.select_set(False)
    S["count"] = len(bpy.data.objects)


def s_fillet_start():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("fillet invoke", bpy.ops.spline_surface.fillet('INVOKE_DEFAULT'))
    seam_mid = edge_polyline(*world_net(bpy.data.objects["Top"]), 'V0', 9)[4]
    S["seam_at"] = to_window(seam_mid)
    send('MOUSEMOVE', 'NOTHING', S["seam_at"])


def s_fillet_pick():
    press('LEFTMOUSE', S["seam_at"])


def s_fillet_enter():
    press('RET', S["seam_at"])


def s_fillet_adjust():
    send('WHEELUPMOUSE', 'PRESS', S["seam_at"])           # radius 0.1 -> 0.11
    press('C', S["seam_at"])                              # G2 -> G1


def s_fillet_confirm():
    press('RET', S["seam_at"])


def s_fillet_check():
    made = [o for o in bpy.data.objects if META_KEY in o
            and json.loads(o[META_KEY]).get("method") == "blend"
            and "fillet" in json.loads(o[META_KEY])]
    check("Fillet: hover seam, click, Enter -> a rounded bridge", len(made) == 1,
          [o.name for o in made])
    trims = [json.loads(o[META_KEY]) for o in bpy.data.objects if META_KEY in o
             and json.loads(o[META_KEY]).get("method") == "fillet_trim"]
    check("Fillet: both patches trimmed back", len(trims) == 2, len(trims))
    if trims:
        rec = trims[0]["fillet"]
        check("Fillet: mouse wheel scales the radius (0.1 -> 0.11)",
              abs(rec["radius"] - 0.11) < 1e-6, rec["radius"])
        check("Fillet: C switches G2 -> G1", rec["continuity"] == "G1", rec["continuity"])
    S["count"] = len(bpy.data.objects)


def s_esc_start():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("flange invoke (to cancel)", bpy.ops.spline_surface.flange('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', S["seam_at"])


def s_esc():
    press('ESC', S["seam_at"])


def s_esc_check():
    win, area, region = view()
    check("Esc cancels a click tool and leaves nothing behind",
          len(bpy.data.objects) == S["count"], len(bpy.data.objects))


def _top_point(u, v):
    from spline_surface.core.nurbs import eval_points
    net, orders = world_net(bpy.data.objects["Top"])
    return to_window(eval_points(net, orders, np.array([u]), np.array([v]))[0])


def _new_curves():
    return [o for o in bpy.data.objects if o.type == 'CURVE' and o.name not in S["curves_before"]]


def _off_iso(ob, direction, t):
    """Largest distance of the drawn curve (handles included) from the patch iso-line."""
    from spline_surface.bridge.read_curves import read_polyline
    net, orders = world_net(bpy.data.objects["Top"])
    want = iso_line(net, orders, direction, t, samples=2000)
    got = read_polyline(ob, None, 16)
    return float(max(np.linalg.norm(want - p, axis=1).min() for p in got))


def s_dup_start():
    S["curves_before"] = {o.name for o in bpy.data.objects if o.type == 'CURVE'}
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("duplicate edge invoke", bpy.ops.spline_surface.duplicate_edge_curve('INVOKE_DEFAULT'))
    S["dup_at"] = [_top_point(1.0, 0.5), _top_point(0.0, 0.5)]     # Top's edges u = 1 and u = 0


def s_dup_shift_clicks():
    for at in S["dup_at"]:
        send('MOUSEMOVE', 'NOTHING', at)
        press('LEFTMOUSE', at, shift=True)


def s_dup_shift_check():
    check("Duplicate Edge: Shift+click keeps picking (nothing made yet)", not _new_curves(),
          [o.name for o in _new_curves()])


def s_dup_enter():
    press('RET', S["dup_at"][0])


def s_dup_check():
    made = _new_curves()
    check("Duplicate Edge: Shift+click two edges, Enter -> two curves", len(made) == 2,
          [o.name for o in made])
    if len(made) == 2:
        offs = sorted(min(_off_iso(o, 'U', 1.0), _off_iso(o, 'U', 0.0)) for o in made)
        check("Duplicate Edge: the curves lie on the patch edges u = 1 and u = 0",
              offs[-1] < 2e-3, offs)
    S["curves_before"] = {o.name for o in bpy.data.objects if o.type == 'CURVE'}


def s_dup_d_start():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("duplicate edge invoke", bpy.ops.spline_surface.duplicate_edge_curve('INVOKE_DEFAULT'))
    S["mid_at"] = _top_point(0.5, 0.5)
    send('MOUSEMOVE', 'NOTHING', S["mid_at"])
    press('D', S["mid_at"])                                   # U -> V


def s_dup_click():
    send('MOUSEMOVE', 'NOTHING', S["mid_at"])
    press('LEFTMOUSE', S["mid_at"])


def s_dup_d_check():
    made = _new_curves()
    check("Duplicate Edge: D switches to V, a plain click makes one curve and finishes",
          len(made) == 1, [o.name for o in made])
    if made:
        off = _off_iso(made[0], 'V', 0.5)
        check("Duplicate Edge: the curve is the iso-line v = 0.5 across the patch", off < 2e-3, off)
    S["count"] = len(bpy.data.objects)


def s_flange_start():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("flange invoke", bpy.ops.spline_surface.flange('INVOKE_DEFAULT'))
    top = bpy.data.objects["Top"]
    far = edge_polyline(*world_net(top), 'V1', 9)[4]     # Top's free edge at y = 2
    S["edge_at"] = to_window(far)
    send('MOUSEMOVE', 'NOTHING', S["edge_at"])


def s_flange_click():
    press('LEFTMOUSE', S["edge_at"])


def _base(meta):
    while meta.get("method") == "fillet_trim":     # Flange rounds its fold with a Fillet
        meta = meta["base"]
    return meta


def s_flange_check():
    walls = [_base(json.loads(o[META_KEY])) for o in bpy.data.objects if META_KEY in o]
    walls = [m for m in walls if m.get("method") == "flange"]
    check("Flange: hover a free edge, click -> a wall (its fold rounded)", len(walls) == 1,
          len(walls))
    if walls:
        check("Flange: the wall is on the edge under the mouse",
              walls[0]["sources"][0] == "Top" and walls[0]["edge"] == "V1", walls[0]["edge"])


def s_fill_setup():
    # A square hole at x, y in [23, 25] ringed by four flat strips.
    def strip(name, x0, x1, y0, y1):
        xs, ys = np.meshgrid(np.linspace(x0, x1, 4), np.linspace(y0, y1, 4), indexing="ij")
        return free(name, np.stack([xs, ys, np.zeros_like(xs)], -1))
    S["ring"] = [strip("L", 22, 23, 23, 25), strip("R", 25, 26, 23, 25),
                 strip("B", 23, 25, 22, 23), strip("T", 23, 25, 25, 26)]
    win, area, region = view()
    area.spaces[0].region_3d.view_rotation = Euler((0.6, 0.0, 0.3)).to_quaternion()
    for o in bpy.context.scene.objects:
        o.select_set(o in S["ring"])
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.view3d.view_selected()
    for o in bpy.context.scene.objects:
        o.select_set(False)


def s_fill_start():
    # The edge of each strip that faces the hole (read once the view has moved).
    rim = {"L": 'U1', "R": 'U0', "B": 'V1', "T": 'V0'}
    S["rim_at"] = [to_window(edge_polyline(*world_net(bpy.data.objects[n]), e, 9)[4])
                   for n, e in rim.items()]
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("fill hole invoke", bpy.ops.spline_surface.fill_hole('INVOKE_DEFAULT'))
    S["fill_clicks"] = list(S["rim_at"])


def s_fill_click():
    at = S["fill_clicks"].pop(0)
    send('MOUSEMOVE', 'NOTHING', at)
    press('LEFTMOUSE', at)
    if S["fill_clicks"]:
        STEPS.insert(0, s_fill_click)


def s_fill_enter():
    press('RET', S["rim_at"][0])


def s_fill_check():
    fills = [json.loads(o[META_KEY]) for o in bpy.data.objects if META_KEY in o
             and json.loads(o[META_KEY]).get("method") == "fill"]
    check("Fill Hole: click the four rim edges, Enter -> one patch fills the hole",
          len(fills) == 1 and sorted(fills[0]["sources"]) == ["B", "L", "R", "T"],
          [f.get("sources") for f in fills])


# ---- E: edges / curves. A curve duplicated from an edge lies right on it.
def _focus(objs):
    win, area, region = view()
    for o in bpy.context.scene.objects:
        o.select_set(o in objs)
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.view3d.view_selected()
    for o in bpy.context.scene.objects:
        o.select_set(False)


def _mid(name, edge):
    return to_window(edge_polyline(*world_net(bpy.data.objects[name]), edge, 9)[4])


def s_kind_setup():
    xs, ys = np.meshgrid(np.linspace(40, 42, 4), np.linspace(0, 2, 4), indexing="ij")
    flat = np.stack([xs, ys, np.zeros_like(xs)], -1)
    S["kp"] = [free("KA", flat), free("KB", flat + np.array([0.0, 4.0, 0.0]))]
    assert bpy.ops.spline_surface.duplicate_edge_curve(
        edges=json.dumps([["KA", "V", 1.0]])) == {'FINISHED'}
    S["kcurve"] = bpy.context.active_object.name                 # lies on KA's edge V1
    area = view()[1]
    area.spaces[0].region_3d.view_rotation = Euler((0.9, 0.0, 0.4)).to_quaternion()
    _focus(S["kp"])


def _bridges():
    return [json.loads(o[META_KEY]) for o in bpy.data.objects if META_KEY in o
            and json.loads(o[META_KEY]).get("method") == "blend"
            and "fillet" not in json.loads(o[META_KEY])]


def s_kind_edges():
    S["kat"], S["kbt"] = _mid("KA", 'V1'), _mid("KB", 'V0')
    S["nb"] = len(_bridges())
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("bridge invoke", bpy.ops.spline_surface.bridge_edges('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', S["kat"])
    press('E', S["kat"])                                          # edges only
    press('LEFTMOUSE', S["kat"])
    send('MOUSEMOVE', 'NOTHING', S["kbt"])
    press('LEFTMOUSE', S["kbt"])


def s_kind_edges_check():
    new = _bridges()[S["nb"]:]
    check("Bridge, E = edges only: the click on the shared spot takes the patch edge",
          len(new) == 1 and new[0]["sources"] == ["KA", "KB"] and new[0]["edges"] == ["V1", "V0"],
          [(b["sources"], b["edges"]) for b in new])
    for o in [o for o in bpy.data.objects if META_KEY in o and o.name.startswith("KA_blend")]:
        bpy.data.objects.remove(o)        # its seams would hide KA's and KB's edges next time
    S["nb"] = len(_bridges())


def s_kind_curves():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("bridge invoke", bpy.ops.spline_surface.bridge_edges('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', S["kat"])
    press('E', S["kat"])
    press('E', S["kat"])                                          # curves only
    press('LEFTMOUSE', S["kat"])
    send('MOUSEMOVE', 'NOTHING', S["kbt"])                        # then edges only, by itself
    press('LEFTMOUSE', S["kbt"])


def s_kind_curves_check():
    new = _bridges()[S["nb"]:]
    check("Bridge, E = curves only: the same spot takes the curve; the second pick is an edge",
          len(new) == 1 and new[0]["sources"] == ["KB", S["kcurve"]]
          and new[0]["edges"] == ["V0", "CURVE"],
          [(b["sources"], b["edges"]) for b in new])


# ---- Loft: click the sections in order
def s_loft_setup():
    from spline_surface.projecting import make_bezier_curve
    S["secs"] = []
    for k, x in enumerate((50.0, 51.5, 53.0)):
        pts = np.array([(x, 0, 0), (x, 1, 0.5 + 0.2 * k), (x, 2, 0)])
        S["secs"].append(make_bezier_curve("Sec%d" % k, pts, bpy.context.collection))
    _focus(S["secs"])


def s_loft_clicks():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("loft invoke", bpy.ops.spline_surface.loft('INVOKE_DEFAULT'))
    at = {o.name: to_window(o.matrix_world @ o.data.splines[0].bezier_points[1].co)
          for o in S["secs"]}
    for name in ("Sec2", "Sec0"):
        send('MOUSEMOVE', 'NOTHING', at[name])
        press('LEFTMOUSE', at[name])
    press('BACK_SPACE', at["Sec0"])                               # drop Sec0 again
    for name in ("Sec1", "Sec0"):
        send('MOUSEMOVE', 'NOTHING', at[name])
        press('LEFTMOUSE', at[name])
    S["loft_at"] = at["Sec0"]


def s_loft_enter():
    press('RET', S["loft_at"])


def s_loft_check():
    lofts = [json.loads(o[META_KEY]) for o in bpy.data.objects if META_KEY in o
             and json.loads(o[META_KEY]).get("method") == "loft"]
    check("Loft: click Sec2, Sec0, Backspace, Sec1, Sec0, Enter -> lofted in click order",
          len(lofts) == 1 and lofts[0]["sources"] == ["Sec2", "Sec1", "Sec0"],
          [m["sources"] for m in lofts])


# ---- Curve Points (modal) and Slide Points (edit mode)
def _chain(name):
    from spline_surface.curveediting import read_spline
    return read_spline(bpy.data.objects[name], 0)


def _chain_at(name, t):
    from spline_surface.core.curvepoint import point_at
    A, L, R, cyc = _chain(name)
    return point_at(A, L, R, [t], cyc)[0]


def _shape_off(before, name):
    from spline_surface.core.curvepoint import nearest, point_at, segments
    A, L, R, cyc = _chain(name)
    m = segments(len(A), cyc)
    return max(nearest(*before[:3], p, before[3])[1]
               for p in point_at(A, L, R, np.linspace(0, m, 30 * m + 1), cyc))


def s_cp_setup():
    from spline_surface.projecting import make_bezier_curve
    ob = make_bezier_curve("CPv", np.array([(60, 0, 0), (62, 1.5, 0), (64, 0, 0), (66, 1.5, 0)],
                                           float), bpy.context.collection)
    ob.data.dimensions = '3D'
    area = view()[1]
    area.spaces[0].region_3d.view_rotation = Euler((0.0, 0.0, 0.0)).to_quaternion()   # top view
    _focus([ob])
    bpy.context.view_layer.update()
    S["cp0"] = _chain("CPv")


def s_cp_insert():
    win, area, region = view()
    S["cp_at"] = to_window(_chain_at("CPv", 0.5))
    send('MOUSEMOVE', 'NOTHING', S["cp_at"])
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("curve points invoke", bpy.ops.spline_surface.curve_points('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', S["cp_at"])
    press('LEFTMOUSE', S["cp_at"])


def s_cp_insert_check():
    A = _chain("CPv")[0]
    size = float(np.ptp(S["cp0"][0], 0).max())
    check("Curve Points: click on a curve adds a point, shape unchanged",
          len(A) == 5 and _shape_off(S["cp0"], "CPv") < 1e-4 * size,
          (len(A), _shape_off(S["cp0"], "CPv")))


def s_cp_slide():
    # drag point 2 (the old second point) halfway towards point 3, along the curve
    S["cp_from"] = _chain("CPv")[0][2].copy()
    start, goal = to_window(S["cp_from"]), to_window(_chain_at("CPv", 2.5))
    send('MOUSEMOVE', 'NOTHING', start)
    send('LEFTMOUSE', 'PRESS', start)
    mid = ((start[0] + goal[0]) // 2, (start[1] + goal[1]) // 2)
    send('MOUSEMOVE', 'NOTHING', mid)
    send('MOUSEMOVE', 'NOTHING', goal)
    send('LEFTMOUSE', 'RELEASE', goal)
    S["cp_goal"] = _chain_at("CPv", 2.5)


def s_cp_slide_check():
    A = _chain("CPv")[0]
    size = float(np.ptp(S["cp0"][0], 0).max())
    moved = float(np.linalg.norm(A[2] - S["cp_from"]))
    off = _shape_off(S["cp0"], "CPv")
    check("Curve Points: drag a point -> it slides along the curve, shape kept",
          len(A) == 5 and moved > 0.3 and float(np.linalg.norm(A[2] - S["cp_goal"])) < 0.05 * size
          and off < 1e-2 * size, (moved, float(np.linalg.norm(A[2] - S["cp_goal"])), off))   # a sharp zigzag


def s_cp_dissolve():
    at = to_window(_chain("CPv")[0][3])
    send('MOUSEMOVE', 'NOTHING', at)
    press('LEFTMOUSE', at, ctrl=True)


def s_cp_dissolve_check():
    n = len(_chain("CPv")[0])
    size = float(np.ptp(S["cp0"][0], 0).max())
    check("Curve Points: Ctrl+click a point removes it, shape close",
          n == 4 and _shape_off(S["cp0"], "CPv") < 2e-2 * size, (n, _shape_off(S["cp0"], "CPv")))


def s_cp_undo_enter():
    at = S["cp_at"]
    press('Z', at, ctrl=True)                    # back to 5 points
    press('RET', at)


def s_cp_undo_check():
    n = len(_chain("CPv")[0])
    check("Curve Points: Ctrl+Z takes back the last edit, Enter keeps the rest", n == 5, n)


def s_es_setup():
    ob = bpy.data.objects["CPv"]
    for o in bpy.context.scene.objects:
        o.select_set(o is ob)
    bpy.context.view_layer.objects.active = ob
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.object.mode_set(mode='EDIT')
    for k, b in enumerate(ob.data.splines[0].bezier_points):
        b.select_control_point = b.select_left_handle = b.select_right_handle = (k == 1)
    S["es0"] = _chain("CPv")
    S["es_from"] = S["es0"][0][1].copy()


def s_es_slide():
    win, area, region = view()
    a, b = to_window(S["es0"][0][1]), to_window(_chain_at("CPv", 1.6))
    send('MOUSEMOVE', 'NOTHING', a)
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("slide points invoke", bpy.ops.spline_surface.slide_curve_points('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))
    send('MOUSEMOVE', 'NOTHING', b)
    press('LEFTMOUSE', b)


def s_es_check():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.object.mode_set(mode='OBJECT')
    A = _chain("CPv")[0]
    size = float(np.ptp(S["es0"][0], 0).max())
    moved = float(np.linalg.norm(A[1] - S["es_from"]))
    off = _shape_off(S["es0"], "CPv")
    check("Slide Points (edit mode): the selected point slides along the curve, shape kept",
          moved > 0.2 and off < 5e-3 * size, (moved, off))


# ---- Curve Pull: drag on a curve; a crossing curve follows
def _gap(a, b):
    """Closest distance between two curves: the nearest sample pair, then projected
    back and forth onto each curve until it settles (samples alone miss a crossing)."""
    from spline_surface.core.curvepoint import nearest, point_at, segments
    A, L, R, c = _chain(a)
    B, M, N, d = _chain(b)
    pa = point_at(A, L, R, np.linspace(0, segments(len(A), c), 400), c)
    pb = point_at(B, M, N, np.linspace(0, segments(len(B), d), 400), d)
    i, j = np.unravel_index(np.linalg.norm(pa[:, None] - pb[None], axis=2).argmin(), (400, 400))
    q = pb[j]
    for _ in range(30):
        ta, _ = nearest(A, L, R, q, c)
        p = point_at(A, L, R, [ta], c)[0]
        tb, _ = nearest(B, M, N, p, d)
        q = point_at(B, M, N, [tb], d)[0]
    return float(np.linalg.norm(p - q))


def s_pull_setup():
    from spline_surface.projecting import make_bezier_curve
    h = make_bezier_curve("PH", np.array([(70, 0, 0), (72, 0.1, 0), (74, -0.1, 0), (76, 0, 0)], float),
                          bpy.context.collection)
    v = make_bezier_curve("PV", np.array([(73, -3, 0), (73.1, -1, 0), (72.9, 1, 0), (73, 3, 0)], float),
                          bpy.context.collection)
    for o in (h, v):
        o.data.dimensions = '3D'
    _focus([h, v])
    for o in bpy.context.scene.objects:
        o.select_set(o in (h, v))
    bpy.context.view_layer.update()
    S["ph0"], S["pv0"] = _chain("PH"), _chain("PV")
    S["gap0"] = _gap("PH", "PV")


def s_pull_drag():
    from bpy_extras.view3d_utils import region_2d_to_location_3d
    win, area, region = view()
    p = _chain_at("PH", 0.8)
    a = to_window(p)
    b = (a[0], a[1] + 60)
    send('MOUSEMOVE', 'NOTHING', a)
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("curve points invoke (pull)", bpy.ops.spline_surface.curve_points('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', a)
    send('LEFTMOUSE', 'PRESS', a)
    send('MOUSEMOVE', 'NOTHING', (a[0], a[1] + 30))
    send('MOUSEMOVE', 'NOTHING', b)
    send('LEFTMOUSE', 'RELEASE', b)
    press('RET', b)
    rv3d = area.spaces[0].region_3d
    S["pull_goal"] = np.array(region_2d_to_location_3d(region, rv3d, (b[0] - region.x, b[1] - region.y),
                                                       Vector(p)))
    S["pull_from"] = p


def s_pull_check():
    from spline_surface.core.curvepoint import nearest
    A, L, R, c = _chain("PH")
    t, d = nearest(A, L, R, S["pull_goal"], c)
    moved = float(np.linalg.norm(S["pull_goal"] - S["pull_from"]))
    check("Curve Pull: drag on a curve -> it bends to the mouse (point count unchanged)",
          len(A) == 4 and moved > 0.3 and d < 0.02 * moved, (len(A), moved, d))
    vmoved = float(np.abs(_chain("PV")[0] - S["pv0"][0]).max())
    check("Curve Pull: the crossing curve follows and the two still cross",
          vmoved > 0.05 and _gap("PH", "PV") < 1e-3, (vmoved, _gap("PH", "PV")))
    ends = (np.allclose(A[0], S["ph0"][0][0]) and np.allclose(A[-1], S["ph0"][0][-1]))
    check("Curve Pull: ends pinned by default", ends, (A[0], A[-1]))


# ---- Constrained Move (edit mode)
def s_cm_setup():
    from spline_surface.projecting import make_bezier_curve
    ob = make_bezier_curve("CMv", np.array([(80, 0, 0), (82, 1.5, 0), (84, 2, 0), (86, 1.5, 0),
                                            (88, 0, 0)], float), bpy.context.collection)
    ob.data.dimensions = '3D'
    area = view()[1]
    area.spaces[0].region_3d.view_rotation = Euler((0.0, 0.0, 0.0)).to_quaternion()
    _focus([ob])
    for o in bpy.context.scene.objects:
        o.select_set(o is ob)
    bpy.context.view_layer.objects.active = ob
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.object.mode_set(mode='EDIT')
    for k, b in enumerate(ob.data.splines[0].bezier_points):
        b.select_control_point = b.select_left_handle = b.select_right_handle = (k == 2)
    S["cm0"] = _chain("CMv")


def s_cm_move():
    from spline_surface.core.curvesmooth import frame
    win, area, region = view()
    A, L, R, c = S["cm0"]
    t, nrm, b = frame(A, L, R, 2, c)
    a = to_window(A[2])
    goal = to_window(A[2] + 0.8 * t)
    send('MOUSEMOVE', 'NOTHING', a)
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("constrained move invoke", bpy.ops.spline_surface.constrained_move('INVOKE_DEFAULT'))
    press('T', a)                                # along the tangent
    send('MOUSEMOVE', 'NOTHING', ((a[0] + goal[0]) // 2, (a[1] + goal[1]) // 2))
    send('MOUSEMOVE', 'NOTHING', goal)
    press('LEFTMOUSE', goal)
    S["cm_t"] = t


def s_cm_check():
    win, area, region = view()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.object.mode_set(mode='OBJECT')
    A = _chain("CMv")[0]
    d = A[2] - S["cm0"][0][2]
    along = float(d @ S["cm_t"])
    across = float(np.linalg.norm(d - along * S["cm_t"]))
    others = float(np.abs(np.delete(A, 2, axis=0) - np.delete(S["cm0"][0], 2, axis=0)).max())
    check("Constrained Move: T, then the mouse -> the point moves along its own tangent only",
          0.5 < along < 1.1 and across < 1e-3 and others < 1e-6, (along, across, others))


# ---- Curve Points: drag a curve end onto a patch edge (snaps)
def s_snap_setup():
    from spline_surface.projecting import make_bezier_curve
    xs, ys = np.meshgrid(np.linspace(90, 92, 4), np.linspace(0, 2, 4), indexing="ij")
    patch = free("EP", np.stack([xs, ys, np.zeros_like(xs)], -1))
    c = make_bezier_curve("EC", np.array([(93.5, 1.4, 0.3), (95, 1.0, 0.6), (96.5, 1.2, 0.2)], float),
                          bpy.context.collection)
    c.data.dimensions = '3D'
    area = view()[1]
    area.spaces[0].region_3d.view_rotation = Euler((0.5, 0.0, 0.2)).to_quaternion()
    _focus([patch, c])
    for o in bpy.context.scene.objects:
        o.select_set(o is c)
    bpy.context.view_layer.objects.active = c


def s_snap_drag():
    win, area, region = view()
    start = to_window(_chain("EC")[0][0])
    goal = to_window(edge_polyline(*world_net(bpy.data.objects["EP"]), 'U1', 9)[4])   # x = 92
    send('MOUSEMOVE', 'NOTHING', start)
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("curve points invoke (end)", bpy.ops.spline_surface.curve_points('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', start)
    send('LEFTMOUSE', 'PRESS', start)
    send('MOUSEMOVE', 'NOTHING', ((start[0] + goal[0]) // 2, (start[1] + goal[1]) // 2))
    send('MOUSEMOVE', 'NOTHING', goal)
    send('LEFTMOUSE', 'RELEASE', goal)
    press('RET', goal)


def s_snap_check():
    p = _chain("EC")[0][0]
    check("Curve Points: drag a curve end onto a patch edge -> it lands on the edge",
          abs(p[0] - 92.0) < 1e-3 and abs(p[2]) < 1e-3 and 0.0 < p[1] < 2.0, p)


# ---- Curve Pull button on a 2-point curve (Blender's default Bezier): it must bend
def s_p2_setup():
    from spline_surface.projecting import make_bezier_curve
    ob = make_bezier_curve("P2", np.array([(100, 0, 0), (104, 0, 0)], float), bpy.context.collection)
    ob.data.dimensions = '3D'
    sp = ob.data.splines[0]
    sp.bezier_points[0].handle_left_type = sp.bezier_points[0].handle_right_type = 'FREE'
    sp.bezier_points[1].handle_left_type = sp.bezier_points[1].handle_right_type = 'FREE'
    sp.bezier_points[0].handle_right = (101, 1, 0)
    sp.bezier_points[1].handle_left = (103, -1, 0)
    area = view()[1]
    area.spaces[0].region_3d.view_rotation = Euler((0.0, 0.0, 0.0)).to_quaternion()
    _focus([ob])
    for o in bpy.context.scene.objects:
        o.select_set(o is ob)
    bpy.context.view_layer.objects.active = ob
    S["p2_0"] = _chain("P2")


def s_p2_drag():
    from bpy_extras.view3d_utils import region_2d_to_location_3d
    win, area, region = view()
    p = _chain_at("P2", 0.5)
    a = to_window(p)
    b = (a[0], a[1] + 50)
    send('MOUSEMOVE', 'NOTHING', a)
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("curve pull invoke", bpy.ops.spline_surface.curve_pull('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', a)
    send('LEFTMOUSE', 'PRESS', a)
    send('MOUSEMOVE', 'NOTHING', (a[0], a[1] + 25))
    send('MOUSEMOVE', 'NOTHING', b)
    send('LEFTMOUSE', 'RELEASE', b)
    press('RET', b)
    rv3d = area.spaces[0].region_3d
    S["p2_goal"] = np.array(region_2d_to_location_3d(region, rv3d, (b[0] - region.x, b[1] - region.y),
                                                     Vector(p)))
    S["p2_from"] = p


def s_p2_check():
    from spline_surface.core.curvepoint import nearest
    A, L, R, c = _chain("P2")
    moved = float(np.linalg.norm(S["p2_goal"] - S["p2_from"]))
    t, d = nearest(A, L, R, S["p2_goal"], c)
    ends = np.allclose(A[0], S["p2_0"][0][0]) and np.allclose(A[-1], S["p2_0"][0][-1])
    check("Curve Pull button, 2-point curve: a point is added, the curve bends to the mouse, "
          "ends stay", len(A) == 3 and moved > 0.3 and d < 0.02 * moved and ends,
          (len(A), moved, d, ends))


# ---- Curve Pull: X, then drag an end -> it moves along X only
def s_ax_setup():
    from spline_surface.projecting import make_bezier_curve
    ob = make_bezier_curve("AXc", np.array([(110, 0, 0), (110.5, 2, 0.6), (110, 4, 0)], float),
                           bpy.context.collection)
    ob.data.dimensions = '3D'
    area = view()[1]
    area.spaces[0].region_3d.view_rotation = Euler((0.9, 0.0, 0.5)).to_quaternion()
    _focus([ob])
    for o in bpy.context.scene.objects:
        o.select_set(o is ob)
    bpy.context.view_layer.objects.active = ob
    S["ax0"] = _chain("AXc")[0][0].copy()


def s_ax_drag():
    win, area, region = view()
    a = to_window(S["ax0"])
    b = (a[0] + 70, a[1] + 40)                    # diagonal on screen
    send('MOUSEMOVE', 'NOTHING', a)
    with bpy.context.temp_override(window=win, area=area, region=region):
        log("curve pull invoke (end, X)", bpy.ops.spline_surface.curve_pull('INVOKE_DEFAULT'))
    send('MOUSEMOVE', 'NOTHING', a)
    press('X', a)
    send('LEFTMOUSE', 'PRESS', a)
    send('MOUSEMOVE', 'NOTHING', (a[0] + 35, a[1] + 20))
    send('MOUSEMOVE', 'NOTHING', b)
    send('LEFTMOUSE', 'RELEASE', b)
    press('RET', b)


def s_ax_check():
    p = _chain("AXc")[0][0]
    d = p - S["ax0"]
    check("Curve Pull: X, then drag an end -> it moves along X only",
          abs(d[0]) > 0.1 and abs(d[1]) < 1e-4 and abs(d[2]) < 1e-4, d)


def s_done():
    ok = all(r for _, r in RESULTS)
    log("SUMMARY", "ALL OK" if ok else "FAILURES", [t for t, r in RESULTS if not r])
    LOG.close()
    bpy.ops.wm.quit_blender()


STEPS = [s_setup, s_fillet_start, s_fillet_pick, s_fillet_enter, s_fillet_adjust,
         s_fillet_confirm, s_fillet_check, s_esc_start, s_esc, s_esc_check,
         s_dup_start, s_dup_shift_clicks, s_dup_shift_check, s_dup_enter, s_dup_check,
         s_dup_d_start, s_dup_click, s_dup_d_check,
         s_flange_start, s_flange_click, s_flange_check,
         s_fill_setup, s_fill_start, s_fill_click, s_fill_enter, s_fill_check,
         s_kind_setup, s_kind_edges, s_kind_edges_check, s_kind_curves, s_kind_curves_check,
         s_loft_setup, s_loft_clicks, s_loft_enter, s_loft_check, s_cp_setup, s_cp_insert, s_cp_insert_check, s_cp_slide, s_cp_slide_check,
         s_cp_dissolve, s_cp_dissolve_check, s_cp_undo_enter, s_cp_undo_check,
         s_es_setup, s_es_slide, s_es_check,
         s_pull_setup, s_pull_drag, s_pull_check,
         s_cm_setup, s_cm_move, s_cm_check,
         s_snap_setup, s_snap_drag, s_snap_check,
         s_p2_setup, s_p2_drag, s_p2_check,
         s_ax_setup, s_ax_drag, s_ax_check, s_done]


def tick():
    fn = STEPS.pop(0)
    try:
        fn()
    except Exception:
        log("EXC in", fn.__name__, traceback.format_exc())
        RESULTS.append(("exception in " + fn.__name__, False))
        STEPS[:] = [s_done] if fn is not s_done else []
    return 0.6 if STEPS else None


bpy.app.timers.register(tick, first_interval=1.5)
