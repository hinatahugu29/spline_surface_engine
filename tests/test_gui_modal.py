"""GUI check of the click tools with real (simulated) mouse and key events.

    blender --factory-startup --enable-event-simulate --python tests/test_gui_modal.py -- LOG

Blender only lets a script send events when started with
--enable-event-simulate. The script hovers a seam and an edge on screen,
clicks, presses Enter / C / the mouse wheel / Esc the way a user would, and
checks what the Fillet, Flange, Fill Hole and Duplicate Edge tools made. LOG (default:
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


def send(kind, value='PRESS', at=None, shift=False):
    win = view()[0]
    x, y = at if at is not None else (win.width // 2, win.height // 2)
    win.event_simulate(type=kind, value=value, x=x, y=y, shift=shift)


def press(kind, at=None, shift=False):
    send(kind, 'PRESS', at, shift)
    send(kind, 'RELEASE', at, shift)


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
         s_fill_setup, s_fill_start, s_fill_click, s_fill_enter, s_fill_check, s_done]


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
