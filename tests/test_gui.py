"""GUI check: Live Update with real undo/redo, renames and far moves.

Runs in a Blender window (undo only exists with a UI), steps through the
scene on a timer, writes one line per check and quits:

    blender --factory-startup --python tests/test_gui.py -- LOG

LOG is where the report goes (default: tests/gui_check.log). Exits after the last
step; the last line is "SUMMARY ALL OK" or lists the failed checks.
"""
import os
import sys
import json
import traceback

import bpy
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import spline_surface  # noqa: E402
from spline_surface import live  # noqa: E402,F401
from spline_surface.bridge.write_surface import build_surface_data  # noqa: E402
from spline_surface.core.continuity import find_shared_edge, grade, measure  # noqa: E402
from spline_surface.core.extend import extend_net  # noqa: E402
from spline_surface.rebuild import compute_from_meta  # noqa: E402
from spline_surface.recipe import META_KEY, hand_edited, raw_meta, store_meta, world_net  # noqa: E402

extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
LOG = open(extra[0] if extra else os.path.join(ROOT, "tests", "gui_check.log"), "w",
           encoding="utf-8")


def log(*a):
    LOG.write(" ".join(str(x) for x in a) + "\n")
    LOG.flush()


spline_surface.register()
S = {}
def win():
    w = bpy.context.window_manager.windows[0]
    area = next(a for a in w.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return dict(window=w, area=area, region=region)
def op(fn, **kw):
    with bpy.context.temp_override(**win()):
        return fn('EXEC_DEFAULT', True, **kw)
def sel(names):
    for o in bpy.context.scene.objects: o.select_set(False)
    for n in names: bpy.data.objects[n].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[names[0]]
def curve(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER'); sp.bezier_points.add(len(pts) - 1)
    for bp, co in zip(sp.bezier_points, pts):
        bp.co = co; bp.handle_left_type = bp.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu); bpy.context.collection.objects.link(ob); return ob
def free(name, net):
    ob = bpy.data.objects.new(name, build_surface_data(name, net, 12)); bpy.context.collection.objects.link(ob)
    store_meta(ob, raw_meta(net, world_net(ob)[1])); return ob
def out_of_sync():
    bad = []
    for ob in bpy.data.objects:
        if META_KEY not in ob or ob.type != 'SURFACE': continue
        meta = json.loads(ob[META_KEY])
        if meta.get("method") == "raw" and not meta.get("matches"): continue
        if hand_edited(ob, meta): bad.append((ob.name, "hand edited")); continue
        try:
            net, orders, _ = compute_from_meta(dict(meta), bpy.context)
        except ValueError as e:
            bad.append((ob.name, "error " + str(e))); continue
        cur, co = world_net(ob)
        size = float(np.ptp(cur.reshape(-1, 3), 0).max()) or 1
        if cur.shape != np.asarray(net).shape: bad.append((ob.name, "shape %s vs %s" % (cur.shape, np.asarray(net).shape))); continue
        d = float(np.abs(cur - net).max()) / size
        if d > 1e-4: bad.append((ob.name, "off by %.2g" % d))
    return bad
def check(tag):
    bad = out_of_sync()
    log(("OK  " if not bad else "BAD ") + tag, bad if bad else "", "| note:", bpy.context.scene.spline_surface.live_note)
    S.setdefault("results", []).append((tag, not bad))
def net_of(n): return world_net(bpy.data.objects[n])[0].copy()

def s_setup():
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    c = [curve("c0", [(0,0,0),(2,-0.2,0.6),(4,0,0)]), curve("c1", [(4,0,0),(4.3,1.5,0.4),(4,3,0)]),
         curve("c2", [(4,3,0),(2,3.3,0.8),(0,3,0)]), curve("c3", [(0,3,0),(-0.2,1.5,0.3),(0,0,0)])]
    sel([o.name for o in c]); log("coons", op(bpy.ops.spline_surface.coons))
    S["coons"] = bpy.context.active_object.name
    secs = [curve("s%d" % k, [(x, 6, 0), (x, 7, 0.5 + 0.2 * k), (x, 8, 0)]) for k, x in enumerate((0, 1.5, 3))]
    sel([o.name for o in secs]); log("loft", op(bpy.ops.spline_surface.loft, rows=0))
    S["loft"] = bpy.context.active_object.name
    xs, ys = np.meshgrid(np.linspace(0, 3, 4), np.linspace(10, 13, 4), indexing="ij")
    A = np.stack([xs, ys, 0.3 * np.sin(xs)], -1)
    a = free("PA", A); b = free("PB", A + np.array([0, 5, 0.5]))
    log("bridge", op(bpy.ops.spline_surface.bridge_edges, source_a="PA", edge_a='V1', source_b="PB", edge_b='V0', continuity='G2'))
    S["bridge"] = bpy.context.active_object.name
    B = np.array(extend_net(A, "U1", 1.5, "G1", degree=3)[0]); B[1:, :, 2] += 0.5
    f = free("Fol", B)
    sel(["Fol", "PA"]); bpy.context.view_layer.objects.active = f
    log("match", op(bpy.ops.spline_surface.match_edge, continuity='G2'))
    sel(["PB"]); log("offset", op(bpy.ops.spline_surface.offset_surface))
    S["offset"] = bpy.context.active_object.name
    S["coons0"] = net_of(S["coons"])
    log("objects", sorted(o.name for o in bpy.data.objects))
def s_check0(): check("after setup")
def s_move():
    sel(["c0"]); log("move c0", op(bpy.ops.transform.translate, value=(0, 0, 0.5)))
def s_check_move():
    check("curve moved: patches follow")
    S["coons1"] = net_of(S["coons"])
    log("    coons moved by", float(np.abs(S["coons1"] - S["coons0"]).max()))
def s_undo(): log("undo", op(bpy.ops.ed.undo))
def s_check_undo():
    check("after undo: patches back with their curves")
    log("    coons vs original", float(np.abs(net_of(S["coons"]) - S["coons0"]).max()))
def s_redo(): log("redo", op(bpy.ops.ed.redo))
def s_check_redo():
    check("after redo")
    log("    coons vs moved", float(np.abs(net_of(S["coons"]) - S["coons1"]).max()))
def s_rename():
    bpy.data.objects["c1"].name = "RenamedRight"
def s_move_renamed():
    cu = bpy.data.objects["RenamedRight"].data
    cu.splines[0].bezier_points[1].co.x += 0.4; cu.update_tag(); log("bent renamed curve")
def s_check_renamed():
    meta = json.loads(bpy.data.objects[S["coons"]][META_KEY])
    log("    coons sources", meta["sources"])
    check("renamed curve moved: Coons follows")
def s_far():
    sel(["PA"]); log("move PA far", op(bpy.ops.transform.translate, value=(0, 0, 2.0)))
def s_check_far():
    check("Match reference moved far (z+2): follower, bridge, offset follow")
    Pf, of = world_net(bpy.data.objects["Fol"]); Pr, orr = world_net(bpy.data.objects["PA"])
    sh = find_shared_edge(Pf, Pr, 1e-3 * 5, of, orr)
    log("    Fol seam", None if sh is None else grade(measure(Pf, Pr, *sh[:3], of, orr)))
def s_addrow():
    S["off_shape"] = world_net(bpy.data.objects[S["offset"]])[0].shape
    sel(["PB"]); log("add row on PB", op(bpy.ops.spline_surface.insert_line, direction='U'))
def s_check_addrow():
    shape = world_net(bpy.data.objects[S["offset"]])[0].shape
    log("    offset grid", S["off_shape"], "->", shape)
    check("Add Row on a source: dependents rebuilt with new point counts")
def all_valid():
    bad = []
    for ob in bpy.data.objects:
        if ob.type == 'SURFACE':
            try:
                world_net(ob)
            except ValueError as e:
                bad.append(str(e))
    return bad
def s_undo2(): log("undo", op(bpy.ops.ed.undo))
def s_check_undo2():
    bad = all_valid()
    log(("OK  " if not bad else "BAD ") + "undo once after Add Row: every patch still valid", bad)
    S["results"].append(("undo once valid", not bad))
    log("    offset grid", world_net(bpy.data.objects[S["offset"]])[0].shape)
    # v0.82.3: Add Row rebuilds what reads it before it returns, so one undo takes back both
    check("undo once: back before Add Row")
def s_undo3(): log("undo", op(bpy.ops.ed.undo))
def s_check_undo3():
    bad = all_valid()
    S["results"].append(("undo twice valid", not bad))
    check("undo twice: back before the far move, followers with it")
def s_redo2(): log("redo", op(bpy.ops.ed.redo)); log("redo", op(bpy.ops.ed.redo))
def s_check_redo2():
    bad = all_valid()
    S["results"].append(("redo valid", not bad))
    log("    offset grid", world_net(bpy.data.objects[S["offset"]])[0].shape)
    check("redo twice: Add Row and its rebuild back")
def s_done():
    ok = all(r for _, r in S.get("results", []))
    log("SUMMARY", "ALL OK" if ok else "FAILURES", [t for t, r in S["results"] if not r])
    LOG.close()
    bpy.ops.wm.quit_blender()

STEPS = [s_setup, s_check0, s_move, s_check_move, s_undo, s_check_undo, s_redo, s_check_redo,
         s_rename, s_move_renamed, s_check_renamed, s_far, s_check_far, s_addrow, s_check_addrow,
         s_undo2, s_check_undo2, s_undo3, s_check_undo3, s_redo2, s_check_redo2, s_done]
def tick():
    if not STEPS: return None
    fn = STEPS.pop(0)
    try:
        fn()
    except Exception:
        log("EXC in", fn.__name__, traceback.format_exc())
        if fn is not s_done:
            STEPS[:] = [s_done]
    return 1.2
bpy.app.timers.register(tick, first_interval=1.5)
