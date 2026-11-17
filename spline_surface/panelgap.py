"""Panel Gap: measure the clearance between facing edges of neighbouring panels.

While on, every pair of free edges from different patches that face each
other (a panel joint) is drawn as short rungs across the gap, green where the
gap is within tolerance of the target and orange where it is off. The Inspect
panel lists each joint's smallest, largest and mean gap and its step (how far
one panel sits proud of the other). Even Gap moves the active panel's edge so
the joint is the target gap all along (and flush), and keeps it so when the
other panel moves. The geometry lives in core.gap.
"""
import json

import bpy
from bpy.app.handlers import persistent
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .core.gap import edge_normals, even_gap_net, find_gap_pairs, measure_gap
from .recipe import (EDGE_KEYS, META_KEY, edge_polyline, hand_edited_net, raw_meta, replace_net,
                     store_meta, surface_edges, world_net)

GOOD = (0.25, 0.9, 0.35, 1.0)
OFF = (1.0, 0.55, 0.15, 1.0)
SAMPLES = 24

_state = {"handle": None, "dirty": False, "joints": []}


def compute_gaps(context):
    """[dict(a, b, edge_a, edge_b, min, max, mean, step, rungs)] for the visible patches."""
    s = context.scene.spline_surface
    edges = surface_edges(context, samples=SAMPLES)
    if not edges:
        return []
    size = max(np.ptp(pl, axis=0).max() for pl in edges.values())
    search = s.pgap_search if s.pgap_search > 0 else 0.05 * size
    nets = {}
    joints = []
    for ka, kb in find_gap_pairs(edges, search, touch=1e-4 * size):
        if ka[0] not in nets:
            nets[ka[0]] = world_net(bpy.data.objects[ka[0]])
        net, orders = nets[ka[0]]
        m = measure_gap(edges[ka], edges[kb], edge_normals(net, orders, ka[1], SAMPLES))
        ok = m["dist"] <= search
        if not ok.any():
            continue
        g = m["gap"][ok]
        joints.append({"a": ka[0], "b": kb[0], "edge_a": ka[1], "edge_b": kb[1],
                       "min": float(g.min()), "max": float(g.max()), "mean": float(g.mean()),
                       "step": float(m["step"][ok].max()),
                       "rungs": (m["a"][ok], m["b"][ok], g)})
    return joints


def refresh(context):
    _state["joints"] = compute_gaps(context)
    _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    if not _state["joints"]:
        return
    s = bpy.context.scene.spline_surface
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", 2.5)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    for j in _state["joints"]:
        a, b, g = j["rungs"]
        good = np.abs(g - s.pgap_target) <= s.pgap_tolerance
        for flag, color in ((True, GOOD), (False, OFF)):
            sel = good == flag
            if not sel.any():
                continue
            pos = np.stack([a[sel], b[sel]], axis=1).reshape(-1, 3).astype(np.float32)
            shader.uniform_float("color", color)
            batch_for_shader(shader, 'LINES', {"pos": pos}).draw(shader)
    gpu.state.blend_set('NONE')


def _timer():
    if _state["handle"] is None:
        return None
    if _state["dirty"]:
        try:
            refresh(bpy.context)
        except Exception:  # never let a timer die on a half-edited scene
            _state["dirty"] = False
    return 0.4


@persistent      # kept when another file is opened
def _on_depsgraph(scene, depsgraph):
    if _state["handle"] is None:
        return
    for u in depsgraph.updates:
        idb = getattr(u.id, "original", u.id)
        if isinstance(idb, (bpy.types.Object, bpy.types.Curve)):
            _state["dirty"] = True
            return


def mark_dirty():
    _state["dirty"] = True


def is_on():
    return _state["handle"] is not None


def joints():
    return _state["joints"] if is_on() else []


def show(context, on):
    if on and _state["handle"] is None:
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
        refresh(context)
        bpy.app.timers.register(_timer, first_interval=0.4, persistent=True)
    elif not on and _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
        _state["joints"] = []
        refresh(context)


class SS_OT_panel_gap(bpy.types.Operator):
    bl_idname = "spline_surface.panel_gap"
    bl_label = "Panel Gap"
    bl_description = ("Show the clearance between facing edges of neighbouring panels: green where "
                      "the gap is within tolerance of the target, orange where it is off")
    bl_options = {'REGISTER'}

    def execute(self, context):
        show(context, not is_on())
        return {'FINISHED'}


def facing_edges(a, b, search=0.0):
    """(edge of a, edge of b, reach) of the joint between patches a and b, or None
    (reach: how far apart the two may be and still count as facing, twice the search).

    The pair whose edges lie closest on average, among those within `search`
    (0: 5% of the larger patch) along most of their length."""
    (na, oa), (nb, ob) = world_net(a), world_net(b)
    lines = {(a.name, e): edge_polyline(na, oa, e, SAMPLES) for e in EDGE_KEYS}
    lines.update({(b.name, e): edge_polyline(nb, ob, e, SAMPLES) for e in EDGE_KEYS})
    size = max(np.ptp(na.reshape(-1, 3), 0).max(), np.ptp(nb.reshape(-1, 3), 0).max())
    search = search if search > 0 else 0.05 * size
    best = None
    for ka, kb in find_gap_pairs(lines, search, touch=0.0, partial=True):
        if ka[0] != a.name:
            ka, kb = kb, ka
        d = measure_gap(lines[ka], lines[kb])["dist"]
        close = d <= search
        d = d[close].mean() / close.mean()          # the facing part, favouring longer joints
        if best is None or d < best[0]:
            best = (d, ka[1], kb[1])
    return None if best is None else best[1:] + (2.0 * search,)


class SS_OT_even_gap(bpy.types.Operator):
    bl_idname = "spline_surface.even_gap"
    bl_label = "Even Gap"
    bl_description = ("Move the active panel's edge so the gap to the other selected panel is the "
                      "Target all along (and the two sit level). The active panel keeps following "
                      "the other one when it moves")
    bl_options = {'REGISTER', 'UNDO'}

    target: bpy.props.FloatProperty(
        name="Gap", default=-1.0, min=-1.0, max=100.0, unit='LENGTH',
        description="The gap to keep; below 0 uses Panel Gap's Target")
    flush: bpy.props.BoolProperty(
        name="Flush", default=True,
        description="Also bring the edge level with the other panel (no step)")
    remove: bpy.props.BoolProperty(
        name="Remove", default=False, options={'SKIP_SAVE'},
        description="Stop keeping the gap (the shape stays as it is)")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return context.mode == 'OBJECT' and ob is not None and ob.type == 'SURFACE'

    def invoke(self, context, event):
        if self.target < 0:
            self.target = context.scene.spline_surface.pgap_target
        return self.execute(context)

    def execute(self, context):
        ob = context.active_object
        meta = json.loads(ob[META_KEY]) if META_KEY in ob else None
        if self.remove:
            if not meta or "gap" not in meta:
                self.report({'INFO'}, "%s keeps no gap" % ob.name)
                return {'CANCELLED'}
            meta.pop("gap")
            store_meta(ob, meta)
            self.report({'INFO'}, "%s no longer keeps its gap" % ob.name)
            return {'FINISHED'}
        others = [o for o in context.selected_objects if o.type == 'SURFACE' and o is not ob]
        if len(others) != 1:
            self.report({'ERROR'}, "Select the other panel too (the active one moves)")
            return {'CANCELLED'}
        ref = others[0]
        s = context.scene.spline_surface
        target = self.target if self.target >= 0 else s.pgap_target
        pair = facing_edges(ob, ref, s.pgap_search)
        if pair is None:
            self.report({'ERROR'}, "No edges of %s and %s face each other (raise Search in "
                                   "Panel Gap or move the panels closer)" % (ob.name, ref.name))
            return {'CANCELLED'}
        net, orders = world_net(ob)
        rnet, rorders = world_net(ref)
        line = edge_polyline(rnet, rorders, pair[1], 64)
        try:
            new, new_orders, (before, after) = even_gap_net(net, orders, pair[0], line, target,
                                                            flush=self.flush, reach=pair[2])
        except (ValueError, np.linalg.LinAlgError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if meta is None or hand_edited_net(meta, net, orders):
            # the shape before the gap becomes the recipe, as Match Edge does
            meta = raw_meta(net, orders, **{k: meta[k] for k in ("matches",) if meta and k in meta})
        replace_net(ob, new, s, new_orders)
        meta["gap"] = {"ref": ref.name, "edge": pair[0], "edge_ref": pair[1],
                       "target": target, "flush": self.flush, "reach": pair[2]}
        store_meta(ob, meta)
        mark_dirty()
        msg = "%s %s: gap %.4g (worst error %.2g -> %.2g)" % (ob.name, pair[0], target, before, after)
        opened = opened_seams(context, ob, net, orders, new, new_orders)
        if opened:
            # the ends of the evened edge moved, and neighbours sharing a side edge
            # (pieces of a split, say) do not follow: say so instead of leaving a crack
            self.report({'WARNING'}, msg + "; this opened the seam with " + ", ".join(opened)
                        + " (even the gap before splitting, or Match Edge them to it)")
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def opened_seams(context, ob, net, orders, new, new_orders):
    """Patches that shared an edge with `ob` before (net) but no longer do (new)."""
    from .core.continuity import find_shared_edge
    size = float(np.ptp(np.asarray(net).reshape(-1, 3), axis=0).max())
    tol = 1e-4 * size + 1e-9
    out = []
    for other in context.visible_objects:
        if other is ob or other.type != 'SURFACE':
            continue
        try:
            onet, oorders = world_net(other)
        except ValueError:
            continue
        if find_shared_edge(net, onet, tol, orders, oorders) is not None \
                and find_shared_edge(new, onet, tol, new_orders, oorders) is None:
            out.append(other.name)
    return sorted(out)


classes = (SS_OT_panel_gap, SS_OT_even_gap)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
