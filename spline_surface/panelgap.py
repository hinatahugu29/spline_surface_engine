"""Panel Gap: measure the clearance between facing edges of neighbouring panels.

While on, every pair of free edges from different patches that face each
other (a panel joint) is drawn as short rungs across the gap, green where the
gap is within tolerance of the target and orange where it is off. The Inspect
panel lists each joint's smallest, largest and mean gap and its step (how far
one panel sits proud of the other). The geometry lives in core.gap.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .core.gap import edge_normals, find_gap_pairs, measure_gap
from .recipe import surface_edges, world_net

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
        bpy.app.timers.register(_timer, first_interval=0.4)
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


classes = (SS_OT_panel_gap,)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
