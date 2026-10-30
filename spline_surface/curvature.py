"""Curvature display: combs along iso-curves and a Gaussian / mean colour map.

Shown for the selected patches, or every visible patch if none is selected,
and recomputed shortly after any patch changes. Display only. Comb teeth
point away from the centre of curvature; their length is scaled so the
largest curvature on screen gets a fixed fraction of the scene size.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .core.curvature import color_ramp, comb, surface_curvature
from .recipe import world_net

TOOTH = (0.75, 0.75, 0.8, 0.7)
ENVELOPE = (1.0, 0.55, 0.15, 1.0)

_state = {"handle": None, "comb": False, "map": False, "dirty": False,
          "teeth": [], "envelopes": [], "tris": None, "kmax": 0.0, "limit": 0.0}


def _patches(context):
    visible = [ob for ob in context.visible_objects if ob.type == 'SURFACE']
    chosen = [ob for ob in visible if ob.select_get()] or visible
    out = []
    for ob in chosen:
        try:
            out.append(world_net(ob))
        except ValueError:
            pass
    return out


def _robust_max(values):
    v = np.abs(np.concatenate([np.ravel(x) for x in values])) if values else np.zeros(1)
    return float(np.percentile(v, 98)) if v.size else 0.0


def refresh(context):
    s = context.scene.spline_surface
    patches = _patches(context)
    pts = np.concatenate([n.reshape(-1, 3) for n, _ in patches]) if patches else np.zeros((1, 3))
    size = float(np.linalg.norm(np.ptp(pts, axis=0))) or 1.0
    _state["teeth"], _state["envelopes"], _state["tris"] = [], [], None
    if _state["comb"] and patches:
        dirs = ('U', 'V') if s.comb_direction == 'BOTH' else (s.comb_direction,)
        combs = [comb(n, o, d, s.comb_count) for n, o in patches for d in dirs]
        kmax = _robust_max([np.linalg.norm(k, axis=2) for _, k in combs])
        _state["kmax"] = kmax
        scale = s.comb_scale * 0.15 * size / max(kmax, 1e-12)
        for P, k in combs:
            tips = P - k * scale  # away from the centre of curvature
            _state["envelopes"] += [row.astype(np.float32) for row in tips]
            _state["teeth"].append(np.stack([P, tips], axis=2).reshape(-1, 3).astype(np.float32))
    if _state["map"] and patches:
        grids = [surface_curvature(n, o) for n, o in patches]
        idx = 2 if s.curvature_type == 'GAUSS' else 3
        limit = _robust_max([g[idx] for g in grids])
        _state["limit"] = limit
        pos, col = [], []
        lift = size * 3e-3  # clears the viewport's tessellated surface
        for g in grids:
            P, N, val = g[0], g[1], g[idx]
            C = color_ramp(val, limit)
            for side in (1.0, -1.0):  # both faces, just off the surface
                Q = P + side * lift * N
                a, b, c, d = Q[:-1, :-1], Q[1:, :-1], Q[1:, 1:], Q[:-1, 1:]
                ca, cb, cc, cd = C[:-1, :-1], C[1:, :-1], C[1:, 1:], C[:-1, 1:]
                pos += [a, b, c, a, c, d]
                col += [ca, cb, cc, ca, cc, cd]
        pos = np.concatenate([p.reshape(-1, 3) for p in pos]).astype(np.float32)
        col = np.concatenate([c.reshape(-1, 4) for c in col]).astype(np.float32)
        _state["tris"] = (pos, col)
    _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    region = bpy.context.region
    gpu.state.blend_set('ALPHA')
    if _state["tris"] is not None:
        pos, col = _state["tris"]
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        gpu.state.depth_test_set('LESS_EQUAL')
        batch_for_shader(shader, 'TRIS', {"pos": pos, "color": col}).draw(shader)
    if _state["teeth"]:
        lines = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        lines.uniform_float("viewportSize", (region.width, region.height))
        gpu.state.depth_test_set('NONE')
        lines.uniform_float("lineWidth", 1.0)
        lines.uniform_float("color", TOOTH)
        for seg in _state["teeth"]:
            batch_for_shader(lines, 'LINES', {"pos": seg}).draw(lines)
        lines.uniform_float("lineWidth", 2.0)
        lines.uniform_float("color", ENVELOPE)
        for row in _state["envelopes"]:
            batch_for_shader(lines, 'LINE_STRIP', {"pos": row}).draw(lines)
    gpu.state.depth_test_set('NONE')
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


def toggle(context, which):
    _state[which] = not _state[which]
    on = _state["comb"] or _state["map"]
    if on and _state["handle"] is None:
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
        bpy.app.timers.register(_timer, first_interval=0.4)
    elif not on and _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    refresh(context)


def is_on(which):
    return _state[which]


def limit():
    return _state["limit"]


class SS_OT_curvature_comb(bpy.types.Operator):
    bl_idname = "spline_surface.curvature_comb"
    bl_label = "Curvature Comb"
    bl_description = ("Show curvature combs along iso-curves: the teeth point away from the "
                      "centre of curvature, longer means more curved. A smooth envelope "
                      "means smooth curvature; jumps at a seam mean G1 or worse")
    bl_options = {'REGISTER'}

    def execute(self, context):
        toggle(context, "comb")
        return {'FINISHED'}


class SS_OT_curvature_map(bpy.types.Operator):
    bl_idname = "spline_surface.curvature_map"
    bl_label = "Curvature Map"
    bl_description = ("Colour the patches by curvature. Gaussian: red dome, green flat or "
                      "cylindrical, blue saddle. Mean: red bends towards the patch normal, "
                      "blue away")
    bl_options = {'REGISTER'}

    def execute(self, context):
        toggle(context, "map")
        return {'FINISHED'}


classes = (SS_OT_curvature_comb, SS_OT_curvature_map)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    _state["comb"] = _state["map"] = False
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
