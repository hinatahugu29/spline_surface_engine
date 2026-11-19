"""Point Continuity: a dot at every point of the shown Bezier curves, coloured by
how smoothly the curve goes through it (core.curvesmooth.point_state):

    green   G2  - the same curvature on both sides
    yellow  G1  - the same direction, the curvature jumps
    red     corner (G0)
    grey    an end

The same colours as Seam Colors. Shown for the curves in edit mode, else the
selected curves, else every visible one; refreshed shortly after any change
(editing in edit mode included). Display only.
"""
import bpy
from bpy.app.handlers import persistent
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .core.curvesmooth import point_state

COLORS = {'G2': (0.2, 0.9, 0.35, 1.0), 'G1': (1.0, 0.82, 0.15, 1.0),
          'CORNER': (1.0, 0.25, 0.2, 1.0), 'END': (0.6, 0.6, 0.6, 1.0)}

_state = {"handle": None, "dirty": False, "marks": {}}


def _curves(context):
    if context.mode == 'EDIT_CURVE':
        return [o for o in context.objects_in_mode if o.type == 'CURVE']
    vis = [o for o in context.visible_objects if o.type == 'CURVE']
    return [o for o in vis if o.select_get()] or vis


def marks(context):
    """{state: [world points]} for the curves shown."""
    from .curveediting import bezier_splines, read_spline
    out = {k: [] for k in COLORS}
    for ob in _curves(context):
        for idx in bezier_splines(ob):
            A, L, R, cyc = read_spline(ob, idx)
            for i in range(len(A)):
                out[point_state(A, L, R, i, cyc)].append(tuple(A[i]))
    return out


def summary(context):
    """'G2 5  G1 2  corner 1' for the panel, or ''."""
    if _state["handle"] is None:
        return ""
    m = _state["marks"]
    parts = [("G2", len(m.get('G2', []))), ("G1", len(m.get('G1', []))),
             ("corner", len(m.get('CORNER', [])))]
    return "  ".join("%s %d" % p for p in parts)


def refresh(context):
    _state["marks"] = marks(context)
    _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    m = _state["marks"]
    if not m:
        return
    shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.point_size_set(10.0)
    for state, pts in m.items():
        if pts:
            shader.uniform_float("color", COLORS[state])
            batch_for_shader(shader, 'POINTS', {"pos": pts}).draw(shader)
    gpu.state.blend_set('NONE')


def _timer():
    if _state["handle"] is None:
        return None
    if _state["dirty"]:
        try:
            refresh(bpy.context)
        except Exception:  # never let a timer die on a half-edited scene
            _state["dirty"] = False
    return 0.15


@persistent      # kept when another file is opened
def _on_depsgraph(scene, depsgraph):
    if _state["handle"] is None:
        return
    for u in depsgraph.updates:
        idb = getattr(u.id, "original", u.id)
        if isinstance(idb, (bpy.types.Object, bpy.types.Curve)):
            _state["dirty"] = True
            return


def is_on():
    return _state["handle"] is not None


class SS_OT_point_continuity(bpy.types.Operator):
    bl_idname = "spline_surface.point_continuity"
    bl_label = "Point Continuity"
    bl_description = ("Colour every point of the shown curves by how smoothly the curve goes "
                      "through it: green G2, yellow G1, red corner, grey end. Follows edits, in "
                      "edit mode too. Press again to hide")
    bl_options = {'REGISTER'}

    def execute(self, context):
        if _state["handle"] is None:
            _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
            bpy.app.timers.register(_timer, first_interval=0.15, persistent=True)
            refresh(context)
        else:
            bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
            _state["handle"] = None
            _state["marks"] = {}
            for win in context.window_manager.windows:
                for area in win.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        return {'FINISHED'}


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)


classes = (SS_OT_point_continuity,)
