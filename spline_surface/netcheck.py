"""Network Check: what Curve Network would make of the curves, drawn in place.

Curves are coloured by direction (blue / green), crossings get a white
mark. Problems: a missing crossing is a red line between the two curves'
closest points, a same-direction crossing a red mark, a curve crossing
nothing grey, a curve whose crossings are out of order orange. Uses the
selected curves (else every visible one), with Curve Network's tolerance.
Display only; recomputed shortly after any change.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .bridge.read_curves import read_polyline
from .core.network import diagnose_network
from .operators import BOUNDARY_TYPES, network_tolerance

FAMILY = {0: (0.35, 0.65, 1.0, 1.0), 1: (0.35, 0.95, 0.45, 1.0), None: (0.6, 0.6, 0.6, 1.0)}
TANGLED = (1.0, 0.6, 0.15, 1.0)
PROBLEM = (1.0, 0.25, 0.2, 1.0)
CROSSING = (1.0, 1.0, 1.0, 0.9)

_state = {"handle": None, "dirty": False, "lines": [], "summary": []}


def _sources(context):
    visible = [o for o in context.visible_objects if o.type in BOUNDARY_TYPES]
    return [o for o in visible if o.select_get()] or visible


def _cross(p, r):
    return [np.array([p - r * a, p + r * a]) for a in np.eye(3)]


def summarise(d):
    """Short lines for the panel, problems first."""
    lines = []
    if d["missing"]:
        lines.append("%d crossing(s) missing (red lines)" % len(d["missing"]))
    if d["conflicts"]:
        lines.append("%d same-direction crossing(s) (red marks)" % len(d["conflicts"]))
    if d["alone"]:
        lines.append("%d curve(s) cross nothing (grey)" % len(d["alone"]))
    if d["tangled"]:
        lines.append("%d curve(s) cross out of order (orange)" % len(d["tangled"]))
    nu, nv = d["counts"]
    if min(nu, nv) < 2:
        lines.append("need 2+ crossing curves each way (now %d x %d)" % (nu, nv))
    head = "%d x %d curves, %d crossings" % (nu, nv, len(d["crossings"]))
    return [head + (": ready" if d["ok"] else "")] + lines


def refresh(context):
    _state["lines"], _state["summary"] = [], []
    obs = _sources(context)
    dg = context.evaluated_depsgraph_get()
    polys = []
    for o in obs:
        try:
            polys.append(read_polyline(o, dg, 120))
        except ValueError:
            pass
    if len(polys) >= 2:
        d = diagnose_network(polys, network_tolerance(polys, context.scene.spline_surface))
        size = float(np.ptp(np.concatenate(polys), axis=0).max())
        r = 0.012 * size
        lines = _state["lines"]
        for i, pts in enumerate(polys):
            color = TANGLED if i in d["tangled"] else FAMILY[d["family"][i]]
            lines.append((pts, color, 3.0))
        for _, _, p, _ in d["crossings"]:
            lines += [(seg, CROSSING, 2.0) for seg in _cross(p, r)]
        for _, _, p in d["conflicts"]:
            lines += [(seg, PROBLEM, 4.0) for seg in _cross(p, 2 * r)]
        for _, _, pa, pb, _ in d["missing"]:
            lines.append((np.array([pa, pb]), PROBLEM, 3.0))
            lines += [(seg, PROBLEM, 3.0) for seg in _cross(pa, r) + _cross(pb, r)]
        _state["summary"] = summarise(d)
    else:
        _state["summary"] = ["select (or show) 2+ curves"]
    _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    if not _state["lines"]:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    for pts, color, width in _state["lines"]:
        shader.uniform_float("color", color)
        shader.uniform_float("lineWidth", width)
        batch_for_shader(shader, 'LINE_STRIP',
                         {"pos": np.asarray(pts, np.float32)}).draw(shader)
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


def show(context, on):
    if on and _state["handle"] is None:
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
        refresh(context)
        bpy.app.timers.register(_timer, first_interval=0.4)
    elif not on and _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
        _state["lines"], _state["summary"] = [], []


def summary():
    return _state["summary"] if _state["handle"] is not None else []


class SS_OT_network_check(bpy.types.Operator):
    bl_idname = "spline_surface.network_check"
    bl_label = "Network Check"
    bl_description = ("Show how Curve Network sees the selected (else visible) curves: "
                      "directions in blue / green, crossings marked; missing crossings red "
                      "lines, same-direction crossings red marks, lone curves grey, "
                      "out-of-order curves orange. Updates as you edit")
    bl_options = {'REGISTER'}

    def execute(self, context):
        show(context, _state["handle"] is None)
        return {'FINISHED'}


classes = (SS_OT_network_check,)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
