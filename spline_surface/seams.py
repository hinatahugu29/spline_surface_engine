"""Seam colours: every seam between visible patches drawn in its continuity grade.

G2 green, G1 yellow, G0 red, "open" (edges close but apart) magenta. The
seams are recomputed shortly after any patch changes while the display is on.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .core.continuity import find_shared_edge, grade, measure
from .operators import _seam_tolerance
from .recipe import edge_polyline, world_net

COLORS = {'G2': (0.25, 0.9, 0.35, 1.0), 'G1': (1.0, 0.85, 0.2, 1.0),
          'G0': (1.0, 0.25, 0.2, 1.0), 'open': (0.9, 0.3, 0.9, 1.0)}
GRADES = ('G2', 'G1', 'G0', 'open')

_state = {"handle": None, "seams": [], "dirty": False, "counts": {}}


def _bbox_overlap(Pa, Pb, pad):
    a, b = Pa.reshape(-1, 3), Pb.reshape(-1, 3)
    return np.all(a.min(0) - pad <= b.max(0)) and np.all(b.min(0) - pad <= a.max(0))


def shared_seams(context):
    """[dict(a, b, edge_a, edge_b, grade, measure, polyline)] for visible patches.

    `polyline` follows edge_a in world space.
    """
    settings = context.scene.spline_surface
    nets = {}
    for ob in context.visible_objects:
        if ob.type == 'SURFACE':
            try:
                nets[ob.name] = world_net(ob)
            except ValueError:
                pass
    names = sorted(nets)
    seams = []
    for i, a in enumerate(names):
        Pa, oa = nets[a]
        for b in names[i + 1:]:
            Pb, ob_ = nets[b]
            tol = _seam_tolerance(Pa, Pb, settings)
            if not _bbox_overlap(Pa, Pb, tol):
                continue
            shared = find_shared_edge(Pa, Pb, tol, oa, ob_)
            if shared is None:
                continue
            m = measure(Pa, Pb, *shared[:3], oa, ob_)
            seams.append({"a": a, "b": b, "edge_a": shared[0], "edge_b": shared[1],
                          "grade": grade(m), "measure": m,
                          "polyline": edge_polyline(Pa, oa, shared[0], 48)})
    return seams


def compute_seams(context):
    """[(grade, polyline, measure, (name_a, name_b))] for visible patches."""
    return [(s["grade"], s["polyline"], s["measure"], (s["a"], s["b"]))
            for s in shared_seams(context)]


def refresh(context):
    _state["seams"] = compute_seams(context)
    counts = {g: 0 for g in GRADES}
    for g, *_ in _state["seams"]:
        counts[g] += 1
    _state["counts"] = counts
    _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    if not _state["seams"]:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", 4.0)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    for g, pts, _, _ in _state["seams"]:
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": [tuple(p) for p in pts]})
        shader.uniform_float("color", COLORS[g])
        batch.draw(shader)
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


def show(context, on):
    if on and _state["handle"] is None:
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
        refresh(context)
        bpy.app.timers.register(_timer, first_interval=0.4)
    elif not on and _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
        _state["seams"] = []
        refresh(context)


def counts():
    return _state["counts"] if _state["handle"] is not None else {}


class SS_OT_seam_colors(bpy.types.Operator):
    bl_idname = "spline_surface.seam_colors"
    bl_label = "Seam Colors"
    bl_description = ("Colour every seam between visible patches by continuity: "
                      "G2 green, G1 yellow, G0 red, open magenta. Updates as you edit")
    bl_options = {'REGISTER'}

    def execute(self, context):
        show(context, _state["handle"] is None)
        return {'FINISHED'}


classes = (SS_OT_seam_colors,)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
