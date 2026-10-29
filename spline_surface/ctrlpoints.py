"""Control-point overlay: which points of a patch are pinned by its seams.

Pinned points (the rows next to a G0/G1/G2 seam, see core.lock) are drawn
orange, free interior points cyan, over a faint control net. Shown for the
selected patches, or every visible patch if none is selected. Display only;
nothing here edits a patch.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .core.lock import lock_mask
from .recipe import world_net
from .seams import shared_seams

LOCKED = (1.0, 0.55, 0.1, 1.0)
FREE = (0.2, 0.85, 1.0, 1.0)
NET = (0.7, 0.7, 0.7, 0.35)

_state = {"handle": None, "patches": [], "dirty": False, "counts": (0, 0)}


def compute_patches(context):
    """[(world net, locked mask)] for the selected (else all visible) patches."""
    visible = [ob for ob in context.visible_objects if ob.type == 'SURFACE']
    chosen = [ob for ob in visible if ob.select_get()] or visible
    pinned = {ob.name: [] for ob in chosen}
    for s in shared_seams(context):
        if s["a"] in pinned:
            pinned[s["a"]].append((s["edge_a"], s["grade"]))
        if s["b"] in pinned:
            pinned[s["b"]].append((s["edge_b"], s["grade"]))
    patches = []
    for ob in chosen:
        try:
            net, _ = world_net(ob)
        except ValueError:
            continue
        patches.append((net, lock_mask(net.shape[:2], pinned[ob.name])))
    return patches


def refresh(context):
    _state["patches"] = compute_patches(context)
    locked = sum(int(m.sum()) for _, m in _state["patches"])
    total = sum(m.size for _, m in _state["patches"])
    _state["counts"] = (total - locked, locked)
    _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    if not _state["patches"]:
        return
    lines = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    points = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
    region = bpy.context.region
    lines.uniform_float("viewportSize", (region.width, region.height))
    lines.uniform_float("lineWidth", 1.0)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    for net, mask in _state["patches"]:
        for grid in (net, net.transpose(1, 0, 2)):
            for row in grid:
                lines.uniform_float("color", NET)
                batch_for_shader(lines, 'LINE_STRIP', {"pos": [tuple(p) for p in row]}).draw(lines)
        gpu.state.point_size_set(9.0)
        for sel, color in ((mask, LOCKED), (~mask, FREE)):
            if sel.any():
                points.uniform_float("color", color)
                pos = [tuple(p) for p in net[sel]]
                batch_for_shader(points, 'POINTS', {"pos": pos}).draw(points)
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
        _state["patches"] = []
        refresh(context)


def counts():
    """(free, locked) control point counts while the overlay is on, else None."""
    return _state["counts"] if _state["handle"] is not None else None


class SS_OT_control_points(bpy.types.Operator):
    bl_idname = "spline_surface.control_points"
    bl_label = "Control Points"
    bl_description = ("Show a patch's control points: orange are pinned by its seams "
                      "(G0 one row, G1 two, G2 three from the edge), cyan are free to edit")
    bl_options = {'REGISTER'}

    def execute(self, context):
        show(context, _state["handle"] is None)
        return {'FINISHED'}


classes = (SS_OT_control_points,)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
