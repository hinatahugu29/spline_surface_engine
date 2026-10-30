"""Viewport gizmos: drag handles for the recipe values of the active patch.

Extend Length: with an Extend patch active, an arrow sits mid-way along the
edge it grew from. Offset Distance: with an Offset patch active, an arrow
stands on the middle of the source along its normal. Fillet Radius: with a
fillet bridge active, an arrow runs from the old crease to the bridge's
middle. Dragging rewrites the recipe value and rebuilds (dependents follow
through Live Update). Hand-edited patches get no arrow, since rebuilding
would throw the edits away.
"""
import json

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import live
from .core.extend import length_handle
from .core.nurbs import eval_points
from .core.offset import distance_handle
from .recipe import META_KEY, edge_polyline, hand_edited, world_net

MIN_LENGTH = 0.02
MIN_DISTANCE = 1e-4


def _recipe(ob, method):
    """(meta, source object) if `ob` is a `method` patch an arrow may rebuild."""
    if ob is None or ob.type != 'SURFACE' or META_KEY not in ob:
        return None
    try:
        meta = json.loads(ob[META_KEY])
    except ValueError:
        return None
    if meta.get("method") != method:
        return None
    src = bpy.data.objects.get(meta["sources"][0])
    if src is None or src.type != 'SURFACE' or hand_edited(ob, meta):
        return None
    return meta, src


class _RecipeArrow:
    """An arrow whose offset is one numeric recipe value of the active patch.

    Subclasses set `method` and `key` and implement `_handle` (arrow base,
    direction, world distance per unit of the value) and `_clamp`.
    """
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}
    method = key = None

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and _recipe(context.active_object, cls.method) is not None

    def setup(self, context):
        gz = self.gizmos.new("GIZMO_GT_arrow_3d")
        gz.draw_style = 'NORMAL'
        gz.color = 1.0, 0.55, 0.15
        gz.alpha = 0.8
        gz.color_highlight = 1.0, 0.8, 0.3
        gz.alpha_highlight = 1.0
        gz.target_set_handler("offset", get=self._get, set=self._set)
        self.arrow = gz
        self.handle_scale = 1.0
        self.was_modal = False
        self._place(context)

    def _place(self, context):
        found = _recipe(context.active_object, self.method)
        if found is None:
            return
        meta, src = found
        net, orders = world_net(src)
        base, direction, self.handle_scale = self._handle(meta, net, orders)
        rot = Vector(direction).to_track_quat('Z', 'Y').to_matrix().to_4x4()
        self.arrow.matrix_basis = Matrix.Translation(Vector(base)) @ rot

    def _get(self):
        found = _recipe(bpy.context.active_object, self.method)
        return found[0][self.key] * self.handle_scale if found else 0.0

    def _set(self, value):
        ctx = bpy.context
        ob = ctx.active_object
        found = _recipe(ob, self.method)
        if found is None:
            return
        meta = found[0]
        meta[self.key] = self._clamp(value / max(self.handle_scale, 1e-12))
        # The fingerprint still matches the patch, so rebuild treats it as ours.
        ob[META_KEY] = json.dumps(meta)
        live.rebuild(ob, ctx)

    def refresh(self, context):
        self._place(context)

    def draw_prepare(self, context):
        modal = self.arrow.is_modal
        if not modal:
            self._place(context)
            if self.was_modal:
                # Not allowed while drawing; a timer runs it right after.
                message = self.bl_label

                def push():
                    try:
                        bpy.ops.ed.undo_push(message=message)
                    except RuntimeError:
                        pass

                bpy.app.timers.register(push, first_interval=0.0)
        self.was_modal = modal


class SS_GGT_extend_length(_RecipeArrow, bpy.types.GizmoGroup):
    bl_idname = "SS_GGT_extend_length"
    bl_label = "Extend Length"
    method, key = "extend", "length"

    def _handle(self, meta, net, orders):
        return length_handle(net, meta["edge"], orders)

    def _clamp(self, value):
        return max(MIN_LENGTH, value)


class SS_GGT_offset_distance(_RecipeArrow, bpy.types.GizmoGroup):
    bl_idname = "SS_GGT_offset_distance"
    bl_label = "Offset Distance"
    method, key = "offset", "distance"

    def _handle(self, meta, net, orders):
        base, normal = distance_handle(net, orders)
        return base, normal, 1.0

    def _clamp(self, value):
        # Either side of the source works; only exactly on it is degenerate.
        if abs(value) >= MIN_DISTANCE:
            return value
        return MIN_DISTANCE if value >= 0 else -MIN_DISTANCE


def _fillet_bridge(ob):
    """(bridge recipe, recipe of its first trimmed patch) for an untouched
    fillet bridge, else None."""
    if ob is None or ob.type != 'SURFACE' or META_KEY not in ob:
        return None
    try:
        meta = json.loads(ob[META_KEY])
    except ValueError:
        return None
    if meta.get("method") != "blend" or "fillet" not in meta or hand_edited(ob, meta):
        return None
    side = bpy.data.objects.get(meta["sources"][0])
    if side is None or META_KEY not in side:
        return None
    side_meta = json.loads(side[META_KEY])
    if side_meta.get("method") != "fillet_trim" or side_meta["fillet"]["id"] != meta["fillet"]:
        return None
    return meta, side_meta


class SS_GGT_fillet_radius(_RecipeArrow, bpy.types.GizmoGroup):
    bl_idname = "SS_GGT_fillet_radius"
    bl_label = "Fillet Radius"

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and _fillet_bridge(context.active_object) is not None

    def _place(self, context):
        found = _fillet_bridge(context.active_object)
        if found is None:
            return
        from .operators import compute_from_meta, fillet_base
        meta, side_meta = found
        try:
            base, base_orders, _ = compute_from_meta(
                fillet_base(side_meta, meta["fillet"]), context)
        except ValueError:
            # A source this fillet was built from (e.g. a boundary curve)
            # was deleted: nothing sane to place the handle on this frame.
            return
        crease = edge_polyline(base, base_orders, meta["edges"][0], 3)[1]
        net, orders = world_net(context.active_object)
        offset = eval_points(net, orders, [0.5], [0.5])[0] - crease
        reach = float(np.linalg.norm(offset))
        # The bridge's middle moves away from the old crease in proportion to the radius.
        self.handle_scale = reach / max(side_meta["fillet"]["radius"], 1e-12)
        rot = Vector(offset / max(reach, 1e-12)).to_track_quat('Z', 'Y').to_matrix().to_4x4()
        self.arrow.matrix_basis = Matrix.Translation(Vector(crease)) @ rot

    def _get(self):
        found = _fillet_bridge(bpy.context.active_object)
        return found[1]["fillet"]["radius"] * self.handle_scale if found else 0.0

    def _set(self, value):
        found = _fillet_bridge(bpy.context.active_object)
        if found is None:
            return
        from .filleting import set_fillet_radius
        try:
            set_fillet_radius(bpy.context, found[0]["fillet"],
                              max(1e-4, value / max(self.handle_scale, 1e-12)))
        except ValueError:
            pass  # e.g. the radius no longer fits: keep the last good one


classes = (SS_GGT_extend_length, SS_GGT_offset_distance, SS_GGT_fillet_radius)
