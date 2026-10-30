"""Viewport gizmos: drag handles for the recipe values of the active patch.

Extend Length: with an Extend patch active, an arrow sits mid-way along the
edge it grew from; dragging it rewrites the recipe's length and rebuilds the
patch (dependents follow through Live Update). Hand-edited patches get no
arrow, since rebuilding would throw the edits away.
"""
import json

import bpy
from mathutils import Matrix, Vector

from . import live
from .core.extend import length_handle
from .recipe import META_KEY, hand_edited, world_net

MIN_LENGTH = 0.02


def _extend_recipe(ob):
    """(meta, source object) if `ob` is an Extend patch the arrow may rebuild."""
    if ob is None or ob.type != 'SURFACE' or META_KEY not in ob:
        return None
    try:
        meta = json.loads(ob[META_KEY])
    except ValueError:
        return None
    if meta.get("method") != "extend":
        return None
    src = bpy.data.objects.get(meta["sources"][0])
    if src is None or src.type != 'SURFACE' or hand_edited(ob, meta):
        return None
    return meta, src


def _push_undo():
    try:
        bpy.ops.ed.undo_push(message="Extend Length")
    except RuntimeError:
        pass
    return None


class SS_GGT_extend_length(bpy.types.GizmoGroup):
    bl_idname = "SS_GGT_extend_length"
    bl_label = "Extend Length"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and _extend_recipe(context.active_object) is not None

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
        found = _extend_recipe(context.active_object)
        if found is None:
            return
        meta, src = found
        net, orders = world_net(src)
        base, direction, self.handle_scale = length_handle(net, meta["edge"], orders)
        rot = Vector(direction).to_track_quat('Z', 'Y').to_matrix().to_4x4()
        self.arrow.matrix_basis = Matrix.Translation(Vector(base)) @ rot

    def _get(self):
        found = _extend_recipe(bpy.context.active_object)
        return found[0]["length"] * self.handle_scale if found else 0.0

    def _set(self, value):
        ctx = bpy.context
        ob = ctx.active_object
        found = _extend_recipe(ob)
        if found is None:
            return
        meta = found[0]
        meta["length"] = max(MIN_LENGTH, value / max(self.handle_scale, 1e-12))
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
                bpy.app.timers.register(_push_undo, first_interval=0.0)
        self.was_modal = modal


classes = (SS_GGT_extend_length,)
