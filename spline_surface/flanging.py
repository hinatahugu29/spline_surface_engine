"""Flange: fold a wall off an open patch edge, with a rounded fold.

Click an edge; a wall is added along it at the chosen angle and, if the
radius is above zero, the fold is rounded with Fillet. The wall is a "flange"
recipe (it follows its source patch through Live Update); the rounded fold is
an ordinary fillet, so its radius can still be changed afterwards. The
geometry lives in core.flange.
"""
import json

import bpy

from . import pick, zebra
from .bridge.write_surface import build_surface_data
from .operators import compute_flange
from .recipe import _make_active, store_meta, surface_edges


# Bridge tension that makes a G1 bridge between the patch edge and the wall's first row
# follow a circular arc (checked in the tests)
_ARC_TENSION = 1.5


class SS_OT_flange(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.flange"
    bl_label = "Flange"
    bl_description = ("Fold a wall off an open patch edge (click the edge) and round the fold. "
                      "Use it for the lip around a panel: a hem, a return, a wheel-arch lip")
    bl_options = {'REGISTER', 'UNDO'}

    source: bpy.props.StringProperty(
        name="Patch", description="Patch to fold the wall off (empty: the active object)")
    edge: bpy.props.EnumProperty(
        name="Edge",
        items=[('V1', "V Max", "Edge at v = 1"), ('V0', "V Min", "Edge at v = 0"),
               ('U1', "U Max", "Edge at u = 1"), ('U0', "U Min", "Edge at u = 0")])
    length: bpy.props.FloatProperty(
        name="Length", default=0.3, min=0.001, max=1000.0, unit='LENGTH',
        description="Depth of the wall")
    angle: bpy.props.FloatProperty(
        name="Angle", default=90.0, min=5.0, max=175.0,
        description="Fold angle: 0 continues the patch straight on, 90 is square to it, "
                    "180 folds back over it")
    radius: bpy.props.FloatProperty(
        name="Radius", default=0.05, min=0.0, max=1000.0, unit='LENGTH',
        description="Rounding of the fold (0: leave it sharp)")
    flip: bpy.props.BoolProperty(
        name="Flip", default=False, description="Fold towards the front of the patch instead of the back")
    outside: bpy.props.BoolProperty(
        name="Outside", default=False,
        description="Leave the patch as it is and add the rounded fold outside its edge, instead "
                    "of cutting the patch back by the radius. Use it where the edge already is the "
                    "outline, e.g. to carry a flange on along a neighbouring patch")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.visible_objects)

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._edges = surface_edges(context)
        if not self._edges:
            self.report({'ERROR'}, "No open patch edges to fold a wall off")
            return {'CANCELLED'}
        self._hover = None
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Flange: click an edge   Esc/RMB: cancel")
        return {'RUNNING_MODAL'}

    def _label(self):
        return pick.edge_label(self._hover, self._edges[self._hover]) if self._hover else None

    def _draw_lines(self):
        lines = [(pts, pick.COLOR_EDGE, 2.0) for k, pts in self._edges.items() if k != self._hover]
        if self._hover:
            lines.append((self._edges[self._hover], pick.COLOR_HOVER, 4.0))
        return lines

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            hover = pick.nearest_polyline(context, self._edges,
                                          (event.mouse_region_x, event.mouse_region_y))
            if hover != self._hover:
                self._hover = hover
                context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self._hover:
            self._finish(context)
            self.source, self.edge = self._hover
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    def execute(self, context):
        source = bpy.data.objects.get(self.source) if self.source else context.active_object
        if source is None or source.type != 'SURFACE':
            self.report({'ERROR'}, "Pick a patch edge to fold a wall off")
            return {'CANCELLED'}
        settings = context.scene.spline_surface
        meta = {"method": "flange", "sources": [source.name], "edge": self.edge,
                "length": self.length, "angle": self.angle, "flip": self.flip}
        outside = self.outside and self.radius > 0.0
        if outside:
            meta["offset"] = self.radius
        try:
            net, orders = compute_flange(source, meta)
            cu = build_surface_data(source.name + "_flange", net, settings.resolution, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        for mat in zebra.real_materials(source):
            cu.materials.append(mat)
        wall = bpy.data.objects.new(source.name + "_flange", cu)
        context.collection.objects.link(wall)
        store_meta(wall, meta)
        _make_active(context, wall)
        if outside:
            result = bpy.ops.spline_surface.bridge_edges(
                source_a=source.name, edge_a=self.edge, source_b=wall.name, edge_b='V0',
                continuity='G1', tension=_ARC_TENSION)
            if result != {'FINISHED'}:
                self.report({'WARNING'}, "The wall was added, but the fold could not be rounded")
        elif self.radius > 0.0:
            seams = json.dumps([[source.name, self.edge, wall.name, 'V0']])
            result = bpy.ops.spline_surface.fillet(
                seams=seams, radius=self.radius, continuity='G2', corners=False)
            if result != {'FINISHED'}:
                self.report({'WARNING'}, "The wall was added, but the fold could not be rounded "
                                         "(radius too large for the wall?)")
        return {'FINISHED'}


classes = (SS_OT_flange,)
