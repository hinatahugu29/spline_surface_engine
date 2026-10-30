"""Presets: drop in a starting shape for a body panel at the 3D cursor.

Hood, Roof, Door, or a half hood with its fender (joined by a rounded
shoulder). Each is made of plain patches to shape with Soft Pull, Extend and
the rest. The shapes live in core.presets.
"""
import json

import bpy
import numpy as np

from .bridge.write_surface import build_surface_data
from .core import presets
from .recipe import _make_active, raw_meta, store_meta

KIND_ITEMS = [
    ('HOOD', "Hood", "A crowned panel for the bonnet"),
    ('ROOF', "Roof", "A crowned panel for the roof"),
    ('DOOR', "Door", "An upright side panel with a little bulge and tumblehome"),
    ('HOOD_FENDER', "Hood + Fender",
     "Half a hood and its fender wall, joined by a rounded shoulder; mirror the group for the other side"),
]


class SS_OT_add_preset(bpy.types.Operator):
    bl_idname = "spline_surface.add_preset"
    bl_label = "Add Preset"
    bl_description = "Add a starting shape for a body panel at the 3D cursor"
    bl_options = {'REGISTER', 'UNDO'}

    kind: bpy.props.EnumProperty(name="Panel", items=KIND_ITEMS, default='HOOD')
    length: bpy.props.FloatProperty(name="Length", default=1.4, min=0.01, max=1000.0, unit='LENGTH',
                                    description="Size along the car (y)")
    width: bpy.props.FloatProperty(name="Width", default=1.6, min=0.01, max=1000.0, unit='LENGTH',
                                   description="Size across the car (x)")
    height: bpy.props.FloatProperty(name="Height", default=0.8, min=0.01, max=1000.0, unit='LENGTH',
                                    description="Door height")
    crown: bpy.props.FloatProperty(name="Crown", default=0.1, min=0.0, max=100.0, unit='LENGTH',
                                   description="How far the middle of the panel rises above its edges")
    depth: bpy.props.FloatProperty(name="Fender Depth", default=0.45, min=0.01, max=1000.0,
                                   unit='LENGTH', description="How far the fender wall drops")
    radius: bpy.props.FloatProperty(name="Shoulder Radius", default=0.08, min=0.0, max=1000.0,
                                    unit='LENGTH', description="Rounding between hood and fender (0: sharp)")
    half: bpy.props.BoolProperty(name="Half", default=False,
                                 description="Only the half on the right of the centre line (x >= 0), "
                                             "to mirror afterwards (Hood + Fender is always a half)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "kind")
        col.prop(self, "length")
        if self.kind != 'DOOR':
            col.prop(self, "width")
            col.prop(self, "crown")
        else:
            col.prop(self, "height")
        if self.kind in ('HOOD', 'ROOF'):
            col.prop(self, "half")
        if self.kind == 'HOOD_FENDER':
            col.prop(self, "depth")
            col.prop(self, "radius")

    def execute(self, context):
        settings = context.scene.spline_surface
        sizes = {"length": self.length, "width": self.width, "height": self.height,
                 "crown": self.crown, "half": self.half, "depth": self.depth}
        nets = presets.build(self.kind, **sizes)
        origin = np.array(context.scene.cursor.location)
        made = []
        for name, net in nets:
            net = net + origin
            cu = build_surface_data(name, net, settings.resolution, (presets.POINTS, presets.POINTS))
            ob = bpy.data.objects.new(cu.name, cu)
            context.collection.objects.link(ob)
            store_meta(ob, raw_meta(net, (presets.POINTS, presets.POINTS)))
            made.append(ob)
        _make_active(context, made[0])
        if self.kind == 'HOOD_FENDER' and self.radius > 0.0:
            context.view_layer.update()
            seams = json.dumps([[made[1].name, 'V0', made[0].name, 'V1']])
            if bpy.ops.spline_surface.fillet(seams=seams, radius=self.radius, continuity='G2',
                                             corners=False) != {'FINISHED'}:
                self.report({'WARNING'}, "The shoulder could not be rounded (radius too large?)")
        return {'FINISHED'}


classes = (SS_OT_add_preset,)
