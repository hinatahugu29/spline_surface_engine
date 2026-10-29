"""Mirror a patch group across a symmetry plane, joined smoothly at the centre."""
import json

import bpy
import numpy as np

from . import live, zebra
from .bridge.write_surface import build_surface_data
from .core.mirror import edges_on_plane
from .operators import compute_from_meta
from .recipe import (META_KEY, _make_active, hand_edited, raw_meta, replace_net, store_meta,
                     world_net)


def _meta(ob):
    return json.loads(ob[META_KEY]) if META_KEY in ob else None


class SS_OT_mirror_group(bpy.types.Operator):
    bl_idname = "spline_surface.mirror_group"
    bl_label = "Mirror Group"
    bl_description = ("Mirror the selected patch groups across a symmetry plane. Edges on the "
                      "plane are squared to it so both halves join with G2; the mirrored "
                      "half follows the original through Live Update")
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Axis", default='X',
        items=[('X', "X", "Mirror across the YZ plane"), ('Y', "Y", "Mirror across the XZ plane"),
               ('Z', "Z", "Mirror across the XY plane")])
    offset: bpy.props.FloatProperty(name="Plane Offset", default=0.0, unit='LENGTH',
                                    description="Position of the symmetry plane on the axis")
    square_centre: bpy.props.BoolProperty(
        name="Smooth Centre", default=True,
        description="Make patch edges lying on the plane cross it at a right angle (G2 join)")
    tolerance: bpy.props.FloatProperty(
        name="On-Plane Tolerance", default=1e-3, min=0.0, unit='LENGTH',
        description="How close an edge must be to the plane to count as lying on it")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.selected_objects)

    def execute(self, context):
        settings = context.scene.spline_surface
        group = live.connected([o.name for o in context.selected_objects])
        down = live.dependency_graph()
        patches = []
        for name in sorted(group):
            ob = bpy.data.objects[name]
            meta = _meta(ob)
            if ob.type != 'SURFACE' or (meta and meta.get("method") == "mirror"):
                continue
            if any((_meta(bpy.data.objects[d]) or {}).get("method") == "mirror"
                   for d in down.get(name, ()) if d in bpy.data.objects):
                continue  # already mirrored
            patches.append(ob)
        if not patches:
            self.report({'WARNING'}, "Nothing new to mirror")
            return {'CANCELLED'}
        squared, made = [], []
        live._state["busy"] = True
        try:
            if self.square_centre:
                for ob in patches:
                    try:
                        net, orders = world_net(ob)
                    except ValueError:
                        continue
                    if not edges_on_plane(net, self.axis, self.offset, self.tolerance):
                        continue
                    meta = _meta(ob)
                    if meta is None or hand_edited(ob, meta):
                        old = meta.get("matches", []) if meta else []
                        meta = raw_meta(net, orders, matches=old)
                    meta["symmetry"] = {"axis": self.axis, "offset": self.offset,
                                        "tol": self.tolerance}
                    try:
                        net, orders, _ = compute_from_meta(meta, context)
                    except ValueError as e:
                        self.report({'WARNING'}, "%s: %s" % (ob.name, e))
                        continue
                    replace_net(ob, net, settings, orders)
                    store_meta(ob, meta)
                    squared.append(ob.name)
                live.update_chain(squared, context)
            for ob in patches:
                meta = {"method": "mirror", "sources": [ob.name], "axis": self.axis,
                        "offset": self.offset}
                try:
                    net, orders, _ = compute_from_meta(meta, context)
                    cu = build_surface_data(ob.name + "_mirror", net, settings.resolution, orders)
                except ValueError as e:
                    self.report({'WARNING'}, "%s: %s" % (ob.name, e))
                    continue
                for mat in zebra.real_materials(ob):
                    cu.materials.append(mat)
                mob = bpy.data.objects.new(cu.name, cu)
                for coll in ob.users_collection:
                    coll.objects.link(mob)
                    break
                else:
                    context.collection.objects.link(mob)
                store_meta(mob, meta)
                made.append(mob)
        finally:
            live._state["busy"] = False
            live._reset()
        if made:
            _make_active(context, made[-1])
        self.report({'INFO'}, "Mirrored %d patch(es), squared %d centre edge(s)"
                    % (len(made), len(squared)))
        return {'FINISHED'}


classes = (SS_OT_mirror_group,)
