"""Cage editing: deform a whole patch group with a lattice, then bake it.

While the cage is active every object shows the lattice deformation as a
modifier preview. Applying bakes the deformation into the *sources* only
(root patches, curves and hand-edited patches); everything generated from
them is rebuilt from its recipe, so the seams come back exact G1/G2.
"""
import json

import bpy
import numpy as np
from mathutils import Vector

from . import live
from .operators import META_KEY, hand_edited, store_meta, world_net

MOD_NAME = "SS_Cage"
CAGE_KEY = "spline_surface_cage"


def _points(ob):
    """World-space control points (incl. Bezier handles) of a surface or curve."""
    M = np.array(ob.matrix_world)
    pts = []
    for sp in ob.data.splines:
        for p in sp.bezier_points:
            pts += [p.co[:], p.handle_left[:], p.handle_right[:]]
        for p in sp.points:
            pts.append(p.co[:3])
    P = np.array(pts, dtype=float).reshape(-1, 3)
    return P @ M[:3, :3].T + M[:3, 3]


def _set_points(ob, P):
    Mi = np.array(ob.matrix_world.inverted())
    L = P @ Mi[:3, :3].T + Mi[:3, 3]
    i = 0
    for sp in ob.data.splines:
        for p in sp.bezier_points:
            p.co, p.handle_left, p.handle_right = L[i], L[i + 1], L[i + 2]
            i += 3
        for p in sp.points:
            p.co = (*L[i], p.co[3])
            i += 1
    ob.data.update_tag()


def _deform(context, lattice, P):
    """Positions of world points P after the lattice deformation."""
    me = bpy.data.meshes.new("_ss_cage_probe")
    me.vertices.add(len(P))
    me.vertices.foreach_set("co", np.asarray(P, dtype=np.float32).ravel())
    probe = bpy.data.objects.new("_ss_cage_probe", me)
    context.scene.collection.objects.link(probe)
    mod = probe.modifiers.new("cage", 'LATTICE')
    mod.object = lattice
    try:
        context.view_layer.update()
        ev = probe.evaluated_get(context.evaluated_depsgraph_get())
        out = np.empty(len(P) * 3)
        ev.data.vertices.foreach_get("co", out)
        return out.reshape(-1, 3)
    finally:
        bpy.data.objects.remove(probe)
        bpy.data.meshes.remove(me)


def _cage_users(lattice):
    return [o for o in bpy.data.objects
            if any(m.type == 'LATTICE' and m.name == MOD_NAME and m.object == lattice
                   for m in o.modifiers)]


def _find_cage(context):
    ob = context.active_object
    if ob is not None and ob.type == 'LATTICE' and CAGE_KEY in ob:
        return ob
    for o in context.selected_objects:
        for m in o.modifiers:
            if m.name == MOD_NAME and m.object is not None:
                return m.object
    return None


class SS_OT_cage_add(bpy.types.Operator):
    bl_idname = "spline_surface.cage_add"
    bl_label = "Add Cage"
    bl_description = ("Wrap the selected patch groups (and their curves) in a lattice cage. "
                      "Edit the lattice, then Apply Cage")
    bl_options = {'REGISTER', 'UNDO'}

    divisions: bpy.props.IntVectorProperty(
        name="Divisions", size=3, default=(3, 3, 3), min=2, max=16)
    margin: bpy.props.FloatProperty(name="Margin", default=0.05, min=0.0, max=1.0,
                                    subtype='FACTOR')

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type in {'SURFACE', 'CURVE'} for o in context.selected_objects)

    def execute(self, context):
        group = live.connected([o.name for o in context.selected_objects])
        objs = [bpy.data.objects[n] for n in sorted(group)
                if bpy.data.objects[n].type in {'SURFACE', 'CURVE'}]
        P = np.concatenate([_points(o) for o in objs])
        lo, hi = P.min(axis=0), P.max(axis=0)
        size = np.maximum(hi - lo, 1e-3 * max((hi - lo).max(), 1e-3))
        size *= 1.0 + 2.0 * self.margin
        lat = bpy.data.lattices.new("SS_Cage")
        lat.points_u, lat.points_v, lat.points_w = self.divisions
        for axis in ("u", "v", "w"):
            setattr(lat, "interpolation_type_" + axis, 'KEY_BSPLINE')
        cage = bpy.data.objects.new("SS_Cage", lat)
        cage.location = Vector((lo + hi) / 2.0)
        cage.scale = Vector(size)
        cage[CAGE_KEY] = True
        context.collection.objects.link(cage)
        for o in objs:
            mod = o.modifiers.new(MOD_NAME, 'LATTICE')
            mod.object = cage
        for o in context.selected_objects:
            o.select_set(False)
        cage.select_set(True)
        context.view_layer.objects.active = cage
        self.report({'INFO'}, "Cage added around %d object(s): edit it, then Apply Cage"
                    % len(objs))
        return {'FINISHED'}


class SS_OT_cage_apply(bpy.types.Operator):
    bl_idname = "spline_surface.cage_apply"
    bl_label = "Apply Cage"
    bl_description = ("Bake the cage deformation into the source patches and curves, "
                      "rebuild everything generated from them and remove the cage")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and _find_cage(context) is not None

    def execute(self, context):
        cage = _find_cage(context)
        users = _cage_users(cage)
        names = {o.name for o in users}
        regenerated = set()
        for o in users:
            meta = json.loads(o[META_KEY]) if META_KEY in o else None
            inputs = set(meta.get("sources", [])) | {m["ref"] for m in meta.get("matches", [])} \
                if meta else set()
            # Generated from something inside the cage: rebuild instead of bake.
            if inputs & names and not hand_edited(o, meta):
                regenerated.add(o.name)
        bake = [o for o in users if o.name not in regenerated]
        moved = {o.name: _deform(context, cage, _points(o)) for o in bake}
        live._state["busy"] = True
        try:
            for o in users:
                for m in [m for m in o.modifiers if m.name == MOD_NAME]:
                    o.modifiers.remove(m)
            for o in bake:
                _set_points(o, moved[o.name])
                if META_KEY in o:
                    meta = json.loads(o[META_KEY])
                    if meta.get("method") == "raw":
                        net, orders = world_net(o)
                        meta["net"], meta["orders"] = net.tolist(), list(orders)
                        store_meta(o, meta)
            context.view_layer.update()
            result = live.update_chain([o.name for o in bake], context)
        finally:
            live._state["busy"] = False
            live._reset()
        lat = cage.data
        bpy.data.objects.remove(cage)
        if lat.users == 0:
            bpy.data.lattices.remove(lat)
        for o in users:
            o.select_set(True)
        context.view_layer.objects.active = users[0] if users else None
        msg = "Baked %d source(s), rebuilt %d patch(es)" % (len(bake), len(result["ok"]))
        if result["skipped"] or result["error"]:
            self.report({'WARNING'}, msg + "; not rebuilt: "
                        + ", ".join(result["skipped"] + result["error"]))
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class SS_OT_cage_cancel(bpy.types.Operator):
    bl_idname = "spline_surface.cage_cancel"
    bl_label = "Remove Cage"
    bl_description = "Remove the cage without changing anything"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and _find_cage(context) is not None

    def execute(self, context):
        cage = _find_cage(context)
        for o in _cage_users(cage):
            for m in [m for m in o.modifiers if m.name == MOD_NAME]:
                o.modifiers.remove(m)
        lat = cage.data
        bpy.data.objects.remove(cage)
        if lat.users == 0:
            bpy.data.lattices.remove(lat)
        return {'FINISHED'}


classes = (SS_OT_cage_add, SS_OT_cage_apply, SS_OT_cage_cancel)
