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

    scope: bpy.props.EnumProperty(
        name="Scope", default='GROUP',
        items=[('GROUP', "Group", "The selected patches and everything linked to them by recipes"),
               ('SELECTED', "Selected", "Only the selected patches"),
               ('DOWNSTREAM', "Selected + Built From Them",
                "The selected patches and the patches made from them (fillets, extensions...), "
                "not the patches they were made from")])
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
        picked = [o.name for o in context.selected_objects]
        down = live.dependency_graph()
        if self.scope == 'SELECTED':
            group = set(picked)
        elif self.scope == 'DOWNSTREAM':
            group = set(picked) | set(live.downstream_order(picked, down)[0])
        else:
            group = live.connected(picked)
        # Patches built from a mirrored patch already live on the mirrored side: mirroring
        # them again would put copies back on the original side, so the group scopes skip them.
        mirrors = [o.name for o in bpy.data.objects
                   if META_KEY in o and (_meta(o) or {}).get("method") == "mirror"]
        on_mirror_side = set(live.downstream_order(mirrors, down)[0])             if self.scope != 'SELECTED' else set()
        skipped_mirror_side = 0
        patches = []
        for name in sorted(group):
            ob = bpy.data.objects[name]
            meta = _meta(ob)
            if ob.type != 'SURFACE' or (meta and meta.get("method") == "mirror"):
                continue
            if name in on_mirror_side:
                skipped_mirror_side += 1
                continue
            if any((_meta(bpy.data.objects[d]) or {}).get("method") == "mirror"
                   for d in down.get(name, ()) if d in bpy.data.objects):
                continue  # already mirrored
            patches.append(ob)
        if not patches:
            self.report({'WARNING'}, "Nothing new to mirror" + (
                " (%d patch(es) built from mirrored patches were left alone)" % skipped_mirror_side
                if skipped_mirror_side else ""))
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
        note = "" if not skipped_mirror_side else (
            "; left %d patch(es) built from mirrored patches (use Scope: Selected to mirror them)"
            % skipped_mirror_side)
        self.report({'INFO'}, "Mirrored %d patch(es), squared %d centre edge(s)%s"
                    % (len(made), len(squared), note))
        return {'FINISHED'}


def _mirrors_of(names, down):
    """Mirror patches among `names`, plus the mirrors made from any of them."""
    out = []
    for name in names:
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != 'SURFACE':
            continue
        if (_meta(ob) or {}).get("method") == "mirror":
            out.append(name)
        for d in sorted(down.get(name, ())):
            dob = bpy.data.objects.get(d)
            if dob is not None and (_meta(dob) or {}).get("method") == "mirror":
                out.append(d)
    return list(dict.fromkeys(out))


def _freeze(ob):
    net, orders = world_net(ob)
    store_meta(ob, raw_meta(net, orders))


class SS_OT_mirror_unlink(bpy.types.Operator):
    bl_idname = "spline_surface.mirror_unlink"
    bl_label = "Unlink Mirror"
    bl_description = ("Cut the mirrored patches (selected, or made from the selected patches) "
                      "loose from their originals. They keep their shape and no longer follow "
                      "the original, so the two sides can differ")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.selected_objects)

    def execute(self, context):
        names = _mirrors_of([o.name for o in context.selected_objects], live.dependency_graph())
        if not names:
            self.report({'WARNING'}, "No mirrored patches among the selection")
            return {'CANCELLED'}
        live._state["busy"] = True
        try:
            for name in names:
                _freeze(bpy.data.objects[name])
        finally:
            live._state["busy"] = False
            live._reset()
        self.report({'INFO'}, "Unlinked %d mirrored patch(es)" % len(names))
        return {'FINISHED'}


class SS_OT_mirror_remove(bpy.types.Operator):
    bl_idname = "spline_surface.mirror_remove"
    bl_label = "Remove Mirror"
    bl_description = ("Delete the mirrored patches (selected, or made from the selected patches). "
                      "Patches built from a deleted mirror keep their shape and are cut loose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.selected_objects)

    def execute(self, context):
        down = live.dependency_graph()
        names = _mirrors_of([o.name for o in context.selected_objects], down)
        if not names:
            self.report({'WARNING'}, "No mirrored patches among the selection")
            return {'CANCELLED'}
        gone = set(names)
        orphans = {d for n in names for d in down.get(n, ()) if d not in gone}
        live._state["busy"] = True
        try:
            for name in sorted(orphans):
                ob = bpy.data.objects.get(name)
                if ob is not None and ob.type == 'SURFACE':
                    _freeze(ob)
            for name in names:
                ob = bpy.data.objects[name]
                data = ob.data
                bpy.data.objects.remove(ob, do_unlink=True)
                if data is not None and data.users == 0:     # keep the name free for a new mirror
                    bpy.data.curves.remove(data)
        finally:
            live._state["busy"] = False
            live._reset()
        self.report({'INFO'}, "Removed %d mirrored patch(es)%s" % (
            len(names), ", froze %d built from them" % len(orphans) if orphans else ""))
        return {'FINISHED'}


classes = (SS_OT_mirror_group, SS_OT_mirror_unlink, SS_OT_mirror_remove)
