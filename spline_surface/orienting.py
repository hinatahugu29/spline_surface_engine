"""Align Normals: make touching patches face the same way.

A patch's normal follows from its u and v directions, and builders such as Bridge Edges
or Sweep derive those from their edges, so a new patch can come out facing against its
neighbour (a different shade in a MatCap, back-face colour in Face Orientation, and
reversed faces in To Mesh). Turning a patch over reverses its u direction; recipe
patches remember that as "normal_flip", which compute_from_meta applies last.
Patches other patches are built from are not turned (their edge names would move under
the patches that use them); the rest of their group is turned to match instead.
"""
import json

import bpy
import numpy as np

from .core.continuity import find_shared_edge
from .core.orient import facing_flips, flip_net, same_facing
from .live import dependency_graph
from .recipe import META_KEY, hand_edited, replace_net, store_meta, world_net


def turn_over(ob, settings):
    """Turn a patch over (same surface, reversed normal)."""
    net, orders = world_net(ob)
    replace_net(ob, flip_net(net), settings, orders)
    if META_KEY in ob:
        meta = json.loads(ob[META_KEY])
        if meta.get("normal_flip"):
            del meta["normal_flip"]
        else:
            meta["normal_flip"] = True
        store_meta(ob, meta)


def face_like(ob, ref, settings):
    """Turn the new patch `ob` over if it faces against `ref`, a patch it touches.
    Returns whether it was turned."""
    try:
        (Pa, oa), (Pb, ob_) = world_net(ob), world_net(ref)
    except ValueError:
        return False
    size = max(np.ptp(Pa.reshape(-1, 3), axis=0).max(), np.ptp(Pb.reshape(-1, 3), axis=0).max())
    shared = find_shared_edge(Pa, Pb, max(size * min(settings.gap_tolerance, 0.02), 1e-6), oa, ob_)
    if shared is None or same_facing(*shared[:3]):
        return False
    turn_over(ob, settings)
    return True


class SS_OT_align_normals(bpy.types.Operator):
    bl_idname = "spline_surface.align_normals"
    bl_label = "Align Normals"
    bl_description = ("Turn patches over where needed so that patches sharing an edge all face "
                      "the same way (the selected patches, or all visible ones if fewer than two "
                      "are selected). Patches that others are built from are left as they are")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(o.type == 'SURFACE' for o in context.visible_objects)

    def execute(self, context):
        settings = context.scene.spline_surface
        obs = [o for o in context.selected_objects if o.type == 'SURFACE']
        if len(obs) < 2:
            obs = [o for o in context.visible_objects if o.type == 'SURFACE']
        patches, used = [], []
        for ob in obs:
            try:
                patches.append(world_net(ob))
                used.append(ob)
            except ValueError:
                pass
        if len(used) < 2:
            self.report({'INFO'}, "Nothing to align")
            return {'CANCELLED'}
        down = dependency_graph()
        pinned = []
        for i, ob in enumerate(used):
            meta = json.loads(ob[META_KEY]) if META_KEY in ob else None
            if down.get(ob.name) or (meta is not None and hand_edited(ob, meta)):
                pinned.append(i)
        active = context.active_object
        root = used.index(active) if active in used else 0
        flips = facing_flips(patches, min(settings.gap_tolerance, 0.02), root, pinned)
        turned, stuck = [], []
        for i, (ob, flip) in enumerate(zip(used, flips)):
            if not flip:
                continue
            if i in pinned:
                stuck.append(ob.name)
            else:
                turn_over(ob, settings)
                turned.append(ob.name)
        msg = "Turned over %d patch(es)" % len(turned)
        if stuck:
            self.report({'WARNING'}, msg + "; could not turn (other patches are built from them, or "
                        "edited by hand): " + ", ".join(stuck))
        else:
            self.report({'INFO'}, msg if turned else "All patches already face the same way")
        return {'FINISHED'}


classes = (SS_OT_align_normals,)
