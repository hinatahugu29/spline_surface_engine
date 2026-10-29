"""Control Mesh: a patch's control points as a plain mesh you edit with G/R/S.

Edit the mesh's vertices in Edit Mode and the patch follows. Vertices pinned by
the patch's seams (core.lock) snap back, so every seam keeps its G0/G1/G2;
only the free interior moves. Like Soft Pull, the patch ends up as if edited
by hand (Live Update leaves it alone, Regenerate resets it).
"""
import bmesh
import bpy
import numpy as np

from .ctrlpoints import pin_masks
from .live import _write_in_place
from .recipe import _make_active, replace_net, world_net

PATCH_KEY = "ss_cp_patch"
GROUP = "Pinned"

_masks = {}          # control mesh name -> pinned mask, computed when it was made
_state = {"busy": False}


def _control_mesh_of(patch):
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.get(PATCH_KEY) == patch.name:
            return ob
    return None


def _grid(net):
    """Vertices (i * nv + j order) and edges of a control net's grid."""
    nu, nv = net.shape[:2]
    edges = [(i * nv + j, (i + 1) * nv + j) for i in range(nu - 1) for j in range(nv)]
    edges += [(i * nv + j, i * nv + j + 1) for i in range(nu) for j in range(nv - 1)]
    return net.reshape(-1, 3), edges


def make_control_mesh(context, patch):
    """Create (or refresh) the control mesh of `patch`; returns the object."""
    net, _, mask = pin_masks(context, [patch])[patch.name]
    verts, edges = _grid(net)
    ob = _control_mesh_of(patch)
    if ob is None:
        me = bpy.data.meshes.new(patch.name + "_CP")
        me.from_pydata(verts.tolist(), edges, [])
        ob = bpy.data.objects.new(patch.name + "_CP", me)
        (patch.users_collection[0] if patch.users_collection else context.scene.collection
         ).objects.link(ob)
        ob[PATCH_KEY] = patch.name
        ob.show_in_front = True
        ob.display_type = 'WIRE'
        ob.lock_location = ob.lock_rotation = ob.lock_scale = (True, True, True)
    else:
        me = ob.data
        if len(me.vertices) == len(verts):
            me.vertices.foreach_set("co", verts.ravel())
            me.update()
        else:
            me.clear_geometry()
            me.from_pydata(verts.tolist(), edges, [])
    group = ob.vertex_groups.get(GROUP) or ob.vertex_groups.new(name=GROUP)
    group.remove(range(len(verts)))
    pinned = np.flatnonzero(mask.ravel()).tolist()
    if pinned:
        group.add(pinned, 1.0, 'REPLACE')
    _masks[ob.name] = mask
    return ob


def _read_vertices(ob, count):
    """World coordinates of the control mesh's vertices, or None if the count changed."""
    me = ob.data
    if ob.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(me)
        if len(bm.verts) != count:
            return None
        co = np.array([v.co for v in bm.verts], float)
    else:
        if len(me.vertices) != count:
            return None
        co = np.empty(count * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def _snap_back(ob, world):
    """Put the (world-space) coordinates back into the control mesh's vertices."""
    M = np.array(ob.matrix_world.inverted())
    co = world @ M[:3, :3].T + M[:3, 3]
    me = ob.data
    if ob.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(me)
        for v, c in zip(bm.verts, co):
            v.co = c
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
    else:
        me.vertices.foreach_set("co", co.ravel())
        me.update()


def sync(context, ob):
    """Apply the control mesh `ob` to its patch. Returns a status string."""
    patch = bpy.data.objects.get(ob.get(PATCH_KEY, ""))
    if patch is None or patch.type != 'SURFACE':
        return "patch missing"
    try:
        net0, orders = world_net(patch)
    except ValueError:
        return "patch unreadable"
    nu, nv = net0.shape[:2]
    mask = _masks.get(ob.name)
    if mask is None or mask.shape != (nu, nv):
        mask = _masks[ob.name] = pin_masks(context, [patch])[patch.name][2]
    co = _read_vertices(ob, nu * nv)
    if co is None:
        return "vertex count changed: add/delete vertices is not supported"
    new = co.reshape(nu, nv, 3)
    snapped = bool(np.abs(new[mask] - net0[mask]).max(initial=0.0) > 1e-9)
    new[mask] = net0[mask]
    if snapped:
        _snap_back(ob, new.reshape(-1, 3))
    if np.abs(new - net0).max() > 1e-9:
        if not _write_in_place(patch, new, orders):
            replace_net(patch, new, context.scene.spline_surface, orders)
    return "ok"


def _on_depsgraph(scene, depsgraph):
    if _state["busy"]:
        return
    touched = []
    for u in depsgraph.updates:
        idb = getattr(u.id, "original", u.id)
        if isinstance(idb, bpy.types.Object) and idb.type == 'MESH' and PATCH_KEY in idb:
            touched.append(idb)
    if not touched:
        return
    _state["busy"] = True
    try:
        for ob in touched:
            status = sync(bpy.context, ob)
            if status != "ok":
                scene.spline_surface.live_note = "Control Mesh: " + status
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        _state["busy"] = False


class SS_OT_control_mesh(bpy.types.Operator):
    bl_idname = "spline_surface.control_mesh"
    bl_label = "Control Mesh"
    bl_description = ("Edit the active patch through its control points as a mesh (Edit Mode, "
                      "G/R/S). Points pinned by seams (vertex group Pinned) snap back, so "
                      "G0/G1/G2 are kept. Use Done to remove the mesh")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and (ob.type == 'SURFACE' or PATCH_KEY in ob) \
            and context.mode in {'OBJECT', 'EDIT_MESH', 'EDIT_SURFACE'}

    def execute(self, context):
        ob = context.active_object
        if ob.type == 'MESH':  # already a control mesh: just go back into it
            cp = ob
        else:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            try:
                cp = make_control_mesh(context, ob)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
        _make_active(context, cp)
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, "Move free points; Pinned points snap back")
        return {'FINISHED'}


class SS_OT_control_mesh_done(bpy.types.Operator):
    bl_idname = "spline_surface.control_mesh_done"
    bl_label = "Done"
    bl_description = "Remove the control mesh and go back to the patch"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and PATCH_KEY in ob

    def execute(self, context):
        cp = context.active_object
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        sync(context, cp)
        patch = bpy.data.objects.get(cp[PATCH_KEY])
        me = cp.data
        _masks.pop(cp.name, None)
        bpy.data.objects.remove(cp)
        if me.users == 0:
            bpy.data.meshes.remove(me)
        if patch is not None:
            _make_active(context, patch)
        return {'FINISHED'}


classes = (SS_OT_control_mesh, SS_OT_control_mesh_done)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
