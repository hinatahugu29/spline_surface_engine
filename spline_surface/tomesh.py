"""To Mesh: sample patches at a uniform density and weld them into one mesh."""
import bpy
import numpy as np

from .core.orient import facing_flips, flip_net
from .core.meshing import segment_counts, t_junctions
from .core.nurbs import eval_points
from . import zebra
from .recipe import world_net


def _stitch_t_junctions(bm, dist):
    """Split boundary edges where another patch's boundary vertex sits on them,
    so the following merge closes the seam. Returns the number of splits."""
    import bmesh
    bm.verts.index_update()
    verts = list(bm.verts)
    boundary = [e for e in bm.edges if e.is_boundary]
    if not boundary:
        return 0
    P = np.array([v.co for v in verts])
    E = np.array([[e.verts[0].index, e.verts[1].index] for e in boundary])
    candidates = sorted({v.index for e in boundary for v in e.verts})
    count = 0
    for row, items in t_junctions(P, E, candidates, dist).items():
        start, end = boundary[row].verts
        edge = boundary[row]
        prev = 0.0
        for f, vi in items:
            _, new = bmesh.utils.edge_split(edge, start, (f - prev) / (1.0 - prev))
            new.co = verts[vi].co
            edge = next(e for e in new.link_edges if end in e.verts)
            start, prev = new, f
            count += 1
    return count


def _split_ngons(bm):
    """Cut faces with more than four corners into quads and triangles.

    Stitching a T-junction turns a quad into a pentagon (or worse). Each such
    face is cut from its straightest corner (the stitched vertex) to a far
    corner, so the leftovers are quads and triangles. Returns the number of cuts.
    """
    import bmesh
    cuts = 0
    for _ in range(10 * len(bm.faces) + 10):
        face = next((f for f in bm.faces if len(f.verts) > 4), None)
        if face is None:
            break
        loops = list(face.loops)
        n = len(loops)
        best, best_i = 2.0, 0
        for i, loop in enumerate(loops):
            a = loops[i - 1].vert.co - loop.vert.co
            b = loops[(i + 1) % n].vert.co - loop.vert.co
            cosang = a.dot(b) / max(a.length * b.length, 1e-30)    # -1 for a straight corner
            if cosang < best:
                best, best_i = cosang, i
        j = (best_i + n // 2) % n
        try:
            bmesh.utils.face_split(face, loops[best_i].vert, loops[j].vert)
        except (ValueError, RuntimeError):
            break
        cuts += 1
    return cuts


def weld_mesh(name, surfaces, resolution, context=None, quads=True, curve_detail=0):
    """One mesh of all patches, sampled uniformly and welded along shared edges.

    Each patch gets segments in proportion to its size (the longest side
    gets `resolution`), so the density is even across big and small patches.
    Patches that share a whole edge get the same count along it and share
    its parametrisation, so their seam vertices meet. Where a
    long edge meets two shorter patches (Fill Hole around a centre, a split
    neighbour), the short side's extra vertices are stitched into the long
    side's edges so the seam still closes.
    With `quads`, the pentagons the stitching leaves are cut into quads and triangles.
    Returns (mesh, fallback_names): fallback_names lists patches that weren't
    a clean control net and so used Blender's own tessellation instead.
    """
    import bmesh
    bm = bmesh.new()
    size = 0.0
    fallback = []
    patches = []
    for ob in surfaces:
        try:
            patches.append(world_net(ob))
        except ValueError:
            # Not a clean patch: fall back to Blender's own tessellation.
            # Its vertex layout generally won't match a neighbour's uniform
            # sampling, so this seam may not fully weld below.
            fallback.append(ob.name)
            dg = (context or bpy.context).evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
            me.transform(ob.matrix_world)
            bm.from_mesh(me)
            bpy.data.meshes.remove(me)
    # patches that touch face the same way, so the mesh's faces do too
    flips = facing_flips(patches, min((context or bpy.context).scene.spline_surface.gap_tolerance, 0.02))
    patches = [(flip_net(net), orders) if f else (net, orders) for (net, orders), f in zip(patches, flips)]
    for (net, orders), (nu, nv) in zip(patches, segment_counts(patches, resolution, curve_detail=curve_detail)):
        size = max(size, np.ptp(net.reshape(-1, 3), axis=0).max())
        U, V = np.meshgrid(np.linspace(0.0, 1.0, nu + 1), np.linspace(0.0, 1.0, nv + 1),
                           indexing="ij")
        pts = eval_points(net, orders, U.ravel(), V.ravel()).reshape(nu + 1, nv + 1, 3)
        verts = [[bm.verts.new(p) for p in row] for row in pts]
        for i in range(nu):
            for j in range(nv):
                bm.faces.new((verts[i][j], verts[i][j + 1], verts[i + 1][j + 1], verts[i + 1][j]))   # Blender's front side
    dist = 1e-4 * max(size, 1e-3)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    if _stitch_t_junctions(bm, dist):
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bmesh.ops.dissolve_degenerate(bm, dist=dist, edges=bm.edges)
    if quads:
        _split_ngons(bm)
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me, fallback


class SS_OT_to_mesh(bpy.types.Operator):
    bl_idname = "spline_surface.to_mesh"
    bl_label = "To Mesh"
    bl_description = ("Create a mesh of the selected surfaces at the given resolution, "
                      "welded into one object by default")
    bl_options = {'REGISTER', 'UNDO'}

    weld: bpy.props.BoolProperty(
        name="Weld", default=True,
        description="Join all patches into one mesh and merge the seams")
    quads: bpy.props.BoolProperty(
        name="Quads", default=True,
        description="Cut the pentagons that welded seams leave into quads and triangles "
                    "(so the mesh subdivides cleanly)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.selected_objects)

    def execute(self, context):
        settings = context.scene.spline_surface
        surfaces = [o for o in context.selected_objects if o.type == 'SURFACE']
        created = []
        weld_fallback = []
        if self.weld:
            me, weld_fallback = weld_mesh(surfaces[0].name + "_mesh", surfaces,
                                          settings.mesh_resolution, context, self.quads,
                                          settings.curve_detail)
            for ob in surfaces:
                for mat in zebra.real_materials(ob):
                    if mat is not None and mat.name not in me.materials:
                        me.materials.append(mat)
            mob = bpy.data.objects.new(me.name, me)
            context.collection.objects.link(mob)
            created.append(mob)
        else:
            for ob in surfaces:
                for sp in ob.data.splines:
                    sp.resolution_u = sp.resolution_v = settings.mesh_resolution
            depsgraph = context.evaluated_depsgraph_get()
            for ob in surfaces:
                mesh = bpy.data.meshes.new_from_object(ob.evaluated_get(depsgraph))
                mesh.name = ob.name + "_mesh"
                mob = bpy.data.objects.new(mesh.name, mesh)
                mob.matrix_world = ob.matrix_world
                context.collection.objects.link(mob)
                created.append(mob)
        for o in context.selected_objects:
            o.select_set(False)
        for mob in created:
            mob.select_set(True)
        context.view_layer.objects.active = created[-1]
        msg = "Created %d mesh object(s)" % len(created)
        if weld_fallback:
            self.report({'WARNING'}, msg + "; not a clean patch, seam may not be welded: "
                        + ", ".join(weld_fallback))
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = (SS_OT_to_mesh,)
