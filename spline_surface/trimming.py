"""Trim (prototype): a mesh of a patch with holes cut along closed curves.

Select the closed curves that outline the holes (a window, a lamp) and the
patch, then run it. The curves are projected onto the patch (nearest point,
along an axis or the view), and the patch is meshed with those outlines cut
out. The result is a plain mesh: it is a snapshot, and the patch itself stays
whole. This is a prototype: edges of the patch are not welded to neighbours
and the hole outline is resolved to the grid. The geometry lives in core.trim.
"""
import bpy
import numpy as np

from .bridge.read_curves import read_loop
from .core.project import project_points
from .core.trim import trim_grid
from .projecting import MODES, SS_OT_project_curve
from .recipe import world_net


class SS_OT_trim_mesh(bpy.types.Operator):
    bl_idname = "spline_surface.trim_mesh"
    bl_label = "Trim to Mesh"
    bl_description = ("Prototype: mesh the selected patch with holes cut out along the selected "
                      "closed curves (window, lamp). The patch itself is not changed")
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(name="Direction", items=MODES, default='NEAREST',
                                 description="How the curves are laid onto the patch")
    resolution: bpy.props.IntProperty(
        name="Resolution", default=48, min=8, max=400,
        description="Grid cells along each side of the patch before the holes are cut")
    points: bpy.props.IntProperty(
        name="Outline Points", default=64, min=8, max=400,
        description="Points taken along each curve to outline its hole")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        patch = next((o for o in context.selected_objects if o.type == 'SURFACE'), None)
        curves = [o for o in context.selected_objects if o.type == 'CURVE']
        if patch is None or not curves:
            self.report({'ERROR'}, "Select a patch and the closed curves that outline the holes")
            return {'CANCELLED'}
        try:
            net, orders = world_net(patch)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        direction = SS_OT_project_curve._direction(self, context)
        loops = []
        for cv in curves:
            line = read_loop(cv, context.evaluated_depsgraph_get(), max(self.points // 4, 8))
            seg = np.linalg.norm(np.diff(np.vstack([line, line[:1]]), axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            t = np.linspace(0.0, s[-1], self.points, endpoint=False)
            closed = np.vstack([line, line[:1]])
            pts = np.stack([np.interp(t, s, closed[:, k]) for k in range(3)], axis=1)
            _, uv, ok = project_points(net, orders, pts, direction)
            if not ok.all():
                self.report({'ERROR'}, "%s does not lie fully on the patch in that direction" % cv.name)
                return {'CANCELLED'}
            loops.append(uv)
        verts, quads, _ = trim_grid(net, orders, loops, self.resolution, self.resolution)
        if not len(quads):
            self.report({'ERROR'}, "The holes cover the whole patch")
            return {'CANCELLED'}
        me = bpy.data.meshes.new(patch.name + "_trim")
        me.from_pydata([tuple(v) for v in verts], [], [tuple(q) for q in quads])
        me.update()
        ob = bpy.data.objects.new(patch.name + "_trim", me)
        context.collection.objects.link(ob)
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        return {'FINISHED'}


classes = (SS_OT_trim_mesh,)
