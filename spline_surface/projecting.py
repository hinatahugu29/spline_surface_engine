"""Project Curve: lay a curve onto a patch.

Select a curve and a patch, then run it: the curve is sampled and every sample
is moved onto the patch, to the nearest point or along an axis or the view
direction. The result is a new Bezier curve lying on the surface, usable like
any boundary curve (Coons, Network, Fill Hole). It is a snapshot: it does not
follow later edits of the curve or the patch. The geometry lives in core.project.
"""
import bpy
import numpy as np

from .bridge.read_curves import read_polyline
from .core.project import project_points
from .recipe import _make_active, world_net

MODES = [
    ('NEAREST', "Nearest", "Move each point to the closest point of the patch"),
    ('X', "Along X", "Slide each point along the X axis onto the patch"),
    ('Y', "Along Y", "Slide each point along the Y axis onto the patch"),
    ('Z', "Along Z", "Slide each point along the Z axis onto the patch"),
    ('VIEW', "View", "Slide each point along the viewing direction onto the patch"),
]
_AXES = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}


def make_bezier_curve(name, points, collection):
    """Bezier curve object (world space, AUTO handles) through `points`."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(points) - 1)
    for bp, co in zip(sp.bezier_points, points):
        bp.co = tuple(co)
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    return ob


class SS_OT_project_curve(bpy.types.Operator):
    bl_idname = "spline_surface.project_curve"
    bl_label = "Project Curve"
    bl_description = ("Lay the selected curve onto the selected patch (nearest point, along an axis "
                      "or along the view) as a new curve lying on its surface")
    bl_options = {'REGISTER', 'UNDO'}

    curve: bpy.props.StringProperty(name="Curve", description="Curve to project (empty: the selected curve)")
    patch: bpy.props.StringProperty(name="Patch", description="Patch to project onto (empty: the selected patch)")
    mode: bpy.props.EnumProperty(name="Direction", items=MODES, default='NEAREST')
    samples: bpy.props.IntProperty(
        name="Points", default=24, min=3, max=200,
        description="How many points of the curve are projected (the new curve passes through them)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def _pick(self, context):
        curve = bpy.data.objects.get(self.curve) if self.curve else next(
            (o for o in context.selected_objects if o.type == 'CURVE'), None)
        patch = bpy.data.objects.get(self.patch) if self.patch else next(
            (o for o in context.selected_objects if o.type == 'SURFACE'), None)
        return curve, patch

    def _direction(self, context):
        if self.mode == 'NEAREST':
            return None
        if self.mode in _AXES:
            return np.array(_AXES[self.mode])
        rv3d = context.region_data
        if rv3d is None:
            return np.array(_AXES['Z'])
        return np.array(rv3d.view_rotation @ __import__("mathutils").Vector((0.0, 0.0, -1.0)))

    def execute(self, context):
        curve, patch = self._pick(context)
        if curve is None or curve.type != 'CURVE' or patch is None or patch.type != 'SURFACE':
            self.report({'ERROR'}, "Select a curve and a patch")
            return {'CANCELLED'}
        try:
            line = read_polyline(curve, context.evaluated_depsgraph_get(), self.samples * 4)
            net, orders = world_net(patch)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        # keep `samples` points, evenly by arc length
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, s[-1], self.samples)
        pts = np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)
        out, _, ok = project_points(net, orders, pts, self._direction(context))
        if ok.sum() < 2:
            self.report({'ERROR'}, "The curve does not reach the patch in that direction")
            return {'CANCELLED'}
        ob = make_bezier_curve(curve.name + "_proj", out[ok], context.collection)
        _make_active(context, ob)
        if not ok.all():
            self.report({'WARNING'}, "%d of %d points missed the patch and were left out"
                        % ((~ok).sum(), len(ok)))
        return {'FINISHED'}


classes = (SS_OT_project_curve,)
