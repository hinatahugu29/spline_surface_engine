"""Making curves from curves (core.curvemake).

  Two-View Curve: a side-view curve and a top-view (or front-view) curve
      become one 3D curve - the classic way to draw a car's character lines.
  Curve Fillet: round corner points with a radius (G1).
  Offset Curve: a parallel copy of a flat curve, in its plane.
  Extend Curve: lengthen ends, continuing the end's own bend or straight on.
"""
import bpy
import numpy as np

from .bridge.read_curves import read_polyline
from .core.curvemake import extend_end, fillet_corner, fit_points, offset_points, two_view
from .core.curvepull import chain_plane
from .core.curvesmooth import point_state
from .curveediting import (bezier_splines, read_spline, selected_points, target_objects,
                           write_spline)


def new_curve(context, name, A, L, R, cyclic=False):
    """A new curve object (world space) with one Bezier spline, made active."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(A) - 1)
    sp.use_cyclic_u = cyclic
    ob = bpy.data.objects.new(name, cu)
    context.collection.objects.link(ob)
    write_spline(ob, 0, A, L, R)
    if context.mode == 'OBJECT':
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
    return ob


def _size(A):
    return max(float(np.ptp(A, axis=0).max()), 1e-9)


class SS_OT_two_view_curve(bpy.types.Operator):
    bl_idname = "spline_surface.two_view_curve"
    bl_label = "Two-View Curve"
    bl_description = ("Make one 3D curve from two flat curves drawn in two views (side and top, "
                      "or side and front): along the axis both run along, each view gives the "
                      "coordinate the other cannot see. Select the two curves")
    bl_options = {'REGISTER', 'UNDO'}

    tolerance: bpy.props.FloatProperty(name="Tolerance %", default=0.1, min=0.001, max=5.0,
                                       description="How closely the new curve follows the two views, "
                                                   "as a percentage of its size (fewer points when larger)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        curves = [o for o in context.selected_objects if o.type == 'CURVE']
        if len(curves) != 2:
            self.report({'ERROR'}, "Select two curves drawn in two views (got %d)" % len(curves))
            return {'CANCELLED'}
        dg = context.evaluated_depsgraph_get()
        try:
            polys = [read_polyline(o, dg, 32) for o in curves]
            pts, axis = two_view(*polys)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        A, L, R, err = fit_points(pts, tolerance=self.tolerance / 100.0 * _size(pts))
        new_curve(context, "TwoViewCurve", A, L, R)
        self.report({'INFO'}, "3D curve along %s from %s and %s: %d points"
                    % ("XYZ"[axis], curves[0].name, curves[1].name, len(A)))
        return {'FINISHED'}


def _corners(context):
    """{(object name, spline index): [corner indices]}: selected ones in edit mode,
    every corner of the selected curves in object mode."""
    out = {}
    if context.mode == 'EDIT_CURVE':
        for key, sel in selected_points(context).items():
            A, L, R, cyc = read_spline(bpy.data.objects[key[0]], key[1])
            c = [i for i in sel if point_state(A, L, R, i, cyc) == 'CORNER']
            if c:
                out[key] = c
        return out
    for ob in target_objects(context):
        for idx in bezier_splines(ob):
            A, L, R, cyc = read_spline(ob, idx)
            c = [i for i in range(len(A)) if point_state(A, L, R, i, cyc) == 'CORNER']
            if c:
                out[(ob.name, idx)] = c
    return out


class SS_OT_curve_fillet(bpy.types.Operator):
    bl_idname = "spline_surface.curve_fillet"
    bl_label = "Curve Fillet"
    bl_description = ("Round the corner points of curves with a radius: the curve is cut that far "
                      "along it on both sides and joined by an arc-like span (G1). Edit mode: the "
                      "selected corner points; object mode: every corner of the selected curves")
    bl_options = {'REGISTER', 'UNDO'}

    radius: bpy.props.FloatProperty(name="Radius", default=0.0, min=0.0, unit='LENGTH',
                                    description="How far along the curve each side is cut back "
                                                "(0: a fifth of the shorter neighbouring span)")

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_CURVE'}

    def execute(self, context):
        work = _corners(context)
        if not work:
            self.report({'WARNING'}, "No corner points to round")
            return {'CANCELLED'}
        done = 0
        for (name, idx), corners in work.items():
            ob = bpy.data.objects[name]
            A, L, R, cyc = read_spline(ob, idx)
            for i in sorted(corners, reverse=True):           # later ones first: indices stay
                r = self.radius
                if r <= 0:
                    n = len(A)
                    near = [A[(i - 1) % n] if (cyc or i > 0) else None,
                            A[(i + 1) % n] if (cyc or i < n - 1) else None]
                    r = 0.2 * min(np.linalg.norm(p - A[i]) for p in near if p is not None)
                try:
                    A, L, R, _ = fillet_corner(A, L, R, i, r, cyc)
                    done += 1
                except ValueError:
                    pass
            write_spline(ob, idx, A, L, R)
        self.report({'INFO'}, "Rounded %d corner(s)" % done)
        return {'FINISHED'} if done else {'CANCELLED'}


class SS_OT_offset_curve(bpy.types.Operator):
    bl_idname = "spline_surface.offset_curve"
    bl_label = "Offset Curve"
    bl_description = ("A parallel copy of each selected flat curve, in its own plane (a panel "
                      "line, a trim's inner edge). Negative Distance offsets to the other side")
    bl_options = {'REGISTER', 'UNDO'}

    distance: bpy.props.FloatProperty(name="Distance", default=0.1, unit='LENGTH')
    tolerance: bpy.props.FloatProperty(name="Tolerance %", default=0.1, min=0.001, max=5.0)

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        made, skipped = [], 0
        for ob in target_objects(context):
            for idx in bezier_splines(ob):
                A, L, R, cyc = read_spline(ob, idx)
                plane = chain_plane(A, L, R)
                if plane is None:
                    skipped += 1
                    continue
                nrm = plane[1] * (1 if plane[1][int(np.argmax(np.abs(plane[1])))] > 0 else -1)
                pts = offset_points(A, L, R, self.distance, nrm, cyc)
                if cyc:
                    pts = pts[:-1]
                B, M, N, _ = fit_points(pts, cyc, self.tolerance / 100.0 * _size(pts))
                made.append(new_curve(context, ob.name + "_offset", B, M, N, cyc))
        if not made:
            self.report({'ERROR'}, "Select flat curves (a curve that is not flat has no plane to offset in)")
            return {'CANCELLED'}
        for o in made:
            o.select_set(True)
        self.report({'INFO'} if not skipped else {'WARNING'},
                    "Offset %d curve(s)%s" % (len(made), ", %d not flat" % skipped if skipped else ""))
        return {'FINISHED'}


class SS_OT_extend_curve(bpy.types.Operator):
    bl_idname = "spline_surface.extend_curve"
    bl_label = "Extend Curve"
    bl_description = ("Lengthen curve ends: Natural continues the end's own bend (perfectly "
                      "smooth), Straight goes straight on along the end tangent. Edit mode: the "
                      "selected end points; object mode: the chosen ends of the selected curves")
    bl_options = {'REGISTER', 'UNDO'}

    length: bpy.props.FloatProperty(name="Length", default=0.5, min=0.0, unit='LENGTH')
    mode: bpy.props.EnumProperty(name="Shape", default='NATURAL',
                                 items=[('NATURAL', "Natural", "Continue the end's own bend"),
                                        ('STRAIGHT', "Straight", "Straight on along the end tangent")])
    ends: bpy.props.EnumProperty(name="Ends", default='END',
                                 items=[('END', "End", "The last point"), ('START', "Start", "The first point"),
                                        ('BOTH', "Both", "Both ends")],
                                 description="Object mode: which ends")

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_CURVE'}

    def execute(self, context):
        work = []
        if context.mode == 'EDIT_CURVE':
            for (name, idx), sel in selected_points(context).items():
                sp = bpy.data.objects[name].data.splines[idx]
                if sp.use_cyclic_u:
                    continue
                n = len(sp.bezier_points)
                which = (['START'] if 0 in sel else []) + (['END'] if n - 1 in sel else [])
                if which:
                    work.append((name, idx, which))
        else:
            which = ['START', 'END'] if self.ends == 'BOTH' else [self.ends]
            for ob in target_objects(context):
                for idx in bezier_splines(ob):
                    if not ob.data.splines[idx].use_cyclic_u:
                        work.append((ob.name, idx, which))
        if not work:
            self.report({'ERROR'}, "Select open curves (object mode) or their end points (edit mode)")
            return {'CANCELLED'}
        count = 0
        for name, idx, which in work:
            ob = bpy.data.objects[name]
            A, L, R, cyc = read_spline(ob, idx)
            for end in which:
                A, L, R = extend_end(A, L, R, end, self.length, self.mode)
                write_spline(ob, idx, A, L, R, inserted=0 if end == 'START' else len(A) - 1)
                count += 1
        self.report({'INFO'}, "Extended %d end(s)" % count)
        return {'FINISHED'}


classes = (SS_OT_two_view_curve, SS_OT_curve_fillet, SS_OT_offset_curve, SS_OT_extend_curve)
