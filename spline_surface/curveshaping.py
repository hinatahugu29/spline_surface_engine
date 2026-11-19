"""Curve shaping in meaningful directions, G2 and refitting (core.curvesmooth).

  Constrained Move (edit mode, modal): the selected points move along their
      own tangent (T), normal - towards the bend - (N) or binormal (B), or
      only their handle lengths change (H). K keeps G2 where it was.
  Make G2: scale handle lengths so each point has the same curvature on both
      sides (edit mode: the selected points; object mode: every point of the
      selected curves).
  Refit Curve: the same shape with fewer (or more) points, by a tolerance or
      a count; corners and ends stay.
"""
import bpy
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Vector

from .core.curvesmooth import frame, make_all_g2, point_state, refit
from .curveediting import (bezier_splines, read_spline, restore, selected_points, snapshot,
                           target_objects, write_spline)

AXIS_TEXT = {'T': "along the tangent", 'N': "along the normal (towards the bend)",
             'B': "along the binormal", 'H': "handle lengths"}


class SS_OT_constrained_move(bpy.types.Operator):
    bl_idname = "spline_surface.constrained_move"
    bl_label = "Constrained Move"
    bl_description = ("Move the selected curve points along their own tangent (T), normal "
                      "towards the bend (N) or binormal (B), or change only their handle lengths "
                      "(H). K keeps G2 where the curve was G2. Shift: finer")
    bl_options = {'REGISTER', 'UNDO'}

    axis: bpy.props.EnumProperty(
        name="Move", default='N',
        items=[('T', "Tangent", "Along each point's tangent"),
               ('N', "Normal", "Along each point's normal, towards where the curve bends"),
               ('B', "Binormal", "Out of the plane the curve bends in"),
               ('H', "Handle Length", "Only the handle lengths change (directions kept)")])
    amount: bpy.props.FloatProperty(name="Amount", default=0.0,
                                    description="Distance (T/N/B) or handle length factor as a "
                                                "power of two (H: 1 doubles, -1 halves)")
    keep_g2: bpy.props.BoolProperty(name="Keep G2", default=True,
                                    description="Afterwards, make the points that were G2 G2 again")

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_CURVE'

    def _setup(self, context):
        self._sel = selected_points(context)
        self._chains = {k: read_spline(bpy.data.objects[k[0]], k[1]) for k in self._sel}
        self._was_g2 = {}
        for key, (A, L, R, cyc) in self._chains.items():
            n = len(A)
            near = set()
            for i in self._sel[key]:
                near |= {(i - 1) % n if cyc else i - 1, i, (i + 1) % n if cyc else i + 1}
            self._was_g2[key] = [j for j in sorted(near) if 0 <= j < n
                                 and point_state(A, L, R, j, cyc) == 'G2']

    def _apply(self, context):
        left = 0
        for key, sel in self._sel.items():
            A, L, R, cyc = (x.copy() if hasattr(x, "copy") else x for x in self._chains[key])
            for i in sel:
                if self.axis == 'H':
                    f = 2.0 ** self.amount
                    L[i] = A[i] + (L[i] - A[i]) * f
                    R[i] = A[i] + (R[i] - A[i]) * f
                    continue
                t, nrm, b = frame(*self._chains[key][:3], i, cyc)
                d = {'T': t, 'N': nrm, 'B': b}[self.axis] * self.amount
                A[i] += d
                L[i] += d
                R[i] += d
            if self.keep_g2 and self._was_g2[key]:
                A, L, R, why = make_all_g2(A, L, R, self._was_g2[key], cyc)
                left += len(why)
            write_spline(bpy.data.objects[key[0]], key[1], A, L, R)
        return left

    def execute(self, context):
        if not hasattr(self, "_sel"):
            self._setup(context)
        if not self._sel:
            self.report({'ERROR'}, "Select curve points to move")
            return {'CANCELLED'}
        left = self._apply(context)
        if left:
            self.report({'WARNING'}, "%d point(s) could not be kept G2" % left)
        return {'FINISHED'}

    def _screen_dir(self, context):
        """Screen direction of the first point's axis and the world length of a pixel."""
        key, sel = next(iter(self._sel.items()))
        A, L, R, cyc = self._chains[key]
        i = sel[0]
        t, nrm, b = frame(A, L, R, i, cyc)
        v = {'T': t, 'N': nrm, 'B': b, 'H': t}[self.axis]
        region, rv3d = context.region, context.region_data
        p = A[i]
        size = max(float(np.ptp(A, axis=0).max()), 1e-6) * 1e-3
        s0 = location_3d_to_region_2d(region, rv3d, Vector(p))
        s1 = location_3d_to_region_2d(region, rv3d, Vector(p + size * v))
        if s0 is None or s1 is None or (s1 - s0).length < 1e-6:
            return np.array([1.0, 0.0]), size          # pointing at the viewer: use the x axis
        return np.array(s1 - s0) / (s1 - s0).length, size / (s1 - s0).length

    def _header(self, context, extra=""):
        context.area.header_text_set(
            "Constrained Move: %s %.4g   T/N/B: axis   H: handle lengths   K: keep G2 %s   "
            "Shift: finer   Click/Enter: keep   Esc/RMB: cancel%s"
            % (AXIS_TEXT[self.axis], self.amount, "on" if self.keep_g2 else "off", extra))

    def invoke(self, context, event):
        self._setup(context)
        if not self._sel:
            self.report({'ERROR'}, "Select curve points to move")
            return {'CANCELLED'}
        self._snaps = {n: snapshot(bpy.data.objects[n]) for n in {k[0] for k in self._sel}}
        self._start = np.array((event.mouse_region_x, event.mouse_region_y), float)
        self._dir, self._unit = self._screen_dir(context)
        self.amount = 0.0
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            m = np.array((event.mouse_region_x, event.mouse_region_y), float)
            px = float(np.dot(m - self._start, self._dir))
            self.amount = px / 100.0 if self.axis == 'H' else px * self._unit
            if event.shift:
                self.amount *= 0.1
            left = self._apply(context)
            self._header(context, "   (%d not kept G2)" % left if left else "")
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type in {'T', 'N', 'B', 'H', 'K'}:
            if event.type == 'K':
                self.keep_g2 = not self.keep_g2
            else:
                self.axis = event.type
                self._dir, self._unit = self._screen_dir(context)
                self._start = np.array((event.mouse_region_x, event.mouse_region_y), float)
                self.amount = 0.0
            self._apply(context)
            self._header(context)
            return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            context.area.header_text_set(None)
            return {'FINISHED'}
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            for n, snap in self._snaps.items():
                restore(bpy.data.objects[n], snap)
            context.area.header_text_set(None)
            return {'CANCELLED'}
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}


def _work(context):
    """{(object name, spline index): [anchor indices]}: the selected points in edit
    mode, every point of the selected (else visible) curves in object mode."""
    if context.mode == 'EDIT_CURVE':
        return selected_points(context)
    out = {}
    for ob in target_objects(context):
        for i in bezier_splines(ob):
            out[(ob.name, i)] = list(range(len(ob.data.splines[i].bezier_points)))
    return out


WHY_TEXT = {'corner': "corners", 'straight': "next to a straight side", 'inflection': "inflections",
            'twisted': "twisted (3D)", 'too far': "too far from G2"}


class SS_OT_make_g2(bpy.types.Operator):
    bl_idname = "spline_surface.make_g2"
    bl_label = "Make G2"
    bl_description = ("Make curve points G2 (the same curvature on both sides) by changing only "
                      "their handle lengths; directions and positions stay. Edit mode: the "
                      "selected points; object mode: every point of the selected curves. "
                      "Corners, inflections and points next to a straight side are left (reported)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_CURVE'}

    def execute(self, context):
        work = _work(context)
        if not work:
            self.report({'ERROR'}, "Select curve points (edit mode) or curves (object mode)")
            return {'CANCELLED'}
        done, reasons = 0, {}
        for (name, idx), points in work.items():
            ob = bpy.data.objects[name]
            A, L, R, cyc = read_spline(ob, idx)
            A, L, R, left = make_all_g2(A, L, R, points, cyc)
            write_spline(ob, idx, A, L, R)
            for why in left.values():
                reasons[why] = reasons.get(why, 0) + 1
            done += sum(1 for i in points if point_state(A, L, R, i, cyc) == 'G2')
        if reasons:
            self.report({'WARNING'}, "G2 at %d point(s); left: %s" % (done, ", ".join(
                "%d %s" % (c, WHY_TEXT.get(w, w)) for w, c in sorted(reasons.items()))))
        else:
            self.report({'INFO'}, "G2 at %d point(s)" % done)
        return {'FINISHED'}


class SS_OT_refit_curve(bpy.types.Operator):
    bl_idname = "spline_surface.refit_curve"
    bl_label = "Refit Curve"
    bl_description = ("Rebuild curves with fewer (or more) points along the same shape: the "
                      "fewest points within a tolerance, or a given count. Corners and ends stay. "
                      "Edit mode: the splines with a selected point; object mode: the selected curves")
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="By", default='TOLERANCE',
        items=[('TOLERANCE', "Tolerance", "The fewest points that keep the shape within the tolerance"),
               ('COUNT', "Count", "This many points")])
    tolerance: bpy.props.FloatProperty(name="Tolerance %", default=0.2, min=0.001, max=10.0,
                                       description="Largest allowed change, as a percentage of "
                                                   "each curve's size")
    count: bpy.props.IntProperty(name="Points", default=5, min=2, max=200)

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_CURVE'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "mode", expand=True)
        layout.prop(self, "tolerance" if self.mode == 'TOLERANCE' else "count")

    def execute(self, context):
        work = _work(context)
        if not work:
            self.report({'ERROR'}, "Select curves (object mode) or points (edit mode)")
            return {'CANCELLED'}
        before = after = 0
        worst = 0.0
        for (name, idx) in work:
            ob = bpy.data.objects[name]
            A, L, R, cyc = read_spline(ob, idx)
            size = max(float(np.ptp(np.concatenate([A, L, R]), axis=0).max()), 1e-9)
            if self.mode == 'COUNT':
                A2, L2, R2, err = refit(A, L, R, cyc, count=self.count)
            else:
                A2, L2, R2, err = refit(A, L, R, cyc, tolerance=self.tolerance / 100.0 * size)
            write_spline(ob, idx, A2, L2, R2)
            before, after = before + len(A), after + len(A2)
            worst = max(worst, err / size)
        self.report({'INFO'}, "Refit: %d -> %d points, shape within %.3g%% of the size"
                    % (before, after, 100.0 * worst))
        return {'FINISHED'}


classes = (SS_OT_constrained_move, SS_OT_make_g2, SS_OT_refit_curve)
