"""Make patches and the curves they are built from: Add Patch, Line / Arc,
Curve Grid, Coons, Loft, Snap Ends, Curve Network."""

import bpy
import numpy as np

from .bridge.write_surface import build_surface_data
from .core.orient import flip_net, part_flips
from .core.extend import bezier_orders
from .core.loft import order_sections
from . import pick
from .recipe import (_make_active, iso_lines, raw_meta, set_origin,
                     store_meta)
from .rebuild import (BOUNDARY_TYPES, compute_coons, compute_loft, compute_network)


class SS_OT_coons(bpy.types.Operator):
    bl_idname = "spline_surface.coons"
    bl_label = "Curves → Coons"
    bl_description = "Fill 3 or 4 selected boundary curves with a Bezier NURBS patch"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.spline_surface
        sources = [o for o in context.selected_objects if o.type in BOUNDARY_TYPES]
        if len(sources) not in (3, 4):
            self.report({'ERROR'}, "Select 3 or 4 curves (got %d)" % len(sources))
            return {'CANCELLED'}
        try:
            net, err = compute_coons(sources, context.evaluated_depsgraph_get(), settings)
            cu = build_surface_data("CoonsPatch", net, settings.resolution)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        ob = bpy.data.objects.new("CoonsPatch", cu)
        context.collection.objects.link(ob)
        store_meta(ob, {"method": "coons", "sources": [o.name for o in sources]})
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        self.report({'INFO'}, "Coons patch created (max fit error %.3g)" % err)
        return {'FINISHED'}


def _end_points(ob):
    """(first, last) editable end points of the first spline of a curve object."""
    sp = ob.data.splines[0]
    pts = sp.bezier_points if sp.type == 'BEZIER' else sp.points
    return pts[0], pts[-1]


def _move_point(pt, delta):
    if hasattr(pt, "handle_left"):
        pt.co += delta
        pt.handle_left += delta
        pt.handle_right += delta
    else:
        x, y, z, w = pt.co
        pt.co = (x + delta.x, y + delta.y, z + delta.z, w)


class SS_OT_loft(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.loft"
    bl_label = "Curves → Loft"
    bl_description = ("Loft a patch through 2 or more section curves. Click the curves in the "
                      "order the patch should pass through them, then Enter (curves selected "
                      "beforehand start out picked). Up to Degree+1 sections are hit exactly")
    bl_options = {'REGISTER', 'UNDO'}

    rows: bpy.props.IntProperty(
        name="Extra Rows", default=2, min=0, max=4,
        description="Control point rows added between the sections without changing the shape "
                    "(more points to shape with; limited by the Degree setting)")
    sections: bpy.props.StringProperty(
        name="Sections", default="",
        description="Curve names in loft order, separated by ';'. Empty: the selected curves, "
                    "ordered automatically")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    # -------------------------------------------------------------- modal
    def invoke(self, context, event):
        self.sections = ""            # Blender remembers the last run's properties: pick afresh
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._curves = pick.curve_lines(context)
        if len(self._curves) < 2:
            self.report({'ERROR'}, "Need at least two visible curves to loft through")
            return {'CANCELLED'}
        chosen = [o for o in context.selected_objects if (o.name, 'CURVE') in self._curves]
        if len(chosen) > 1:           # keep the old way: selected curves, ordered automatically
            polys = [self._curves[(o.name, 'CURVE')] for o in chosen]
            chosen = [chosen[i] for i, _ in order_sections(polys)]
        self._chosen = [(o.name, 'CURVE') for o in chosen]
        self._hover = None
        self._preview = []
        self._preview_key = None
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        self._update_preview(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        context.area.header_text_set(
            "Loft: click the section curves in order (%d picked)   Click a picked one: remove   "
            "Backspace: remove the last   Enter: loft   Esc/RMB: cancel" % len(self._chosen))

    def _label(self):
        if self._hover is None:
            return None
        pts = self._curves[self._hover]
        n = self._chosen.index(self._hover) + 1 if self._hover in self._chosen else None
        text = "%s  #%d" % (self._hover[0], n) if n else self._hover[0]
        return text, pts[len(pts) // 2]

    def _draw_lines(self):
        lines = [(pts, pick.COLOR_EDGE, 2.0) for k, pts in self._curves.items()
                 if k != self._hover and k not in self._chosen]
        lines += [(self._curves[k], pick.COLOR_SELECTED, 4.0) for k in self._chosen]
        if self._hover and self._hover not in self._chosen:
            lines.append((self._curves[self._hover], pick.COLOR_HOVER, 4.0))
        lines += [(pts, pick.COLOR_PREVIEW, 1.0) for pts in self._preview]
        return lines

    def _update_preview(self, context):
        keys = self._chosen + ([self._hover] if self._hover and self._hover not in self._chosen
                               else [])
        if tuple(keys) == self._preview_key:
            return
        self._preview_key = tuple(keys)
        self._preview = []
        if len(keys) >= 2:
            try:
                net, _, _ = compute_loft([bpy.data.objects[k[0]] for k in keys],
                                         context.evaluated_depsgraph_get(),
                                         context.scene.spline_surface, keep_order=True,
                                         rows=self.rows)
                self._preview = iso_lines(net, bezier_orders(net), count=5, samples=24)
            except Exception:
                # Preview-only: any geometry failure just means no preview.
                pass
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            hover = pick.nearest_polyline(context, self._curves,
                                          (event.mouse_region_x, event.mouse_region_y))
            if hover != self._hover:
                self._hover = hover
                self._update_preview(context)
                context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._hover is None:
                return {'RUNNING_MODAL'}
            if self._hover in self._chosen:
                self._chosen.remove(self._hover)
            else:
                self._chosen.append(self._hover)
            self._header(context)
            self._update_preview(context)
        elif event.type == 'BACK_SPACE' and event.value == 'PRESS':
            if self._chosen:
                self._chosen.pop()
                self._header(context)
                self._update_preview(context)
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if len(self._chosen) < 2:
                self.report({'WARNING'}, "Pick at least two curves")
                return {'RUNNING_MODAL'}
            self.sections = ";".join(k[0] for k in self._chosen)
            self._finish(context)
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    # -------------------------------------------------------------- execute
    def execute(self, context):
        settings = context.scene.spline_surface
        if self.sections:
            names = [n for n in self.sections.split(";") if n]
            sources = [bpy.data.objects.get(n) for n in names]
            missing = [n for n, o in zip(names, sources) if o is None or o.type not in BOUNDARY_TYPES]
            if missing:
                self.report({'ERROR'}, "Not a curve any more: %s" % ", ".join(missing))
                return {'CANCELLED'}
        else:
            sources = [o for o in context.selected_objects if o.type in BOUNDARY_TYPES]
        if len(sources) < 2:
            self.report({'ERROR'}, "Select 2 or more section curves (got %d)" % len(sources))
            return {'CANCELLED'}
        try:
            net, err, ordered = compute_loft(sources, context.evaluated_depsgraph_get(), settings,
                                             keep_order=bool(self.sections), rows=self.rows)
            cu = build_surface_data("LoftPatch", net, settings.resolution)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        ob = bpy.data.objects.new("LoftPatch", cu)
        context.collection.objects.link(ob)
        store_meta(ob, {"method": "loft", "sources": [o.name for o in ordered], "rows": self.rows})
        _make_active(context, ob)
        note = "" if len(sources) <= settings.degree + 1 else \
            " - more sections than Degree+1, middle ones are approximated"
        self.report({'INFO'}, "Loft created (max fit error %.3g)%s" % (err, note))
        return {'FINISHED'}


class SS_OT_snap_ends(bpy.types.Operator):
    bl_idname = "spline_surface.snap_ends"
    bl_label = "Snap Ends"
    bl_description = ("Move the nearest end points of the selected curves together "
                      "(within Gap Tolerance) so they form a closed boundary")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and sum(
            o.type == 'CURVE' for o in context.selected_objects) >= 2

    def execute(self, context):
        curves = [o for o in context.selected_objects if o.type == 'CURVE' and o.data.splines]
        ends = []  # (curve index, point, world position)
        for ci, ob in enumerate(curves):
            for pt in _end_points(ob):
                ends.append((ci, pt, ob.matrix_world @ pt.co.to_3d()))
        cos = np.array([e[2][:] for e in ends])
        diag = np.linalg.norm(cos.max(axis=0) - cos.min(axis=0))
        tol = max(diag * context.scene.spline_surface.gap_tolerance, 1e-6)

        pairs = sorted(
            ((np.linalg.norm(cos[a] - cos[b]), a, b)
             for a in range(len(ends)) for b in range(a + 1, len(ends))
             if ends[a][0] != ends[b][0]),
            key=lambda x: x[0])
        used, snapped = set(), 0
        for d, a, b in pairs:
            if d > tol or a in used or b in used:
                continue
            used.update((a, b))
            mid = (ends[a][2] + ends[b][2]) / 2.0
            for idx in (a, b):
                ci, pt, world = ends[idx]
                inv = curves[ci].matrix_world.to_3x3().inverted()
                _move_point(pt, inv @ (mid - world))
            snapped += 1
        for ob in curves:
            ob.data.update_tag()
        unmatched = len(ends) - 2 * snapped
        level = {'WARNING'} if unmatched else {'INFO'}
        self.report(level, "Snapped %d corner(s), %d end(s) left open" % (snapped, unmatched))
        return {'FINISHED'}


def _arc_points(radius, angle_deg, segments):
    """Bezier (point, handle_left, handle_right) triples tracing a circular
    arc of `angle_deg` degrees and `radius`, centred at the origin in the XY
    plane, starting at angle 0. Handles are exact (not AUTO), so the curve
    is a true circular arc rather than an approximation of one."""
    angle = np.radians(angle_deg)
    theta = angle / segments
    h = radius * 4.0 / 3.0 * np.tan(theta / 4.0)
    pts = []
    for i in range(segments + 1):
        a = angle * i / segments
        co = np.array([radius * np.cos(a), radius * np.sin(a), 0.0])
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        pts.append((co, co - h * tangent, co + h * tangent))
    return pts


class SS_OT_add_boundary_curve(bpy.types.Operator):
    bl_idname = "spline_surface.add_boundary_curve"
    bl_label = "Add Boundary Curve"
    bl_description = ("Add a straight line or circular arc at the 3D cursor, sized for use "
                      "as a patch boundary (Coons, Loft, Curve Network...)")
    bl_options = {'REGISTER', 'UNDO'}

    shape: bpy.props.EnumProperty(
        name="Shape", default='LINE',
        items=[('LINE', "Line", "Straight line"),
               ('ARC', "Arc", "Circular arc; set Angle to 180 for a half circle")])
    length: bpy.props.FloatProperty(name="Length", default=2.0, min=0.001, unit='LENGTH',
                                    description="Line shape only")
    radius: bpy.props.FloatProperty(name="Radius", default=1.0, min=0.001, unit='LENGTH',
                                    description="Arc shape only")
    angle: bpy.props.FloatProperty(name="Angle", default=90.0, min=1.0, max=359.0,
                                   subtype='FACTOR', description="Arc shape only, in degrees")
    segments: bpy.props.IntProperty(name="Segments", default=1, min=1, max=8,
                                    description="Arc shape only: Bezier segments (more for a "
                                                "large angle)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        if self.shape == 'LINE':
            pts = [(np.array([-self.length / 2.0, 0.0, 0.0]), None, None),
                  (np.array([self.length / 2.0, 0.0, 0.0]), None, None)]
            handle_type = 'VECTOR'
        else:
            pts = _arc_points(self.radius, self.angle, self.segments)
            handle_type = 'FREE'
        cu = bpy.data.curves.new("BoundaryCurve", 'CURVE')
        cu.dimensions = '3D'
        sp = cu.splines.new('BEZIER')
        sp.bezier_points.add(len(pts) - 1)
        for bp, (co, hl, hr) in zip(sp.bezier_points, pts):
            bp.co = co
            if handle_type == 'VECTOR':
                bp.handle_left_type = bp.handle_right_type = 'VECTOR'
            else:
                bp.handle_left_type = bp.handle_right_type = 'FREE'
                bp.handle_left = hl
                bp.handle_right = hr
        ob = bpy.data.objects.new(cu.name, cu)
        ob.location = context.scene.cursor.location      # the object's origin sits at the 3D cursor
        context.collection.objects.link(ob)
        _make_active(context, ob)
        return {'FINISHED'}


class SS_OT_add_curve_grid(bpy.types.Operator):
    bl_idname = "spline_surface.add_curve_grid"
    bl_label = "Add Curve Grid"
    bl_description = ("Add a ready-made grid of crossing Bezier curves at the 3D cursor, "
                      "selected and ready for Curve Network. Edit the curves to taste")
    bl_options = {'REGISTER', 'UNDO'}

    count_u: bpy.props.IntProperty(name="Curves Along X", default=3, min=2, max=12)
    count_v: bpy.props.IntProperty(name="Curves Along Y", default=4, min=2, max=12)
    size_x: bpy.props.FloatProperty(name="Size X", default=3.0, min=0.01, unit='LENGTH')
    size_y: bpy.props.FloatProperty(name="Size Y", default=2.0, min=0.01, unit='LENGTH')
    bulge: bpy.props.FloatProperty(name="Bulge", default=0.4, unit='LENGTH',
                                   description="Height of the dome the curves follow")
    overhang: bpy.props.FloatProperty(name="Overhang", default=0.1, min=0.0, max=0.5,
                                      subtype='FACTOR',
                                      description="How far curves run past the outer ones")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        c = np.array(context.scene.cursor.location)
        hx, hy = self.size_x / 2.0, self.size_y / 2.0
        z = lambda x, y: self.bulge * (1 - (x / hx) ** 2) * (1 - (y / hy) ** 2)
        made = []

        def curve(name, pts):
            cu = bpy.data.curves.new(name, 'CURVE')
            cu.dimensions = '3D'
            sp = cu.splines.new('BEZIER')
            sp.bezier_points.add(len(pts) - 1)
            for bp, co in zip(sp.bezier_points, pts):
                bp.co = co - c
                bp.handle_left_type = bp.handle_right_type = 'AUTO'
            ob = bpy.data.objects.new(name, cu)
            ob.location = c                                  # origin at the 3D cursor
            context.collection.objects.link(ob)
            made.append(ob)

        ext = 1.0 + self.overhang
        for y in np.linspace(-hy, hy, self.count_v):     # curves running along X
            xs = np.linspace(-hx * ext, hx * ext, 5)
            curve("GridX", [c + (x, y, z(min(max(x, -hx), hx), y)) for x in xs])
        for x in np.linspace(-hx, hx, self.count_u):     # curves running along Y
            ys = np.linspace(-hy * ext, hy * ext, 5)
            curve("GridY", [c + (x, y, z(x, min(max(y, -hy), hy))) for y in ys])
        for o in context.selected_objects:
            o.select_set(False)
        for ob in made:
            ob.select_set(True)
        context.view_layer.objects.active = made[0]
        return {'FINISHED'}


class SS_OT_curve_network(bpy.types.Operator):
    bl_idname = "spline_surface.curve_network"
    bl_label = "Curve Network"
    bl_description = ("Skin a grid of crossing curves: select curves running in two directions "
                      "(each crossing every curve of the other direction). One patch per "
                      "cell, joined smoothly across the inner curves")
    bl_options = {'REGISTER', 'UNDO'}

    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G2',
        items=[('G0', "G0", "Cells only touch along the curves"),
               ('G1', "G1", "Cells share tangents across the inner curves"),
               ('G2', "G2", "Cells share tangents and curvature across the inner curves")])

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and sum(
            o.type in BOUNDARY_TYPES for o in context.selected_objects) >= 4

    def execute(self, context):
        settings = context.scene.spline_surface
        sources = sorted((o for o in context.selected_objects if o.type in BOUNDARY_TYPES),
                         key=lambda o: o.name)
        meta = {"method": "network", "sources": [o.name for o in sources],
                "continuity": self.continuity}
        try:
            cells, info = compute_network(sources, meta, settings,
                                          context.evaluated_depsgraph_get())
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        made = []
        turns = part_flips([], [(c[2], c[3]) for c in cells], min(settings.gap_tolerance, 0.02))
        for part, ((i, j, net, orders), turned) in enumerate(zip(cells, turns)):
            if turned:
                net = flip_net(net)
            name = "Network_%d_%d" % (i, j)
            cu = build_surface_data(name, net, settings.resolution, orders)
            ob = bpy.data.objects.new(name, cu)
            context.collection.objects.link(ob)
            store_meta(ob, dict(meta, part=part, **({"normal_flip": True} if turned else {})))
            made.append(ob)
        for o in context.selected_objects:
            o.select_set(False)
        for ob in made:
            ob.select_set(True)
        context.view_layer.objects.active = made[-1]
        self.report({'INFO'}, "Curve network %d x %d: %d patches (curve fit %.2g, "
                    "largest crossing gap %.2g)" % (info["size"][0], info["size"][1],
                                                    len(made), info["fit_error"],
                                                    info["max_gap"]))
        return {'FINISHED'}


def primitive_net(shape, size_x, size_y, points_u, points_v, bulge):
    """Control net of a starter patch centred on the origin, z up."""
    u = np.linspace(-1.0, 1.0, points_u)[:, None]
    v = np.linspace(-1.0, 1.0, points_v)[None, :]
    if shape == 'PLANE':
        z = 0.0 * u * v
    elif shape == 'ARC':
        z = bulge * (1.0 - u ** 2) + 0.0 * v
    else:  # DOME
        z = bulge * (1.0 - u ** 2) * (1.0 - v ** 2)
    x = -0.5 * size_x * u + 0.0 * v                  # u runs toward -x: Blender then shows the top (+z) as the front
    y = 0.5 * size_y * v + 0.0 * u
    return np.stack([x, y, z], axis=-1)


class SS_OT_add_patch(bpy.types.Operator):
    bl_idname = "spline_surface.add_patch"
    bl_label = "Add Patch"
    bl_description = "Add a starter NURBS patch at the 3D cursor, ready to Extend and Match"
    bl_options = {'REGISTER', 'UNDO'}

    shape: bpy.props.EnumProperty(
        name="Shape", default='PLANE',
        items=[('PLANE', "Plane", "Flat patch"),
               ('ARC', "Arc", "Curved in one direction, like a roof or door panel"),
               ('DOME', "Dome", "Curved in both directions, like a hood")])
    size_x: bpy.props.FloatProperty(name="Size X", default=2.0, min=0.001, unit='LENGTH')
    size_y: bpy.props.FloatProperty(name="Size Y", default=2.0, min=0.001, unit='LENGTH')
    bulge: bpy.props.FloatProperty(name="Bulge", default=0.3, unit='LENGTH',
                                   description="Height of the curve for Arc and Dome")
    points_u: bpy.props.IntProperty(name="Points U", default=6, min=2, max=32)
    points_v: bpy.props.IntProperty(name="Points V", default=6, min=2, max=32)
    order: bpy.props.IntProperty(name="Order", default=6, min=2, max=6,
                                 description="Order = degree + 1; capped by the point count")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.spline_surface
        net = primitive_net(self.shape, self.size_x, self.size_y,
                            self.points_u, self.points_v, self.bulge)
        net = net + np.array(context.scene.cursor.location)
        orders = (min(self.order, self.points_u), min(self.order, self.points_v))
        name = "SurfPatch"
        cu = build_surface_data(name, net, settings.resolution, orders)
        ob = bpy.data.objects.new(name, cu)
        context.collection.objects.link(ob)
        set_origin(ob, context.scene.cursor.location)      # the origin sits at the 3D cursor
        store_meta(ob, raw_meta(net, orders))
        _make_active(context, ob)
        return {'FINISHED'}


classes = (SS_OT_add_patch, SS_OT_add_boundary_curve, SS_OT_coons, SS_OT_loft, SS_OT_snap_ends, SS_OT_add_curve_grid, SS_OT_curve_network,)
