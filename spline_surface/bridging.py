"""Corner Fill, Bridge Edges and Sweep: the modal tools that pick two patch edges."""
import bpy

from .bridge.read_curves import read_polyline
from .bridge.write_surface import build_surface_data
from . import pick
from . import zebra
from .recipe import EDGE_KEYS, OPPOSITE_EDGE, _make_active, iso_lines, store_meta, surface_edges
from . import operators
from .operators import compute_blend, compute_corner, compute_sweep
from .orienting import face_like


class _EdgePairPicker(pick.ModalPicker):
    """Modal two-edge picking shared by Corner Fill and Bridge Edges."""

    # Bridge Edges can close a single patch's two opposite edges into a
    # loop; Corner Fill joins two patches meeting at a corner, where that
    # has no meaning, so it keeps the plain "different patch" rule.
    _allow_same_patch = False

    def _candidates(self, context):
        return surface_edges(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._edges = self._candidates(context)
        if len(self._edges) < 2:
            self.report({'ERROR'}, "Need at least two open patch edges")
            return {'CANCELLED'}
        self._first = None
        self._hover = None
        self._preview = []
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        step = "click the first edge" if self._first is None else "click the second edge"
        context.area.header_text_set(
            self._title + ": %s   C: %s   Wheel: tension %.2f   Esc/RMB: cancel"
            % (step, self.continuity, self.tension))

    def _label(self):
        return pick.edge_label(self._hover, self._edges[self._hover]) if self._hover else None

    def _draw_lines(self):
        lines = [(pts, pick.COLOR_EDGE, 2.0) for k, pts in self._edges.items()
                 if k not in (self._hover, self._first)]
        if self._first:
            lines.append((self._edges[self._first], pick.COLOR_SELECTED, 4.0))
        if self._hover and self._hover != self._first:
            lines.append((self._edges[self._hover], pick.COLOR_HOVER, 4.0))
        lines += [(pts, pick.COLOR_PREVIEW, 1.0) for pts in self._preview]
        return lines

    def _set_pair(self, a, b):
        self.source_a, self.edge_a = a
        self.source_b, self.edge_b = b

    def _update_preview(self, context):
        self._preview = []
        pair_ok = self._first and self._hover and (
            self._hover[0] != self._first[0]
            or (self._allow_same_patch and OPPOSITE_EDGE.get(self._first[1]) == self._hover[1]))
        if pair_ok:
            self._set_pair(self._first, self._hover)
            try:
                net, orders = self._compute(
                    [bpy.data.objects[self.source_a], bpy.data.objects[self.source_b]],
                    self._meta(), context.scene.spline_surface)
                self._preview = iso_lines(net, orders)
            except Exception:
                # Preview-only: any geometry failure just means no preview.
                pass
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            hover = pick.nearest_polyline(context, self._edges,
                                          (event.mouse_region_x, event.mouse_region_y))
            if hover != self._hover:
                self._hover = hover
                self._update_preview(context)
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
            f = 1.1 if event.type == 'WHEELUPMOUSE' else 1 / 1.1
            self.tension = min(3.0, max(0.05, self.tension * f))
            self._header(context)
            self._update_preview(context)
        elif event.type == 'C' and event.value == 'PRESS':
            order = ['G0', 'G1', 'G2']
            self.continuity = order[(order.index(self.continuity) + 1) % 3]
            self._header(context)
            self._update_preview(context)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._hover is None:
                return {'RUNNING_MODAL'}
            if self._first is None or self._hover == self._first:
                self._first = None if self._hover == self._first else self._hover
                self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if self._hover[0] == self._first[0]:
                if not self._allow_same_patch:
                    self.report({'WARNING'}, "Pick an edge of a different patch")
                    return {'RUNNING_MODAL'}
                if OPPOSITE_EDGE.get(self._first[1]) != self._hover[1]:
                    self.report({'WARNING'},
                                "On the same patch, pick its opposite edge (U0/U1 or V0/V1) "
                                "to close it into a loop")
                    return {'RUNNING_MODAL'}
            self._set_pair(self._first, self._hover)
            if self._more_picks(context):
                return {'RUNNING_MODAL'}
            self._finish(context)
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    def _more_picks(self, context):
        """True if the picked pair still needs another click before execute (Sweep's target)."""
        return False

    def _meta(self):
        return {"method": self._method, "sources": [self.source_a, self.source_b],
                "edges": [self.edge_a, self.edge_b],
                "continuity": self.continuity, "tension": self.tension}

    def execute(self, context):
        settings = context.scene.spline_surface
        meta = self._meta()
        try:
            sources = [bpy.data.objects[n] for n in meta["sources"]]
        except KeyError:
            self.report({'ERROR'}, "Pick two patch edges")
            return {'CANCELLED'}
        try:
            net, orders = self._compute(sources, meta, settings)
            name = sources[0].name + self._suffix
            cu = build_surface_data(name, net, settings.resolution, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        for mat in zebra.real_materials(sources[0]):
            cu.materials.append(mat)
        ob = bpy.data.objects.new(name, cu)
        context.collection.objects.link(ob)
        store_meta(ob, meta)
        face_like(ob, sources[0], settings)
        _make_active(context, ob)
        return {'FINISHED'}


class SS_OT_corner_fill(_EdgePairPicker, bpy.types.Operator):
    bl_idname = "spline_surface.corner_fill"
    bl_label = "Corner Fill"
    bl_description = ("Fill the notch between two patch edges that meet at a corner. "
                      "Click the two edges in the viewport")
    bl_options = {'REGISTER', 'UNDO'}

    source_a: bpy.props.StringProperty(name="Patch A")
    edge_a: bpy.props.EnumProperty(name="Edge A", items=[(e, e, "") for e in EDGE_KEYS])
    source_b: bpy.props.StringProperty(name="Patch B")
    edge_b: bpy.props.EnumProperty(name="Edge B", items=[(e, e, "") for e in EDGE_KEYS])
    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G2',
        items=[('G0', "G0", "Touch the edges only"),
               ('G1', "G1", "Match tangents across both edges"),
               ('G2', "G2", "Match tangents and curvature across both edges")])
    tension: bpy.props.FloatProperty(
        name="Tension", default=1.0, min=0.05, max=3.0,
        description="How far the neighbours' tangents reach into the fill")

    _method, _suffix, _title = "corner", "_corner", "Corner Fill"

    def _compute(self, sources, meta, settings):
        return compute_corner(sources, meta, settings)


class SS_OT_bridge_edges(_EdgePairPicker, bpy.types.Operator):
    bl_idname = "spline_surface.bridge_edges"
    bl_label = "Bridge Edges"
    bl_description = ("Bridge the gap between two separate patch edges, or from a patch edge "
                      "to a curve, with a smooth blend. Click the two in the viewport")
    bl_options = {'REGISTER', 'UNDO'}

    source_a: bpy.props.StringProperty(name="Patch A")
    edge_a: bpy.props.EnumProperty(name="Edge A", items=[(e, e, "") for e in EDGE_KEYS]
                                   + [('CURVE', "Curve", "Patch A is a curve object")])
    source_b: bpy.props.StringProperty(name="Patch B")
    edge_b: bpy.props.EnumProperty(name="Edge B", items=[(e, e, "") for e in EDGE_KEYS]
                                   + [('CURVE', "Curve", "Patch B is a curve object")])
    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G2',
        items=[('G0', "G0", "Straight ruled bridge"),
               ('G1', "G1", "Match tangents at both edges"),
               ('G2', "G2", "Match tangents and curvature at both edges")])
    tension: bpy.props.FloatProperty(
        name="Tension", default=1.0, min=0.05, max=3.0,
        description="How far each side's tangent reaches into the bridge")
    flip: bpy.props.BoolProperty(
        name="Flip", default=False,
        description="Pair the edge ends the other way (twists the bridge)")
    curve_tangent: bpy.props.EnumProperty(
        name="At Curve", default='SMOOTH',
        items=[('SMOOTH', "Smooth", "Arrive at the curve naturally"),
               ('STRAIGHT', "Straight", "Leave the curve heading straight for the patch edge")],
        description="How the bridge meets a curve (curves only give position, G0)")

    _method, _suffix, _title = "blend", "_blend", "Bridge Edges (F: flip)"
    _allow_same_patch = True

    def _compute(self, sources, meta, settings):
        net, orders, self._curve_error = compute_blend(sources, meta, settings)
        return net, orders

    def _candidates(self, context):
        lines = surface_edges(context)
        dg = context.evaluated_depsgraph_get()
        for ob in context.visible_objects:
            if ob.type == 'CURVE' and ob.data.splines:
                try:
                    lines[(ob.name, 'CURVE')] = read_polyline(ob, dg, 48)
                except ValueError:
                    pass
        return lines

    def _set_pair(self, a, b):
        if a[1] == 'CURVE':  # the patch always comes first
            a, b = b, a
        super()._set_pair(a, b)

    def _meta(self):
        meta = super()._meta()
        meta["flip"] = self.flip
        meta["curve_tangent"] = self.curve_tangent
        return meta

    def execute(self, context):
        if self.edge_a == 'CURVE':
            self.report({'ERROR'}, "Pick at least one patch edge (two curves make a Loft)")
            return {'CANCELLED'}
        self._curve_error = None
        result = super().execute(context)
        if result == {'FINISHED'} and self._curve_error is not None:
            self.report({'INFO'}, "Bridged to curve (max deviation %.3g)" % self._curve_error)
        return result

    def modal(self, context, event):
        if event.type == 'F' and event.value == 'PRESS':
            self.flip = not self.flip
            self._update_preview(context)
            return {'RUNNING_MODAL'}
        return super().modal(context, event)


class SS_OT_sweep(_EdgePairPicker, bpy.types.Operator):
    bl_idname = "spline_surface.sweep"
    bl_label = "Sweep"
    bl_description = ("Carry the shape of one patch edge (the profile) along another patch edge "
                      "(the rail). Click the profile edge, then the rail edge. "
                      "F in the viewport: start from the rail's other end; S: let the profile's other end "
                      "run straight out of its patch instead of following the rail")
    bl_options = {'REGISTER', 'UNDO'}

    source_a: bpy.props.StringProperty(name="Profile Patch")
    edge_a: bpy.props.EnumProperty(name="Profile Edge", items=[(e, e, "") for e in EDGE_KEYS])
    source_b: bpy.props.StringProperty(name="Rail Patch")
    edge_b: bpy.props.EnumProperty(name="Rail Edge", items=[(e, e, "") for e in EDGE_KEYS])
    flip: bpy.props.BoolProperty(
        name="Flip", default=False,
        description="Sweep from the rail's far end instead of the end nearest the profile")

    straight: bpy.props.BoolProperty(
        name="Straight", default=False,
        description="The profile's other end does not follow the rail but runs straight out of "
                    "its patch (the profile is stretched between the two)")
    length: bpy.props.FloatProperty(
        name="Length", default=0.0, min=0.0, max=1000.0, unit='LENGTH',
        description="How far the straight end runs (0: as far as the rail's end has gone in that direction, so the ends line up)")
    target_mode: bpy.props.EnumProperty(
        name="Target", default='NONE',
        items=[('NONE', "Length", "Auto or typed length"),
               ('EDGE', "Edge", "Stop where the straight end comes closest to a patch edge or curve"),
               ('PLANE', "Plane", "Stop where the straight end reaches a plane square to an axis")],
        description="Where the straight end stops (needs Straight)")
    target_source: bpy.props.StringProperty(name="Target Object")
    target_edge: bpy.props.EnumProperty(name="Target Edge", items=[(e, e, "") for e in EDGE_KEYS]
                                        + [('CURVE', "Curve", "The target is a curve object")])
    target_axis: bpy.props.EnumProperty(
        name="Plane Axis", default='2',
        items=[('0', "X", "The plane YZ at this X"), ('1', "Y", "The plane XZ at this Y"),
               ('2', "Z", "The plane XY at this Z")])
    target_value: bpy.props.FloatProperty(name="Plane At", default=0.0, unit='LENGTH',
                                          description="Coordinate of the plane along its axis")

    _method, _suffix, _title = "sweep", "_sweep", "Sweep"
    _target_stage = False

    def _compute(self, sources, meta, settings):
        return compute_sweep(sources, meta)

    def _header(self, context):
        if self._target_stage:
            context.area.header_text_set("Sweep: click the target edge or curve   T: change target   Esc/RMB: cancel")
            return
        step = "click the profile edge" if self._first is None else "click the rail edge"
        tgt = {'NONE': "length", 'EDGE': "edge",
               'PLANE': "plane %s=%.3g" % ("XYZ"[int(self.target_axis)], self.target_value)}[self.target_mode]
        context.area.header_text_set(
            "Sweep: %s   F: flip   S: straight %s   T: target (%s)%s   Esc/RMB: cancel"
            % (step, "on" if self.straight else "off", tgt,
               "   X/Y/Z: plane axis" if self.target_mode == 'PLANE' else ""))

    def _meta(self):
        meta = {"method": "sweep", "sources": [self.source_a, self.source_b],
                "edges": [self.edge_a, self.edge_b], "flip": self.flip,
                "straight": self.straight, "length": self.length}
        if self.target_mode == 'EDGE' and self.target_source:
            meta["target"] = {"kind": "edge", "source": self.target_source, "edge": self.target_edge}
        elif self.target_mode == 'PLANE':
            meta["target"] = {"kind": "plane", "axis": int(self.target_axis), "value": self.target_value}
        return meta

    def _plane_from_cursor(self, context):
        self.target_value = float(context.scene.cursor.location[int(self.target_axis)])

    def _more_picks(self, context):
        if self.target_mode != 'EDGE' or self.target_source:
            return False
        self._target_stage = True
        self._pair_edges = self._edges
        self._edges = SS_OT_bridge_edges._candidates(self, context)      # curves can be targets too
        self._first, self._hover, self._preview = None, None, []
        self._header(context)
        context.area.tag_redraw()
        return True

    def _update_preview(self, context):
        if not self._target_stage:
            return super()._update_preview(context)
        self._preview = []
        if self._hover:
            self.target_source, self.target_edge = self._hover
            try:
                net, orders = compute_sweep([bpy.data.objects[self.source_a],
                                             bpy.data.objects[self.source_b]], self._meta())
                self._preview = iso_lines(net, orders)
            except Exception:
                pass
        context.area.tag_redraw()

    def _cycle_target(self, context):
        order = ['NONE', 'EDGE', 'PLANE']
        self.target_mode = order[(order.index(self.target_mode) + 1) % 3]
        if self.target_mode != 'NONE':
            self.straight = True
        if self.target_mode == 'PLANE':
            self._plane_from_cursor(context)
        self.target_source = ""
        self._header(context)
        self._update_preview(context)

    def modal(self, context, event):
        if self._target_stage:
            if event.type == 'MOUSEMOVE':
                hover = pick.nearest_polyline(context, self._edges,
                                              (event.mouse_region_x, event.mouse_region_y))
                if hover != self._hover:
                    self._hover = hover
                    self._update_preview(context)
            elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self._hover:
                self.target_source, self.target_edge = self._hover
                self._finish(context)
                return self.execute(context)
            elif event.type == 'T' and event.value == 'PRESS':
                self._target_stage = False
                self._edges = self._pair_edges
                self._first = None
                self._cycle_target(context)
            elif self._common_modal(context, event):
                return self._common_result
            return {'RUNNING_MODAL'}
        if event.type == 'F' and event.value == 'PRESS':
            self.flip = not self.flip
            self._update_preview(context)
            return {'RUNNING_MODAL'}
        if event.type == 'S' and event.value == 'PRESS':
            self.straight = not self.straight
            self._header(context)
            self._update_preview(context)
            return {'RUNNING_MODAL'}
        if event.type == 'T' and event.value == 'PRESS':
            self._cycle_target(context)
            return {'RUNNING_MODAL'}
        if event.type in {'X', 'Y', 'Z'} and event.value == 'PRESS' and self.target_mode == 'PLANE':
            self.target_axis = str("XYZ".index(event.type))
            self._plane_from_cursor(context)
            self._header(context)
            self._update_preview(context)
            return {'RUNNING_MODAL'}
        if event.type in {'C', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        return super().modal(context, event)

    def execute(self, context):
        if self.target_mode == 'EDGE' and not self.target_source:
            self.report({'ERROR'}, "Pick a target edge or curve")
            return {'CANCELLED'}
        if self.target_mode != 'NONE' and not self.straight:
            self.straight = True
        result = super().execute(context)
        if result == {'FINISHED'} and operators.sweep_note:
            self.report({'WARNING'}, operators.sweep_note)
        return result


classes = (SS_OT_corner_fill, SS_OT_bridge_edges, SS_OT_sweep)
