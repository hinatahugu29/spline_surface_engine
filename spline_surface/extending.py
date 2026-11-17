"""Extend Edge (with its reach-a-second-edge length solver) and Offset Surface."""

import bpy
import numpy as np

from .bridge.write_surface import build_surface_data
from .core.offset import offset_net
from .core.extend import extend_net
from . import pick
from . import zebra
from .recipe import (EDGE_KEYS, OPPOSITE_EDGE, _make_active, edge_polyline,
                     iso_lines, store_meta, surface_edges, world_net)
from .matching import _selected_surfaces
from .rebuild import compute_extend


def _extend_far_edge(net0, orders0, edge, new_net, new_orders):
    """Which edge key of the extension `new_net` is the new, far (tip) edge.

    Matched by centroid distance, not point-by-point distance: the new
    patch's own parametrization can sample its shared edge in the opposite
    order from `edge_polyline(net0, ...)`, which would make a point-by-point
    comparison see a large (wrong) distance for the true shared edge on a
    patch whose far edge happens to be geometrically no further away (e.g. a
    short, flat, symmetric extension).
    """
    src_centre = edge_polyline(net0, orders0, edge, 16).mean(axis=0)
    shared = min(EDGE_KEYS, key=lambda k: np.linalg.norm(
        edge_polyline(new_net, new_orders, k, 16).mean(axis=0) - src_centre))
    return OPPOSITE_EDGE[shared]


def _solve_extend_length(net0, orders0, edge, continuity, degree, orders, target_pts,
                         lo=0.01, hi=5.0):
    """Length in [lo, hi] whose extension's far edge comes closest to `target_pts`.

    Length is the extension's only free parameter -- its far edge's own shape
    is fixed by the source's own tangent/curvature, it can't bend to follow a
    curved target. Scored by the far edge's *average* gap to the target
    (sampled along its length, each to its own nearest target point) rather
    than just the gap at its centre, so a curved or offset target still picks
    the length that fits it best overall instead of matching only the middle
    and leaving the ends off.
    """
    def far_dist(length):
        net, out_orders = extend_net(net0, edge, length, continuity, degree, orders)
        far = _extend_far_edge(net0, orders0, edge, net, out_orders)
        far_pts = edge_polyline(net, out_orders, far, 16)
        # For each point along the far edge, distance to its own nearest
        # target point (not one shared nearest point for all of them).
        gaps = np.linalg.norm(far_pts[:, None, :] - target_pts[None, :, :], axis=2).min(axis=1)
        return float(gaps.mean())
    xs = np.linspace(lo, hi, 16)
    ds = [far_dist(x) for x in xs]
    i = int(np.argmin(ds))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    for _ in range(16):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        a, b = (a, m2) if far_dist(m1) < far_dist(m2) else (m1, b)
    return float(np.clip((a + b) / 2, lo, hi))


class SS_OT_extend_edge(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.extend_edge"
    bl_label = "Extend Edge"
    bl_description = ("Grow a new patch off an open edge of any visible patch (click it), "
                      "matching its tangent (G1) or tangent and curvature (G2). Then click a "
                      "second edge to stretch the extension until it reaches it")
    bl_options = {'REGISTER', 'UNDO'}

    source: bpy.props.StringProperty(
        name="Patch", description="Patch to extend (empty: the active object)")
    edge: bpy.props.EnumProperty(
        name="Edge",
        items=[('V1', "V Max", "Edge at v = 1"), ('V0', "V Min", "Edge at v = 0"),
               ('U1', "U Max", "Edge at u = 1"), ('U0', "U Min", "Edge at u = 0")])
    length: bpy.props.FloatProperty(
        name="Length", default=0.5, min=0.01, max=5.0, step=5,
        description="Length of the extension relative to the source patch")
    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G2',
        items=[('G1', "G1", "Match tangents (straight continuation)"),
               ('G2', "G2", "Match tangents and curvature")])

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.visible_objects)

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._edges = surface_edges(context)
        if not self._edges:
            self.report({'ERROR'}, "No open patch edges to extend")
            return {'CANCELLED'}
        self._nets = {}
        self._hover = None
        self._source = None
        self._preview = []
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        if self._source is None:
            step = "click an edge"
        else:
            step = "click a target edge to reach it, or Enter/click again to confirm"
        context.area.header_text_set(
            "Extend Edge: %s   Wheel: length %.2f   C: %s   Esc/RMB: cancel"
            % (step, self.length, self.continuity))

    def _label(self):
        return pick.edge_label(self._hover, self._edges[self._hover]) if self._hover else None

    def _draw_lines(self):
        lines = [(pts, pick.COLOR_EDGE, 2.0) for e, pts in self._edges.items()
                 if e not in (self._hover, self._source)]
        if self._source:
            lines.append((self._edges[self._source], pick.COLOR_SELECTED, 4.0))
        if self._hover and self._hover != self._source:
            lines.append((self._edges[self._hover], pick.COLOR_HOVER, 4.0))
        lines += [(pts, pick.COLOR_PREVIEW, 1.0) for pts in self._preview]
        return lines

    def _net(self, name):
        if name not in self._nets:
            self._nets[name] = world_net(bpy.data.objects[name])
        return self._nets[name]

    def _update_preview(self, context):
        self._preview = []
        source = self._source or self._hover
        if not source:
            context.area.tag_redraw()
            return
        name, edge = source
        try:
            net0, orders0 = self._net(name)
            length = self.length
            target = self._hover if (self._source and self._hover
                                     and self._hover != self._source) else None
            if target:
                length = _solve_extend_length(net0, orders0, edge, self.continuity,
                                              context.scene.spline_surface.degree, orders0,
                                              self._edges[target])
            net, orders = extend_net(net0, edge, length, self.continuity,
                                     context.scene.spline_surface.degree, orders0)
            self._preview = iso_lines(net, orders)
        except Exception:
            # Preview-only: any geometry failure just means no preview.
            pass
        context.area.tag_redraw()

    def _solve_to_target(self, context, target):
        name, edge = self._source
        try:
            net0, orders0 = self._net(name)
            self.length = _solve_extend_length(
                net0, orders0, edge, self.continuity, context.scene.spline_surface.degree,
                orders0, self._edges[target])
        except Exception:
            pass  # keep whatever length was last set manually

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            hover = pick.nearest_polyline(context, self._edges,
                                          (event.mouse_region_x, event.mouse_region_y))
            if hover != self._hover:
                self._hover = hover
                self._update_preview(context)
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
            f = 1.1 if event.type == 'WHEELUPMOUSE' else 1 / 1.1
            self.length = min(5.0, max(0.01, self.length * f))
            self._header(context)
            self._update_preview(context)
        elif event.type == 'C' and event.value == 'PRESS':
            self.continuity = 'G1' if self.continuity == 'G2' else 'G2'
            self._header(context)
            self._update_preview(context)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._hover is None:
                return {'RUNNING_MODAL'}
            if self._source is None:
                self._source = self._hover
                self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if self._hover != self._source:
                self._solve_to_target(context, self._hover)
            self._finish(context)
            self.source, self.edge = self._source
            return self.execute(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS' and self._source:
            self._finish(context)
            self.source, self.edge = self._source
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    def execute(self, context):
        source = bpy.data.objects.get(self.source) if self.source else context.active_object
        if source is None or source.type != 'SURFACE':
            self.report({'ERROR'}, "Pick a patch edge to extend")
            return {'CANCELLED'}
        settings = context.scene.spline_surface
        meta = {"method": "extend", "sources": [source.name], "edge": self.edge,
                "length": self.length, "continuity": self.continuity}
        try:
            net, orders = compute_extend(source, meta, settings)
            cu = build_surface_data(source.name + "_ext", net, settings.resolution, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        for mat in zebra.real_materials(source):
            cu.materials.append(mat)
        ob = bpy.data.objects.new(source.name + "_ext", cu)
        context.collection.objects.link(ob)
        store_meta(ob, meta)
        _make_active(context, ob)
        return {'FINISHED'}


class SS_OT_offset_surface(bpy.types.Operator):
    bl_idname = "spline_surface.offset_surface"
    bl_label = "Offset Surface"
    bl_description = ("Create a copy of each selected patch at a constant distance along its "
                      "normal (panel thickness, inner skins). The copies follow the source")
    bl_options = {'REGISTER', 'UNDO'}

    distance: bpy.props.FloatProperty(
        name="Distance", default=0.05, unit='LENGTH', step=1,
        description="Offset along the patch normal; negative for the other side")
    refine: bpy.props.IntProperty(
        name="Refine", default=2, min=1, max=6,
        description="Spans per source span in the fitted offset (higher = closer, heavier)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(_selected_surfaces(context))

    def execute(self, context):
        settings = context.scene.spline_surface
        made, worst = [], 0.0
        for src in _selected_surfaces(context):
            meta = {"method": "offset", "sources": [src.name], "distance": self.distance,
                    "refine": self.refine}
            try:
                net, orders = world_net(src)
                net, orders, err = offset_net(net, orders, self.distance, self.refine)
                cu = build_surface_data(src.name + "_offset", net, settings.resolution, orders)
            except ValueError as e:
                self.report({'WARNING'}, "%s: %s" % (src.name, e))
                continue
            worst = max(worst, err)
            for mat in zebra.real_materials(src):
                cu.materials.append(mat)
            ob = bpy.data.objects.new(cu.name, cu)
            (src.users_collection[0] if src.users_collection
             else context.collection).objects.link(ob)
            store_meta(ob, meta)
            made.append(ob)
        if not made:
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for ob in made:
            ob.select_set(True)
        context.view_layer.objects.active = made[-1]
        self.report({'INFO'}, "Offset %d patch(es), max deviation %.2g" % (len(made), worst))
        return {'FINISHED'}


classes = (SS_OT_extend_edge, SS_OT_offset_surface,)
