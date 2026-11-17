"""Fill Hole: pick the edges and curves around a hole and fill it."""

import bpy

from .bridge.write_surface import build_surface_data
from .core.orient import flip_net, part_flips
from . import pick
from . import zebra
from .recipe import (iso_lines, store_meta, surface_edges, world_net)
from .rebuild import compute_fill


class SS_OT_fill_hole(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.fill_hole"
    bl_label = "Fill Hole"
    bl_description = ("Fill a hole bounded by patch edges (and curves). Click the edges "
                      "around it, then Enter. Four sides give one patch, other counts "
                      "one patch per side around a centre")
    bl_options = {'REGISTER', 'UNDO'}

    picks: bpy.props.StringProperty(
        name="Edges", description="object:edge pairs separated by ';'")
    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G2',
        items=[('G0', "G0", "Touch the edges only: a flat lid when the hole's outline is flat"),
               ('G1', "G1", "Match tangents across every edge"),
               ('G2', "G2", "Match tangents and curvature across every edge")])
    tension: bpy.props.FloatProperty(
        name="Tension", default=1.0, min=0.05, max=3.0,
        description="How far the neighbours' tangents reach into the fill")
    layout: bpy.props.EnumProperty(
        name="Layout", default='AUTO',
        items=[('AUTO', "Auto", "Radial when the outline has smooth corners, grid otherwise"),
               ('GRID', "Grid", "One patch for four sides, otherwise one per corner"),
               ('RADIAL', "Radial", "One patch per side meeting at a centre (caps, tube ends)")])

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._edges = surface_edges(context)
        self._edges.update(pick.curve_lines(context))
        if len(self._edges) < 3:
            self.report({'ERROR'}, "Need at least three open edges or curves")
            return {'CANCELLED'}
        self._chosen = []
        self._hover = None
        self._preview = []
        self._kind = 'ALL'
        self._mouse = (event.mouse_region_x, event.mouse_region_y)
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        context.area.header_text_set(
            "Fill Hole: click the edges around the hole (%d picked)   Enter: fill   C: %s   "
            "Wheel: tension %.2f%s   Esc/RMB: cancel"
            % (len(self._chosen), self.continuity, self.tension,
               "   E: %s" % pick.KIND_TEXT[self._kind] if pick.has_both(self._edges) else ""))

    def _rehover(self, context):
        hover = pick.nearest_polyline(context, pick.of_kind(self._edges, self._kind), self._mouse)
        if hover != self._hover:
            self._hover = hover
            self._update_preview(context)
        context.area.tag_redraw()

    def _label(self):
        return pick.edge_label(self._hover, self._edges[self._hover]) if self._hover else None

    def _draw_lines(self):
        lines = [(pts, pick.COLOR_EDGE, 2.0) for k, pts in pick.of_kind(self._edges, self._kind).items()
                 if k != self._hover and k not in self._chosen]
        lines += [(self._edges[k], pick.COLOR_SELECTED, 4.0) for k in self._chosen]
        if self._hover and self._hover not in self._chosen:
            lines.append((self._edges[self._hover], pick.COLOR_HOVER, 4.0))
        lines += [(pts, pick.COLOR_PREVIEW, 1.0) for pts in self._preview]
        return lines

    def _set_picks(self, keys):
        self.picks = ";".join("%s:%s" % k for k in keys)

    def _update_preview(self, context):
        self._preview = []
        keys = self._chosen + ([self._hover] if self._hover and self._hover not in self._chosen
                               else [])
        if len(keys) >= 2:
            self._set_picks(keys)
            try:
                for net, orders in compute_fill(self._meta(), context.scene.spline_surface):
                    self._preview += iso_lines(net, orders, count=4, samples=20)
            except Exception:
                # Preview-only: any geometry failure just means no preview.
                pass
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self._mouse = (event.mouse_region_x, event.mouse_region_y)
            self._rehover(context)
        elif event.type == 'E' and event.value == 'PRESS' and pick.has_both(self._edges):
            self._kind = pick.next_kind(self._kind)
            self._header(context)
            self._rehover(context)
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
            if self._hover in self._chosen:
                self._chosen.remove(self._hover)
            else:
                self._chosen.append(self._hover)
            self._header(context)
            self._update_preview(context)
        elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            if len(self._chosen) < 2:
                self.report({'WARNING'}, "Pick at least two edges")
                return {'RUNNING_MODAL'}
            self._set_picks(self._chosen)
            self._finish(context)
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    def _meta(self):
        pairs = [p.rsplit(":", 1) for p in self.picks.split(";") if p]
        return {"method": "fill", "sources": [p[0] for p in pairs],
                "edges": [p[1] for p in pairs],
                "continuity": self.continuity, "tension": self.tension, "layout": self.layout}

    def execute(self, context):
        settings = context.scene.spline_surface
        meta = self._meta()
        if len(meta["sources"]) < 2:
            self.report({'ERROR'}, "Pick at least two edges")
            return {'CANCELLED'}
        try:
            parts = compute_fill(meta, settings)
        except (ValueError, KeyError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        patch_src = next((bpy.data.objects[n] for n, e in zip(meta["sources"], meta["edges"])
                          if e != 'CURVE'), None)
        made = []
        refs = [world_net(bpy.data.objects[n]) for n, e in zip(meta["sources"], meta["edges"])
                if e != 'CURVE']
        turns = part_flips(refs, parts, min(settings.gap_tolerance, 0.02))
        for i, ((net, orders), turned) in enumerate(zip(parts, turns)):
            if turned:                     # face like the patches around the hole
                net = flip_net(net)
            name = meta["sources"][0] + "_fill"
            cu = build_surface_data(name, net, settings.resolution, orders)
            if patch_src is not None:
                for mat in zebra.real_materials(patch_src):
                    cu.materials.append(mat)
            ob = bpy.data.objects.new(name, cu)
            context.collection.objects.link(ob)
            store_meta(ob, dict(meta, part=i, **({"normal_flip": True} if turned else {})))
            made.append(ob)
        for o in context.selected_objects:
            o.select_set(False)
        for ob in made:
            ob.select_set(True)
        context.view_layer.objects.active = made[-1]
        self.report({'INFO'}, "Filled with %d patch(es)" % len(made))
        return {'FINISHED'}


classes = (SS_OT_fill_hole,)
