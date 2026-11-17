import json

import bpy
import numpy as np

from .bridge.write_surface import build_surface_data
from .core.orient import flip_net, part_flips
from .core.offset import offset_net
from .core.continuity import find_shared_edge, grade, measure
from .core.extend import bezier_orders, extend_net
from .core.loft import order_sections
from . import pick
from . import zebra
from .recipe import (EDGE_KEYS, META_KEY, OPPOSITE_EDGE, _make_active,
                     edge_polyline, hand_edited, hand_edited_net, iso_lines, raw_meta,
                     replace_net, set_origin, store_meta, surface_edges, world_net)
from .rebuild import (BOUNDARY_TYPES, _seam_tolerance, apply_match, compute_coons,
                      compute_extend, compute_fill, compute_from_meta, compute_loft,
                      compute_network)


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


class SS_OT_regenerate(bpy.types.Operator):
    bl_idname = "spline_surface.regenerate"
    bl_label = "Regenerate"
    bl_description = "Rebuild the active patch from its source curves with the current settings"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return context.mode == 'OBJECT' and ob is not None and META_KEY in ob

    def execute(self, context):
        ob = context.active_object
        meta = json.loads(ob[META_KEY])
        edited = hand_edited(ob, meta)
        try:
            net, orders, msg = compute_from_meta(meta, context)
            replace_net(ob, net, context.scene.spline_surface, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        store_meta(ob, meta)
        if edited:
            self.report({'WARNING'}, "Regenerated (%s) - hand edits to this patch were "
                        "discarded, Ctrl+Z to get them back" % msg)
        else:
            self.report({'INFO'}, "Regenerated (%s)" % msg)
        return {'FINISHED'}


def _selected_surfaces(context):
    return [o for o in context.selected_objects if o.type == 'SURFACE']


class SS_OT_match_edge(bpy.types.Operator):
    bl_idname = "spline_surface.match_edge"
    bl_label = "Match Edge"
    bl_description = ("Make the active patch meet the other selected patch along their "
                      "closest edges with G0/G1/G2 continuity (the other patch is not changed)")
    bl_options = {'REGISTER', 'UNDO'}

    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G2',
        items=[('G0', "G0", "Close the gap (position)"),
               ('G1', "G1", "Match tangents"),
               ('G2', "G2", "Match tangents and curvature")])
    spread: bpy.props.BoolProperty(
        name="Spread", default=True,
        description="Ease the seam's move into the whole patch (the opposite edge stays put) "
                    "instead of only the 1-3 rows next to the seam. Keeps the patch smooth "
                    "when the other patch is later moved far")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (context.mode == 'OBJECT' and ob is not None and ob.type == 'SURFACE'
                and len(_selected_surfaces(context)) == 2)

    def execute(self, context):
        target = context.active_object
        ref = next(o for o in _selected_surfaces(context) if o is not target)
        settings = context.scene.spline_surface
        meta = json.loads(target[META_KEY]) if META_KEY in target else None
        entry = {"ref": ref.name, "continuity": self.continuity, "spread": self.spread}
        try:
            tnet, torders = world_net(target)
            net, orders = apply_match(tnet, torders, *world_net(ref), self.continuity, settings,
                                      entry)
            replace_net(target, net, settings, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if meta is None or hand_edited_net(meta, tnet, torders):
            # Keep the hand-made shape as the recipe so Regenerate does not undo it.
            old_matches = meta.get("matches", []) if meta else []
            meta = raw_meta(tnet, torders, matches=old_matches)
        matches = [m for m in meta.get("matches", []) if m["ref"] != ref.name]
        matches.append(entry)
        meta["matches"] = matches
        store_meta(target, meta)
        self.report({'INFO'}, "%s matched to %s (%s)" % (target.name, ref.name, self.continuity))
        return {'FINISHED'}


_GRADE_ORDER = {'open': -1, 'G0': 0, 'G1': 1, 'G2': 2}


class SS_OT_seam_jump(bpy.types.Operator):
    bl_idname = "spline_surface.seam_jump"
    bl_label = "Go to Seam"
    bl_description = "Select the two patches of this seam and bring them into view"
    bl_options = {'REGISTER', 'UNDO'}

    a: bpy.props.StringProperty()
    b: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        obs = [bpy.data.objects.get(n) for n in (self.a, self.b)]
        if not all(obs):
            self.report({'ERROR'}, "A patch of this seam no longer exists; run Check Continuity again")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in obs:
            o.select_set(True)
        context.view_layer.objects.active = obs[0]
        if context.area and context.area.type == 'VIEW_3D':
            bpy.ops.view3d.view_selected()
        return {'FINISHED'}


class SS_OT_check_continuity(bpy.types.Operator):
    bl_idname = "spline_surface.check_continuity"
    bl_label = "Check Continuity"
    bl_description = ("Measure gap, normal angle and curvature jump on seams between the selected "
                      "patches (all visible patches when fewer than two are selected). The worst "
                      "seams come first; the arrow button next to one jumps to it")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and len([o for o in context.visible_objects
                                                 if o.type == 'SURFACE']) >= 2

    def execute(self, context):
        settings = context.scene.spline_surface
        nets = {}
        pool = _selected_surfaces(context)
        if len(pool) < 2:
            pool = [o for o in context.visible_objects if o.type == 'SURFACE']
        for o in pool:
            try:
                nets[o.name] = world_net(o)
            except ValueError as e:
                self.report({'WARNING'}, str(e))
        names = sorted(nets)
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                (Pa, oa), (Pb, ob_) = nets[a], nets[b]
                shared = find_shared_edge(Pa, Pb, _seam_tolerance(Pa, Pb, settings), oa, ob_)
                if shared is None:
                    continue
                m = measure(Pa, Pb, *shared[:3], oa, ob_)
                rows.append({"a": a, "b": b, "edge_a": shared[0], "edge_b": shared[1],
                            "grade": grade(m), **m})
        rows.sort(key=lambda r: (_GRADE_ORDER.get(r["grade"], 0), -r["angle"], -r["curvature"]))
        settings.continuity_report = json.dumps(rows)
        if not rows:
            self.report({'WARNING'}, "No shared edges found between the selected patches")
        for r in rows:
            self.report({'INFO'},
                        "%s(%s) | %s(%s): %s (gap %.2g, angle %.3g deg, curvature %.1f%%)" % (
                            r["a"], r["edge_a"], r["b"], r["edge_b"], r["grade"],
                            r["gap"], r["angle"], 100 * r["curvature"]))
        return {'FINISHED'}


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


classes = (SS_OT_add_patch, SS_OT_add_boundary_curve, SS_OT_coons, SS_OT_loft, SS_OT_snap_ends,
           SS_OT_regenerate, SS_OT_extend_edge, SS_OT_fill_hole, SS_OT_offset_surface, SS_OT_add_curve_grid,
           SS_OT_curve_network,
           SS_OT_match_edge, SS_OT_seam_jump, SS_OT_check_continuity)
