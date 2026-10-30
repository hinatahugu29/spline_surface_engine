"""Fillet: round the creases at seams where two patch edges already coincide.

Pick one or more seams, set the radius; the patches are trimmed back, the
cut edges bridged, and the holes where fillets meet filled with corner
patches. The geometry lives in core.fillet.
"""
import json

import bpy
import numpy as np

from . import pick, zebra
from .bridge.write_surface import build_surface_data
from .core.continuity import find_shared_edge
from .core.fill import fill_patches, side_from_patch
from .core.fillet import corner_loops, multi_fillet
from .operators import _seam_tolerance, compute_fill
from .recipe import (META_KEY, edge_polyline, iso_lines, raw_meta, replace_net, store_meta,
                     world_net)


def compute_multi_fillet(seams, radius, continuity, settings):
    """multi_fillet on the objects named in `seams` [(a, edge_a, b, edge_b)]."""
    names = {n for a, _, b, _ in seams for n in (a, b)}
    missing = sorted(n for n in names if n not in bpy.data.objects)
    if missing:
        raise ValueError("Patches missing: %s" % ", ".join(missing))
    patches = {n: world_net(bpy.data.objects[n]) for n in names}
    # Each seam was found at its own pair's tolerance; accept the loosest.
    tol = max(_seam_tolerance(patches[a][0], patches[b][0], settings) for a, _, b, _ in seams)
    return multi_fillet(patches, seams, radius, continuity, tol)


def find_fillet_seams(context, settings):
    """{(obj_a name, obj_b name): (edge_a, edge_b, polyline)} for every pair of
    visible patches that share a coincident edge (the only valid Fillet input).

    Uses the same curve-sampling match as Check Continuity, at the same
    tolerance, so a seam offered here is one Check Continuity would call G0.
    """
    nets = {}
    for ob in context.visible_objects:
        if ob.type != 'SURFACE':
            continue
        try:
            nets[ob.name] = world_net(ob)
        except ValueError:
            continue
    names = sorted(nets)
    seams = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (Pa, oa), (Pb, ob_) = nets[a], nets[b]
            shared = find_shared_edge(Pa, Pb, _seam_tolerance(Pa, Pb, settings), oa, ob_)
            if shared is None:
                continue
            ea, eb, _, _ = shared
            seams[(a, b)] = (ea, eb, edge_polyline(Pa, oa, ea, 24))
    return seams


class SS_OT_fillet(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.fillet"
    bl_label = "Fillet"
    bl_description = ("Round the creases at seams where two patch edges already coincide "
                      "(e.g. after Match Edge): the patches are trimmed back and bridged. "
                      "Click one or more seams, Enter, then set the radius")
    bl_options = {'REGISTER', 'UNDO'}

    seams: bpy.props.StringProperty(
        name="Seams", description="Picked seams as JSON [[patch a, edge a, patch b, edge b], ...]")
    radius: bpy.props.FloatProperty(
        name="Radius", default=0.1, min=1e-4, max=1000.0,
        description="Rolling-ball radius: sets how far both patches are trimmed back")
    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G2',
        items=[('G1', "G1", "Tangent match at the trimmed edges; closest to a true arc"),
               ('G2', "G2", "Tangent and curvature match at the trimmed edges")])
    corners: bpy.props.BoolProperty(
        name="Fill Corners", default=True,
        description="Close the holes where fillets meet (e.g. a box corner) with corner patches")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        settings = context.scene.spline_surface
        self._seams = find_fillet_seams(context, settings)
        if not self._seams:
            self.report({'ERROR'}, "No coincident edges between visible patches "
                                   "(use Match Edge first, or raise Gap Tolerance)")
            return {'CANCELLED'}
        self._hover = None
        self._picked = []     # (a, b) keys of self._seams, in click order
        self._adjust = False  # picking seams, then setting the radius
        self._preview = []
        self.radius = 0.1
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        if not self._adjust:
            context.area.header_text_set(
                "Fillet: click seams to round (%d picked, click again to drop)   "
                "Enter/Space: set radius   Esc/RMB: cancel" % len(self._picked))
        else:
            context.area.header_text_set(
                "Fillet: %d seam(s)   Wheel: radius %.4g   C: %s   V: corner patches %s   "
                "Enter/click: confirm   Backspace: back to picking   Esc/RMB: cancel"
                % (len(self._picked), self.radius, self.continuity,
                   "on" if self.corners else "off"))

    def _label(self):
        if not self._adjust and self._hover is not None:
            a, b = self._hover
            ea, eb, pts = self._seams[self._hover]
            return "%s(%s) | %s(%s)" % (a, ea, b, eb), pts[len(pts) // 2]
        return None

    def _draw_lines(self):
        lines = [(v[2], pick.COLOR_EDGE, 2.0) for k, v in self._seams.items()
                 if k != self._hover and k not in self._picked]
        lines += [(self._seams[k][2], pick.COLOR_SELECTED, 4.0) for k in self._picked]
        if self._hover is not None and not self._adjust:
            lines.append((self._seams[self._hover][2], pick.COLOR_HOVER, 4.0))
        lines += [(pts, pick.COLOR_PREVIEW, 1.0) for pts in self._preview]
        return lines

    def _seam_list(self):
        return [list(s) for s in json.loads(self.seams or "[]")]

    def _update_preview(self, context):
        self._preview = []
        if self._adjust:
            try:
                seams = self._seam_list()
                trimmed, bridges, _ = compute_multi_fillet(
                    seams, self.radius, self.continuity, context.scene.spline_surface)
                for net, orders, _ in bridges:
                    self._preview += iso_lines(net, orders)
                for a, ea, b, eb in seams:
                    self._preview += [edge_polyline(*trimmed[a], ea, 24),
                                      edge_polyline(*trimmed[b], eb, 24)]
                if self.corners:
                    nets = [(n, o) for n, o, _ in bridges]
                    pts = np.concatenate([n.reshape(-1, 3) for n, _ in nets])
                    tol = 1e-3 * np.ptp(pts, axis=0).max() + 1e-9
                    for loop in corner_loops(nets):
                        sides = [side_from_patch(*nets[i], e) for i, e in loop]
                        for net, orders in fill_patches(sides, self.continuity, tol=tol):
                            self._preview += iso_lines(net, orders)
            except Exception:
                pass  # preview-only
        context.area.tag_redraw()

    def modal(self, context, event):
        if not self._adjust:
            if event.type == 'MOUSEMOVE':
                hover = pick.nearest_polyline(
                    context, {k: v[2] for k, v in self._seams.items()},
                    (event.mouse_region_x, event.mouse_region_y))
                if hover != self._hover:
                    self._hover = hover
                    context.area.tag_redraw()
            elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                if self._hover is None:
                    return {'RUNNING_MODAL'}
                if self._hover in self._picked:
                    self._picked.remove(self._hover)
                else:
                    self._picked.append(self._hover)
                self._header(context)
                context.area.tag_redraw()
            elif event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
                if not self._picked:
                    return {'RUNNING_MODAL'}
                self.seams = json.dumps([[a, self._seams[(a, b)][0], b, self._seams[(a, b)][1]]
                                         for a, b in self._picked])
                self._adjust = True
                self._header(context)
                self._update_preview(context)
            elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
                return {'PASS_THROUGH'}  # zoom while hunting for seams
            elif self._common_modal(context, event):
                return self._common_result
        else:
            if event.type == 'BACK_SPACE' and event.value == 'PRESS':
                self._adjust = False
                self._preview = []
                self._header(context)
                context.area.tag_redraw()
            elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
                f = 1.1 if event.type == 'WHEELUPMOUSE' else 1 / 1.1
                self.radius = min(1000.0, max(1e-4, self.radius * f))
                self._header(context)
                self._update_preview(context)
            elif event.type in {'C', 'V'} and event.value == 'PRESS':
                if event.type == 'C':
                    self.continuity = 'G1' if self.continuity == 'G2' else 'G2'
                else:
                    self.corners = not self.corners
                self._header(context)
                self._update_preview(context)
            elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                self._finish(context)
                return self.execute(context)
            elif self._common_modal(context, event):
                return self._common_result
        return {'RUNNING_MODAL'}

    def execute(self, context):
        settings = context.scene.spline_surface
        try:
            seams = self._seam_list()
            if not seams:
                raise ValueError("Pick at least one seam")
            trimmed, bridges, infos = compute_multi_fillet(
                seams, self.radius, self.continuity, settings)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        partners = {}
        for a, _, b, _ in seams:
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)
        from . import live  # live imports this module
        live._state["busy"] = True
        made = []
        try:
            for name, (P, o) in trimmed.items():
                src = bpy.data.objects[name]
                old = json.loads(src[META_KEY]) if META_KEY in src else {}
                # Patches filleted together used to meet at the crease (e.g.
                # matched G0); after the trim that match would snap them back.
                matches = [m for m in old.get("matches", []) if m["ref"] not in partners[name]]
                replace_net(src, P, settings, o)
                store_meta(src, raw_meta(P, o, matches=matches))
            for (a, ea, b, eb), (net, orders, tension) in zip(seams, bridges):
                cu = build_surface_data(a + "_fillet", net, settings.resolution, orders)
                for mat in zebra.real_materials(bpy.data.objects[a]):
                    cu.materials.append(mat)
                ob = bpy.data.objects.new(cu.name, cu)
                context.collection.objects.link(ob)
                # Stored as a bridge between the trimmed edges, so Live Update
                # and Regenerate keep it following both patches.
                store_meta(ob, {"method": "blend", "sources": [a, b], "edges": [ea, eb],
                                "continuity": self.continuity, "tension": tension,
                                "flip": False})
                made.append(ob)
            context.view_layer.update()
            corner_count = corner_failed = 0
            if self.corners:
                names = [ob.name for ob in made]
                for loop in corner_loops([(n, o) for n, o, _ in bridges]):
                    # Saved as a Fill Hole between the bridges, so it follows them.
                    meta = {"method": "fill", "sources": [names[i] for i, _ in loop],
                            "edges": [e for _, e in loop], "continuity": self.continuity,
                            "tension": 1.0, "layout": 'AUTO'}
                    try:
                        parts = compute_fill(meta, settings)
                    except ValueError:
                        corner_failed += 1
                        continue
                    for part, (net, orders) in enumerate(parts):
                        cu = build_surface_data(names[loop[0][0]] + "_corner", net,
                                                settings.resolution, orders)
                        for mat in zebra.real_materials(made[loop[0][0]]):
                            cu.materials.append(mat)
                        cob = bpy.data.objects.new(cu.name, cu)
                        context.collection.objects.link(cob)
                        store_meta(cob, dict(meta, part=part))
                        made.append(cob)
                    corner_count += 1
            live.update_chain(sorted(trimmed), context)
        finally:
            live._state["busy"] = False
            live._reset()
        for o in context.selected_objects:
            o.select_set(False)
        for ob in made:
            ob.select_set(True)
        context.view_layer.objects.active = made[-1]
        trims = [i["trim"] for i in infos]
        msg = "fillet: %d seam(s), trimmed %.3g-%.3g, %d corner(s) filled" % (
            len(seams), min(trims), max(trims), corner_count)
        if corner_failed:
            self.report({'WARNING'}, msg + "; %d corner(s) left open (use Fill Hole)"
                        % corner_failed)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = (SS_OT_fillet,)
