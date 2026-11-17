"""Split a patch in two along an iso-line, keeping the group's links intact."""
import json

import bpy
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d

from . import live, pick, zebra
from .bridge.write_surface import build_surface_data
from .core.continuity import find_shared_edge
from .core.nurbs import eval_points
from .core.split import iso_bezier, iso_line, split_net
from .recipe import META_KEY, _make_active, raw_meta, replace_net, store_meta, world_net

# Which edges of the original end up on which piece, per cut direction.
_EDGE_MAP = {'U': ({'U0': 'a', 'U1': 'b'}, ('V0', 'V1')),
             'V': ({'V0': 'a', 'V1': 'b'}, ('U0', 'U1'))}


def _meta(ob):
    return json.loads(ob[META_KEY]) if META_KEY in ob else None


def _freeze(ob, meta):
    """Turn a dependent into a stored net so it no longer reads a vanished edge."""
    net, orders = world_net(ob)
    store_meta(ob, raw_meta(net, orders))


def _fillet_partner(meta, name):
    """True if `meta` is a fillet trim (at any depth) made together with patch `name`."""
    m = meta
    while m and m.get("method") == "fillet_trim":
        if name in m.get("fillet", {}).get("members", []):
            return True
        m = m.get("base")
    return False


def retarget(source, piece_b, direction, net_a, orders_a, net_b, orders_b):
    """Point recipes that read `source` at the right piece. Returns frozen names."""
    edge_map, cut_edges = _EDGE_MAP[direction]
    frozen = []
    for ob in bpy.data.objects:
        meta = _meta(ob)
        if meta is None or ob in (source, piece_b):
            continue
        changed = False
        if _fillet_partner(meta, source.name):
            # trimmed together with the split patch, which no longer carries the
            # fillet: the trim cannot be redone, so keep the shape as it is
            _freeze(ob, meta)
            frozen.append(ob.name)
            continue
        names = meta.get("sources", [])
        if source.name in names:
            method = meta["method"]
            if method == "mirror":
                continue  # handled by the caller: the mirror of the new piece is added
            if method == "extend":
                edges = [meta["edge"]]
            elif "edges" in meta:
                edges = list(meta["edges"])
            else:
                edges = None
            if edges is None:
                continue
            idx = [i for i, n in enumerate(names) if n == source.name]
            if method == "extend":
                idx = [0]
            if any(edges[i] in cut_edges for i in idx):
                _freeze(ob, meta)
                frozen.append(ob.name)
                continue
            for i in idx:
                if edge_map[edges[i]] == 'b':
                    names[i] = piece_b.name
                    changed = True
            meta["sources"] = names
        matches = []
        for m in meta.get("matches", []):
            if m["ref"] != source.name:
                matches.append(m)
                continue
            try:
                tnet, torders = world_net(ob)
            except ValueError:
                continue
            for ref, rnet, rord in ((source, net_a, orders_a), (piece_b, net_b, orders_b)):
                # Tighter than the usual ~1e-3 seam tolerance elsewhere: a
                # match here should land on exactly the piece it used to
                # reference, not loosely "close enough" after the split.
                if find_shared_edge(tnet, rnet, 1e-4 * np.ptp(rnet.reshape(-1, 3), 0).max(),
                                    torders, rord):
                    # The remembered edge pair named the unsplit patch's edges.
                    kept = {k: v for k, v in m.items()
                            if k not in ("edge_t", "edge_r", "reversed")}
                    matches.append(dict(kept, ref=ref.name))
                    break
            changed = True
        if "matches" in meta:
            meta["matches"] = matches
        if changed:
            store_meta(ob, meta)
    return frozen


def _mirrors_of(name):
    return [o for o in bpy.data.objects
            if (_meta(o) or {}).get("method") == "mirror" and _meta(o)["sources"] == [name]]


class SS_OT_split_patch(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.split_patch"
    bl_label = "Split Patch"
    bl_description = ("Cut a patch in two along an iso-line. Click on the patch where to cut; "
                      "the pieces join with G2 and the group's links follow")
    bl_options = {'REGISTER', 'UNDO'}

    source: bpy.props.StringProperty(name="Patch")
    direction: bpy.props.EnumProperty(
        name="Cut", default='U',
        items=[('U', "Across U", "Cut along the iso-line u = t"),
               ('V', "Across V", "Cut along the iso-line v = t")])
    t: bpy.props.FloatProperty(name="Position", default=0.5, min=0.001, max=0.999,
                               subtype='FACTOR')
    snap: bpy.props.BoolProperty(
        name="Snap to Spans", default=True,
        description="On multi-span patches cut only at span boundaries (exact); "
                    "off: anywhere, the pieces are fitted (still G2 at the cut)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.visible_objects)

    # -------------------------------------------------------------- modal
    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._patches = {}
        for ob in context.visible_objects:
            if ob.type == 'SURFACE':
                try:
                    self._patches[ob.name] = world_net(ob)
                except ValueError:
                    pass
        if not self._patches:
            self.report({'ERROR'}, "No patches to split")
            return {'CANCELLED'}
        g = np.linspace(0.0, 1.0, 33)
        self._U, self._V = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
        self._grid = {n: eval_points(net, o, self._U, self._V)
                      for n, (net, o) in self._patches.items()}
        self._hover = None
        self._line = None
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        context.area.header_text_set(
            "Split Patch: click where to cut   D: direction (%s)   S: snap to spans (%s)   "
            "Esc/RMB: cancel" % (self.direction, "on" if self.snap else "off"))

    def _label(self):
        if self._line is None:
            return None
        name, u, v = self._hover
        return ("%s  cut at %.3f" % (name, self._cut_t(name, u, v)),
                self._line[len(self._line) // 2])

    def _draw_lines(self):
        return [(self._line, pick.COLOR_HOVER, 3.0)] if self._line is not None else []

    def _cut_t(self, name, u, v):
        net, orders = self._patches[name]
        t = u if self.direction == 'U' else v
        axis = 0 if self.direction == 'U' else 1
        spans = net.shape[axis] - orders[axis] + 1
        if self.snap and spans > 1:
            t = round(t * spans) / spans
        return t

    def _update(self, context, mouse):
        region, rv3d = context.region, context.region_data
        best, best_d = None, pick.PICK_RADIUS
        for name, pts in self._grid.items():
            for i, q in enumerate(pts):
                p = location_3d_to_region_2d(region, rv3d, q)
                if p is not None:
                    d = (p[0] - mouse[0]) ** 2 + (p[1] - mouse[1]) ** 2
                    if d < best_d ** 2:
                        best, best_d = (name, self._U[i], self._V[i]), d ** 0.5
        self._hover = best
        self._line = None
        if best is not None:
            name, u, v = best
            t = self._cut_t(name, u, v)
            if 0.0 < t < 1.0:
                net, orders = self._patches[name]
                self._line = iso_line(net, orders, self.direction, t)
        context.area.tag_redraw()

    def modal(self, context, event):
        mouse = (event.mouse_region_x, event.mouse_region_y)
        if event.type == 'MOUSEMOVE':
            self._update(context, mouse)
        elif event.type == 'D' and event.value == 'PRESS':
            self.direction = 'V' if self.direction == 'U' else 'U'
            self._header(context)
            self._update(context, mouse)
        elif event.type == 'S' and event.value == 'PRESS':
            self.snap = not self.snap
            self._header(context)
            self._update(context, mouse)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._line is None:
                return {'RUNNING_MODAL'}
            name, u, v = self._hover
            self.source = name
            self.t = u if self.direction == 'U' else v
            self._finish(context)
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    # -------------------------------------------------------------- execute
    def execute(self, context):
        settings = context.scene.spline_surface
        source = bpy.data.objects.get(self.source) if self.source else context.active_object
        if source is None or source.type != 'SURFACE':
            self.report({'ERROR'}, "Pick a patch to split")
            return {'CANCELLED'}
        try:
            net, orders = world_net(source)
            A, B, oa, ob_, t, err = split_net(net, orders, self.direction, self.t, self.snap)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        mirrors = _mirrors_of(source.name)
        live._state["busy"] = True
        try:
            replace_net(source, A, settings, oa)
            store_meta(source, raw_meta(A, oa))
            cu = build_surface_data(source.name + "_split", B, settings.resolution, ob_)
            for mat in zebra.real_materials(source):
                cu.materials.append(mat)
            piece = bpy.data.objects.new(cu.name, cu)
            (source.users_collection[0] if source.users_collection
             else context.collection).objects.link(piece)
            store_meta(piece, raw_meta(B, ob_))
            frozen = retarget(source, piece, self.direction, A, oa, B, ob_)
            for m in mirrors:
                meta = dict(_meta(m), sources=[piece.name])
                meta.pop("fp", None)
                mcu = build_surface_data(piece.name + "_mirror", B, settings.resolution, ob_)
                for mat in zebra.real_materials(m):
                    mcu.materials.append(mat)
                mob = bpy.data.objects.new(mcu.name, mcu)
                (m.users_collection[0] if m.users_collection
                 else context.collection).objects.link(mob)
                store_meta(mob, meta)
            context.view_layer.update()
            result = live.update_chain([source.name, piece.name], context)
        finally:
            live._state["busy"] = False
            live._reset()
        _make_active(context, piece)
        msg = "Split %s at %.3f" % (source.name, t)
        if err > 1e-6:
            msg += " (fitted, max deviation %.3g)" % err
        if frozen:
            self.report({'WARNING'}, msg + "; frozen (their edge was cut): " + ", ".join(frozen))
        elif result["skipped"] or result["error"]:
            self.report({'WARNING'}, msg + "; not rebuilt: "
                        + ", ".join(result["skipped"] + result["error"]))
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def _bezier_object(name, points, collection):
    """Bezier curve object (world space) with the given (co, left, right) points.

    The handles come from the patch, so they are kept as given: ALIGNED inside
    (the line is smooth there, and moving one handle keeps it smooth), FREE at
    corners of a degree-1 line and at the two ends.
    """
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(points) - 1)
    for i, (bp, (co, left, right)) in enumerate(zip(sp.bezier_points, points)):
        bp.handle_left_type = bp.handle_right_type = 'FREE'
        bp.co, bp.handle_left, bp.handle_right = tuple(co), tuple(left), tuple(right)
        a, b = np.asarray(co) - np.asarray(left), np.asarray(right) - np.asarray(co)
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if 0 < i < len(points) - 1 and la > 0 and lb > 0 and np.dot(a, b) > (1 - 1e-9) * la * lb:
            bp.handle_left_type = bp.handle_right_type = 'ALIGNED'
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    return ob


class SS_OT_duplicate_edge_curve(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.duplicate_edge_curve"
    bl_label = "Duplicate Edge as Curve"
    bl_description = ("Copy an iso-line of a patch out as its own curve, usable like any boundary "
                      "curve (Coons, Network, Fill Hole, Project Curve). Click on the patch where "
                      "to cut; Shift+click to pick several edges (across any visible patches) "
                      "before finishing with Enter. The patch itself is left untouched")
    bl_options = {'REGISTER', 'UNDO'}

    source: bpy.props.StringProperty(name="Patch")
    direction: bpy.props.EnumProperty(
        name="Cut", default='U',
        items=[('U', "Across U", "Duplicate the iso-line u = t"),
               ('V', "Across V", "Duplicate the iso-line v = t")])
    t: bpy.props.FloatProperty(name="Position", default=0.5, min=0.0, max=1.0, subtype='FACTOR')
    snap: bpy.props.BoolProperty(
        name="Snap to Spans", default=True,
        description="On multi-span patches snap to span boundaries (exact at the control net); "
                    "off: anywhere on the patch")
    edges: bpy.props.StringProperty(
        name="Edges", default="",
        description="JSON [[patch, direction, t], ...]; non-empty overrides source/direction/t "
                    "(used for Shift-click multi-pick, and to repeat the operator from the redo panel)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.visible_objects)

    # -------------------------------------------------------------- modal
    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._patches = {}
        for ob in context.visible_objects:
            if ob.type == 'SURFACE':
                try:
                    self._patches[ob.name] = world_net(ob)
                except ValueError:
                    pass
        if not self._patches:
            self.report({'ERROR'}, "No patches to duplicate an edge from")
            return {'CANCELLED'}
        g = np.linspace(0.0, 1.0, 33)
        self._U, self._V = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
        self._grid = {n: eval_points(net, o, self._U, self._V)
                      for n, (net, o) in self._patches.items()}
        self._hover = None
        self._line = None
        self._picked = []  # [(name, direction, t, line)], accumulated by Shift-click
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        context.area.header_text_set(
            "Duplicate Edge as Curve: click to copy (and finish)   Shift+click: add and keep "
            "picking (%d so far)   Enter: finish the picked edges   D: direction (%s)   "
            "S: snap to spans (%s)   Esc/RMB: cancel"
            % (len(self._picked), self.direction, "on" if self.snap else "off"))

    def _label(self):
        if self._line is None:
            return None
        name, u, v = self._hover
        return ("%s  at %.3f" % (name, self._edge_t(name, u, v)), self._line[len(self._line) // 2])

    def _draw_lines(self):
        lines = [(line, pick.COLOR_SELECTED, 3.0) for _, _, _, line in self._picked]
        if self._line is not None:
            lines.append((self._line, pick.COLOR_HOVER, 3.0))
        return lines

    def _edge_t(self, name, u, v):
        net, orders = self._patches[name]
        t = u if self.direction == 'U' else v
        axis = 0 if self.direction == 'U' else 1
        spans = net.shape[axis] - orders[axis] + 1
        if self.snap and spans > 1:
            t = round(t * spans) / spans
        return t

    def _update(self, context, mouse):
        region, rv3d = context.region, context.region_data
        best, best_d = None, pick.PICK_RADIUS
        for name, pts in self._grid.items():
            for i, q in enumerate(pts):
                p = location_3d_to_region_2d(region, rv3d, q)
                if p is not None:
                    d = (p[0] - mouse[0]) ** 2 + (p[1] - mouse[1]) ** 2
                    if d < best_d ** 2:
                        best, best_d = (name, self._U[i], self._V[i]), d ** 0.5
        self._hover = best
        self._line = None
        if best is not None:
            name, u, v = best
            t = self._edge_t(name, u, v)
            net, orders = self._patches[name]
            self._line = iso_line(net, orders, self.direction, t)
        context.area.tag_redraw()

    def modal(self, context, event):
        mouse = (event.mouse_region_x, event.mouse_region_y)
        if event.type == 'MOUSEMOVE':
            self._update(context, mouse)
        elif event.type == 'D' and event.value == 'PRESS':
            self.direction = 'V' if self.direction == 'U' else 'U'
            self._header(context)
            self._update(context, mouse)
        elif event.type == 'S' and event.value == 'PRESS':
            self.snap = not self.snap
            self._header(context)
            self._update(context, mouse)
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._line is None:
                return {'RUNNING_MODAL'}
            name, u, v = self._hover
            t = self._edge_t(name, u, v)
            if event.shift:
                if not any(n == name and d == self.direction and abs(tt - t) < 1e-9
                           for n, d, tt, _ in self._picked):
                    self._picked.append((name, self.direction, t, self._line))
                self._header(context)
                return {'RUNNING_MODAL'}
            self.edges = json.dumps([[n, d, tt] for n, d, tt, _ in self._picked]
                                    + [[name, self.direction, t]])
            self._finish(context)
            return self.execute(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if not self._picked:
                return {'RUNNING_MODAL'}
            self.edges = json.dumps([[n, d, tt] for n, d, tt, _ in self._picked])
            self._finish(context)
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    # -------------------------------------------------------------- execute
    def _duplicate(self, context, source_name, direction, t):
        source = bpy.data.objects.get(source_name)
        if source is None or source.type != 'SURFACE':
            return None
        try:
            net, orders = world_net(source)
        except ValueError:
            return None
        axis = 0 if direction == 'U' else 1
        spans = net.shape[axis] - orders[axis] + 1
        if self.snap and spans > 1:
            t = round(t * spans) / spans
        ob = _bezier_object(source.name + "_edge", iso_bezier(net, orders, direction, t),
                            context.collection)
        return ob, source.name, direction, t

    def execute(self, context):
        picks = json.loads(self.edges) if self.edges else None
        if not picks:
            source = bpy.data.objects.get(self.source) if self.source else context.active_object
            if source is None or source.type != 'SURFACE':
                self.report({'ERROR'}, "Pick a patch to duplicate an edge from")
                return {'CANCELLED'}
            picks = [[source.name, self.direction, self.t]]
        created = []
        for name, direction, t in picks:
            result = self._duplicate(context, name, direction, t)
            if result is not None:
                created.append(result)
        if not created:
            self.report({'ERROR'}, "No valid patch to duplicate an edge from")
            return {'CANCELLED'}
        _make_active(context, created[-1][0])
        for ob, *_ in created:
            ob.select_set(True)
        if len(created) == 1:
            ob, name, direction, t = created[0]
            self.report({'INFO'}, "Duplicated %s's %s = %.3f edge as %s"
                        % (name, direction.lower(), t, ob.name))
        else:
            self.report({'INFO'}, "Duplicated %d edges as curves" % len(created))
        return {'FINISHED'}


classes = (SS_OT_split_patch, SS_OT_duplicate_edge_curve)
