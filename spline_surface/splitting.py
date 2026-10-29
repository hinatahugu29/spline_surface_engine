"""Split a patch in two along an iso-line, keeping the group's links intact."""
import json

import bpy
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d

from . import live, pick, zebra
from .bridge.write_surface import build_surface_data
from .core.continuity import find_shared_edge
from .core.nurbs import eval_points
from .core.split import iso_line, split_net
from .operators import META_KEY, _make_active, raw_meta, replace_net, store_meta, world_net

# Which edges of the original end up on which piece, per cut direction.
_EDGE_MAP = {'U': ({'U0': 'a', 'U1': 'b'}, ('V0', 'V1')),
             'V': ({'V0': 'a', 'V1': 'b'}, ('U0', 'U1'))}


def _meta(ob):
    return json.loads(ob[META_KEY]) if META_KEY in ob else None


def _freeze(ob, meta):
    """Turn a dependent into a stored net so it no longer reads a vanished edge."""
    net, orders = world_net(ob)
    store_meta(ob, raw_meta(net, orders))


def retarget(source, piece_b, direction, net_a, orders_a, net_b, orders_b):
    """Point recipes that read `source` at the right piece. Returns frozen names."""
    edge_map, cut_edges = _EDGE_MAP[direction]
    frozen = []
    for ob in bpy.data.objects:
        meta = _meta(ob)
        if meta is None or ob in (source, piece_b):
            continue
        changed = False
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
                if find_shared_edge(tnet, rnet, 1e-4 * np.ptp(rnet.reshape(-1, 3), 0).max(),
                                    torders, rord):
                    matches.append(dict(m, ref=ref.name))
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


class SS_OT_split_patch(bpy.types.Operator):
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

    def _finish(self, context):
        self._picker.remove()
        context.area.header_text_set(None)
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
        elif event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._finish(context)
            return {'CANCELLED'}
        elif event.type in {'MIDDLEMOUSE', 'NUMPAD_PERIOD'} or event.type.startswith('NDOF') \
                or event.type.startswith('TRACKPAD'):
            return {'PASS_THROUGH'}
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


classes = (SS_OT_split_patch,)
