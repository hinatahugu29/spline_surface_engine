"""Edits that spread to neighbouring patches.

* Linked Edit: when a patch changes (edit mode, transform), every patch near
  it follows through a smooth falloff field; patches that shared an edge with
  it are re-matched with the continuity they had (G0/G1/G2).
* Edge Chain: a run of open edges across patches becomes one NURBS curve;
  editing that curve moves all of those edges (and their surroundings).
* Merge Patches: a rectangular grid of patches refitted as one patch.
"""
import json

import bpy
import numpy as np

from . import pick
from . import zebra
from .bridge.write_surface import build_surface_data
from .core.continuity import find_shared_edge, grade, match_net, measure
from .core.linked import (chain_polyline, deform, edge_order, edge_row, falloff_field,
                          grid_layout, merge_grid, order_chain)
from .core.nurbs import basis_matrix, fit_polyline
from .recipe import (ADDON_WRITES, META_KEY, _make_active, hand_edited, raw_meta,
                     replace_net, store_meta, surface_edges, world_net)

CHAIN_KEY = "spline_surface_chain"
CHAIN_SAMPLES = 96

_cache = {}  # object name -> (world net, orders) as last seen


def reset():
    _cache.clear()
    ADDON_WRITES.clear()


def _size(net):
    return float(np.ptp(np.asarray(net).reshape(-1, 3), axis=0).max())


def _write(ob, net, orders, context):
    from .live import _write_in_place
    if not _write_in_place(ob, net, orders):
        replace_net(ob, net, context.scene.spline_surface, orders)
    if META_KEY in ob:
        meta = json.loads(ob[META_KEY])
        if meta.get("method") == "raw":  # a free patch: its recipe is its net
            meta["net"], meta["orders"] = np.asarray(net).tolist(), list(orders)
            store_meta(ob, meta)
    _cache[ob.name] = (np.array(net, float), tuple(orders))


def _regenerated(names, context):
    """Patches Live Update will rebuild from `names`; they must not be deformed too."""
    if not context.scene.spline_surface.live_update:
        return set()
    from .live import _meta, downstream_order
    order, _ = downstream_order(names)
    out = set()
    for n in order:
        ob = bpy.data.objects.get(n)
        meta = _meta(ob) if ob is not None else None
        if meta is not None and not hand_edited(ob, meta):
            out.add(n)
    return out


def _targets(context, centre_box, radius, skip):
    """Visible patches whose box lies within `radius` of `centre_box`."""
    lo, hi = centre_box[0] - radius, centre_box[1] + radius
    out = []
    for ob in context.visible_objects:
        if ob.type != 'SURFACE' or ob.name in skip:
            continue
        try:
            net, orders = world_net(ob)
        except ValueError:
            continue
        flat = net.reshape(-1, 3)
        if np.any(flat.max(0) < lo) or np.any(flat.min(0) > hi):
            continue
        out.append((ob, net, orders))
    return out


def _deform_around(context, field, box, radius, skip):
    """Move every nearby patch by `field`. Returns {name: (old net, new net, orders)}."""
    moved = {}
    cands = _targets(context, box, radius, skip)
    regen = _regenerated([c[0].name for c in cands] + list(skip), context)
    for ob, net, orders in cands:
        if ob.name in regen:
            continue
        new = deform(net, field)
        if np.abs(new - net).max() <= 1e-9 * max(_size(net), 1e-6):
            continue
        moved[ob.name] = (net, new, orders)
    return moved


def _rejoin(order, before, after):
    """Re-match the seams the patches had before they moved.

    `order` lists names, earlier ones win: each patch is matched to the
    earlier patches it shared an edge with, with the continuity it had.
    before/after: {name: (net, orders)}. Updates `after` in place.
    """
    for j, b in enumerate(order):
        Nb, ob_ = before[b]
        tol = 1e-3 * max(_size(Nb), 1e-6) + 1e-6
        for a in order[:j]:
            Na, oa = before[a]
            sh = find_shared_edge(Nb, Na, tol, ob_, oa)
            if sh is None:
                continue
            g = grade(measure(Nb, Na, *sh[:3], ob_, oa))
            if g not in ('G0', 'G1', 'G2'):
                continue
            try:
                after[b] = match_net(after[b][0], after[a][0], *sh[:3], g,
                                     after[b][1], after[a][1])
            except ValueError:
                pass


def propagate(ob, old, new, context):
    """Spread the change old -> new of patch `ob` to its surroundings."""
    settings = context.scene.spline_surface
    (P0, o0), (P1, o1) = old, new
    if P0.shape != P1.shape or tuple(o0) != tuple(o1):
        return []  # control net was rebuilt: no point-to-point displacement
    disp = P1 - P0
    size = max(_size(P0), 1e-6)
    if np.abs(disp).max() <= 1e-7 * size:
        return []
    radius = settings.link_falloff * size
    field = falloff_field(P0.reshape(-1, 3), disp.reshape(-1, 3), radius)
    flat = np.concatenate([P0.reshape(-1, 3), P1.reshape(-1, 3)])
    moved = _deform_around(context, field, (flat.min(0), flat.max(0)), radius, {ob.name})
    before = {ob.name: (P0, o0)}
    after = {ob.name: (P1, o1)}
    for name, (N, Nn, on) in moved.items():
        before[name], after[name] = (N, on), (Nn, on)
    # The edited patch first, then outwards.
    order = [ob.name] + sorted(moved, key=lambda n: np.linalg.norm(
        moved[n][0].mean((0, 1)) - P0.mean((0, 1))))
    _rejoin(order, before, after)
    for name in moved:
        _write(bpy.data.objects[name], *after[name], context)
    return list(moved)


# ---------------------------------------------------------------- edge chain

def chain_samples(cob, count=CHAIN_SAMPLES):
    """World points along a chain curve, evenly in its parameter (None if not NURBS)."""
    if not cob.data.splines:
        return None
    sp = cob.data.splines[0]
    if sp.type != 'NURBS' or len(sp.points) < 2:
        return None
    co = np.array([p.co[:3] for p in sp.points])
    k = min(sp.order_u, len(co))
    pts = basis_matrix(len(co), k, np.linspace(0.0, 1.0, count)) @ co
    M = np.array(cob.matrix_world)
    return pts @ M[:3, :3].T + M[:3, 3]


def _section(samples, t0, t1, count=32, t=None):
    """Points between parameters t0 and t1 of a sampled curve (samples at t, default even)."""
    t = np.linspace(0.0, 1.0, len(samples)) if t is None else t
    s = np.linspace(t0, t1, count)
    return np.stack([np.interp(s, t, samples[:, d]) for d in range(3)], axis=1)


def sync_chain(cob, context):
    """Bring the chain's edges (and their surroundings) to the edited curve."""
    data = json.loads(cob[CHAIN_KEY])
    new = chain_samples(cob)
    rest = np.array(data["rest"], float)
    if new is None or new.shape != rest.shape:
        return []
    disp = new - rest
    if np.abs(disp).max() <= 1e-7 * max(data["size"], 1e-6):
        return []
    settings = context.scene.spline_surface
    radius = settings.link_falloff * data["size"]
    field = falloff_field(rest, disp, radius)
    flat = np.concatenate([rest, new])
    # The chain's own patches get their edge row snapped to the curve exactly
    # below; skip them here so they aren't *also* smoothly falloff-deformed
    # as if they were a nearby, unrelated patch. That double deformation
    # doesn't cancel out on its own: fitting the exact edge row again after
    # a round trip leaves the deformed interior control points drifted.
    own = {seg["ob"] for seg in data["segments"]}
    moved = _deform_around(context, field, (flat.min(0), flat.max(0)), radius, own)
    overshoot = []
    for seg in data["segments"]:
        ob = bpy.data.objects.get(seg["ob"])
        if ob is None:
            continue
        if ob.name in moved:
            N, Nn, orders = moved[ob.name]
        else:
            try:
                N, orders = world_net(ob)
            except ValueError:
                continue
            Nn = N.copy()
        row = edge_row(Nn, seg["edge"])
        n, k = Nn[row].shape[0], edge_order(orders, seg["edge"])
        pts = _section(new, seg["t0"], seg["t1"])
        if seg["rev"]:
            pts = pts[::-1]
        new_row = fit_polyline(pts, n, k)[0]
        # A tight bend fit through few control points (n close to k) can
        # overshoot wildly *between* the sample points even though it still
        # passes near them -- fit_polyline's own deviation won't catch that.
        # Flag a row whose control points land well outside the target
        # polyline's own extent.
        lo, hi = pts.min(0), pts.max(0)
        margin = 0.5 * max(np.ptp(pts, axis=0).max(), 1e-6)
        beyond = np.maximum(lo - margin - new_row, 0) + np.maximum(new_row - (hi + margin), 0)
        if beyond.max() > 0:
            overshoot.append(ob.name)
        Nn[row] = new_row
        moved[ob.name] = (N, Nn, orders)
    if overshoot:
        context.scene.spline_surface.live_note = (
            "Edge Chain: %s overshot a sharp bend (too few control points for it); "
            "raise Control Points on the chain" % ", ".join(sorted(set(overshoot))))
    before = {n: (v[0], v[2]) for n, v in moved.items()}
    after = {n: (v[1], v[2]) for n, v in moved.items()}
    # The chain's own patches first (in chain order), then everything else
    # caught by the falloff, nearest to the chain first: a deterministic
    # order, matching propagate()'s "closer wins" rule.
    chain = [g["ob"] for g in data["segments"] if g["ob"] in moved]
    centre = new.mean(0)
    rest_names = sorted((n for n in moved if n not in chain),
                        key=lambda n: np.linalg.norm(before[n][0].mean((0, 1)) - centre))
    _rejoin(chain + rest_names, before, after)
    for name in moved:
        _write(bpy.data.objects[name], *after[name], context)
    data["rest"] = new.tolist()
    cob[CHAIN_KEY] = json.dumps(data)
    return list(moved)


def on_update(touched, context):
    """Depsgraph hook (called by live): returns the patches written."""
    settings = context.scene.spline_surface
    written = []
    for name in sorted(touched):
        cob = bpy.data.objects.get(name)
        if cob is not None and cob.type == 'CURVE' and CHAIN_KEY in cob:
            written += sync_chain(cob, context)
    if not settings.linked_edit:
        reset()
        return written
    if not _cache:
        remember_all(context)
    active = context.view_layer.objects.active
    for name in sorted(touched):
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != 'SURFACE' or name in written:
            continue
        try:
            now = world_net(ob)
        except ValueError:
            continue
        before = _cache.get(name)
        _cache[name] = (now[0].copy(), tuple(now[1]))
        if name in ADDON_WRITES:
            ADDON_WRITES.discard(name)
            continue
        # Only a patch the user is working on drives its neighbours.
        if before is None or ob != active:
            continue
        written += propagate(ob, before, now, context)
    return written


def remember_all(context):
    """Baseline every visible patch (so the first edit has something to compare)."""
    for ob in context.visible_objects:
        if ob.type == 'SURFACE':
            try:
                net, orders = world_net(ob)
            except ValueError:
                continue
            _cache[ob.name] = (net.copy(), tuple(orders))


# ---------------------------------------------------------------- operators

class SS_OT_edge_chain(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.edge_chain"
    bl_label = "Edge Chain"
    bl_description = ("Click an open edge: the run of open edges joined to it becomes one "
                      "curve. Edit that curve and all of those edges follow, regardless of "
                      "the Linked Edit toggle")
    bl_options = {'REGISTER', 'UNDO'}

    source: bpy.props.StringProperty(name="Patch")
    edge: bpy.props.StringProperty(name="Edge")
    points: bpy.props.IntProperty(
        name="Control Points", default=0, min=0, max=64,
        description="Points of the control curve (0: automatic, 3 per edge)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.visible_objects)

    def _chain(self, edges, key):
        size = max((np.ptp(pl, axis=0).max() for pl in edges.values()), default=1.0)
        keys = [key] + [k for k in edges if k != key]
        return order_chain({k: edges[k] for k in keys}, 1e-3 * size + 1e-6)

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return self.execute(context)
        self._edges = surface_edges(context)
        if not self._edges:
            self.report({'ERROR'}, "No open patch edges")
            return {'CANCELLED'}
        self._hover, self._run = None, []
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Edge Chain: click an open edge   Esc/RMB: cancel")
        return {'RUNNING_MODAL'}

    def _label(self):
        if self._hover is None:
            return None
        text, at = pick.edge_label(self._hover, self._edges[self._hover])
        return "%s  (+%d joined)" % (text, len(self._run) - 1), at

    def _draw_lines(self):
        run = {k for k, _ in self._run}
        lines = [(pts, pick.COLOR_EDGE, 2.0) for e, pts in self._edges.items() if e not in run]
        lines += [(self._edges[k], pick.COLOR_HOVER, 4.0) for k in run]
        return lines

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            hover = pick.nearest_polyline(context, self._edges,
                                          (event.mouse_region_x, event.mouse_region_y))
            if hover != self._hover:
                self._hover = hover
                self._run = self._chain(self._edges, hover) if hover else []
                context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._hover is None:
                return {'RUNNING_MODAL'}
            self._finish(context)
            self.source, self.edge = self._hover
            return self.execute(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}

    def execute(self, context):
        edges = surface_edges(context, samples=32)
        key = (self.source, self.edge)
        if key not in edges:
            self.report({'ERROR'}, "Pick an open patch edge")
            return {'CANCELLED'}
        chain = self._chain(edges, key)
        pts, ranges = chain_polyline(edges, chain)
        n = self.points or min(64, max(4, 3 * len(chain) + 1))
        ctrl, _ = fit_polyline(pts, n, 4)
        cu = bpy.data.curves.new("EdgeChain", 'CURVE')
        cu.dimensions = '3D'
        sp = cu.splines.new('NURBS')
        sp.points.add(n - 1)
        for p, c in zip(sp.points, ctrl):
            p.co = (*c, 1.0)
        sp.order_u = 4
        sp.use_endpoint_u = True
        sp.resolution_u = 24
        cob = bpy.data.objects.new("EdgeChain", cu)
        context.collection.objects.link(cob)
        sizes = []
        segs = []
        for ((name, edge), rev), (t0, t1) in zip(chain, ranges):
            sizes.append(_size(world_net(bpy.data.objects[name])[0]))
            segs.append({"ob": name, "edge": edge, "rev": rev, "t0": t0, "t1": t1})
        # Rest = the edges as they are now: the first sync snaps them onto the curve.
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-12)
        rest = _section(pts, 0.0, 1.0, CHAIN_SAMPLES, t)
        cob[CHAIN_KEY] = json.dumps({"segments": segs, "rest": rest.tolist(),
                                     "size": float(np.mean(sizes))})
        _make_active(context, cob)
        context.view_layer.update()
        from . import live
        live._state["busy"] = True
        try:
            written = sync_chain(cob, context)
            if context.scene.spline_surface.live_update and written:
                live.update_chain(written, context)
        finally:
            live._state["busy"] = False
            live._reset()
        self.report({'INFO'}, "Chain of %d edge(s): edit the EdgeChain curve" % len(chain))
        return {'FINISHED'}


class SS_OT_merge_patches(bpy.types.Operator):
    bl_idname = "spline_surface.merge_patches"
    bl_label = "Merge Patches"
    bl_description = ("Refit the selected patches, laid edge to edge on a rectangular grid, "
                      "as one patch. The originals are hidden (or deleted)")
    bl_options = {'REGISTER', 'UNDO'}

    detail: bpy.props.FloatProperty(
        name="Detail", default=1.0, min=0.25, max=4.0,
        description="Control points of the merged patch, relative to the originals")
    delete: bpy.props.BoolProperty(
        name="Delete Originals", default=False,
        description="Delete the selected patches instead of hiding them")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and sum(
            o.type == 'SURFACE' for o in context.selected_objects) >= 2

    def execute(self, context):
        settings = context.scene.spline_surface
        obs = [o for o in context.selected_objects if o.type == 'SURFACE']
        try:
            patches = [world_net(o) for o in obs]
            size = max(_size(p[0]) for p in patches)
            grid = grid_layout(patches, 1e-3 * size + 1e-6)
            net, orders, err = merge_grid(grid, detail=self.detail)
            cu = build_surface_data("Merged", net, settings.resolution, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        for mat in zebra.real_materials(obs[0]):
            cu.materials.append(mat)
        ob = bpy.data.objects.new("Merged", cu)
        context.collection.objects.link(ob)
        store_meta(ob, raw_meta(net, orders))
        from .live import dependency_graph
        down = dependency_graph()
        orphans = sorted({d for o in obs for d in down.get(o.name, ())} - {o.name for o in obs})
        for o in obs:
            if self.delete:
                bpy.data.objects.remove(o)
            else:
                o.hide_set(True)
        _make_active(context, ob)
        ni = max(c[0] for c in grid) + 1
        nj = max(c[1] for c in grid) + 1
        msg = "Merged %dx%d patches, max deviation %.2g%%" % (ni, nj, 100 * err / size)
        if orphans:
            msg += "; still built from the originals: " + ", ".join(orphans)
        self.report({'WARNING'} if orphans or err > 0.01 * size else {'INFO'}, msg)
        return {'FINISHED'}


classes = (SS_OT_edge_chain, SS_OT_merge_patches)
