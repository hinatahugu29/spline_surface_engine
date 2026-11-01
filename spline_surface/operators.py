import json

import bpy
import numpy as np

from .bridge.read_curves import read_polyline
from .bridge.write_surface import build_surface_data
from .core.bezier import fit_curve
from .core.coons import close_loop, coons_net, loop_to_sides, orient_loop, snap_corners
from .core.blend import blend_curve_net, blend_net
from .core.corner import corner_net
from .core.fill import fill_patches, side_from_curve, side_from_patch
from .core.fillet import fillet_net, multi_fillet
from .core.flange import flange_net
from .core.orient import flip_net, part_flips
from .core.sweep import sweep_net
from .core.mirror import mirror_net, symmetrize
from .core.network import build_network
from .core.offset import offset_net
from .core.continuity import find_shared_edge, grade, match_net, measure
from .core.extend import bezier_orders, extend_net
from .core.loft import add_rows, loft_net, order_sections
from . import pick
from . import zebra
from .recipe import (EDGE_KEYS, META_KEY, META_VERSION, NET_OVERRIDE, OPPOSITE_EDGE, _make_active,
                     edge_polyline, hand_edited, hand_edited_net, iso_lines, raw_meta,
                     replace_net, set_origin, store_meta, surface_edges, world_net)

BOUNDARY_TYPES = {'CURVE', 'GREASEPENCIL'}


def _auto_tolerance(polys, rel):
    allpts = np.concatenate(polys)
    diag = np.linalg.norm(allpts.max(axis=0) - allpts.min(axis=0))
    return max(diag * rel, 1e-6)


def compute_coons(sources, depsgraph, settings):
    """Fit boundary curves and build a Coons control net. Returns (net, max_fit_error)."""
    polys = [read_polyline(o, depsgraph, settings.samples) for o in sources]
    loop = close_loop(orient_loop(polys, _auto_tolerance(polys, settings.gap_tolerance)))
    fits = [fit_curve(p, settings.degree) for p in loop]
    sides = snap_corners(*loop_to_sides([f[0] for f in fits]))
    return coons_net(*sides), max(f[1] for f in fits)


def compute_loft(sources, depsgraph, settings, keep_order=False, rows=0):
    """Fit section curves and loft through them.

    Returns (net, max_fit_error, ordered source objects).
    """
    polys = [read_polyline(o, depsgraph, settings.samples) for o in sources]
    order = order_sections(polys, keep_order)
    fits = [fit_curve(polys[i][::-1] if flip else polys[i], settings.degree) for i, flip in order]
    net, _ = loft_net([f[0] for f in fits], settings.degree)
    net = add_rows(net, rows, settings.degree)
    return net, max(f[1] for f in fits), [sources[i] for i, _ in order]


def compute_extend(source, meta, settings):
    net, orders = world_net(source)
    return extend_net(net, meta["edge"], meta["length"], meta["continuity"],
                      settings.degree, orders)


def compute_flange(source, meta):
    # Blender shows the side against du x dv as the front, so folding along it goes to the back
    net, orders = world_net(source)
    return flange_net(net, meta["edge"], meta["length"], meta["angle"], not meta.get("flip", False),
                      orders, meta.get("offset", 0.0))


sweep_note = None     # why the last compute_sweep fell back to Auto length (None: it did not)


def sweep_target(meta):
    """('plane', axis, value) / ('line', points) for the recipe's target, or None."""
    t = meta.get("target")
    if not t or not meta.get("straight"):
        return None
    if t["kind"] == 'plane':
        return 'plane', t["axis"], t["value"]
    ob = bpy.data.objects.get(t["source"])
    if ob is None:
        raise ValueError("the target %s is gone" % t["source"])
    if t["edge"] == 'CURVE':
        return 'line', read_polyline(ob, bpy.context.evaluated_depsgraph_get(), 48)
    net, orders = world_net(ob)
    return 'line', edge_polyline(net, orders, t["edge"], 48)


def compute_sweep(sources, meta):
    global sweep_note
    sweep_note = None
    (Pa, oa), (Pb, ob) = world_net(sources[0]), world_net(sources[1])
    args = (Pa, oa, meta["edges"][0], Pb, ob, meta["edges"][1], meta.get("flip", False),
            meta.get("straight", False), meta.get("length", 0.0))
    try:
        return sweep_net(*args, target=sweep_target(meta))
    except ValueError as e:
        if not meta.get("target"):
            raise
        sweep_note = "Target not used (%s); the length is automatic" % e
        return sweep_net(*args)


def compute_corner(sources, meta, settings):
    (Pa, oa), (Pb, ob) = world_net(sources[0]), world_net(sources[1])
    size = max(np.ptp(Pa.reshape(-1, 3), axis=0).max(), np.ptp(Pb.reshape(-1, 3), axis=0).max())
    return corner_net(Pa, oa, meta["edges"][0], Pb, ob, meta["edges"][1],
                      meta["continuity"], meta.get("tension", 1.0),
                      tol=settings.gap_tolerance * 0.1 * size + 1e-6)


def compute_fillet(sources, meta, settings):
    (Pa, oa), (Pb, ob) = world_net(sources[0]), world_net(sources[1])
    return fillet_net(Pa, oa, meta["edges"][0], Pb, ob, meta["edges"][1],
                      meta["radius"], tol=_seam_tolerance(Pa, Pb, settings))


def compute_fill(meta, settings):
    """All patches of a hole fill: [(net, orders)] (one, or one per side)."""
    sides, size = [], 0.0
    dg = bpy.context.evaluated_depsgraph_get()
    for name, edge in zip(meta["sources"], meta["edges"]):
        ob = bpy.data.objects.get(name)
        if ob is None:
            raise ValueError("Source object missing: %s" % name)
        if edge == 'CURVE':
            pts = read_polyline(ob, dg, 96)
            side, _ = side_from_curve(pts)
        else:
            P, orders = world_net(ob)
            pts = P.reshape(-1, 3)
            side = side_from_patch(P, orders, edge)
        size = max(size, np.ptp(pts, axis=0).max())
        sides.append(side)
    return fill_patches(sides, meta["continuity"], meta.get("tension", 1.0),
                        tol=settings.gap_tolerance * 0.1 * size + 1e-6,
                        layout=meta.get("layout", 'AUTO'))


def compute_blend(sources, meta, settings):
    """Returns (net, orders, curve_fit_error). curve_fit_error is None unless
    the second side is a curve (meta["edges"][1] == 'CURVE')."""
    Pa, oa = world_net(sources[0])
    if meta["edges"][1] == 'CURVE':
        pts = read_polyline(sources[1], bpy.context.evaluated_depsgraph_get(), 96)
        net, orders, err = blend_curve_net(
            Pa, oa, meta["edges"][0], pts, meta["continuity"], meta.get("tension", 1.0),
            meta.get("flip", False), meta.get("curve_tangent", 'SMOOTH'))
        return net, orders, err
    Pb, ob = world_net(sources[1])
    net, orders = blend_net(Pa, oa, meta["edges"][0], Pb, ob, meta["edges"][1],
                            meta["continuity"], meta.get("tension", 1.0), meta.get("flip", False))
    return net, orders, None


def fillet_base(meta, fillet_id):
    """The recipe a patch had before fillet `fillet_id` trimmed it.

    A patch filleted again keeps its earlier trim as the base of the later
    one, so walk down the chain to the right fillet.
    """
    m = meta
    while m.get("method") == "fillet_trim":
        if m["fillet"]["id"] == fillet_id:
            return m["base"]
        m = m["base"]
    raise ValueError("this patch is no longer part of the fillet")


def compute_fillet_trim(meta, context):
    """A fillet member's trimmed net: rebuild every member's untrimmed shape
    from its base recipe and fillet them all again (each reach depends on
    both sides of its seam). Returns (net, orders)."""
    from . import live
    fs = meta["fillet"]
    bases = {}
    for name in fs["members"]:
        ob = bpy.data.objects.get(name)
        if ob is None or META_KEY not in ob:
            raise ValueError("Fillet patch missing: %s" % name)
        member = meta if name == meta["member"] else json.loads(ob[META_KEY])
        bases[name] = fillet_base(member, fs["id"])
    # A member built from another member (a flange wall off the patch it was filleted with) must
    # read that one untrimmed, as it was when the fillet was made, so build in that order.
    patches, pending = {}, list(bases)
    try:
        while pending:
            ready = [n for n in pending if not (set(live._inputs(bases[n])) & set(pending) - {n})]
            if not ready:
                raise ValueError("Fillet members depend on each other in a loop")
            for name in ready:
                net, orders, _ = compute_from_meta(bases[name], context)
                patches[name] = (net, orders)
                NET_OVERRIDE[name] = (net, orders)
                pending.remove(name)
    finally:
        for name in bases:
            NET_OVERRIDE.pop(name, None)
    seams = [tuple(s) for s in fs["seams"]]
    tol = max(_seam_tolerance(patches[a][0], patches[b][0], context.scene.spline_surface)
              for a, _, b, _ in seams)
    trimmed, _, _ = multi_fillet(patches, seams, fs["radius"], fs["continuity"], tol)
    return trimmed[meta["member"]]


def network_tolerance(polys, settings):
    """How far apart two curves may pass and still count as crossing."""
    # A typical curve's size, not the whole selection's: one stray curve far
    # away must not widen the tolerance until neighbouring curves "cross".
    size = float(np.median([np.ptp(p, axis=0).max() for p in polys]))
    # Curves are bent through the crossings, so allow generous misses (5% of size).
    return settings.gap_tolerance * 0.5 * size + 1e-6


def compute_network(sources, meta, settings, depsgraph):
    """All cells of a curve network: ([(i, j, net, orders)], info)."""
    polys = [read_polyline(o, depsgraph, 120) for o in sources]
    return build_network(polys, network_tolerance(polys, settings), meta["continuity"])


def compute_from_meta(meta, context):
    """Rebuild a patch from its stored recipe. Returns (net, orders, message)."""
    if meta.get("ver", 1) > META_VERSION:
        raise ValueError("this patch was built by a newer version of Spline Surface; "
                         "update the add-on to rebuild it")
    settings = context.scene.spline_surface
    names = meta["sources"]
    missing = [n for n in names if n not in bpy.data.objects]
    if missing:
        raise ValueError("Source objects missing: %s" % ", ".join(missing))
    sources = [bpy.data.objects[n] for n in names]
    orders = None
    if meta["method"] == "coons":
        net, err = compute_coons(sources, context.evaluated_depsgraph_get(), settings)
        msg = "max fit error %.3g" % err
    elif meta["method"] == "loft":
        net, err, _ = compute_loft(sources, context.evaluated_depsgraph_get(), settings,
                                   keep_order=True, rows=meta.get("rows", 0))
        msg = "max fit error %.3g" % err
    elif meta["method"] == "extend":
        (net, orders), msg = compute_extend(sources[0], meta, settings), meta["continuity"]
    elif meta["method"] == "flange":
        net, orders = compute_flange(sources[0], meta)
        msg = "flange %.4g at %.4g deg" % (meta["length"], meta["angle"])
    elif meta["method"] == "sweep":
        net, orders = compute_sweep(sources, meta)
        msg = "sweep along %s" % sources[1].name
    elif meta["method"] == "corner":
        net, orders = compute_corner(sources, meta, settings)
        msg = meta["continuity"]
    elif meta["method"] == "fill":
        parts = compute_fill(meta, settings)
        part = meta.get("part", 0)
        if part >= len(parts):
            raise ValueError("the hole now has a different number of sides")
        net, orders = parts[part]
        msg = meta["continuity"]
    elif meta["method"] == "blend":
        net, orders, _ = compute_blend(sources, meta, settings)
        msg = meta["continuity"]
    elif meta["method"] == "fillet":
        net, orders, info = compute_fillet(sources, meta, settings)
        msg = "fillet r=%.4g (reach %.3g-%.3g)" % (meta["radius"], info["reach"].min(), info["reach"].max())
    elif meta["method"] == "fillet_trim":
        net, orders = compute_fillet_trim(meta, context)
        msg = "trimmed by a r=%.4g fillet" % meta["fillet"]["radius"]
    elif meta["method"] == "network":
        cells, info = compute_network(sources, meta, settings, context.evaluated_depsgraph_get())
        part = meta.get("part", 0)
        if part >= len(cells):
            raise ValueError("the curve network now has a different number of cells")
        _, _, net, orders = cells[part]
        msg = "network cell %d/%d" % (part + 1, len(cells))
    elif meta["method"] == "offset":
        src_net, src_orders = world_net(sources[0])
        net, orders, err = offset_net(src_net, src_orders, meta["distance"],
                                      meta.get("refine", 2))
        msg = "offset %.4g (max deviation %.2g)" % (meta["distance"], err)
    elif meta["method"] == "mirror":
        net, orders = world_net(sources[0])
        net = mirror_net(net, meta["axis"], meta.get("offset", 0.0))
        msg = "mirror of %s" % sources[0].name
    elif meta["method"] == "raw":
        net, orders, msg = np.array(meta["net"]), tuple(meta["orders"]), "stored net"
    else:
        raise ValueError("unknown method %r" % meta["method"])
    orders = orders or bezier_orders(net)
    for match in meta.get("matches", []):
        ref = bpy.data.objects.get(match["ref"])
        if ref is None:
            raise ValueError("Matched patch missing: %s" % match["ref"])
        net, orders = apply_match(net, orders, *world_net(ref), match["continuity"], settings)
        msg += ", %s to %s" % (match["continuity"], ref.name)
    sym = meta.get("symmetry")
    if sym:
        net, fixed = symmetrize(net, sym["axis"], sym.get("offset", 0.0), sym["tol"])
        if fixed:
            msg += ", square to the mirror plane"
    if meta.get("normal_flip"):                      # turned over by Align Normals: applied last
        net = np.asarray(net)[::-1]
    return net, orders, msg


def _seam_tolerance(Pa, Pb, settings):
    pts = np.concatenate([Pa.reshape(-1, 3), Pb.reshape(-1, 3)])
    return max(np.ptp(pts, axis=0).max() * settings.gap_tolerance, 1e-6)


def apply_match(net, orders, ref_net, ref_orders, continuity, settings):
    shared = find_shared_edge(net, ref_net, _seam_tolerance(net, ref_net, settings),
                              orders, ref_orders)
    if shared is None:
        raise ValueError("no edge lies close to the reference patch "
                         "(raise Gap Tolerance or move the patches together)")
    return match_net(net, ref_net, *shared[:3], continuity, orders, ref_orders)


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


class SS_OT_loft(bpy.types.Operator):
    bl_idname = "spline_surface.loft"
    bl_label = "Curves → Loft"
    bl_description = ("Loft a patch through 2 or more section curves (ordered and oriented "
                      "automatically; up to 6 sections are hit exactly)")
    bl_options = {'REGISTER', 'UNDO'}

    rows: bpy.props.IntProperty(
        name="Extra Rows", default=2, min=0, max=4,
        description="Control point rows added between the sections without changing the shape "
                    "(more points to shape with; limited by the Degree setting)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.spline_surface
        sources = [o for o in context.selected_objects if o.type in BOUNDARY_TYPES]
        if len(sources) < 2:
            self.report({'ERROR'}, "Select 2 or more section curves (got %d)" % len(sources))
            return {'CANCELLED'}
        try:
            net, err, ordered = compute_loft(sources, context.evaluated_depsgraph_get(), settings,
                                             rows=self.rows)
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
        try:
            tnet, torders = world_net(target)
            net, orders = apply_match(tnet, torders, *world_net(ref), self.continuity, settings)
            replace_net(target, net, settings, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if meta is None or hand_edited_net(meta, tnet, torders):
            # Keep the hand-made shape as the recipe so Regenerate does not undo it.
            old_matches = meta.get("matches", []) if meta else []
            meta = raw_meta(tnet, torders, matches=old_matches)
        matches = [m for m in meta.get("matches", []) if m["ref"] != ref.name]
        matches.append({"ref": ref.name, "continuity": self.continuity})
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
        dg = context.evaluated_depsgraph_get()
        for ob in context.visible_objects:
            if ob.type == 'CURVE' and ob.data.splines:
                try:
                    self._edges[(ob.name, 'CURVE')] = read_polyline(ob, dg, 48)
                except ValueError:
                    pass
        if len(self._edges) < 3:
            self.report({'ERROR'}, "Need at least three open edges or curves")
            return {'CANCELLED'}
        self._chosen = []
        self._hover = None
        self._preview = []
        self._picker = pick.Overlay(context, self._draw_lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        context.area.header_text_set(
            "Fill Hole: click the edges around the hole (%d picked)   Enter: fill   C: %s   "
            "Wheel: tension %.2f   Esc/RMB: cancel"
            % (len(self._chosen), self.continuity, self.tension))

    def _label(self):
        return pick.edge_label(self._hover, self._edges[self._hover]) if self._hover else None

    def _draw_lines(self):
        lines = [(pts, pick.COLOR_EDGE, 2.0) for k, pts in self._edges.items()
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
