"""Rebuilding a patch from its recipe: the compute_* builders, the recipe
registry (RECIPES, @recipe) and compute_from_meta, plus Match Edge's seam
helpers that every rebuild applies last.

No operators live here; operators.py and the tool modules call into this one.
"""
import json

import bpy
import numpy as np

from .bridge.read_curves import read_polyline
from .core.bezier import fit_curve
from .core.coons import close_loop, coons_net, loop_to_sides, orient_loop, snap_corners
from .core.blend import blend_curve_net, blend_net
from .core.corner import corner_net
from .core.fill import fill_patches, side_from_curve, side_from_patch
from .core.fillet import fillet_net, multi_fillet
from .core.flange import flange_net
from .core.sweep import sweep_net
from .core.mirror import mirror_net, symmetrize
from .core.network import build_network
from .core.offset import offset_net
from .core.continuity import edge_pair_gap, find_shared_edge, match_net
from .core.gap import even_gap_net
from .core.extend import bezier_orders, extend_net
from .core.loft import add_rows, loft_net, order_sections
from .recipe import META_KEY, META_VERSION, NET_OVERRIDE, edge_polyline, world_net

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
    if meta.get("rotate"):
        return sweep_net(*args, rotate=True)
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


RECIPES = {}


def recipe(method):
    """Register how a recipe `method` is rebuilt.

    The function takes (meta, sources, context, settings) and returns
    (net, orders, message); orders may be None for a plain Bezier net. A new
    kind of patch registers itself here and needs no other change to the
    rebuild (Live Update, Regenerate, gizmos and Mirror all go through
    compute_from_meta)."""
    def register(fn):
        RECIPES[method] = fn
        return fn
    return register


@recipe("coons")
def _rebuild_coons(meta, sources, context, settings):
    net, err = compute_coons(sources, context.evaluated_depsgraph_get(), settings)
    return net, None, "max fit error %.3g" % err


@recipe("loft")
def _rebuild_loft(meta, sources, context, settings):
    net, err, _ = compute_loft(sources, context.evaluated_depsgraph_get(), settings,
                               keep_order=True, rows=meta.get("rows", 0))
    return net, None, "max fit error %.3g" % err


@recipe("extend")
def _rebuild_extend(meta, sources, context, settings):
    net, orders = compute_extend(sources[0], meta, settings)
    return net, orders, meta["continuity"]


@recipe("flange")
def _rebuild_flange(meta, sources, context, settings):
    net, orders = compute_flange(sources[0], meta)
    return net, orders, "flange %.4g at %.4g deg" % (meta["length"], meta["angle"])


@recipe("sweep")
def _rebuild_sweep(meta, sources, context, settings):
    net, orders = compute_sweep(sources, meta)
    return net, orders, "sweep along %s" % sources[1].name


@recipe("corner")
def _rebuild_corner(meta, sources, context, settings):
    net, orders = compute_corner(sources, meta, settings)
    return net, orders, meta["continuity"]


@recipe("fill")
def _rebuild_fill(meta, sources, context, settings):
    parts = compute_fill(meta, settings)
    part = meta.get("part", 0)
    if part >= len(parts):
        raise ValueError("the hole now has a different number of sides")
    net, orders = parts[part]
    return net, orders, meta["continuity"]


@recipe("blend")
def _rebuild_blend(meta, sources, context, settings):
    net, orders, _ = compute_blend(sources, meta, settings)
    return net, orders, meta["continuity"]


@recipe("fillet")
def _rebuild_fillet(meta, sources, context, settings):
    net, orders, info = compute_fillet(sources, meta, settings)
    return net, orders, "fillet r=%.4g (reach %.3g-%.3g)" % (
        meta["radius"], info["reach"].min(), info["reach"].max())


@recipe("fillet_trim")
def _rebuild_fillet_trim(meta, sources, context, settings):
    net, orders = compute_fillet_trim(meta, context)
    return net, orders, "trimmed by a r=%.4g fillet" % meta["fillet"]["radius"]


@recipe("network")
def _rebuild_network(meta, sources, context, settings):
    cells, info = compute_network(sources, meta, settings, context.evaluated_depsgraph_get())
    part = meta.get("part", 0)
    if part >= len(cells):
        raise ValueError("the curve network now has a different number of cells")
    _, _, net, orders = cells[part]
    return net, orders, "network cell %d/%d" % (part + 1, len(cells))


@recipe("offset")
def _rebuild_offset(meta, sources, context, settings):
    src_net, src_orders = world_net(sources[0])
    net, orders, err = offset_net(src_net, src_orders, meta["distance"], meta.get("refine", 2))
    return net, orders, "offset %.4g (max deviation %.2g)" % (meta["distance"], err)


@recipe("mirror")
def _rebuild_mirror(meta, sources, context, settings):
    net, orders = world_net(sources[0])
    net = mirror_net(net, meta["axis"], meta.get("offset", 0.0))
    return net, orders, "mirror of %s" % sources[0].name


@recipe("raw")
def _rebuild_raw(meta, sources, context, settings):
    return np.array(meta["net"]), tuple(meta["orders"]), "stored net"


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
    build = RECIPES.get(meta["method"])
    if build is None:
        raise ValueError("unknown method %r" % meta["method"])
    net, orders, msg = build(meta, sources, context, settings)
    orders = orders or bezier_orders(net)
    for match in meta.get("matches", []):
        ref = bpy.data.objects.get(match["ref"])
        if ref is None:
            raise ValueError("Matched patch missing: %s" % match["ref"])
        net, orders = apply_match(net, orders, *world_net(ref), match["continuity"], settings,
                                  match)
        msg += ", %s to %s" % (match["continuity"], ref.name)
    gap = meta.get("gap")
    if gap:
        net, orders = apply_gap(net, orders, gap)
        msg += ", %.4g gap to %s" % (gap["target"], gap["ref"])
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


MATCH_PAIR_KEYS = ("edge_t", "edge_r", "reversed")


def match_pair(net, orders, ref_net, ref_orders, settings, match=None):
    """(edge_t, edge_r, reversed) a match works on, or None if it can't tell.

    The edge pair is remembered in the match the first time it is found. A
    remembered pair that still lies close is used as is; otherwise the closest
    pair within Gap Tolerance (after a split, say, the remembered names may be
    stale); otherwise the remembered pair again: the reference has moved
    further away than the tolerance, and the seam must follow it anyway
    (searching alone, the match was lost once a source moved ~10% away)."""
    tol = _seam_tolerance(net, ref_net, settings)
    rec = None
    if match is not None and all(k in match for k in MATCH_PAIR_KEYS):
        rec = (match["edge_t"], match["edge_r"], bool(match["reversed"]))
        if edge_pair_gap(net, ref_net, *rec, orders, ref_orders) <= tol:
            return rec
    shared = find_shared_edge(net, ref_net, tol, orders, ref_orders)
    return tuple(shared[:3]) if shared is not None else rec


def apply_match(net, orders, ref_net, ref_orders, continuity, settings, match=None,
                spread=None):
    """Match `net` to `ref_net`. With a recipe `match` entry, its edge pair is
    used and recorded in it, and its "spread" flag read (old entries: off)."""
    pair = match_pair(net, orders, ref_net, ref_orders, settings, match)
    if pair is None:
        raise ValueError("no edge lies close to the reference patch "
                         "(raise Gap Tolerance or move the patches together)")
    if spread is None:
        spread = bool(match.get("spread", False)) if match is not None else False
    out = match_net(net, ref_net, *pair, continuity, orders, ref_orders, spread=spread)
    if match is not None:
        match.update(zip(MATCH_PAIR_KEYS, pair))
    return out


def apply_gap(net, orders, gap):
    """Even Gap's recipe step: run `gap["edge"]` of the patch `gap["target"]`
    away from edge `gap["edge_ref"]` of patch `gap["ref"]` (flush if asked)."""
    ref = bpy.data.objects.get(gap["ref"])
    if ref is None:
        raise ValueError("Gap partner missing: %s" % gap["ref"])
    rnet, rorders = world_net(ref)
    line = edge_polyline(rnet, rorders, gap["edge_ref"], 64)
    net, orders, _ = even_gap_net(net, orders, gap["edge"], line, gap["target"],
                                  flush=gap.get("flush", True), reach=gap.get("reach"))
    return net, orders
