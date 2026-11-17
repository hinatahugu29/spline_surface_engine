"""Seam colours: every seam between visible patches drawn in its continuity grade.

G2 green, G1 yellow, G0 red, "open" (edges close but apart) magenta. The
seams are recomputed shortly after any patch changes while the display is on.
"""
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .core.diagnose import drop_intended
from .recipe import edge_polyline, world_net

COLORS = {'G2': (0.25, 0.9, 0.35, 1.0), 'G1': (1.0, 0.85, 0.2, 1.0),
          'G0': (1.0, 0.25, 0.2, 1.0), 'open': (0.9, 0.3, 0.9, 1.0)}
GRADES = ('G2', 'G1', 'G0', 'open')

_state = {"handle": None, "seams": [], "dirty": False, "counts": {}}
# measured pairs, keyed by both nets' bytes: an edit re-measures only the pairs it touched
_pairs = {}


def _key(net):
    return hash(net.tobytes())


def _contact(Pa, oa, ea, Pb, ob_, eb, tol, samples=120):
    """The stretch of edge ea of patch a lying on edge eb of patch b (world points)."""
    pa = edge_polyline(Pa, oa, ea, samples)
    pb = edge_polyline(Pb, ob_, eb, 241)
    a0, ab = pb[:-1], pb[1:] - pb[:-1]
    f = np.clip(((pa[:, None] - a0[None]) * ab[None]).sum(2)
                / np.maximum((ab * ab).sum(1), 1e-30)[None], 0.0, 1.0)
    d = np.linalg.norm(pa[:, None] - (a0[None] + f[..., None] * ab[None]), axis=2).min(1)
    on = np.flatnonzero(d <= tol)
    return pa[on.min():on.max() + 1] if len(on) >= 2 else pa


def shared_seams(context, among=None):
    """[dict(a, b, edge_a, edge_b, grade, measure, polyline)] for visible patches.

    Measured as Check Continuity does (seams along part of an edge included),
    without the seams Diagnose knows are apart on purpose (two patches a
    fillet trimmed apart, a split fill's piece against a whole edge). With
    `among` (names), only seams with one of those patches. `polyline` follows
    edge_a in world space (only the touching stretch along part of an edge).
    """
    from .matching import measure_pair, recipes
    settings = context.scene.spline_surface
    nets = {}
    for ob in context.visible_objects:
        if ob.type == 'SURFACE':
            try:
                nets[ob.name] = world_net(ob)
            except ValueError:
                pass
    keys = {n: _key(P) for n, (P, _) in nets.items()}
    names = sorted(nets)
    rows = []
    seen = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if among is not None and a not in among and b not in among:
                continue
            k = (a, b, keys[a], keys[b], settings.gap_tolerance)
            seen.add(k)
            if k not in _pairs:
                _pairs[k] = measure_pair(a, b, nets, settings)
            if _pairs[k] is not None:
                rows.append(_pairs[k])
    for k in [k for k in _pairs if k not in seen and (among is None or k[0] in among or k[1] in among)]:
        del _pairs[k]
    seams = []
    for r in drop_intended(rows, recipes({r["a"] for r in rows} | {r["b"] for r in rows})):
        (Pa, oa), (Pb, ob_) = nets[r["a"]], nets[r["b"]]
        m = {k: r[k] for k in ("gap", "angle", "curvature", "size")}
        line = (_contact(Pa, oa, r["edge_a"], Pb, ob_, r["edge_b"], 2e-3 * r["size"])
                if r.get("partial") else edge_polyline(Pa, oa, r["edge_a"], 48))
        seams.append({"a": r["a"], "b": r["b"], "edge_a": r["edge_a"], "edge_b": r["edge_b"],
                      "grade": r["grade"], "measure": m, "polyline": line,
                      "partial": bool(r.get("partial"))})
    return seams


def compute_seams(context):
    """[(grade, polyline, measure, (name_a, name_b))] for visible patches."""
    return [(s["grade"], s["polyline"], s["measure"], (s["a"], s["b"]))
            for s in shared_seams(context)]


def refresh(context):
    _state["seams"] = compute_seams(context)
    counts = {g: 0 for g in GRADES}
    for g, *_ in _state["seams"]:
        counts[g] += 1
    _state["counts"] = counts
    _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _draw():
    if not _state["seams"]:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", 4.0)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    for g, pts, _, _ in _state["seams"]:
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": [tuple(p) for p in pts]})
        shader.uniform_float("color", COLORS[g])
        batch.draw(shader)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def _timer():
    if _state["handle"] is None:
        return None
    if _state["dirty"]:
        try:
            refresh(bpy.context)
        except Exception:  # never let a timer die on a half-edited scene
            _state["dirty"] = False
    return 0.4


def _on_depsgraph(scene, depsgraph):
    if _state["handle"] is None:
        return
    for u in depsgraph.updates:
        idb = getattr(u.id, "original", u.id)
        if isinstance(idb, (bpy.types.Object, bpy.types.Curve)):
            _state["dirty"] = True
            return


def show(context, on):
    if on and _state["handle"] is None:
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
        refresh(context)
        bpy.app.timers.register(_timer, first_interval=0.4)
    elif not on and _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
        _state["seams"] = []
        refresh(context)


def counts():
    return _state["counts"] if _state["handle"] is not None else {}


class SS_OT_seam_colors(bpy.types.Operator):
    bl_idname = "spline_surface.seam_colors"
    bl_label = "Seam Colors"
    bl_description = ("Colour every seam between visible patches by continuity: "
                      "G2 green, G1 yellow, G0 red, open magenta. Updates as you edit")
    bl_options = {'REGISTER'}

    def execute(self, context):
        show(context, _state["handle"] is None)
        return {'FINISHED'}


classes = (SS_OT_seam_colors,)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
