"""Follow Map: who follows whom, drawn in the viewport and listed in the panel.

Every recipe input is a link "source -> patch built on it". Match Edge links
are drawn on the seam itself, with chevrons pointing into the patch that
follows; every other link is an arc from the source (the edge it was built
from, when the recipe names one) to the patch, with the arrowhead and a
label near the patch. Colours tell the state of the follower:

    following       cyan (Match) / violet (built from)
    hand edited     grey, dashed: Live Update leaves it alone
    failed          red: its last rebuild failed (reason in the panel)
    missing source  magenta
"""
import json

import blf
import bpy
from bpy.app.handlers import persistent
import gpu
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

from .core.continuity import find_shared_edge
from .core.nurbs import eval_points
from .recipe import META_KEY, edge_polyline, hand_edited, world_net

COLORS = {
    "match": (0.25, 0.85, 1.0, 0.95),
    "source": (0.78, 0.5, 1.0, 0.95),
    "hand": (0.82, 0.82, 0.82, 0.9),
    "failed": (1.0, 0.25, 0.2, 0.95),
    "missing": (0.95, 0.3, 0.9, 0.95),
}
METHOD_NAMES = {
    "coons": "Coons", "loft": "Loft", "extend": "Extend", "flange": "Flange",
    "sweep": "Sweep", "corner": "Corner Fill", "fill": "Fill Hole", "blend": "Bridge",
    "fillet": "Fillet", "fillet_trim": "Fillet trim", "network": "Curve Network",
    "offset": "Offset", "mirror": "Mirror", "raw": "Patch",
}
ARROW_PX = 13.0
LIST_LIMIT = 40

_state = {"handle": None, "text_handle": None, "dirty": False, "draw": [], "labels": []}


def _meta(ob):
    try:
        return json.loads(ob[META_KEY])
    except (KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------- links

def follower_state(ob, meta):
    """'ok', 'hand' or 'failed' for a patch built from a recipe (a link whose own
    source is gone is 'missing', see link_state)."""
    from .live import FAILED
    if ob.name in FAILED:
        return "failed"
    if hand_edited(ob, meta):
        return "hand"
    return "ok"


def links(names=None):
    """[dict(src, dst, kind, label, state, meta)] over the patches `names` (all if None).

    kind: 'match' (Match Edge), 'source' (built from), 'target' (a Sweep's stop)."""
    out = []
    obs = bpy.data.objects if names is None else [bpy.data.objects[n] for n in sorted(names)
                                                    if n in bpy.data.objects]
    for ob in obs:
        if META_KEY not in ob:
            continue
        meta = _meta(ob)
        if meta is None:
            continue
        state = follower_state(ob, meta)
        method = METHOD_NAMES.get(meta.get("method"), meta.get("method", "?"))
        for i, src in enumerate(meta.get("sources", [])):
            out.append({"src": src, "dst": ob.name, "kind": "source", "label": method,
                        "state": state, "meta": meta, "index": i})
        for m in meta.get("matches", []):
            label = "Match %s%s" % (m.get("continuity", "?"), " Spread" if m.get("spread") else "")
            out.append({"src": m.get("ref", "?"), "dst": ob.name, "kind": "match", "label": label,
                        "state": state, "meta": meta, "match": m})
        tgt = meta.get("target") or {}
        if tgt.get("source"):
            out.append({"src": tgt["source"], "dst": ob.name, "kind": "target",
                        "label": method + " target", "state": state, "meta": meta})
    return out


def scope_names(context):
    """The patches whose links the map shows (None: everything).

    SELECTED: the selected patches and the patches directly linked to them
    (links between two neighbours that are both unselected are left out by
    scope_links). GROUP: everything connected to the selection."""
    from .live import connected, dependency_graph
    settings = context.scene.spline_surface
    if settings.follow_scope == 'ALL':
        return None
    sel = {o.name for o in context.selected_objects}
    if not sel:
        return set()
    if settings.follow_scope == 'GROUP':
        return connected(sel)
    out = set(sel)
    for src, deps in dependency_graph().items():
        if src in sel:
            out |= deps
    return out


def scope_links(context):
    """links() limited to the current scope."""
    names = scope_names(context)
    rows = links(names)
    if context.scene.spline_surface.follow_scope == 'SELECTED':
        sel = {o.name for o in context.selected_objects}
        rows = [r for r in rows if r["src"] in sel or r["dst"] in sel]
    return rows


def link_state(link):
    return "missing" if link["src"] not in bpy.data.objects else link["state"]


# ---------------------------------------------------------------- geometry

def _surface(ob):
    if ob is None or ob.type != 'SURFACE':
        return None
    try:
        return world_net(ob)
    except ValueError:
        return None


def _anchor(ob, edge=None):
    """(point, normal or None) to hang an arrow on: an edge's middle, or the patch centre."""
    srf = _surface(ob)
    if srf is not None:
        net, orders = srf
        if edge:
            pl = edge_polyline(net, orders, edge, 9)
            return pl[4], None
        p = eval_points(net, orders, [0.5], [0.5])[0]
        du = eval_points(net, orders, [0.5], [0.5], 1, 0)[0]
        dv = eval_points(net, orders, [0.5], [0.5], 0, 1)[0]
        n = np.cross(du, dv)
        ln = np.linalg.norm(n)
        return p, (n / ln if ln > 1e-12 else None)
    if ob is not None and ob.type == 'CURVE':
        M = np.array(ob.matrix_world)
        pts = []
        for sp in ob.data.splines:
            pts += [tuple(b.co) for b in sp.bezier_points] + [tuple(p.co[:3]) for p in sp.points]
        if pts:
            pts = np.array(pts)
            pts = pts @ M[:3, :3].T + M[:3, 3]
            return pts[len(pts) // 2] if len(pts) > 2 else pts.mean(0), None
    if ob is not None:
        return np.array(ob.matrix_world.translation), None
    return None, None


def _size(ob):
    srf = _surface(ob)
    if srf is None:
        return 1.0
    return float(np.ptp(srf[0].reshape(-1, 3), axis=0).max()) or 1.0


def _inward(net, orders, edge, t, depth=0.12):
    """Points just inside the patch from `edge`, at the edge parameters t."""
    c = np.full(len(t), 1.0 - depth if edge[1] == '1' else depth)
    return eval_points(net, orders, c, t) if edge[0] == 'U' else eval_points(net, orders, t, c)


def _match_geometry(link):
    """(seam polyline, [(base, tip)] chevrons) for a Match link, or None if it can't tell."""
    fol, ref = bpy.data.objects.get(link["dst"]), bpy.data.objects.get(link["src"])
    fs, rs = _surface(fol), _surface(ref)
    if fs is None or rs is None:
        return None
    (net, orders), (rnet, rord) = fs, rs
    edge = link["match"].get("edge_t")
    if edge is None:
        size = np.ptp(np.concatenate([net.reshape(-1, 3), rnet.reshape(-1, 3)]), 0).max()
        sh = find_shared_edge(net, rnet, 1e-2 * size, orders, rord)
        if sh is None:
            return None
        edge = sh[0]
    pl = edge_polyline(net, orders, edge, 32)
    at = [6, 16, 25]
    tips = _inward(net, orders, edge, np.linspace(0.0, 1.0, 32)[at])
    return pl, list(zip(pl[at], tips))


def _arc(p0, p1, normal, lift_min):
    d = p1 - p0
    length = np.linalg.norm(d)
    lift = max(0.2 * length, lift_min)
    up = normal if normal is not None else np.array([0.0, 0.0, 1.0])
    if length > 1e-9:
        # Bulge sideways to the link, not along it (an Offset sits straight above its source).
        dh = d / length
        up = up - (up @ dh) * dh
        if np.linalg.norm(up) < 0.3:
            up = np.cross(dh, [0.0, 0.0, 1.0] if abs(dh[2]) < 0.9 else [1.0, 0.0, 0.0])
        up = up / np.linalg.norm(up)
    mid = 0.5 * (p0 + p1) + up * lift
    s = np.linspace(0.0, 1.0, 24)[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * mid + s * s * p1


def _source_edge(link):
    """The edge of the source the recipe built from, when it names one."""
    meta, i = link["meta"], link.get("index", 0)
    edges = meta.get("edges")
    if isinstance(edges, list) and i < len(edges) and isinstance(edges[i], str):
        return edges[i] if len(edges[i]) == 2 else None
    edge = meta.get("edge")
    if i == 0 and isinstance(edge, str) and len(edge) == 2:
        return edge
    return None


def compute(context):
    """Lines (3D) and arrowheads/labels (anchored in 3D) for the visible links."""
    visible = {o.name for o in context.visible_objects}
    draw, labels = [], []
    for link in scope_links(context):
        if link["dst"] not in visible:
            continue
        state = link_state(link)
        missing = state == "missing"
        color = COLORS[link["kind"] if link["kind"] == "match" else "source"] \
            if state == "ok" else COLORS[state]
        dashed = state == "hand"
        dst = bpy.data.objects[link["dst"]]
        text = link["label"] + {"ok": "", "hand": " (hand edited)", "failed": " (failed)",
                                "missing": " (missing %s)" % link["src"]}[state]
        if link["kind"] == "match" and not missing:
            geo = _match_geometry(link)
            if geo is not None:
                pl, chevrons = geo
                draw.append({"pts": pl, "color": color, "dashed": dashed, "width": 4.0,
                             "heads": chevrons})
                labels.append((pl[len(pl) // 2], text, color))
                continue
        p1, n1 = _anchor(dst)
        if p1 is None:
            continue
        if missing:
            labels.append((p1, text, color))
            continue
        p0, _ = _anchor(bpy.data.objects[link["src"]], _source_edge(link))
        if p0 is None:
            continue
        arc = _arc(p0, p1, n1, 0.25 * _size(dst))
        draw.append({"pts": arc, "color": color, "dashed": dashed, "width": 3.0,
                     "heads": [(arc[-5], arc[-3])]})
        labels.append((arc[12], text, color))
    _state["draw"], _state["labels"] = draw, labels


# ---------------------------------------------------------------- drawing

def _dashes(pts):
    segs = []
    for i in range(0, len(pts) - 1, 2):
        segs += [tuple(pts[i]), tuple(pts[i + 1])]
    return segs


def _draw_view():
    if not _state["draw"]:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    for item in _state["draw"]:
        shader.uniform_float("lineWidth", item["width"])
        shader.uniform_float("color", item["color"])
        if item["dashed"]:
            batch = batch_for_shader(shader, 'LINES', {"pos": _dashes(item["pts"])})
        else:
            batch = batch_for_shader(shader, 'LINE_STRIP',
                                     {"pos": [tuple(p) for p in item["pts"]]})
        batch.draw(shader)
    gpu.state.blend_set('NONE')


def _head_2d(region, rv3d, base, tip):
    """Two strokes of an arrowhead at `tip`, pointing from base to tip, in pixels."""
    a = location_3d_to_region_2d(region, rv3d, base)
    b = location_3d_to_region_2d(region, rv3d, tip)
    if a is None or b is None:
        return None
    a, b = np.array(a), np.array(b)
    d = b - a
    ln = np.linalg.norm(d)
    if ln < 1e-6:
        return None
    d /= ln
    n = np.array([-d[1], d[0]])
    back = b - d * ARROW_PX
    return [(*(back + n * ARROW_PX * 0.6), 0.0), (*b, 0.0), (*(back - n * ARROW_PX * 0.6), 0.0)]


def _draw_pixel():
    if not _state["draw"] and not _state["labels"]:
        return
    ctx = bpy.context
    region, rv3d = ctx.region, ctx.region_data
    if rv3d is None:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", 2.5)
    gpu.state.blend_set('ALPHA')
    for item in _state["draw"]:
        shader.uniform_float("color", item["color"])
        for base, tip in item["heads"]:
            stroke = _head_2d(region, rv3d, base, tip)
            if stroke:
                batch_for_shader(shader, 'LINE_STRIP', {"pos": stroke}).draw(shader)
    blf.size(0, 12)
    blf.enable(0, blf.SHADOW)
    blf.shadow(0, 3, 0.0, 0.0, 0.0, 0.85)
    blf.shadow_offset(0, 1, -1)
    for at, text, color in _state["labels"]:
        p = location_3d_to_region_2d(region, rv3d, at)
        if p is None:
            continue
        blf.position(0, p[0] + 6, p[1] + 4, 0)
        blf.color(0, *color[:3], 1.0)
        blf.draw(0, text)
    blf.disable(0, blf.SHADOW)
    gpu.state.blend_set('NONE')


def refresh(context):
    try:
        compute(context)
    finally:
        _state["dirty"] = False
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def is_on():
    return _state["handle"] is not None


def _timer():
    if _state["handle"] is None:
        return None
    if _state["dirty"]:
        try:
            refresh(bpy.context)
        except Exception:  # never let a timer die on a half-edited scene
            _state["dirty"] = False
    return 0.3


@persistent      # kept when another file is opened
def _on_depsgraph(scene, depsgraph):
    if _state["handle"] is not None:
        _state["dirty"] = True  # selection changes matter too (Group scope)


def mark_dirty(*_):
    _state["dirty"] = True


def show(context, on):
    if on and _state["handle"] is None:
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_view, (), 'WINDOW', 'POST_VIEW')
        _state["text_handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_pixel, (), 'WINDOW', 'POST_PIXEL')
        refresh(context)
        bpy.app.timers.register(_timer, first_interval=0.3, persistent=True)
    elif not on and _state["handle"] is not None:
        _remove_handlers()
        _state["draw"], _state["labels"] = [], []
        refresh(context)


def _remove_handlers():
    for key in ("handle", "text_handle"):
        if _state[key] is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_state[key], 'WINDOW')
            _state[key] = None


# ---------------------------------------------------------------- panel list

def list_rows(context):
    """Rows for the panel: (links shown, how many more). Followers in dependency order."""
    from .live import downstream_order
    rows = scope_links(context)
    if rows:
        order, looped = downstream_order([r["src"] for r in rows])
        rank = {n: i for i, n in enumerate(order + looped)}
        rows.sort(key=lambda r: (rank.get(r["dst"], -1), r["dst"], r["kind"] != "match", r["src"]))
    return rows[:LIST_LIMIT], max(0, len(rows) - LIST_LIMIT)


STATE_ICONS = {"ok": 'CHECKMARK', "hand": 'GREASEPENCIL', "failed": 'ERROR', "missing": 'GHOST_DISABLED'}


def draw_list(layout, context):
    settings = context.scene.spline_surface
    row = layout.row(align=True)
    row.operator("spline_surface.follow_map", text="Follow Arrows", icon='DECORATE_LINKED',
                 depress=is_on())
    row.prop(settings, "follow_scope", text="")
    if not settings.follow_list:
        layout.prop(settings, "follow_list", text="List Links", icon='DISCLOSURE_TRI_RIGHT',
                    emboss=False)
        return
    layout.prop(settings, "follow_list", text="List Links", icon='DISCLOSURE_TRI_DOWN',
                emboss=False)
    rows, more = list_rows(context)
    if not rows:
        layout.label(text="Nothing follows anything here" if settings.follow_scope == 'ALL'
                     else "Select a patch to list its links", icon='INFO')
        return
    from .live import FAILED
    col = layout.column(align=True)
    for r in rows:
        line = col.row(align=True)
        state = link_state(r)
        sub = line.row(align=True)
        sub.alert = state in ("failed", "missing")
        arrow = "⇒" if r["kind"] == "match" else "→"
        short = r["label"].replace(" Spread", "").replace("Match ", "")
        sub.label(text="%s %s %s  %s" % (r["src"], arrow, r["dst"], short),
                  icon=STATE_ICONS[state])
        op = line.operator("spline_surface.follow_jump", text="", icon='RESTRICT_SELECT_OFF')
        op.a, op.b = r["src"], r["dst"]
        if state == "failed" and r["dst"] in FAILED:
            col.label(text="    " + FAILED[r["dst"]][:80])
    if more:
        col.label(text="... and %d more" % more)


# ---------------------------------------------------------------- operators

class SS_OT_follow_map(bpy.types.Operator):
    bl_idname = "spline_surface.follow_map"
    bl_label = "Follow Arrows"
    bl_description = ("Show who follows whom: Match Edge links as chevrons on the seam, "
                      "patches built from sources as arcs. Grey dashed: edited by hand, "
                      "red: last rebuild failed, magenta: source missing")
    bl_options = {'REGISTER'}

    def execute(self, context):
        show(context, not is_on())
        return {'FINISHED'}


class SS_OT_follow_jump(bpy.types.Operator):
    bl_idname = "spline_surface.follow_jump"
    bl_label = "Go to Link"
    bl_description = "Select the follower (active) and what it follows, and bring them into view"
    bl_options = {'REGISTER', 'UNDO'}

    a: bpy.props.StringProperty()
    b: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        obs = [bpy.data.objects.get(n) for n in (self.a, self.b)]
        obs = [o for o in obs if o is not None and context.view_layer.objects.get(o.name)]
        if not obs:
            self.report({'ERROR'}, "Neither object is in this view layer any more")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        for o in obs:
            o.select_set(True)
        context.view_layer.objects.active = obs[-1]
        if context.area and context.area.type == 'VIEW_3D':
            bpy.ops.view3d.view_selected()
        return {'FINISHED'}


classes = (SS_OT_follow_map, SS_OT_follow_jump)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    _remove_handlers()
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
