"""Patch groups: dependency tracking, chained regeneration and live update.

Every generated patch stores its recipe (sources + matches). Read together
those recipes form a dependency graph; a connected component is a patch
group. When a source changes, everything downstream is rebuilt in dependency
order so the seams stay exact. Patches edited by hand are left alone.
"""
import json
import traceback

import bpy
import numpy as np

from .operators import compute_from_meta
from .recipe import (ADDON_WRITES, META_KEY, ROUND_DECIMALS, hand_edited, replace_net,
                     store_meta, world_net)


def _meta(ob):
    try:
        return json.loads(ob[META_KEY])
    except (KeyError, ValueError, TypeError):
        return None


def _inputs(meta):
    """Names of the objects a recipe reads."""
    names = list(meta.get("sources", []))
    names += [m["ref"] for m in meta.get("matches", [])]
    if meta.get("target", {}).get("source"):
        names.append(meta["target"]["source"])
    return names


_graph_cache = {"key": None, "graph": {}}


def dependency_graph():
    """{object name: set of names that depend on it} over all patches.

    Recipes are only parsed again when some recipe string changed, which keeps
    the per-edit cost of live update independent of how many patches exist.
    """
    recipes = [(ob.name, ob[META_KEY]) for ob in bpy.data.objects if META_KEY in ob]
    key = hash(tuple(recipes))
    if _graph_cache["key"] == key:
        return _graph_cache["graph"]
    down = {}
    for name, raw in recipes:
        try:
            meta = json.loads(raw)
        except (ValueError, TypeError):
            continue
        for src in _inputs(meta):
            down.setdefault(src, set()).add(name)
    _graph_cache["key"], _graph_cache["graph"] = key, down
    return down


def downstream_order(names, down=None):
    """(patches depending on `names` sources first, patches caught in a loop)."""
    down = dependency_graph() if down is None else down
    reach, stack = set(), list(names)
    while stack:
        for dep in down.get(stack.pop(), ()):
            if dep not in reach:
                reach.add(dep)
                stack.append(dep)
    # Kahn's algorithm restricted to the reached sub-graph.
    indeg = {n: 0 for n in reach}
    for src, deps in down.items():
        if src in reach:
            for d in deps:
                indeg[d] += 1
    order = []
    ready = sorted(n for n, k in indeg.items() if k == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for d in sorted(down.get(n, ())):
            if d in indeg:
                indeg[d] -= 1
                if indeg[d] == 0:
                    ready.append(d)
    # Patches on a dependency loop (e.g. a patch matched to its own extension)
    # have no consistent order; callers skip them.
    return order, sorted(reach - set(order))


def connected(names):
    """All objects linked to `names` through recipes, in either direction."""
    down = dependency_graph()
    up = {}
    for src, deps in down.items():
        for d in deps:
            up.setdefault(d, set()).add(src)
    seen, stack = set(names), list(names)
    while stack:
        n = stack.pop()
        for m in down.get(n, set()) | up.get(n, set()):
            if m not in seen and m in bpy.data.objects:
                seen.add(m)
                stack.append(m)
    return seen


def off_scene_inputs(names, scene):
    """Inputs of the patches `names` that exist but are not in `scene`.

    A recipe names its sources; if those objects live in another scene (e.g. a
    full scene copy made by another add-on), editing the look-alike in this
    scene never reaches the patch, and nothing else would say why."""
    out = set()
    for name in names:
        ob = bpy.data.objects.get(name)
        meta = _meta(ob) if ob is not None else None
        for src in _inputs(meta or {}):
            sob = bpy.data.objects.get(src)
            if sob is not None and scene.objects.get(src) is None:
                out.add(src)
    return sorted(out)


def _write_in_place(ob, net, orders):
    """Move the control points without rebuilding the datablock (fast path)."""
    cu = ob.data
    if len(cu.splines) != 1:
        return False
    sp = cu.splines[0]
    nu, nv = net.shape[:2]
    if (sp.point_count_u, sp.point_count_v, sp.order_u, sp.order_v) != (nu, nv) + tuple(orders):
        return False
    M = np.array(ob.matrix_world.inverted())
    local = net @ M[:3, :3].T + M[:3, 3]
    co = np.concatenate([local.transpose(1, 0, 2).reshape(-1, 3),
                         np.ones((nu * nv, 1))], axis=1)
    sp.points.foreach_set("co", co.ravel())
    cu.update_tag()
    ADDON_WRITES.add(ob.name)
    return True


def rebuild(ob, context, force=False):
    """Regenerate one patch. Returns (status, message); status in ok/skipped/error."""
    meta = _meta(ob)
    if meta is None:
        return "skipped", "no recipe"
    if not force and hand_edited(ob, meta):
        return "skipped", "edited by hand"
    try:
        net, orders, msg = compute_from_meta(meta, context)
        if not _write_in_place(ob, net, orders):
            replace_net(ob, net, context.scene.spline_surface, orders)
    except ValueError as e:
        return "error", str(e)
    store_meta(ob, meta)
    return "ok", msg


def update_chain(names, context):
    """Rebuild everything downstream of `names`. Returns {status: [object names]}.

    A patch whose rebuild came out identical stops the chain there: what only
    reads it is listed under "unchanged" and left alone.
    """
    result = {"ok": [], "skipped": [], "error": [], "unchanged": []}
    blocked = set()
    changed = set(names)
    order, looped = downstream_order(names, dependency_graph())
    result["error"] += looped
    for name in order:
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        meta = _meta(ob) or {}
        if blocked & set(_inputs(meta)):
            blocked.add(name)
            result["skipped"].append(name)
            continue
        if not changed & set(_inputs(meta)):
            # nothing it reads moved (an upstream rebuild gave the same shape)
            result["unchanged"].append(name)
            continue
        before = _signature(ob)
        status, _ = rebuild(ob, context)
        result[status].append(name)
        if status != "ok":
            blocked.add(name)
        elif _signature(ob) != before:
            changed.add(name)
    return result


# ---------------------------------------------------------------- renames

# Recipe fields that hold settings, never object names.
_NOT_NAMES = {"method", "edge", "edges", "continuity", "kind", "axis", "fp", "ver",
              "layout", "net", "orders"}
_uids = {}      # object name -> session_uid, for every object some recipe reads
_missing = set()    # recipe inputs that are gone and could not be traced


def _rename_in(value, old, new, key=None):
    if key in _NOT_NAMES:
        return value, False
    if isinstance(value, str):
        return (new, True) if value == old else (value, False)
    if isinstance(value, list):
        out, hit = [], False
        for v in value:
            v, h = _rename_in(v, old, new)
            out.append(v)
            hit |= h
        return out, hit
    if isinstance(value, dict):
        out, hit = {}, False
        for k, v in value.items():
            v, h = _rename_in(v, old, new, k)
            out[k] = v
            hit |= h
        return out, hit
    return value, False


def rename_input(old, new):
    """Point every recipe that reads `old` at `new`. Returns the patches changed."""
    changed = []
    for ob in bpy.data.objects:
        if META_KEY not in ob:
            continue
        meta = _meta(ob)
        if meta is None:
            continue
        meta, hit = _rename_in(meta, old, new)
        if hit:
            ob[META_KEY] = json.dumps(meta)   # the fingerprint stays: the shape did not change
            changed.append(ob.name)
    if old in _state["fps"]:
        _state["fps"][new] = _state["fps"].pop(old)
    return changed


def follow_renames():
    """Rewrite recipes whose inputs were renamed; remember the ones that vanished.

    Recipes name their inputs, so renaming a curve would cut it loose without a
    word. Each input's session_uid is remembered; when a name disappears, the
    object still carrying that uid under a new name takes its place."""
    down = dependency_graph()
    by_uid = None
    gone = []
    for name in list(down):
        ob = bpy.data.objects.get(name)
        if ob is not None:
            _uids[name] = ob.session_uid
        else:
            gone.append(name)
    renamed = []
    still = set()
    for name in gone:
        uid = _uids.get(name)
        if uid is not None:
            if by_uid is None:
                by_uid = {o.session_uid: o for o in bpy.data.objects}
            ob = by_uid.get(uid)
            if ob is not None and ob.name != name:
                rename_input(name, ob.name)
                _uids.pop(name, None)
                _uids[ob.name] = uid
                renamed.append((name, ob.name))
                continue
        still.add(name)
    _missing.clear()
    _missing.update(still)
    return renamed


def missing_inputs(names):
    """Inputs of the patches `names` that no longer exist."""
    out = set()
    for name in names:
        ob = bpy.data.objects.get(name)
        meta = _meta(ob) if ob is not None else None
        out.update(n for n in _inputs(meta or {}) if n not in bpy.data.objects)
    return sorted(out)


# ---------------------------------------------------------------- live update

_state = {"busy": False, "fps": {}}


def _signature(ob):
    """Cheap fingerprint of what a dependent would read from `ob`."""
    try:
        if ob.type == 'SURFACE':
            net, _ = world_net(ob)
            return hash(np.round(net, ROUND_DECIMALS).tobytes())
        if ob.type == 'CURVE':
            pts = []
            for sp in ob.data.splines:
                for p in sp.bezier_points:
                    pts += [*p.co, *p.handle_left, *p.handle_right]
                for p in sp.points:
                    pts += [*p.co]
            return hash((tuple(np.round(pts, ROUND_DECIMALS)),
                        tuple(np.round(np.array(ob.matrix_world), ROUND_DECIMALS).ravel())))
    except ValueError:
        return None
    return None


def _on_depsgraph(scene, depsgraph):
    settings = getattr(scene, "spline_surface", None)
    if _state["busy"] or settings is None:
        return
    ctx = bpy.context
    if ctx.mode != 'OBJECT':
        return
    # Only objects the depsgraph reports as updated can have changed; sources
    # seen for the first time just get a baseline signature.
    touched = set()
    for u in depsgraph.updates:
        idb = getattr(u.id, "original", u.id)
        if isinstance(idb, bpy.types.Object):
            touched.add(idb.name)
        elif isinstance(idb, bpy.types.Curve):
            touched.update(o.name for o in bpy.data.objects if o.data == idb)
    from . import linked
    _state["busy"] = True
    missing_before = set(_missing)
    try:
        linked.on_update(touched, ctx)
        for old, new in follow_renames():
            touched.add(new)
            print("[Spline Surface] recipes now read %s (was %s)" % (new, old))
    except Exception:
        # Never let a geometry-math bug escape the handler: an uncaught
        # exception here reprints on every future scene edit.
        traceback.print_exc()
    finally:
        _state["busy"] = False
    if _missing != missing_before:
        settings.live_note = ("sources missing (renamed in another way or deleted): "
                              + ", ".join(sorted(_missing))) if _missing else ""
    if not settings.live_update:
        return
    down = dependency_graph()
    if not down:
        return
    changed = []
    for name in down:
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        if name in _state["fps"] and name not in touched:
            continue
        sig = _signature(ob)
        if _state["fps"].get(name) != sig:
            if name in _state["fps"]:
                changed.append(name)
            _state["fps"][name] = sig
    if not changed:
        return
    _state["busy"] = True
    try:
        result = update_chain(changed, ctx)
        for name in result["ok"]:
            ob = bpy.data.objects.get(name)
            if ob is not None:
                _state["fps"][name] = _signature(ob)
        notes = []
        if result["skipped"]:
            notes.append("not updated (hand edited): " + ", ".join(result["skipped"]))
        if result["error"]:
            notes.append("failed or looped: " + ", ".join(result["error"]))
        if _missing:
            notes.append("sources missing: " + ", ".join(sorted(_missing)))
        away = off_scene_inputs(result["ok"] + result["unchanged"], scene)
        if away:
            notes.append("sources not in this scene: " + ", ".join(away))
        settings.live_note = "; ".join(notes)
    except Exception:
        traceback.print_exc()
        settings.live_note = "live update failed (see console)"
    finally:
        _state["busy"] = False


def _reset(*_):
    _state["busy"] = False      # a stuck flag would leave Live Update off until Blender restarts
    _state["fps"].clear()
    from . import linked
    linked.reset()


def resync(context):
    """Rebuild every live patch from its sources.

    Blender stores an undo step when the transform ends, before Live Update
    reshapes the patches, so each step holds the sources one edit ahead of
    their patches. After undo/redo the patches are brought back in line."""
    settings = getattr(context.scene, "spline_surface", None)
    if settings is None or not settings.live_update or context.mode != 'OBJECT':
        return None
    down = dependency_graph()
    names = [n for n in down if n in bpy.data.objects]
    if not names:
        return None
    _state["busy"] = True
    try:
        return update_chain(names, context)
    except Exception:
        traceback.print_exc()
        return None
    finally:
        _state["busy"] = False


def _on_load(*_):
    # session_uids belong to the previous file
    _uids.clear()
    _missing.clear()


def _on_undo(*_):
    _reset()
    resync(bpy.context)


# ---------------------------------------------------------------- operators

class SS_OT_regenerate_all(bpy.types.Operator):
    bl_idname = "spline_surface.regenerate_all"
    bl_label = "Regenerate Group"
    bl_description = ("Rebuild every patch connected to the selection, sources first. "
                      "Hand-edited patches are skipped unless Force is on")
    bl_options = {'REGISTER', 'UNDO'}

    force: bpy.props.BoolProperty(
        name="Force", default=False,
        description="Also rebuild hand-edited patches (their edits are discarded)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and context.selected_objects

    def execute(self, context):
        follow_renames()
        group = connected([o.name for o in context.selected_objects])
        down = dependency_graph()
        roots = [n for n in group if not (_meta(bpy.data.objects[n]) or {}).get("sources")
                 and not (_meta(bpy.data.objects[n]) or {}).get("matches")]
        order, looped = downstream_order(roots or group, down)
        done, skipped, failed = 0, [], ["%s (dependency loop)" % n for n in looped]
        _state["busy"] = True
        try:
            for name in order:
                status, msg = rebuild(bpy.data.objects[name], context, self.force)
                if status == "ok":
                    done += 1
                elif status == "skipped":
                    skipped.append(name)
                else:
                    failed.append("%s (%s)" % (name, msg))
        finally:
            _state["busy"] = False
            _reset()
        away = off_scene_inputs(order, context.scene)
        gone = missing_inputs(order)
        if gone:
            failed.append("sources missing (deleted, or renamed before this session): "
                          + ", ".join(gone))
        if failed:
            self.report({'ERROR'}, "Rebuilt %d, failed: %s" % (done, ", ".join(failed)))
        elif away:
            self.report({'WARNING'}, "Rebuilt %d, but these sources are in another scene "
                        "(edits here will not reach the patches): %s" % (done, ", ".join(away)))
        elif skipped:
            self.report({'WARNING'}, "Rebuilt %d, skipped hand-edited: %s"
                        % (done, ", ".join(skipped)))
        else:
            self.report({'INFO'}, "Rebuilt %d patch(es)" % done)
        return {'FINISHED'}


class SS_OT_select_group(bpy.types.Operator):
    bl_idname = "spline_surface.select_group"
    bl_label = "Select Group"
    bl_description = "Select every patch and source connected to the selection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and context.selected_objects

    def execute(self, context):
        group = connected([o.name for o in context.selected_objects])
        count = 0
        for name in group:
            ob = context.view_layer.objects.get(name)
            if ob is not None and ob.visible_get():
                ob.select_set(True)
                count += 1
        self.report({'INFO'}, "Selected %d object(s)" % count)
        return {'FINISHED'}


classes = (SS_OT_regenerate_all, SS_OT_select_group)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    if _reset not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_reset)
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    for hs in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _on_undo not in hs:
            hs.append(_on_undo)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    if _reset in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_reset)
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    for hs in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _on_undo in hs:
            hs.remove(_on_undo)
    _reset()
