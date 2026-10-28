"""Patch groups: dependency tracking, chained regeneration and live update.

Every generated patch stores its recipe (sources + matches). Read together
those recipes form a dependency graph; a connected component is a patch
group. When a source changes, everything downstream is rebuilt in dependency
order so the seams stay exact. Patches edited by hand are left alone.
"""
import json

import bpy
import numpy as np

from .operators import (ADDON_WRITES, META_KEY, compute_from_meta, hand_edited, replace_net,
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
    return names


def dependency_graph():
    """{object name: set of names that depend on it} over all patches."""
    down = {}
    for ob in bpy.data.objects:
        meta = _meta(ob) if META_KEY in ob else None
        if meta is None:
            continue
        for name in _inputs(meta):
            down.setdefault(name, set()).add(ob.name)
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
    """Rebuild everything downstream of `names`. Returns {status: [object names]}."""
    result = {"ok": [], "skipped": [], "error": []}
    blocked = set()
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
        status, _ = rebuild(ob, context)
        result[status].append(name)
        if status != "ok":
            blocked.add(name)
    return result


# ---------------------------------------------------------------- live update

_state = {"busy": False, "fps": {}}


def _signature(ob):
    """Cheap fingerprint of what a dependent would read from `ob`."""
    try:
        if ob.type == 'SURFACE':
            net, _ = world_net(ob)
            return hash(np.round(net, 6).tobytes())
        if ob.type == 'CURVE':
            pts = []
            for sp in ob.data.splines:
                for p in sp.bezier_points:
                    pts += [*p.co, *p.handle_left, *p.handle_right]
                for p in sp.points:
                    pts += [*p.co]
            return hash((tuple(np.round(pts, 6)), tuple(np.round(np.array(ob.matrix_world), 6).ravel())))
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
    try:
        linked.on_update(touched, ctx)
    finally:
        _state["busy"] = False
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
        settings.live_note = "; ".join(notes)
    finally:
        _state["busy"] = False


def _reset(*_):
    _state["fps"].clear()
    from . import linked
    linked.reset()


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
        if failed:
            self.report({'ERROR'}, "Rebuilt %d, failed: %s" % (done, ", ".join(failed)))
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
    for hs in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post,
               bpy.app.handlers.redo_post):
        if _reset not in hs:
            hs.append(_reset)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    for hs in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post,
               bpy.app.handlers.redo_post):
        if _reset in hs:
            hs.remove(_reset)
    _reset()
