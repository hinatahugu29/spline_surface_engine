"""Patch groups: dependency tracking, chained regeneration and live update.

Every generated patch stores its recipe (sources + matches). Read together
those recipes form a dependency graph; a connected component is a patch
group. When a source changes, everything downstream is rebuilt in dependency
order so the seams stay exact. Patches edited by hand are left alone.
"""
import json
import traceback

import bpy
from bpy.app.handlers import persistent
import numpy as np

from .rebuild import compute_from_meta
from .recipe import (ADDON_WRITES, META_KEY, ROUND_DECIMALS, hand_edited, replace_net,
                     store_meta, world_net)

from .graph import (FAILED, _inputs, _meta, _rename_in,  # noqa: F401
                    connected, dependency_graph, downstream_order,
                    off_scene_inputs, relink_scene_copies, scene_copy_relinks)


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


HAND_EDITED = "edited by hand"
POINT_COUNT = "point count changed"


def rebuild(ob, context, force=False, in_place_only=False, allow_new_data=True):
    """Regenerate one patch. Returns (status, message); status in
    ok/skipped/error/deferred.

    A patch whose point count changed needs a new Curve datablock. With
    `in_place_only` it is skipped; without `allow_new_data` it is "deferred"
    (Live Update's depsgraph handler runs after the undo step was stored:
    a datablock made there is missing from undo memory, and undoing leaves
    the patch empty or crashes Blender -- run_deferred() makes it from a
    timer and stores an undo step)."""
    meta = _meta(ob)
    if meta is None:
        return "skipped", "no recipe"
    if not force and hand_edited(ob, meta):
        return "skipped", HAND_EDITED
    try:
        net, orders, msg = compute_from_meta(meta, context)
        if not _write_in_place(ob, net, orders):
            if in_place_only:
                return "skipped", POINT_COUNT
            if not allow_new_data:
                return "deferred", POINT_COUNT
            replace_net(ob, net, context.scene.spline_surface, orders)
    except ValueError as e:
        FAILED[ob.name] = str(e)
        return "error", str(e)
    store_meta(ob, meta)
    FAILED.pop(ob.name, None)
    return "ok", msg


def update_chain(names, context, in_place_only=False, allow_new_data=True):
    """Rebuild everything downstream of `names`. Returns {status: [object names]}.

    A patch whose rebuild came out identical stops the chain there: what only
    reads it is listed under "unchanged" and left alone. Without
    `allow_new_data`, patches whose point count changed are listed under
    "deferred" (what reads them waits too, under "waiting") for run_deferred().
    """
    result = {"ok": [], "skipped": [], "error": [], "unchanged": [], "deferred": [],
              "waiting": [], "stale": []}
    blocked = set()
    waiting = set()
    changed = set(names)
    order, looped = downstream_order(names, dependency_graph())
    result["error"] += looped
    for name in looped:
        FAILED[name] = "dependency loop"
    for name in order:
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        meta = _meta(ob) or {}
        if waiting & set(_inputs(meta)):
            waiting.add(name)
            result["waiting"].append(name)
            continue
        if blocked & set(_inputs(meta)):
            blocked.add(name)
            result["skipped"].append(name)
            continue
        if not changed & set(_inputs(meta)):
            # nothing it reads moved (an upstream rebuild gave the same shape)
            result["unchanged"].append(name)
            continue
        before = _signature(ob)
        status, why = rebuild(ob, context, in_place_only=in_place_only,
                              allow_new_data=allow_new_data)
        result[status].append(name)
        if status == "deferred":
            waiting.add(name)
        elif status == "skipped" and why == POINT_COUNT:
            result["stale"].append(name)     # in place only (after an undo): left as it was
            blocked.add(name)
        elif status == "skipped" and why == HAND_EDITED:
            # kept as the user shaped it, which is what its dependents read: they
            # follow it (a mirror copy of a hand-moved panel, after an undo)
            changed.add(name)
        elif status != "ok":
            blocked.add(name)
        elif _signature(ob) != before:
            changed.add(name)
    return result


# ---------------------------------------------------------------- renames


_uids = {}      # object name -> session_uid, for every object some recipe reads
_missing = set()    # recipe inputs that are gone and could not be traced
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


@persistent
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
        # Inside the depsgraph handler only control points move: a patch whose
        # point count changed is finished from a timer (see rebuild()).
        result = update_chain(changed, ctx, allow_new_data=False)
        if result["deferred"]:
            _defer(result["deferred"])
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


def settle(context):
    """Rebuild what an operator changed before its undo step is stored.

    Otherwise Live Update runs from the depsgraph handler after the step, and a
    patch whose point count changed is rebuilt from a timer that stores a step
    of its own: the first Ctrl+Z then only undid that (Door refined, its mirror
    copy back to the old point count), or seemed to do nothing. Called by
    finish.py as each undoable operator of the add-on returns."""
    settings = getattr(context.scene, "spline_surface", None)
    if (settings is None or not settings.live_update or _state["busy"]
            or context.mode != 'OBJECT'):
        return None
    changed = []
    for name in dependency_graph():
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        sig = _signature(ob)
        if name in _state["fps"] and _state["fps"][name] != sig:
            changed.append(name)
        _state["fps"][name] = sig
    if not changed:
        return None
    _state["busy"] = True
    try:
        # still inside the operator: new datablocks are part of its undo step
        result = update_chain(changed, context)
        for name in result["ok"]:
            ob = bpy.data.objects.get(name)
            if ob is not None:
                _state["fps"][name] = _signature(ob)
        _deferred.difference_update(result["ok"])
        return result
    except Exception:
        traceback.print_exc()
        return None
    finally:
        _state["busy"] = False


_deferred = set()   # patches whose point count changed, rebuilt from a timer


def _defer(names):
    _deferred.update(names)
    if not bpy.app.timers.is_registered(_deferred_timer):
        bpy.app.timers.register(_deferred_timer, first_interval=0.05)


def _busy_elsewhere(context):
    """True while it is unsafe to run operators: a modal tool (a transform
    drag, a picker) is running or the file is in some edit mode."""
    if context.mode != 'OBJECT':
        return True
    wm = getattr(context, "window_manager", None)
    for win in getattr(wm, "windows", ()):
        if len(getattr(win, "modal_operators", ())):
            return True
    return False


def run_deferred(context):
    """Rebuild the deferred patches and what reads them, then store an undo
    step so undo memory knows the new datablocks. Returns the
    update_chain-style result, or None if nothing was waiting."""
    names = sorted(n for n in _deferred if n in bpy.data.objects)
    _deferred.clear()
    if not names:
        return None
    result = {"ok": [], "skipped": [], "error": [], "unchanged": [], "deferred": [],
              "waiting": [], "stale": []}
    _state["busy"] = True
    try:
        done = []
        for name in names:
            status, _ = rebuild(bpy.data.objects[name], context)
            result[status].append(name)
            if status == "ok":
                done.append(name)
        if done:
            down = update_chain(done, context)
            for key, vals in down.items():
                result[key] += vals
        for name in result["ok"]:
            ob = bpy.data.objects.get(name)
            if ob is not None:
                _state["fps"][name] = _signature(ob)
                ob.data.update_tag()
        if result["ok"]:
            _undo_push(context, "Spline Surface: Live Update")
    finally:
        _state["busy"] = False
    return result


def _undo_push(context, message):
    wm = getattr(context, "window_manager", None)
    wins = list(getattr(wm, "windows", ()))
    if not wins:
        return                              # background mode: no undo stack
    try:
        with context.temp_override(window=wins[0]):
            bpy.ops.ed.undo_push(message=message)
    except (RuntimeError, TypeError):
        traceback.print_exc()


def _deferred_timer():
    ctx = bpy.context
    if not _deferred:
        return None
    try:
        if _busy_elsewhere(ctx):
            return 0.2                     # wait for the drag or edit to end
        run_deferred(ctx)
    except Exception:
        traceback.print_exc()
        _deferred.clear()
    return None


@persistent
def _reset(*_):
    _state["busy"] = False      # a stuck flag would leave Live Update off until Blender restarts
    _state["fps"].clear()
    from . import linked
    linked.reset()


def resync(context):
    """Rebuild every live patch from its sources.

    Blender stores an undo step when the transform ends, before Live Update
    reshapes the patches, so each step holds the sources one edit ahead of
    their patches. After undo/redo the patches are brought back in line.

    Only control points are moved: rebuilding a datablock runs operators and
    adds/removes IDs, which right after an undo corrupts Blender's undo memory
    (the next Ctrl+Z crashed in curve_foreach_id)."""
    settings = getattr(context.scene, "spline_surface", None)
    if settings is None or not settings.live_update or context.mode != 'OBJECT':
        return None
    down = dependency_graph()
    names = [n for n in down if n in bpy.data.objects]
    if not names:
        return None
    _state["busy"] = True
    try:
        return update_chain(names, context, in_place_only=True)
    except Exception:
        traceback.print_exc()
        return None
    finally:
        _state["busy"] = False


def _prime():
    """Record what every source looks like now.

    The depsgraph handler rebuilds a patch when a source's fingerprint differs
    from the recorded one; a source seen for the first time only gets recorded.
    Without this, the first edit after opening a file (or enabling the add-on)
    became the baseline, and a single change - a value typed in the sidebar, one
    snapped move - was not followed until the next one."""
    try:
        for name in dependency_graph():
            ob = bpy.data.objects.get(name)
            if ob is not None:
                _state["fps"][name] = _signature(ob)
    except Exception:       # restricted data while Blender is still starting up
        traceback.print_exc()


def _prime_timer():
    _prime()
    return None


@persistent
def _on_load(*_):
    # Every handler of the add-on is @persistent: Blender drops the others when a
    # file is opened, which left Live Update off after File > Open (before v0.82.1).
    # session_uids belong to the previous file
    _uids.clear()
    _missing.clear()
    FAILED.clear()
    _deferred.clear()
    _prime()
    # overlays that are on (Seam Colors, Panel Gap, ...) redraw for the new file
    from . import ctrlpoints, curvature, followmap, netcheck, panelgap, seams
    for module in (ctrlpoints, curvature, followmap, netcheck, panelgap, seams):
        if "dirty" in module._state:
            module._state["dirty"] = True


def _deferred_resync():
    try:
        result = resync(bpy.context)
        settings = getattr(bpy.context.scene, "spline_surface", None)
        if result and result["stale"] and settings is not None:
            # their point count differs from what their recipe now builds; making new
            # data right after an undo would corrupt undo memory, so say so instead
            settings.live_note = ("not rebuilt after undo (point count changed): %s - "
                                  "select and Regenerate" % ", ".join(result["stale"]))
    except Exception:
        traceback.print_exc()
    return None     # run once


@persistent
def _on_undo(*_):
    _reset()
    _deferred.clear()   # the undone state is resynced in place; never extrude right after undo
    # leave the undo system alone while it is still restoring; fix up right after
    if not bpy.app.timers.is_registered(_deferred_resync):
        bpy.app.timers.register(_deferred_resync, first_interval=0.0)


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


class SS_OT_relink_scene_copies(bpy.types.Operator):
    bl_idname = "spline_surface.relink_scene_copies"
    bl_label = "Use This Scene's Copies"
    bl_description = ("Patches here were copied from another scene (Scene > Full Copy, shot "
                      "managers) and still follow that scene's curves and patches. Make them "
                      "follow the copies in this scene instead")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(scene_copy_relinks(context.scene))

    def execute(self, context):
        changed = relink_scene_copies(context.scene)
        _reset()
        context.scene.spline_surface.live_note = ""
        self.report({'INFO'}, "%d patch(es) now follow this scene's copies" % len(changed))
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


classes = (SS_OT_regenerate_all, SS_OT_select_group, SS_OT_relink_scene_copies)


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
    # enabled with a file already open: record its sources once Blender is ready
    bpy.app.timers.register(_prime_timer, first_interval=0.0)


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
