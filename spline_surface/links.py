"""Links that can be edited after a patch is made: the seam continuity of
Extend / Bridge / Corner Fill, and Match Edge's follow links (list, change,
remove). Changes rebuild the patch and its group through live.update_chain.
"""
import json

import bpy

from .recipe import META_KEY, hand_edited, raw_meta, store_meta, world_net

from .graph import _meta, dependency_graph
from . import live
from .live import update_chain


# Recipes whose seam continuity / tension can be changed after the fact.
CONTINUITY_METHODS = {"extend": "Extend", "blend": "Bridge", "corner": "Corner Fill"}


def continuity_recipe(ob):
    """The recipe of `ob` if its continuity can be edited, else None."""
    if ob is None or ob.type != 'SURFACE' or META_KEY not in ob:
        return None
    meta = _meta(ob)
    if meta and meta.get("method") in CONTINUITY_METHODS and "continuity" in meta:
        return meta
    return None


def set_continuity(context, ob, continuity, tension=None):
    """Rebuild `ob` with another continuity (and tension); what reads it follows.
    Raises ValueError and leaves everything as it was if that can't be done."""
    meta = continuity_recipe(ob)
    if meta is None:
        raise ValueError("%s has no editable continuity" % getattr(ob, "name", "?"))
    if hand_edited(ob, meta):
        raise ValueError("%s was edited by hand (Regenerate first)" % ob.name)
    new = dict(meta, continuity=continuity)
    if tension is not None:
        new["tension"] = float(tension)
    old_raw = ob[META_KEY]
    ob[META_KEY] = json.dumps(new)
    live._state["busy"] = True
    try:
        status, msg = live.rebuild(ob, context)
        if status != "ok":
            ob[META_KEY] = old_raw
            raise ValueError(msg)
        res = update_chain([ob.name], context)
        _redraw(context, [ob.name] + res["ok"])
        return res
    finally:
        live._state["busy"] = False
        live._reset()


def _redraw(context, names):
    """A button press runs outside any depsgraph pass: tag what moved and redraw,
    or the viewport keeps showing the old shape until something else updates."""
    for name in names:
        o = bpy.data.objects.get(name)
        if o is not None:
            o.data.update_tag()
            o.update_tag(refresh={'OBJECT', 'DATA'})
    context.view_layer.update()
    for win in getattr(context.window_manager, "windows", ()):
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# ---------------------------------------------------------------- match links

def match_entries(ob):
    """The Match Edge links of `ob` (patches it follows), [] if none."""
    if ob is None or ob.type != 'SURFACE' or META_KEY not in ob:
        return []
    return list((_meta(ob) or {}).get("matches", []))


def match_followers(name):
    """[(patch, continuity)] of the patches matched to `name` (that follow it)."""
    out = []
    for dep in sorted(dependency_graph().get(name, ())):
        ob = bpy.data.objects.get(dep)
        for m in (_meta(ob) or {}).get("matches", []) if ob is not None else ():
            if m.get("ref") == name:
                out.append((dep, m.get("continuity", "?")))
    return out


def _frozen(ob, meta, matches):
    """`ob`'s current shape as a raw recipe carrying `matches` (and its symmetry)."""
    net, orders = world_net(ob)
    extra = {"symmetry": meta["symmetry"]} if meta.get("symmetry") else {}
    return raw_meta(net, orders, matches=matches, **extra)


def set_match(context, ob, ref, continuity=None, spread=None, remove=False):
    """Change (or with `remove`, drop) `ob`'s Match Edge link to the patch `ref`.

    A changed link rebuilds `ob` and what reads it. Dropping a link keeps the
    patch where it is when its recipe is its own net (a free or hand-edited
    patch); a generated patch is rebuilt from its recipe without the link.
    Raises ValueError and leaves everything as it was if that can't be done."""
    meta = _meta(ob) if ob is not None and META_KEY in ob else None
    entries = (meta or {}).get("matches", [])
    if not any(m.get("ref") == ref for m in entries):
        raise ValueError("%s does not follow %s" % (getattr(ob, "name", "?"), ref))
    old_raw = ob[META_KEY]
    if remove:
        rest = [m for m in entries if m.get("ref") != ref]
        if meta.get("method") == "raw" or hand_edited(ob, meta):
            new, keep = _frozen(ob, meta, rest), True
        else:
            new, keep = dict(meta, matches=rest), False
    else:
        changed = []
        for m in entries:
            m = dict(m)
            if m.get("ref") == ref:
                if continuity is not None:
                    m["continuity"] = continuity
                if spread is not None:
                    m["spread"] = bool(spread)
            changed.append(m)
        if hand_edited(ob, meta):
            new = _frozen(ob, meta, changed)
        else:
            new = dict(meta, matches=changed)
        keep = False
    live._state["busy"] = True
    try:
        if keep:
            store_meta(ob, new)
            res = {"ok": [], "skipped": [], "error": [], "unchanged": []}
        else:
            ob[META_KEY] = json.dumps(new)
            status, msg = live.rebuild(ob, context)
            if status != "ok":
                ob[META_KEY] = old_raw
                raise ValueError(msg)
            res = update_chain([ob.name], context)
        _redraw(context, [ob.name] + res["ok"])
        return res
    finally:
        live._state["busy"] = False
        live._reset()


class SS_OT_match_link_edit(bpy.types.Operator):
    bl_idname = "spline_surface.match_link_edit"
    bl_label = "Edit Match Link"
    bl_description = ("Change how the active patch follows a patch it was matched to "
                      "(continuity, Spread). Patches built on it follow")
    bl_options = {'REGISTER', 'UNDO'}

    ref: bpy.props.StringProperty(name="Reference")
    continuity: bpy.props.EnumProperty(
        name="Continuity",
        items=[('G0', "G0", "Position only"),
               ('G1', "G1", "Tangent continuous"),
               ('G2', "G2", "Curvature continuous")],
        default='G2')
    spread: bpy.props.BoolProperty(
        name="Spread", default=True,
        description="Ease the seam's move into the whole patch instead of only the rows "
                    "next to the seam")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(match_entries(context.active_object))

    def execute(self, context):
        ob = context.active_object
        try:
            res = set_match(context, ob, self.ref, self.continuity, self.spread)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        bad = res["skipped"] + res["error"]
        msg = "%s follows %s (%s%s)" % (ob.name, self.ref, self.continuity,
                                        ", spread" if self.spread else "")
        if bad:
            self.report({'WARNING'}, msg + "; not updated: " + ", ".join(bad))
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


class SS_OT_match_link_remove(bpy.types.Operator):
    bl_idname = "spline_surface.match_link_remove"
    bl_label = "Unlink Match"
    bl_description = ("Stop the active patch following a patch it was matched to. A free or "
                      "hand-edited patch stays where it is; a generated patch goes back to "
                      "the shape its own recipe gives")
    bl_options = {'REGISTER', 'UNDO'}

    ref: bpy.props.StringProperty(name="Reference")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(match_entries(context.active_object))

    def execute(self, context):
        ob = context.active_object
        try:
            set_match(context, ob, self.ref, remove=True)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "%s no longer follows %s" % (ob.name, self.ref))
        return {'FINISHED'}


class SS_OT_edit_continuity(bpy.types.Operator):
    bl_idname = "spline_surface.edit_continuity"
    bl_label = "Edit Continuity"
    bl_description = ("Change the continuity (G0/G1/G2) or tension of the active Extend, "
                      "Bridge or Corner Fill patch without picking the edges again. "
                      "Patches built on it follow")
    bl_options = {'REGISTER', 'UNDO'}

    continuity: bpy.props.EnumProperty(
        name="Continuity",
        items=[('G0', "G0", "Position only"),
               ('G1', "G1", "Tangent continuous"),
               ('G2', "G2", "Curvature continuous")],
        default='G2')
    tension: bpy.props.FloatProperty(name="Tension", default=1.0, min=0.05, max=5.0,
                                     description="How far the seam's shape reaches in")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and continuity_recipe(context.active_object) is not None

    def invoke(self, context, event):
        meta = continuity_recipe(context.active_object)
        if not self.properties.is_property_set("continuity"):
            self.continuity = meta["continuity"]
        if not self.properties.is_property_set("tension"):
            self.tension = meta.get("tension", 1.0)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        layout.row().prop(self, "continuity", expand=True)
        layout.prop(self, "tension")

    def execute(self, context):
        ob = context.active_object
        try:
            res = set_continuity(context, ob, self.continuity, self.tension)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        bad = res["skipped"] + res["error"]
        if bad:
            self.report({'WARNING'}, "%s is %s; not updated: %s"
                        % (ob.name, self.continuity, ", ".join(bad)))
        else:
            self.report({'INFO'}, "%s is %s" % (ob.name, self.continuity))
        return {'FINISHED'}


# ---------------------------------------------------------------- tension field
# Object.ss_seam_tension: the panel's Tension field under G0 / G1 / G2. It stores
# nothing itself: it reads the recipe's tension and setting it rebuilds the patch
# (and what reads it) as Edit Continuity does - so dragging it shows the shape live.

def _tension_get(ob):
    meta = continuity_recipe(ob)
    return float(meta.get("tension", 1.0)) if meta is not None else 1.0


def _tension_set(ob, value):
    meta = continuity_recipe(ob)
    if meta is None or abs(float(meta.get("tension", 1.0)) - value) < 1e-6:
        return
    try:
        set_continuity(bpy.context, ob, meta["continuity"], value)
    except ValueError as e:
        print("Spline Surface: tension not changed:", e)


def register():
    bpy.types.Object.ss_seam_tension = bpy.props.FloatProperty(
        name="Tension", min=0.05, max=5.0, soft_min=0.2, soft_max=3.0, step=5, precision=2,
        description="How far the seam's shape reaches in (rebuilds the patch, and the patches "
                    "built on it, as you drag)",
        get=_tension_get, set=_tension_set, options=set())


def unregister():
    del bpy.types.Object.ss_seam_tension


classes = (SS_OT_edit_continuity, SS_OT_match_link_edit, SS_OT_match_link_remove)
