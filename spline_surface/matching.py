"""Regenerate, Match Edge and the continuity report (Check Continuity,
jump to a seam)."""
import json

import bpy

from .core.diagnose import diagnose
from .core.continuity import find_shared_edge, grade, measure
from .recipe import (META_KEY, hand_edited, hand_edited_net, raw_meta,
                     replace_net, store_meta, world_net)
from .rebuild import (_seam_tolerance, apply_match, compute_from_meta)


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
    spread: bpy.props.BoolProperty(
        name="Spread", default=True,
        description="Ease the seam's move into the whole patch (the opposite edge stays put) "
                    "instead of only the 1-3 rows next to the seam. Keeps the patch smooth "
                    "when the other patch is later moved far")

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
        entry = {"ref": ref.name, "continuity": self.continuity, "spread": self.spread}
        try:
            tnet, torders = world_net(target)
            net, orders = apply_match(tnet, torders, *world_net(ref), self.continuity, settings,
                                      entry)
            replace_net(target, net, settings, orders)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if meta is None or hand_edited_net(meta, tnet, torders):
            # Keep the hand-made shape as the recipe so Regenerate does not undo it.
            old_matches = meta.get("matches", []) if meta else []
            meta = raw_meta(tnet, torders, matches=old_matches)
        matches = [m for m in meta.get("matches", []) if m["ref"] != ref.name]
        matches.append(entry)
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


def patch_nets(context, report=None):
    """{name: (net, orders)} of the selected patches, or every visible one when
    fewer than two are selected."""
    pool = _selected_surfaces(context)
    if len(pool) < 2:
        pool = [o for o in context.visible_objects if o.type == 'SURFACE']
    nets = {}
    for o in pool:
        try:
            nets[o.name] = world_net(o)
        except ValueError as e:
            if report:
                report({'WARNING'}, str(e))
    return nets


def measure_seams(nets, settings):
    """[dict(a, b, edge_a, edge_b, grade, gap, angle, curvature, size)] for every seam."""
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
    return rows


class SS_OT_diagnose(bpy.types.Operator):
    bl_idname = "spline_surface.diagnose"
    bl_label = "Diagnose"
    bl_description = ("Look for quality problems among the selected patches (all visible ones when "
                      "fewer than two are selected): seams weaker than their recipe asks for, kinks, "
                      "and wavy patches whose curvature changes sign or bumps. Worst first, each "
                      "with a suggestion")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(o.type == 'SURFACE' for o in context.visible_objects)

    def execute(self, context):
        settings = context.scene.spline_surface
        nets = patch_nets(context, self.report)
        metas = {}
        for name in nets:
            ob = bpy.data.objects[name]
            try:
                metas[name] = json.loads(ob[META_KEY]) if META_KEY in ob else None
            except ValueError:
                metas[name] = None
        issues = diagnose(nets, metas, measure_seams(nets, settings))
        settings.diagnose_report = json.dumps(issues)
        settings.diagnose_done = True
        if not issues:
            self.report({'INFO'}, "No problems found in %d patch(es)" % len(nets))
        for i in issues:
            self.report({'INFO'}, "%s -> %s" % (i["text"], i["hint"]))
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
        nets = patch_nets(context, self.report)
        rows = measure_seams(nets, settings)
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


classes = (SS_OT_regenerate, SS_OT_match_edge, SS_OT_seam_jump, SS_OT_check_continuity,
           SS_OT_diagnose)
