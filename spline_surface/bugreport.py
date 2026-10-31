"""Copy Bug Report: put the facts a bug report needs on the clipboard.

Versions, the add-on's settings, and for the selected patches (or the active
one) their size, recipe and seams. Recipes that store a whole control net are
shortened to their shape, so the text stays small enough to paste anywhere.
"""
import json
import platform

import bpy

from .core.continuity import find_shared_edge, grade, measure
from .recipe import META_KEY, hand_edited, world_net

MAX_PATCHES = 12


def _addon_version():
    import sys
    mod = sys.modules.get(__package__)
    info = getattr(mod, "bl_info", {}) if mod else {}
    return ".".join(str(v) for v in info.get("version", ("?",)))


def _short_meta(meta):
    meta = dict(meta)
    if "net" in meta:
        meta["net"] = "<%d x %d points>" % (len(meta["net"]), len(meta["net"][0]))
    meta.pop("fp", None)
    return meta


def _settings_lines(settings):
    lines = []
    for prop in settings.bl_rna.properties:
        if prop.identifier == "rna_type" or prop.is_hidden:
            continue
        value = getattr(settings, prop.identifier)
        if isinstance(value, float):
            value = round(value, 5)
        if isinstance(value, str) and len(value) > 60:
            continue
        lines.append("  %s = %s" % (prop.identifier, value))
    return lines


def build_report(context):
    """The report text for the current selection."""
    settings = context.scene.spline_surface
    lines = ["Spline Surface %s | Blender %s | %s" % (
        _addon_version(), bpy.app.version_string, platform.platform())]
    lines += ["", "Settings:"] + _settings_lines(settings)
    if settings.live_note:
        lines += ["", "Live Update note: " + settings.live_note]
    patches = [o for o in context.selected_objects if o.type == 'SURFACE']
    if not patches and context.active_object is not None and context.active_object.type == 'SURFACE':
        patches = [context.active_object]
    lines += ["", "Patches (%d selected%s):" % (
        len(patches), ", first %d shown" % MAX_PATCHES if len(patches) > MAX_PATCHES else "")]
    nets = {}
    for ob in patches[:MAX_PATCHES]:
        try:
            net, orders = world_net(ob)
        except ValueError as e:
            lines.append("- %s: not a clean patch (%s)" % (ob.name, e))
            continue
        nets[ob.name] = (net, orders)
        meta = None
        if META_KEY in ob:
            try:
                meta = json.loads(ob[META_KEY])
            except ValueError:
                pass
        edited = ""
        if meta is not None and hand_edited(ob, meta):
            edited = " [edited by hand]"
        lines.append("- %s: %d x %d points, orders %s%s" % (
            ob.name, net.shape[0], net.shape[1], tuple(orders), edited))
        lines.append("    recipe: " + (json.dumps(_short_meta(meta), sort_keys=True)
                                       if meta is not None else "none"))
    names = sorted(nets)
    seam_lines = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (Pa, oa), (Pb, ob_) = nets[a], nets[b]
            pts = Pa.reshape(-1, 3), Pb.reshape(-1, 3)
            size = max(float(abs(p.max(0) - p.min(0)).max()) for p in pts)
            shared = find_shared_edge(Pa, Pb, max(size * settings.gap_tolerance, 1e-6), oa, ob_)
            if shared is None:
                continue
            m = measure(Pa, Pb, *shared[:3], oa, ob_)
            seam_lines.append("- %s(%s) | %s(%s): %s, gap %.3g, angle %.3g deg, curvature %.1f%%" % (
                a, shared[0], b, shared[1], grade(m), m["gap"], m["angle"], 100 * m["curvature"]))
    if seam_lines:
        lines += ["", "Seams between them:"] + seam_lines
    return "\n".join(lines)


class SS_OT_copy_report(bpy.types.Operator):
    bl_idname = "spline_surface.copy_report"
    bl_label = "Copy Bug Report"
    bl_description = ("Copy versions, settings and the selected patches' recipes and seams to the "
                      "clipboard, to paste into a bug report")
    bl_options = {'REGISTER'}

    def execute(self, context):
        text = build_report(context)
        context.window_manager.clipboard = text
        self.report({'INFO'}, "Copied %d lines to the clipboard" % (text.count("\n") + 1))
        return {'FINISHED'}


classes = (SS_OT_copy_report,)
