"""Zebra (reflection stripe) display for checking surface smoothness.

Stripes come from the reflected view vector, so they behave like the light
strips used in car studios: kinks in the stripes reveal G1 breaks, sudden
changes in stripe width reveal curvature (G2) jumps. Visible in Material
Preview / Rendered shading.
"""
import json

import bpy

MAT_NAME = "SS_Zebra"
ORIG_KEY = "spline_surface_orig_mats"
AXES = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}


def _get_material():
    mat = bpy.data.materials.get(MAT_NAME)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MAT_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    neg = nt.nodes.new("ShaderNodeVectorMath")
    neg.operation = 'SCALE'
    neg.inputs["Scale"].default_value = -1.0
    refl = nt.nodes.new("ShaderNodeVectorMath")
    refl.operation = 'REFLECT'
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = 'DOT_PRODUCT'
    dot.name = "axis"
    freq = nt.nodes.new("ShaderNodeMath")
    freq.operation = 'MULTIPLY'
    freq.name = "frequency"
    sine = nt.nodes.new("ShaderNodeMath")
    sine.operation = 'SINE'
    step = nt.nodes.new("ShaderNodeMath")
    step.operation = 'GREATER_THAN'
    step.inputs[1].default_value = 0.0
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    link = nt.links.new
    link(geo.outputs["Incoming"], neg.inputs[0])
    link(neg.outputs["Vector"], refl.inputs[0])
    link(geo.outputs["Normal"], refl.inputs[1])
    link(refl.outputs["Vector"], dot.inputs[0])
    link(dot.outputs["Value"], freq.inputs[0])
    link(freq.outputs[0], sine.inputs[0])
    link(sine.outputs[0], step.inputs[0])
    link(step.outputs[0], emit.inputs["Color"])
    link(emit.outputs[0], out.inputs["Surface"])
    for k, node in enumerate((geo, neg, refl, dot, freq, sine, step, emit, out)):
        node.location = (k * 180, 0)
    return mat


def update_material(settings):
    mat = bpy.data.materials.get(MAT_NAME)
    if mat is None:
        return
    nodes = mat.node_tree.nodes
    nodes["axis"].inputs[1].default_value = AXES[settings.zebra_axis]
    nodes["frequency"].inputs[1].default_value = settings.zebra_frequency


def is_on(ob):
    mats = getattr(ob.data, "materials", ())
    return ORIG_KEY in ob or any(m is not None and m.name == MAT_NAME for m in mats)


def enable(ob, settings):
    if is_on(ob):
        return
    mat = _get_material()
    update_material(settings)
    mats = ob.data.materials
    ob[ORIG_KEY] = json.dumps([m.name if m else None for m in mats])
    mats.clear()
    mats.append(mat)


def disable(ob):
    if not is_on(ob):
        return
    mats = ob.data.materials
    if ORIG_KEY in ob:
        names = json.loads(ob[ORIG_KEY])
        del ob[ORIG_KEY]
        mats.clear()
        for n in names:
            mats.append(bpy.data.materials.get(n) if n else None)
    else:  # a patch copied from a striped one: just drop the stripes
        keep = [m for m in mats if m is None or m.name != MAT_NAME]
        mats.clear()
        for m in keep:
            mats.append(m)


def _targets(context):
    """Every visible patch, plus selected meshes (e.g. To Mesh results)."""
    obs = [o for o in context.visible_objects if o.type == 'SURFACE']
    obs += [o for o in context.selected_objects if o.type == 'MESH']
    return obs


class SS_OT_zebra_toggle(bpy.types.Operator):
    bl_idname = "spline_surface.zebra_toggle"
    bl_label = "Zebra"
    bl_description = ("Toggle zebra stripes on all visible patches at once (and selected "
                      "meshes). New patches get stripes too while it is on")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.spline_surface
        settings.zebra_on = not settings.zebra_on
        if settings.zebra_on:
            for ob in _targets(context):
                enable(ob, settings)
            for area in context.screen.areas if context.screen else ():
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D' and space.shading.type == 'SOLID':
                            space.shading.type = 'MATERIAL'
        else:
            for ob in bpy.data.objects:
                if ob.type in {'SURFACE', 'MESH'}:
                    disable(ob)
        return {'FINISHED'}


def _on_depsgraph(scene, depsgraph):
    settings = getattr(scene, "spline_surface", None)
    if settings is None or not settings.zebra_on:
        return
    for u in depsgraph.updates:
        idb = getattr(u.id, "original", u.id)
        if isinstance(idb, bpy.types.Object) and idb.type == 'SURFACE' and not is_on(idb):
            enable(idb, settings)


classes = (SS_OT_zebra_toggle,)


def register():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
