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
    return ORIG_KEY in ob


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
    names = json.loads(ob[ORIG_KEY])
    del ob[ORIG_KEY]
    mats = ob.data.materials
    mats.clear()
    for n in names:
        mats.append(bpy.data.materials.get(n) if n else None)


class SS_OT_zebra_toggle(bpy.types.Operator):
    bl_idname = "spline_surface.zebra_toggle"
    bl_label = "Zebra"
    bl_description = ("Toggle zebra stripes on the selected surfaces/meshes "
                      "(view in Material Preview)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(o.type in {'SURFACE', 'MESH'} for o in context.selected_objects)

    def execute(self, context):
        targets = [o for o in context.selected_objects if o.type in {'SURFACE', 'MESH'}]
        turn_on = not all(is_on(o) for o in targets)
        for ob in targets:
            if turn_on:
                enable(ob, context.scene.spline_surface)
            else:
                disable(ob)
        if turn_on:
            for area in context.screen.areas if context.screen else ():
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D' and space.shading.type == 'SOLID':
                            space.shading.type = 'MATERIAL'
        return {'FINISHED'}


classes = (SS_OT_zebra_toggle,)
