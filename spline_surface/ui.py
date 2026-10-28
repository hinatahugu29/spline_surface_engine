import json

import bpy

from . import zebra

GRADE_ICONS = {"G2": 'CHECKMARK', "G1": 'INFO', "G0": 'ERROR', "open": 'CANCEL'}


def _zebra_update(self, context):
    zebra.update_material(self)


class SS_Settings(bpy.types.PropertyGroup):
    live_update: bpy.props.BoolProperty(
        name="Live Update", default=True,
        description="Rebuild dependent patches automatically when a source patch or curve changes")
    live_note: bpy.props.StringProperty(options={'HIDDEN'})
    degree: bpy.props.IntProperty(
        name="Degree", default=5, min=2, max=5,
        description="Bezier degree of each patch side (Blender NURBS allows up to 5)")
    gap_tolerance: bpy.props.FloatProperty(
        name="Gap Tolerance", default=0.1, min=0.0, max=0.5, subtype='FACTOR',
        description="Max allowed gap between curve ends, relative to the selection size")
    samples: bpy.props.IntProperty(
        name="Samples", default=32, min=4, max=256,
        description="Samples per Bezier segment when reading boundary curves")
    resolution: bpy.props.IntProperty(
        name="Preview Resolution", default=12, min=1, max=64)
    mesh_resolution: bpy.props.IntProperty(
        name="Mesh Resolution", default=24, min=1, max=256,
        description="Surface resolution used by To Mesh")
    zebra_frequency: bpy.props.FloatProperty(
        name="Stripes", default=40.0, min=1.0, max=500.0, update=_zebra_update)
    zebra_axis: bpy.props.EnumProperty(
        name="Axis", items=[(k, k, "") for k in zebra.AXES], default='Z',
        update=_zebra_update)
    continuity_report: bpy.props.StringProperty(
        name="Continuity Report", default="[]", options={'HIDDEN', 'SKIP_SAVE'})


class _SubPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Spline Surface"
    bl_parent_id = "SS_PT_main"


class SS_PT_main(bpy.types.Panel):
    bl_label = "Spline Surface"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Spline Surface"

    def draw(self, context):
        s = context.scene.spline_surface
        col = self.layout.column(align=True)
        col.label(text="Edge tools: click in the viewport,", icon='INFO')
        col.label(text="options in the Adjust Last Operation panel")
        if s.live_note:
            self.layout.label(text=s.live_note, icon='ERROR')


class SS_PT_create(_SubPanel, bpy.types.Panel):
    bl_label = "Create"

    def draw_header(self, context):
        self.layout.label(icon='SURFACE_NSURFACE')

    def draw(self, context):
        s = context.scene.spline_surface
        layout = self.layout
        layout.operator("spline_surface.add_patch", icon='SURFACE_NSURFACE')
        col = layout.column(align=True)
        col.label(text="From curves:")
        row = col.row(align=True)
        row.operator("spline_surface.coons", icon='MOD_MESHDEFORM', text="Coons")
        row.operator("spline_surface.loft", icon='MOD_SCREW', text="Loft")
        col.operator("spline_surface.curve_network", icon='MESH_GRID')
        row = layout.row(align=True)
        row.prop(s, "gap_tolerance")
        row.operator("spline_surface.snap_ends", icon='SNAP_ON', text="")
        layout.operator("spline_surface.regenerate", icon='FILE_REFRESH')


class SS_PT_create_settings(_SubPanel, bpy.types.Panel):
    bl_label = "Fit Settings"
    bl_parent_id = "SS_PT_create"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.spline_surface
        col = self.layout.column(align=True)
        col.prop(s, "degree")
        col.prop(s, "samples")
        col.prop(s, "resolution")


class SS_PT_grow(_SubPanel, bpy.types.Panel):
    bl_label = "Grow & Connect"

    def draw_header(self, context):
        self.layout.label(icon='FORWARD')

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("spline_surface.extend_edge", icon='MOD_SOLIDIFY')
        col.operator("spline_surface.bridge_edges", icon='MOD_CURVE')
        col.operator("spline_surface.match_edge", icon='AUTOMERGE_ON')
        col = layout.column(align=True)
        row = col.row(align=True)
        row.operator("spline_surface.corner_fill", icon='MOD_TRIANGULATE')
        row.operator("spline_surface.fill_hole", icon='MOD_MESHDEFORM')
        col.operator("spline_surface.split_patch", icon='MOD_EDGESPLIT')
        layout.operator("spline_surface.offset_surface", icon='MOD_OFFSET')


class SS_PT_group(_SubPanel, bpy.types.Panel):
    bl_label = "Group"

    def draw_header(self, context):
        self.layout.label(icon='LINKED')

    def draw(self, context):
        s = context.scene.spline_surface
        layout = self.layout
        layout.prop(s, "live_update")
        row = layout.row(align=True)
        row.operator("spline_surface.select_group", icon='RESTRICT_SELECT_OFF', text="Select")
        row.operator("spline_surface.regenerate_all", icon='FILE_REFRESH', text="Regenerate")
        layout.operator("spline_surface.mirror_group", icon='MOD_MIRROR')
        row = layout.row(align=True)
        row.operator("spline_surface.cage_add", icon='OUTLINER_OB_LATTICE')
        row.operator("spline_surface.cage_apply", icon='CHECKMARK', text="Apply")
        row.operator("spline_surface.cage_cancel", icon='X', text="")


class SS_PT_inspect(_SubPanel, bpy.types.Panel):
    bl_label = "Inspect"

    def draw_header(self, context):
        self.layout.label(icon='HIDE_OFF')

    def draw(self, context):
        s = context.scene.spline_surface
        layout = self.layout
        layout.operator("spline_surface.zebra_toggle", icon='SEQ_STRIP_DUPLICATE')
        row = layout.row(align=True)
        row.prop(s, "zebra_frequency")
        row.prop(s, "zebra_axis", text="")
        from . import seams
        on = seams._state["handle"] is not None
        layout.operator("spline_surface.seam_colors", icon='COLORSET_03_VEC', depress=on,
                        text="Seam Colors: On" if on else "Seam Colors")
        if on:
            c = seams.counts()
            row = layout.row(align=True)
            for g in seams.GRADES:
                if c.get(g):
                    row.label(text="%s %d" % (g, c[g]),
                              icon=GRADE_ICONS.get(g, 'QUESTION'))
        layout.operator("spline_surface.check_continuity", icon='VIEWZOOM')
        try:
            rows = json.loads(s.continuity_report or "[]")
        except ValueError:
            rows = []
        col = layout.column(align=True)
        for r in rows:
            col.label(text="%s | %s: %s" % (r["a"], r["b"], r["grade"]),
                      icon=GRADE_ICONS.get(r["grade"], 'QUESTION'))
            col.label(text="    angle %.3g°, curvature %.1f%%" % (r["angle"], 100 * r["curvature"]))


class SS_PT_output(_SubPanel, bpy.types.Panel):
    bl_label = "Output"

    def draw_header(self, context):
        self.layout.label(icon='MESH_DATA')

    def draw(self, context):
        s = context.scene.spline_surface
        self.layout.prop(s, "mesh_resolution")
        self.layout.operator("spline_surface.to_mesh", icon='MESH_GRID')


classes = (SS_Settings, SS_PT_main, SS_PT_create, SS_PT_create_settings, SS_PT_grow,
           SS_PT_group, SS_PT_inspect, SS_PT_output)
