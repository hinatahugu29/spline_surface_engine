import json

import bpy

from . import presetting, zebra

GRADE_ICONS = {"G2": 'CHECKMARK', "G1": 'INFO', "G0": 'ERROR', "open": 'CANCEL'}


def _zebra_update(self, context):
    zebra.update_material(self)


def _gap_update(self, context):
    from . import panelgap
    if panelgap.is_on():
        panelgap.mark_dirty()


def _curvature_update(self, context):
    from . import curvature
    curvature.mark_dirty()


class SS_Settings(bpy.types.PropertyGroup):
    live_update: bpy.props.BoolProperty(
        name="Live Update", default=True,
        description="Rebuild dependent patches automatically when a source patch or curve changes")
    live_note: bpy.props.StringProperty(options={'HIDDEN'})
    linked_edit: bpy.props.BoolProperty(
        name="Linked Edit", default=False,
        description=("When you edit a patch (edit mode, move), nearby patches follow and "
                     "patches sharing an edge with it stay joined with their G0/G1/G2. "
                     "Doesn't affect Edge Chain curves, which always follow"))
    link_falloff: bpy.props.FloatProperty(
        name="Falloff", default=1.0, min=0.05, max=10.0, step=10,
        description="How far edits spread, relative to the size of the edited patch")
    zebra_on: bpy.props.BoolProperty(options={'HIDDEN'})
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
        description="To Mesh: segments along the longest patch side; the other patches get "
                    "the same density (unwelded To Mesh: segments per patch)")
    zebra_frequency: bpy.props.FloatProperty(
        name="Stripes", default=40.0, min=1.0, max=500.0, update=_zebra_update)
    zebra_axis: bpy.props.EnumProperty(
        name="Axis", items=[(k, k, "") for k in zebra.AXES], default='Z',
        update=_zebra_update)
    continuity_report: bpy.props.StringProperty(
        name="Continuity Report", default="[]", options={'HIDDEN', 'SKIP_SAVE'})
    comb_scale: bpy.props.FloatProperty(
        name="Scale", default=1.0, min=0.05, max=20.0, update=_curvature_update,
        description="Length of the longest comb tooth, relative to the scene size")
    comb_count: bpy.props.IntProperty(
        name="Combs", default=5, min=1, max=40, update=_curvature_update,
        description="Iso-curves per patch and direction that get a comb")
    comb_direction: bpy.props.EnumProperty(
        name="Direction", default='BOTH', update=_curvature_update,
        items=[('U', "U", "Combs along U"), ('V', "V", "Combs along V"),
               ('BOTH', "U+V", "Combs in both directions")])
    pgap_target: bpy.props.FloatProperty(
        name="Target", default=0.01, min=0.0, max=100.0, unit='LENGTH', update=_gap_update,
        description="Panel Gap: the gap the joints should have")
    pgap_tolerance: bpy.props.FloatProperty(
        name="Tolerance", default=0.003, min=0.0, max=100.0, unit='LENGTH', update=_gap_update,
        description="Panel Gap: how far from the target a gap may be and still show green")
    pgap_search: bpy.props.FloatProperty(
        name="Search", default=0.0, min=0.0, max=1000.0, unit='LENGTH', update=_gap_update,
        description="Panel Gap: edges farther apart than this are not a joint "
                    "(0: 5% of the scene size)")
    curvature_type: bpy.props.EnumProperty(
        name="Map", default='GAUSS', update=_curvature_update,
        items=[('GAUSS', "Gaussian", "Product of the principal curvatures: dome, flat, saddle"),
               ('MEAN', "Mean", "Average of the principal curvatures, signed by the normal")])


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
        col.label(text="Body panel presets:")
        grid = col.grid_flow(row_major=True, columns=2, align=True)
        for kind, label, _ in presetting.KIND_ITEMS:
            grid.operator("spline_surface.add_preset", text=label).kind = kind
        layout.operator("spline_surface.add_blueprint", icon='IMAGE_REFERENCE')
        col = layout.column(align=True)
        col.label(text="Boundary curves:")
        row = col.row(align=True)
        row.operator("spline_surface.add_boundary_curve", icon='LINCURVE',
                     text="Line").shape = 'LINE'
        row.operator("spline_surface.add_boundary_curve", icon='SPHERECURVE',
                     text="Arc").shape = 'ARC'
        col.label(text="From curves:")
        row = col.row(align=True)
        row.operator("spline_surface.coons", icon='MOD_MESHDEFORM', text="Coons")
        row.operator("spline_surface.loft", icon='MOD_SCREW', text="Loft")
        row = col.row(align=True)
        row.operator("spline_surface.curve_network", icon='MESH_GRID')
        row.operator("spline_surface.add_curve_grid", icon='ADD', text="")
        from . import netcheck
        on = netcheck._state["handle"] is not None
        col.operator("spline_surface.network_check", icon='VIEWZOOM', depress=on,
                     text="Network Check: On" if on else "Network Check")
        for i, line in enumerate(netcheck.summary()):
            col.label(text=line, icon='CHECKMARK' if line.endswith("ready") else
                      ('NONE' if i == 0 else 'ERROR'))
        col.operator("spline_surface.project_curve", icon='MOD_SHRINKWRAP')
        row = col.row(align=True)
        row.operator("spline_surface.move_crossing", icon='TRANSFORM_ORIGINS')
        row.operator("spline_surface.fair_curves", icon='MOD_SMOOTH')
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
        col.operator("spline_surface.fillet", icon='MOD_BEVEL')
        col.operator("spline_surface.flange", icon='MOD_SKIN')
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
        row.prop(s, "linked_edit")
        row.prop(s, "link_falloff")
        row = layout.row(align=True)
        row.operator("spline_surface.edge_chain", icon='CURVE_PATH')
        row.operator("spline_surface.merge_patches", icon='SELECT_EXTEND', text="Merge")
        row = layout.row(align=True)
        row.operator("spline_surface.select_group", icon='RESTRICT_SELECT_OFF', text="Select")
        row.operator("spline_surface.regenerate_all", icon='FILE_REFRESH', text="Regenerate")
        layout.operator("spline_surface.mirror_group", icon='MOD_MIRROR')
        row = layout.row(align=True)
        row.operator("spline_surface.cage_add", icon='OUTLINER_OB_LATTICE')
        row.operator("spline_surface.cage_apply", icon='CHECKMARK', text="Apply")
        row.operator("spline_surface.cage_cancel", icon='X', text="")


class SS_PT_shape(_SubPanel, bpy.types.Panel):
    bl_label = "Shape"

    def draw_header(self, context):
        self.layout.label(icon='MOD_SMOOTH')

    def draw(self, context):
        layout = self.layout
        layout.operator("spline_surface.soft_pull", icon='SMOOTHCURVE')
        row = layout.row(align=True)
        row.operator("spline_surface.control_mesh", icon='EDITMODE_HLT')
        row.operator("spline_surface.control_mesh_done", icon='CHECKMARK', text="Done")
        row = layout.row(align=True)
        row.operator("spline_surface.fair", icon='MOD_SMOOTH')
        row.operator("spline_surface.refine", icon='MOD_SUBSURF')


class SS_PT_inspect(_SubPanel, bpy.types.Panel):
    bl_label = "Inspect"

    def draw_header(self, context):
        self.layout.label(icon='HIDE_OFF')

    def draw(self, context):
        s = context.scene.spline_surface
        layout = self.layout
        layout.operator("spline_surface.zebra_toggle", icon='SEQ_STRIP_DUPLICATE',
                        depress=s.zebra_on, text="Zebra: On" if s.zebra_on else "Zebra")
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
        from . import curvature
        on = curvature.is_on("comb")
        layout.operator("spline_surface.curvature_comb", icon='FORCE_HARMONIC', depress=on,
                        text="Curvature Comb: On" if on else "Curvature Comb")
        if on:
            row = layout.row(align=True)
            row.prop(s, "comb_scale")
            row.prop(s, "comb_count")
            layout.prop(s, "comb_direction", expand=True)
        on = curvature.is_on("map")
        layout.operator("spline_surface.curvature_map", icon='COLOR', depress=on,
                        text="Curvature Map: On" if on else "Curvature Map")
        if on:
            layout.prop(s, "curvature_type", expand=True)
            layout.label(text="Full colour at ±%.3g" % curvature.limit())
        from . import ctrlpoints
        on = ctrlpoints._state["handle"] is not None
        layout.operator("spline_surface.control_points", icon='PARTICLE_POINT', depress=on,
                        text="Control Points: On" if on else "Control Points")
        if on:
            free, locked = ctrlpoints.counts()
            layout.label(text="Free %d | Pinned %d" % (free, locked))
        from . import panelgap
        on = panelgap.is_on()
        layout.operator("spline_surface.panel_gap", icon='ARROW_LEFTRIGHT', depress=on,
                        text="Panel Gap: On" if on else "Panel Gap")
        if on:
            row = layout.row(align=True)
            row.prop(s, "pgap_target")
            row.prop(s, "pgap_tolerance")
            layout.prop(s, "pgap_search")
            col = layout.column(align=True)
            for j in panelgap.joints():
                col.label(text="%s | %s" % (j["a"], j["b"]), icon='ARROW_LEFTRIGHT')
                col.label(text="    gap %.3g - %.3g (mean %.3g), step %.3g"
                               % (j["min"], j["max"], j["mean"], j["step"]))
        layout.operator("spline_surface.check_continuity", icon='VIEWZOOM')
        try:
            rows = json.loads(s.continuity_report or "[]")
        except ValueError:
            rows = []
        col = layout.column(align=True)
        for r in rows:
            col.label(text="%s(%s) | %s(%s): %s" % (
                          r["a"], r.get("edge_a", "?"), r["b"], r.get("edge_b", "?"), r["grade"]),
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
           SS_PT_group, SS_PT_shape, SS_PT_inspect, SS_PT_output)
