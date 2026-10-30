bl_info = {
    "name": "Spline Surface",
    "author": "hinata",
    "version": (0, 38, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Spline Surface",
    "description": "Build smooth NURBS patches from boundary curves",
    "category": "Add Curve",
}

import bpy  # noqa: E402

from . import cage, controlmesh, ctrlpoints, curvature, filleting, gizmos, linked, live, netcheck, operators, pulling, seams, shaping, splitting, symmetry, ui, zebra  # noqa: E402

classes = (ui.classes + operators.classes + filleting.classes + live.classes + linked.classes
           + cage.classes + symmetry.classes + splitting.classes + seams.classes + ctrlpoints.classes
           + curvature.classes + netcheck.classes + pulling.classes + controlmesh.classes
           + shaping.classes + zebra.classes + gizmos.classes)


def _add_menu(self, context):
    self.layout.separator()
    self.layout.operator("spline_surface.add_patch", icon='SURFACE_NSURFACE')
    self.layout.operator("spline_surface.add_curve_grid", icon='MESH_GRID')


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.spline_surface = bpy.props.PointerProperty(type=ui.SS_Settings)
    bpy.types.VIEW3D_MT_surface_add.append(_add_menu)
    live.register()
    seams.register()
    ctrlpoints.register()
    curvature.register()
    netcheck.register()
    controlmesh.register()
    zebra.register()


def unregister():
    zebra.unregister()
    controlmesh.unregister()
    netcheck.unregister()
    curvature.unregister()
    ctrlpoints.unregister()
    seams.unregister()
    live.unregister()
    bpy.types.VIEW3D_MT_surface_add.remove(_add_menu)
    del bpy.types.Scene.spline_surface
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
