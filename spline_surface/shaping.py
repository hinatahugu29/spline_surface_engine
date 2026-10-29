"""Fair and Refine: one-shot interior tools (no dragging needed).

Fair irons out the free interior of the selected patches; rows pinned by seams
(and, by default, the patch boundary) stay put, so continuity is kept. Refine
adds control points without changing the shape, giving Soft Pull and Control
Mesh more to work with. Both leave the patch as if edited by hand.
"""
import bpy
import numpy as np

from .core.fair import fair_net, refine_net
from .ctrlpoints import pin_masks
from .live import _write_in_place
from .recipe import replace_net, world_net


def _patches(context):
    return [o for o in context.selected_objects if o.type == 'SURFACE']


class SS_OT_fair(bpy.types.Operator):
    bl_idname = "spline_surface.fair"
    bl_label = "Fair"
    bl_description = ("Smooth the interior of the selected patches. Points pinned by seams "
                      "stay, so G0/G1/G2 are kept")
    bl_options = {'REGISTER', 'UNDO'}

    strength: bpy.props.FloatProperty(
        name="Strength", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        description="0 leaves the patch as it is, 1 smooths it fully")
    keep_boundary: bpy.props.BoolProperty(
        name="Keep Boundary", default=True,
        description="Also keep the patch's outer ring of points, so open edges don't move")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(_patches(context))

    def execute(self, context):
        patches = _patches(context)
        masks = pin_masks(context, patches)
        worst = 0.0
        for ob in patches:
            if ob.name not in masks:
                continue
            net, orders, pinned = masks[ob.name]
            fixed = pinned.copy()
            if self.keep_boundary:
                fixed[[0, -1]] = fixed[:, [0, -1]] = True
            new = fair_net(net, fixed, self.strength)
            size = max(float(np.ptp(net.reshape(-1, 3), axis=0).max()), 1e-12)
            worst = max(worst, float(np.abs(new - net).max()) / size)
            if not _write_in_place(ob, new, orders):
                replace_net(ob, new, context.scene.spline_surface, orders)
        self.report({'INFO'}, "Faired %d patch(es), largest move %.2f%% of the patch size"
                    % (len(masks), 100 * worst))
        return {'FINISHED'}


class SS_OT_refine(bpy.types.Operator):
    bl_idname = "spline_surface.refine"
    bl_label = "Refine"
    bl_description = ("Add control points without changing the shape (every span is split in "
                      "two). More points to shape with; seams keep their continuity but a "
                      "neighbour must match the new point count to be re-matched")
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        name="Direction", default='BOTH',
        items=[('BOTH', "U and V", ""), ('U', "U", ""), ('V', "V", "")])

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(_patches(context))

    def execute(self, context):
        done = 0
        for ob in _patches(context):
            try:
                net, orders = world_net(ob)
                new, new_orders = refine_net(net, orders, self.direction)
            except ValueError as e:
                self.report({'WARNING'}, "%s: %s" % (ob.name, e))
                continue
            replace_net(ob, new, context.scene.spline_surface, new_orders)
            done += 1
        if not done:
            return {'CANCELLED'}
        self.report({'INFO'}, "Refined %d patch(es)" % done)
        return {'FINISHED'}


classes = (SS_OT_fair, SS_OT_refine)
