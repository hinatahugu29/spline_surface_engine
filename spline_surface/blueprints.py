"""Blueprints: place a reference drawing at its real size.

Adds an image as a Blender image empty standing in the plane of its view, at a
size you give in scene units: the length of the car in the side and top views,
its width in the front view. The origin sits at the bottom middle of front and
side drawings (so z = 0 is the ground) and at the centre of a top drawing. The
car's centre line is x = 0, the front of the car is +y, as in the body presets.
"""
import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .recipe import _make_active

VIEWS = [
    ('SIDE', "Side", "Seen from the right: the plane of y (front to back) and z (up)"),
    ('FRONT', "Front", "Seen from the front: the plane of x (across) and z (up)"),
    ('TOP', "Top", "Seen from above: the plane of x (across) and y (front to back)"),
]
# euler rotation that stands the image in each view's plane
_ROTATION = {'SIDE': (math.pi / 2, 0.0, math.pi / 2), 'FRONT': (math.pi / 2, 0.0, 0.0),
             'TOP': (0.0, 0.0, 0.0)}


def add_blueprint(context, image, view, length, offset=(0.0, 0.0, 0.0), opacity=0.5,
                  flip=False, name=None):
    """Image empty for `image` in `view`; `length` is its width in scene units."""
    w, h = image.size[0] or 1, image.size[1] or 1
    ob = bpy.data.objects.new(name or "Blueprint %s" % view.title(), None)
    ob.empty_display_type = 'IMAGE'
    ob.data = image
    # the empty scales its longer image side to empty_display_size
    ob.empty_display_size = length if w >= h else length * h / w
    ob.empty_image_offset = (-0.5, -0.5) if view == 'TOP' else (-0.5, 0.0)
    ob.use_empty_image_alpha = True
    ob.color[3] = opacity
    ob.empty_image_depth = 'BACK'
    ob.show_empty_image_only_axis_aligned = True
    ob.rotation_euler = _ROTATION[view]
    ob.location = offset
    if flip:
        ob.scale.x = -1.0
    context.collection.objects.link(ob)
    return ob


class SS_OT_add_blueprint(bpy.types.Operator, ImportHelper):
    bl_idname = "spline_surface.add_blueprint"
    bl_label = "Add Blueprint"
    bl_description = ("Place a reference drawing (side, front or top view) at its real size, "
                      "ground at z = 0, centre line at x = 0, front of the car towards +y")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp", options={'HIDDEN'})
    view: EnumProperty(name="View", items=VIEWS, default='SIDE')
    length: FloatProperty(
        name="Length", default=4.5, min=0.01, max=10000.0, unit='LENGTH',
        description="Real width of the drawing as it appears: the car's length in the side and "
                    "top views, its width in the front view")
    offset: bpy.props.FloatVectorProperty(
        name="Offset", size=3, subtype='TRANSLATION', unit='LENGTH',
        description="Where the drawing's origin goes (bottom middle; centre for the top view)")
    opacity: FloatProperty(name="Opacity", default=0.5, min=0.0, max=1.0)
    flip: BoolProperty(name="Mirror", default=False,
                       description="Flip the drawing left to right (the car faces the other way)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        try:
            image = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as e:
            self.report({'ERROR'}, "Cannot read the image: %s" % e)
            return {'CANCELLED'}
        ob = add_blueprint(context, image, self.view, self.length, tuple(self.offset),
                           self.opacity, self.flip)
        _make_active(context, ob)
        return {'FINISHED'}


classes = (SS_OT_add_blueprint,)
