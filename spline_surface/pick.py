"""Viewport overlay and screen-space picking for modal operators."""

import blf
import bpy
import gpu
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

COLOR_EDGE = (0.3, 0.7, 1.0, 0.8)
COLOR_HOVER = (1.0, 0.6, 0.1, 1.0)
COLOR_SELECTED = (0.2, 1.0, 0.4, 1.0)
COLOR_PREVIEW = (1.0, 0.85, 0.4, 0.6)
PICK_RADIUS = 30.0  # pixels


class Overlay:
    """Draws polylines from `get_lines()` -> [(points, rgba, width)] in the 3D view,
    plus an optional text label from `get_label()` -> (text, 3D anchor) or None."""

    def __init__(self, context, get_lines, get_label=None):
        self.get_lines = get_lines
        self.get_label = get_label
        self.handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (), 'WINDOW', 'POST_VIEW')
        self.text_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_text, (), 'WINDOW', 'POST_PIXEL') if get_label else None
        context.area.tag_redraw()

    def _draw_text(self):
        label = self.get_label()
        if not label:
            return
        text, anchor = label
        ctx = bpy.context
        p = location_3d_to_region_2d(ctx.region, ctx.region_data, anchor)
        if p is None:
            return
        size = int(13 * ctx.preferences.system.ui_scale)
        blf.size(0, size)
        w, h = blf.dimensions(0, text)
        x, y = p[0] + 12, p[1] + 12
        # Dark backing so the label reads on any surface colour.
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        pad = 4
        verts = [(x - pad, y - pad), (x + w + pad, y - pad),
                 (x + w + pad, y + h + pad), (x - pad, y + h + pad)]
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
        gpu.state.blend_set('ALPHA')
        shader.uniform_float("color", (0.0, 0.0, 0.0, 0.6))
        batch.draw(shader)
        gpu.state.blend_set('NONE')
        blf.position(0, x, y, 0)
        blf.color(0, *COLOR_HOVER)
        blf.draw(0, text)

    def _draw(self):
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        for pts, color, width in self.get_lines():
            pts = [tuple(p) for p in pts]
            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
            shader.uniform_float("color", color)
            shader.uniform_float("lineWidth", width)
            batch.draw(shader)
        gpu.state.blend_set('NONE')

    def remove(self):
        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
            self.handle = None
        if self.text_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.text_handle, 'WINDOW')
            self.text_handle = None


class ModalPicker:
    """Shared boilerplate for modal operators that run a pick.Overlay:
    self._finish() and the Esc/RMB-cancel, middle-mouse/NDOF/trackpad
    pass-through tail every one of them repeated identically. A subclass's
    own modal() still owns its own hover/pick logic; it only needs to end
    its elif chain with:
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}
    """

    def _finish(self, context):
        self._picker.remove()
        context.area.header_text_set(None)
        context.area.tag_redraw()

    def _common_modal(self, context, event):
        """True (and sets self._common_result) if this event was the shared
        cancel/pass-through case; the caller returns self._common_result."""
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._finish(context)
            self._common_result = {'CANCELLED'}
            return True
        if event.type in {'MIDDLEMOUSE', 'NUMPAD_PERIOD'} or event.type.startswith('NDOF') \
                or event.type.startswith('TRACKPAD'):
            self._common_result = {'PASS_THROUGH'}
            return True
        return False


EDGE_NAMES = {'U0': "U min", 'U1': "U max", 'V0': "V min", 'V1': "V max", 'CURVE': "curve"}


def edge_label(key, points):
    """Label for a hovered (object name, edge) key, anchored mid-edge."""
    if key is None:
        return None
    return "%s  %s" % (key[0], EDGE_NAMES.get(key[1], key[1])), points[len(points) // 2]


def _segment_distance(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + t * ab - p)


def nearest_polyline(context, polylines, mouse, radius=PICK_RADIUS):
    """Key of the polyline closest to the mouse on screen, or None if all are far."""
    region, rv3d = context.region, context.region_data
    p = np.array(mouse, dtype=float)
    best, best_d = None, radius
    for key, pts in polylines.items():
        screen = [location_3d_to_region_2d(region, rv3d, q) for q in pts]
        for a, b in zip(screen, screen[1:]):
            if a is None or b is None:
                continue
            d = _segment_distance(p, np.array(a), np.array(b))
            if d < best_d:
                best, best_d = key, d
    return best
