"""Soft Pull: click a point on a patch, drag, and the free control points around
it follow. Pinned points (rows next to G0/G1/G2 seams) never move, so every
seam keeps its continuity; the patch ends up like one edited by hand.

Strokes repeat like a sculpt mode until Enter/Space; Ctrl+Z takes back the
last stroke and Esc all of them. The whole session is one undo step.

L switches the origin from a point to an iso-line through it (D: which way,
T: taper the ends), so a whole line - a character line, an open edge - is
pulled with a band around it.
"""
import bpy
import numpy as np
from bpy_extras.view3d_utils import (location_3d_to_region_2d, region_2d_to_origin_3d,
                                     region_2d_to_vector_3d)
from mathutils import Vector
from mathutils.geometry import intersect_line_plane

from . import pick
from .core.nurbs import eval_points, eval_surface
from .core.softpull import line_pull, soft_pull
from .core.split import iso_line
from .ctrlpoints import pin_masks
from .live import _write_in_place
from .recipe import iso_lines, replace_net

GRID = 33          # surface samples per direction used for picking
PICK_PIXELS = 40.0
DEFAULT_RADIUS = 0.25  # of the patch size
EDGE_SNAP = 0.06   # a line this close (in parameter) to the boundary snaps onto it
AXES = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}
MODE_NAMES = {'NORMAL': "normal", 'VIEW': "view plane", 'X': "X axis", 'Y': "Y axis", 'Z': "Z axis"}


def _size(net):
    return float(np.ptp(np.asarray(net).reshape(-1, 3), axis=0).max())


def _circle(centre, radius, right, up, count=48):
    a = np.linspace(0.0, 2.0 * np.pi, count + 1)
    return centre + radius * (np.cos(a)[:, None] * right + np.sin(a)[:, None] * up)


class SS_OT_soft_pull(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.soft_pull"
    bl_label = "Soft Pull"
    bl_description = ("Click a point on a patch and drag: the free control points around it "
                      "follow. Rows pinned by seams stay put, so G0/G1/G2 are kept. "
                      "Wheel: radius, N: surface normal, X/Y/Z: that axis, V: view plane, "
                      "L: pull a whole line (D: direction, T: taper)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            o.type == 'SURFACE' for o in context.visible_objects)

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run Soft Pull in the 3D viewport")
            return {'CANCELLED'}
        obs = [o for o in context.visible_objects if o.type == 'SURFACE']
        selected = [o for o in obs if o.select_get()]
        self._patches = pin_masks(context, selected or obs)
        if not self._patches:
            self.report({'ERROR'}, "No patches to pull")
            return {'CANCELLED'}
        self._samples = {}
        for name in self._patches:
            self._sample(name)
        self._orig = {name: net for name, (net, _, _) in self._patches.items()}
        self._history = []   # (patch name, net before the stroke)
        self._size = max(_size(n) for n, _, _ in self._patches.values())
        self._radius = DEFAULT_RADIUS * self._size
        self._hover = None   # (name, u, v, point)
        self._drag = None    # dict while the button is held
        self._mode = 'NORMAL'
        self._line = False   # origin: a point, or the iso-line through it
        self._dir = 'U'      # line u = const ('U') or v = const ('V')
        self._taper = False
        self._picker = pick.Overlay(context, self._lines)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------ helpers

    def _header(self, context, extra=""):
        if self._line:
            origin = "L line %s=const (D)   T taper: %s   Wheel: width %.3g" % (
                self._dir.lower(), "on" if self._taper else "off", self._radius)
        else:
            origin = "L point   Wheel: radius %.3g" % self._radius
        context.area.header_text_set(
            "Soft Pull: drag (%d strokes)   Origin: %s   Move: %s   N / X Y Z / V   "
            "Ctrl+Z: undo stroke   Enter/Space: done   Esc: cancel all%s"
            % (len(self._history), origin, MODE_NAMES[self._mode], extra))

    def _sample(self, name):
        net, orders, _ = self._patches[name]
        g = np.linspace(0.0, 1.0, GRID)
        pts = eval_surface(net, orders, g, g)
        self._samples[name] = (pts.reshape(-1, 3), pts.shape[:2])

    def _set_net(self, context, name, net):
        """Make `net` the patch's current shape, in the object and for picking."""
        _, orders, mask = self._patches[name]
        self._patches[name] = (net, orders, mask)
        self._sample(name)
        ob = bpy.data.objects.get(name)
        if ob is not None:
            self._write(context, ob, net, orders)

    def _view_axes(self, context):
        rot = context.region_data.view_rotation
        return np.array(rot @ Vector((1, 0, 0))), np.array(rot @ Vector((0, 1, 0)))

    def _line_param(self, u, v):
        """(t0, along) of the iso-line through (u, v), snapped onto a boundary."""
        t0, along = (u, v) if self._dir == 'U' else (v, u)
        if t0 < EDGE_SNAP:
            t0 = 0.0
        elif t0 > 1.0 - EDGE_SNAP:
            t0 = 1.0
        return t0, along

    def _lines(self):
        ctx = bpy.context
        if self._hover is None and self._drag is None:
            return []
        right, up = self._view_axes(ctx)
        if self._drag is not None:
            d = self._drag
            lines = [(pts, pick.COLOR_PREVIEW, 1.5) for pts in iso_lines(d["net"], d["orders"], 8)]
            if "t0" in d:
                lines.append((iso_line(d["net"], d["orders"], d["dir"], d["t0"]),
                              pick.COLOR_HOVER, 3.0))
            else:
                lines.append((_circle(d["anchor"], self._radius, right, up), pick.COLOR_HOVER, 2.0))
            lines.append((np.array([d["anchor"], d["anchor"] + d["delta"]]),
                          pick.COLOR_SELECTED, 3.0))
            return lines
        if self._line:
            name, u, v, _ = self._hover
            net, orders, _ = self._patches[name]
            t0 = self._line_param(u, v)[0]
            return [(iso_line(net, orders, self._dir, t0), pick.COLOR_HOVER, 3.0)]
        return [(_circle(self._hover[3], self._radius, right, up), pick.COLOR_HOVER, 2.0)]

    def _nearest(self, context, mouse):
        region, rv3d = context.region, context.region_data
        m = np.array(mouse, float)
        best, best_d = None, PICK_PIXELS
        for name, (pts, shape) in self._samples.items():
            scr = np.array([(p.x, p.y) if p is not None else (1e9, 1e9)
                            for p in (location_3d_to_region_2d(region, rv3d, Vector(q))
                                      for q in pts)])
            d = np.linalg.norm(scr - m, axis=1)
            k = int(d.argmin())
            if d[k] < best_d:
                i, j = divmod(k, shape[1])
                best, best_d = (name, i / (shape[0] - 1), j / (shape[1] - 1), pts[k]), d[k]
        return best

    def _delta(self, context, event):
        region, rv3d = context.region, context.region_data
        pos = (event.mouse_region_x, event.mouse_region_y)
        o = region_2d_to_origin_3d(region, rv3d, pos)
        d = region_2d_to_vector_3d(region, rv3d, pos)
        a = Vector(self._drag["anchor"])
        if self._mode != 'VIEW':
            n = Vector(self._drag["normal"] if self._mode == 'NORMAL' else AXES[self._mode])
            w = a - o
            b, dn, e = n.dot(d), n.dot(w), d.dot(w)
            den = 1.0 - b * b
            t = (b * e - dn) / den if abs(den) > 1e-4 else 0.0
            return np.array(n * t)
        view = Vector(rv3d.view_rotation @ Vector((0, 0, -1)))
        hit = intersect_line_plane(o, o + d, a, view)
        return np.array(hit - a) if hit is not None else np.zeros(3)

    def _write(self, context, ob, net, orders):
        if not _write_in_place(ob, net, orders):
            replace_net(ob, net, context.scene.spline_surface, orders)

    def _start_drag(self, context):
        name, u, v, point = self._hover
        net, orders, _ = self._patches[name]
        line = {}
        if self._line:
            t0, along = self._line_param(u, v)
            u, v = (t0, along) if self._dir == 'U' else (along, t0)
            point = eval_points(net, orders, [u], [v])[0]
            line = {"t0": t0, "along": along, "dir": self._dir}
        # Just inside the boundary, so a snapped edge still has a usable normal.
        cu, cv = np.clip(u, 1e-4, 1 - 1e-4), np.clip(v, 1e-4, 1 - 1e-4)
        du = eval_points(net, orders, [cu], [cv], 1, 0)[0]
        dv = eval_points(net, orders, [cu], [cv], 0, 1)[0]
        normal = np.cross(du, dv)
        length = np.linalg.norm(normal)
        normal = normal / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])
        self._drag = {"name": name, "uv": (u, v), "anchor": np.array(point), "normal": normal,
                      "delta": np.zeros(3), "net": net, "orders": orders, "gain": 1.0, **line}

    def _apply(self, context, event):
        d = self._drag
        net, orders, mask = self._patches[d["name"]]
        d["delta"] = self._delta(context, event)
        if "t0" in d:
            d["net"], d["gain"] = line_pull(net, orders, mask, d["dir"], d["t0"], d["along"],
                                            d["delta"], self._radius, self._taper)
        else:
            d["net"], d["gain"] = soft_pull(net, orders, mask, d["uv"], d["delta"], self._radius)
        ob = bpy.data.objects.get(d["name"])
        if ob is not None:
            self._write(context, ob, d["net"], orders)
        self._header(context, "   (anchor is pinned)" if d["gain"] == 0.0 else "")

    def _revert(self, context):
        d = self._drag
        net, orders, _ = self._patches[d["name"]]
        ob = bpy.data.objects.get(d["name"])
        if ob is not None:
            self._write(context, ob, net, orders)
        self._drag = None

    # ------------------------------------------------------------ modal

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self._drag:  # drop just the stroke in progress
                self._revert(context)
                self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHTMOUSE':
                return {'RUNNING_MODAL'}
            for name in {n for n, _ in self._history}:
                self._set_net(context, name, self._orig[name])
            self._finish(context)
            return {'CANCELLED'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS' \
                and not self._drag:
            self._finish(context)
            return {'FINISHED'} if self._history else {'CANCELLED'}
        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            if self._history and not self._drag:
                name, net = self._history.pop()
                self._set_net(context, name, net)
                self._header(context)
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE':
            if self._drag:
                self._apply(context, event)
            else:
                self._hover = self._nearest(context, (event.mouse_region_x, event.mouse_region_y))
            context.area.tag_redraw()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' and not self._drag:
            if self._hover is not None:
                self._start_drag(context)
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self._drag:
            d, self._drag = self._drag, None
            if d["gain"] > 0.0 and np.any(d["delta"]):
                self._history.append((d["name"], self._patches[d["name"]][0]))
                self._set_net(context, d["name"], d["net"])
            else:
                self._set_net(context, d["name"], self._patches[d["name"]][0])
            self._hover = self._nearest(context, (event.mouse_region_x, event.mouse_region_y))
            self._header(context)
            context.area.tag_redraw()
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            self._radius *= 1.1 if event.type == 'WHEELUPMOUSE' else 1 / 1.1
            if self._drag:
                self._apply(context, event)
            self._header(context)
            context.area.tag_redraw()
        elif event.type in {'L', 'D'} and event.value == 'PRESS' and not self._drag:
            if event.type == 'L':
                self._line = not self._line
            else:
                self._dir = 'V' if self._dir == 'U' else 'U'
            self._header(context)
            context.area.tag_redraw()
        elif event.type == 'T' and event.value == 'PRESS':
            self._taper = not self._taper
            if self._drag:
                self._apply(context, event)
            self._header(context)
            context.area.tag_redraw()
        elif event.type in {'N', 'V', 'X', 'Y', 'Z'} and event.value == 'PRESS':
            key = {'N': 'NORMAL', 'V': 'VIEW'}.get(event.type, event.type)
            # pressing the active axis again goes back to the surface normal
            self._mode = 'NORMAL' if key == self._mode and key in AXES else key
            if self._drag:
                self._apply(context, event)
            self._header(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}


classes = (SS_OT_soft_pull,)
