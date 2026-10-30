"""Curve Network editing: move a crossing, fair curves (Bezier curves only).

Move Crossing drags a crossing of the network: both curves through it follow
exactly, their other crossings stay put, and each Bezier point moves with its
handles so the curves stay smooth (core.curveedit.pull_chain). Strokes repeat
until Enter like Soft Pull. Fair Curves irons the wiggles out of the selected
curves, keeping their ends, end tangents and crossings.

Edited curves get ALIGNED handles (positions kept): AUTO handles would be
recomputed by Blender and undo the edit.
"""
import bpy
import numpy as np
from bpy_extras.view3d_utils import (location_3d_to_region_2d, region_2d_to_origin_3d,
                                     region_2d_to_vector_3d)
from mathutils import Vector
from mathutils.geometry import intersect_line_plane

from . import pick
from .bridge.read_curves import read_polyline
from .core.curveedit import fair_chain, nearest_param, pull_chain
from .core.network import diagnose_network
from .operators import network_tolerance

PICK_PIXELS = 30.0
AXES = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}


def _bezier(ob):
    """The first spline of a curve object if it is an open Bezier spline."""
    if ob.type != 'CURVE' or not ob.data.splines:
        return None
    sp = ob.data.splines[0]
    if sp.type != 'BEZIER' or sp.use_cyclic_u or len(sp.bezier_points) < 2:
        return None
    return sp


def read_chain(ob):
    """World-space (anchors, left handles, right handles) of the first spline."""
    M = np.array(ob.matrix_world)
    world = lambda vs: np.array([tuple(v) for v in vs]) @ M[:3, :3].T + M[:3, 3]
    bp = _bezier(ob).bezier_points
    return (world([p.co for p in bp]), world([p.handle_left for p in bp]),
            world([p.handle_right for p in bp]))


def write_chain(ob, A, L, R):
    Minv = np.array(ob.matrix_world.inverted())
    local = lambda X: np.asarray(X) @ Minv[:3, :3].T + Minv[:3, 3]
    A, L, R = local(A), local(L), local(R)
    for i, p in enumerate(_bezier(ob).bezier_points):
        if p.handle_left_type in {'AUTO', 'VECTOR'}:
            p.handle_left_type = 'ALIGNED'
        if p.handle_right_type in {'AUTO', 'VECTOR'}:
            p.handle_right_type = 'ALIGNED'
        p.co, p.handle_left, p.handle_right = A[i], L[i], R[i]
    ob.data.update_tag()


def network_curves(context):
    """Selected curves (else every visible one) that Curve Network would read."""
    visible = [o for o in context.visible_objects if o.type == 'CURVE']
    return [o for o in visible if o.select_get()] or visible


def crossings(context, objects):
    """[(object a, object b, world point)] for every crossing among `objects`."""
    dg = context.evaluated_depsgraph_get()
    obs, polys = [], []
    for o in objects:
        try:
            polys.append(read_polyline(o, dg, 120))
            obs.append(o)
        except ValueError:
            pass
    if len(polys) < 2:
        return []
    d = diagnose_network(polys, network_tolerance(polys, context.scene.spline_surface))
    return [(obs[a], obs[b], p) for a, b, p, _ in d["crossings"]]


def _fixed_params(ob, chain, found, skip=None):
    """Parameters on `ob`'s chain of its crossings (except the point `skip`)."""
    out = []
    for a, b, p in found:
        if ob in (a, b) and (skip is None or np.linalg.norm(p - skip) > 1e-9):
            out.append(nearest_param(*chain, p))
    return out


class SS_OT_move_crossing(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.move_crossing"
    bl_label = "Move Crossing"
    bl_description = ("Drag a crossing of a curve network: both curves through it follow, their "
                      "other crossings stay. Bezier curves only. X/Y/Z: along that axis, "
                      "V: view plane, Ctrl+Z: undo a stroke, Enter: done, Esc: cancel all")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run Move Crossing in the 3D viewport")
            return {'CANCELLED'}
        self._curves = [o for o in network_curves(context) if _bezier(o)]
        self._found = crossings(context, self._curves)
        if not self._found:
            self.report({'ERROR'}, "No crossings between Bezier curves (Network Check shows why)")
            return {'CANCELLED'}
        self._orig = {o.name: read_chain(o) for o in self._curves}
        self._history = []  # [{name: chain before the stroke}]
        self._hover = None
        self._drag = None
        self._mode = 'VIEW'
        self._size = float(np.ptp(np.concatenate([c[0] for c in self._orig.values()]), axis=0).max())
        self._picker = pick.Overlay(context, self._lines)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        context.area.header_text_set(
            "Move Crossing: drag a crossing (%d strokes)   Move: %s   X Y Z axis / V view   "
            "Ctrl+Z: undo stroke   Enter/Space: done   Esc: cancel all"
            % (len(self._history), "view plane" if self._mode == 'VIEW' else self._mode + " axis"))

    def _lines(self):
        r = 0.015 * self._size
        ctx = bpy.context
        rot = ctx.region_data.view_rotation
        right, up = np.array(rot @ Vector((1, 0, 0))), np.array(rot @ Vector((0, 1, 0)))
        lines = []
        for i, (_, _, p) in enumerate(self._found):
            color = pick.COLOR_HOVER if i == self._hover else pick.COLOR_EDGE
            width = 4.0 if i == self._hover else 2.0
            lines += [(np.array([p - r * right, p + r * right]), color, width),
                      (np.array([p - r * up, p + r * up]), color, width)]
        if self._drag is not None:
            d = self._drag
            lines.append((np.array([d["point"], d["point"] + d["delta"]]), pick.COLOR_SELECTED, 3.0))
        return lines

    def _nearest(self, context, mouse):
        region, rv3d = context.region, context.region_data
        best, best_d = None, PICK_PIXELS
        for i, (_, _, p) in enumerate(self._found):
            s = location_3d_to_region_2d(region, rv3d, Vector(p))
            if s is not None:
                dist = np.hypot(s.x - mouse[0], s.y - mouse[1])
                if dist < best_d:
                    best, best_d = i, dist
        return best

    def _delta(self, context, event):
        region, rv3d = context.region, context.region_data
        pos = (event.mouse_region_x, event.mouse_region_y)
        o = region_2d_to_origin_3d(region, rv3d, pos)
        v = region_2d_to_vector_3d(region, rv3d, pos)
        a = Vector(self._drag["point"])
        if self._mode in AXES:
            n = Vector(AXES[self._mode])
            w = a - o
            b, dn, e = n.dot(v), n.dot(w), v.dot(w)
            den = 1.0 - b * b
            return np.array(n * ((b * e - dn) / den if abs(den) > 1e-4 else 0.0))
        hit = intersect_line_plane(o, o + v, a, Vector(rv3d.view_rotation @ Vector((0, 0, -1))))
        return np.array(hit - a) if hit is not None else np.zeros(3)

    def _start(self, context):
        a, b, p = self._found[self._hover]
        sides = []
        for ob in (a, b):
            chain = read_chain(ob)
            sides.append((ob, chain, nearest_param(*chain, p),
                          _fixed_params(ob, chain, self._found, skip=p)))
        self._drag = {"point": np.array(p), "delta": np.zeros(3), "sides": sides}

    def _apply(self, context, event):
        d = self._drag
        d["delta"] = self._delta(context, event)
        for ob, chain, t, fixed in d["sides"]:
            write_chain(ob, *pull_chain(*chain, t, d["delta"], fixed))

    def _restore(self, chains):
        for name, chain in chains.items():
            ob = bpy.data.objects.get(name)
            if ob is not None:
                write_chain(ob, *chain)

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self._drag:
                self._restore({ob.name: chain for ob, chain, _, _ in self._drag["sides"]})
                self._drag = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHTMOUSE':
                return {'RUNNING_MODAL'}
            self._restore({n: c for n, c in self._orig.items()
                           if any(n in h for h in self._history)})
            self._finish(context)
            return {'CANCELLED'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS' \
                and not self._drag:
            self._finish(context)
            return {'FINISHED'} if self._history else {'CANCELLED'}
        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            if self._history and not self._drag:
                self._restore(self._history.pop())
                self._found = crossings(context, self._curves)
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
                self._start(context)
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self._drag:
            d, self._drag = self._drag, None
            if np.any(d["delta"]):
                self._history.append({ob.name: chain for ob, chain, _, _ in d["sides"]})
            context.view_layer.update()
            self._found = crossings(context, self._curves)
            self._hover = None
            self._header(context)
            context.area.tag_redraw()
        elif event.type in {'X', 'Y', 'Z', 'V'} and event.value == 'PRESS':
            key = 'VIEW' if event.type == 'V' else event.type
            self._mode = 'VIEW' if key == self._mode else key
            if self._drag:
                self._apply(context, event)
            self._header(context)
        elif self._common_modal(context, event):
            return self._common_result
        return {'RUNNING_MODAL'}


class SS_OT_fair_curves(bpy.types.Operator):
    bl_idname = "spline_surface.fair_curves"
    bl_label = "Fair Curves"
    bl_description = ("Iron the wiggles out of the selected Bezier curves, keeping their ends, "
                      "end tangents and where they cross the other visible curves")
    bl_options = {'REGISTER', 'UNDO'}

    strength: bpy.props.FloatProperty(
        name="Strength", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
        description="0 leaves the curves as they are, 1 makes them as fair as the "
                    "fixed points allow")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(_bezier(o) for o in context.selected_objects)

    def execute(self, context):
        targets = [o for o in context.selected_objects if _bezier(o)]
        found = crossings(context, [o for o in context.visible_objects if o.type == 'CURVE'])
        worst = 0.0
        for ob in targets:
            chain = read_chain(ob)
            fixed = _fixed_params(ob, chain, found)
            new = fair_chain(*chain, fixed, self.strength)
            worst = max(worst, float(np.abs(new[0] - chain[0]).max()))
            write_chain(ob, *new)
        self.report({'INFO'}, "Faired %d curve(s) keeping their crossings, largest move %.3g"
                    % (len(targets), worst))
        return {'FINISHED'}


classes = (SS_OT_move_crossing, SS_OT_fair_curves)
