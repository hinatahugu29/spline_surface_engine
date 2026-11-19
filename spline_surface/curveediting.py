"""Curve editing beyond moving points and handles one by one.

  Curve Points (object or edit mode, modal): click on a curve to add a point
      (exact, core.curvepoint), drag on a curve to pull it with a soft falloff
      along the curve (core.curvepull; crossing curves of a network follow),
      drag a point to slide it along the curve, Ctrl+click a point to remove it
      (its neighbours keep their tangents). Strokes stack; Ctrl+Z takes one
      back, Enter keeps them, Esc drops them all.
  Slide Points (edit mode, modal): the selected points slide along the curve
      as the mouse moves, like Vertex Slide on a mesh.
  Dissolve Points (edit mode): remove the selected points, refitting the
      curve so it keeps its shape and its neighbours' tangents.

Edited splines get ALIGNED / FREE handles: Blender recomputes AUTO and VECTOR
handles from the neighbours, which would undo the edit.
"""
import bpy
import numpy as np
from bpy_extras.view3d_utils import (location_3d_to_region_2d, region_2d_to_location_3d,
                                     region_2d_to_origin_3d, region_2d_to_vector_3d)
from mathutils import Vector
from mathutils.geometry import intersect_line_plane

from . import pick
from .core.curvepoint import (arc_param, arc_table, dissolve_point, insert_point, nearest,
                              point_at, segments, slide_point)
from .core.curvepull import chain_plane, pull_curve, smooth_points
from .core.softpull import PROFILES

AXES = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}

POINT_PIXELS = 12.0        # how near an anchor counts as "on the point"
DRAG_PIXELS = 4.0          # movement that turns a click into a drag
PER_SEGMENT = 24           # screen samples per segment for picking and drawing


# ---------------------------------------------------------------- curve data
_SPLINE_PROPS = ("type", "use_cyclic_u", "use_smooth", "resolution_u", "material_index",
                 "tilt_interpolation", "radius_interpolation", "order_u", "use_endpoint_u",
                 "use_bezier_u", "hide")
_BEZ_PROPS = ("handle_left_type", "handle_right_type", "tilt", "radius", "weight_softbody",
              "select_control_point", "select_left_handle", "select_right_handle", "hide")
_PT_PROPS = ("tilt", "radius", "weight_softbody", "weight", "select", "hide")


def snapshot(ob):
    """Everything about the object's splines needed to rebuild them."""
    out = []
    for sp in ob.data.splines:
        d = {p: getattr(sp, p) for p in _SPLINE_PROPS if hasattr(sp, p)}
        if sp.type == 'BEZIER':
            d["points"] = [dict({p: getattr(b, p) for p in _BEZ_PROPS},
                                co=tuple(b.co), handle_left=tuple(b.handle_left),
                                handle_right=tuple(b.handle_right))
                           for b in sp.bezier_points]
        else:
            d["points"] = [dict({p: getattr(b, p) for p in _PT_PROPS if hasattr(b, p)},
                                co=tuple(b.co)) for b in sp.points]
        out.append(d)
    return out


def restore(ob, snap):
    """Rebuild the object's splines from a snapshot (same order as before)."""
    cu = ob.data
    cu.splines.clear()
    for d in snap:
        sp = cu.splines.new(d["type"])
        pts = d["points"]
        if d["type"] == 'BEZIER':
            sp.bezier_points.add(len(pts) - 1)
            for b, p in zip(sp.bezier_points, pts):
                b.handle_left_type = b.handle_right_type = 'FREE'
                b.co, b.handle_left, b.handle_right = p["co"], p["handle_left"], p["handle_right"]
                for k in _BEZ_PROPS:
                    setattr(b, k, p[k])
        else:
            sp.points.add(len(pts) - 1)
            for b, p in zip(sp.points, pts):
                b.co = p["co"]
                for k in _PT_PROPS:
                    if k in p:
                        setattr(b, k, p[k])
        for k, v in d.items():
            if k not in ("type", "points"):
                try:
                    setattr(sp, k, v)
                except (AttributeError, TypeError):
                    pass
    cu.update_tag()


def _world(ob):
    M = np.array(ob.matrix_world)
    return M[:3, :3], M[:3, 3]


def read_spline(ob, idx):
    """World (A, L, R, cyclic) of Bezier spline idx."""
    sp = ob.data.splines[idx]
    R3, t = _world(ob)
    get = lambda attr: np.array([tuple(getattr(b, attr)) for b in sp.bezier_points]) @ R3.T + t
    return get("co"), get("handle_left"), get("handle_right"), sp.use_cyclic_u


def bezier_splines(ob):
    return [i for i, sp in enumerate(ob.data.splines)
            if sp.type == 'BEZIER' and len(sp.bezier_points) >= 2]


def _collinear(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return na > 1e-9 and nb > 1e-9 and np.dot(a, b) > (1 - 1e-7) * na * nb


def write_spline(ob, idx, A, L, R, inserted=None, removed=None):
    """Write world (A, L, R) to Bezier spline idx; `inserted` / `removed` is the
    index of an anchor added / taken away (the count changes)."""
    Minv = np.linalg.inv(np.array(ob.matrix_world))
    loc = lambda X: np.asarray(X, float) @ Minv[:3, :3].T + Minv[:3, 3]
    A, L, R = loc(A), loc(L), loc(R)
    sp = ob.data.splines[idx]
    if len(A) != len(sp.bezier_points):
        snap = snapshot(ob)
        pts = snap[idx]["points"]
        if inserted is not None:
            a, b = pts[inserted - 1], pts[inserted % len(pts)]
            new = dict(a)
            for k in ("tilt", "radius", "weight_softbody"):
                new[k] = 0.5 * (a[k] + b[k])
            for k in ("select_control_point", "select_left_handle", "select_right_handle"):
                new[k] = False
            pts.insert(inserted, new)
        elif removed is not None:
            pts.pop(removed)
        for p, a, l, r in zip(pts, A, L, R):
            p["co"], p["handle_left"], p["handle_right"] = tuple(a), tuple(l), tuple(r)
        restore(ob, snap)
        sp = ob.data.splines[idx]
    for b, a, l, r in zip(sp.bezier_points, A, L, R):
        b.handle_left_type = b.handle_right_type = 'FREE'
        b.co, b.handle_left, b.handle_right = a, l, r
    for b, a, l, r in zip(sp.bezier_points, A, L, R):
        if _collinear(a - l, r - a):
            b.handle_left_type = b.handle_right_type = 'ALIGNED'
    ob.data.update_tag()


def target_objects(context):
    """Curve objects to work on: the ones in edit mode, else the selected
    visible curves, else every visible curve."""
    if context.mode == 'EDIT_CURVE':
        return [o for o in context.objects_in_mode if o.type == 'CURVE']
    vis = [o for o in context.visible_objects if o.type == 'CURVE' and bezier_splines(o)]
    return [o for o in vis if o.select_get()] or vis


# ---------------------------------------------------------------- Curve Points
class SS_OT_curve_points(pick.ModalPicker, bpy.types.Operator):
    bl_idname = "spline_surface.curve_points"
    bl_label = "Curve Points"
    bl_description = ("Edit Bezier curves in the viewport. Click on a curve: add a point (the "
                      "shape does not change). Drag on a curve: pull it, the rest follows softly "
                      "(Wheel: range, F: falloff, E: pin the ends, N: crossing curves follow, "
                      "V / P / X / Y / Z: view plane / the curve's own plane / an axis). Drag a "
                      "point: slide it along the curve. Ctrl+click a point: remove it, keeping "
                      "the shape. Ctrl+Z takes back one edit, Enter keeps them, Esc drops all")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_CURVE'}

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run Curve Points in the 3D viewport")
            return {'CANCELLED'}
        self._obs = [o.name for o in target_objects(context)]
        if not self._obs:
            self.report({'ERROR'}, "No Bezier curve to edit")
            return {'CANCELLED'}
        self._orig = {n: snapshot(bpy.data.objects[n]) for n in self._obs}
        self._history = []          # strokes: [[(object name, snapshot before)], ...]
        self._chains = {}
        self._read_all()
        size = max((float(np.ptp(c[0], axis=0).max()) for c in self._chains.values()), default=1.0)
        self._radius = 0.3 * (size or 1.0)
        self._profile = 'SMOOTH'
        self._pin = True
        self._net = True
        self._move = 'VIEW'         # VIEW, PLANE (the curve's own, when flat), X, Y, Z
        self._hover = None          # ('POINT', key, i, world) or ('CURVE', key, t, world)
        self._press = None
        self._slide = None
        self._pull = None
        self._note = ""
        self._picker = pick.Overlay(context, self._lines, self._label)
        context.window_manager.modal_handler_add(self)
        self._header(context)
        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------ data
    def _read_all(self):
        self._chains = {}
        for n in self._obs:
            ob = bpy.data.objects.get(n)
            if ob is None:
                continue
            for i in bezier_splines(ob):
                self._chains[(n, i)] = read_spline(ob, i)

    def _samples(self, key):
        A, L, R, cyc = self._chains[key]
        m = segments(len(A), cyc)
        t = np.linspace(0.0, m, m * PER_SEGMENT + 1)
        return t, point_at(A, L, R, t, cyc)

    def _header(self, context):
        move = {'VIEW': "view plane", 'PLANE': "curve's plane"}.get(self._move, self._move)
        context.area.header_text_set(
            "Curve Points (%d edits): click curve: add point   drag curve: pull   drag point: "
            "slide   Ctrl+click point: remove   Wheel: range %.3g   F: %s   E: ends %s   "
            "N: crossings %s   V/P/X/Y/Z: %s   Ctrl+Z / Enter / Esc%s"
            % (len(self._history), self._radius, self._profile.lower(),
               "pinned" if self._pin else "free", "follow" if self._net else "stay", move,
               self._note))

    # ------------------------------------------------------------ drawing
    def _reach(self, key, t):
        """Curve points within the pull range of parameter t (for the preview)."""
        A, L, R, cyc = self._chains[key]
        tt, ss = arc_table(A, L, R, cyc, 16)
        d = np.abs(ss - np.interp(t, tt, ss))
        if cyc:
            d = np.minimum(d, ss[-1] - d)
        pts = point_at(A, L, R, tt, cyc)
        inside = d < self._radius
        runs, cur = [], []
        for p, ok in zip(pts, inside):
            if ok:
                cur.append(p)
            elif cur:
                runs.append(np.array(cur))
                cur = []
        if cur:
            runs.append(np.array(cur))
        return [r for r in runs if len(r) > 1]

    def _lines(self):
        out = []
        for key in self._chains:
            out.append((self._samples(key)[1], pick.COLOR_EDGE, 1.5))
        for key, (A, L, R, cyc) in self._chains.items():
            for a in A:
                out.append((_cross(a, self._mark_size), pick.COLOR_EDGE, 2.0))
        hover = self._hover
        if self._pull:
            for run in self._reach(self._pull["key"], self._pull["t"]):
                out.append((run, pick.COLOR_SELECTED, 3.0))
        elif hover is not None and hover[0] == 'CURVE':
            for run in self._reach(hover[1], hover[2]):
                out.append((run, pick.COLOR_PREVIEW, 2.5))
        if hover is not None:
            kind, key, _, p = hover
            col = pick.COLOR_SELECTED if (self._slide or self._pull) else pick.COLOR_HOVER
            out.append((_cross(p, self._mark_size * (1.6 if kind == 'POINT' else 1.2)), col, 3.0))
        return out

    def _label(self):
        if self._hover is None or self._slide or self._pull:
            return None
        kind, key, _, p = self._hover
        text = ("drag: slide   Ctrl+click: remove" if kind == 'POINT'
                else "click: add a point   drag: pull")
        return text, p

    # ------------------------------------------------------------ picking
    def _screen(self, context, pts):
        region, rv3d = context.region, context.region_data
        out = np.full((len(pts), 2), 1e9)
        for i, q in enumerate(pts):
            s = location_3d_to_region_2d(region, rv3d, Vector(q))
            if s is not None:
                out[i] = (s.x, s.y)
        return out

    def _pick(self, context, mouse):
        m = np.array(mouse, float)
        best, best_d = None, POINT_PIXELS
        for key, (A, L, R, cyc) in self._chains.items():
            d = np.linalg.norm(self._screen(context, A) - m, axis=1)
            i = int(d.argmin())
            if d[i] < best_d:
                best, best_d = ('POINT', key, i, A[i]), d[i]
        if best is not None:
            return best
        best_d = pick.PICK_RADIUS
        for key in self._chains:
            t, pts = self._samples(key)
            scr = self._screen(context, pts)
            d = _seg_dist(scr, m)
            j = int(d.argmin())
            if d[j] < best_d:
                f = _seg_frac(scr[j], scr[j + 1], m)
                tt = t[j] + f * (t[j + 1] - t[j])
                A, L, R, cyc = self._chains[key]
                best, best_d = ('CURVE', key, tt, point_at(A, L, R, [tt], cyc)[0]), d[j]
        return best

    def _set_mark_size(self, context):
        region, rv3d = context.region, context.region_data
        if self._hover is None:
            return
        p = Vector(self._hover[3])
        s = location_3d_to_region_2d(region, rv3d, p)
        if s is None:
            return
        q = region_2d_to_location_3d(region, rv3d, (s.x + 6, s.y), p)
        self._mark_size = (q - p).length

    _mark_size = 0.02

    # ------------------------------------------------------------ history
    def _begin(self):
        self._history.append([])

    def _touch(self, name):
        """Remember the object's state before the current stroke first changes it."""
        stroke = self._history[-1]
        if all(n != name for n, _ in stroke):
            stroke.append((name, snapshot(bpy.data.objects[name])))

    def _undo(self, context):
        if not self._history:
            return
        for name, snap in reversed(self._history.pop()):
            ob = bpy.data.objects.get(name)
            if ob is not None:
                restore(ob, snap)
        self._read_all()
        self._hover = None
        self._note = "   (undone)"

    def _cancel_all(self, context):
        touched = {n for stroke in self._history for n, _ in stroke}
        for name in touched:
            ob = bpy.data.objects.get(name)
            if ob is not None:
                restore(ob, self._orig[name])

    # ------------------------------------------------------------ insert / dissolve
    def _insert(self, context, key, t):
        name, idx = key
        A, L, R, cyc = self._chains[key]
        A2, L2, R2, i = insert_point(A, L, R, t, cyc)
        if len(A2) == len(A):
            return
        self._begin()
        self._touch(name)
        write_spline(bpy.data.objects[name], idx, A2, L2, R2, inserted=i)
        self._note = "   (point added, shape unchanged)"
        self._read_all()

    def _dissolve(self, context, key, i):
        name, idx = key
        A, L, R, cyc = self._chains[key]
        if len(A) <= (3 if cyc else 2):
            self._note = "   (too few points to remove one)"
            return
        A2, L2, R2, err = dissolve_point(A, L, R, i, cyc)
        self._begin()
        self._touch(name)
        write_spline(bpy.data.objects[name], idx, A2, L2, R2, removed=i)
        self._note = "   (point removed, shape off by %.3g)" % err
        self._read_all()

    # ------------------------------------------------------------ slide
    def _start_slide(self, context):
        kind, key, i, p = self._press["hover"]
        self._begin()
        self._touch(key[0])
        A, L, R, cyc = self._chains[key]
        t, pts = self._samples(key)
        self._slide = {"key": key, "i": i, "chain": (A, L, R, cyc), "t": t, "pts": pts}

    def _drag_slide(self, context, mouse):
        s = self._slide
        A, L, R, cyc = s["chain"]
        i, n = s["i"], len(A)
        m = segments(n, cyc)
        t, pts = s["t"], s["pts"]
        # only the stretch between the neighbours: the point cannot pass them
        if cyc:
            tt = np.where(t > m - 1.0, t - m, t) if i == 0 else t
            sel = (tt > i - 1.0) & (tt < i + 1.0)
            tsel, psel = tt[sel], pts[sel]
            order = np.argsort(tsel)
            tsel, psel = tsel[order], psel[order]
        else:
            sel = (t > i - 1.0) & (t < i + 1.0)
            tsel, psel = t[sel], pts[sel]
        if len(tsel) < 2:
            return
        scr = self._screen(context, psel)
        mm = np.array(mouse, float)
        d = _seg_dist(scr, mm)
        j = int(d.argmin())
        f = _seg_frac(scr[j], scr[j + 1], mm)
        target = tsel[j] + f * (tsel[j + 1] - tsel[j])
        A2, L2, R2, err = slide_point(A, L, R, i, target, cyc, samples=24)
        name, idx = s["key"]
        write_spline(bpy.data.objects[name], idx, A2, L2, R2)
        self._chains[s["key"]] = (A2, L2, R2, cyc)
        self._hover = ('POINT', s["key"], i, A2[i])
        self._note = "   (sliding: shape off by %.3g)" % err

    # ------------------------------------------------------------ pull
    def _crossing_links(self, context, key, A, L, R, cyc):
        """[(own parameter, partner key, partner parameter, partner's other crossings)]
        for the curves crossing this one (single-spline curves, as Curve Network reads)."""
        from .curvetools import crossings
        name, idx = key
        if idx != 0:
            return []
        obs = [bpy.data.objects[n] for n in self._obs
               if bpy.data.objects.get(n) is not None and len(bezier_splines(bpy.data.objects[n])) == 1
               and bezier_splines(bpy.data.objects[n])[0] == 0]
        me = bpy.data.objects[name]
        if me not in obs or len(obs) < 2:
            return []
        found = crossings(context, obs)
        out = []
        for a, b, p in found:
            if me not in (a, b):
                continue
            other = b if a is me else a
            okey = (other.name, 0)
            if okey not in self._chains:
                continue
            oA, oL, oR, ocyc = self._chains[okey]
            t_own = nearest(A, L, R, p, cyc)[0]
            t_other = nearest(oA, oL, oR, p, ocyc)[0]
            rest = [nearest(oA, oL, oR, q, ocyc)[0] for x, y, q in found
                    if other in (x, y) and me not in (x, y)]
            out.append((t_own, okey, t_other, rest))
        return out

    def _start_pull(self, context):
        _, key, t, p = self._press["hover"]
        A, L, R, cyc = self._chains[key]
        self._begin()
        self._touch(key[0])
        plane = chain_plane(A, L, R)
        if self._move == 'PLANE' and plane is None:
            self._move = 'VIEW'
        self._pull = {"key": key, "t": t, "anchor": np.array(p), "chain": (A, L, R, cyc),
                      "plane": plane, "smooth": smooth_points(A, L, R),
                      "links": self._crossing_links(context, key, A, L, R, cyc) if self._net else []}

    def _delta(self, context, mouse):
        region, rv3d = context.region, context.region_data
        o = region_2d_to_origin_3d(region, rv3d, mouse)
        d = region_2d_to_vector_3d(region, rv3d, mouse)
        a = Vector(self._pull["anchor"])
        if self._move in AXES:
            n = Vector(AXES[self._move])
            w = a - o
            b, dn, e = n.dot(d), n.dot(w), d.dot(w)
            den = 1.0 - b * b
            t = (b * e - dn) / den if abs(den) > 1e-4 else 0.0
            return np.array(n * t)
        normal = Vector(rv3d.view_rotation @ Vector((0, 0, -1)))
        if self._move == 'PLANE' and self._pull["plane"] is not None:
            pn = Vector(self._pull["plane"][1])
            if abs(pn.dot(d)) > 0.05:          # not looking along the plane
                normal = pn
        hit = intersect_line_plane(o, o + d, a, normal)
        return np.array(hit - a) if hit is not None else np.zeros(3)

    def _drag_pull(self, context, mouse):
        P = self._pull
        A, L, R, cyc = P["chain"]
        delta = self._delta(context, mouse)
        A2, L2, R2 = pull_curve(A, L, R, P["t"], delta, self._radius, cyc, self._profile,
                                self._pin, keep_smooth=P["smooth"])
        name, idx = P["key"]
        write_spline(bpy.data.objects[name], idx, A2, L2, R2)
        moved = 0
        for t_own, okey, t_other, rest in P["links"]:
            disp = point_at(A2, L2, R2, [t_own], cyc)[0] - point_at(A, L, R, [t_own], cyc)[0]
            if np.linalg.norm(disp) < 1e-9:
                continue
            oA, oL, oR, ocyc = self._chains[okey]
            self._touch(okey[0])
            o2 = pull_curve(oA, oL, oR, t_other, disp, self._radius, ocyc, self._profile,
                            self._pin, fixed=rest)
            write_spline(bpy.data.objects[okey[0]], okey[1], *o2)
            moved += 1
        self._hover = ('CURVE', P["key"], P["t"], point_at(A2, L2, R2, [P["t"]], cyc)[0])
        self._note = "   (pulling %.3g%s)" % (float(np.linalg.norm(delta)),
                                               ", %d crossing curve(s) follow" % moved if moved else "")

    # ------------------------------------------------------------ modal
    def _keys(self, context, event):
        """Option keys; True if the event was one."""
        if event.value != 'PRESS':
            return False
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            self._radius *= 1.15 if event.type == 'WHEELUPMOUSE' else 1 / 1.15
        elif event.type == 'F':
            self._profile = PROFILES[(PROFILES.index(self._profile) + 1) % len(PROFILES)]
        elif event.type == 'E':
            self._pin = not self._pin
        elif event.type == 'N':
            self._net = not self._net
        elif event.type in {'V', 'P', 'X', 'Y', 'Z'} and not event.ctrl:
            self._move = {'V': 'VIEW', 'P': 'PLANE'}.get(event.type, event.type)
        else:
            return False
        if self._pull and event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'F', 'E'}:
            self._drag_pull(context, self._mouse)
        self._header(context)
        context.area.tag_redraw()
        return True

    def modal(self, context, event):
        mouse = (event.mouse_region_x, event.mouse_region_y)
        self._mouse = mouse
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self._slide or self._pull:       # drop just the stroke in progress
                self._undo(context)
                self._slide = self._pull = self._press = None
                self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHTMOUSE':
                return {'RUNNING_MODAL'}
            self._cancel_all(context)
            self._finish(context)
            return {'CANCELLED'}
        busy = self._slide or self._pull
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS' and not busy:
            self._finish(context)
            return {'FINISHED'} if self._history else {'CANCELLED'}
        if event.type == 'Z' and event.ctrl and event.value == 'PRESS' and not busy:
            self._undo(context)
            self._header(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._keys(context, event):
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE':
            if self._slide:
                self._drag_slide(context, mouse)
                self._header(context)
            elif self._pull:
                self._drag_pull(context, mouse)
                self._header(context)
            elif self._press is not None:
                moved = np.linalg.norm(np.subtract(mouse, self._press["mouse"]))
                if moved > DRAG_PIXELS:
                    if self._press["hover"][0] == 'POINT':
                        self._start_slide(context)
                        self._drag_slide(context, mouse)
                    else:
                        self._start_pull(context)
                        self._drag_pull(context, mouse)
                    self._header(context)
            else:
                self._hover = self._pick(context, mouse)
                self._set_mark_size(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._hover = self._pick(context, mouse)
            self._set_mark_size(context)
            if self._hover is None:
                return {'RUNNING_MODAL'}
            kind, key, where, _ = self._hover
            if kind == 'POINT' and event.ctrl:
                self._dissolve(context, key, where)
                self._hover = None
                self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            self._press = {"mouse": mouse, "hover": self._hover}
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self._slide or self._pull:
                self._slide = self._pull = None
                self._read_all()
            elif self._press is not None and self._press["hover"][0] == 'CURVE':
                _, key, t, _ = self._press["hover"]
                self._insert(context, key, t)
                self._hover = self._pick(context, mouse)
            self._press = None
            self._header(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self._common_modal(context, event):
            if self._common_result == {'CANCELLED'}:
                self._cancel_all(context)
            return self._common_result
        return {'RUNNING_MODAL'}


def _cross(p, size):
    p = np.asarray(p, float)
    s = max(size, 1e-6)
    return np.array([p - (s, 0, 0), p + (s, 0, 0), p, p - (0, s, 0), p + (0, s, 0), p,
                     p - (0, 0, s), p + (0, 0, s)])


def _seg_dist(scr, m):
    """Screen distance from m to each segment of the polyline scr."""
    a, b = scr[:-1], scr[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", m - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12),
                0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * ab - m, axis=1)


def _seg_frac(a, b, m):
    ab = b - a
    return float(np.clip(np.dot(m - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))


# ---------------------------------------------------------------- edit mode
def selected_points(context):
    """{(object name, spline index): [anchor indices]} of the selected anchors."""
    out = {}
    for ob in context.objects_in_mode:
        if ob.type != 'CURVE':
            continue
        for i in bezier_splines(ob):
            sel = [j for j, b in enumerate(ob.data.splines[i].bezier_points)
                   if b.select_control_point]
            if sel:
                out[(ob.name, i)] = sel
    return out


class SS_OT_slide_curve_points(bpy.types.Operator):
    bl_idname = "spline_surface.slide_curve_points"
    bl_label = "Slide Points"
    bl_description = ("Slide the selected curve points along the curve, keeping its shape "
                      "(like Vertex Slide on a mesh). Move the mouse, click or Enter to keep, "
                      "right-click or Esc to cancel")
    bl_options = {'REGISTER', 'UNDO'}

    distance: bpy.props.FloatProperty(name="Distance", default=0.0, unit='LENGTH',
                                      description="How far along the curve (signed)")

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_CURVE'

    def _apply(self, context):
        worst = 0.0
        for (name, idx), sel in self._sel.items():
            ob = bpy.data.objects[name]
            A0, L0, R0, cyc = self._chains[(name, idx)]
            targets = []
            for i in sel:
                if not cyc and ((i == 0 and self.distance < 0)
                                or (i == len(A0) - 1 and self.distance > 0)):
                    targets.append(None)            # an end only slides inwards
                    continue
                t = arc_param(A0, L0, R0, float(i), self.distance, cyc)
                targets.append(point_at(A0, L0, R0, [t], cyc)[0])
            A, L, R = A0, L0, R0
            for i, p in zip(sel, targets):
                if p is None:
                    continue
                t, _ = nearest(A, L, R, p, cyc)
                A, L, R, err = slide_point(A, L, R, i, t, cyc, samples=24)
                worst = max(worst, err)
            write_spline(ob, idx, A, L, R)
        return worst

    def execute(self, context):
        if not hasattr(self, "_sel"):
            self._sel = selected_points(context)
            self._chains = {k: read_spline(bpy.data.objects[k[0]], k[1]) for k in self._sel}
        if not self._sel:
            self.report({'ERROR'}, "Select curve points to slide")
            return {'CANCELLED'}
        self._apply(context)
        return {'FINISHED'}

    def invoke(self, context, event):
        self._sel = selected_points(context)
        if not self._sel:
            self.report({'ERROR'}, "Select curve points to slide")
            return {'CANCELLED'}
        self._chains = {k: read_spline(bpy.data.objects[k[0]], k[1]) for k in self._sel}
        self._snaps = {n: snapshot(bpy.data.objects[n]) for n in {k[0] for k in self._sel}}
        # The screen direction of the first point's tangent and the world size of a pixel there.
        (name, idx), sel = next(iter(self._sel.items()))
        A, L, R, cyc = self._chains[(name, idx)]
        p = A[sel[0]]
        tan = point_at(A, L, R, [min(float(sel[0]), segments(len(A), cyc) - 1e-3)], cyc, 1)[0]
        region, rv3d = context.region, context.region_data
        s0 = location_3d_to_region_2d(region, rv3d, Vector(p))
        s1 = location_3d_to_region_2d(region, rv3d, Vector(p + 1e-3 * tan / max(np.linalg.norm(tan), 1e-12)))
        if s0 is None or s1 is None or (s1 - s0).length < 1e-9:
            self.report({'ERROR'}, "The point is not in view")
            return {'CANCELLED'}
        self._dir = np.array(s1 - s0) / (s1 - s0).length
        self._unit = 1e-3 / (s1 - s0).length          # world length of one pixel along the tangent
        self._start = np.array((event.mouse_region_x, event.mouse_region_y), float)
        self.distance = 0.0
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            m = np.array((event.mouse_region_x, event.mouse_region_y), float)
            self.distance = float(np.dot(m - self._start, self._dir)) * self._unit
            if event.shift:
                self.distance *= 0.1
            err = self._apply(context)
            context.area.header_text_set("Slide Points: %.4g along the curve (shape off by %.3g)   "
                                         "Click/Enter: keep   Esc/RMB: cancel" % (self.distance, err))
            return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            context.area.header_text_set(None)
            return {'FINISHED'}
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            for n, snap in self._snaps.items():
                restore(bpy.data.objects[n], snap)
            context.area.header_text_set(None)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


class SS_OT_dissolve_curve_points(bpy.types.Operator):
    bl_idname = "spline_surface.dissolve_curve_points"
    bl_label = "Dissolve Points"
    bl_description = ("Remove the selected curve points and refit the curve to keep its shape; "
                      "the neighbouring points keep their tangents (stays G1)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_CURVE'

    def execute(self, context):
        sel = selected_points(context)
        if not sel:
            self.report({'ERROR'}, "Select curve points to dissolve")
            return {'CANCELLED'}
        worst, count, skipped = 0.0, 0, 0
        for (name, idx), points in sel.items():
            ob = bpy.data.objects[name]
            for i in sorted(points, reverse=True):
                A, L, R, cyc = read_spline(ob, idx)
                if len(A) <= (3 if cyc else 2):
                    skipped += 1
                    continue
                A, L, R, err = dissolve_point(A, L, R, i, cyc)
                write_spline(ob, idx, A, L, R, removed=i)
                worst, count = max(worst, err), count + 1
        if not count:
            self.report({'WARNING'}, "Nothing dissolved: a curve needs at least two points")
            return {'CANCELLED'}
        self.report({'INFO'}, "Dissolved %d point(s), shape off by %.3g%s"
                    % (count, worst, ", %d kept (too few left)" % skipped if skipped else ""))
        return {'FINISHED'}


def _context_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.operator("spline_surface.slide_curve_points")
    layout.operator("spline_surface.dissolve_curve_points")
    layout.operator("spline_surface.curve_points")


def register():
    bpy.types.VIEW3D_MT_edit_curve_context_menu.append(_context_menu)


def unregister():
    bpy.types.VIEW3D_MT_edit_curve_context_menu.remove(_context_menu)


classes = (SS_OT_curve_points, SS_OT_slide_curve_points, SS_OT_dissolve_curve_points)
