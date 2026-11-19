"""Attach Curve Ends: put curve ends on what they should meet (core.curveends).

Each end goes to the nearest of, within reach:
  another curve's end - G0 joins it, G1 also continues its tangent, G2 its bend
  a patch            - G0 onto the surface, G1 with the tangent in its tangent
                       plane (a patch built from the curve can then meet it G1),
                       or Normal: leaving straight off the surface
  another curve      - G0 onto it (a T junction)
"""
import bpy
import numpy as np

from .core.curveends import attach_end, end_frame
from .core.curvepoint import nearest
from .core.nurbs import eval_points
from .curveediting import bezier_splines, read_spline, selected_points, target_objects, write_spline
from .recipe import world_net


def nearest_on_patch(net, orders, p, iterations=8):
    """(point, unit normal, distance) of the patch point closest to p."""
    g = np.linspace(0.0, 1.0, 33)
    U, V = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
    pts = eval_points(net, orders, U, V)
    k = int(np.argmin(np.linalg.norm(pts - p, axis=1)))
    u, v = U[k], V[k]
    for _ in range(iterations):                 # Newton on the squared distance
        s = eval_points(net, orders, [u], [v])[0]
        su = eval_points(net, orders, [u], [v], 1, 0)[0]
        sv = eval_points(net, orders, [u], [v], 0, 1)[0]
        r = s - p
        J = np.array([[su @ su, su @ sv], [su @ sv, sv @ sv]])
        g2 = np.array([r @ su, r @ sv])
        try:
            du, dv = np.linalg.solve(J + 1e-12 * np.eye(2), -g2)
        except np.linalg.LinAlgError:
            break
        u, v = float(np.clip(u + du, 0.0, 1.0)), float(np.clip(v + dv, 0.0, 1.0))
    s = eval_points(net, orders, [u], [v])[0]
    cu, cv = np.clip(u, 1e-4, 1 - 1e-4), np.clip(v, 1e-4, 1 - 1e-4)
    nrm = np.cross(eval_points(net, orders, [cu], [cv], 1, 0)[0],
                   eval_points(net, orders, [cu], [cv], 0, 1)[0])
    ln = np.linalg.norm(nrm)
    return s, (nrm / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])), float(np.linalg.norm(s - p))


def _ends_to_attach(context):
    """[(object name, spline index, 'START' / 'END')]."""
    out = []
    if context.mode == 'EDIT_CURVE':
        for (name, idx), sel in selected_points(context).items():
            sp = bpy.data.objects[name].data.splines[idx]
            if sp.use_cyclic_u:
                continue
            n = len(sp.bezier_points)
            out += [(name, idx, 'START')] * (0 in sel) + [(name, idx, 'END')] * (n - 1 in sel)
        return out
    for ob in target_objects(context):
        for idx in bezier_splines(ob):
            if not ob.data.splines[idx].use_cyclic_u:
                out += [(ob.name, idx, 'START'), (ob.name, idx, 'END')]
    return out


class SS_OT_attach_curve_ends(bpy.types.Operator):
    bl_idname = "spline_surface.attach_curve_ends"
    bl_label = "Attach Curve Ends"
    bl_description = ("Put curve ends on what they should meet: another curve's end (G0 / G1 / "
                      "G2 continuing it), a patch (onto the surface; G1: tangent in its tangent "
                      "plane, or straight off it), or another curve (onto it). Object mode: both "
                      "ends of the selected curves; edit mode: the selected end points")
    bl_options = {'REGISTER', 'UNDO'}

    continuity: bpy.props.EnumProperty(
        name="Continuity", default='G1',
        items=[('G0', "G0", "Only the position"),
               ('G1', "G1", "Position and tangent"),
               ('G2', "G2", "Position, tangent and bend (curve ends; on a patch this is G1)")])
    on_patch: bpy.props.EnumProperty(
        name="On a Patch", default='TANGENT',
        items=[('TANGENT', "Along the Surface", "The tangent lies in the patch's tangent plane"),
               ('NORMAL', "Off the Surface", "The curve leaves along the patch normal")])
    reach: bpy.props.FloatProperty(
        name="Reach", default=0.0, min=0.0, unit='LENGTH',
        description="How far an end may move (0: 5% of the curve's size)")

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_CURVE'}

    def execute(self, context):
        ends = _ends_to_attach(context)
        if not ends:
            self.report({'ERROR'}, "Select curves (object mode) or curve end points (edit mode)")
            return {'CANCELLED'}
        chains = {}
        for ob in context.visible_objects:
            if ob.type == 'CURVE':
                for idx in bezier_splines(ob):
                    chains[(ob.name, idx)] = read_spline(ob, idx)
        patches = []
        for ob in context.visible_objects:
            if ob.type == 'SURFACE':
                try:
                    patches.append(world_net(ob))
                except ValueError:
                    pass
        done = {'end': 0, 'patch': 0, 'curve': 0}
        missed = 0
        for name, idx, end in ends:
            key = (name, idx)
            A, L, R, cyc = chains[key]
            p, t, k = end_frame(A, L, R, end)
            size = max(float(np.ptp(A, axis=0).max()), 1e-6)
            reach = self.reach or 0.05 * size
            best = None                                     # (distance, kind, data)
            for okey, (B, M, N, ocyc) in chains.items():
                if okey == key:
                    continue
                if not ocyc:
                    for oend, q in (('START', B[0]), ('END', B[-1])):
                        d = float(np.linalg.norm(q - p))
                        if d <= reach and (best is None or best[1] != 'end' or d < best[0]):
                            best = (d, 'end', (okey, oend))
            if best is None:
                for net, orders in patches:
                    q, nrm, d = nearest_on_patch(net, orders, p)
                    if d <= reach and (best is None or d < best[0]):
                        best = (d, 'patch', (q, nrm))
                for okey, (B, M, N, ocyc) in chains.items():
                    if okey == key:
                        continue
                    tt, d = nearest(B, M, N, p, ocyc)
                    if d <= reach and (best is None or d < best[0] - 1e-9):
                        from .core.curvepoint import point_at
                        best = (d, 'curve', point_at(B, M, N, [tt], ocyc)[0])
            if best is None:
                missed += 1
                continue
            _, kind, data = best
            if kind == 'end':
                okey, oend = data
                q, ot, ok = end_frame(*chains[okey][:3], oend)
                A, L, R = attach_end(A, L, R, end, q,
                                     direction=-ot if self.continuity != 'G0' else None,
                                     curvature=ok if self.continuity == 'G2' else None)
            elif kind == 'patch':
                q, nrm = data
                direction = None
                if self.continuity != 'G0':
                    if self.on_patch == 'NORMAL':
                        direction = nrm if t @ nrm >= 0 else -nrm
                    else:
                        direction = t - (t @ nrm) * nrm
                        if np.linalg.norm(direction) < 1e-9:
                            direction = None
                A, L, R = attach_end(A, L, R, end, q, direction=direction)
            else:
                A, L, R = attach_end(A, L, R, end, data)
            chains[key] = (A, L, R, cyc)
            write_spline(bpy.data.objects[name], idx, A, L, R)
            done[kind] += 1
        total = sum(done.values())
        msg = "Attached %d end(s): %d to curve ends, %d to patches, %d onto curves" % (
            total, done['end'], done['patch'], done['curve'])
        if missed:
            msg += "; %d with nothing in reach" % missed
        self.report({'INFO'} if total else {'WARNING'}, msg)
        return {'FINISHED'} if total else {'CANCELLED'}


classes = (SS_OT_attach_curve_ends,)
