"""Read Blender curve-like objects as world-space polylines."""
import numpy as np
from mathutils.geometry import interpolate_bezier


def _bezier_spline(spline, samples):
    pts = spline.bezier_points
    segs = list(zip(pts[:-1], pts[1:]))
    if spline.use_cyclic_u:
        raise ValueError("cyclic curves cannot be used as a patch boundary")
    out = []
    for a, b in segs:
        seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, samples + 1)
        out.extend(seg if not out else seg[1:])
    return [tuple(p) for p in out]


def _evaluated_polyline(obj, depsgraph):
    """Use Blender's own tessellation (NURBS / poly curves)."""
    ob_eval = obj.evaluated_get(depsgraph)
    mesh = ob_eval.to_mesh()
    try:
        if len(mesh.polygons):
            raise ValueError("%s: disable bevel/extrude to use it as a boundary" % obj.name)
        return [tuple(v.co) for v in mesh.vertices]
    finally:
        ob_eval.to_mesh_clear()


def _grease_pencil(obj):
    layer = obj.data.layers.active
    frame = layer.current_frame() if layer else None
    if frame is None or not frame.drawing.strokes:
        raise ValueError("%s: no stroke on the active layer" % obj.name)
    return [tuple(p.position) for p in frame.drawing.strokes[0].points]


def read_polyline(obj, depsgraph, samples=32):
    """Return an (k, 3) world-space polyline for the first spline / stroke."""
    if obj.type == 'CURVE':
        if not obj.data.splines:
            raise ValueError("%s: empty curve" % obj.name)
        spline = obj.data.splines[0]
        if spline.type == 'BEZIER':
            local = _bezier_spline(spline, samples)
        else:
            local = _evaluated_polyline(obj, depsgraph)
    elif obj.type == 'GREASEPENCIL':
        local = _grease_pencil(obj)
    else:
        raise ValueError("%s: unsupported object type %s" % (obj.name, obj.type))
    if len(local) < 2:
        raise ValueError("%s: too few points" % obj.name)
    pts = np.array(local, dtype=float)
    M = np.array(obj.matrix_world)
    return pts @ M[:3, :3].T + M[:3, 3]


def read_loop(obj, depsgraph, samples=16):
    """World-space (k, 3) points around a closed curve, the first point not repeated.

    A cyclic Bezier spline is closed by its last segment; any other curve is
    taken as drawn and closed by joining its ends.
    """
    if obj.type != 'CURVE' or not obj.data.splines:
        raise ValueError("%s: not a curve" % obj.name)
    spline = obj.data.splines[0]
    if spline.type == 'BEZIER':
        pts = spline.bezier_points
        pairs = list(zip(pts[:-1], pts[1:])) + ([(pts[-1], pts[0])] if spline.use_cyclic_u else [])
        local = []
        for a, b in pairs:
            seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, samples + 1)
            local.extend(seg if not local else seg[1:])
        local = [tuple(p) for p in local]
    else:
        local = _evaluated_polyline(obj, depsgraph)
    if len(local) < 3:
        raise ValueError("%s: too few points" % obj.name)
    arr = np.array(local, dtype=float)
    M = np.array(obj.matrix_world)
    arr = arr @ M[:3, :3].T + M[:3, 3]
    if np.linalg.norm(arr[0] - arr[-1]) < 1e-6 * max(np.ptp(arr, axis=0).max(), 1e-9):
        arr = arr[:-1]
    return arr
