"""Write a Bezier control net into a Blender NURBS surface.

The Python API cannot set a surface spline's V point count directly, so a
single U row is created and extruded in edit mode until the grid has the
right size; then every point is overwritten.
"""
import bpy


def build_surface_data(name, net, resolution=12, orders=None):
    """Create SURFACE curve data from a control net of shape (nu, nv, 3).

    `orders` = (order_u, order_v); None means a single-span Bezier patch
    (order = point count). Endpoint is always on.
    """
    nu, nv = net.shape[0], net.shape[1]
    ku, kv = orders or (nu, nv)
    if max(ku, kv) > 6:
        raise ValueError("Blender NURBS order is limited to 6 (degree 5)")
    cu = bpy.data.curves.new(name, 'SURFACE')
    sp = cu.splines.new('NURBS')
    sp.points.add(nu - 1)
    for pt in sp.points:
        pt.select = True

    tmp = bpy.data.objects.new("_ss_tmp", cu)
    scene = bpy.context.scene
    scene.collection.objects.link(tmp)
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    prev_sel = [o for o in view_layer.objects if o.select_get()]
    try:
        for o in prev_sel:
            o.select_set(False)
        view_layer.objects.active = tmp
        tmp.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        for _ in range(nv - 1):
            bpy.ops.curve.extrude(mode='TRANSLATION')
        bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        bpy.data.objects.remove(tmp)
        for o in prev_sel:
            if o.name in view_layer.objects:
                o.select_set(True)
        if prev_active and prev_active.name in view_layer.objects:
            view_layer.objects.active = prev_active

    sp = cu.splines[0]
    assert (sp.point_count_u, sp.point_count_v) == (nu, nv)
    for j in range(nv):
        for i in range(nu):
            x, y, z = net[i, j]
            sp.points[j * nu + i].co = (x, y, z, 1.0)
    sp.order_u, sp.order_v = ku, kv
    sp.use_endpoint_u = sp.use_endpoint_v = True
    sp.resolution_u = sp.resolution_v = resolution
    sp.use_smooth = True
    return cu


def read_surface_net(cu):
    """Inverse of build_surface_data: control net of the first spline."""
    return read_spline_net(cu.splines[0])[0]


def read_spline_net(sp):
    """(net (nu, nv, 3), weights (nu, nv)) of a surface spline."""
    import numpy as np

    nu, nv = sp.point_count_u, sp.point_count_v
    co = np.array([p.co[:] for p in sp.points]).reshape(nv, nu, 4).transpose(1, 0, 2)
    return co[..., :3].copy(), co[..., 3].copy()


def check_patch(ob):
    """Validate a surface object for the patch tools; returns its (order_u, order_v).

    Accepts single- and multi-span NURBS with Endpoint on in both directions.
    Raises ValueError naming exactly what does not fit.
    """
    import numpy as np

    if not ob.data.splines:
        raise ValueError("%s: surface has no splines" % ob.name)
    sp = ob.data.splines[0]
    problems = []
    for axis in ("u", "v"):
        n = getattr(sp, "point_count_" + axis)
        k = getattr(sp, "order_" + axis)
        if n < 2:
            problems.append("only %d point in %s (a single curve, not a surface: extrude it "
                            "or use Add Patch)" % (n, axis.upper()))
        elif k > n:
            problems.append("order %s is %d but only %d points" % (axis.upper(), k, n))
        if getattr(sp, "use_cyclic_" + axis):
            problems.append("%s is cyclic" % axis.upper())
        if not getattr(sp, "use_endpoint_" + axis):
            problems.append("Endpoint %s is off (turn it on so the surface reaches its edges)"
                            % axis.upper())
        if getattr(sp, "use_bezier_" + axis) and n != k:
            problems.append("Bezier %s is on with %d points / order %d (turn it off)"
                            % (axis.upper(), n, k))
    if not np.allclose(read_spline_net(sp)[1], 1.0):
        problems.append("control point weights are not all 1")
    if problems:
        raise ValueError("%s: %s" % (ob.name, "; ".join(problems)))
    return (sp.order_u, sp.order_v)
