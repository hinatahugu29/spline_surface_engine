"""Reading/writing a patch's control net and recipe (the spline_surface custom
property), and the open-edge helpers built on top of world_net().

Pure IO/geometry primitives with no dependency on the compute_* recipe
dispatch (that stays in operators.py, since it in turn depends on every
core.* builder); every other module here (live, linked, splitting, symmetry,
cage, seams) only needs this layer, not operators.py's much larger surface.
"""
import hashlib
import json

import bpy
import numpy as np

from .bridge.write_surface import build_surface_data, check_patch, read_surface_net
from .core.nurbs import eval_points

META_KEY = "spline_surface"
# Patches the add-on itself just rewrote: Linked Edit must not treat them as user edits.
ADDON_WRITES = set()

# Recipe schema version: bump when a field's meaning changes incompatibly
# (merely adding an optional field, read with meta.get(..., default), doesn't
# need a bump). compute_from_meta() refuses to guess at a newer schema than
# this add-on understands, rather than failing deep inside with a stray
# KeyError.
META_VERSION = 1

# Decimal places for rounding control-point coordinates before hashing/fingerprinting,
# shared by fingerprint() here and live.py's change-detection signature so hand-edit
# and staleness detection can't quietly drift out of sync with each other.
ROUND_DECIMALS = 6

EDGE_KEYS = ('V1', 'V0', 'U1', 'U0')
# The only same-patch edge pairing that isn't a degenerate, corner-sharing
# pair: a patch's two opposite edges (e.g. to close a tube into a loop).
OPPOSITE_EDGE = {'U0': 'U1', 'U1': 'U0', 'V0': 'V1', 'V1': 'V0'}


def world_net(ob):
    """(world-space control net, (order_u, order_v)) of a NURBS surface patch."""
    orders = check_patch(ob)
    net = read_surface_net(ob.data)
    M = np.array(ob.matrix_world)
    return net @ M[:3, :3].T + M[:3, 3], orders


def fingerprint(net, orders):
    """Short hash of a patch, used to notice hand edits since we last wrote it."""
    data = np.round(np.asarray(net, dtype=float), ROUND_DECIMALS).tobytes() + bytes(orders)
    return hashlib.md5(data).hexdigest()[:16]


def hand_edited_net(meta, net, orders):
    return "fp" in meta and fingerprint(net, orders) != meta["fp"]


def hand_edited(ob, meta):
    """True if the patch changed (edit mode, transform...) since the add-on wrote it."""
    try:
        return hand_edited_net(meta, *world_net(ob))
    except ValueError:
        return "fp" in meta


def store_meta(ob, meta):
    """Save the recipe with a fingerprint of the patch as Blender stores it."""
    meta.setdefault("ver", META_VERSION)
    meta["fp"] = fingerprint(*world_net(ob))
    ob[META_KEY] = json.dumps(meta)


def raw_meta(net, orders, **extra):
    """A "raw" recipe: no sources, just the control net as stored. Used to
    freeze a patch's current shape in place of whatever built it (e.g. a
    dependent whose source edge vanished, or a hand-edited patch that's
    about to gain an unrelated recipe field like a match or symmetry)."""
    return {"method": "raw", "sources": [], "net": np.asarray(net).tolist(),
           "orders": list(orders), **extra}


def replace_net(ob, net, settings, orders=None):
    """Swap the object's surface for one built from a world-space net."""
    # the object keeps its transform (and so its origin); the net is written in its local space
    M = np.array(ob.matrix_world)
    if not np.allclose(M, np.eye(4)):
        Mi = np.linalg.inv(M)
        net = np.asarray(net) @ Mi[:3, :3].T + Mi[:3, 3]
    cu = build_surface_data(ob.data.name, net, settings.resolution, orders)
    old = ob.data
    for mat in old.materials:
        cu.materials.append(mat)
    ob.data = cu
    ADDON_WRITES.add(ob.name)
    if old.users == 0:
        bpy.data.curves.remove(old)


def set_origin(ob, point):
    """Move the object's origin to the world position `point`; the surface does not move."""
    from mathutils import Matrix, Vector
    M = ob.matrix_world.copy()
    c = M.inverted() @ Vector(point)
    for sp in ob.data.splines:
        for p in sp.points:
            p.co = (p.co.x - c.x, p.co.y - c.y, p.co.z - c.z, p.co.w)
    ob.data.update_tag()
    ob.matrix_world = M @ Matrix.Translation(c)
    ADDON_WRITES.add(ob.name)


def _make_active(context, ob):
    for o in context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob


def edge_polyline(net, orders, edge, samples=48):
    """World-space points along one boundary edge of a patch."""
    t = np.linspace(0.0, 1.0, samples)
    c = np.full(samples, 1.0 if edge[1] == '1' else 0.0)
    return eval_points(net, orders, c, t) if edge[0] == 'U' else eval_points(net, orders, t, c)


def iso_lines(net, orders, count=6, samples=32):
    """A few u and v iso-curves of a patch, for a wireframe preview."""
    t = np.linspace(0.0, 1.0, samples)
    lines = []
    for c in np.linspace(0.0, 1.0, count):
        cc = np.full(samples, c)
        lines.append(eval_points(net, orders, cc, t))
        lines.append(eval_points(net, orders, t, cc))
    return lines


def surface_edges(context, samples=24, free_only=True):
    """{(object name, edge): world polyline} over visible patches, skipping seams."""
    lines = {}
    for ob in context.visible_objects:
        if ob.type != 'SURFACE':
            continue
        try:
            net, orders = world_net(ob)
        except ValueError:
            continue
        for e in EDGE_KEYS:
            lines[(ob.name, e)] = edge_polyline(net, orders, e, samples)
    if not free_only:
        return lines
    size = max((np.ptp(pl, axis=0).max() for pl in lines.values()), default=1.0)
    tol = 1e-3 * size + 1e-6
    shared = set()
    keys = list(lines)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0]:
                continue
            d = np.linalg.norm(lines[a][:, None] - lines[b][None], axis=2)
            if d.min(axis=1).max() < tol and d.min(axis=0).max() < tol:
                shared.update((a, b))
    return {k: v for k, v in lines.items() if k not in shared}
