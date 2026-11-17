"""The patch dependency graph read from the recipes: who reads whom, rebuild
order, connected groups, recipes pointing at another scene's objects, and
renaming an input inside a recipe. No rebuilding happens here (live.py).
"""
import json
import re

import bpy

from .recipe import META_KEY


def _meta(ob):
    try:
        return json.loads(ob[META_KEY])
    except (KeyError, ValueError, TypeError):
        return None


def _inputs(meta):
    """Names of the objects a recipe reads."""
    names = list(meta.get("sources", []))
    names += [m["ref"] for m in meta.get("matches", [])]
    if meta.get("target", {}).get("source"):
        names.append(meta["target"]["source"])
    return names


_graph_cache = {"key": None, "graph": {}}

# patch name -> why its last rebuild in a chain failed (cleared once it rebuilds again);
# the Follow Map draws these links red.
FAILED = {}


def dependency_graph():
    """{object name: set of names that depend on it} over all patches.

    Recipes are only parsed again when some recipe string changed, which keeps
    the per-edit cost of live update independent of how many patches exist.
    """
    recipes = [(ob.name, ob[META_KEY]) for ob in bpy.data.objects if META_KEY in ob]
    key = hash(tuple(recipes))
    if _graph_cache["key"] == key:
        return _graph_cache["graph"]
    down = {}
    for name, raw in recipes:
        try:
            meta = json.loads(raw)
        except (ValueError, TypeError):
            continue
        for src in _inputs(meta):
            down.setdefault(src, set()).add(name)
    _graph_cache["key"], _graph_cache["graph"] = key, down
    return down


def downstream_order(names, down=None):
    """(patches depending on `names` sources first, patches caught in a loop)."""
    down = dependency_graph() if down is None else down
    reach, stack = set(), list(names)
    while stack:
        for dep in down.get(stack.pop(), ()):
            if dep not in reach:
                reach.add(dep)
                stack.append(dep)
    # Kahn's algorithm restricted to the reached sub-graph.
    indeg = {n: 0 for n in reach}
    for src, deps in down.items():
        if src in reach:
            for d in deps:
                indeg[d] += 1
    order = []
    ready = sorted(n for n, k in indeg.items() if k == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for d in sorted(down.get(n, ())):
            if d in indeg:
                indeg[d] -= 1
                if indeg[d] == 0:
                    ready.append(d)
    # Patches on a dependency loop (e.g. a patch matched to its own extension)
    # have no consistent order; callers skip them.
    return order, sorted(reach - set(order))


def connected(names):
    """All objects linked to `names` through recipes, in either direction."""
    down = dependency_graph()
    up = {}
    for src, deps in down.items():
        for d in deps:
            up.setdefault(d, set()).add(src)
    seen, stack = set(names), list(names)
    while stack:
        n = stack.pop()
        for m in down.get(n, set()) | up.get(n, set()):
            if m not in seen and m in bpy.data.objects:
                seen.add(m)
                stack.append(m)
    return seen


def off_scene_inputs(names, scene):
    """Inputs of the patches `names` that exist but are not in `scene`.

    A recipe names its sources; if those objects live in another scene (e.g. a
    full scene copy made by another add-on), editing the look-alike in this
    scene never reaches the patch, and nothing else would say why."""
    out = set()
    for name in names:
        ob = bpy.data.objects.get(name)
        meta = _meta(ob) if ob is not None else None
        for src in _inputs(meta or {}):
            sob = bpy.data.objects.get(src)
            if sob is not None and scene.objects.get(src) is None:
                out.add(src)
    return sorted(out)


_relink_cache = {"key": None, "map": {}}


def scene_copy_relinks(scene):
    """{patch: {input: copy}} for patches in `scene` whose recipe reads an object
    of another scene that has exactly one same-named copy here.

    Scene > New > Full Copy (and add-ons built on it, e.g. shot managers)
    copies patches and curves as "<name>.001", but the copied recipes still
    name the originals, so the copies follow the other scene's curves."""
    dependency_graph()                     # refreshes _graph_cache["key"] when recipes change
    key = (_graph_cache["key"], scene.name, tuple(o.name for o in scene.objects))
    if _relink_cache["key"] == key:
        return _relink_cache["map"]
    here = {o.name: o for o in scene.objects}
    out = {}
    for ob in scene.objects:
        meta = _meta(ob) if META_KEY in ob else None
        for src in _inputs(meta or {}):
            other = bpy.data.objects.get(src)
            if src in here or other is None:
                continue
            pat = re.compile(re.escape(src) + r"\.\d{3,}$")
            cands = [n for n, o in here.items() if pat.match(n) and o.type == other.type]
            if len(cands) == 1:
                out.setdefault(ob.name, {})[src] = cands[0]
    _relink_cache["key"], _relink_cache["map"] = key, out
    return out


def relink_scene_copies(scene):
    """Point the recipes found by scene_copy_relinks() at this scene's copies.
    Returns the patches changed. Shapes are left as they are."""
    changed = []
    for name, pairs in scene_copy_relinks(scene).items():
        ob = bpy.data.objects.get(name)
        meta = _meta(ob) if ob is not None else None
        if meta is None:
            continue
        for old, new in pairs.items():
            meta, _ = _rename_in(meta, old, new)
        ob[META_KEY] = json.dumps(meta)       # the fingerprint stays: the shape did not change
        changed.append(name)
    _relink_cache["key"] = None
    return changed


# Recipe fields that hold settings, never object names.
_NOT_NAMES = {"method", "edge", "edges", "continuity", "kind", "axis", "fp", "ver",
              "layout", "net", "orders", "edge_t", "edge_r"}


def _rename_in(value, old, new, key=None):
    if key in _NOT_NAMES:
        return value, False
    if isinstance(value, str):
        return (new, True) if value == old else (value, False)
    if isinstance(value, list):
        out, hit = [], False
        for v in value:
            v, h = _rename_in(v, old, new)
            out.append(v)
            hit |= h
        return out, hit
    if isinstance(value, dict):
        out, hit = {}, False
        for k, v in value.items():
            v, h = _rename_in(v, old, new, k)
            out[k] = v
            hit |= h
        return out, hit
    return value, False
