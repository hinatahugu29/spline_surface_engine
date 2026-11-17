"""Quality check of a set of patches, in the spirit of Class-A surfacing.

Two kinds of problems are looked for, each with a suggestion how to fix it:

- Seams: a seam weaker than its patch's recipe asks for (a G2 Match that only
  reaches G1, say: the patch was edited by hand or could not follow), or a seam
  with a kink (G0) that nothing asked for.
- Waviness inside a patch: along the iso-lines, the curvature across them
  changes sign (an inflection: the reflection lines S-bend) or rises and falls
  more often than a fair panel does. Both show as wobbly zebra stripes.

Pure geometry: the caller measures the seams (core.continuity.measure/grade)
and passes the recipes in.
"""
import numpy as np

from .extend import bezier_orders
from .nurbs import eval_points

RANK = {'open': 0, 'G0': 1, 'G1': 2, 'G2': 3}
# How bad each issue is, for sorting: higher first.
# An open seam is reported only up to this gap (relative to the size): a near miss.
NEAR_MISS = 0.02
SEVERITY = {"seam_open": 5, "seam_below": 4, "seam_kink": 3, "inflection": 2, "bumpy": 1}


def _normal_curvature(net, orders, along, count, samples):
    """Signed curvature of `count` iso-lines running in `along` ('U'/'V'): (count, samples)."""
    t = np.linspace(0.02, 0.98, samples)
    w = np.linspace(0.05, 0.95, count)
    W, T = np.meshgrid(w, t, indexing="ij")
    if along == 'U':
        u, v, d1, d2 = T.ravel(), W.ravel(), (1, 0), (2, 0)
    else:
        u, v, d1, d2 = W.ravel(), T.ravel(), (0, 1), (0, 2)
    C1 = eval_points(net, orders, u, v, *d1)
    C2 = eval_points(net, orders, u, v, *d2)
    N = np.cross(eval_points(net, orders, u, v, 1, 0), eval_points(net, orders, u, v, 0, 1))
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-24)
    speed2 = np.maximum(np.einsum("ij,ij->i", C1, C1), 1e-24)
    k = np.einsum("ij,ij->i", C2, N) / speed2
    return k.reshape(count, samples)


def _sign_changes(k, floor):
    """Sign changes of k, ignoring values within `floor` of zero (flat stretches)."""
    s = np.sign(np.where(np.abs(k) > floor, k, 0.0))
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum())


def _bumps(k, floor):
    """Local maxima of |k| that stand out by more than `floor` on both sides."""
    a = np.abs(k)
    n = 0
    i = 1
    while i < len(a) - 1:
        if a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            left = a[i] - a[:i].min()
            right = a[i] - a[i + 1:].min()
            if left > floor and right > floor:
                n += 1
        i += 1
    return n


def waviness(net, orders=None, count=7, samples=48, rel=0.08):
    """How wavy a patch is: dict(inflections, bumps, line) for its worst iso-line.

    An iso-line counts an inflection where its normal curvature changes sign
    and a bump for every extra curvature peak beyond the first (a fair panel
    has at most one). Curvature smaller than `rel` of the patch's largest (or
    of 1/size, whichever is larger) counts as flat, so a flat panel with noise
    in the last digits stays clean.
    """
    net = np.asarray(net, dtype=float)
    orders = tuple(orders or bezier_orders(net))
    size = max(float(np.ptp(net.reshape(-1, 3), axis=0).max()), 1e-12)
    worst = {"inflections": 0, "bumps": 0, "line": None}
    for along in ('U', 'V'):
        K = _normal_curvature(net, orders, along, count, samples)
        floor = rel * max(float(np.abs(K).max()), 1.0 / size)
        for i, k in enumerate(K):
            inf = _sign_changes(k, floor)
            bumps = max(0, _bumps(k, floor) - 1)
            if (inf, bumps) > (worst["inflections"], worst["bumps"]):
                worst = {"inflections": inf, "bumps": bumps, "line": (along, i)}
    return worst


def expected_seams(meta):
    """{other patch: continuity} the recipe `meta` asks for along a seam."""
    if not meta:
        return {}
    out = {}

    def want(name, c):
        if c in RANK and RANK[c] > RANK.get(out.get(name), -1):
            out[name] = c

    for m in meta.get("matches", []):
        want(m.get("ref"), m.get("continuity"))
    method = meta.get("method")
    srcs = meta.get("sources", [])
    if method in ("extend", "blend", "corner", "fill") and "continuity" in meta:
        for s in srcs:
            want(s, meta["continuity"])
    return out


def _fillet_id(meta):
    if meta and meta.get("method") == "fillet_trim":
        return meta.get("fillet", {}).get("id")
    return None


def _through_mirrors(metas):
    """Recipes with every Mirror Group copy given its original's recipe, the
    names in it turned into their mirror copies' names."""
    copy_of = {m["sources"][0]: n for n, m in metas.items()
               if m and m.get("method") == "mirror" and m.get("sources")}
    out = dict(metas)
    for name, m in metas.items():
        if not (m and m.get("method") == "mirror" and m.get("sources")):
            continue
        orig = metas.get(m["sources"][0])
        if not orig:
            continue
        mapped = dict(orig)
        mapped["matches"] = [dict(x, ref=copy_of.get(x.get("ref"), x.get("ref")))
                             for x in orig.get("matches", [])]
        mapped["sources"] = [copy_of.get(s, s) for s in orig.get("sources", [])]
        out[name] = mapped
    return out


def diagnose(patches, metas, seams):
    """Issues found among `patches`, worst first.

    patches: {name: (net, orders)}; metas: {name: recipe dict or None};
    seams: [dict(a, b, grade, angle, curvature)] as Check Continuity measures
    them. Returns [dict(kind, a, b, text, hint, severity)].
    """
    issues = []
    metas = _through_mirrors(metas)
    for r in seams:
        a, b, got = r["a"], r["b"], r["grade"]
        fa = _fillet_id(metas.get(a))
        if fa is not None and fa == _fillet_id(metas.get(b)):
            continue                     # trimmed apart by their fillet, which joins them
        asked = max((expected_seams(metas.get(a)).get(b), expected_seams(metas.get(b)).get(a)),
                    key=lambda c: RANK.get(c, -1))
        if asked is not None and RANK[got] < RANK[asked]:
            follower = a if b in expected_seams(metas.get(a)) else b
            fm = metas.get(follower) or {}
            if fm.get("method") == "fill" and fm.get("merge") and got == 'open':
                continue    # one piece of a split fill covers only part of that edge
            if fm.get("method") == "fill":
                hint = "Fill Hole is only approximate near its corners (curvature, G2)"
            else:
                hint = "Regenerate %s, or redo its Match Edge (%s)" % (follower, asked)
            issues.append({"kind": "seam_below", "a": a, "b": b,
                           "text": "%s | %s: %s, the recipe asks for %s" % (a, b, got, asked),
                           "hint": hint})
        elif got == 'open':
            if r.get("gap", 0.0) > NEAR_MISS * r.get("size", np.inf):
                continue                 # well apart: a panel between them, or on purpose
            issues.append({"kind": "seam_open", "a": a, "b": b,
                           "text": "%s | %s: the edges almost meet (gap %.3g)" % (a, b, r.get("gap", 0.0)),
                           "hint": "Match Edge G0 or higher closes it"})
        elif got == 'G0' and asked is None:
            issues.append({"kind": "seam_kink", "a": a, "b": b,
                           "text": "%s | %s: kink of %.3g deg" % (a, b, r["angle"]),
                           "hint": "Match Edge G1/G2 if it should be smooth (a crease is fine as is)"})
    for name, (net, orders) in patches.items():
        w = waviness(net, orders)
        if w["inflections"]:
            issues.append({"kind": "inflection", "a": name, "b": name,
                           "text": "%s: curvature changes sign %d time(s) along %s" % (
                               name, w["inflections"], w["line"][0]),
                           "hint": "Fair, or Soft Pull the hollow out"})
        elif w["bumps"]:
            issues.append({"kind": "bumpy", "a": name, "b": name,
                           "text": "%s: %d extra curvature bump(s) along %s" % (
                               name, w["bumps"], w["line"][0]),
                           "hint": "Fair evens it out"})
    for i in issues:
        i["severity"] = SEVERITY[i["kind"]]
    issues.sort(key=lambda i: -i["severity"])
    return issues
