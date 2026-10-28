"""Offset surface: a patch at constant distance along the normals of another.

The true offset of a NURBS surface is not a NURBS surface, so it is sampled
and fitted back in least squares onto a refined copy of the source's knot
structure. Refining (nested, uniform) keeps the edges in step with
neighbours that were offset the same way.
"""
import numpy as np

from .nurbs import basis_matrix, eval_points


def _normals(net, orders, U, V):
    eps = 1e-4
    u, v = np.clip(U, eps, 1 - eps), np.clip(V, eps, 1 - eps)  # avoid collapsed corners
    du = eval_points(net, orders, u, v, 1, 0)
    dv = eval_points(net, orders, u, v, 0, 1)
    N = np.cross(du, dv)
    length = np.linalg.norm(N, axis=1, keepdims=True)
    bad = length[:, 0] < 1e-12
    N = N / np.maximum(length, 1e-12)
    if bad.any():  # fall back to the mean normal for degenerate samples
        mean = N[~bad].mean(axis=0) if (~bad).any() else np.array([0.0, 0.0, 1.0])
        N[bad] = mean / max(np.linalg.norm(mean), 1e-12)
    return N


def offset_net(net, orders, distance, refine=2):
    """Net of the surface offset by `distance` along cross(S_u, S_v).

    Returns (net, orders, max fit deviation).
    """
    P = np.asarray(net, dtype=float)
    (nu, nv), (ku, kv) = P.shape[:2], orders
    nu2 = (nu - ku + 1) * refine + ku - 1
    nv2 = (nv - kv + 1) * refine + kv - 1
    mu, mv = max(4 * nu2, 24), max(4 * nv2, 24)
    su, sv = np.linspace(0.0, 1.0, mu), np.linspace(0.0, 1.0, mv)
    U, V = (a.ravel() for a in np.meshgrid(su, sv, indexing="ij"))
    S = eval_points(P, orders, U, V)
    Q = (S + distance * _normals(P, orders, U, V)).reshape(mu, mv, 3)
    Bu, Bv = basis_matrix(nu2, ku, su), basis_matrix(nv2, kv, sv)
    Pu, Pv = np.linalg.pinv(Bu), np.linalg.pinv(Bv)
    out = np.einsum("ia,abd,jb->ijd", Pu, Q, Pv)
    fit = np.einsum("ai,ijd,bj->abd", Bu, out, Bv)
    err = float(np.linalg.norm(fit - Q, axis=2).max())
    return out, (ku, kv), err
