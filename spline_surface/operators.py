"""The core modelling operators, gathered for registration.

They live in creating.py (Add Patch, Coons, Loft, Curve Network...),
matching.py (Regenerate, Match Edge, continuity report), extending.py
(Extend Edge, Offset Surface) and holefill.py (Fill Hole). The recipe
rebuild is in rebuild.py, the recipe storage in recipe.py.

The names below are re-exported so scripts written against the old,
single-file operators.py keep working.
"""
from . import creating, extending, holefill, matching
from .creating import primitive_net  # noqa: F401
from .extending import _extend_far_edge, _solve_extend_length  # noqa: F401
from .recipe import (META_KEY, edge_polyline, hand_edited, store_meta,  # noqa: F401
                     surface_edges, world_net)
from .rebuild import compute_from_meta  # noqa: F401

classes = creating.classes + extending.classes + holefill.classes + matching.classes
