"""What every undoable operator of the add-on does as it finishes.

Blender stores an operator's undo step right after it returns. Anything the
add-on changes later - from a depsgraph handler or a timer - is outside that
step: undo then lands on a state the user never saw, or (when an ID pointer
changed) crashes. So the follow-up work runs here, before the operator returns:
Live Update for what the operator changed, and stripes for new patches while
Zebra is on.
"""
import functools
import traceback

from . import live, zebra


def finish(context):
    live.settle(context)
    zebra.stripe_new(context)


def _after(context, result):
    if isinstance(result, set) and 'FINISHED' in result:
        try:
            finish(context)
        except Exception:
            traceback.print_exc()
    return result


def _wrap(meth, fn):
    # Blender checks the argument count, so each method keeps its own signature
    if meth == "execute":
        def run(self, context):
            return _after(context, fn(self, context))
    else:
        def run(self, context, event):
            return _after(context, fn(self, context, event))
    functools.update_wrapper(run, fn)
    run._finishes = True
    return run


def wrap_operators(classes):
    """Make the add-on's undoable operators call finish() before they return."""
    for cls in classes:
        if (not getattr(cls, "bl_idname", "").startswith("spline_surface.")
                or 'UNDO' not in getattr(cls, "bl_options", ())):
            continue
        for meth in ("execute", "invoke", "modal"):
            fn = cls.__dict__.get(meth)
            if fn is not None and not getattr(fn, "_finishes", False):
                setattr(cls, meth, _wrap(meth, fn))
