"""Retake the listing screenshots in docs/images from the bundled samples.

Opens a Blender window (overlays and the sidebar only draw with a UI), sets up
each view on a timer, saves a JPEG and quits. One run per sample:

    blender --factory-startup --window-geometry 0 0 1600 1100 samples/hood_fender.blend \
        --python tools/take_screenshots.py -- [OUTDIR]
    blender --factory-startup --window-geometry 0 0 1600 1100 samples/body_front_cabin.blend \
        --python tools/take_screenshots.py -- [OUTDIR]

OUTDIR defaults to docs/images. Which shots are taken depends on the file:
hood_fender gives hero_zebra, seam_colors, follow_arrows, to_mesh and
ui_overview; body_front_cabin gives body_zebra and diagnose.
"""
import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import spline_surface  # noqa: E402
from spline_surface import ui  # noqa: E402
from spline_surface.recipe import world_net  # noqa: E402

extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.abspath(extra[0] if extra else os.path.join(ROOT, "docs", "images"))
os.makedirs(OUT, exist_ok=True)

# Only the Inspect panel open in the sidebar, so the screenshots show it whole.
for cls in (ui.SS_PT_create, ui.SS_PT_create_settings, ui.SS_PT_grow, ui.SS_PT_group,
            ui.SS_PT_shape, ui.SS_PT_output, ui.SS_PT_help):
    cls.bl_options = {'DEFAULT_CLOSED'}
spline_surface.register()

SAMPLE = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
STEPS = []
FAILED = []


def step(wait=1.0):
    def deco(fn):
        STEPS.append((fn, wait))
        return fn
    return deco


def run():
    if not STEPS:
        print("SCREENSHOTS", "FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
        bpy.ops.wm.quit_blender()
        return None
    fn, wait = STEPS.pop(0)
    try:
        fn()
    except Exception:
        traceback.print_exc()
        FAILED.append(fn.__name__)
    return wait


def view3d():
    win = bpy.context.window_manager.windows[0]
    area = max((a for a in win.screen.areas if a.type == 'VIEW_3D'),
               key=lambda a: a.width * a.height)
    return win, area, area.spaces.active


def override():
    win, area, _ = view3d()
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return bpy.context.temp_override(window=win, area=area, region=region)


def surfaces():
    return [o for o in bpy.data.objects if o.type == 'SURFACE' and o.visible_get()]


def look(direction, distance=1.5, centre_shift=(0.0, 0.0, 0.0), objects=None):
    """Point the viewport at the visible patches from `direction`."""
    _, _, space = view3d()
    obs = objects or surfaces()
    pts = np.concatenate([world_net(o)[0].reshape(-1, 3) for o in obs])
    centre = 0.5 * (pts.min(0) + pts.max(0)) + np.asarray(centre_shift)
    size = np.ptp(pts, axis=0).max()
    r3d = space.region_3d
    r3d.view_perspective = 'PERSP'
    r3d.view_location = Vector(centre)
    d = Vector(direction).normalized()
    r3d.view_rotation = (-d).to_track_quat('-Z', 'Y')
    r3d.view_distance = size * distance


def sidebar(show):
    view3d()[2].show_region_ui = show


def widen(px=150):
    """Drag the sidebar's edge to the left so the lists aren't cut short.
    One event per timer tick, so each is handled before the next arrives."""
    win, area, _ = view3d()
    r = next(r for r in area.regions if r.type == 'UI')
    x, y = r.x + 1, r.y + r.height // 2
    events = [('MOUSEMOVE', 'NOTHING', x), ('LEFTMOUSE', 'PRESS', x)]
    events += [('MOUSEMOVE', 'NOTHING', x - px * k // 6) for k in range(1, 7)]
    events += [('LEFTMOUSE', 'RELEASE', x - px), ('MOUSEMOVE', 'NOTHING', x // 3)]

    def pump():
        if not events:
            return None
        kind, value, at = events.pop(0)
        win.event_simulate(type=kind, value=value, x=at, y=y)
        return 0.05
    bpy.app.timers.register(pump, first_interval=0.05)


def tab():
    """Switch the sidebar to our tab (only once it has been drawn: before that
    the region has no tabs to choose from)."""
    _, area, _ = view3d()
    region = next(r for r in area.regions if r.type == 'UI')
    region.active_panel_category = "Spline Surface"


def select(*obs):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in obs:
        o.select_set(True)
    if obs:
        bpy.context.view_layer.objects.active = obs[-1]


def shoot(name, whole=False):
    png = os.path.join(OUT, name + ".png")
    with override():
        if whole:
            bpy.ops.screen.screenshot(filepath=png)
        else:
            bpy.ops.screen.screenshot_area(filepath=png)
    img = bpy.data.images.load(png)
    jpg = os.path.join(OUT, name + ".jpg")
    # Image.save keeps the loaded file's format: write through the scene's output
    # settings instead, with a view transform that leaves the colours alone
    scene = bpy.context.scene
    view = scene.view_settings
    kept = (view.view_transform, view.look, view.exposure, view.gamma)
    view.view_transform, view.look, view.exposure, view.gamma = 'Standard', 'None', 0.0, 1.0
    out = scene.render.image_settings
    out.file_format, out.color_mode, out.quality = 'JPEG', 'RGB', 88
    img.save_render(jpg, scene=scene)
    view.view_transform, view.look, view.exposure, view.gamma = kept
    bpy.data.images.remove(img)
    os.remove(png)
    print("saved", jpg)


def setup_view():
    _, _, space = view3d()
    space.overlay.show_cursor = False
    space.overlay.show_text = True
    space.shading.type = 'SOLID'
    sidebar(False)


# ---------------------------------------------------------------- hood_fender

if SAMPLE == "hood_fender":
    HOOD = (-1.0, -1.1, 0.95)

    @step(4.0)
    def hero_setup():
        setup_view()
        select()
        look(HOOD, 1.35)
        with override():
            bpy.ops.spline_surface.zebra_toggle()

    @step(1.0)
    def hero_zebra():
        shoot("hero_zebra")

    @step(1.5)
    def seams_setup():
        with override():
            bpy.ops.spline_surface.zebra_toggle()
            bpy.ops.spline_surface.seam_colors()
        view3d()[2].shading.type = 'SOLID'

    @step(1.0)
    def seam_colors():
        shoot("seam_colors")

    @step(1.5)
    def follow_setup():
        with override():
            bpy.ops.spline_surface.seam_colors()
        select(bpy.data.objects["Hood"])
        with override():
            bpy.ops.spline_surface.follow_map()

    @step(1.0)
    def follow_arrows():
        shoot("follow_arrows")

    @step(2.0)
    def ui_setup():
        with override():
            bpy.ops.spline_surface.follow_map()
            bpy.ops.spline_surface.seam_colors()
        select(bpy.data.objects["Hood"])
        sidebar(True)
        look(HOOD, 1.6, centre_shift=(0.0, 0.0, 0.0))

    @step(1.5)
    def ui_tab():
        tab()
        widen()

    @step(1.0)
    def ui_overview():
        shoot("ui_overview", whole=True)

    @step(2.0)
    def mesh_setup():
        with override():
            bpy.ops.spline_surface.seam_colors()
        sidebar(False)
        pats = surfaces()
        select(*pats)
        with override():
            bpy.ops.spline_surface.to_mesh()
        for o in pats:
            o.hide_set(True)
        select()
        space = view3d()[2]
        space.overlay.show_wireframes = True
        space.overlay.wireframe_threshold = 1.0
        look(HOOD, 1.35, objects=pats)

    @step(1.0)
    def to_mesh():
        shoot("to_mesh")


# ---------------------------------------------------------------- body_front_cabin

elif SAMPLE == "body_front_cabin":
    BODY = (1.0, 1.1, 0.6)

    @step(5.0)
    def body_setup():
        setup_view()
        select()
        look(BODY, 1.45)
        with override():
            bpy.ops.spline_surface.zebra_toggle()

    @step(1.0)
    def body_zebra():
        shoot("body_zebra")

    @step(2.0)
    def diagnose_setup():
        with override():
            bpy.ops.spline_surface.zebra_toggle()
        view3d()[2].shading.type = 'SOLID'
        select()
        with override():
            bpy.ops.spline_surface.diagnose()
            bpy.ops.spline_surface.seam_colors()
        sidebar(True)
        look(BODY, 1.25, centre_shift=(0.0, 0.25, 0.0))

    @step(1.5)
    def diagnose_tab():
        tab()
        widen()

    @step(1.0)
    def diagnose():
        shoot("diagnose", whole=True)

else:
    print("no screenshots for", SAMPLE or "an unsaved file")

bpy.app.timers.register(run, first_interval=3.0, persistent=True)
