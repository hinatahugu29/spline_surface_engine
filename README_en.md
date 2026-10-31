# Spline Surface

A Blender 5.x add-on for building smooth, car-body-like surfaces without NURBS expertise.
Surfaces are Blender NURBS surface patches; when the shape is right, weld them into one mesh.

The full documentation is in Japanese (see `docs/MANUAL.md`). The panel and button names are in English.

## Install
1. Take `spline_surface-x.y.z.zip` (from `dist/` or the release).
2. In Blender: Edit > Preferences > Add-ons > menu at the top right > Install from Disk, and pick the ZIP (do not unpack it).
3. In the 3D Viewport press N: a **Spline Surface** tab appears.

Requires Blender 5.0 or later (tested with 5.2 on Windows).

## What it does
- **Join surfaces smoothly**: match neighbouring patches at G0 (position), G1 (tangent) or G2 (curvature). Extend an edge, bridge two edges, fill a corner or a hole, round a corner (Fillet), fold a wall off an edge (Flange).
- **Build from curves**: Coons patch from 3 to 4 boundary curves, Loft through section curves, Curve Network from a grid of crossing curves (G2 inside).
- **Body panel presets and blueprints**: start from a Hood, Roof, Door or Hood + Fender; place side, front and top drawings at real size.
- **It remembers how a patch was made**: move a source patch or curve and everything built from it is rebuilt (Live Update). Mirror Group keeps the other half in step.
- **Check the result**: seam colours (G2 green, G1 yellow, G0 red), zebra stripes, curvature comb and map, a continuity list that jumps to the worst seam, panel gap measurement.
- **Shape it**: Soft Pull, Fair, Refine, Add Row, Cage (lattice).
- **Output**: To Mesh welds the patches into one gap-free mesh, mostly quads.

## Not covered
- Trimming a hole into a patch (build openings from separate patches instead; see the manual).
- NURBS export (IGES / STEP). The output is Blender surfaces and meshes.

## Documents
- `docs/MANUAL.md` manual (Japanese), `docs/FEATURES.md` feature list, `docs/WORKFLOWS.md` six step-by-step workflows, `docs/CHANGELOG.md` history.

## Reporting a bug
Select the patches involved and press **Copy Bug Report** at the top of the sidebar, then paste the text into a new issue at
https://github.com/hinatahugu29/spline_surface_engine/issues
together with what you did and what you expected. A small .blend that reproduces the problem helps a lot.
