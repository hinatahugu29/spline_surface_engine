# 更新履歴

バージョンごとの変更です（新しい順）。細かな内容はコミットログと ROADMAP.md を見てください。

| バージョン | 日付 | 内容 |
|---|---|---|
| 0.65.1 | 2026-11-07 | Fix a crash on Ctrl+Z after Live Update: the post-undo rebuild only moves control points and waits for the undo to finish |
| 0.65.0 | 2026-11-07 | Renaming a source curve or patch keeps the patches built from it following; deleted sources are reported |
| 0.64.3 | 2026-11-07 | Undo/Redo rebuild live patches from their curves, so they no longer lag one step behind |
| 0.64.2 | 2026-11-07 | Regenerate / Live Update warn when a patch's sources are in another scene |
| 0.64.1 | 2026-11-03 | Live Update no longer stays off after a Fillet fails part-way; undo/load switch it back on |
| 0.64.0 | 2026-11-01 | Edit Flange changes angle and depth afterwards; gizmos no longer vanish until reselected; a flange wall no longer shifts when its fillet is redone |
| 0.63.1 | 2026-11-01 | Sweep Target (Edge): the third click no longer skipped by a remembered target |
| 0.63.0 | 2026-11-01 | To Mesh Curve Detail: a minimum number of segments per 90 degrees of curve |
| 0.62.0 | 2026-11-01 | Fillet bridges and corners, Fill Hole and Curve Network patches face like their neighbours; operators.py split (bridging.py, tomesh.py) |
| 0.61.0 | 2026-11-01 | Sweep Straight stops at a target (edge, curve or plane) |
| 0.60.1 | 2026-11-01 | Add Patch faces up in Blender; To Mesh winding matches Blender's front |
| 0.60.0 | 2026-11-01 | Align Normals; new patches and To Mesh face like their neighbours |
| 0.59.1 | 2026-11-01 | Sweep Straight lines its end up with the rail's end |
| 0.59.0 | 2026-11-01 | Sweep Straight - the profile's other end runs straight out of its patch |
| 0.58.0 | 2026-11-01 | Sweep carries a profile edge along a rail edge |
| 0.57.1 | 2026-11-01 | Copy Bug Report moves to a collapsed Help panel at the bottom |
| 0.57.0 | 2026-11-01 | patches keep their object transform when rebuilt; Origin to Center; Add Patch origin at the 3D cursor |
| 0.56.0 | 2026-11-01 | Unlink Mirror and Remove Mirror |
| 0.55.0 | 2026-11-01 | Copy Bug Report button |
| 0.54.0 | 2026-11-01 | Flange Outside adds the rounded fold outside the edge without cutting the patch back |
| 0.53.0 | 2026-11-01 | Mirror Group scope; no mirroring back of patches built from mirrored ones |
| 0.52.1 | 2026-10-31 | Line, Arc and Curve Grid objects get their origin at the 3D cursor |
| 0.52.0 | 2026-10-31 | Add Row U/V adds one control point line, also in Edit Mode |
| 0.51.0 | 2026-10-31 | Loft Extra Rows adds control rows between sections |
| 0.50.0 | 2026-10-31 | lighter Live Update (cached dependency graph, early cutoff) |
| 0.49.0 | 2026-10-31 | continuity report over all patches, worst first, with jump to seam |
| 0.48.0 | 2026-10-31 | Zebra width and soft edges |
| 0.47.0 | 2026-10-31 | To Mesh cuts welded-seam pentagons into quads and triangles |
| 0.46.1 | 2026-10-31 | no trim - document building window openings from separate patches |
| 0.46.0 | 2026-10-31 | Trim to Mesh prototype - cut holes along closed curves |
| 0.45.0 | 2026-10-31 | Project Curve lays a curve onto a patch |
| 0.44.0 | 2026-10-31 | Add Blueprint places reference drawings at real size |
| 0.43.0 | 2026-10-31 | Presets - starting shapes for hood, roof, door and hood + fender |
| 0.42.0 | 2026-10-31 | Panel Gap measures the gap and step between facing panels |
| 0.41.0 | 2026-10-31 | Flange - fold a rounded wall off a patch edge |
| 0.40.5 | 2026-10-30 | mitre the ends of two fillets that meet at an unrounded edge |
| 0.40.4 | 2026-10-30 | curvature display recomputes only the patches that changed |
| 0.40.3 | 2026-10-30 | keep the end tangents of curve sides in Fill Hole |
| 0.39.0 | 2026-10-30 | Move Crossing and Fair Curves for curve networks |
| 0.38.0 | 2026-10-30 | Fillet can be redone (radius later, bases that move) |
| 0.37.0 | 2026-10-30 | Network Check and curvature combs on curves |
| 0.36.0 | 2026-10-30 | split out Fillet, add a zip build script, bring test_blender up to date |
| 0.35.0 | 2026-10-30 | corner patches where fillets meet |
| 0.34.0 | 2026-10-30 | Fillet several seams at once |
| 0.33.3 | 2026-10-30 | To Mesh gives every patch the same mesh density |
| 0.33.2 | 2026-10-30 | To Mesh stitches T-junctions so seams don't crack |
| 0.33.1 | 2026-10-30 | arrow gizmo to drag an Offset patch's distance |
| 0.33.0 | 2026-10-30 | arrow gizmo to drag an Extend patch's length |
| 0.32.0 | 2026-10-30 | Soft Pull from a control point (Point Pull) |
| 0.31.0 | 2026-10-30 | Soft Pull falloff shapes (F key) |
| 0.30.1 | 2026-10-30 | Soft Pull keeps going stroke after stroke |
| 0.30.0 | 2026-10-30 | Soft Pull can pull a whole iso-line (Line Pull) |
| 0.29.0 | 2026-10-30 | Curvature Comb and Curvature Map overlays |
| 0.28.6 | 2026-10-30 | Fillet trims both patches back and bridges the new edges |
| 0.28.5 | 2026-10-30 | Fillet rolls the ball between the surfaces, not beyond them |
| 0.28.4 | 2026-10-30 | Fillet picks the seam in one click instead of two edges |
| 0.28.3 | 2026-10-30 | fix Fillet's coincidence check comparing the wrong thing |
| 0.28.2 | 2026-10-30 | fix Fillet never offering already-coincident edges as candidates |
| 0.28.1 | 2026-10-30 | Check Continuity reports the matched edge labels |
| 0.28.0 | 2026-10-30 | バージョン番号を更新 |
| 0.27.0 | 2026-10-29 | Fair and Refine, new Shape panel |
| 0.26.0 | 2026-10-29 | Control Mesh - edit a patch via its control points as a mesh, pinned rows snap back |
| 0.25.1 | 2026-10-29 | Soft Pull gains X/Y/Z axis constraints and a view-plane key |
| 0.25.0 | 2026-10-29 | Soft Pull modal that moves free control points and keeps seam rows pinned |
| 0.24.0 | 2026-10-29 | Control Points overlay showing seam-pinned vs free control points |
| 0.21.0 | 2026-10-29 | Linked Edit, Edge Chain and Merge Patches |
| 0.20.1 | 2026-10-29 | skip one-row surfaces instead of crashing the edge pickers |
| 0.20.0 | 2026-10-29 | fix Fill Hole on smooth outlines (tube end caps) |
| 0.19.0 | 2026-10-29 | global zebra, top-view curve networks, curve grid starter |
