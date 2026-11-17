# 更新履歴

バージョンごとの変更です（新しい順）。細かな内容はコミットログと ROADMAP.md を見てください。

| バージョン | 日付 | 内容 |
|---|---|---|
| 0.81.1 | 2026-11-17 | Check Continuity and Diagnose read a partial seam's normals right on the edge, at the exact matching point on both sides (they stepped a hair inside, which on a tight fillet read up to 0.5 deg on a perfect seam). The sample's fill against the fillet is 0.069 deg, not 0.56. A partial seam shows G1 at best (its curvature is not measured). No shapes change |
| 0.81.0 | 2026-11-17 | Fill Hole: the pieces of a fill split around a centre (5 or more sides, or a triangle) now join each other G1 right up to the centre (the sample: 0.07-0.31 deg -> 0.03-0.09 deg). Each quad's cross tangent along a spoke may differ from its neighbour's by a multiple of the spoke's own tangent, so both can meet their own corner at the centre instead of averaging. Existing fills of that kind change shape near their centre when rebuilt (same number of pieces) |
| 0.80.0 | 2026-11-17 | Diagnose: mark a finding as intended (kept on the patch, hidden from then on, cleared with a button). Check Continuity and Diagnose measure seams along part of an edge (a split fill's piece on its neighbour), and no longer take a piece for the same edge as a longer one. Fill Hole's joined side is refitted finer: the sample's fill meets the fender within 0.046 deg (was 0.127). The body sample no longer has a rear corner fill lying on the door |
| 0.79.0 | 2026-11-17 | Split Patch keeps the patch's Match Edge and Even Gap links: each goes to the piece holding that edge (an Even Gap on a cut edge to every piece facing the partner), a Match on a cut edge is named in a warning, and the piece left without links follows its sibling across the cut so no crack opens. To Mesh warns where three or more patches share an edge (not manifold) and puts those vertices in a 'non_manifold' vertex group. docs/CHECKLIST_hands_on.md lists what to check by hand |
| 0.78.0 | 2026-11-17 | Fill Hole no longer folds where two picked edges run straight on (a fillet's end edge into its wall): such edges count as one side, so a "four-sided" hole that is really a triangle is filled by three pieces around its real corners, the centre taken from a one-patch fill of the hole. Only for fills made from now on (older recipes keep their shape and part count). To Mesh welds a curved seam to a coarser neighbour edge: the stitching allowance grows with how much the boundary bends there (straight runs and panel gaps keep 5%). Diagnose skips a split fill's piece that covers only part of a neighbour edge. Both samples rebuilt |
| 0.77.0 | 2026-11-17 | From building a larger sample (front of a body with its cabin, samples/body_front_cabin.blend; findings in docs/VALIDATION_body.md): Even Gap works on joints that only partly face each other (a door edge longer than the fender edge) and warns when it opens a seam with a neighbour; Split Patch on a filleted panel freezes the fillet's other panel instead of leaving it unbuildable; Diagnose reports an open seam only for a near miss. New test tests/test_body.py |
| 0.76.0 | 2026-11-17 | Fill Hole: near the corners, where two sides want the same control points, the tangent rows now win over the curvature rows. On the sample the Fender to Fill seams go from G0 (0.55-0.58 deg) to G1 (0.07-0.09 deg); nothing else changes. The sample is rebuilt; its test checks these seams |
| 0.75.0 | 2026-11-17 | Diagnose (Inspect): lists quality problems worst first with a suggestion each: seams weaker than the recipe asks for, kinks nothing asked for, and wavy patches (curvature changing sign or bumping along the iso-lines). Fillet members and Mirror Group copies are judged by their recipes. On the sample it finds the known Fill Hole corner approximation |
| 0.74.0 | 2026-11-17 | Presets: Trunk (lid and rear panel), Roof + Side (half roof and the side of the greenhouse), Bumper (a fascia whose ends sweep back); the joined ones are rounded with Fillet like Hood + Fender. Sweep: R turns the profile with the rail as it bends (without twisting); the recipe keeps it |
| 0.73.0 | 2026-11-17 | Even Gap (Panel Gap): select the other panel, make the panel to move active and press Even Gap: its edge is moved so the gap is the Target all along and the two panels sit level (Flush). The active panel keeps the gap when the other one moves (stored in its recipe like Match Edge; X stops it). Inside: operators.py and live.py split by tool (creating, matching, extending, holefill, rebuild with a recipe registry, graph, links); tools/run_tests.py runs every suite, CI on GitHub, and a shape snapshot of the sample guards refactors |
| 0.72.0 | 2026-11-16 | Loft picks in the viewport: click the section curves in the order to pass through (Backspace drops the last, Enter lofts; curves selected beforehand start out picked), with a preview. E in Bridge Edges, Fill Hole and Sweep's target switches the candidates between edges + curves, edges only and curves only, so a curve lying on an edge (Duplicate Edge as Curve) can be told apart; after a curve, Bridge's second pick is edges only. GUI tests for both |
| 0.71.0 | 2026-11-16 | Duplicate Edge as Curve (Grow & Connect): copy a patch edge or iso-line out as a Bezier curve, Shift+click to pick several. The curve is the patch line itself (exact up to degree 3, one point per span), so Coons from duplicated edges is exact and immediate. Tests for it in tests/test_core.py, tests/test_blender.py and tests/test_gui_modal.py |
| 0.70.1 | 2026-11-16 | Use This Scene's Copies: after Scene > Full Copy (or a shot manager), copied patches still read the original scene's curves; the Group panel offers to point them at this scene's copies. New tests: click tools driven by simulated events (tests/test_gui_modal.py), the sample end to end with a To Mesh check (tests/test_sample.py). Listing texts and images updated |
| 0.70.0 | 2026-11-16 | Release candidate. Follow Arrows gets a Selected scope (the default): only what the selected patches follow and what follows them. The sample is rebuilt with the current version (the stray default cube is gone). Screenshots in docs/images and the README. ROADMAP rewritten around the release |
| 0.69.1 | 2026-11-16 | Fix: undoing an edit that changed a source's point count (Add Row, etc.) could leave a dependent patch empty or crash Blender. Such rebuilds now run right after the edit and store their own undo step (one extra Ctrl+Z). Fix a gizmo error on startup. New GUI test (tests/test_gui.py) with real undo/redo |
| 0.69.0 | 2026-11-16 | Live Update no longer switches to edit mode inside Blender's update handler: a patch that changes its point count copies a same-size surface, or is finished right after the drag ends. LICENSE (GPL-3.0-or-later) ships in the zip; the manifest declares its clipboard and files permissions |
| 0.68.0 | 2026-11-16 | Follow Arrows: who follows whom drawn in the viewport (Match links as chevrons on the seam, built-from links as arcs; grey dashed hand edited, red failed, magenta missing source) and listed in the Group panel with a button to go to each link |
| 0.67.0 | 2026-11-16 | Match Edge keeps following when the other patch moves far (it was lost beyond ~10% of the size: the matched edge pair is now remembered); Spread eases the seam's move into the whole patch; the panel lists Match links to change G0/G1/G2 or unlink |
| 0.66.1 | 2026-11-08 | The G0/G1/G2 buttons redraw the patch right away (it only updated after touching the redo panel) |
| 0.66.0 | 2026-11-08 | Change the continuity (G0/G1/G2) of an Extend, Bridge or Corner Fill patch afterwards from the Grow & Connect panel; patches built on it follow |
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
