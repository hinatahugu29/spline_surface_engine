# コマンド一覧（UI の棚卸し）

サイドバー（N キー → **Spline Surface** タブ）とメニューに出る操作を、画面に出る順に全部並べた一覧です。実機で確かめるときのチェック表として使えます（確認の列に ✔ を付けていく）。

- **ボタン**：画面に出る名前。（ ）の中は、状況で変わる名前
- **モード**：押せるモード。オブジェクト／編集＝オブジェクトモードとカーブの編集モードの両方。どのモードでも＝モードの制限なし（表示の切り替えなど）
- **種類**：**ビュー** は押すと 3D ビューでクリック・ドラッグする操作（Esc でやめる）。**即時** は押すとすぐ実行（多くは左下の「最後の操作を調整」で値を変えられる）
- **出るとき**：いつも出ているか、ある状態のときだけ出るか（条件はソースの式のまま）
- **説明**：ボタンにマウスを乗せると出る説明（英語）と同じ文。使い方の詳しい説明は [説明書](MANUAL.md)

この一覧はソースから作っています（`py tools/list_commands.py`）。ボタンを足したのに作り直していないと、テストが失敗します。

## Create

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
|  | Add Patch | `add_patch` | オブジェクト | 即時 | いつも | Add a starter NURBS patch at the 3D cursor, ready to Extend and Match | |
| Body panel presets | Hood / Roof / Door / Hood + Fender / Trunk / Roof + Side / Bumper | `add_preset` | オブジェクト | 即時 | いつも | Add a starting shape for a body panel at the 3D cursor | |
| Body panel presets | Add Blueprint | `add_blueprint` | オブジェクト | 即時 | いつも | Place a reference drawing (side, front or top view) at its real size, ground at z = 0, centre line at x = 0, front of the car towards +y | |
| Boundary curves | Line | `add_boundary_curve` | オブジェクト | 即時 | いつも | Add a straight line or circular arc at the 3D cursor, sized for use as a patch boundary (Coons, Loft, Curve Network...) | |
| Boundary curves | Arc | `add_boundary_curve` | オブジェクト | 即時 | いつも | Add a straight line or circular arc at the 3D cursor, sized for use as a patch boundary (Coons, Loft, Curve Network...) | |
| From curves | Coons | `coons` | オブジェクト | 即時 | いつも | Fill 3 or 4 selected boundary curves with a Bezier NURBS patch | |
| From curves | Loft | `loft` | オブジェクト | ビュー | いつも | Loft a patch through 2 or more section curves. Click the curves in the order the patch should pass through them, then Enter (curves selected beforehand start out picked). Up to Degree+1 sections are hit exactly | |
| From curves | Curve Network | `curve_network` | オブジェクト | 即時 | いつも | Skin a grid of crossing curves: select curves running in two directions (each crossing every curve of the other direction). One patch per cell, joined smoothly across the inner curves | |
| From curves | （アイコンだけ）Add Curve Grid | `add_curve_grid` | オブジェクト | 即時 | いつも | Add a ready-made grid of crossing Bezier curves at the 3D cursor, selected and ready for Curve Network. Edit the curves to taste | |
| From curves | Network Check（オンの間は「Network Check: On」） | `network_check` | どのモードでも | 即時 | いつも | Show how Curve Network sees the selected (else visible) curves: directions in blue / green, crossings marked; missing crossings red lines, same-direction crossings red marks, lone curves grey, out-of-order curves orange. Updates as you edit | |
| From curves | Project Curve | `project_curve` | オブジェクト | 即時 | いつも | Lay the selected curve onto the selected patch (nearest point, along an axis or along the view) as a new curve lying on its surface | |
| From curves | Trim to Mesh | `trim_mesh` | オブジェクト | 即時 | いつも | Prototype: mesh the selected patch with holes cut out along the selected closed curves (window, lamp). The patch itself is not changed | |
| From curves | Move Crossing | `move_crossing` | オブジェクト | ビュー | いつも | Drag a crossing of a curve network: both curves through it follow, their other crossings stay. Bezier curves only. X/Y/Z: along that axis, V: view plane, Ctrl+Z: undo a stroke, Enter: done, Esc: cancel all | |
| From curves | Fair Curves | `fair_curves` | オブジェクト | 即時 | いつも | Iron the wiggles out of the selected Bezier curves, keeping their ends, end tangents and where they cross the other visible curves | |
| From curves | （アイコンだけ）Snap Ends | `snap_ends` | オブジェクト | 即時 | いつも | Move the nearest end points of the selected curves together (within Gap Tolerance) so they form a closed boundary | |
| From curves | Regenerate | `regenerate` | オブジェクト | 即時 | いつも | Rebuild the active patch from its source curves with the current settings | |

## Curves

カーブ編集（v0.83〜）。編集モードでは Slide・Dissolve・Constrained Move が加わる

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
| Points (shape kept) | Curve Pull | `curve_pull` | オブジェクト／編集 | ビュー | いつも | Grab anywhere on a curve and drag: that spot follows the mouse and the curve bends smoothly around it (a point is added there first if none is free nearby). Wheel: range, F: falloff, E: pin the ends, N: crossing curves follow, V / P / X / Y / Z: view plane / the curve's own plane / an axis. Dragging a point pulls too (Shift+drag slides it along the curve). Also adds and removes points like Curve Points. Ctrl+Z takes back one pull, Enter keeps, Esc drops all | |
| Points (shape kept) | Curve Points | `curve_points` | オブジェクト／編集 | ビュー | いつも | Edit Bezier curves in the viewport. Click on a curve: add a point (the shape does not change). Drag on a curve: pull it, the rest follows softly (Wheel: range, F: falloff, E: pin the ends, N: crossing curves follow, V / P / X / Y / Z: view plane / the curve's own plane / an axis). Drag a point: slide it along the curve (Shift+drag: pull it). Ctrl+click a point: remove it, keeping the shape. Ctrl+Z takes back one edit, Enter keeps them, Esc drops all | |
| Points (shape kept) | Slide | `slide_curve_points` | カーブの編集モード | ビュー | カーブの編集モードのとき | Slide the selected curve points along the curve, keeping its shape (like Vertex Slide on a mesh). Move the mouse, click or Enter to keep, right-click or Esc to cancel | |
| Points (shape kept) | Dissolve | `dissolve_curve_points` | カーブの編集モード | 即時 | カーブの編集モードのとき | Remove the selected curve points and refit the curve to keep its shape; the neighbouring points keep their tangents (stays G1) | |
| Points (shape kept) | Constrained Move | `constrained_move` | カーブの編集モード | ビュー | カーブの編集モードのとき | Move the selected curve points along their own tangent (T), normal towards the bend (N) or binormal (B), or change only their handle lengths (H). K keeps G2 where the curve was G2. Shift: finer | |
| Smooth | Make G2 | `make_g2` | オブジェクト／編集 | 即時 | いつも | Make curve points G2 (the same curvature on both sides) by changing only their handle lengths; directions and positions stay. Edit mode: the selected points; object mode: every point of the selected curves. Corners, inflections and points next to a straight side are left (reported) | |
| Smooth | Refit Curve | `refit_curve` | オブジェクト／編集 | 即時 | いつも | Rebuild curves with fewer (or more) points along the same shape: the fewest points within a tolerance, or a given count. Corners and ends stay. Edit mode: the splines with a selected point; object mode: the selected curves | |
| Smooth | Fair Curves | `fair_curves` | オブジェクト | 即時 | いつも | Iron the wiggles out of the selected Bezier curves, keeping their ends, end tangents and where they cross the other visible curves | |
| Connect | Attach Curve Ends | `attach_curve_ends` | オブジェクト／編集 | 即時 | いつも | Put curve ends on what they should meet: another curve's end (G0 / G1 / G2 continuing it), a patch (onto the surface; G1: tangent in its tangent plane, or straight off it), or another curve (onto it). Object mode: both ends of the selected curves; edit mode: the selected end points | |
| Make | Two-View Curve | `two_view_curve` | オブジェクト | 即時 | いつも | Make one 3D curve from two flat curves drawn in two views (side and top, or side and front): along the axis both run along, each view gives the coordinate the other cannot see. Select the two curves | |
| Make | Fillet | `curve_fillet` | オブジェクト／編集 | 即時 | いつも | Round the corner points of curves with a radius: the curve is cut that far along it on both sides and joined by an arc-like span (G1). Edit mode: the selected corner points; object mode: every corner of the selected curves | |
| Make | Offset | `offset_curve` | オブジェクト | 即時 | いつも | A parallel copy of each selected flat curve, in its own plane (a panel line, a trim's inner edge). Negative Distance offsets to the other side | |
| Make | Extend | `extend_curve` | オブジェクト／編集 | 即時 | いつも | Lengthen curve ends: Natural continues the end's own bend (perfectly smooth), Straight goes straight on along the end tangent. Edit mode: the selected end points; object mode: the chosen ends of the selected curves | |
| See | Point Continuity（オンの間は「Point Continuity: On」） | `point_continuity` | どのモードでも | 即時 | いつも | Colour every point of the shown curves by how smoothly the curve goes through it: green G2, yellow G1, red corner, grey end. Follows edits, in edit mode too. Press again to hide | |
| See | Curvature Comb（オンの間は「Curvature Comb: On」） | `curvature_comb` | どのモードでも | 即時 | いつも | Show curvature combs along patch iso-curves and along curves: the teeth point away from the centre of curvature, longer means more curved. A smooth envelope means smooth curvature; a jump at a seam or a curve's control point means G1 or worse there | |

## Grow & Connect

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
|  | Extend Edge | `extend_edge` | オブジェクト | ビュー | いつも | Grow a new patch off an open edge of any visible patch (click it), matching its tangent (G1) or tangent and curvature (G2). Then click a second edge to stretch the extension until it reaches it | |
|  | Bridge Edges | `bridge_edges` | オブジェクト | ビュー | いつも | Bridge the gap between two separate patch edges, or from a patch edge to a curve, with a smooth blend. Click the two in the viewport | |
|  | Sweep | `sweep` | オブジェクト | ビュー | いつも | Carry the shape of one patch edge (the profile) along another patch edge (the rail). Click the profile edge, then the rail edge. F in the viewport: start from the rail's other end; S: let the profile's other end run straight out of its patch instead of following the rail; R: turn the profile with the rail | |
|  | Match Edge | `match_edge` | オブジェクト | 即時 | いつも | Make the active patch meet the other selected patch along their closest edges with G0/G1/G2 continuity (the other patch is not changed) | |
|  | Corner Fill | `corner_fill` | オブジェクト | ビュー | いつも | Fill the notch between two patch edges that meet at a corner. Click the two edges in the viewport | |
|  | Fill Hole | `fill_hole` | オブジェクト | ビュー | いつも | Fill a hole bounded by patch edges (and curves). Click the edges around it, then Enter. Four sides give one patch, other counts one patch per side around a centre | |
|  | Fillet | `fillet` | オブジェクト | ビュー | いつも | Round the creases at seams where two patch edges already coincide (e.g. after Match Edge): the patches are trimmed back and bridged. Click one or more seams, Enter, then set the radius | |
|  | Flange | `flange` | オブジェクト | ビュー | いつも | Fold a wall off an open patch edge (click the edge) and round the fold. Use it for the lip around a panel: a hem, a return, a wheel-arch lip | |
|  | Edit | `flange_edit` | オブジェクト | 即時 | いつも | Change the angle or depth of the active flange wall afterwards (the rounded fold and anything built on it follow). Tweak in the panel that opens at the bottom left | |
|  | Split Patch | `split_patch` | オブジェクト | ビュー | いつも | Cut a patch in two along an iso-line. Click on the patch where to cut; the pieces join with G2 and the group's links follow | |
|  | Duplicate Edge as Curve | `duplicate_edge_curve` | オブジェクト | ビュー | いつも | Copy an iso-line of a patch out as its own curve, usable like any boundary curve (Coons, Network, Fill Hole, Project Curve). Click on the patch where to cut; Shift+click to pick several edges (across any visible patches) before finishing with Enter. The patch itself is left untouched | |
|  | Offset Surface | `offset_surface` | オブジェクト | 即時 | いつも | Create a copy of each selected patch at a constant distance along its normal (panel thickness, inner skins). The copies follow the source | |
|  | G0 / G1 / G2 | `edit_continuity` | オブジェクト | 即時 | Extend・Bridge Edges・Corner Fill で作った面がアクティブのとき | Change the continuity (G0/G1/G2) or tension of the active Extend, Bridge or Corner Fill patch without picking the edges again. Patches built on it follow | |
|  | G0 / G1 / G2 | `match_link_edit` | オブジェクト | 即時 | アクティブな面に Match Edge の追従があるとき（追従ごとに1行） | Change how the active patch follows a patch it was matched to (continuity, Spread). Patches built on it follow | |
|  | （アイコンだけ）Edit Match Link | `match_link_edit` | オブジェクト | 即時 | アクティブな面に Match Edge の追従があるとき（追従ごとに1行） | Change how the active patch follows a patch it was matched to (continuity, Spread). Patches built on it follow | |
|  | （アイコンだけ）Unlink Match | `match_link_remove` | オブジェクト | 即時 | アクティブな面に Match Edge の追従があるとき（追従ごとに1行） | Stop the active patch following a patch it was matched to. A free or hand-edited patch stays where it is; a generated patch goes back to the shape its own recipe gives | |

## Group

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
|  | Edge Chain | `edge_chain` | オブジェクト | ビュー | いつも | Click an open edge: the run of open edges joined to it becomes one curve. Edit that curve and all of those edges follow, regardless of the Linked Edit toggle | |
|  | Merge | `merge_patches` | オブジェクト | 即時 | いつも | Refit the selected patches, laid edge to edge on a rectangular grid, as one patch. The originals are hidden (or deleted) | |
|  | Select | `select_group` | オブジェクト | 即時 | いつも | Select every patch and source connected to the selection | |
|  | Regenerate | `regenerate_all` | オブジェクト | 即時 | いつも | Rebuild every patch connected to the selection, sources first. Hand-edited patches are skipped unless Force is on | |
|  | Use This Scene's Copies | `relink_scene_copies` | オブジェクト | 即時 | シーンを Full Copy して、元のシーンのカーブを読んでいる面があるとき | Patches here were copied from another scene (Scene > Full Copy, shot managers) and still follow that scene's curves and patches. Make them follow the copies in this scene instead | |
|  | Follow Arrows | `follow_map` | どのモードでも | 即時 | いつも | Show who follows whom: Match Edge links as chevrons on the seam, patches built from sources as arcs. Grey dashed: edited by hand, red: last rebuild failed, magenta: source missing | |
|  | （アイコンだけ）Go to Link | `follow_jump` | オブジェクト | 即時 | 一覧の行ごと（結果があるとき） | Select the follower (active) and what it follows, and bring them into view | |
|  | Mirror Group | `mirror_group` | オブジェクト | 即時 | いつも | Mirror the selected patch groups across a symmetry plane. Edges on the plane are squared to it so both halves join with G2; the mirrored half follows the original through Live Update | |
|  | Unlink Mirror | `mirror_unlink` | オブジェクト | 即時 | いつも | Cut the mirrored patches (selected, or made from the selected patches) loose from their originals. They keep their shape and no longer follow the original, so the two sides can differ | |
|  | Remove Mirror | `mirror_remove` | オブジェクト | 即時 | いつも | Delete the mirrored patches (selected, or made from the selected patches). Patches built from a deleted mirror keep their shape and are cut loose | |
|  | Add Cage | `cage_add` | オブジェクト | 即時 | いつも | Wrap the selected patch groups (and their curves) in a lattice cage. Edit the lattice, then Apply Cage | |
|  | Apply | `cage_apply` | オブジェクト | 即時 | いつも | Bake the cage deformation into the source patches and curves, rebuild everything generated from them and remove the cage | |
|  | （アイコンだけ）Remove Cage | `cage_cancel` | オブジェクト | 即時 | いつも | Remove the cage without changing anything | |

## Shape

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
|  | Soft Pull | `soft_pull` | オブジェクト | ビュー | いつも | Click a point on a patch and drag: the free control points around it follow. Rows pinned by seams stay put, so G0/G1/G2 are kept. Wheel: radius, N: surface normal, X/Y/Z: that axis, V: view plane, L: origin point / line (D: direction, T: taper) / control point (Wheel / Shift+Wheel: rows along U / V), F: falloff shape | |
|  | Control Mesh | `control_mesh` | オブジェクト | 即時 | いつも | Edit the active patch through its control points as a mesh (Edit Mode, G/R/S). Points pinned by seams (vertex group Pinned) snap back, so G0/G1/G2 are kept. Use Done to remove the mesh | |
|  | Done | `control_mesh_done` | どのモードでも | 即時 | いつも | Remove the control mesh and go back to the patch | |
|  | Fair | `fair` | オブジェクト | 即時 | いつも | Smooth the interior of the selected patches. Points pinned by seams stay, so G0/G1/G2 are kept | |
|  | Refine | `refine` | オブジェクト | 即時 | いつも | Add control points without changing the shape (every span is split in two). More points to shape with; seams keep their continuity but a neighbour must match the new point count to be re-matched | |
|  | Origin to Center | `origin_to_center` | オブジェクト | 即時 | いつも | Move the origin of the selected patches to the centre of their control points. The surface does not move; rotating and scaling then pivot around the patch, and the origin stays there when the patch is rebuilt | |
|  | Align Normals | `align_normals` | オブジェクト | 即時 | いつも | Turn patches over where needed so that patches sharing an edge all face the same way (the selected patches, or all visible ones if fewer than two are selected). Patches that others are built from are left as they are | |
|  | Add Row U / Add Row V | `insert_line` | オブジェクト | 即時 | いつも | Add one row or column of control points to the patch, in Object or Edit Mode. Exact when the direction is a single span; otherwise the net is refitted and the shape moves slightly (the change is reported) | |

## Inspect

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
|  | Zebra（オンの間は「Zebra: On」） | `zebra_toggle` | どのモードでも | 即時 | いつも | Toggle zebra stripes on all visible patches at once (and selected meshes). New patches get stripes too while it is on | |
|  | Seam Colors（オンの間は「Seam Colors: On」） | `seam_colors` | どのモードでも | 即時 | いつも | Colour every seam between visible patches by continuity: G2 green, G1 yellow, G0 red, open magenta. Updates as you edit | |
|  | Curvature Comb（オンの間は「Curvature Comb: On」） | `curvature_comb` | どのモードでも | 即時 | いつも | Show curvature combs along patch iso-curves and along curves: the teeth point away from the centre of curvature, longer means more curved. A smooth envelope means smooth curvature; a jump at a seam or a curve's control point means G1 or worse there | |
|  | Curvature Map（オンの間は「Curvature Map: On」） | `curvature_map` | どのモードでも | 即時 | いつも | Colour the patches by curvature. Gaussian: red dome, green flat or cylindrical, blue saddle. Mean: red bends towards the patch normal, blue away | |
|  | Control Points（オンの間は「Control Points: On」） | `control_points` | どのモードでも | 即時 | いつも | Show a patch's control points: orange are pinned by its seams (G0 one row, G1 two, G2 three from the edge), cyan are free to edit | |
|  | Panel Gap（オンの間は「Panel Gap: On」） | `panel_gap` | どのモードでも | 即時 | いつも | Show the clearance between facing edges of neighbouring panels: green where the gap is within tolerance of the target, orange where it is off | |
|  | Even Gap | `even_gap` | オブジェクト | 即時 | Panel Gap がオンのとき | Move the active panel's edge so the gap to the other selected panel is the Target all along (and the two sit level). The active panel keeps following the other one when it moves | |
|  | （アイコンだけ）Even Gap | `even_gap` | オブジェクト | 即時 | Panel Gap がオンのとき | Move the active panel's edge so the gap to the other selected panel is the Target all along (and the two sit level). The active panel keeps following the other one when it moves | |
|  | Check Continuity | `check_continuity` | オブジェクト | 即時 | いつも | Measure gap, normal angle and curvature jump on seams between the selected patches (all visible patches when fewer than two are selected). The worst seams come first; the arrow button next to one jumps to it | |
|  | （アイコンだけ）Go to Seam | `seam_jump` | オブジェクト | 即時 | 一覧の行ごと（結果があるとき） | Select the two patches of this seam and bring them into view | |
|  | Diagnose | `diagnose` | オブジェクト | 即時 | いつも | Look for quality problems among the selected patches (all visible ones when fewer than two are selected): seams weaker than their recipe asks for, kinks, and wavy patches whose curvature changes sign or bumps. Worst first, each with a suggestion | |
|  | （アイコンだけ）Intended | `diagnose_accept` | どのモードでも | 即時 | 一覧の行ごと（結果があるとき） | Mark this finding as intended (a crease, a tumblehome): Diagnose stops listing it. The mark is kept on the patch | |
|  | （アイコンだけ）Clear Intended Marks | `diagnose_reset` | どのモードでも | 即時 | Diagnose で「意図した形」の印を付けた所があるとき | Forget the findings marked as intended on the selected patches (all visible patches when none are selected), so Diagnose lists them again | |

## Output

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
|  | To Mesh | `to_mesh` | オブジェクト | 即時 | いつも | Create a mesh of the selected surfaces at the given resolution, welded into one object by default | |

## Help

| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |
|---|---|---|---|---|---|---|---|
|  | Copy Bug Report | `copy_report` | どのモードでも | 即時 | いつも | Copy versions, settings and the selected patches' recipes and seams to the clipboard, to paste into a bug report | |

## メニュー

| 場所 | ボタン | 操作 | モード | 種類 | 確認 |
|---|---|---|---|---|---|
| 3D ビュー → Add → Surface（オブジェクトモード） | Add Patch | `add_patch` | オブジェクト | 即時 | |
| 3D ビュー → Add → Surface（オブジェクトモード） | Add Curve Grid | `add_curve_grid` | オブジェクト | 即時 | |
| カーブの編集モード → 右クリックメニュー | Slide Points | `slide_curve_points` | カーブの編集モード | ビュー | |
| カーブの編集モード → 右クリックメニュー | Dissolve Points | `dissolve_curve_points` | カーブの編集モード | 即時 | |
| カーブの編集モード → 右クリックメニュー | Constrained Move | `constrained_move` | カーブの編集モード | ビュー | |
| カーブの編集モード → 右クリックメニュー | Make G2 | `make_g2` | オブジェクト／編集 | 即時 | |
| カーブの編集モード → 右クリックメニュー | Refit Curve | `refit_curve` | オブジェクト／編集 | 即時 | |
| カーブの編集モード → 右クリックメニュー | Attach Curve Ends | `attach_curve_ends` | オブジェクト／編集 | 即時 | |
| カーブの編集モード → 右クリックメニュー | Curve Fillet | `curve_fillet` | オブジェクト／編集 | 即時 | |
| カーブの編集モード → 右クリックメニュー | Extend Curve | `extend_curve` | オブジェクト／編集 | 即時 | |
| カーブの編集モード → 右クリックメニュー | Curve Pull | `curve_pull` | オブジェクト／編集 | ビュー | |
| カーブの編集モード → 右クリックメニュー | Curve Points | `curve_points` | オブジェクト／編集 | ビュー | |

## パネルやメニューに出ないもの

ほかの操作の中から呼ばれる、または一覧の行の中のボタンとして出るものです。

| 操作 | 名前 | 説明 |
|---|---|---|
| （なし：すべての操作がパネルかメニューに出ています） | | |

合計：78 の操作（パネル・メニューに出るもの 78）
