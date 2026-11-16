# Spline Surface ロードマップ

車のような滑らかな曲面を、NURBS の慣れなしで作れるようにする Blender 5.x アドオン。
最終出力はメッシュ。操作はボタンとクリック選択が基本で、ギズモは操作が固まってから載せる。

## 現在地（2026-11-16、v0.72.0）

**v0.72.0 を公開候補（Release Candidate）とする。** 第1〜第4段階の作業はすべて終わった。公開前の確認も、自動で確かめられるものはすべてテストにした。残りは公開の手続き（アカウントでの操作）と、手で触ったときの感触の確認だけ。

| 段階 | 目的 | 状況 |
|---|---|---|
| 第1段階 v0.41〜 | 実機で固める（基準モデルを通しで作る） | ✅ 完了。基準モデル＝同梱のサンプル（ボンネット＋フェンダー、`tools/build_sample_hood.py`）が最後まで詰まらずに作れる |
| 第2段階 v0.5x | 車体パネルを作る道具（縁と付き方） | ✅ 完了（Flange、Panel Gap、Presets、Blueprints） |
| 第3段階 v0.6x | 開口部と投影 | ✅ 完了（Project Curve。トリムは試作して見送り） |
| 第4段階 v0.7x | 品質と配布 | ✅ 作業は完了（下の「公開の準備」）。公開の手続きが残り |

### 公開の準備

| 項目 | 状況 |
|---|---|
| ライセンス | ✅ GPL-3.0-or-later。`LICENSE` をリポジトリ・ZIP・Booth 一式に同梱（v0.69.0） |
| Blender Extensions の要件 | ✅ マニフェストに copyright と権限（clipboard：Copy Bug Report、files：Blueprints の画像読み込み）。`blender --command extension validate` と `extension build` が通り、別の設定フォルダに入れて有効化できることを確認。外部への通信・`sys.path` の書き換え・同梱バイナリなし |
| 説明書・機能一覧・作業の流れ・更新履歴 | ✅ `docs/` |
| 不具合報告の導線 | ✅ Copy Bug Report、Issue の雛形 |
| サンプル .blend | ✅ 現行版で作り直し、開いて全体を作り直しても失敗ゼロ（v0.70.0。紛れ込んでいた既定の Cube を除いた） |
| 掲載用のスクリーンショット | ✅ `docs/images/`（Zebra、Seam Colors、Follow Arrows、To Mesh、パネル全体）。README に掲載 |
| 自動テスト | ✅ 5本。`tests/test_core.py`（計算）、`tests/test_blender.py`（Blender、バックグラウンド）、`tests/test_gui.py`（画面つき：本物の Undo／Redo、名前の変更、大きな移動、点の数が変わる作り直し）、`tests/test_gui_modal.py`（画面つき、マウスとキーのイベントを再現：Fillet・Flange・Fill Hole・Esc）、`tests/test_sample.py`（サンプルを開いて全体を作り直し、To Mesh の結果を検査）。走らせ方は各ファイルの先頭 |
| 公開前の確認（下の6項目） | ✅ 6項目とも自動テストで確認（v0.70.1） |
| シーンを複製するアドオンとの併用 | ✅ Blender の Scene > Full Copy（シーンを複製するアドオンが使う仕組み）で再現して確認。複製した面が元のシーンのカーブを読み続ける問題が見つかり、Group 欄に **Use This Scene's Copies** を追加（v0.70.1）。Shot Manager 本体での確認は、このPCに入っていないため未 |
| 手で触ったときの感触 | ⬜ **人の作業**。ドラッグ中の追従の滑らかさ、パネルの見え方の好みなど、感じ方の確認。自動テストでは結果しか確かめられない |
| 公開の手続き | ⬜ **人の作業**。Blender Extensions への登録（アカウント）、Booth／Superhive のページ更新。文面（`docs/booth_description.txt`、`docs/superhive_page.html`）は最新の機能・更新履歴・画像に合わせ済み。画像は `docs/images/` からアップロードする。マニフェストの `website` は、公開するページが決まったら入れる |

### 公開前の確認と、それを確かめるテスト

| 確認すること | テスト |
|---|---|
| 1. 元を動かすとつながった面が付いてくる。Ctrl+Z／Ctrl+Shift+Z で行き来しても崩れない | `test_gui.py`（カーブの移動 → Undo → Redo、各段階で全部の面がレシピどおり） |
| 2. Match Edge の相手を大きく動かしても付いてくる。追従の一覧で連続性を変える・解除する | `test_gui.py`（高さ +2 の移動）、`test_blender.py`（match_links_far_move） |
| 3. Follow Arrows の範囲の切り替え | `test_blender.py`（follow_map_checks：Selected／Group／All と状態の色） |
| 4. Add Row → 作った面が作り直される。Ctrl+Z 2回で Add Row の前に戻る | `test_gui.py`（Undo 1回・2回、Redo 2回で全部の面が有効） |
| 5. Fillet・Flange・Fill Hole のクリック操作（クリック、Enter、ホイール、C キー、Esc） | `test_gui_modal.py`（画面上の継ぎ目・辺の位置にイベントを送る） |
| 6. To Mesh の結果に穴や裏返りがない | `test_sample.py`（境界はボディの外周1周のみ、向きの食い違い・非多様体・つぶれた面が 0） |

## 方針

- **機能を足すより、実際に1つのものを作って詰まりを拾う**。機能の網羅は一段落した
- 計算は `core/`（Blender 非依存）に置いて単体テストを書く。Blender 側は `test_blender.py`／`test_gui.py` と実機確認で補う
- 設計の原則（下）は変えない
- **Live Update は Blender の更新ハンドラの中で制御点を動かすだけにする**。データの作り替え（点の数が変わる作り直し）はタイマーで行い、Undo の区切りを足す（v0.69.1）。ハンドラの中で作ったデータは Undo の記録に入らず、Undo すると面が空になったり Blender が落ちたりする

## 設計の原則

| 用途 | 方式 |
|---|---|
| 一発で結果が決まる処理 | ボタン（Operator）。Undo できれば足りる |
| ドラッグしながら形を見て決める | モーダル（pick.ModalPicker）。開始〜確定〜Esc 取消 |
| 制御点を選んで動かす | 制御メッシュ＋Blender 標準の G/R/S |
| 状態の表示 | GPU オーバーレイ（描画のみ） |
| 作った後の値の直接ドラッグ | ギズモ（レシピの値を書き換えて作り直す） |

## できていること（v0.72.0）

| 分類 | 機能 |
|---|---|
| 作る | Add Patch（Plane / Arc / Dome）、Presets（Hood／Roof／Door／Hood + Fender）、Blueprints（下絵）、Line / Arc（境界用のカーブ）、Coons、Loft（Extra Rows）、Curve Network（＋ Add Curve Grid、Network Check、Move Crossing、Fair Curves）、Snap Ends、Project Curve |
| 伸ばす・つなぐ | Extend Edge、Bridge Edges、Sweep（Straight、Target）、Match Edge（G0/G1/G2、Spread、大きく動かしても切れない追従）、Corner Fill、Fill Hole（3辺以上、カーブ混在、G0 は平らなフタ）、Flange（Outside、Edit Flange）、Fillet（複数の継ぎ目、トリム、コーナーパッチ、半径の変更）、Split Patch、Duplicate Edge as Curve（辺・等パラメータ線をカーブに。Shift+クリックで複数）、Offset Surface、Continuity の後からの変更 |
| まとめて扱う | Live Update、Linked Edit、Edge Chain、Merge Patches、Cage、Mirror Group（Scope、Unlink、Remove）、Select／Regenerate Group、Follow Arrows／List Links（追従関係の表示） |
| 面の内側の編集 | Soft Pull（起点：点／線／制御点、減衰5種、連続ストローク）、Control Mesh、Fair、Refine、Add Row、Origin to Center、Align Normals |
| 確認 | Zebra（Width、Soft）、Seam Colors、Curvature Comb、Curvature Map、Control Points、Check Continuity（悪い順、ジャンプ）、Panel Gap |
| ギズモ | Extend の長さ、Offset Surface の距離、Fillet の半径 |
| 出力 | To Mesh（均一な密度、T 字頂点の接続と溶接、四角形化、Curve Detail、面の向きの統一）、Trim to Mesh（見た目の確認用の試作） |
| 基盤 | 複数スパンの NURBS、面ごとの作り方の記録（レシピ）と Regenerate、手で直した面の検出、名前の変更の追従、Undo／Redo 後の同期、Copy Bug Report |

## 公開後の候補（v0.7x〜、必要になったら）

公開後は、使った人から出た詰まりを最優先にする。それまでの候補：

| 候補 | 内容 |
|---|---|
| Panel Gap をそろえる操作 | 今は測って表示するだけ。どちらの面を動かすかの決め方を、実機で使ってから決める |
| Sweep の断面の回転 | レールの向きに合わせて断面を回すモード（今は平行移動のみ） |
| Presets の追加 | Roof の側面、Trunk、Bumper など |
| シーン複製の自動の付け直し | 今は Group 欄のボタンで付け直す。複製を見つけたら自動で付け直すかは、使い方を聞いてから |
| 相互の追従 | Match Edge の追従は一方向のみ（相互にするとループで両方作り直されない）。両側を同時に合わせる仕組みが要るかは使ってから |
| Follow Arrows の弧 | 弧の膨らみは視点によらず固定。真上に重なる面（Offset など）は見る向きによって短く見える |

### 残っている近似（必要になったら）
- Fillet：切り詰める量は継ぎ目全体で一定（角度が辺に沿って大きく変わる形では近似）
- Fill Hole の角：端点付近（コーナーのごく近く）だけ、隣の面との法線が 0.6〜1.2° ずれて G0 判定になる（G1 判定のしきい値 0.1°）。角の近くを平均して作る仕組み上の近似で、内側は 0° にそろう。サンプルでも Fender と Fill の継ぎ目4本がこれ
- Fill Hole：4辺以外の穴は中心点の付近で G2 がわずかに崩れる（最大 1° 程度）
- 2本だけ丸めた角：端の形は両方の弧のずれを足し合わせた近似。元の面が曲面だと、角の端のごく近くで G2 がわずかに崩れることがある

## 遠い目標（案）

| 目標 | 内容 | 前提 |
|---|---|---|
| 車体1台を通しで作る | 左右対称の車体全体（ボディ、パネル、開口部）をこのアドオンだけで作る。製品として見せられる作例にする | 公開後の詰まりの解消 |
| 真の NURBS 出力 | メッシュではなく、STEP / IGES などに書き出す。CAD や他ツールにつなげる | 内部の NURBS 表現の整理、トリムの扱い |
| 履歴つきの編集 | レシピをたどって、上流の境界を動かすと全体が作り直される。いまの Live Update を全体に広げ、手動修正との共存ルールを決める | レシピの網羅、手で直した面の扱い |
| スケッチから面へ | 下絵の上に描いたカーブから、ネットワークと面を自動で提案する | Curve Network の判定の頑健化 |
| 品質の自動診断 | 継ぎ目の連続性、曲率の跳び、反射線の乱れを自動で見つけて直し方を提案する（Class-A 的な基準） | 解析機能の拡充 |
| Geometry Nodes 連携 | 面や制御点を Geometry Nodes から読める形で出し、パラメトリックなバリエーションを作れるようにする | 出力形式の整理 |

## これまでの経過

細かな内容は `docs/CHANGELOG.md` とコミットログ。

| 版 | 内容 |
|---|---|
| v0.13〜v0.21 | Mirror、Split、5辺以上・カーブ混在の穴埋め、Seam Colors と Offset、Curve Network、Fill Hole の Radial、Linked Edit / Edge Chain / Merge |
| v0.24〜v0.27 | 面の内側の編集：ロック領域（G0/G1/G2 で 1〜3 列を固定）、Soft Pull、Control Mesh、Fair と Refine |
| v0.28〜v0.35 | Fillet（1クリックで選び、半径分トリムして G1/G2 でつなぐ。複数の継ぎ目、コーナーパッチ）、Curvature Comb / Map、Soft Pull の拡張、ギズモ、To Mesh の T 字頂点の接続 |
| v0.36〜v0.40 | 整理（Fillet を `filleting.py` へ、`tools/build_zip.py`）、Network Check、Fillet のやり直しと半径ギズモ、Move Crossing・Fair Curves、Add Boundary Curve（Line / Arc） |
| v0.40.3〜v0.40.5 | Fill Hole の自由カーブの端の向き（59° → 0°）、曲率表示の差分更新（面40枚で 0.62 秒 → 0.03 秒）、2本だけ丸めた角のマイター |
| v0.41 | Flange（辺から壁を立ち上げ、折れ目を Fillet で丸める） |
| v0.42〜v0.45 | Panel Gap（測定と表示）、Presets、Blueprints、Project Curve |
| v0.46 | Trim の試作。パッチにトリムを持たせるのは To Mesh の再設計に近く見送り。窓・ライトは別パッチで囲む組み方（マニュアルの作例 C） |
| v0.47 | To Mesh の四角形化 |
| v0.48〜v0.50 | Zebra の Width／Soft、連続性レポート（悪い順、ジャンプ）、Live Update の軽量化（グラフのキャッシュ、同じ形で止める） |
| v0.51〜v0.57 | Loft の Extra Rows、Add Row、Mirror の Scope、Flange Outside、Copy Bug Report、Mirror の Unlink / Remove、面の原点を保つ・Origin to Center |
| v0.58〜v0.61 | Sweep（Straight、Target） |
| v0.60〜v0.62 | 面の向き（Align Normals、作った面と To Mesh の向きをそろえる）、`bridging.py`・`tomesh.py` の分離 |
| v0.63〜v0.64 | To Mesh の Curve Detail、Edit Flange、Fillet の途中失敗で Live Update が止まる不具合の修正、Undo／Redo で作り直しが一歩遅れる問題の修正 |
| v0.65 | 名前を変えた source に追従する。消えた source を知らせる。Undo 後の作り直しで落ちる不具合の修正 |
| v0.66 | Extend・Bridge・Corner Fill の連続性を後から変える |
| v0.67 | Match Edge の追従を切れなくする（合わせた辺の組を記録。それまでは相手が大きさの約10%以上動くと切れていた）。Spread。追従の一覧・変更・解除 |
| v0.68 | Follow Arrows／List Links（追従関係を矢印と一覧で表示。状態の色分け） |
| v0.69 | LICENSE の同梱、マニフェストの権限。Live Update のデータの作り替えをハンドラの外へ移し、Undo すると依存する面が空になって落ちる不具合を修正（v0.68 以前からあった）。`tests/test_gui.py`。ギズモの起動時のエラーの修正 |
| v0.70 | 公開候補。Follow Arrows の Selected 範囲（既定）、サンプルの作り直し、掲載用スクリーンショット、ROADMAP の整理。v0.70.1：Use This Scene's Copies（Full Copy したシーンの面を、そのシーンの複製に付け直す）、公開前の確認をすべて自動テストに（`test_gui_modal.py`、`test_sample.py`）、掲載ページの文面と画像 |
| v0.71 | Duplicate Edge as Curve（面の辺・等パラメータ線を、区間ごとに1点のベジェカーブとして正確に複製。Shift+クリックで複数）。点の多いカーブだと Coons が極端に遅くなるため、面の線をそのまま区間単位で書き出す方式にした |
| v0.72 | Loft をクリックで選ぶ方式に（通す順＝クリック順、Backspace、プレビュー、事前の選択を引き継ぐ）。辺とカーブを選ぶツール（Bridge Edges、Fill Hole、Sweep の Target）に E キーの絞り込み。辺に重なったカーブが選び分けられないという実機での指摘から |
