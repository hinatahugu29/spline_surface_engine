# Spline Surface ロードマップ

車のような滑らかな曲面を、NURBS の慣れなしで作れるようにする Blender 5.x アドオン。
最終出力はメッシュ。操作はボタンとクリック選択が基本で、ギズモは操作が固まってから載せる。

## 現在地（2026-11-19、v0.86.1）

**v0.72.0 を公開候補（Release Candidate）とした。** その後 v0.73〜v0.76 で、内部の整理（下の「v0.73〜v0.76 の整理」）と、Even Gap・Presets の追加・Sweep の Rotate・Diagnose・Fill Hole の角の改善を入れた。v0.77〜v0.82 では、2つ目のサンプル（車体の前半＋キャビン、`tools/build_sample_body.py`）を作って詰まりを拾い、直した（`docs/VALIDATION_body.md`）。v0.82.1〜v0.82.3 では、Blender の画面を開いてランダムに操作を続ける確認で、Undo とファイルの開き直しまわりの不具合（Zebra と Ctrl+Z で落ちる、File → Open で Live Update が止まる、など）を見つけて直した。第1〜第4段階の作業はすべて終わった。公開前の確認も、自動で確かめられるものはすべてテストにした。残りは公開の手続き（アカウントでの操作）と、手で触ったときの感触の確認（`docs/CHECKLIST_hands_on.md`）だけ。

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
| サンプル .blend | ✅ 2つ。ボンネット＋フェンダー（`tools/build_sample_hood.py`）と車体の前半＋キャビン（`tools/build_sample_body.py`）。どちらもスクリプトで毎回同じ手順で作り直せて、開いて全体を作り直しても失敗ゼロ。Booth 一式に両方入る |
| 掲載用のスクリーンショット | ✅ `docs/images/`（Zebra、Seam Colors、Follow Arrows、To Mesh、パネル全体、車体の Zebra、Diagnose）。`tools/take_screenshots.py` がサンプルから撮り直す（v0.82.0 で現行版に更新）。README に掲載 |
| 自動テスト | ✅ 6本。`tests/test_core.py`（計算）、`tests/test_blender.py`（Blender、バックグラウンド）、`tests/test_gui.py`（画面つき：本物の Undo／Redo、名前の変更、大きな移動、点の数が変わる作り直し）、`tests/test_gui_modal.py`（画面つき、マウスとキーのイベントを再現：Fillet・Flange・Fill Hole・Esc）、`tests/test_sample.py`（ボンネットのサンプルを開いて全体を作り直し、継ぎ目の角度と To Mesh の結果を検査）、`tests/test_body.py`（車体のサンプルで同じこと、加えて Even Gap の追従や Diagnose の印）。走らせ方は各ファイルの先頭、まとめては `py tools/run_tests.py` |
| 公開前の確認（下の6項目） | ✅ 6項目とも自動テストで確認（v0.70.1） |
| シーンを複製するアドオンとの併用 | ✅ Blender の Scene > Full Copy（シーンを複製するアドオンが使う仕組み）で再現して確認。複製した面が元のシーンのカーブを読み続ける問題が見つかり、Group 欄に **Use This Scene's Copies** を追加（v0.70.1）。Shot Manager 本体での確認は、このPCに入っていないため未 |
| 手で触ったときの感触 | ⬜ **人の作業**。ドラッグ中の追従の滑らかさ、パネルの見え方の好みなど、感じ方の確認。自動テストでは結果しか確かめられない。項目は `docs/CHECKLIST_hands_on.md` |
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

### v0.73〜v0.76 の整理

| 項目 | 内容 |
|---|---|
| テストの一括実行 | `py tools/run_tests.py`（5本、`--no-gui` で画面つきを除く。Blender は環境変数 `BLENDER` か既定の場所） |
| CI | `.github/workflows/tests.yml`：計算のテストと、Blender 5.0 のバックグラウンドのテスト（blender、sample、body）。push のたびに GitHub で走る |
| 形の回帰チェック | `tests/sample_snapshot.json`：サンプルを作り直した制御点を記録と比べる。形を変える修正のあとは `-- --update` で書き直す |
| `rebuild.py` | 作り直しの計算と、作り方の登録（`@recipe("name")`）。新しい種類の面は登録だけで Live Update・Regenerate に乗る |
| 分割 | `operators.py` → `creating.py`・`matching.py`・`extending.py`・`holefill.py`。`live.py` → `graph.py`（依存関係）・`links.py`（連続性と追従の変更）・`live.py`（作り直しとハンドラ） |

## カーブ編集（v0.83〜）

面の道具がそろったあと、「カーブ系は？」という声が増えた。このアドオンの中でのカーブに限らず、Blender のベジェ編集そのものの扱いにくさ（点とハンドルしかつかめない、形を保ったまま点を足す・消す・ずらす手段がない、意味のある方向に動かせない、G2 を保てない）を埋める。面がなくてもふつうのベジェカーブに使え、面と連動する部分（端の条件、網、Live Update）はその上に重ねる。

| 段階 | 内容 | 状況 |
|---|---|---|
| A 点の構成を変える（形は保つ） | Curve Points（クリックで足す・ドラッグで滑らせる・Ctrl+クリックで消す、モーダル）、編集モードの Slide Points・Dissolve Points | ✅ v0.83.0 |
| B 形を変える | Curve Pull（曲線上のどこでもつかんで引く。範囲・減衰・方向、端の固定、網の交点も付いてくる。P でカーブ自身の平面に固定） | ✅ v0.83.0 |
| B' 意味のある向きに動かす | Constrained Move（編集モード。接線・法線方向、ハンドルの長さだけ、G2 を保つ） | ✅ v0.84.0 |
| C 整える | Refit Curve（形を保って点を減らす）、Make G2（ハンドルの長さで曲率をそろえる。全部の点をまとめて解く）、Fair Curves（既存） | ✅ v0.84.0 |
| D つなぐ・条件 | Attach Curve Ends（端をほかのカーブの端に G0/G1/G2 で、面に沿う／面から立ち上がる向きで）、端を辺やカーブの上に乗せて滑らせる（Curve Points） | ✅ v0.85.0 |
| E 作る | Two-View Curve（2方向の図から 3D カーブ）、Curve Fillet、Offset Curve、Extend Curve | ✅ v0.86.0 |
| F 見る | Point Continuity（点ごとの G2／G1／角の色）、曲率の櫛の追従を速く（0.15 秒） | ✅ v0.85.0 |

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

## できていること（v0.82.3）

| 分類 | 機能 |
|---|---|
| 作る | Add Patch（Plane / Arc / Dome）、Presets（Hood／Roof／Door／Trunk／Bumper／Hood + Fender／Roof + Side）、Blueprints（下絵）、Line / Arc（境界用のカーブ）、Coons、Loft（Extra Rows）、Curve Network（＋ Add Curve Grid、Network Check、Move Crossing、Fair Curves）、Snap Ends、Project Curve |
| 伸ばす・つなぐ | Extend Edge、Bridge Edges、Sweep（Straight、Target、Rotate）、Match Edge（G0/G1/G2、Spread、大きく動かしても切れない追従）、Corner Fill、Fill Hole（3辺以上、カーブ混在、G0 は平らなフタ、まっすぐ続く辺は1辺に）、Flange（Outside、Edit Flange）、Fillet（複数の継ぎ目、トリム、コーナーパッチ、半径の変更）、Split Patch（追従の指定を引き継ぐ）、Duplicate Edge as Curve（辺・等パラメータ線をカーブに。Shift+クリックで複数）、Offset Surface、Continuity の後からの変更 |
| まとめて扱う | Live Update、Linked Edit、Edge Chain、Merge Patches、Cage、Mirror Group（Scope、Unlink、Remove）、Select／Regenerate Group、Follow Arrows／List Links（追従関係の表示） |
| 面の内側の編集 | Soft Pull（起点：点／線／制御点、減衰5種、連続ストローク）、Control Mesh、Fair、Refine、Add Row、Origin to Center、Align Normals |
| 確認 | Zebra（Width、Soft）、Seam Colors、Curvature Comb、Curvature Map、Control Points、Check Continuity（悪い順、ジャンプ、辺の一部だけで接する継ぎ目も）、Diagnose（直し方の案つき、意図どおりの印）、Panel Gap と Even Gap |
| ギズモ | Extend の長さ、Offset Surface の距離、Fillet の半径 |
| 出力 | To Mesh（均一な密度、T 字頂点の接続と溶接、四角形化、Curve Detail、面の向きの統一、3枚以上が付く辺の印）、Trim to Mesh（見た目の確認用の試作） |
| 基盤 | 複数スパンの NURBS、面ごとの作り方の記録（レシピ）と Regenerate、手で直した面の検出、名前の変更の追従、Undo／Redo 後の同期、Copy Bug Report |

## 公開後の候補（v0.7x〜、必要になったら）

公開後は、使った人から出た詰まりを最優先にする。それまでの候補：

| 候補 | 内容 |
|---|---|
| シーン複製の自動の付け直し | 今は Group 欄のボタンで付け直す。複製を見つけたら自動で付け直すかは、使い方を聞いてから |
| 相互の追従 | Match Edge の追従は一方向のみ（相互にするとループで両方作り直されない）。両側を同時に合わせる仕組みが要るかは使ってから |
| Follow Arrows の弧 | 弧の膨らみは視点によらず固定。真上に重なる面（Offset など）は見る向きによって短く見える |
| To Mesh で3枚が付く辺 | いまは警告して頂点グループに入れるだけ（車体のサンプルのモールの付け根）。そういう面を別のメッシュにする選択肢 |
| Even Gap のはみ出した部分 | 向かい合わない部分は元の位置にゆるく留まるだけ。そろえた部分へなめらかに曲げてつなぐか |
| Refine・Add Row と追従 | Match Edge や Even Gap で付いていく面に Refine・Add Row をかけると、点の数が変わって以後付いていかない。指定を新しい点の数で取り直すか |
| 操作が続いたときの重さ | 面が数十枚あるファイルで操作を繰り返したときの所要時間は、まだ測っていない（車体のサンプルで、36枚に6種類の操作を順にかけると全体で10分前後） |

### 残っている近似（必要になったら）
- Fillet：切り詰める量は継ぎ目全体で一定（角度が辺に沿って大きく変わる形では近似）
- Fill Hole の角：端点付近では曲率（G2）が崩れる。v0.76 で角の付近は接平面（G1）を優先するようにし、法線のずれは 0.55° → 0.07° 前後（G1 判定）になった。曲率まで合わせるには角のねじれの整合が要る
- Fill Hole：4辺以外の穴は、中心で面どうしが G1（v0.81。それまでは中心の付近で最大 1° 程度折れていた）。曲率（G2）は中心の付近で合わない
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
| v0.73〜v0.76 | Even Gap、Presets（Trunk、Roof + Side、Bumper）、Sweep の Rotate、Diagnose、Fill Hole の角を G1 に。内部の整理（`rebuild.py` のレシピ登録、`operators.py`・`live.py` の分割）、テストの一括実行と CI、形の回帰チェック |
| v0.77〜v0.81 | 2つ目のサンプル（車体の前半＋キャビン）で拾った詰まりの修正：一部だけ向かい合う縁の Even Gap、Fill Hole の角のひだ（本当の角の数で埋め、分けた面どうしは中心まで G1）、Split Patch の後の追従、辺の一部だけで接する継ぎ目の測定、Diagnose の「意図どおり」の印、To Mesh の非多様体の印 |
| v0.82 | Seam Colors を Check Continuity・Diagnose と同じ測り方に（フィレットで切り離した2枚を「open」と塗っていた）。掲載用の画像をスクリプトで撮り直し、Booth 一式に車体のサンプル、CI の更新。v0.82.1〜v0.82.3：画面つきのランダム操作で見つけた不具合の修正（Zebra と Ctrl+Z で落ちる、File → Open で Live Update が止まる、開いた直後の最初の変更、操作と作り直しを Ctrl+Z 1回で戻す、何度か Undo すると付いてこなくなる） |
| v0.83 | カーブ編集 A・B：Curve Points（点を足す・滑らせる・消す、どれも形を保つ。カーブ上をドラッグで引く Curve Pull）、編集モードの Slide Points・Dissolve Points。下の「カーブ編集」の段階表を参照 |
| v0.84 | カーブ編集 B'・C：Constrained Move（点自身の向きに動かす・ハンドルの長さだけ・G2 を保つ）、Make G2、Refit Curve |
| v0.85 | カーブ編集 D・F：Attach Curve Ends、端の吸着、Point Continuity、曲率の櫛の追従 |
| v0.86 | カーブ編集 E：Two-View Curve、Curve Fillet、Offset Curve、Extend Curve。カーブ編集の段階表（A〜F）がすべて完了。v0.86.1：Curve Pull の専用ボタン、2点だけのカーブなど動かせる点が近くにないときに引けなかった不具合を修正（つかんだ所に点を足してから引く） |
