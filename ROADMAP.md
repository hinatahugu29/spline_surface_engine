# Spline Surface ロードマップ

車のような滑らかな曲面を、NURBS の慣れなしで作れるようにする Blender 5.x アドオン。
最終出力はメッシュ。操作はボタンとクリック選択が基本で、ギズモは操作が固まってから載せる。

## できていること（〜 v0.35）

| 分類 | 機能 |
|---|---|
| 作る | Add Patch（Plane / Arc / Dome）、Coons、Loft、Curve Network（＋ Add Curve Grid）、Snap Ends |
| 伸ばす・つなぐ | Extend Edge、Bridge Edges、Match Edge（G0/G1/G2）、Corner Fill、Fill Hole（3辺以上、カーブ混在、G0 は平らなフタ）、Fillet（複数の継ぎ目、トリム、コーナーパッチ）、Split Patch、Offset Surface |
| まとめて扱う | Live Update、Linked Edit、Edge Chain、Merge Patches、Cage、Mirror Group |
| 面の内側の編集 | Soft Pull（起点：点／線／制御点、減衰5種、連続ストローク）、Control Mesh、Fair、Refine |
| 確認 | Zebra、Seam Colors、Curvature Comb、Curvature Map、Control Points、Check Continuity（対応する辺の名前つき） |
| ギズモ | Extend の長さ、Offset Surface の距離 |
| 出力 | To Mesh（面の大きさに合わせた均一な密度、継ぎ目の T 字頂点をつないで溶接） |
| 基盤 | 複数スパンの NURBS、面ごとの作り方の記録（レシピ）と Regenerate、手で直した面の検出 |

## これまでの経過

| 版 | 内容 |
|---|---|
| v0.13〜v0.21 | Mirror、Split、5辺以上・カーブ混在の穴埋め、使い勝手の整理、Seam Colors と Offset、Curve Network、Fill Hole の Radial、Linked Edit / Edge Chain / Merge |
| v0.24〜v0.27 | 面の内側の編集：ロック領域（G0/G1/G2 で 1〜3 列を固定）、Soft Pull、Control Mesh、Fair と Refine |
| v0.28 | Fillet：ぴったり接して折れた継ぎ目を1クリックで選び、半径分トリムして G1/G2 でつなぐ |
| v0.29 | Curvature Comb / Curvature Map |
| v0.30〜v0.32 | Soft Pull の拡張：線起点、連続ストローク、減衰カーブ、制御点起点 |
| v0.33 | ギズモ（Extend の長さ、Offset の距離）、To Mesh の T 字頂点の接続と密度の均一化 |
| v0.34〜v0.35 | Fillet を複数の継ぎ目でまとめて、コーナーパッチ、Fill Hole G0 の平らなフタ |

## 設計の原則（引き続き）

| 用途 | 方式 |
|---|---|
| 一発で結果が決まる処理 | ボタン（Operator）。Undo できれば足りる |
| ドラッグしながら形を見て決める | モーダル（pick.ModalPicker）。開始〜確定〜Esc 取消 |
| 制御点を選んで動かす | 制御メッシュ＋Blender 標準の G/R/S |
| 状態の表示 | GPU オーバーレイ（描画のみ） |
| 作った後の値の直接ドラッグ | ギズモ（レシピの値を書き換えて作り直す） |

- 計算は `core/`（Blender 非依存）に置いて単体テストを書く。Blender 側のつなぎ込み（レシピ、Live Update、Undo）は実機確認で補う
- 実機確認は並行で継続し、ロードマップの進行はブロックしない

## これから（v0.36〜）

### v0.36 コードの整理 ← 次（小さく安全、この後の作業を楽にする）
- Fillet 関連（約 250 行）を `operators.py`（1,699 行）から専用ファイルへ分ける（Split Patch と同じ形）。v0.38 で Fillet を作り直すので先に済ませる
- ZIP のビルドを小さなスクリプトに：`blender_manifest.toml` と `__init__.py` のバージョンが揃っているかも確かめて `dist/` に出す
- `test_blender.py`（Blender 上での手動確認スクリプト）に v0.28 以降の追加分を入れる：Fillet（複数・コーナー）、ギズモ、Soft Pull の各モード、To Mesh

### v0.37 Curve Network の表示（確認系、描画のみ）
- **網の診断表示**：横・縦に分類されたカーブを色分け、見つかった交点に印、交差していない箇所はどれだけ離れているかを表示。今は「全部交差」「同じ向きどうしは交差しない」を守れていないとエラーになるだけなので、どこを直すかが見えるようにする
- **カーブの曲率コーム**：Curvature Comb をカーブにも出す。キャラクターラインのうねりを面にする前に直せる

### v0.38 Fillet をやり直せるように（設計変更）
- 今の制約：トリムした面は作り方の記録を失った「固定された面」になり、元の面を動かしても付いてこない。半径も後から変えられない
- Fillet のレシピに、トリム前の面の形・継ぎ目・半径を残す。半径の変更や元の面の変更は「トリム前に戻してやり直す」で処理する
- その上で Fillet の半径ギズモ（v0.33 の矢印の仕組みを流用）

### v0.39 Curve Network の編集
- **交点をつまんで動かす**：交点をドラッグすると、そこを通る縦横2本のカーブが一緒に動く
- **カーブのならし（Fair）**：端点と交点は固定したまま、カーブのうねりだけを取る
- **左右対称のカーブ編集**：片側を描けば反対側が付いてくる（Mirror Group のカーブ版）

### 残課題（必要になったら）
- Fillet：切り詰める量は継ぎ目全体で一定（角度が辺に沿って大きく変わる形では近似）。2本だけ丸めた角は、つなぎ面の端が斜めになりきれいなコーナーにならない
- 曲率表示：面を1つ動かすたびに全部の面を計算し直す。面が数十枚で重くなるようなら、変わった面だけ計算し直す
- To Mesh：T 字頂点をつないだ所は5角形以上の面になる
- Fill Hole：4辺以外の穴は中心点の付近で G2 がわずかに崩れる（最大 1° 程度）

### 保留・後回し
- **開口部のトリム**（窓・ライト）：NURBS 上のブール的処理で実装コストが高く効果が読みにくい
- **Flange / Return**（縁の折り返し）：Offset と Fillet の仕組みを組み合わせれば作れそう
- **Project Curve**（カーブを面に投影）：開口部のトリムの前段として要るようになったら
- **Presets**（ボンネット・ルーフ等の初期構成）、下絵配置の補助、サンプル .blend と使い方ドキュメント
