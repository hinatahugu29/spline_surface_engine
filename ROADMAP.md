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

### v0.36 コードの整理 ← 完了
- Fillet 関連（約 280 行）を `operators.py` から `filleting.py` へ（`operators.py` は 1,699 → 1,421 行）。使われていない import も整理
- `tools/build_zip.py`：バージョンの食い違いを検出して止め、`dist/` などに ZIP を出す
- `test_blender.py` に v0.28 以降の追加分（Fillet の複数・コーナー・追従、Extend の矢印の作り直し、To Mesh の T 字頂点、Soft Pull の線・制御点、曲率表示）を入れ、Steam 版 Blender 5.2 で画面なしで通しで実行（`ALL OK`）
- 実行して分かったこと：v0.33.3 で To Mesh の分割数を変えた際、既存の頂点数の確認が古いままだった（今回、新しい仕様に合わせて更新）

### v0.37 Curve Network の表示 ← 完了（実機確認は並行継続）
- **Network Check**：Curve Network と同じ判定を、最初の問題で止めずに最後まで行い（`core.network.diagnose_network`、判定の前半は `build_network` と共通化）、方向の色分け・交点・足りない交点（最接近点を結ぶ赤線）・同じ方向どうしの交差・孤立したカーブ・順番の絡みを表示。パネルに要約
- **カーブの曲率コーム**：Curvature Comb がカーブにも出る。ベジェは制御点とハンドルから厳密に計算（制御点で曲率が跳ぶと歯が2本立つ）、ほかのカーブは分割した点列から
- Curve Network の許容誤差を「選んだ全体の大きさ」から「カーブ1本の大きさの中央値」に：遠くの1本が混ざっても許容誤差が膨らまない（膨らむと、隣り合う同じ方向のカーブまで「交差」とみなされていた）

### v0.38 Fillet をやり直せるように ← 完了（実機確認は並行継続）
- 切り詰めた面は新しいレシピ `fillet_trim`：切り詰める前のレシピ（base、手で直されていたらその時の形）と、フィレットの組（ID・継ぎ目・半径・G1/G2・メンバー）を持つ。作り直すときはメンバー全員の base から切り詰め前の形を計算して `multi_fillet` をやり直す（`operators.compute_fillet_trim`）
- base が依存する外部の面を sources に載せるので、それが動くと切り詰めからやり直され、つなぎ面・コーナーは Live Update で付いてくる
- 入れ子（フィレット済みの面をさらに丸める）は、base の中を辿って該当する組の切り詰め前を探す（`fillet_base`）
- 半径の変更 `filleting.set_fillet_radius`：メンバー全員の新しい形を先に計算し、全部できた時だけ書く（大きすぎる半径は何も変えずに拒否）。手で直したメンバーがいると止める
- 半径の矢印ギズモ：つなぎ面を選ぶと元の角からつなぎ面の中央へ（v0.33 の矢印の仕組みを流用）
- 制約：切り詰めた面を手で動かすと「手で直した面」扱いで、半径の変更は Regenerate するまで止まる（他の面と同じ考え方）

### v0.39 Curve Network の編集 ← 完了（実機確認は並行継続）
- **Move Crossing**：交点をつまんで動かすと、縦横2本のカーブがぴったり付いてくる。各ベジェ点をハンドルごと平行移動させ（向きが変わらないので G1 のまま）、「つまんだ点は目標へ・ほかの交点は動かない」を満たす中で一番小さい動きを解く（`core.curveedit.pull_chain`）。Soft Pull と同じ連続ストローク
- **Fair Curves**：制御多角形の2階差分でならし、端点・端の向き・交点は拘束で固定、最後にハンドルを一直線に揃えて G1（`core.curveedit.fair_chain`）
- 編集したカーブのハンドルは Aligned に（Auto のままだと Blender が計算し直して編集が消える）
- **左右対称のカーブ編集は見送り**：Curve Network は横方向のカーブが中心線をまたいで1本につながっている必要があり、カーブを半分ずつミラーすると網にならない。片側の網から面を作って Mirror Group で反転する方が確実

### 次の候補（決まり次第ここに）
- 残課題・保留から選ぶ。実機確認で見つかった不便を優先する

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
