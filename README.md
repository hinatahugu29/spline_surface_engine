# Spline Surface

車のボディのような滑らかな曲面を、NURBS に慣れていなくても作れるようにする Blender 5.x 用アドオンです。
面は Blender の NURBS サーフェスで作り、最後にメッシュにして使います。

## インストール
1. `dist/` にある最新の `spline_surface-x.y.z.zip` を用意する
2. Blender の「編集 → プリファレンス → アドオン」の右上メニューから「ディスクからインストール」で ZIP を選ぶ
3. 3D ビューのサイドバー（N キー）に **Spline Surface** タブが出ます

## 基本の流れ
1. **作る**：Add Patch で面を出すか、カーブを3〜4本描いて Coons、断面カーブを並べて Loft、縦横に交差するカーブの網から Curve Network
2. **伸ばす・つなぐ**：Extend Edge で辺から伸ばし、Corner Fill・Fill Hole・Bridge Edges で隙間を埋める
3. **確認する**：Seam Colors（G2 緑・G1 黄・G0 赤）と Zebra（縞模様）でつなぎ目の滑らかさを見る
4. **まとめて直す**：元の面を動かせば Live Update で全体が付いてくる。大きな変形は Cage、左右対称は Mirror Group
5. **出力する**：To Mesh で1つのメッシュに溶接

## パネル
| 欄 | 内容 |
|---|---|
| Create | Add Patch、Coons、Loft、Curve Network、Snap Ends（カーブの端をくっつける）、Regenerate、Fit Settings |
| Grow & Connect | Extend Edge、Bridge Edges、Match Edge、Corner Fill、Fill Hole、Split Patch、Offset Surface |
| Group | Live Update、Select / Regenerate、Mirror Group、Cage（Add / Apply / ✕） |
| Inspect | Zebra、Seam Colors（つなぎ目を継続性で色分け）、Check Continuity（結果の一覧） |
| Output | Mesh Resolution、To Mesh |

## クリックで選ぶツールの操作
Extend Edge・Corner Fill・Bridge Edges・Fill Hole・Split Patch は、押すと 3D ビューで選ぶモードに入ります。

| 操作 | 内容 |
|---|---|
| マウスを近づける | 候補がオレンジに光り、名前とプレビューが出る |
| 左クリック | 選ぶ（選び済みをもう一度クリックすると外す） |
| ホイール | Extend は長さ、ほかは Tension（両側の面の向きを引き継ぐ強さ） |
| C | 継続性の切り替え（G0 / G1 / G2） |
| F | Bridge Edges：端の組み合わせを反転 |
| D / S | Split Patch：切る向き／区間の境目への吸着 |
| Enter / Space | Fill Hole：選んだ辺で埋める |
| 中ボタン | 視点の回転（モード中も使える） |
| Esc / 右クリック | やめる |

作成後は、左下の「最後の操作を調整」パネルで値を変えられます。

## 継続性の目安
- **G0**：位置だけつながる（折れ目が見える）
- **G1**：向きまでそろう（縞模様が折れずにつながる）
- **G2**：曲がり具合までそろう（縞模様の太さも途切れない。車の外板の目標）

## 知っておくと良いこと
- 手で編集した面は「手で直した」と記録され、Regenerate や Live Update で上書きされないようにしています（Regenerate を押した場合は警告が出て、Ctrl+Z で戻せます）
- 対応する面：Endpoint がオン、Cyclic なし、制御点の Weight がすべて 1
- Corner Fill の頂点と、4辺以外の Fill Hole の中心付近は、わずかに近似になります
- Split Patch で切った面は、作り方の記録を持たない「固定された面」になります
- Curve Network：縦方向と横方向のカーブを描き、全部選んで押す。どのカーブも反対方向のカーブ全部と交差している必要がある（少しずれていても、サイズの5%程度までなら自動で交点を通るように曲げる）。外周より外にはみ出た部分は無視される
- Offset Surface は近似（標準で誤差は距離の1%程度以下）。元の面に連動します

## 試してほしいこと（v0.16 の調整用チェックリスト）
気になった点があれば、番号と一緒に教えてください。

1. Add Patch → Extend Edge を何回か：辺の選びやすさ、ラベルの見やすさ、ホイールでの長さ調整
2. Corner Fill / Bridge Edges / Fill Hole：2本目以降の選び方、プレビューの分かりやすさ
3. 元の面を G で動かす：Live Update の追従と重さ（面が10枚を超えるくらいで）
4. Mirror Group → 片側を編集：反対側の追従、中心線のゼブラ
5. Cage：Add → Lattice 編集 → Apply の流れ
6. Split Patch：切る位置の分かりやすさ、D / S キー
7. To Mesh：溶接結果、シェーディング
8. パネル全体：並び順、名前、足りない／要らないボタン
