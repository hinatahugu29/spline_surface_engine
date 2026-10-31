"""Build the zip to upload to Booth: the add-on zip, the documents and the sample.

    py tools/build_booth_pack.py

Runs build_zip first (so the version check applies), then writes
dist/spline_surface-booth-<version>.zip holding one folder with the add-on zip
(install it as it is, do not unpack it), docs/ and samples/. The sample is
built with tools/build_sample_hood.py; it is skipped with a warning if missing.
"""
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_zip  # noqa: E402

ROOT = build_zip.ROOT
DOCS = ("MANUAL.md", "FEATURES.md", "WORKFLOWS.md", "CHANGELOG.md")
README = """Spline Surface %(v)s

1. Blender の「編集 → プリファレンス → アドオン」を開く
2. 右上のメニューから「ディスクからインストール」を選び、spline_surface-%(v)s.zip を指定する
   （ZIP は解凍せず、そのまま選びます）
3. 3D ビューで N キーを押し、サイドバーの「Spline Surface」タブを開く

docs/MANUAL.md      説明書
docs/WORKFLOWS.md   目的別の作業の流れ（6例）
docs/FEATURES.md    機能一覧
docs/CHANGELOG.md   更新履歴
samples/            サンプル（ボンネット+フェンダー）

Blender 5.0 以降 / 不具合は説明書の「不具合報告」を見てください
"""


def main():
    version = build_zip.manifest_version()
    build_zip.write_zip(os.path.join(ROOT, "dist", "spline_surface-%s.zip" % version)) \
        if build_zip.bl_info_version() == version else sys.exit("version mismatch")
    addon = os.path.join(ROOT, "dist", "spline_surface-%s.zip" % version)
    out = os.path.join(ROOT, "dist", "spline_surface-booth-%s.zip" % version)
    top = "Spline_Surface_%s" % version
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(addon, "%s/spline_surface-%s.zip" % (top, version))
        zf.writestr("%s/README.txt" % top, README % {"v": version})
        for name in DOCS:
            zf.write(os.path.join(ROOT, "docs", name), "%s/docs/%s" % (top, name))
        sample = os.path.join(ROOT, "samples", "hood_fender.blend")
        if os.path.exists(sample):
            zf.write(sample, "%s/samples/hood_fender.blend" % top)
        else:
            print("warning: samples/hood_fender.blend is missing (run tools/build_sample_hood.py)")
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
