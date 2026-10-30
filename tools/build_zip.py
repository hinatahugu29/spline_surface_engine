"""Build the add-on zip for Blender's "Install from Disk".

    py tools/build_zip.py [--out DIR ...]

Reads the version from blender_manifest.toml, stops if __init__.py's
bl_info version disagrees, and writes spline_surface-<version>.zip (the
manifest at the zip root, no caches or dotfiles) into every --out directory
(default: dist/ next to this repository's spline_surface/).
"""
import argparse
import ast
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "spline_surface")


def manifest_version():
    text = open(os.path.join(PKG, "blender_manifest.toml"), encoding="utf-8").read()
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


def bl_info_version():
    tree = ast.parse(open(os.path.join(PKG, "__init__.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "bl_info":
            info = ast.literal_eval(node.value)
            return ".".join(str(n) for n in info["version"])
    return None


def write_zip(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(PKG):
            dirnames[:] = sorted(d for d in dirnames
                                 if d != "__pycache__" and not d.startswith("."))
            for name in sorted(filenames):
                if name.startswith(".") or name.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, name)
                zf.write(full, os.path.relpath(full, PKG))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", action="append",
                        help="directory to put the zip in (repeatable; default dist/)")
    args = parser.parse_args()
    version, info = manifest_version(), bl_info_version()
    if version != info:
        sys.exit("version mismatch: blender_manifest.toml says %s, __init__.py bl_info says %s"
                 % (version, info))
    outs = args.out or [os.path.join(ROOT, "dist")]
    name = "spline_surface-%s.zip" % version
    first = os.path.join(outs[0], name)
    os.makedirs(outs[0], exist_ok=True)
    write_zip(first)
    print("wrote", first, os.path.getsize(first), "bytes")
    for out in outs[1:]:
        os.makedirs(out, exist_ok=True)
        shutil.copy2(first, os.path.join(out, name))
        print("copied to", os.path.join(out, name))


if __name__ == "__main__":
    main()
