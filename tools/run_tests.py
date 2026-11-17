"""Run every test and print one line per suite.

    py tools/run_tests.py            # all five suites
    py tools/run_tests.py core sample  # only these

Blender is taken from the BLENDER environment variable, else from the usual
install folders. The GUI suites (gui, modal) open a Blender window for a short
while; skip them with  --no-gui  (CI does this).
"""
import glob
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_blender():
    env = os.environ.get("BLENDER")
    if env:
        return env
    candidates = sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"))
    candidates += [r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
                   "/usr/bin/blender", "/Applications/Blender.app/Contents/MacOS/Blender"]
    for c in reversed(candidates):
        if os.path.exists(c):
            return c
    return "blender"


def suites(blender, logdir):
    py = sys.executable
    return {
        "core": ([py, "tests/test_core.py"], None),
        "blender": ([blender, "-b", "--factory-startup", "--python", "tests/test_blender.py"], "ALL OK"),
        "sample": ([blender, "-b", "--factory-startup", "samples/hood_fender.blend",
                    "--python", "tests/test_sample.py"], "ALL OK"),
        "gui": ([blender, "--factory-startup", "--python", "tests/test_gui.py", "--",
                 os.path.join(logdir, "gui.log")], os.path.join(logdir, "gui.log")),
        "modal": ([blender, "--factory-startup", "--enable-event-simulate", "--python",
                   "tests/test_gui_modal.py", "--", os.path.join(logdir, "modal.log")],
                  os.path.join(logdir, "modal.log")),
    }


def passed(name, proc, check):
    if proc.returncode != 0:
        return False
    out = proc.stdout + proc.stderr
    if name == "core":
        return "Traceback" not in out
    if name in ("blender", "sample"):
        return check in out.splitlines()
    with open(check, encoding="utf-8") as f:       # gui / modal: the log's last line
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return bool(lines) and lines[-1].startswith("SUMMARY ALL OK")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_gui = "--no-gui" in sys.argv
    logdir = tempfile.mkdtemp(prefix="ss_tests_")
    table = suites(find_blender(), logdir)
    names = args or [n for n in table if not (no_gui and n in ("gui", "modal"))]
    failed = []
    for name in names:
        cmd, check = table[name]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        ok = passed(name, proc, check)
        print("%-8s %s  (%.0fs)" % (name, "OK" if ok else "FAILED", time.time() - t0))
        if not ok:
            failed.append(name)
            tail = (proc.stdout + proc.stderr).splitlines()[-30:]
            print("    " + "\n    ".join(tail))
    print("ALL OK" if not failed else "FAILED: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
