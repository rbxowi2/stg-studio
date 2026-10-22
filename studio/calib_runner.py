"""Start the calibrate.py subprocess, run a read-only pre-check, read the acceptance report.

No tkinter here: pure functions plus one state machine, testable from the
command line (like prep_runner.py).

The tab's behaviour = `calibrate.py <calibration set> --edge x6`. The UI also
passes --yes, which only matters when no scale is given; the UI requires all
six distances, so behaviour is the same. Without --yes, calibrate.py could
wait on input() in a subprocess that has no console.

Command-line self-test:
    python studio/calib_runner.py <calibration set>
"""
import glob
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from joblimit import adopt                                    # noqa: E402
import params as P                                            # noqa: E402

CALIB = os.path.join(_ROOT, "pipeline", "calibrate.py")
PY = sys.executable
LOG_NAME = "calib.log"
MIN_SHOTS = 10           # same number as calibrate.py's warning when checking the set


def precheck(calib):
    """Read-only: reads session.json and lists file names; writes nothing.

    The checks are the warnings calibrate.py's main() prints when checking
    the set (the same criteria).
    Returns {"ok", "why", "warn": [...], "cams", "shots", "res", "has_rig"}.
    """
    res = {"ok": False, "why": "", "warn": [], "cams": [], "shots": 0,
           "res": None, "has_rig": False}
    if not calib or not os.path.isdir(calib):
        res["why"] = "Choose a calibration set folder"
        return res
    sj = os.path.join(calib, "session.json")
    if not os.path.isfile(sj):
        res["why"] = "This folder has no session.json; it is not a calibration set"
        return res
    try:
        with open(sj, encoding="utf-8") as f:
            ses = json.load(f)
        names = [s["name"] for s in ses["streams"]]
        res["res"] = tuple(ses["capture"]["resolution"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        res["why"] = f"session.json could not be read: {e}"
        return res
    res["cams"] = names
    if not names:
        res["why"] = "session.json lists no camera"
        return res
    counts = {n: len(glob.glob(os.path.join(calib, n, "shot_*.*")))
              for n in names}
    shots = counts[names[0]]
    res["shots"] = shots
    if shots == 0:
        res["why"] = f"No shot_* images under {names[0]}"
        return res
    if ses.get("mode") != "calib":
        res["warn"].append(f"session mode is {ses.get('mode')}, not calib")
    for n in names[1:]:
        if counts[n] != shots:
            res["warn"].append(f"{n} has only {counts[n]} images ({names[0]} has {shots})")
    if shots < MIN_SHOTS:
        res["warn"].append(f"Only {shots} sets; 15 or more recommended")
    res["has_rig"] = os.path.isdir(os.path.join(calib, "rig"))
    if res["has_rig"]:
        res["warn"].append("A rig folder exists and will be overwritten")
    res["ok"] = True
    return res


def edges_problem(edges):
    """Whether all six distances are filled in. edges = {"A-D": "153.0", ...} (strings or numbers).

    Returns None (fine) or a reason.
    """
    for key, _a, _b, _d in P.CALIB_EDGES:
        raw = edges.get(key)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return "All six distances are required"
        if not v > 0:
            return "Distances must be positive"
    return None


def build(calib, edges, values):
    """Build the command without running it, so it can be tested on its own."""
    cmd = [PY, "-u", CALIB, os.path.abspath(calib)]
    for key, a, b, _d in P.CALIB_EDGES:
        cmd += ["--edge", f"{a},{b},{float(edges[key])}"]
    for p in P.CALIB_PARAMS:
        v = values.get(p.key, p.default)
        if p.kind is bool:
            if v:
                cmd.append(p.cli)
        elif v != p.default:
            cmd += [p.cli, str(v)]
    cmd.append("--yes")
    return cmd


def read_report(calib):
    """<calibration set>/rig/calib_report.json. If unreadable, {"error": reason}."""
    path = os.path.join(calib, "rig", "calib_report.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        return {"error": f"Cannot read {path}: {e}"}


class CalibRunner:
    """One full calibration. The UI calls poll() every 300 ms. Same state machine as PrepRunner."""

    def __init__(self):
        self.proc = None
        self.state = "idle"
        self._asked_stop = False
        self.log_path = ""
        self._log = None
        self.calib = ""
        self.returncode = None
        self.started = 0.0

    def start(self, calib, edges, values):
        import time
        self.calib = os.path.abspath(calib)
        self.log_path = os.path.join(self.calib, LOG_NAME)
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            build(calib, edges, values), cwd=_ROOT, stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        # bound to the studio's lifetime; it starts COLMAP processes of its own
        adopt(self.proc.pid)
        self.started = time.time()
        self.state = "running"
        return self.proc

    def stop(self):
        """Calibration has no checkpoint; it can only be terminated.

        The whole process tree is ended, not only calibrate.py: terminate()
        kills only Python, and a running COLMAP would keep holding
        _work/database.db, so the next run would fail to delete _work.
        taskkill /T must run while the parent is alive to find the children.
        """
        if self.proc and self.proc.poll() is None:
            self._asked_stop = True
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                    capture_output=True, timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.SubprocessError):
                pass
            if self.proc.poll() is None:
                self.proc.terminate()               # fallback if taskkill failed
            self.state = "stopping"
            return
        self.state = "idle"

    def poll(self):
        if self.proc is None:
            return self.state
        rc = self.proc.poll()
        if rc is None:
            return self.state
        self.returncode = rc
        self.proc = None
        try:
            self._log.close()
        except Exception:                                     # noqa: BLE001
            pass
        # a non-zero exit code after a requested stop is not a failure (as in PrepRunner.poll)
        if self._asked_stop:
            self.state = "stopped"
        else:
            self.state = "done" if rc == 0 else "failed"
        self._asked_stop = False
        return self.state

    def log_tail(self, n=8):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                lines = [x.rstrip() for x in f if x.strip()]
        except OSError:
            return []
        return lines[-n:]


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(precheck(target), ensure_ascii=False, indent=2))
    defaults = {k: d for k, _a, _b, d in P.CALIB_EDGES}
    print(edges_problem(defaults), edges_problem({**defaults, "A-D": ""}))
    print(" ".join(build(target or ".", defaults,
                         {p.key: p.default for p in P.CALIB_PARAMS})))
