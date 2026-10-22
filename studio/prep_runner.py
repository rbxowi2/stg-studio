"""Start the prep_4d.py subprocess and parse its output.

No tkinter here: pure functions plus one state machine, testable from the
command line (like runner.py).

The tab's behaviour = `python prep_4d.py <shooting root>`, with no flags that
change defaults. In particular --rig is never passed: it would copy an
outside calibration into the root and delete the existing one; without it
<root>/rig is used (resolve_rig in prep_4d.py).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from joblimit import adopt                                    # noqa: E402
from tools.train_cmd import find_dataset, is_dataset          # noqa: E402,F401

PREP = os.path.join(_ROOT, "pipeline", "prep_4d.py")
PY = sys.executable          # the studio's own interpreter runs preprocessing as well
LOG_NAME = "prep.log"

# Two lines printed by prep_4d.py (matched by content, not line number):
#   "  Valid groups 300/301 (99.7%)  sampled 300 time steps"
RE_RATE = re.compile(r"Valid groups\s+(\d+)/(\d+)\s+\(([\d.]+)%\).*?sampled\s+(\d+)\s+time steps")
#   "  Dataset -> stg301 (named after the number of valid time steps)"
#   Not \S+: it would also swallow the parenthesised note, so a dataset with
#   the same name would be taken for a different one.
RE_DS = re.compile(r"Dataset\s*->\s*([^\s(]+)")


def is_stage1_only(path):
    """Whether a folder is an empty shell left by stage 1 only.

    Stage 1 already creates the dataset folder and writes window.json and
    phase_report.md, but no colmap_* folders. Every pre-check with different
    parameters would leave such a shell, which find_dataset() would count as
    a real dataset.
    """
    try:
        return (os.path.isfile(os.path.join(path, "window.json"))
                and not any(d.startswith("colmap_") for d in os.listdir(path)))
    except OSError:
        return False


def datasets(root, complete_only=True):
    """Datasets under the root.

    complete_only=True (default) uses `train_cmd.is_dataset()`, the same test
    as training: `window.json` + `colmap_0` + `times_group.json`.
    `window.json` alone is not enough, because a model folder trained with a
    time range has one too (see runner.write_model_window).

    complete_only=False accepts any folder with window.json, to find stage-1
    shells (see is_stage1_only).
    """
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    if complete_only:
        return [d for d in names if is_dataset(os.path.join(root, d))]
    return [d for d in names
            if os.path.isfile(os.path.join(root, d, "window.json"))]


def find_recording(root):
    """The only folder under the shooting root that contains session.json = the recording.

    Same rule as prep_4d.find_recording() (which raises SystemExit; this returns None).
    """
    try:
        hits = [d for d in sorted(os.listdir(root))
                if os.path.isfile(os.path.join(root, d, "session.json"))]
    except OSError:
        return None
    return os.path.join(root, hits[0]) if len(hits) == 1 else None


def build(root, values, stage1=False, out=None):
    """Build the command without running it, so it can be tested on its own.

    With `out`, prep_4d.py's older form is used (recording + output folder as
    two positional arguments), so everything is written into `out`; the
    pre-check uses this to leave the user's dataset untouched.
    """
    if out is not None:
        rec = find_recording(root)
        if rec is None:
            return None
        cmd = [PY, "-u", PREP, rec, os.path.abspath(out)]
    else:
        cmd = [PY, "-u", PREP, os.path.abspath(root)]
    steps = int(values.get("prep_steps") or 0)
    if steps:
        cmd += ["--steps", str(steps)]
    cmd += ["--span-ms", str(values.get("prep_span_ms", 25.0))]
    cmd += ["--background", str(values.get("prep_background", 0.0))]
    anchor = (values.get("prep_anchor") or "").strip()
    if anchor:
        cmd += ["--anchor", anchor]
    if values.get("prep_keep_work"):
        cmd.append("--keep-work")
    if stage1:
        cmd.append("--stage1")
    return cmd


def parse(text):
    """Extract the valid rate, number of time steps and dataset name. Missing values stay missing (never 0)."""
    out = {}
    m = RE_RATE.search(text)
    if m:
        out["ok_groups"] = int(m.group(1))
        out["all_groups"] = int(m.group(2))
        out["rate"] = float(m.group(3))
        out["nsteps"] = int(m.group(4))
    m = RE_DS.search(text)
    if m:
        out["dataset"] = m.group(1)
    return out


def precheck(root, values, timeout=180):
    """Run --stage1 and parse it. Returns a dict the UI displays as-is.

    Read-only: everything is written to a temporary folder that is deleted
    afterwards; the user's folders are not touched.
    """
    res = {"ok": False, "why": "", "raw": ""}

    # prep_4d.py --stage1 would create the dataset folder and rewrite
    # window.json, wiping the geometry fields of an already finished dataset
    # (near_far / wh / fov / ctr / win / unit_mm). The older form
    # "recording + output folder" is used instead, with a temporary output.
    tmp = tempfile.mkdtemp(prefix="stg_precheck_")
    cmd = build(root, values, stage1=True, out=os.path.join(tmp, "probe"))
    if cmd is None:
        shutil.rmtree(tmp, ignore_errors=True)
        res["why"] = "No recording found (the root must contain exactly one folder with session.json)"
        return res

    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        res["why"] = f"The pre-check did not finish within {timeout} s"
        return res
    except OSError as e:
        res["why"] = f"Could not run prep_4d.py: {e}"
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)      # temporary output deleted after reading

    res["raw"] = (p.stdout or "") + (p.stderr or "")
    res.update(parse(res["raw"]))
    if p.returncode != 0:
        tail = [x for x in res["raw"].splitlines() if x.strip()][-3:]
        res["why"] = "Pre-check failed: " + " / ".join(tail)
        return res
    if "nsteps" not in res:
        # Reached when prep_4d.py's output format changes. Say so instead of
        # pretending the valid rate is 0.
        res["why"] = "Could not read the valid rate (prep_4d.py's output format may have changed)"
        return res
    # With an explicit output folder prep does not print the "Dataset -> stgN"
    # line. The naming rule comes from prep_4d.py: out = root / f"stg{len(sel)}".
    res["dataset"] = f"stg{res['nsteps']}"

    res["existing"] = datasets(root)
    other = [d for d in res["existing"] if d != res["dataset"]]
    if other:
        # Running would leave two datasets side by side, and find_dataset()
        # refuses two, so the training tab would stop working.
        res["why"] = (f"This run would create {res['dataset']}, but the root already has "
                      + ", ".join(other) + ". With two datasets the training tab cannot tell "
                      "which one to use; delete the unwanted one first.")
        return res
    res["ok"] = True
    return res


class PrepRunner:
    """One full preprocessing run. The UI calls poll() every 200 ms."""

    def __init__(self):
        self.proc = None
        self.state = "idle"
        self._asked_stop = False    # stopped on request: a non-zero exit code is expected
        self.log_path = ""
        self._log = None
        self.out = ""            # dataset folder (from the pre-check)

    def start(self, root, values, dataset):
        self.out = os.path.join(os.path.abspath(root), dataset)
        os.makedirs(self.out, exist_ok=True)
        self.log_path = os.path.join(self.out, LOG_NAME)
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            build(root, values), cwd=_ROOT, stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        # Bound to the studio's lifetime, as in runner.start. Even more needed
        # here: preprocessing starts COLMAP, so an orphan would be a whole tree.
        adopt(self.proc.pid)
        self.state = "running"
        return self.proc

    def stop(self):
        """Preprocessing has no checkpoint; it can only be terminated. Outputs are kept and overwritten next time."""
        if self.proc and self.proc.poll() is None:
            self._asked_stop = True
            self.proc.terminate()
            self.state = "stopping"
            return
        # With no process to stop, return to idle directly; otherwise poll()
        # would report "stopping" forever and keep the UI locked.
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
        # terminate() gives a non-zero exit code when the user stopped the
        # run; that is not a failure and must not raise a failure dialog.
        if self._asked_stop:
            self.state = "stopped"
        else:
            self.state = "done" if rc == 0 else "failed"
        self._asked_stop = False
        return self.state

    def log_tail(self, n=6):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                lines = [x.rstrip() for x in f if x.strip()]
        except OSError:
            return []
        return lines[-n:]

    def result(self):
        """Acceptance numbers after completion, all from the dataset's own window.json."""
        out = {}
        try:
            with open(os.path.join(self.out, "window.json"), encoding="utf-8") as f:
                out["window"] = json.load(f)
        except (OSError, ValueError) as e:
            out["error"] = f"Cannot read window.json: {e}"
        try:
            with open(os.path.join(self.out, "phase_report.md"),
                      encoding="utf-8") as f:
                out["phase"] = f.read()
        except OSError:
            out["phase"] = ""
        return out
