"""Start the training subprocess, write flag files, read the progress JSONL and preview JPG.

This layer does not use tkinter: pure functions plus one state machine, so it
can be tested from the command line. The UI (train_tab.py) only draws and polls.

State machine:

    idle --start--> running --pause--> paused --continue--> running
                      |                  |
                      +--stop------------+--> stopping --> idle
                      +--completed------------------> done

"Pause" keeps the training process waiting in place (see the STG_CHECKPOINT
patch); GPU memory is not released. Only stopping releases it.
"""
import json
import os
import re
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tools.train_cmd import PY, STG, TIERS, find_dataset, default_holdout  # noqa: E402
import params as P                                                        # noqa: E402
from joblimit import adopt                                                # noqa: E402
import orphans                                                            # noqa: E402

CTRL_DIR = "_studio"          # flags, progress and preview live in this folder under the model output
PROGRESS_NAME = "progress.jsonl"
PREVIEW_NAME = "preview.jpg"
VIDEO_NAME = "preview.mp4"
RENDER_FLAG = "render.flag"
SAVE_FLAG = "save.flag"
# The trainer's save / stop lines all read "... at iteration N ..." but do not
# always end with the number (the save while waiting adds " in 0.1s"), so a
# regex is used instead of taking the last word.
ITER_RE = re.compile(r"at iteration (\d+)")
STOP_FLAG = "stop.flag"
# cmd.txt holds the state (empty = run, pause = paused); each of the three
# actions has its own flag file. Sharing one file would let an action sent
# while paused overwrite "pause", and training would resume by itself after
# the action.
LOG_NAME = "train.log"
# The UI half of the state (parameters, progress, resume history). Stored in
# the root of the output folder, not in _studio/, because it describes the
# model rather than one run. The other half (Gaussians, optimizer state,
# counters) is _studio/state.pth, written by the training process.
META_NAME = "train_meta.json"
META_SCHEMA = "stg_studio/train_meta/1"


def meta_path(out):
    return os.path.join(out, META_NAME)


def read_meta(out):
    """Read train_meta.json. Missing or corrupt -> None; callers only ask "can this resume"."""
    try:
        with open(meta_path(out), encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) and m.get("schema") == META_SCHEMA else None


def write_meta(out, **fields):
    """Update a few fields in place (create the file if needed). Written then renamed, so it is never half-read."""
    m = read_meta(out) or {"schema": META_SCHEMA}
    m.update(fields)
    m["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = meta_path(out) + ".tmp"
    try:
        os.makedirs(out, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        os.replace(tmp, meta_path(out))
    except OSError:
        return None
    return m


def read_points(path):
    """Read the whole progress.jsonl (only once when resuming; afterwards reads are incremental)."""
    pts = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    pts.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        return []
    return pts


def resume_state(out):
    """Which mode an existing output folder puts the tab in.

    Returns ("resume" | "no_state" | "foreign", meta):
      resume    meta and state.pth exist     -> can resume
      no_state  meta exists, no checkpoint   -> cannot resume (a PLY has no optimizer state)
      foreign   neither exists               -> someone else's or an old folder; blocked
    """
    m = read_meta(out)
    if m is None:
        return "foreign", None
    if os.path.isfile(os.path.join(out, CTRL_DIR, "state.pth")):
        return "resume", m
    return "no_state", m


def parse_range(text, nsteps):
    """Parse the time range field into (a, b), both inclusive. Empty -> None.

    Errors raise ValueError with a message meant to be shown as-is; the UI
    and build() share this one check.
    """
    text = (text or "").strip().replace(" ", "")
    if not text:
        return None
    parts = [x for x in text.split(",") if x]
    if len(parts) != 2:
        raise ValueError("Write the time range as two numbers, start,end, e.g. 50,105")
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError("The time range accepts numbers only, e.g. 50,105") from None
    if a < 0 or b >= nsteps:
        raise ValueError(f"The time range must be within 0..{nsteps - 1}")
    if b < a:
        raise ValueError("The end of the time range cannot be before its start")
    return a, b


def range_holdout(a, b):
    """Five held-out time steps spread evenly inside the range.

    `default_holdout(nsteps)` would number them over the full length; with a
    range in the second half, all of them would fall outside it and nothing
    would be held out.
    """
    n = b - a + 1
    return ",".join(str(a + int(x))
                    for x in default_holdout(n).split(",") if x)


def write_model_window(out, ds, rng):
    """With a time range, write the model folder's own `window.json`.

    The player needs no change: it looks for `window.json` next to the PLY,
    then in the model folder, then in the shooting root, so this copy wins.

    Three fields change; the geometry fields are copied:

        nsteps       = number of time steps in the range
        duration_s   = length of this part
        audio.t0_offset_s += position of the range start within the original length

    Derived from the player's mapping `audio seconds = t x duration_s +
    t0_offset_s`: t inside the range is renormalized to 0..1 and both sides
    must agree. The audio file is not trimmed; only its start offset moves.
    """
    a, b = rng
    win = json.load(open(f"{ds}/window.json", encoding="utf-8"))
    times = json.load(open(f"{ds}/times_group.json", encoding="utf-8"))
    ts = [float(v) for k, v in times.items() if a <= int(k.split("/")[0]) <= b]
    if not ts:
        raise ValueError(f"times_group.json has no data for time steps {a}..{b}")
    lo, hi = min(ts), max(ts)
    dur0 = float(win.get("duration_s") or 0.0)
    win["nsteps"] = b - a + 1
    win["duration_s"] = round((hi - lo) * dur0, 6)
    if isinstance(win.get("audio"), dict):
        win["audio"]["t0_offset_s"] = round(
            float(win["audio"].get("t0_offset_s") or 0.0) + lo * dur0, 6)
    win["time_range"] = [a, b]          # for reference
    with open(os.path.join(out, "window.json"), "w", encoding="utf-8") as f:
        json.dump(win, f, ensure_ascii=False, indent=2)
    return win


def dataset_info(root):
    """Find the dataset under a shooting root and compute the layer-A (automatic) values.

    The formulas are those of tools/train_cmd.py; they are not redefined here.
    """
    ds = find_dataset(root).replace("\\", "/")
    win = json.load(open(f"{ds}/window.json", encoding="utf-8"))
    nsteps = win["nsteps"]
    return {
        "dataset": ds,
        "name": os.path.basename(ds),
        "duration": nsteps,
        "background": win.get("background", 0.0),
        "cameras_extent": round(win["near_far"][1] * 1.1, 4),
        "times": f"{ds}/times_group.json",
        "holdout": default_holdout(nsteps),
        "config": f"{ds}/cfg_lite.json",
        "source_path": f"{ds}/colmap_0",
    }


# PLY fields the player needs (load_ply in the player's render_at.py).
# Without the first group it cannot open the model; without the time group
# it falls back to a static model.
PLAYER_NEEDS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
PLAYER_TIME = ("trbf_center", "trbf_scale", "motion_0", "omega_0")

# Only this combination produces PLYs the player can read. ours_full uses
# 9-dimensional features and a small MLP for colour, while the player only
# has the ch3 rasterizer.
OK_MODEL, OK_RDPIP = "ours_lite", "train_ours_lite"


def compat_problem(info):
    """Before starting, check whether this setup would produce a model the player cannot open.

    Returns a message or None. It checks the dataset's config (--config
    points to it), not the UI parameters: no exposed parameter changes the
    PLY's field list (fixed by construct_list_of_attributes in ourslite.py),
    and the three that would are deliberately not exposed.
    """
    try:
        with open(info["config"], encoding="utf-8") as f:
            cfg = json.load(f)
    except OSError:
        return f"Cannot read {os.path.basename(info['config'])}"
    except ValueError as e:
        return f"{os.path.basename(info['config'])} is not valid JSON: {e}"

    bad = []
    if cfg.get("model", OK_MODEL) != OK_MODEL:
        bad.append(f"model={cfg['model']} (needs {OK_MODEL})")
    if cfg.get("rdpip", OK_RDPIP) != OK_RDPIP:
        bad.append(f"rdpip={cfg['rdpip']} (needs {OK_RDPIP})")
    if bad:
        return ("This dataset's config would train a model the player cannot open: "
                + ", ".join(bad))
    return None


def ply_problem(path):
    """Check whether the player can read a trained PLY. Returns a message or None.

    For use after training (or on older models). Before starting, use compat_problem.
    """
    try:
        from plyfile import PlyData
        names = {p.name for p in PlyData.read(path)["vertex"].properties}
    except Exception as e:                                    # noqa: BLE001
        return f"Cannot read {path}: {e}"
    miss = [k for k in PLAYER_NEEDS if k not in names]
    if miss:
        return "The PLY lacks fields the player needs: " + ", ".join(miss)
    if any(n.startswith("f_rest_") for n in names):
        return "The PLY has spherical-harmonic coefficients (f_rest_*): a 3DGS/full model the player's ch3 cannot open"
    miss_t = [k for k in PLAYER_TIME if k not in names]
    if miss_t:
        return "The PLY has no time fields (" + ", ".join(miss_t) + "); it would play as a static model"
    return None


def tier_values(tier):
    """The tier's row (layer B) plus the advanced defaults (layer C)."""
    v = {p.key: p.value_for(tier) for p in P.TIERED}
    v.update({p.key: p.default for p in P.ADVANCED})
    return v


def save_iterations(iterations, pct):
    """Convert "save every N%" into the iteration list for --save_iterations.

    Upstream already accepts this flag, so no patch is needed.
    pct <= 0 or >= 100 saves once at the end; the last iteration is always included.
    """
    it = max(int(iterations), 1)
    if not pct or pct >= 100 or pct < 1:
        return [it]
    pts = {int(it * p / 100) for p in range(int(pct), 100, int(pct))}
    return sorted(p for p in pts | {it} if p >= 1)


def build(root, tier, values, name=None):
    """Return (cmd, env, out_dir) without running anything, so it can be tested on its own."""
    info = dataset_info(root)
    name = name or tier
    out = os.path.join(os.path.dirname(info["dataset"]), f"{name}_out")
    out = out.replace("\\", "/")
    ctrl = f"{out}/{CTRL_DIR}"

    env = dict(os.environ)
    env.update({
        "STG_TIMES": info["times"],
        "STG_CAMERAS_EXTENT": str(info["cameras_extent"]),
        "STG_BACKGROUND": str(info["background"]),
        "STG_ZNEAR_PRUNE": str(values["znear_prune"]),
        "STG_GNUM_MAX": str(values["gnum_max"]),
        "STG_OMEGA_ITER": str(values["omega_iter"]),
        "STG_HOLDOUT_T": values.get("holdout_t") or info["holdout"],
        # interfaces added by the trainer patch (upstream/stg_427abfc.patch)
        "STG_CHECKPOINT": ctrl,
        # How often training checks for commands = how long a pause takes to
        # take effect. 200 (about 6 s) feels unresponsive; 50 is about 1.5 s
        # at the cost of a few more file checks (not measurable at 32 it/s).
        "STG_CHECKPOINT_EVERY": "50",
        "STG_PROGRESS": f"{ctrl}/{PROGRESS_NAME}",
        "STG_PREVIEW": f"{ctrl}/{PREVIEW_NAME}",
        # Keep the process after training finishes, so the user can decide
        # whether to save a checkpoint while the model is still in GPU memory.
        # A finished run does not write state.pth by itself and could not be resumed.
        "STG_HOLD": "1",
    })
    # Iterations for automatic checkpoints. Separate from --save_iterations,
    # which writes playback PLYs (no optimizer state); this writes the
    # resumable state.pth. 0 = off.
    _ck_pct = int(values.get("ckpt_pct") or 0)
    if _ck_pct > 0:
        env["STG_CKPT_ITERS"] = ",".join(
            str(i) for i in save_iterations(values["iterations"], _ck_pct))
    else:
        env.pop("STG_CKPT_ITERS", None)
    # Time window: train only time steps a..b.
    rng = parse_range(values.get("time_range"), info["duration"])
    duration = info["duration"]
    if rng:
        a, b = rng
        duration = b - a + 1
        env["STG_TIME_RANGE"] = f"{a},{b}"
        # held-out steps must follow the range, or none of them falls inside it
        env["STG_HOLDOUT_T"] = values.get("holdout_t") or range_holdout(a, b)
    else:
        env.pop("STG_TIME_RANGE", None)
    if values.get("bg_random"):
        env["STG_BG_RANDOM"] = "1"
    else:
        env.pop("STG_BG_RANDOM", None)

    it = values["iterations"]
    cmd = [PY, "-u", "train.py", "--quiet", "--eval",
           "--source_path", info["source_path"],
           "--model_path", out,
           "--config", info["config"],
           # --duration is the number of time steps, not seconds; with a
           # range it must change too, or STG's time axis would not match the
           # loaded cameras.
           "--duration", str(duration),
           "--densify", str(values["densify"]),
           "--desicnt", str(values["desicnt"]),
           "--densify_until_iter", str(values["densify_until_iter"]),
           "--densify_from_iter", str(values["densify_from_iter"]),
           "--densify_grad_threshold", str(values["densify_grad_threshold"]),
           "--densification_interval", str(values["densification_interval"]),
           "--opacity_reset_interval", str(values["opacity_reset_interval"]),
           "--opthr", str(values["opthr"]),
           "--percent_dense", str(values["percent_dense"]),
           "--lambda_dssim", str(values["lambda_dssim"]),
           "--emsstart", str(values["emsstart"]),
           "--emsthr", str(values["emsthr"]),
           "--farray", str(values["farray"]),
           "--gtisint8", str(values["gtisint8"]),
           "--resolution", str(values["resolution"]),
           "--iterations", str(it),
           # must follow iterations, or the learning rate bottoms out early (false convergence)
           "--position_lr_max_steps", str(P.position_lr_max_steps(values) or it),
           "--save_iterations"] + [
        str(s) for s in save_iterations(it, values.get("autosave_pct", 100))]
    return cmd, env, out


class Runner:
    """One training run. The UI calls poll() every 200 ms."""

    def __init__(self):
        self.proc = None
        self.state = "idle"
        self.out = ""
        self.ctrl = ""
        self.log_path = ""
        self._log = None
        self._read = 0            # read position in progress.jsonl
        self._video_mtime = None  # not None = render requested, still waiting
        self.points = []          # [{iter, loss, n_gauss, max_scale, elapsed_s}]
        self.returncode = None
        self._asked_stop = False
        self._log_from = 0        # where this run starts in train.log (resumes append)

    # ---------------------------------------------------------- start
    def start(self, root, tier, values, name=None, resume=False):
        """`resume=True` continues from `_studio/state.pth`, keeping progress and the curve."""
        cmd, env, out = build(root, tier, values, name)
        self.out = out
        self.ctrl = os.path.join(out, CTRL_DIR)
        os.makedirs(self.ctrl, exist_ok=True)
        # With a time range, the model folder gets its own window.json. It is
        # written at start, not at the end, so a stopped model is playable too.
        _info = dataset_info(root)
        _rng = parse_range(values.get("time_range"), _info["duration"])
        if _rng:
            write_model_window(out, _info["dataset"], _rng)
        state_pth = os.path.join(self.ctrl, "state.pth")
        if resume and not os.path.isfile(state_pth):
            raise FileNotFoundError(f"Resume requested but there is no checkpoint: {state_pth}")
        # When resuming, progress.jsonl is kept (the curve continues); other control files are cleared.
        stale = [RENDER_FLAG, SAVE_FLAG, STOP_FLAG, "cmd.txt"]
        if not resume:
            stale += [PROGRESS_NAME, PREVIEW_NAME, VIDEO_NAME]
        for f in stale:
            try:
                os.remove(os.path.join(self.ctrl, f))
            except OSError:
                pass
        self._asked_stop = False
        if resume:
            env["STG_RESUME"] = state_pth
            self.points = read_points(os.path.join(self.ctrl, PROGRESS_NAME))
            self._read = os.path.getsize(os.path.join(self.ctrl, PROGRESS_NAME)) \
                if os.path.isfile(os.path.join(self.ctrl, PROGRESS_NAME)) else 0
            self._video_mtime = None
        else:
            env.pop("STG_RESUME", None)
            self.points, self._read, self._video_mtime = [], 0, None

        meta = read_meta(out) if resume else None
        resumes = list((meta or {}).get("resumes", []))
        if resume:
            # The resume point is where the previous run really stopped (meta
            # last_iter, from the log), not the last curve point: the curve
            # has one point every 200 iterations, which would misplace the line.
            mark = int((meta or {}).get("last_iter") or 0) or (
                self.points[-1]["iter"] if self.points else 0)
            if mark:
                resumes.append(mark)
        try:
            ds_name = dataset_info(root)["name"]      # the dataset name, not the shooting root's
        except Exception:                                     # noqa: BLE001
            ds_name = ""
        write_meta(out,
                   dataset=ds_name,
                   tier=tier,
                   params=dict(values),
                   total_iterations=int(values["iterations"]),
                   # Do not touch last_iter when resuming: the stored value comes
                   # from the log (exact); the last curve point is coarser.
                   last_iter=int((meta or {}).get("last_iter") or 0),
                   finished=False,
                   resumes=resumes,
                   started_at=(meta or {}).get(
                       "started_at", time.strftime("%Y-%m-%d %H:%M:%S")))

        self.log_path = os.path.join(self.ctrl, LOG_NAME)
        # When resuming, the log is appended to so earlier messages remain
        # available. Its earlier content then includes the previous run's
        # messages, so the start position of this run is recorded and only
        # read from there on; otherwise "the previous run finished" would be
        # taken for "this run finished".
        try:
            self._log_from = os.path.getsize(self.log_path) if resume else 0
        except OSError:
            self._log_from = 0
        self._log = open(self.log_path, "a" if resume else "w",
                         encoding="utf-8", errors="replace")
        # CREATE_NO_WINDOW: no console window. On Windows a python.exe child
        # gets its own console, and this is a GUI; output already goes to
        # train.log.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, cwd=STG, env=env, stdout=self._log,
                                     stderr=subprocess.STDOUT,
                                     creationflags=flags)
        # Bind to the studio's lifetime: if the studio crashes or is closed,
        # the OS also ends the training (and its own children). While the
        # studio is alive this never acts, so a running training is never
        # killed by mistake.
        adopt(self.proc.pid)
        # Record the PID and creation time so a later studio session can
        # recognize a run left over from this one (joblimit cannot cover
        # processes that are already outside).
        orphans.write_pid(self.ctrl, self.proc.pid)
        self.state = "running"
        return cmd

    # ---------------------------------------------------------- control
    def _write_state(self, text):
        """cmd.txt holds the state only (empty or pause). Actions use flag files."""
        with open(os.path.join(self.ctrl, "cmd.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    def _touch(self, name):
        with open(os.path.join(self.ctrl, name), "w") as f:
            f.write("")

    def pause(self):
        self._write_state("pause")
        self.state = "paused"

    def resume(self):
        self._write_state("")
        self.state = "running"

    def save(self):
        """Save a resumable checkpoint. Allowed while paused; it stays paused afterwards."""
        self._touch(SAVE_FLAG)

    def stop(self):
        self._touch(STOP_FLAG)
        self._asked_stop = True       # distinguishes "stopped by the user" from "finished"
        self.state = "stopping"

    @property
    def asked_stop(self):
        """Whether this run was asked to stop (set by stop(), cleared by start()).

        After the process exits, use this, not `state`, to tell whether the
        user pressed stop: poll() has already set the state to done by then.
        """
        return self._asked_stop

    def render_video(self):
        """Ask the trainer to render the full window-center video; it returns to its state (running or paused) afterwards.

        Uses its own flag file rather than cmd.txt: cmd.txt holds a lasting
        state (pause), render is a one-off action, and sharing the file would
        let a render while paused clear the pause.
        """
        self._video_mtime = self._mtime(VIDEO_NAME)
        with open(os.path.join(self.ctrl, RENDER_FLAG), "w") as f:
            f.write("")

    def _mtime(self, name):
        try:
            return os.path.getmtime(os.path.join(self.ctrl, name))
        except OSError:
            return 0

    @property
    def rendering(self):
        """The flag still exists = the trainer has not finished rendering (it deletes the flag when done)."""
        return bool(self.ctrl) and os.path.isfile(
            os.path.join(self.ctrl, RENDER_FLAG))

    @property
    def video(self):
        p = os.path.join(self.ctrl, VIDEO_NAME) if self.ctrl else ""
        return p if p and os.path.isfile(p) else None

    @property
    def checkpoint(self):
        p = os.path.join(self.ctrl, "state.pth")
        return p if os.path.isfile(p) else None

    @property
    def saving(self):
        """The flag still exists = the trainer has not finished saving (it deletes the flag when done)."""
        return bool(self.ctrl) and os.path.isfile(
            os.path.join(self.ctrl, SAVE_FLAG))

    def last_saved(self):
        """(iteration, time) of the latest save, or (None, None).

        The iteration comes from `[checkpoint] saved at iteration N` in the
        log, which is far cheaper than loading the checkpoint file just to
        show one line.
        """
        p = self.checkpoint
        if not p:
            return None, None
        it = None
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "] saved at iteration " in line:
                        m = ITER_RE.search(line)
                        if m:
                            it = int(m.group(1))
        except OSError:
            pass
        return it, os.path.getmtime(p)

    def unsaved_progress(self, cur_iter):
        """Whether the current progress lies beyond the latest checkpoint.

        Stopping does not save, so this part is what stopping would discard.
        """
        it, _ = self.last_saved()
        if it is None:
            return True
        return bool(cur_iter and cur_iter > it)

    @property
    def preview(self):
        p = os.path.join(self.ctrl, PREVIEW_NAME)
        return p if os.path.isfile(p) else None

    # ---------------------------------------------------------- polling
    def poll(self):
        """Return (state, new points). When the process exits, settle the state to done/failed."""
        new = self._read_progress()
        if self.proc is None:
            return self.state, new
        rc = self.proc.poll()
        if rc is None:
            # Finished, but the process still holds GPU memory (STG_HOLD).
            # This is a state of its own, which the button rules understand.
            if self.state == "running" and self.holding:
                self.state = "finished"
            return self.state, new
        self.returncode = rc
        self.proc = None
        try:
            self._log.close()
        except Exception:                                     # noqa: BLE001
            pass
        self.state = "done" if rc == 0 else "failed"
        orphans.clear_pid(self.ctrl)
        self._update_meta()
        return self.state, new

    @property
    def holding(self):
        """Finished, process still alive (holding GPU memory), waiting for commands.

        "The process is alive" cannot tell running from finished-and-waiting;
        the trainer's own log line decides.
        """
        return self._log_has("[checkpoint] finished")

    def _log_has(self, mark):
        """Whether this run's part of the log contains the line (earlier resumed runs are ignored)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(getattr(self, "_log_from", 0))
                return mark.encode() in f.read()
        except OSError:
            return False

    def _update_meta(self):
        """At the end, store the UI half of the state (decides later whether the model can resume)."""
        if not self.out:
            return
        # The last line of progress.jsonl is not "where it ended": it is
        # written every STG_PROGRESS_EVERY iterations, so the final stretch is
        # missing and a finished run would look unfinished. The real end point
        # is in the log, printed by the trainer.
        last = max([self.log_iteration() or 0,
                    self.points[-1]["iter"] if self.points else 0])
        # Finished means "not stopped by the user", not the exit code: a
        # requested stop also exits with 0.
        finished = (self.returncode == 0 and not self._asked_stop)
        write_meta(self.out, last_iter=last, finished=finished)

    def log_iteration(self):
        """The last iteration the trainer printed in train.log (when saving or stopping).

        Only this run's part of the log is read (see `_log_has`).
        """
        it = None
        try:
            with open(self.log_path, "rb") as fb:
                fb.seek(getattr(self, "_log_from", 0))
                text = fb.read().decode("utf-8", "replace")
            for line in text.splitlines():
                for mark in (" saved at iteration ",
                             " stopped by request at iteration "):
                    if mark in line:
                        m = ITER_RE.search(line)
                        if m:
                            it = int(m.group(1))
        except OSError:
            return None
        return it

    def _read_progress(self):
        """Read only the new part; re-reading the whole file would grow with every point."""
        path = os.path.join(self.ctrl, PROGRESS_NAME) if self.ctrl else ""
        if not path or not os.path.isfile(path):
            return []
        new = []
        try:
            # read() instead of `for line in f`: f.tell() inside that loop
            # raises OSError ("telling position disabled by next() call"),
            # which the except below would swallow, looking like an empty
            # file. Binary mode, because text-mode tell() returns an opaque
            # cookie that cannot be used for arithmetic.
            with open(path, "rb") as f:
                f.seek(self._read)
                raw = f.read()
        except OSError:
            return []
        cut = raw.rfind(b"\n") + 1            # keep an unfinished last line for next time
        if not cut:
            return []
        self._read += cut
        chunk = raw[:cut].decode("utf-8", "replace")
        for line in chunk.splitlines():
            try:
                new.append(json.loads(line))
            except ValueError:
                pass
        self.points.extend(new)
        return new

    def log_tail(self, n=200):
        try:
            lines = open(self.log_path, encoding="utf-8",
                         errors="replace").read().splitlines()
        except OSError:
            return "(no messages)"
        return "\n".join([x for x in lines if x.strip()][-n:]) or "(no messages)"
