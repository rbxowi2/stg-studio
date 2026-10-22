"""Combined training tab: split a time range into segments, train each, then merge them into one model.

A subclass of the training tab rather than a copy, so parameters, curve,
preview, resume and stop behave the same and cannot drift apart. It adds
three things:

    1. a "time steps per segment" field and a segment plan preview
    2. running every segment in order (`_on_run_done()` starts the next one)
    3. calling `tools/merge_segments.py` once all segments are done

Each segment is an ordinary training run (output `<name>_s01_out`,
`_s02_out`, ...), so checkpoints, resume, video rendering and the process
watchdog all work unchanged.
"""
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

import params as P
import runner as R
from joblimit import adopt
from tooltip import Tooltip
from train_tab import GREEN, GREY, RED, TrainTab

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tools.merge_segments import split_segments                  # noqa: E402

MERGE_TOOL = os.path.join(_ROOT, "tools", "merge_segments.py")


class ComboTab(TrainTab):
    def __init__(self, master, get_root, show_error, set_busy=None):
        self._plan = []              # [(first step, last step)]
        self._plan_i = 0             # index of the segment being trained
        self._running_plan = False   # whether this run was started by combined training
        self._merge = None           # merge subprocess
        self._auto_release = False   # this stop was sent here (finished segment, next one follows)
        super().__init__(master, get_root, show_error, set_busy)

        # The field sits in the parameter table, right below the time range:
        # it is one of this tab's main knobs.
        box = self.widgets["time_range"].master
        row = box.grid_size()[1]
        ttk.Label(box, text="Steps per segment").grid(row=row, column=0, sticky="w",
                                          pady=2)
        self.seg_steps = tk.StringVar(value="30")
        self.seg_ent = ttk.Entry(box, textvariable=self.seg_steps, width=14)
        self.seg_ent.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        tip = ("Number of time steps per segment (unrelated to fps).\n"
               "Range 10,90 with 30 gives 10..40, 40..70, 70..90;\n"
               "neighbouring segments share their boundary step, where they cross-fade.\n"
               "A tail shorter than a third of a segment is merged into the previous one.")
        Tooltip(self.seg_ent, tip)
        # Only the number of segments is shown; listing every interval would
        # break the layout when there are many. The full list is in the tooltip.
        self.plan_lab = ttk.Label(box, text="", foreground=GREY)
        self.plan_lab.grid(row=row, column=2, sticky="w", padx=(4, 0))
        self.plan_tip = Tooltip(self.plan_lab, "")

        # "Merge again" sits with the other buttons
        self.b_merge = ttk.Button(self.b_start.master, text="Merge again",
                                  command=self._merge_now)
        self.b_merge.pack(side="left", padx=(0, 8))
        Tooltip(self.b_merge,
                "For when every segment is trained and only the merge needs redoing. Does not train again.")
        self.seg_steps.trace_add("write", lambda *_a: self._show_out())
        self._show_out()

    # ------------------------------------------------------------ segments
    def _per(self):
        try:
            return int(self.seg_steps.get())
        except (TypeError, ValueError):
            return 0

    def plan(self):
        """The segments to train. An empty list on invalid input (the reason is shown on screen)."""
        if self.info is None:
            return []
        try:
            rng = R.parse_range(self.vars["time_range"].get(),
                                int(self.info.get("duration") or 0))
        except ValueError:
            return []
        per = self._per()
        if per < 1:
            return []
        a, b = rng if rng else (0, int(self.info.get("duration") or 1) - 1)
        return split_segments(a, b, per)

    def seg_name(self, i):
        return f"{self.name.get().strip()}_s{i + 1:02d}"

    def seg_done(self, i):
        """Whether a segment is finished (it has a playable model)."""
        out = self._out_dir(self.seg_name(i))
        pc = os.path.join(out, "point_cloud")
        try:
            return any(d.startswith("iteration_") for d in os.listdir(pc))
        except OSError:
            return False

    # ------------------------------------------------------------ display
    def _show_out(self):
        """Override: combined training concerns a whole set of segments, not one output folder."""
        if not hasattr(self, "plan_lab"):     # the parent calls this once during construction
            return
        self.resume_meta = None
        name = self.name.get().strip()
        segs = self.plan()
        if not name:
            self.out_lab.configure(text="Enter an output name", foreground=RED)
            self.plan_lab.configure(text="")
        elif not segs:
            self.out_lab.configure(text="Invalid time range or steps per segment",
                                   foreground=RED)
            self.plan_lab.configure(text="")
        else:
            done = [i for i in range(len(segs)) if self.seg_done(i)]
            shortest = min(b - a + 1 for a, b in segs)
            tail = f", {len(done)} done" if done else ""
            self.plan_lab.configure(
                text=f"{len(segs)} segments{tail}",
                foreground=RED if shortest < 30 else GREY)
            # the interval list goes in the tooltip; the screen shows only the count
            lines = [f"segment {i + 1}  steps {a}..{b} ({b - a + 1} steps)"
                     f"{'  done' if i in done else ''}"
                     for i, (a, b) in enumerate(segs)]
            if len(lines) > 12:          # with many segments the tooltip would leave the screen
                lines = lines[:12] + [f"... ({len(lines) - 12} more)"]
            if shortest < 30:
                lines.append("The shortest segment has fewer than 30 steps: motion is poorly constrained and may distort.")
            self.plan_tip.text = chr(10).join(lines)
            merged = self._out_dir(name)
            self.out_lab.configure(
                text=f"Merged model -> {merged}",
                foreground=GREEN if len(done) == len(segs) else GREY)
        self._sync_buttons()

    def _name_ok(self):
        """Combined training does not check one folder for a name clash; it needs a name and a valid plan."""
        return bool(self.name.get().strip()) and bool(self.plan())

    def _sync_buttons(self):
        super()._sync_buttons()
        if not hasattr(self, "b_merge"):
            return
        idle = self.run.state == "idle" and not self._merge
        self.seg_ent.state(["!disabled"] if idle else ["disabled"])
        all_done = bool(self.plan()) and all(
            self.seg_done(i) for i in range(len(self.plan())))
        self.b_merge.state(["!disabled"] if (idle and all_done) else ["disabled"])
        self.b_start.configure(
            text="Continue combined training" if any(
                self.seg_done(i) for i in range(len(self.plan() or [])))
            else "Start combined training")

    def _show_status(self):
        if self._merge:
            self.msg.configure(text="Merging...  joining the segments into one model",
                               foreground=RED)
            return
        if self._running_plan and self.run.state == "finished":
            # Not the training tab's "save before resuming" message: combined
            # training needs no decision here, the next segment starts by itself.
            self.msg.configure(
                text=f"Segment {self._plan_i + 1}/{len(self._plan)} done, "
                     "freeing GPU memory...", foreground=GREY)
            return
        super()._show_status()
        if self._running_plan and self.run.state in ("running", "paused"):
            self.msg.configure(
                text=f"Segment {self._plan_i + 1}/{len(self._plan)} "
                     f"(steps {self._plan[self._plan_i][0]}.."
                     f"{self._plan[self._plan_i][1]})  " + self.msg["text"])

    # ------------------------------------------------------------ running
    def _start(self):
        """Override: run the whole set of segments, skipping finished ones."""
        if self.info is None or self.run.state != "idle" or self._merge:
            return
        self._plan = self.plan()
        if not self._plan or not self.name.get().strip():
            return
        self._plan_i = 0
        self._running_plan = True
        self._next_segment(first=True)

    def _next_segment(self, first=False):
        """Start the next unfinished segment; merge once all are done."""
        while self._plan_i < len(self._plan) and self.seg_done(self._plan_i):
            self._plan_i += 1
        if self._plan_i >= len(self._plan):
            self._running_plan = False
            self._merge_now()
            return
        a, b = self._plan[self._plan_i]
        vals = self.values()
        vals["time_range"] = f"{a},{b}"
        name = self.seg_name(self._plan_i)
        out = self._out_dir(name)
        kind, _meta = R.resume_state(out)
        try:
            self.run.start(self.get_root(), self.tier.get(), vals, name=name,
                           resume=(kind == "resume"))
        except Exception as e:                                # noqa: BLE001
            self._running_plan = False
            self.show_error("Could not start this segment", str(e))
            self._sync_buttons()
            return
        self.curve.set_points([], marks=[])       # one curve per segment
        self.msg.configure(
            text=f"Segment {self._plan_i + 1}/{len(self._plan)} (steps {a}..{b})"
                 f"  output {out}", foreground=GREY)
        self._sync_buttons()

    def _on_run_done(self, stopped):
        """A segment ended. If the user pressed stop, everything stops here; otherwise the next segment starts."""
        if not self._running_plan:
            return
        if self._auto_release:
            stopped = False           # that stop was sent here, not by the user
            self._auto_release = False
        if stopped:
            # A user stop ends the whole combined run. Finished segments are
            # kept; "Continue combined training" skips them next time.
            self._running_plan = False
            self.msg.configure(
                text=f"Stopped at segment {self._plan_i + 1}/{len(self._plan)}. "
                     "Finished segments are kept; \"Continue combined training\" picks up from here.",
                foreground=RED)
            self._sync_buttons()
            return
        self._plan_i += 1
        self.after(500, self._next_segment)       # let the previous process release GPU memory first

    # ------------------------------------------------------------ merge
    def _merge_now(self):
        """Merge the segments into one model, in a subprocess so the UI does not freeze."""
        if self._merge:
            return
        segs = self.plan()
        name = self.name.get().strip()
        if not segs or not name or self.info is None:
            return
        args = []
        for i, (a, b) in enumerate(segs):
            ply = self._latest_ply(self._out_dir(self.seg_name(i)))
            if not ply:
                self.show_error("Cannot merge yet",
                                f"Segment {i + 1} (steps {a}..{b}) has no playable model yet.")
                return
            args.append(f"{ply}:{a}:{b}")
        cmd = [sys.executable, "-u", MERGE_TOOL,
               "--dataset", self.info["dataset"],
               "--out", self._out_dir(name), *args]
        log = os.path.join(self._out_dir(name) + "_merge.log")
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        self._merge_log = open(log, "w", encoding="utf-8", errors="replace")
        self._merge = subprocess.Popen(
            cmd, stdout=self._merge_log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        adopt(self._merge.pid)                    # must not outlive the studio
        self._sync_buttons()

    @staticmethod
    def _latest_ply(out):
        pc = os.path.join(out, "point_cloud")
        try:
            its = sorted((d for d in os.listdir(pc) if d.startswith("iteration_")),
                         key=lambda s: int(s.split("_")[-1]))
        except (OSError, ValueError):
            return None
        if not its:
            return None
        return os.path.join(pc, its[-1], "point_cloud.ply").replace("\\", "/")

    def _poll(self):
        # The preview window says which segment it shows: it is the current
        # segment's model at that segment's own midpoint (t=0.5 in the
        # trainer's preview camera). Without the label the image jumps
        # between segments with no explanation.
        if self.preview is not None:
            if self._running_plan and self._plan_i < len(self._plan):
                a, b = self._plan[self._plan_i]
                self.preview.set_caption(
                    f"segment {self._plan_i + 1}/{len(self._plan)} (steps {a}..{b})")
            else:
                self.preview.set_caption("")
        # A finished segment's training process does not exit by itself (it
        # waits so the user can decide about a checkpoint). Combined training
        # cannot wait: the next segment needs the same GPU, so it sends the
        # stop itself. The PLY of the last iteration has already been saved.
        if (self._running_plan and self.run.state == "finished"
                and not self._auto_release):
            self._auto_release = True
            self.run.stop()
        if self._merge is not None and self._merge.poll() is not None:
            rc, self._merge = self._merge.returncode, None
            try:
                self._merge_log.close()
            except Exception:                                 # noqa: BLE001
                pass
            merged = self._out_dir(self.name.get().strip())
            if rc == 0:
                self.msg.configure(
                    text=f"Combined training finished  merged model in {merged}",
                    foreground=GREEN)
            else:
                self.msg.configure(text=f"Merge failed (exit code {rc}), "
                                        f"see {merged}_merge.log",
                                   foreground=RED)
            self._show_out()
        super()._poll()
