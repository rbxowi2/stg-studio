"""Training tab: parameter form, buttons, curve.

Parameter names and help text all come from params.py; this module only lays
them out. Enabling and disabling buttons happens in one place,
`_sync_buttons()`, because logic spread over several places always misses a path.
"""
import json
import os
import time
import tkinter as tk
from tkinter import ttk

import params as P
import runner as R
from curve import Curve
from preview_win import PreviewWindow
from tooltip import Tooltip

GREY = "#888888"
RED = "#c02020"
GREEN = "#207020"
POLL_MS = 200
ADV_HEIGHT = 170           # visible height of the advanced section (px); scrolls beyond
# The three states in which the training process holds GPU memory.
# finished = done but not unloaded, waiting for commands; like running/paused
# it must allow saving and stopping.
LIVE = ("running", "paused", "finished")


class TrainTab(ttk.Frame):
    def __init__(self, master, get_root, show_error, set_busy=None):
        super().__init__(master, padding=10)
        self.get_root = get_root            # the shooting root shared by the main window
        self.show_error = show_error
        # while training, the main window locks its own controls (root, other tabs)
        self.set_busy = set_busy or (lambda busy: None)
        self.run = R.Runner()
        self.vars = {}
        self.widgets = {}       # key -> input widget (locked individually when resuming)
        self.info = None
        self.compat = None          # not None = reason this setup would produce an unplayable model
        self._was_rendering = False
        self._chain = None           # progress of the one-click resume (see _finish_resume)
        self.preview = None
        # not None = the output name points to a training run that can be resumed
        self.resume_meta = None
        self._filled_from = None    # the output folder whose parameters were filled in

        self.columnconfigure(1, weight=1)
        row = 0

        # -- tier
        bar = ttk.Frame(self)
        bar.grid(row=row, column=0, columnspan=3, sticky="we", pady=(0, 8))
        ttk.Label(bar, text="Tier").pack(side="left")
        self.tier = tk.StringVar(value="balanced")
        self.tier_rbs = []
        for name in P.TIER_NAMES:
            rb = ttk.Radiobutton(bar, text=name, value=name, variable=self.tier,
                                 command=self._apply_tier)
            rb.pack(side="left", padx=(10, 0))
            self.tier_rbs.append(rb)
        self.custom = ttk.Label(bar, text="", foreground=GREY)
        self.custom.pack(side="left", padx=12)
        row += 1

        # -- output name
        # The name is exposed and an existing target is never overwritten
        # silently: a default name per tier would write into a folder that
        # often already holds an accepted model.
        nb = ttk.Frame(self)
        nb.grid(row=row, column=0, columnspan=3, sticky="we")
        ttk.Label(nb, text="Output name").pack(side="left")
        self.name = tk.StringVar(value="")
        self.name_ent = ent = ttk.Entry(nb, textvariable=self.name, width=24)
        ent.pack(side="left", padx=8)
        self.name.trace_add("write", lambda *_a: self._show_out())
        Tooltip(ent, "The model is saved in the <name>_out folder under the shooting root.")
        row += 1

        # The output path on its own line, so it is clear where the model is written
        self.out_lab = ttk.Label(self, text="", foreground=GREY)
        self.out_lab.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 6))
        row += 1

        # -- layer A: automatic, read-only
        self.auto_box = ttk.LabelFrame(self, text="From the dataset (read-only)",
                                       padding=8)
        self.auto_box.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        self.auto_labels = {}
        for i, p in enumerate(P.AUTO):
            ttk.Label(self.auto_box, text=p.label).grid(row=i, column=0, sticky="w")
            lab = ttk.Label(self.auto_box, text="-", foreground=GREY)
            lab.grid(row=i, column=1, sticky="w", padx=10)
            Tooltip(lab, p.tip)
            self.auto_labels[p.key] = lab
        row += 1

        # -- layer B: main knobs filled by the tier
        self.main_box = ttk.LabelFrame(self, text="Main parameters (filled by the tier)",
                                       padding=8)
        self.main_box.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        self._fields(self.main_box, P.TIERED)
        row += 1

        # -- layer C: advanced, in a collapsible section
        self.adv_open = tk.BooleanVar(value=False)
        self.adv_btn = ttk.Checkbutton(self, text="Advanced parameters", style="Toolbutton",
                                       variable=self.adv_open,
                                       command=self._toggle_adv)
        self.adv_btn.grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        # The advanced section has 16 fields; fully expanded the window would
        # be taller than the screen, so they sit in a fixed-height scrolling area.
        self.adv_wrap = ttk.Frame(self)
        self.adv_wrap.columnconfigure(0, weight=1)
        self.adv_canvas = tk.Canvas(self.adv_wrap, height=ADV_HEIGHT,
                                    highlightthickness=0)
        adv_sb = ttk.Scrollbar(self.adv_wrap, orient="vertical",
                               command=self.adv_canvas.yview)
        self.adv_canvas.configure(yscrollcommand=adv_sb.set)
        self.adv_canvas.grid(row=0, column=0, sticky="nsew")
        adv_sb.grid(row=0, column=1, sticky="ns")
        self.adv_box = ttk.Frame(self.adv_canvas, padding=(8, 4))
        self._fields(self.adv_box, P.ADVANCED)
        self.adv_canvas.create_window((0, 0), window=self.adv_box, anchor="nw")
        self.adv_box.bind(
            "<Configure>",
            lambda e: self.adv_canvas.configure(
                scrollregion=self.adv_canvas.bbox("all")))
        self.adv_canvas.bind(
            "<MouseWheel>",
            lambda e: self.adv_canvas.yview_scroll(-e.delta // 120, "units"))
        self.adv_row = row
        row += 1

        # -- buttons
        btns = ttk.Frame(self)
        btns.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        self.b_start = ttk.Button(btns, text="Start training", command=self._start)
        self.b_pause = ttk.Button(btns, text="Pause", command=self._pause)
        self.b_save = ttk.Button(btns, text="Save checkpoint and model", command=self._save)
        self.b_stop = ttk.Button(btns, text="Stop", command=self._stop)
        self.b_prev = ttk.Button(btns, text="Preview window", command=self._toggle_preview)
        for b in (self.b_start, self.b_pause, self.b_save, self.b_stop, self.b_prev):
            b.pack(side="left", padx=(0, 8))
        Tooltip(self.b_pause,
                "Training waits in place. GPU memory is not released; use Stop to release it.")
        Tooltip(self.b_save,
                "Save a resumable checkpoint plus a model the player can open.\n"
                "Stop does not save; besides this button, only the two\n"
                "autosave percentages keep anything.")
        Tooltip(self.b_start,
                "When the output name matches a run stopped here, this button becomes\n"
                "\"Resume from checkpoint\". A resumed model is not bit-identical to an\n"
                "uninterrupted run: training itself is not deterministic (the rasterizer\n"
                "accumulates gradients with CUDA atomics). Near the guided sampling start,\n"
                "points may be added one time more or fewer.")
        row += 1

        self.msg = ttk.Label(self, text="", foreground=GREY)
        self.msg.grid(row=row, column=0, columnspan=2, sticky="w")
        # Always visible: the latest save. The status line is overwritten by
        # the next message, while "last saved where" must stay visible.
        self.saved_lab = ttk.Label(self, text="No checkpoint saved yet", foreground=GREY)
        self.saved_lab.grid(row=row, column=2, sticky="e")
        row += 1

        self.curve = Curve(self)
        self.curve.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        self.rowconfigure(row, weight=1)

        self._apply_tier()
        self._sync_buttons()
        self._after_id = self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------ layout
    def _fields(self, box, plist):
        box.columnconfigure(1, weight=1)
        for i, p in enumerate(plist):
            lab = ttk.Label(box, text=p.label)
            lab.grid(row=i, column=0, sticky="w", pady=2)
            if p.kind is bool:
                var = tk.BooleanVar(value=bool(p.default))
                w = ttk.Checkbutton(box, variable=var)
            elif p.choices:
                var = tk.StringVar(value=str(p.default))
                w = ttk.Combobox(box, textvariable=var, state="readonly", width=12,
                                 values=[str(c) for c in p.choices])
            else:
                var = tk.StringVar(value="" if p.default is None else str(p.default))
                w = ttk.Entry(box, textvariable=var, width=14)
                var.trace_add("write", lambda *_a: self._mark_custom())
            w.grid(row=i, column=1, sticky="w", padx=8, pady=2)
            # help on both the label and the field: the pointer may rest on either
            Tooltip(lab, p.tip)
            Tooltip(w, p.tip)
            self.vars[p.key] = var
            self.widgets[p.key] = w
            if p.key == "time_range":
                # Next to the time range: its length and whether it is too short,
                # shown beside the field so a mistake is visible immediately.
                self.range_lab = ttk.Label(box, text="", foreground=GREY)
                self.range_lab.grid(row=i, column=2, sticky="w", padx=(4, 0))

    def _toggle_adv(self):
        if self.adv_open.get():
            self.adv_wrap.grid(row=self.adv_row, column=0, columnspan=3,
                               sticky="we")
        else:
            self.adv_wrap.grid_forget()

    # ------------------------------------------------------------ parameters
    def _apply_tier(self):
        for p in P.TIERED:
            self.vars[p.key].set(str(p.value_for(self.tier.get())))
        self.custom.configure(text="")
        self._show_out()

    def _mark_custom(self):
        tier = self.tier.get()
        for p in P.TIERED:
            if self.vars[p.key].get() != str(p.value_for(tier)):
                self.custom.configure(text="(custom)")
                return
        self.custom.configure(text="")

    # Bounded fields: out-of-range values are clamped and the clamped value
    # is written back, so the field never shows a value that does not apply.
    CLAMP = {"autosave_pct": (0, 100), "ckpt_pct": (0, 100)}

    def values(self):
        out = {}
        for p in P.TIERED + P.ADVANCED:
            raw = self.vars[p.key].get()
            if p.kind is bool:
                out[p.key] = bool(raw)
            elif p.kind is str:
                out[p.key] = raw
            else:
                try:
                    out[p.key] = p.kind(raw)
                except (TypeError, ValueError):
                    out[p.key] = p.default
            if p.key in self.CLAMP:
                lo, hi = self.CLAMP[p.key]
                v = min(max(out[p.key], lo), hi)
                if v != out[p.key]:
                    out[p.key] = v
                    self.vars[p.key].set(str(v))
        return out

    def refresh_root(self):
        """Called when the shooting root changes. No dataset is not an error; it just cannot run yet."""
        try:
            self.info = R.dataset_info(self.get_root())
        except SystemExit as e:              # find_dataset reports with SystemExit
            self.info = None
            self.msg.configure(text=str(e).strip(), foreground=RED)
        except KeyError as e:
            # window.json lacks a field, usually near_far. The geometry fields
            # are written at the very end of preprocessing stage 2, so a
            # missing one means preprocessing did not finish. A bare KeyError
            # would only show 'near_far'.
            self.info = None
            self.msg.configure(
                text=f"This dataset's window.json lacks {e}: "
                     "preprocessing did not finish (geometry fields are written last in stage 2). "
                     "Run preprocessing again.", foreground=RED)
        except Exception as e:                                # noqa: BLE001
            self.info = None
            self.msg.configure(text=str(e), foreground=RED)
        else:
            # Block setups that would train a model the player cannot open
            # before starting; finding out after training costs too much time.
            self.compat = R.compat_problem(self.info)
            for p in P.AUTO:
                v = self.info.get(p.key, self.info.get(p.key.replace("_", "")))
                self.auto_labels[p.key].configure(text=self._short(v))
            if self.compat:
                self.msg.configure(text=self.compat, foreground=RED)
            else:
                self.msg.configure(text=f"Dataset {self.info['name']}  "
                                        f"{self.info['duration']} time steps",
                                   foreground=GREY)
            self._show_out()
        self._sync_buttons()

    def _fill_meta(self, meta):
        """Resume mode: fill in the previous run's settings, tier included.

        The tier first, then the parameters: changing the tier refills all
        layer-B parameters with that tier's defaults. `self.tier` is set
        directly (`_apply_tier` is bound to user clicks only; `set()` does
        not trigger it).

        Filled once per output folder (`_filled_from`): this runs on every
        keystroke in the name field, and refilling each time would undo the
        user's edits. The values are filled in first; then the user may change them.
        """
        tier = meta.get("tier")
        if tier in P.TIER_NAMES:
            self.tier.set(tier)
        values = meta.get("params") or {}
        for key, var in self.vars.items():
            if key not in values:
                continue
            try:
                var.set(values[key] if isinstance(var, tk.BooleanVar)
                        else str(values[key]))
            except tk.TclError:
                pass
        self._mark_custom()          # mark "(custom)" when it differs from the tier

    def _desicnt(self):
        try:
            return int(self.vars["desicnt"].get())
        except (TypeError, ValueError):
            return None

    def _resume_marks(self):
        """Iterations to mark on the curve (from `resumes` in train_meta.json)."""
        meta = R.read_meta(self.run.out) if self.run.out else None
        return list((meta or {}).get("resumes", []))

    def _out_dir(self, name):
        if self.info is None:
            return ""
        return os.path.join(os.path.dirname(self.info["dataset"]),
                            f"{name}_out").replace("\\", "/")

    def _show_out(self):
        """The output path line; also decides between "new run" and "resume".

        An existing name is not always blocked: a folder with
        `train_meta.json` and `_studio/state.pth` is a run stopped here,
        which can be resumed.
        """
        name = self.name.get().strip()
        out = self._out_dir(name) if name else ""
        self.resume_meta = None
        # When the name points elsewhere, forget which folder was filled in,
        # so returning to it fills the values in again.
        if self._filled_from and self._filled_from != out:
            self._filled_from = None
        if not name:
            self.out_lab.configure(text="Enter an output name", foreground=RED)
        elif not out:
            self.out_lab.configure(text="", foreground=GREY)
        elif not os.path.isdir(out):
            self.out_lab.configure(text=out, foreground=GREY)
        else:
            kind, meta = R.resume_state(out)
            if kind == "resume":
                self.resume_meta = meta
                # Fill in the previous settings once per visit to this folder
                # (this runs on every keystroke). Leaving and coming back
                # counts as a new visit (the marker is cleared above).
                if self._filled_from != out:
                    self._filled_from = out
                    self._fill_meta(meta)
                done = meta.get("last_iter") or 0
                total = meta.get("total_iterations") or 0
                self.out_lab.configure(
                    text=f"{out}  can be resumed: stopped at {done:,} of {total:,}",
                    foreground=GREEN)
            elif kind == "no_state":
                self.out_lab.configure(
                    text=f"{out}  has a record but no checkpoint, so it cannot be resumed "
                         "(a PLY has no optimizer state); choose another name", foreground=RED)
            else:
                self.out_lab.configure(text=f"{out}  already exists; choose another name",
                                       foreground=RED)
        self._sync_buttons()

    def _name_ok(self):
        """A name is required; an existing folder is accepted only when it can be resumed (see `_show_out`)."""
        name = self.name.get().strip()
        if not name:
            return False
        out = self._out_dir(name)
        if not out:
            return False
        if not os.path.isdir(out):
            return True
        if self.resume_meta is None:
            return False
        # A finished run needs no further start, unless the iterations were increased
        done = self.resume_meta.get("last_iter") or 0
        return self._iterations() > done

    def _range_note(self):
        """The line next to the time range field, recomputed from the current state on every poll.

        Returns True when the range can be used (empty counts as valid).
        """
        lab = getattr(self, "range_lab", None)
        if lab is None:
            return True
        if self.info is None:
            lab.configure(text="", foreground=GREY)
            return True
        nsteps = int(self.info.get("duration") or 0)
        if not nsteps:                      # no dataset recognized yet: no judgement, no block
            lab.configure(text="", foreground=GREY)
            return True
        try:
            rng = R.parse_range(self.vars["time_range"].get(), nsteps)
        except ValueError as e:
            lab.configure(text=str(e), foreground=RED)
            return False
        if rng is None:
            lab.configure(text=f"all {nsteps} time steps", foreground=GREY)
            return True
        a, b = rng
        n = b - a + 1
        secs = self._range_seconds(a, b)
        tail = f", about {secs:.2f} s" if secs else ""
        if n < 30:
            # Not blocked: a short range is valid, but motion is poorly
            # constrained and the image may distort.
            lab.configure(text=f"{n} time steps{tail} - fewer than 30, motion may distort",
                          foreground=RED)
        else:
            lab.configure(text=f"{n} time steps{tail}", foreground=GREY)
        return True

    def _range_seconds(self, a, b):
        """Length of the range in seconds; 0 if it cannot be computed (then no seconds are shown)."""
        ds = (self.info or {}).get("dataset")
        if not ds:
            return 0.0
        # Recomputed every 200 ms while the two JSON files do not change, so
        # results are cached instead of re-reading the files on every poll.
        memo = self.__dict__.setdefault("_secs_memo", {})
        key = (ds, a, b)
        if key in memo:
            return memo[key]
        try:
            win = json.load(open(f"{ds}/window.json", encoding="utf-8"))
            times = json.load(open(f"{ds}/times_group.json", encoding="utf-8"))
            ts = [float(v) for k, v in times.items()
                  if a <= int(k.split("/")[0]) <= b]
            memo[key] = (max(ts) - min(ts)) * float(win.get("duration_s") or 0.0)
        except (OSError, ValueError, KeyError):
            memo[key] = 0.0
        return memo[key]

    def _iterations(self):
        """The iterations entered in the field; invalid (empty, not a number) counts as 0."""
        try:
            return int(self.vars["iterations"].get())
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _short(v):
        s = str(v)
        return "..." + s[-46:] if len(s) > 48 else s

    # ------------------------------------------------------------ control
    def _start(self):
        if self.run.state == "finished":
            self._finish_resume()
            return
        if self.info is None or self.run.state in LIVE:
            return
        vals = self.values()
        # No name or a taken name: no start (the button is disabled; this is
        # a second check). An existing model may already be accepted, and
        # overwriting it cannot be undone.
        if not self._name_ok():
            return
        name = self.name.get().strip()
        resume = self.resume_meta is not None
        try:
            self.run.start(self.get_root(), self.tier.get(), vals, name=name,
                           resume=resume)
        # SystemExit too: find_dataset uses it for "no dataset / two
        # datasets", and as a BaseException it would otherwise break the Tk
        # callback and look like a hang.
        except (Exception, SystemExit) as e:                  # noqa: BLE001
            self.show_error("Could not resume" if resume else "Could not start training", str(e))
            return
        if resume:
            # The curve continues; the resume line comes from `resumes` in train_meta.json
            self.curve.set_points(self.run.points, desicnt=self._desicnt(),
                                  marks=self._resume_marks())
        else:
            # marks=[] is required: the curve remembers previous resume
            # points, and a new run must not show another run's lines
            self.curve.set_points([], marks=[])
        self.msg.configure(text=f"Training - output {self.run.out}", foreground=GREY)
        self._sync_buttons()

    def _finish_resume(self):
        """Continue a finished-and-waiting run with changed parameters.

        A running process cannot change its parameters (they are command-line
        arguments given at start), so this takes four steps: save -> stop and
        release -> change parameters -> start a new process with them. The
        user presses one button; the middle steps advance with polling in
        `_advance_chain()`, and the status line reports each one, since every
        step takes a few seconds.
        """
        if self._chain:
            return
        self._chain = {"stage": "save", "tier": self.tier.get(),
                       "values": self.values(), "name": self.name.get().strip()}
        self.run.save()
        self.saved_lab.configure(text="Saving...", foreground=RED)
        self._sync_buttons()

    def _advance_chain(self, state):
        """Steps two and three of the one-click resume. On failure, fall back to an understandable state."""
        if not self._chain:
            return
        if self._chain["stage"] == "save":
            if self.run.saving:
                return
            if not self.run.checkpoint:
                self._chain = None
                self.show_error("Could not resume", "No checkpoint was written; training is still waiting.")
                return
            self._chain["stage"] = "stop"
            self.run.stop()
            self.msg.configure(text="Freeing GPU memory...  then resuming with the new parameters",
                               foreground=RED)
            return
        if self._chain["stage"] == "stop" and self.run.state == "idle":
            c, self._chain = self._chain, None
            try:
                self.run.start(self.get_root(), c["tier"], c["values"],
                               name=c["name"], resume=True)
            except Exception as e:                            # noqa: BLE001
                self.show_error("Could not resume", str(e))
                self._show_out()
                return
            self.curve.set_points(self.run.points, desicnt=self._desicnt(),
                                  marks=self._resume_marks())
            self.msg.configure(text=f"Resumed - output {self.run.out}",
                               foreground=GREY)

    def _pause(self):
        if self.run.state == "running":
            self.run.pause()
        elif self.run.state == "paused":
            self.run.resume()
        self._sync_buttons()

    def _save(self):
        self.run.save()
        # Saving is not immediate: training stops at its next check (every
        # 50 iterations, about 1.5 s); a state saved mid-iteration would be
        # inconsistent and break resuming.
        self.saved_lab.configure(text="Saving...", foreground=RED)

    def _stop(self):
        self.request_stop()
        self._sync_buttons()

    def request_stop(self, title="Stop training"):
        """Shared by the Stop button and closing the main window. Returns True once stopping has begun.

        Stop acts immediately, without a confirmation dialog. To keep
        something, set an autosave or press "Save checkpoint and model"
        first. Stop does one thing: end training and release GPU memory.
        """
        self.run.stop()
        return True

    def _current_iter(self):
        """Current iteration (the larger of the curve and the log)."""
        return max(self.run.log_iteration() or 0,
                   self.run.points[-1]["iter"] if self.run.points else 0)

    def _toggle_preview(self):
        if self.preview is None:
            self.preview = PreviewWindow(self.winfo_toplevel())
            self.preview.bind_render(self.run.render_video)
        if self.preview.visible:
            self.preview.hide()          # refuses while rendering (see preview_win)
        else:
            self.preview.show(self.run.preview or "")

    # ------------------------------------------------------------ polling
    def stop_polling(self):
        """Must be called before the window is destroyed.

        Checking winfo_exists inside _poll is not enough: when the window is
        destroyed, the scheduled `after` command becomes invalid at the Tcl
        level before Python runs, raising `invalid command name "..._poll"`.
        Cancelling the callback is the clean way.
        """
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:                                 # noqa: BLE001
                pass
        self._after_id = None
        if self.preview is not None:
            self.preview.stop_polling()

    def _poll(self):
        self._after_id = None
        state, new = self.run.poll()
        self._advance_chain(state)
        if new:
            self.curve.set_points(self.run.points, desicnt=self._desicnt())
            if self.preview is not None and self.preview.visible:
                self.preview.path = self.run.preview or self.preview.path
        if self.preview is not None:
            # Two conditions for rendering a video:
            #   1. training is alive (the model is in its GPU memory, not here)
            #   2. the first window-center preview exists; before that there is
            #      nothing to draw, and the render would only stall training
            self.preview.set_render_state(
                busy=self.run.rendering,
                can_render=(self.run.state in LIVE
                            and self.run.preview is not None),
                video=self.run.video)
        self._refresh_saved()
        self._show_status()
        # Buttons are recomputed on every poll, not only on events: an input
        # change (for example a larger iteration count in resume mode) is not
        # an event that would otherwise trigger it. Setting a few dozen widget
        # states every 200 ms costs nothing measurable.
        self._was_rendering = self.run.rendering
        self._sync_buttons()
        if state == "failed":
            self.run.state = "idle"
            # The failed run already created its output folder, so the name
            # now counts as taken; recheck, or Start would overwrite its remains.
            self._show_out()
            self.show_error(f"Training failed (exit code {self.run.returncode})",
                            self.run.log_tail())
            self._sync_buttons()
        elif state == "done":
            # Not self.run.state == "stopping": poll() has already changed it
            # to done. See Runner.asked_stop.
            stopped = self.run.asked_stop
            self.run.state = "idle"
            # Stop does not save, so whether a playable model exists is
            # checked in the folder, not inferred. Claiming a model was saved
            # when it was not is the dangerous direction: people trust it and
            # close the window.
            head = "Stopped" if stopped else "Training finished"
            it = self._latest_iteration()
            if it:
                self.msg.configure(
                    text=f"{head}  a playable model was saved (iteration {it:,})  "
                         f"- in {self.run.out}", foreground=GREY)
            else:
                self.msg.configure(
                    text=f"{head}  no playable model in the output folder - {self.run.out}",
                    foreground=RED)
            self._check_output()
            self._show_out()          # the new model now counts as existing
            self._sync_buttons()
            # The last step of the one-click resume waits until the process has
            # really ended (GPU memory released), so it runs after this cleanup.
            self._advance_chain(self.run.state)
            # Hook for subclasses (combined training starts the next segment);
            # also after the cleanup, for the same reason.
            self._on_run_done(stopped)
        self._after_id = self.after(POLL_MS, self._poll)

    def _on_run_done(self, stopped):
        """A training run ended (`stopped` = the user pressed stop).

        Nothing to do in the training tab; the combined training tab overrides it.
        """

    def _show_status(self):
        """The status line is written only here, recomputed from the current state on every poll.

        A message written once when an event happens, relying on something
        else to replace it later, gets stuck: for example "Rendering video"
        written at render start would remain forever while paused, because
        no new data arrives to overwrite it.

        Idle is not handled here: "training finished", "training failed" and
        similar end messages must stay on screen.
        """
        if self._chain:
            self.msg.configure(
                text="Saving checkpoint...  then freeing GPU memory and resuming with the new parameters"
                     if self._chain["stage"] == "save"
                     else "Freeing GPU memory...  then resuming with the new parameters",
                foreground=RED)
            return
        if self.run.state == "finished":
            # Finished but still holding GPU memory. Never stopped
            # automatically (that would lose things while the user is away),
            # so the reminder stays.
            self.msg.configure(
                text="Finished, GPU memory still in use - to resume later, press \"Save checkpoint and model\" first; "
                     "otherwise press \"Stop\" to free GPU memory",
                foreground=RED)
            return
        if self.run.state == "stopping":
            # Stopping is not instant: training handles the flag at its next
            # check. Stop does not save; anything to keep was saved before.
            self.msg.configure(text="Stopping...  freeing GPU memory",
                               foreground=RED)
            return
        if self.run.state not in ("running", "paused"):
            return
        if self.run.rendering:
            back = "paused" if self.run.state == "paused" else "training"
            self.msg.configure(text=f"Rendering video...  training paused, back to {back} when done",
                               foreground=GREY)
            return
        head = "Paused  " if self.run.state == "paused" else ""
        if not self.run.points:
            self.msg.configure(text=head or f"Training - output {self.run.out}",
                               foreground=GREY)
            return
        last = self.run.points[-1]
        dn = last.get("densify_n")
        grow = f"  densify {dn}/{self.vars['desicnt'].get()}" if dn is not None else ""
        # No speed or remaining time while paused; a rate means nothing then
        tail = "" if self.run.state == "paused" else f"  {self._speed_eta()}"
        self.msg.configure(
            text=f"{head}iteration {last['iter']:,}  loss {last['loss']:.5f}  "
                 f"{last['n_gauss']:,} Gaussians{grow}  "
                 f"max scale {last['max_scale']:.2f}{tail}",
            foreground=GREY)

    @staticmethod
    def _this_run(points):
        """Keep only the last part (where `elapsed_s` goes backwards is a resume seam)."""
        cut = 0
        for i in range(1, len(points)):
            if points[i].get("elapsed_s", 0) < points[i - 1].get("elapsed_s", 0):
                cut = i
        return points[cut:]

    def _speed_eta(self, window=5):
        """"N it/s  elapsed x  left y".

        Speed comes from the last few points, not the whole run: training
        slows noticeably as the Gaussian count grows, and a whole-run average
        would make the remaining time too optimistic.

        After a resume only this part's points are used: `elapsed_s` restarts
        from 0 in the new process, and mixing parts would give negative time
        differences. "Elapsed" is therefore this part's time, not cumulative.
        """
        pts = self._this_run(self.run.points)
        cur = pts[-1]
        done = self._hms(cur["elapsed_s"])
        if len(pts) < 2:
            return f"elapsed {done}"
        a = pts[max(0, len(pts) - 1 - window)]
        di, dt = cur["iter"] - a["iter"], cur["elapsed_s"] - a["elapsed_s"]
        if dt <= 0 or di <= 0:
            return f"elapsed {done}"
        rate = di / dt
        total = self._iterations()
        left = ""
        if total > cur["iter"]:
            left = f"  left {self._hms((total - cur['iter']) / rate)}"
        return f"{rate:.1f} it/s  elapsed {done}{left}"

    @staticmethod
    def _hms(sec):
        sec = int(max(sec, 0))
        h, m, s = sec // 3600, sec // 60 % 60, sec % 60
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _refresh_saved(self):
        """The always-visible "last saved" label: red while saving, then iteration and time."""
        if self.run.saving:
            self.saved_lab.configure(text="Saving...", foreground=RED)
            self._saved_mtime = None
            return
        it, mt = self.run.last_saved()
        if mt is None:
            self.saved_lab.configure(text="No checkpoint saved yet", foreground=GREY)
            return
        if mt != getattr(self, "_saved_mtime", None):
            self._saved_mtime = mt
            when = time.strftime("%H:%M:%S", time.localtime(mt))
            where = f"iter {it:,}" if it else "?"
            self.saved_lab.configure(text=f"Last saved  {where}  {when}",
                                     foreground=GREY)

    def _latest_iteration(self):
        """Iteration of the newest PLY in the output folder, or None."""
        pc = os.path.join(self.run.out or "", "point_cloud")
        try:
            its = [int(d.split("_")[-1]) for d in os.listdir(pc)
                   if d.startswith("iteration_")]
        except (OSError, ValueError):
            return None
        return max(its) if its else None

    def _check_output(self):
        """After training, confirm the player can read the produced PLY.

        A second check after the one before starting (compat_problem): what
        finally counts is the content of the file, not the configuration.
        """
        pc = os.path.join(self.run.out, "point_cloud")
        try:
            its = sorted((d for d in os.listdir(pc) if d.startswith("iteration_")),
                         key=lambda s: int(s.split("_")[-1]))
        except OSError:
            return
        if not its:
            return
        why = R.ply_problem(os.path.join(pc, its[-1], "point_cloud.ply"))
        if why:
            self.msg.configure(text=why, foreground=RED)
            self.show_error("The player cannot open this model", why)

    def _sync_buttons(self):
        """Button enabling and disabling happens only here.

        State machine:

            idle --start / resume from checkpoint--> running <-> pause/continue <-> paused
              ^                                         |                             |
              |                                         +-- completed --> finished    |
              |                                         |                    |        |
              +-- stopping <-- stop (frees GPU memory, does not save) -------+--------+

        running, paused and finished (= LIVE) hold GPU memory; all three can
        save, render and stop. Pause / continue only applies to the first two.

        Rendering is not a state but a busy period on top of those three;
        afterwards the state given by the flags applies again.

        While rendering, every control that changes the training state is
        frozen: pause / continue, save, stop, and the preview window's play
        and close. Those actions would run in the same training process as
        the video rendering; preventing the conflict is simpler than handling
        it. The zoom controls stay enabled: they do not touch training, and
        rendering is exactly when the image is being watched.
        """
        st = self.run.state
        # finished (done but still holding GPU memory): save, render and stop
        # work; pause / continue do not (nothing is running), and a new start
        # requires stopping first to release GPU memory.
        live = st in LIVE and not self.run.rendering
        running = st in ("running", "paused") and not self.run.rendering
        ready = (self.info is not None and self.compat is None
                 and self._name_ok() and self._range_note())
        busy = bool(self._chain)            # during the one-click resume everything is locked
        # resume mode: the name matches a run stopped here (see _show_out)
        resuming = st == "idle" and self.resume_meta is not None
        # In finished state this button runs "save -> stop -> restart with the
        # new parameters", but only if the iterations were increased;
        # otherwise there is nothing left to run.
        more = self._iterations() > self._current_iter()
        can_start = ((st == "idle" and ready)
                     or (st == "finished" and not self.run.rendering and more))
        self.b_start.state(["!disabled"] if can_start and not busy else ["disabled"])
        self.b_start.configure(
            text="Resume from checkpoint" if (resuming or st == "finished") else "Start training")
        live = live and not busy
        running = running and not busy
        self.b_pause.state(["!disabled"] if running else ["disabled"])
        self.b_pause.configure(text="Continue" if st == "paused" else "Pause")
        self.b_save.state(["!disabled"] if live else ["disabled"])
        self.b_stop.state(["!disabled"] if live else ["disabled"])
        # Once running, parameters, tier and output name are locked: changes
        # would not take effect, and seeming to would be worse (the run is
        # already writing to that folder).
        idle = st == "idle"
        # In finished state parameters are editable (to change them before
        # resuming), but as this is a resume, the frozen ones stay locked.
        editable = (idle or st == "finished") and not busy
        for key, w in self.widgets.items():
            # when resuming, also lock the parameters that cannot change
            ok = editable and not ((resuming or st == "finished")
                                   and key in P.RESUME_FROZEN)
            w.state(["!disabled"] if ok else ["disabled"])
        # The tier is locked when resuming: changing it would refill every
        # parameter, including the frozen ones
        for rb in self.tier_rbs:
            rb.state(["!disabled"] if (idle and not resuming) else ["disabled"])
        self.name_ent.state(["!disabled"] if idle else ["disabled"])
        self.set_busy(not idle)             # root and other tabs are in the main window
