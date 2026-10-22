"""Preprocessing tab: recording -> dataset.

Behaviour = `python prep_4d.py <shooting root>`, with no flags that change defaults.

Flow: choose a root (or change a parameter) -> automatic pre-check (seconds)
-> "Run preprocessing" unlocks only if it passes.
"""
import os
import tkinter as tk
from tkinter import ttk

import params as P
import prep_runner as PR
from tooltip import Tooltip

GREY = "#888888"
RED = "#c02020"
GREEN = "#207020"
BLUE = "#1f6fb4"           # same blue as the loss curve: numbers worth reading
POLL_MS = 300

# Acceptance thresholds
RATE_MIN = 90.0        # valid rate %
REPROJ_MAX = 2.5       # tri_reproj_px

SHOW = [("nsteps", "Time steps"), ("duration_s", "Length (s)"),
        ("background", "Background"), ("unit_mm", "1 unit = ? mm"),
        ("npoints", "Initial points"), ("tri_points", "Triangulated pts"),
        ("fps", "fps"), ("wh", "Train resolution")]

# Only these sections of the phase report are shown. Sections are selected,
# not truncated: the per-camera offset table comes first, and cutting the
# tail would also remove the sections of interest. The full report is
# <dataset>/phase_report.md.
PHASE_KEEP = ("## Group time span", "## Discarded groups")


def _trim_phase(text):
    out, keeping = [], True          # everything before the first ## is the header; kept
    for line in text.strip().splitlines():
        if line.startswith("## "):
            keeping = line.startswith(PHASE_KEEP)
        if keeping:
            out.append(line)
    return "\n".join(out).strip()


class PrepTab(ttk.Frame):
    def __init__(self, master, get_root, show_error, set_busy=None):
        super().__init__(master, padding=10)
        self.get_root = get_root
        self.show_error = show_error
        self.set_busy = set_busy or (lambda busy: None)
        self.run = PR.PrepRunner()
        self.vars = {}
        self.check = None            # result of the latest pre-check
        self._after_id = None

        self.columnconfigure(1, weight=1)
        row = 0

        # No tiers: preprocessing has one main knob (time steps).
        box = ttk.LabelFrame(self, text="Parameters", padding=8)
        box.grid(row=row, column=0, columnspan=3, sticky="we", pady=4)
        self.param_box = box
        self._fields(box, P.PREP_PARAMS)
        row += 1

        # The pre-check result: the most important line of this tab
        self.chk_lab = ttk.Label(self, text="Choose a shooting root", foreground=GREY,
                                 justify="left", wraplength=700)
        self.chk_lab.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 2))
        row += 1

        btns = ttk.Frame(self)
        btns.grid(row=row, column=0, columnspan=3, sticky="we", pady=6)
        self.b_run = ttk.Button(btns, text="Run preprocessing", command=self._start)
        self.b_stop = ttk.Button(btns, text="Stop", command=self._stop)
        self.b_recheck = ttk.Button(btns, text="Pre-check again", command=self.refresh_root)
        for b in (self.b_run, self.b_stop, self.b_recheck):
            b.pack(side="left", padx=(0, 8))
        Tooltip(self.b_run, "Full preprocessing took about 13 minutes for 301 time steps (measured).\n"
                            "A dataset with the same name is overwritten.")
        Tooltip(self.b_stop, "Preprocessing has no checkpoint; stopping terminates it.\n"
                             "Outputs are kept and overwritten by the next run.")
        row += 1

        self.msg = ttk.Label(self, text="", foreground=GREY)
        self.msg.grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # Progress and results are both shown here, not in separate windows.
        # Preprocessing only prints text (no structured data), so progress is
        # the tail of its log.
        wrap = ttk.Frame(self)
        wrap.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        self.log = tk.Text(wrap, height=10, wrap="none", state="disabled",
                           background="#f7f7f7")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        # Fixed colour meanings: green = passes, red = fails or bad news,
        # blue = numbers worth reading, grey = everything else
        self.log.tag_configure("good", foreground=GREEN)
        self.log.tag_configure("bad", foreground=RED)
        self.log.tag_configure("key", foreground=BLUE)
        self.log.tag_configure("head", foreground=GREY)
        self.log.tag_configure("title", foreground="black",
                               font=("", 9, "bold"))
        self.rowconfigure(row, weight=1)

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
            else:
                var = tk.StringVar(value="" if p.default is None else str(p.default))
                w = ttk.Entry(box, textvariable=var, width=14)
            w.grid(row=i, column=1, sticky="w", padx=8, pady=2)
            Tooltip(lab, p.tip)
            Tooltip(w, p.tip)
            self.vars[p.key] = var

    def values(self):
        out = {}
        for p in P.PREP_PARAMS:
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
        return out

    # ------------------------------------------------------------ pre-check
    def refresh_root(self):
        """Called when the shooting root or a parameter changes.

        The pre-check takes seconds, so it runs synchronously, after a message
        so the window does not look frozen. It is read-only (see
        prep_runner.precheck).
        """
        if self.run.state == "running":
            return
        root = self.get_root()
        if not root or not os.path.isdir(root):
            self.check = None
            self.chk_lab.configure(text="Choose a shooting root", foreground=GREY)
            self._sync_buttons()
            return
        self.chk_lab.configure(text="Pre-checking... (a few seconds)", foreground=GREY)
        self.update_idletasks()
        self.check = PR.precheck(root, self.values())
        c = self.check
        if c.get("rate") is not None:
            head = (f"Valid rate {c['rate']:.1f}% "
                    f"({c['ok_groups']}/{c['all_groups']} groups)  "
                    f"sampled {c['nsteps']} time steps  -> {c['dataset']}")
        else:
            head = ""
        if c["ok"]:
            same = c["dataset"] in c.get("existing", [])
            tail = "  the existing dataset with this name will be overwritten" if same else ""
            self.chk_lab.configure(text=head + tail, foreground=GREEN)
        else:
            self.chk_lab.configure(text=(head + "\n" if head else "") + c["why"],
                                   foreground=RED)
        self._sync_buttons()

    # ------------------------------------------------------------ run
    def _start(self):
        if not (self.check and self.check.get("ok")):
            return
        try:
            self.run.start(self.get_root(), self.values(), self.check["dataset"])
        except Exception as e:                                # noqa: BLE001
            self.show_error("Could not start preprocessing", str(e))
            return
        self.msg.configure(text=f"Preprocessing - output {self.run.out}", foreground=GREY)
        self._sync_buttons()

    def _stop(self):
        self.run.stop()
        self._sync_buttons()

    def _show_result(self):
        """Write the acceptance numbers into the text area below (no separate window)."""
        res = self.run.result()
        win = res.get("window", {})
        self.log.configure(state="normal")
        # Appended below the progress output rather than replacing it, so
        # the process and the result can be read together.
        if self.log.get("1.0", "end").strip():
            self.log.insert("end", "\n" + "-" * 60 + "\n\n", "head")
        if res.get("error"):
            self.log.insert("end", res["error"] + "\n", "bad")
        self.log.insert("end", f"Preprocessing result  {os.path.basename(self.run.out)}\n",
                        "title")

        # Two thresholds: the number is coloured (green pass, red fail), label and threshold grey
        rate = (self.check or {}).get("rate")
        if rate is not None:
            self._kv("Valid rate", f"{rate:.1f} %",
                     "good" if rate > RATE_MIN else "bad", f"needs > {RATE_MIN:.0f}")
        rp = win.get("tri_reproj_px")
        if rp is not None:
            self._kv("Reproj. error", f"{rp} px",
                     "good" if rp < REPROJ_MAX else "bad", f"needs < {REPROJ_MAX}")

        # the remaining numbers describe the data; blue sets them apart from the text
        for key, label in SHOW:
            if key in win:
                self._kv(label, str(win[key]), "key")
        ph = _trim_phase(res.get("phase") or "")
        if ph:
            self.log.insert("end", "\n")
            for line in ph.splitlines():
                self.log.insert("end", line + "\n", self._phase_tag(line))
        self.log.see("end")
        self.log.configure(state="disabled")

    def _kv(self, label, value, tag, note=""):
        """One line: label (grey), value (coloured), note (grey)."""
        self.log.insert("end", f"  {label}".ljust(16), "head")
        self.log.insert("end", value, tag)
        if note:
            self.log.insert("end", f"  ({note})", "head")
        self.log.insert("end", "\n")

    @staticmethod
    def _phase_tag(line):
        """Lines of the phase report worth noticing: jitter and valid rate."""
        if "jitter" in line or "valid" in line:
            return "key"
        return "head"

    # ------------------------------------------------------------ polling
    def stop_polling(self):
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:                                 # noqa: BLE001
                pass
        self._after_id = None

    def _poll(self):
        self._after_id = None
        state = self.run.poll()
        if state in ("running", "stopping"):
            self._set_log(self.run.log_tail(8))
        elif state == "done":
            self.run.state = "idle"
            self._set_log(self.run.log_tail(8))
            self.msg.configure(text=f"Preprocessing finished - {self.run.out}",
                               foreground=GREY)
            self._show_result()
            # No pre-check here: the result was just written, and nothing
            # should touch the new dataset afterwards.
            self._sync_buttons()
        elif state == "stopped":
            self.run.state = "idle"
            self._set_log(self.run.log_tail(8))
            self.msg.configure(text="Stopped - outputs kept; the next run overwrites them",
                               foreground=GREY)
            self.refresh_root()
            self._sync_buttons()
        elif state == "failed":
            self.run.state = "idle"
            self._set_log(self.run.log_tail(8))
            self.msg.configure(
                text=f"Preprocessing did not finish normally (exit code {self.run.returncode})",
                foreground=RED)
            self.show_error(f"Preprocessing failed (exit code {self.run.returncode})",
                            "\n".join(self.run.log_tail(200)))
            self._sync_buttons()
        self._after_id = self.after(POLL_MS, self._poll)

    # The live log is coloured too: numbers and bad news stand out
    BAD_WORDS = ("failed", "Failed", "Error", "Traceback", "ERROR")
    KEY_WORDS = ("Valid", "triangulat", "reprojection", "Dataset", "scale", "time step")

    def _set_log(self, lines):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for line in lines:
            if any(w in line for w in self.BAD_WORDS):
                tag = "bad"
            elif any(w in line for w in self.KEY_WORDS):
                tag = "key"
            else:
                tag = "head"
            self.log.insert("end", line + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _sync_buttons(self):
        """Button enabling and disabling happens only here (as in the training tab)."""
        busy = self.run.state in ("running", "stopping")
        ok = bool(self.check and self.check.get("ok"))
        self.b_run.state(["!disabled"] if (ok and not busy) else ["disabled"])
        self.b_stop.state(["!disabled"] if busy else ["disabled"])
        self.b_recheck.state(["disabled"] if busy else ["!disabled"])
        for w in self.param_box.winfo_children():
            if isinstance(w, (ttk.Entry, ttk.Checkbutton)):
                w.state(["disabled"] if busy else ["!disabled"])
        self.set_busy(busy)
