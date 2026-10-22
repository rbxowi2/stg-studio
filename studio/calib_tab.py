"""Calibration tab: calibration set -> rig.

Behaviour = calibrate.py <calibration set> --edge x6.

Flow: choose a calibration set -> automatic read-only pre-check (instant)
-> "Run calibration" unlocks once all six distances are filled in.
The output goes to <calibration set>/rig.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk

import calib_runner as CR
import params as P
from tooltip import Tooltip

GREY = "#888888"
RED = "#c02020"
GREEN = "#207020"
AMBER = "#a06000"
BLUE = "#1f6fb4"
POLL_MS = 300


class CalibTab(ttk.Frame):
    def __init__(self, master, settings, save_settings, show_error,
                 set_busy=None):
        """settings: settings.json as read at startup (dict); save_settings(**fields)."""
        super().__init__(master, padding=10)
        self.save_settings = save_settings
        self.show_error = show_error
        self.set_busy = set_busy or (lambda busy: None)
        self.run = CR.CalibRunner()
        self.check = None
        self._after_id = None
        self.columnconfigure(1, weight=1)

        # ---- calibration set ----
        last = settings.get("last_calib") or ""
        self.path = tk.StringVar(value=last if os.path.isdir(last) else "")
        row = ttk.Frame(self)
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="Calibration set").grid(row=0, column=0, sticky="w")
        self.path_ent = ttk.Entry(row, textvariable=self.path)
        self.path_ent.grid(row=0, column=1, sticky="we", padx=8)
        self.pick_btn = ttk.Button(row, text="Browse...", command=self._pick)
        self.pick_btn.grid(row=0, column=2)

        # ---- six distances ----
        box = ttk.LabelFrame(self, text="Measured distances (mm)", padding=8)
        box.pack(fill="x", pady=(10, 4))
        self.edge_box = box
        saved = settings.get("calib_edges") or {}
        self.edges = {}
        for i, (key, a, b, default) in enumerate(P.CALIB_EDGES):
            r, c = divmod(i, 3)
            lab = ttk.Label(box, text=f"{a[-1]} - {b[-1]}")
            lab.grid(row=r, column=c * 2, sticky="w", padx=(0 if c == 0 else 18, 6),
                     pady=2)
            var = tk.StringVar(value=str(saved.get(key, default)))
            ent = ttk.Entry(box, textvariable=var, width=10)
            ent.grid(row=r, column=c * 2 + 1, sticky="w", pady=2)
            Tooltip(lab, P.CALIB_EDGE_TIP)
            Tooltip(ent, P.CALIB_EDGE_TIP)
            var.trace_add("write", lambda *_a: self._edges_changed())
            self.edges[key] = var
        self.edge_msg = ttk.Label(box, text="", foreground=RED)
        self.edge_msg.grid(row=2, column=0, columnspan=6, sticky="w")

        # ---- advanced (collapsible) ----
        self.adv_open = tk.BooleanVar(value=False)
        self.adv_btn = ttk.Checkbutton(self, text="Advanced", variable=self.adv_open,
                                       command=self._toggle_adv)
        self.adv_btn.pack(anchor="w", pady=(6, 0))
        self.adv_box = ttk.Frame(self, padding=(18, 2, 0, 2))
        self.vars = {}
        self.adv_box.columnconfigure(1, weight=1)
        for i, p in enumerate(P.CALIB_PARAMS):
            lab = ttk.Label(self.adv_box, text=p.label)
            lab.grid(row=i, column=0, sticky="w", pady=2)
            if p.kind is bool:
                var = tk.BooleanVar(value=bool(p.default))
                w = ttk.Checkbutton(self.adv_box, variable=var)
            else:
                var = tk.StringVar(value=str(p.default))
                w = ttk.Entry(self.adv_box, textvariable=var, width=10)
            w.grid(row=i, column=1, sticky="w", padx=8, pady=2)
            Tooltip(lab, p.tip)
            Tooltip(w, p.tip)
            self.vars[p.key] = var

        # ---- pre-check result ----
        self.chk_lab = ttk.Label(self, text="", foreground=GREY,
                                 justify="left", wraplength=700)
        self.chk_lab.pack(anchor="w", pady=(8, 2))

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=6)
        self.b_run = ttk.Button(btns, text="Run calibration", command=self._start)
        self.b_stop = ttk.Button(btns, text="Stop", command=self._stop)
        for b in (self.b_run, self.b_stop):
            b.pack(side="left", padx=(0, 8))
        Tooltip(self.b_run, "About 6 minutes for 25 sets (measured).\n"
                            "The result goes to the rig folder in the calibration set; an existing one is overwritten.")
        Tooltip(self.b_stop, "Calibration has no checkpoint; stopping terminates it.")

        self.msg = ttk.Label(self, text="", foreground=GREY)
        self.msg.pack(anchor="w")

        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True, pady=(6, 0))
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        self.log = tk.Text(wrap, height=10, wrap="none", state="disabled",
                           background="#f7f7f7")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        # same colour meanings as the preprocessing tab: green pass, red fail, blue numbers, grey other
        self.log.tag_configure("good", foreground=GREEN)
        self.log.tag_configure("bad", foreground=RED)
        self.log.tag_configure("key", foreground=BLUE)
        self.log.tag_configure("head", foreground=GREY)
        self.log.tag_configure("title", foreground="black", font=("", 9, "bold"))

        self.path.trace_add("write", lambda *_a: self.refresh_root())
        self._edges_changed()
        self.refresh_root()
        self._after_id = self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------ input
    def _pick(self):
        d = filedialog.askdirectory(title="Choose a calibration set",
                                    initialdir=self.path.get() or os.getcwd())
        if d:
            self.path.set(os.path.normpath(d))

    def _toggle_adv(self):
        if self.adv_open.get():
            self.adv_box.pack(fill="x", after=self.adv_btn)
        else:
            self.adv_box.pack_forget()

    def edge_values(self):
        return {k: v.get().strip() for k, v in self.edges.items()}

    def _edges_changed(self):
        why = CR.edges_problem(self.edge_values())
        self.edge_msg.configure(text=why or "")
        if why is None:
            self.save_settings(calib_edges={
                k: float(v) for k, v in self.edge_values().items()})
        self._sync_buttons()

    def values(self):
        out = {}
        for p in P.CALIB_PARAMS:
            raw = self.vars[p.key].get()
            if p.kind is bool:
                out[p.key] = bool(raw)
            else:
                try:
                    out[p.key] = p.kind(raw)
                except (TypeError, ValueError):
                    out[p.key] = p.default
        return out

    # ------------------------------------------------------------ pre-check
    def refresh_root(self):
        """Called when a calibration set is chosen or the tab is shown. Read-only, so it can run any time."""
        if self.run.state in ("running", "stopping"):
            return
        calib = self.path.get().strip()
        self.check = c = CR.precheck(calib)
        if not c["ok"]:
            color = GREY if not calib else RED
            self.chk_lab.configure(text=c["why"], foreground=color)
        else:
            w, h = c["res"]
            head = f"{len(c['cams'])} cameras  {c['shots']} sets  {w}x{h}"
            if c["warn"]:
                self.chk_lab.configure(text=head + "\n" + "\n".join(c["warn"]),
                                       foreground=AMBER)
            else:
                self.chk_lab.configure(text=head, foreground=GREEN)
            self.save_settings(last_calib=os.path.abspath(calib))
        self._sync_buttons()

    # ------------------------------------------------------------ run
    def _start(self):
        if not self._can_run():
            return
        try:
            self.run.start(self.path.get().strip(), self.edge_values(),
                           self.values())
        except Exception as e:                                # noqa: BLE001
            self.show_error("Could not start calibration", str(e))
            return
        self.msg.configure(text=f"Calibrating - log in {self.run.log_path}",
                           foreground=GREY)
        self._sync_buttons()

    def _stop(self):
        self.run.stop()
        self._sync_buttons()

    def _show_result(self):
        """All acceptance numbers come from rig/calib_report.json (thresholds included, as strings)."""
        rep = CR.read_report(self.run.calib)
        self.log.configure(state="normal")
        if self.log.get("1.0", "end").strip():
            self.log.insert("end", "\n" + "-" * 60 + "\n\n", "head")
        if rep.get("error"):
            self.log.insert("end", rep["error"] + "\n", "bad")
        else:
            self.log.insert("end", "Calibration result\n", "title")
            for ck in rep.get("checks", []):
                self.log.insert("end", f"  {ck['name']}".ljust(20), "head")
                self.log.insert("end", str(ck["value"]),
                                "good" if ck.get("pass") else "bad")
                self.log.insert("end", f"  (threshold {ck['threshold']})\n", "head")
            passed = rep.get("passed")
            self.log.insert("end", "\n  ")
            self.log.insert("end", "PASSED" if passed else "FAILED",
                            "good" if passed else "bad")
            if rep.get("elapsed_s") is not None:
                self.log.insert("end", f"  took {rep['elapsed_s']:.0f} s", "head")
            self.log.insert("end", "\n  rig: ", "head")
            self.log.insert("end", os.path.join(self.run.calib, "rig") + "\n",
                            "key")
        self.log.see("end")
        self.log.configure(state="disabled")

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
            self.msg.configure(text="Calibration finished", foreground=GREY)
            self._show_result()
            self.refresh_root()
        elif state == "stopped":
            self.run.state = "idle"
            self._set_log(self.run.log_tail(8))
            self.msg.configure(text="Stopped", foreground=GREY)
            self.refresh_root()
        elif state == "failed":
            self.run.state = "idle"
            self._set_log(self.run.log_tail(8))
            self.msg.configure(
                text=f"Calibration did not finish normally (exit code {self.run.returncode})",
                foreground=RED)
            self.show_error(f"Calibration failed (exit code {self.run.returncode})",
                            "\n".join(self.run.log_tail(40)))
            self.refresh_root()
        self._after_id = self.after(POLL_MS, self._poll)

    BAD_WORDS = ("failed", "Failed", "Error", "Traceback", "FAIL")

    def _set_log(self, lines):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for line in lines:
            if any(w in line for w in self.BAD_WORDS):
                tag = "bad"
            elif line.lstrip().startswith("["):
                tag = "key"                     # [n/8] step titles
            else:
                tag = "head"
            self.log.insert("end", line + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _can_run(self):
        return (bool(self.check and self.check.get("ok"))
                and CR.edges_problem(self.edge_values()) is None)

    def _sync_buttons(self):
        """Button enabling and disabling happens only here (as in the preprocessing tab)."""
        if not hasattr(self, "b_run"):
            return                              # still being constructed; no buttons yet
        busy = self.run.state in ("running", "stopping")
        self.b_run.state(["!disabled"] if (self._can_run() and not busy)
                         else ["disabled"])
        self.b_stop.state(["!disabled"] if busy else ["disabled"])
        st = ["disabled"] if busy else ["!disabled"]
        for w in (self.path_ent, self.pick_btn, self.adv_btn,
                  *self.edge_box.winfo_children(),
                  *self.adv_box.winfo_children()):
            if isinstance(w, (ttk.Entry, ttk.Button, ttk.Checkbutton)):
                w.state(st)
        self.set_busy(busy)
