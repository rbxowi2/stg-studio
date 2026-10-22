"""STG studio - main window, custom tab bar, path pickers in each tab.

Shooting root: the preprocessing, training and combined training tabs each
have a picker, all bound to one variable (the three tabs work on the same
shooting root). The calibration tab chooses a calibration set, with a
variable of its own.

Run:
    .venv/Scripts/pythonw.exe studio/stg_studio.py
Use the repository's virtual environment: training subprocesses use the
interpreter given in tools/train_cmd.py, and the preview needs Pillow, which
is installed there as well.
"""
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from calib_tab import CalibTab                                # noqa: E402
from combo_tab import ComboTab                                # noqa: E402
from prep_tab import PrepTab                                  # noqa: E402
import orphans                                               # noqa: E402
from version import STUDIO_VERSION                            # noqa: E402
from train_tab import LIVE as TRAIN_LIVE, TrainTab            # noqa: E402

TITLE = "STG studio"
SETTINGS = "settings.json"
GREY = "#888888"
RED = "#c02020"


ICON_RGB = (0x2E, 0x7D, 0xD2)      # colour of the dot (R,G,B as stored in BGRA)


def _dot_ico():
    """Create a 32x32 icon (transparent background, one solid dot); returns its path.

    Built from raw bytes without Pillow.

    Windows always draws something in the title bar: a 1x1 transparent
    PhotoImage, a fully transparent 32-bit .ico and an invisible 1-bit .ico
    all end up as a black square. So the icon is a minimal mark, a dot, not a logo.

    The player's launcher contains an identical copy on purpose; the two
    applications do not depend on each other. Change both together.
    """
    import struct
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "stg_dot.ico")
    if os.path.isfile(path):
        return path
    n, r = 32, 11
    b, g, rr = ICON_RGB[2], ICON_RGB[1], ICON_RGB[0]
    rows = []
    for y in range(n - 1, -1, -1):                 # DIB rows go bottom-up
        px = bytearray()
        for x in range(n):
            d = (x - (n - 1) / 2) ** 2 + (y - (n - 1) / 2) ** 2
            inside = d <= r * r
            px += bytes((b, g, rr, 255)) if inside else b"\x00\x00\x00\x00"
        rows.append(bytes(px))
    xor = b"".join(rows)
    and_mask = b"\x00" * (n * 4)                   # 32-bit images use alpha; all zero here
    hdr = struct.pack("<IiiHHIIiiII", 40, n, n * 2, 1, 32, 0,
                      len(xor) + len(and_mask), 0, 0, 0, 0)
    img = hdr + xor + and_mask
    ico = struct.pack("<HHH", 0, 1, 1) + struct.pack(
        "<BBBBHHII", n, n, 0, 0, 1, 32, len(img), 22) + img
    with open(path, "wb") as f:
        f.write(ico)
    return path


def _no_icon(win):
    """Replace the Tk feather icon (no logo in the title bar). Failure is ignored; the icon is not a feature."""
    try:
        win.iconbitmap(default=_dot_ico())
    except Exception:                                         # noqa: BLE001
        pass


def settings_path():
    return os.path.join(HERE, SETTINGS)


def load_settings():
    """The whole settings.json. Unreadable -> empty dict (not an error, no message)."""
    import json
    try:
        with open(settings_path(), encoding="utf-8") as f:
            d = json.load(f)
    except Exception:                                         # noqa: BLE001
        return {}
    return d if isinstance(d, dict) else {}


def save_settings(**fields):
    """Read -> merge -> write.

    Never overwrite the whole file: it holds the shooting root, the
    calibration set and the six distances, and writing one field alone would
    erase the others. Write failures are ignored silently; a convenience
    feature must not get in the way.
    """
    import json
    d = load_settings()
    d.update(fields)
    try:
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:                                         # noqa: BLE001
        pass


def load_last_root():
    """The last shooting root; None if unreadable or no longer present."""
    p = load_settings().get("last_root")
    return p if p and os.path.isdir(p) else None


def save_last_root(path):
    save_settings(last_root=os.path.abspath(path))


BUILD_INFO = "build_info.txt"


def build_tag():
    """The version text in the lower right of the main window.

    With `build_info.txt` next to the program: `v1.0.0 (a3f9c21)` from that file.
    In a git checkout: `v1.0.0 (a3f9c21)` from git.
    Neither: `v1.0.0` only.

    The hash is the reliable identity (a version bump can be forgotten, a
    hash cannot), so both are shown. git is called with `CREATE_NO_WINDOW`:
    the studio runs under pythonw, and a console window would flash otherwise.
    """
    tag = f"v{STUDIO_VERSION}"
    info = os.path.join(HERE, BUILD_INFO)
    try:
        for line in open(info, encoding="utf-8"):
            if line.startswith("git "):
                return f"{tag} ({line.split(None, 1)[1].strip()})"
    except OSError:
        pass
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=HERE, capture_output=True, text=True,
                             timeout=5,
                             creationflags=getattr(subprocess,
                                                   "CREATE_NO_WINDOW", 0))
        if out.returncode == 0 and out.stdout.strip():
            return f"{tag} ({out.stdout.strip()})"
    except (OSError, subprocess.SubprocessError):
        pass
    return tag                      # no git and no build_info: version only


class Studio:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(TITLE)
        _no_icon(self.root)          # default=... also applies to every window opened later
        try:
            ttk.Style().theme_use("vista")
        except Exception:                                     # noqa: BLE001
            pass
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.minsize(760, 560)
        # default height chosen to fit a 1080-pixel screen (tallest with the advanced section open)
        self.root.geometry("800x860")

        self.path = tk.StringVar(value=load_last_root() or "")
        self.pickers = []               # path pickers of all tabs (locked together while busy)

        # Custom tab bar: preprocessing / training / combined training on the
        # left, calibration on the right. ttk.Notebook cannot split tabs left
        # and right, so it is built from Toolbutton-style Radiobuttons (the
        # pressed one stays down) and stacked pages.
        bar = ttk.Frame(self.root, padding=(8, 8, 8, 0))
        bar.pack(fill="x")
        ttk.Separator(self.root).pack(fill="x", pady=(6, 0))
        self.stack = stack = ttk.Frame(self.root)
        stack.pack(fill="both", expand=True, padx=8, pady=8)
        self.cur = tk.StringVar(value="train")
        self.tab_btn = {}
        self.pages = {}
        # tabs call _set_busy during construction, so these two must exist first
        self.tabs = []
        self.busy_tab = None            # the tab that is busy (it is not locked itself)
        for key, text, side in (("prep", "Preprocess", "left"),
                                ("train", "Train", "left"),
                                ("combo", "Combined training", "left"),
                                ("calib", "Calibrate", "right")):
            b = ttk.Radiobutton(bar, text=text, value=key, variable=self.cur,
                                style="Toolbutton", command=self._show_tab)
            b.pack(side=side, padx=(0, 2) if side == "left" else 0)
            self.tab_btn[key] = b
            holder = ttk.Frame(stack)
            holder.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.pages[key] = holder

        # Calibration tab: it chooses a calibration set, not a shooting root, so it has its own path
        self.calib = None
        self.calib = CalibTab(self.pages["calib"], load_settings(), save_settings,
                              self._show_error,
                              lambda b: self._set_busy(b, "calib"))
        self.calib.pack(fill="both", expand=True)
        self.tabs.append(self.calib)

        # The lambdas must not refer to self.prep / self.train: tabs call
        # set_busy while they are being constructed, before the attribute is
        # assigned (AttributeError at startup). They pass a name instead.
        self.prep = None
        self.train = None
        self.combo = None
        # Preprocessing / training / combined training share one shooting root:
        # each tab has a picker, all bound to self.path.
        for key in ("prep", "train", "combo"):
            self._root_picker(self.pages[key])
        self.prep = PrepTab(self.pages["prep"], self.path.get, self._show_error,
                            lambda b: self._set_busy(b, "prep"))
        self.prep.pack(fill="both", expand=True)
        self.tabs.append(self.prep)

        self.train = TrainTab(self.pages["train"], self.path.get,
                              self._show_error,
                              lambda b: self._set_busy(b, "train"))
        self.train.pack(fill="both", expand=True)
        self.tabs.append(self.train)

        # Combined training: separate from the training tab. Only one tab runs
        # at a time; locking goes through set_busy like the others.
        self.combo = ComboTab(self.pages["combo"], self.path.get,
                              self._show_error,
                              lambda b: self._set_busy(b, "combo"))
        self.combo.pack(fill="both", expand=True)
        self.tabs.append(self.combo)
        # Start on the training tab. Only raise it here; the startup check is scheduled below
        self.pages["train"].tkraise()

        # Unattended training runs (left over from a crash or a closed
        # studio). Only listed, never killed automatically.
        self.orphan_bar = ttk.Frame(self.root)
        self.orphan_lab = ttk.Label(self.orphan_bar, text="", foreground=RED,
                                    wraplength=620, justify="left")
        self.orphan_lab.pack(side="left", padx=(12, 8))
        self.orphan_btn = ttk.Button(self.orphan_bar, text="Ask them to stop",
                                     command=self._release_orphans)
        self.orphan_btn.pack(side="left")
        self._orphans = []

        # Version in the lower right: screenshots then carry the version automatically
        ttk.Label(self.root, text=build_tag(), foreground=GREY).pack(
            anchor="e", padx=12, pady=(0, 6))

        self.path.trace_add("write", lambda *_a: self._root_changed())
        self._scan_orphans()
        self.root.after(3000, self._orphan_tick)
        if self.path.get():
            self.train.refresh_root()
            # the preprocessing pre-check waits until the window is drawn (it takes seconds)
            self.root.after(300, self.prep.refresh_root)

    def _scan_orphans(self):
        """List training runs under this root that are alive but unattended. They are not touched."""
        root = self.path.get()
        # Processes started by this window are not orphans, and several tabs
        # start processes. A tab missing from the list would show its own run
        # as an orphan, and "Ask them to stop" would stop it. The list is
        # therefore built from self.tabs, so a new tab cannot be forgotten.
        mine = {t.run.proc.pid for t in self.tabs
                if getattr(t, "run", None) is not None
                and getattr(t.run, "proc", None) is not None
                and t.run.proc.poll() is None}
        # As a safeguard, the control folders of running tabs are excluded
        # too (even if a PID was missed).
        live_ctrl = {os.path.normcase(os.path.abspath(t.run.ctrl))
                     for t in self.tabs
                     if getattr(t, "run", None) is not None
                     and getattr(t.run, "ctrl", "")
                     and getattr(t.run, "proc", None) is not None
                     and t.run.proc.poll() is None}
        found = orphans.scan(root, skip=mine) if root else []
        self._orphans = [o for o in found
                         if os.path.normcase(os.path.abspath(o["ctrl"]))
                         not in live_ctrl]
        if not self._orphans:
            self.orphan_bar.pack_forget()
            return
        when = ", ".join(
            f"{o['name']} ({time.strftime('%H:%M', time.localtime(o['started']))}"
            f", PID {o['pid']})" for o in self._orphans)
        self.orphan_lab.configure(
            text=f"{len(self._orphans)} training run(s) are still running but were not started by this window: "
                 f"{when} - they hold GPU memory.")
        self.orphan_bar.pack(fill="x", pady=(0, 4))

    def _orphan_tick(self):
        """Recomputed every 3 s from the current state; the bar disappears once they end by themselves."""
        self._scan_orphans()
        self.root.after(3000, self._orphan_tick)

    def _release_orphans(self):
        """Ask them to finish by themselves (stop.flag). No force-kill: they may be writing files."""
        n = sum(1 for o in self._orphans if orphans.release(o))
        self.orphan_lab.configure(
            text=f"Asked {n} training run(s) to stop; waiting for them to finish (they will disappear from here).")

    def _quit(self):
        """The window's X.

        Closing during training would leave the subprocess holding the GPU
        with nothing left on screen to stop it, so training is stopped
        cleanly first. Stopping does not save. A finished-and-waiting run
        also holds GPU memory and is stopped as well.
        """
        # training and combined training each have a Runner; check both
        for tab in (self.train, self.combo):
            if tab.run.state in TRAIN_LIVE:
                self._quit_training(tab)
                return
        if self.calib.run.state in ("running", "stopping"):
            if not messagebox.askokcancel(
                    "Calibration is running",
                    "Calibration must be stopped before closing.\n\n"
                    "Calibration has no checkpoint; stopping interrupts it. Continue?"):
                return
            self.calib.run.stop()
            self._wait_stop()
            return
        if self.prep.run.state in ("running", "stopping"):
            # preprocessing has no checkpoint; stopping terminates it and keeps the outputs
            if not messagebox.askokcancel(
                    "Preprocessing is running",
                    "Preprocessing must be stopped before closing.\n\n"
                    "Preprocessing has no checkpoint; stopping interrupts it,\n"
                    "and what it produced so far is kept. Continue?"):
                return
            self.prep.run.stop()
            self._wait_stop()
            return
        self._destroy()

    def _quit_training(self, tab):
        """Closing while training (or combined training) is running."""
        if not messagebox.askokcancel(
                "Training is running",
                "Training must be stopped before closing.\n\n"
                "Stopping does not save; unsaved progress is lost.\n"
                "Continue?"):
            return
        tab.request_stop()
        tab.msg.configure(text="Stopping... (the window closes once GPU memory is freed)")
        self._wait_stop()

    def _wait_stop(self):
        """Close the window only after the subprocesses have really ended."""
        if (self.train.run.poll()[0] in TRAIN_LIVE + ("stopping",)
                or self.combo.run.poll()[0] in TRAIN_LIVE + ("stopping",)
                or self.prep.run.poll() in ("running", "stopping")
                or self.calib.run.poll() in ("running", "stopping")):
            self.root.after(200, self._wait_stop)
            return
        self._destroy()

    def _destroy(self):
        # cancel scheduled callbacks first, then destroy (see train_tab.stop_polling)
        self.train.stop_polling()
        self.combo.stop_polling()
        self.prep.stop_polling()
        self.calib.stop_polling()
        self.root.destroy()

    def _set_busy(self, busy, who=None):
        """While one tab is busy, lock the path pickers and every other tab.

        Changing the root would make the running job and the displayed one
        differ, and the tabs work on the same shooting root's files; two jobs
        at once could corrupt each other's output.

        who = "prep" / "train" / ... (a name, because the tab object may not exist yet).
        """
        if busy:
            self.busy_tab = who
        elif self.busy_tab == who:
            self.busy_tab = None
        locked = self.busy_tab is not None
        st = ["disabled"] if locked else ["!disabled"]
        for w in self.pickers:
            w.state(st)
        for key, btn in self.tab_btn.items():
            btn.state(["!disabled"] if (not locked or key == self.busy_tab)
                      else ["disabled"])

    def _root_picker(self, parent):
        """The "Shooting root" row. Bound to self.path, so every tab shows the same value."""
        row = ttk.Frame(parent, padding=(10, 10, 10, 0))
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="Shooting root").grid(row=0, column=0, sticky="w")
        ent = ttk.Entry(row, textvariable=self.path)
        ent.grid(row=0, column=1, sticky="we", padx=8)
        btn = ttk.Button(row, text="Browse...", command=self._pick)
        btn.grid(row=0, column=2)
        self.pickers += [ent, btn]

    def _show_tab(self):
        """A tab button was pressed: raise that page, then refresh it."""
        self.pages[self.cur.get()].tkraise()
        self._tab_changed()

    def _pick(self):
        d = filedialog.askdirectory(title="Choose a shooting root",
                                    initialdir=self.path.get() or os.getcwd())
        if d:
            self.path.set(d)

    def _tab_changed(self, _e=None):
        """Refresh the tab that was switched to.

        Every refresh_root() can be called repeatedly: training only reads
        window.json, preprocessing runs a read-only pre-check. Nothing is
        refreshed while a job runs (tabs cannot be switched then anyway).
        """
        if self.busy_tab is not None:
            return
        cur = getattr(self, self.cur.get(), None)
        if cur is self.train:
            self.train.refresh_root()
        elif cur is self.combo:
            self.combo.refresh_root()
        elif cur is self.prep:
            self.prep.refresh_root()
        elif cur is self.calib:
            self.calib.refresh_root()

    def _root_changed(self):
        p = self.path.get()
        if p and os.path.isdir(p):
            save_last_root(p)
            self.train.refresh_root()
            self.combo.refresh_root()
            self.prep.refresh_root()      # the preprocessing pre-check follows the new root

    def _show_error(self, title, body):
        """An error window whose text can be selected and copied.

        messagebox text cannot be selected or copied, and copying the message
        is usually the first thing needed after an error.
        """
        w = tk.Toplevel(self.root)
        w.title(title)
        w.transient(self.root)
        fr = ttk.Frame(w, padding=10)
        fr.pack(fill="both", expand=True)
        txt = tk.Text(fr, width=100, height=22, wrap="none")
        ys = ttk.Scrollbar(fr, orient="vertical", command=txt.yview)
        xs = ttk.Scrollbar(fr, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="we")
        fr.rowconfigure(0, weight=1)
        fr.columnconfigure(0, weight=1)
        txt.insert("1.0", body)
        txt.see("end")

        bar = ttk.Frame(fr)
        bar.grid(row=2, column=0, columnspan=2, sticky="we", pady=(8, 0))

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(body)
            copied.configure(text="Copied to the clipboard")

        ttk.Button(bar, text="Copy all", command=copy).pack(side="left")
        copied = ttk.Label(bar, text="", foreground=GREY)
        copied.pack(side="left", padx=10)
        ttk.Button(bar, text="Close", command=w.destroy).pack(side="right")
        txt.bind("<Key>", lambda e: "break" if e.state & 4 == 0 else None)
        w.grab_set()

    def run(self):
        self.root.mainloop()


def main():
    """Entry point.

    Under pythonw (the shortcut) there is no console, so an error at startup
    would show nothing at all and look like a broken shortcut. Exceptions are
    therefore caught, written to a log next to the program, and shown in a
    minimal window whose text can be copied.
    """
    try:
        Studio().run()
    except Exception:                                         # noqa: BLE001
        import traceback
        body = traceback.format_exc()
        try:
            with open(os.path.join(HERE, "startup_error.log"), "w",
                      encoding="utf-8") as f:
                f.write(body)
        except OSError:
            pass
        try:
            r = tk.Tk()
            r.title("STG studio failed to start")
            t = tk.Text(r, width=100, height=24, wrap="none")
            t.pack(fill="both", expand=True)
            t.insert("1.0", body)
            ttk.Button(r, text="Close", command=r.destroy).pack(pady=6)
            r.mainloop()
        except Exception:                                     # noqa: BLE001
            raise                                             # no window possible: re-raise as is


if __name__ == "__main__":
    main()
