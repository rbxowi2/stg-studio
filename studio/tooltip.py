"""Hover tooltips (tkinter has none built in).

Usage: `Tooltip(widget, "text")`. Appears 0.5 s after the pointer enters,
closes when it leaves.
"""
import tkinter as tk
from tkinter import ttk

DELAY_MS = 500
BG = "#ffffe1"          # the pale yellow of native Windows tooltips
WRAP_PX = 460


class Tooltip:
    def __init__(self, widget, text, delay=DELAY_MS):
        self.widget, self.text, self.delay = widget, text, delay
        self._after = None
        self._win = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        # close on click: a tooltip covering a drop-down list gets in the way
        widget.bind("<ButtonPress>", self._leave, add="+")

    def _enter(self, _e=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _leave(self, _e=None):
        self._cancel()
        if self._win is not None:
            self._win.destroy()
            self._win = None

    def _cancel(self):
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self):
        if self._win is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._win = w = tk.Toplevel(self.widget)
        w.wm_overrideredirect(True)          # no border, no title bar
        w.wm_geometry(f"+{x}+{y}")
        ttk.Label(w, text=self.text, background=BG, relief="solid",
                  borderwidth=1, padding=(8, 5), justify="left",
                  wraplength=WRAP_PX).pack()
