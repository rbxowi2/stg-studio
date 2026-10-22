"""Separate window for the live window-center preview.

Shows the JPG written by the training process (see the STG_PREVIEW patch) and
reloads it whenever its modification time changes. It never renders by
itself, which would compete with training for GPU memory.

Controls:
    Fit       scale to the window and keep following its size (default)
    In / Out  fixed zoom; the image can then be dragged with the mouse
The window's X only hides it.
While a video is being rendered the window cannot be closed or hidden
(`busy`); see `hide()`.
"""
import os
import tkinter as tk
from tkinter import ttk

POLL_MS = 1000
TITLE = "Window-center preview"
ZOOM_STEP = 1.25
ZOOM_MIN, ZOOM_MAX = 0.05, 8.0


class PreviewWindow:
    def __init__(self, master):
        self.master = master
        self.path = ""
        self._mtime = 0
        self._src = None            # original image (PIL)
        self._img = None            # image on the canvas (PhotoImage; keep a reference)
        self.zoom = None            # None = fit to window

        self.busy = False           # rendering: no closing, no playback (zoom still works)

        self.win = tk.Toplevel(master)
        self.win.title(TITLE)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.geometry("900x560")

        bar = ttk.Frame(self.win, padding=(6, 6, 6, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="Zoom in", width=8,
                   command=lambda: self._zoom_by(ZOOM_STEP)).pack(side="left")
        ttk.Button(bar, text="Zoom out", width=8,
                   command=lambda: self._zoom_by(1 / ZOOM_STEP)).pack(side="left",
                                                                     padx=6)
        ttk.Button(bar, text="Fit", width=10,
                   command=self._fit).pack(side="left")
        self.zoom_lab = ttk.Label(bar, text="", foreground="#888888")
        self.zoom_lab.pack(side="left", padx=10)

        # Rendering the full video is done by the training process (the model
        # is in its GPU memory); this window only sends the command. While
        # rendering, the button turns red and is disabled, and training waits.
        self.b_render = tk.Button(bar, text="Render video", width=12,
                                  command=self._render)
        self.b_render.pack(side="right")
        self.b_play = tk.Button(bar, text="Play", width=8, command=self._play,
                                state="disabled")
        self.b_play.pack(side="right", padx=6)
        self._render_cb = None       # set by train_tab
        self._video = None           # video path (playback only when present)
        self.video_win = None
        self._after_id = None

        self.canvas = tk.Canvas(self.win, background="#202020",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        # Drag to pan after zooming (scan_mark/scan_dragto is the Canvas's built-in panning)
        self.canvas.bind("<ButtonPress-1>",
                         lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>",
                         lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<MouseWheel>", self._wheel)

        self.win.withdraw()
        self._tick()

    # ------------------------------------------------------------------
    def set_caption(self, text=""):
        """Append what is being shown to the title (combined training: which segment).

        The title only says what is shown now.
        """
        self.win.title(f"{TITLE}  {text}" if text else TITLE)

    def show(self, path=""):
        if path:
            self.path = path
        self._mtime = 0                     # force one reload
        self.win.deiconify()
        self.win.lift()

    def hide(self):
        # Not while rendering: the training process is drawing, and this window
        # is its only progress display. Hiding it would leave several seconds
        # without feedback, inviting other button presses.
        if self.busy:
            return
        self.win.withdraw()

    def _on_close(self):
        """The window's X. While rendering, beep: silently ignoring it looks like a hang."""
        if self.busy:
            self.win.bell()
            return
        self.hide()

    @property
    def visible(self):
        return self.win.state() != "withdrawn"

    # ------------------------------------------------------------ video
    def bind_render(self, cb):
        """train_tab attaches the action that sends the render command."""
        self._render_cb = cb

    def _render(self):
        if self._render_cb is None:
            return
        self._render_cb()
        # Change colour immediately instead of waiting for the next poll
        # (200 ms); a button that does not react invites a second press.
        self.set_render_state(busy=True, can_render=True, video=self._video)

    def set_render_state(self, busy, can_render, video=None):
        """Called by train_tab on every poll. Button state is changed only here."""
        self.busy = bool(busy)
        self.b_render.configure(
            text="Rendering..." if busy else "Render video",
            fg="white" if busy else "black",
            bg="#c02020" if busy else "SystemButtonFace",
            state="disabled" if (busy or not can_render) else "normal")
        if video and video != self._video:
            self._video = video
            if self.video_win is not None and self.video_win.alive:
                self.video_win.reload(video)     # switch to the newly rendered video
        self.b_play.configure(
            state="normal" if (self._video and not busy) else "disabled")

    def _play(self):
        if not self._video:
            return
        from video_win import VideoWindow
        if self.video_win is not None and self.video_win.alive:
            self.video_win.win.lift()
            return
        self.video_win = VideoWindow(self.master, self._video)

    # ------------------------------------------------------------ zoom
    def _fit(self):
        self.zoom = None
        self._redraw()

    def _zoom_by(self, k):
        base = self.zoom or self._fit_scale() or 1.0
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, base * k))
        self._redraw()

    def _wheel(self, e):
        self._zoom_by(ZOOM_STEP if e.delta > 0 else 1 / ZOOM_STEP)

    def _fit_scale(self):
        if self._src is None:
            return None
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        w, h = self._src.size
        return min(cw / w, ch / h)

    # ------------------------------------------------------------ image
    def stop_polling(self):
        """Cancel the scheduled callback before the window is destroyed (see train_tab.stop_polling)."""
        if self._after_id:
            try:
                self.master.after_cancel(self._after_id)
            except Exception:                                 # noqa: BLE001
                pass
        self._after_id = None
        if self.video_win is not None and self.video_win.alive:
            self.video_win.close()

    def _tick(self):
        self._after_id = None
        try:
            if self.visible and self.path and os.path.isfile(self.path):
                m = os.path.getmtime(self.path)
                if m != self._mtime:
                    self._mtime = m
                    self._load()
        except Exception:                                    # noqa: BLE001
            pass                                             # file half-written; try next time
        self._after_id = self.master.after(POLL_MS, self._tick)

    def _load(self):
        # tkinter only reads GIF/PNG; JPG needs Pillow. Without Pillow say so
        # instead of silently showing nothing.
        try:
            from PIL import Image
        except ImportError:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor="nw", fill="#dddddd",
                                    text="Pillow is required: pip install pillow")
            return
        im = Image.open(self.path)
        im.load()                                # read fully so the file is not locked
        self._src = im
        self._redraw()

    def _redraw(self):
        from PIL import Image, ImageTk
        self.canvas.delete("all")
        if self._src is None:
            self.canvas.create_text(10, 10, anchor="nw", fill="#dddddd",
                                    text="No image yet")
            self.zoom_lab.configure(text="")
            return
        scale = self.zoom or self._fit_scale() or 1.0
        w, h = self._src.size
        sw, sh = max(int(w * scale), 1), max(int(h * scale), 1)
        # LANCZOS when shrinking (quality), NEAREST when enlarging: the point
        # is to see noise and stretched Gaussians, which interpolation hides.
        rs = Image.LANCZOS if scale < 1 else Image.NEAREST
        self._img = ImageTk.PhotoImage(self._src.resize((sw, sh), rs))
        self.canvas.create_image(0, 0, anchor="nw", image=self._img)
        self.canvas.configure(scrollregion=(0, 0, sw, sh))
        self.zoom_lab.configure(
            text=f"{scale * 100:.0f}%  {w}x{h}"
                 + ("" if self.zoom else " (fit)"))
