"""The rendered window-center video: looping playback and "save as".

The image fits the window (no zoom or drag; that is the preview window's job).
The video is rendered by the training process (the render command of the
STG_CHECKPOINT patch); this module only reads the file.
"""
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

TITLE = "Window-center video"


class VideoWindow:
    def __init__(self, master, path):
        self.master = master
        self.path = path
        self.cap = None
        self.fps = 30.0
        self.frames = 0
        self._img = None
        self._after = None

        self.win = tk.Toplevel(master)
        self.win.title(TITLE)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.geometry("960x600")

        bar = ttk.Frame(self.win, padding=(6, 6, 6, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="Save as...", command=self._save).pack(side="left")
        self.info = ttk.Label(bar, text="", foreground="#888888")
        self.info.pack(side="left", padx=10)

        self.canvas = tk.Canvas(self.win, background="#202020",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)

        self._open()
        self._tick()

    # ------------------------------------------------------------------
    def _open(self):
        import cv2
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info.configure(
            text=f"{self.frames} frames  {self.fps:.0f} fps  "
                 f"{self.frames / max(self.fps, 1):.1f} s")

    def reload(self, path=None):
        """Call after the video has been rendered again."""
        if path:
            self.path = path
        if self.cap is not None:
            self.cap.release()
        self._open()

    def close(self):
        if self._after is not None:
            self.master.after_cancel(self._after)
            self._after = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.win.destroy()

    @property
    def alive(self):
        return bool(self.win.winfo_exists())

    # ------------------------------------------------------------------
    def _tick(self):
        import cv2
        delay = int(1000 / max(self.fps, 1))
        try:
            ok, frame = self.cap.read()
            if not ok:                                   # loop back to the start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if ok:
                self._draw(frame)
        except Exception:                                # noqa: BLE001
            pass
        self._after = self.master.after(delay, self._tick)

    def _draw(self, frame):
        import cv2
        from PIL import Image, ImageTk
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        h, w = frame.shape[:2]
        s = min(cw / w, ch / h)                          # fit the window
        sw, sh = max(int(w * s), 1), max(int(h * s), 1)
        # cv2 resizing is faster than PIL, which matters for frame-by-frame playback
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        self._img = ImageTk.PhotoImage(
            Image.fromarray(small[:, :, ::-1]))          # BGR -> RGB
        self.canvas.delete("all")
        self.canvas.create_image((cw - sw) // 2, (ch - sh) // 2, anchor="nw",
                                 image=self._img)

    def _save(self):
        p = filedialog.asksaveasfilename(
            parent=self.win, title="Save window-center video", defaultextension=".mp4",
            initialfile="window_view.mp4",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")])
        if not p:
            return
        try:
            shutil.copyfile(self.path, p)
        except OSError as e:
            messagebox.showerror("Could not save", str(e), parent=self.win)
            return
        self.info.configure(text=f"Saved to {os.path.basename(p)}")
