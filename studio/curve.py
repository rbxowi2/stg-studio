"""Loss curve drawn on a Canvas (no matplotlib, to keep the package small).

Two lines, two Y axes:

    left axis   training loss (log scale)
    right axis  largest Gaussian scale. It rises together with visible
                degradation: once the Gaussian count stops growing it can
                climb steeply while colour splatter appears in the image,
                which the loss curve does not show.

Both axes have ticks and values, and the current value is marked at its
actual position on the axis (with a coloured label), so where the run is and
where it is heading can be read at a glance.
"""
import math
import tkinter as tk

PAD_L, PAD_R, PAD_T, PAD_B = 62, 62, 18, 26
LOSS_COLOR = "#1f6fb4"
SCALE_COLOR = "#c05000"
GROW_COLOR = "#2e8b57"      # densification count (third line, no axis of its own)
MARK_COLOR = "#8040a0"      # dashed vertical line at a resume point
GRID = "#e8e8e8"
AXIS = "#b0b0b0"
TICK = 4                    # tick length (px)
YTICKS, XTICKS = 4, 5       # desired number of intervals (adjusted to the data)
FONT = ("", 8)


def _nice_ticks(lo, hi, n):
    """About n "nice" tick values (1/2/5 x 10^k)."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + step * 1e-9:
        out.append(v)
        v += step
    return out or [lo, hi]


def _fmt(v):
    """Value labels: thousands separators for large values, enough digits for small ones."""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


class Curve(tk.Canvas):
    def __init__(self, master, width=560, height=200, **kw):
        super().__init__(master, width=width, height=height,
                         background="white", highlightthickness=1,
                         highlightbackground="#c0c0c0", **kw)
        self.points = []
        self.desicnt = None          # densification limit; marks the full-scale position
        self.marks = []              # iterations of resume points, one vertical line each
        self.bind("<Configure>", lambda e: self.redraw())

    def set_points(self, points, desicnt=None, marks=None):
        self.points = points
        if desicnt:
            self.desicnt = desicnt
        if marks is not None:
            self.marks = list(marks)
        self.redraw()

    # ------------------------------------------------------------------
    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 120 or h < 90:
            return
        x0, x1 = PAD_L, w - PAD_R
        y0, y1 = PAD_T, h - PAD_B

        pts = [p for p in self.points if p.get("loss", 0) > 0]
        if len(pts) < 2:
            self.create_rectangle(x0, y0, x1, y1, outline=AXIS)
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="Waiting for the first data...",
                             fill="#888888", font=FONT)
            return

        it = [p["iter"] for p in pts]
        loss = [p["loss"] for p in pts]
        scale = [p.get("max_scale", 0.0) for p in pts]

        # loss on a log scale (a linear scale shows nothing once it drops by orders of magnitude)
        lgl = [math.log10(v) for v in loss]
        self._x_axis(x0, x1, y0, y1, it)
        self._y_axis(x0, x1, y0, y1, lgl, LOSS_COLOR, left=True, log=True)
        self._y_axis(x0, x1, y0, y1, scale, SCALE_COLOR, left=False, log=False)
        self._line(x0, x1, y0, y1, it, lgl, LOSS_COLOR)
        self._line(x0, x1, y0, y1, it, scale, SCALE_COLOR)
        # Densification count: a third line with no axis of its own. What
        # matters is how far it is from the limit, so 0..desicnt is the full
        # height; a flat line means densification has stopped.
        grow = [p.get("densify_n") for p in pts]
        if all(g is not None for g in grow):
            top = self.desicnt or max(max(grow), 1)
            self._grow_line(x0, x1, y0, y1, it, grow, top)
        # Resume point: a vertical line. The two sides come from different
        # runs, so a step in the loss there is expected (a resumed run is
        # never bit-identical to an uninterrupted one); without the line it
        # would look like a training problem.
        self._marks(x0, x1, y0, y1, it)
        self.create_rectangle(x0, y0, x1, y1, outline=AXIS)

        self.create_text(x0 + 2, y0 - 9, anchor="w", text="Loss (log)",
                         fill=LOSS_COLOR, font=FONT)
        self.create_text(x1 - 2, y0 - 9, anchor="e", text="Largest Gaussian scale",
                         fill=SCALE_COLOR, font=FONT)
        self.create_text((x0 + x1) / 2, h - 8, text="Iteration", fill="#888888",
                         font=FONT)

    def _marks(self, x0, x1, y0, y1, it):
        lo, hi = self._span(it)
        for v in self.marks:
            if not (lo <= v <= hi):
                continue
            x = x0 + (v - lo) / (hi - lo) * (x1 - x0)
            self.create_line(x, y0, x, y1, fill=MARK_COLOR, dash=(2, 3))
            self.create_text(x + 3, y0 + 2, anchor="nw", text="resumed",
                             fill=MARK_COLOR, font=FONT)

    # ------------------------------------------------------------------
    @staticmethod
    def _span(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:                 # all equal: add some room to avoid dividing by zero
            pad = abs(hi) * 0.05 or 1.0
            lo, hi = lo - pad, hi + pad
        return lo, hi

    def _x_axis(self, x0, x1, y0, y1, it):
        lo, hi = self._span(it)
        for v in _nice_ticks(lo, hi, XTICKS):
            x = x0 + (v - lo) / (hi - lo) * (x1 - x0)
            self.create_line(x, y0, x, y1, fill=GRID)
            self.create_line(x, y1, x, y1 + TICK, fill=AXIS)
            self.create_text(x, y1 + TICK + 2, anchor="n", text=_fmt(v),
                             fill="#666666", font=FONT)

    def _y_axis(self, x0, x1, y0, y1, vals, color, left, log):
        """One Y axis per side: ticks and values, then the current value at its real position."""
        lo, hi = self._span(vals)
        for v in _nice_ticks(lo, hi, YTICKS):
            y = y1 - (v - lo) / (hi - lo) * (y1 - y0)
            if left:
                self.create_line(x0 - TICK, y, x1, y, fill=GRID)
                self.create_line(x0 - TICK, y, x0, y, fill=AXIS)
                self.create_text(x0 - TICK - 2, y, anchor="e",
                                 text=_fmt(10 ** v if log else v),
                                 fill=color, font=FONT)
            else:
                self.create_line(x1, y, x1 + TICK, y, fill=AXIS)
                self.create_text(x1 + TICK + 2, y, anchor="w", text=_fmt(v),
                                 fill=color, font=FONT)

        # Minimum: the best value reached so far, which matters more than the
        # current one because the loss fluctuates. Only the loss (left axis)
        # gets it; the largest scale only grows, so its minimum is the start.
        # Drawn even when it overlaps the current value.
        mn = min(vals)
        if left:
            y = y1 - (mn - lo) / (hi - lo) * (y1 - y0)
            self.create_line(x0, y, x1, y, fill=color, dash=(2, 3))
            txt = f"min {_fmt(10 ** mn if log else mn)}"
            # inside the plot on a white background, so it does not collide with the tick labels
            x, anchor = (x0 + 6, "w") if left else (x1 - 6, "e")
            t = self.create_text(x, y - 7, anchor=anchor, text=txt,
                                 fill=color, font=FONT)
            bx = self.bbox(t)
            self.create_rectangle(bx[0] - 2, bx[1], bx[2] + 2, bx[3],
                                  fill="white", outline="")
            self.tag_raise(t)

        # Current value at its real height on the axis, over the tick labels
        cur = vals[-1]
        y = y1 - (cur - lo) / (hi - lo) * (y1 - y0)
        txt = _fmt(10 ** cur if log else cur)
        if left:
            x, anchor = x0 - TICK - 2, "e"
        else:
            x, anchor = x1 + TICK + 2, "w"
        t = self.create_text(x, y, anchor=anchor, text=txt, fill="white",
                             font=("", 8, "bold"))
        bx = self.bbox(t)
        self.create_rectangle(bx[0] - 2, bx[1] - 1, bx[2] + 2, bx[3] + 1,
                              fill=color, outline=color)
        self.tag_raise(t)

    def _grow_line(self, x0, x1, y0, y1, xs, ys, top):
        """Densification count as a step line (it grows one step at a time), labelled N/limit."""
        xlo, xhi = self._span(xs)
        flat = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            px = x0 + (x - xlo) / (xhi - xlo) * (x1 - x0)
            py = y1 - min(y / top, 1.0) * (y1 - y0)
            if i:                                   # horizontal first, then up = a step
                flat += [px, flat[-1]]
            flat += [px, py]
        self.create_line(*flat, fill=GROW_COLOR, width=1, dash=(3, 2))
        self.create_text(flat[-2] - 4, flat[-1] - 7, anchor="e",
                         text=f"densify {ys[-1]}/{top}", fill=GROW_COLOR,
                         font=("", 8, "bold"))

    def _line(self, x0, x1, y0, y1, xs, ys, color):
        xlo, xhi = self._span(xs)
        lo, hi = self._span(ys)
        flat = []
        for x, y in zip(xs, ys):
            flat.append(x0 + (x - xlo) / (xhi - xlo) * (x1 - x0))
            flat.append(y1 - (y - lo) / (hi - lo) * (y1 - y0))
        self.create_line(*flat, fill=color, width=2)
        self.create_oval(flat[-2] - 3, flat[-1] - 3, flat[-2] + 3, flat[-1] + 3,
                         fill=color, outline="white")      # current point
