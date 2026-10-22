"""Studio: freezing during video rendering, and whether the status line gets stuck.

**Why this check exists**

During the dozen seconds of rendering the window-centre video, the training
process is drawing frames. Pressing pause / save / stop then would make the
command compete with `_preview_video()` for the same process. The assumption
that "commands queue up" was never tested, so everything is blocked instead.

A related bug: the status line's "Rendering video..." was written only when
rendering started and nobody wrote it back afterwards. While training runs,
the next JSONL record overwrote it so it went unnoticed, but **while paused
no new data arrives and it stayed stuck forever**.

Both are hard to catch by eye every time (a real training run and a real
render are needed), so this check needs no GPU and no dataset: it drives
the two tkinter components directly.

    python tools/check_studio_buttons.py

**Needs a desktop** (tkinter windows are created, all hidden). Skipped without a display.
ttk widgets must be queried with `instate(["disabled"])` - `w["state"]` is the
   old-style option and does not hold the flags set by `w.state([...])`.
"""
import os
import sys
import tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "studio"))

FAILED = []


def chk(name, got, want):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {got!r}"
          + ("" if ok else f" (expected {want!r})"))


def check_preview_window():
    """Preview window: while rendering, no playing, no collapsing, no closing."""
    from preview_win import PreviewWindow

    root = tk.Tk()
    root.withdraw()
    try:
        pv = PreviewWindow(root)
        pv.show("")

        print("\n[Preview window / idle]")
        pv.set_render_state(busy=False, can_render=True, video="x.mp4")
        chk("play enabled", str(pv.b_play["state"]), "normal")
        pv.hide()
        chk("can be hidden", pv.visible, False)
        pv.show("")

        print("[Preview window / rendering]")
        pv.set_render_state(busy=True, can_render=True, video="x.mp4")
        chk("render disabled", str(pv.b_render["state"]), "disabled")
        chk("play disabled", str(pv.b_play["state"]), "disabled")
        pv.hide()
        chk("cannot be hidden", pv.visible, True)
        pv._on_close()
        chk("X does not close it", pv.visible, True)

        print("[Preview window / rendering finished]")
        pv.set_render_state(busy=False, can_render=True, video="x.mp4")
        chk("play enabled again", str(pv.b_play["state"]), "normal")
        pv.hide()
        chk("can be hidden again", pv.visible, False)
        pv.stop_polling()
    finally:
        root.destroy()


def check_train_tab():
    """Train tab: freezing of the four entry points, and the status line recomputed every time."""
    from train_tab import TrainTab

    root = tk.Tk()
    root.withdraw()
    try:
        busy = []          # the main window's lock (shooting root + other tabs)
        tab = TrainTab(root, get_root=lambda: "", show_error=lambda *a: None,
                       set_busy=busy.append)
        # pretend the dataset was detected and a name entered - this check does not touch the disk
        tab.info = {"dataset": os.path.join(ROOT, "_nonexistent", "stg1", "x")}
        tab.compat = None
        tab.name.set("check_out")
        run = tab.run
        run.points = [{"iter": 1000, "loss": 0.05, "n_gauss": 12345,
                       "max_scale": 1.2, "densify_n": 3, "elapsed_s": 60.0}]

        def set_rendering(flag):
            # rendering is a Runner property; the test replaces the class attribute
            type(run).rendering = property(lambda self, f=flag: f)

        for st in ("running", "paused"):
            for rendering in (False, True):
                run.state = st
                set_rendering(rendering)
                tab._sync_buttons()
                tab._show_status()
                print(f"\n[Train tab / {st}{' + rendering' if rendering else ''}]")
                for label, w in (("pause/continue", tab.b_pause),
                                 ("save checkpoint", tab.b_save),
                                 ("stop", tab.b_stop)):
                    chk(f"{label} disabled", w.instate(["disabled"]), rendering)
                text = tab.msg["text"]
                chk("status shows rendering", text.startswith("Rendering video"), rendering)
                if not rendering:
                    chk("pause is indicated", text.startswith("Paused"), st == "paused")

        # Finished without unloading (STG_HOLD): the process still holds GPU
        # memory and waits for commands. finished is a **real state**, not
        # "running plus a holding flag".
        print("\n[Train tab / finished but still holding GPU memory]")
        set_rendering(False)
        run.state = "finished"
        tab._sync_buttons()
        tab._show_status()
        chk("pause/continue disabled", tab.b_pause.instate(["disabled"]), True)
        chk("save enabled", tab.b_save.instate(["disabled"]), False)
        chk("stop enabled", tab.b_stop.instate(["disabled"]), False)
        chk("status says GPU memory still in use", "GPU memory still in use" in tab.msg["text"], True)
        chk("reminder is red", str(tab.msg["foreground"]), "#c02020")
        # while waiting, GPU memory is held, and COLMAP also uses GPU memory -> other tabs stay locked
        chk("other tabs stay locked", busy[-1], True)
        # "change parameters and resume" requires editable parameters; the four frozen ones stay frozen
        chk("parameters editable (desicnt)",
            tab.widgets["desicnt"].instate(["disabled"]), False)
        chk("frozen ones still locked (resolution)",
            tab.widgets["resolution"].instate(["disabled"]), True)
        chk("output name stays locked", tab.name_ent.instate(["disabled"]), True)
        run.state = "running"

        # The case actually hit: rendering while paused; afterwards no new
        #    JSONL arrives, and the old version stayed on "Rendering video..."
        print("\n[Train tab / paused -> rendering -> finished (no new data meanwhile)]")
        run.state = "paused"
        set_rendering(True)
        tab._show_status()
        chk("rendering", tab.msg["text"].startswith("Rendering video"), True)
        set_rendering(False)
        tab._show_status()
        chk("returns to paused by itself", tab.msg["text"].startswith("Paused"), True)

        # Stop = free GPU memory, **no saving**. An older version said
        #    "saving" while stopping; that behaviour was dropped.
        print("\n[Train tab / stopping]")
        run.state = "stopping"
        tab._show_status()
        tab._refresh_saved()
        chk("status says freeing memory", "freeing GPU memory" in tab.msg["text"], True)
        chk("does not claim to be saving", "saving" in tab.msg["text"], False)
        chk("last save does not show Saving", tab.saved_lab["text"] != "Saving...", True)

        tab.stop_polling()
    finally:
        root.destroy()


def main():
    try:
        tk.Tk().destroy()
    except tk.TclError as e:
        print(f"  skipped: no display available ({e})")
        return 0
    check_preview_window()
    check_train_tab()
    if FAILED:
        print(f"\n  FAIL {len(FAILED)} check(s) did not pass:")
        for name in FAILED:
            print(f"     {name}")
        print("\n  Allowing those buttons while rendering lets commands compete with _preview_video for the same "
              "training process; a stuck status line makes the program look frozen.")
        return 1
    print("\n  OK freezing during rendering and the status line are correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
