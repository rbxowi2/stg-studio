"""Resuming in the studio: `train_meta.json`, the three kinds of name clash, freezing and enabling rules.

**Why this check exists**

Resuming changed "always block a name clash" into "if it clashes with our
own stopped training, continue it". That path spans three files (meta and
`resume_state` in `runner.py`, `_show_out` / `_name_ok` in `train_tab.py`,
`RESUME_FROZEN` in `params.py`), and **breaking any of them shows up only as
a button being enabled or disabled, which is hard to notice by eye**.

A real resume needs a GPU and a few minutes (left to manual testing). This
only checks the **decisions**, with no GPU, dataset or training: a temporary
folder fakes the three kinds of clash.

    python tools/check_resume_mode.py

**Needs a desktop** (tkinter windows are created, all hidden). Skipped without a display.
ttk widgets must be queried with `instate(["disabled"])`, not `w["state"]`.
"""
import os
import shutil
import sys
import tempfile
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


def make_case(tmp, name, meta=True, state=True, last_iter=1050, total=3000):
    """Fake an output folder. Whether meta/state exist decides the kind of clash."""
    import runner as R
    out = os.path.join(tmp, f"{name}_out")
    os.makedirs(os.path.join(out, R.CTRL_DIR), exist_ok=True)
    if meta:
        # The tier is deliberately max and a parameter deliberately **differs**
        #    from max's default, so both "the tier is restored" and "the
        #    parameter is not overwritten by the tier" can be verified.
        R.write_meta(out, dataset="stg301", tier="max",
                     params={"iterations": total, "desicnt": 60,
                             "resolution": 2},
                     total_iterations=total, last_iter=last_iter,
                     finished=False, resumes=[])
    if state:
        open(os.path.join(out, R.CTRL_DIR, "state.pth"), "w").close()
    return out


def check_meta(tmp):
    import runner as R
    print("\n[train_meta.json]")
    out = make_case(tmp, "meta")
    m = R.read_meta(out)
    chk("can be read back", bool(m), True)
    chk("schema is right", m["schema"], R.META_SCHEMA)
    need = {"dataset", "tier", "params", "total_iterations", "last_iter",
            "finished", "resumes", "updated_at"}
    chk("all fields present", sorted(need - set(m)), [])
    R.write_meta(out, last_iter=2000)
    chk("in-place update keeps other fields", R.read_meta(out)["tier"], "max")
    chk("update applied", R.read_meta(out)["last_iter"], 2000)
    # a broken file counts as "none"; it must not crash the UI
    with open(R.meta_path(out), "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    chk("broken meta counts as none", R.read_meta(out), None)


def check_kinds(tmp):
    import runner as R
    print("\n[Three kinds of name clash]")
    chk("meta + checkpoint -> can resume",
        R.resume_state(make_case(tmp, "a"))[0], "resume")
    chk("meta without checkpoint -> cannot resume",
        R.resume_state(make_case(tmp, "b", state=False))[0], "no_state")
    chk("neither -> another program's folder",
        R.resume_state(make_case(tmp, "c", meta=False, state=False))[0],
        "foreign")


def check_stale_log(tmp):
    """When resuming, the log is appended: messages from the previous run must not count as this run's state.

    Observed: after resuming, the status line stayed on "finished", because
    `holding` searched the whole log for `[checkpoint] finished`, a line
    written by the **previous** run.
    """
    import runner as R
    print("\n[Resume: the previous run's log does not count]")
    out = os.path.join(tmp, "stale_out")
    ctrl = os.path.join(out, R.CTRL_DIR)
    os.makedirs(ctrl, exist_ok=True)
    log = os.path.join(ctrl, R.LOG_NAME)
    with open(log, "w", encoding="utf-8") as f:      # previous run: finished and waited for commands
        f.write("Training progress: 100%\n"
                "[checkpoint] finished, holding for commands\n"
                "[checkpoint] saved at iteration 50000\n"
                "[checkpoint] released\n")
    run = R.Runner()
    run.log_path = log
    run._log_from = os.path.getsize(log)             # this run starts writing here
    chk("previous finished does not count", run.holding, False)
    chk("previous iteration does not count", run.log_iteration(), None)
    with open(log, "a", encoding="utf-8") as f:      # this run really finished
        f.write("[checkpoint] saved at iteration 60000\n"
                "[checkpoint] finished, holding for commands\n")
    chk("this run's finished counts", run.holding, True)
    chk("this run's iteration counts", run.log_iteration(), 60000)

    # Finished-and-waiting is a **real state**: the process is alive and GPU
    #    memory is held, but state must say finished, so the button rules do
    #    not work by coincidence.
    class _Alive:                     # fake process: poll() returns None = alive
        def poll(self):
            return None

    run.proc = _Alive()
    run.ctrl = ctrl
    run.state = "running"
    chk("finished and waiting -> finished", run.poll()[0], "finished")
    run.proc = None


def check_ui(tmp):
    import params as P
    from train_tab import TrainTab
    print("\n[Screen: freezing and enabling in resume mode]")
    ds = os.path.join(tmp, "stg301")
    os.makedirs(ds, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    try:
        tab = TrainTab(root, get_root=lambda: tmp, show_error=lambda *a: None)
        tab.info = {"dataset": ds.replace("\\", "/")}
        tab.compat = None

        def use(name):
            tab.name.set(name)          # the trace calls _show_out
            tab._sync_buttons()

        use("a")                        # the resumable one
        chk("entered resume mode", tab.resume_meta is not None, True)
        chk("button reads Resume from checkpoint", str(tab.b_start["text"]), "Resume from checkpoint")
        chk("enabled", tab.b_start.instate(["disabled"]), False)
        chk("parameters restored", tab.vars["iterations"].get(), "3000")
        # the tier is restored too (resuming restores the whole setting)
        chk("tier restored", tab.tier.get(), "max")
        chk("parameter not overwritten by the tier", tab.vars["desicnt"].get(), "60")
        # after restoring, the user may edit; the edit must survive the next _show_out
        tab.vars["desicnt"].set("99")
        tab.name.set("a")               # trigger again (like typing one more character)
        tab._sync_buttons()
        chk("user edit is not overwritten", tab.vars["desicnt"].get(), "99")

        # switching to another name, changing something and switching back
        #    -> **restore again** (originally only the first time triggered it)
        use("brand_new")
        tab.tier.set("fast")
        tab._apply_tier()               # the user changed the tier on another name
        use("a")
        chk("switching back restores the tier", tab.tier.get(), "max")
        chk("switching back restores parameters", tab.vars["desicnt"].get(), "60")
        for key in P.RESUME_FROZEN:
            chk(f"frozen {key}", tab.widgets[key].instate(["disabled"]), True)
        chk("unfrozen ones stay editable (desicnt)",
            tab.widgets["desicnt"].instate(["disabled"]), False)
        chk("tier locked", tab.tier_rbs[0].instate(["disabled"]), True)

        tab.vars["iterations"].set("1000")      # less than the 1050 already done
        tab._sync_buttons()
        chk("disabled when already that far", tab.b_start.instate(["disabled"]), True)
        tab.vars["iterations"].set("5000")      # larger = extend training
        tab._sync_buttons()
        chk("enabled when larger", tab.b_start.instate(["disabled"]), False)

        # Observed: after finishing, raising the iterations still left
        #    "Resume from checkpoint" disabled, because buttons were only
        #    recomputed on events and "the user edited a field" was not one
        #    of them. Criterion: **polling alone** (no manual _sync_buttons)
        #    must re-enable it.
        tab.vars["iterations"].set("1000")     # first make it disabled
        tab._poll()
        chk("disabled with too few iterations (polling only)", tab.b_start.instate(["disabled"]), True)
        tab.vars["iterations"].set("9000")     # the user types a larger value
        tab._poll()                            # only wait for polling, touch nothing else
        chk("enabled by polling alone after raising", tab.b_start.instate(["disabled"]), False)

        use("b")                        # meta without checkpoint
        chk("cannot resume", tab.resume_meta is None, True)
        chk("button disabled", tab.b_start.instate(["disabled"]), True)
        chk("reason given", "cannot be resumed" in tab.out_lab["text"], True)

        use("c")                        # another program's folder
        chk("still blocked", tab.b_start.instate(["disabled"]), True)
        chk("button back to Start training", str(tab.b_start["text"]), "Start training")

        use("brand_new")                # brand-new name
        chk("a new name can start", tab.b_start.instate(["disabled"]), False)

        # when resuming, elapsed time covers only this run: elapsed_s goes back at the resume point
        pts = [{"iter": i, "loss": .1, "n_gauss": 1, "max_scale": 1.,
                "elapsed_s": e}
               for i, e in ((200, 7.), (400, 14.), (600, 21.), (800, 28.3),
                            (1000, 5.), (1200, 15.4), (1400, 24.8))]
        chk("only the last run counts", [p["iter"] for p in tab._this_run(pts)],
            [1000, 1200, 1400])
        tab.stop_polling()
    finally:
        root.destroy()


def main():
    try:
        tk.Tk().destroy()
    except tk.TclError as e:
        print(f"  skipped: no display available ({e})")
        return 0
    tmp = tempfile.mkdtemp()
    try:
        check_meta(tmp)
        check_kinds(tmp)
        check_stale_log(tmp)
        check_ui(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAILED:
        print(f"\n  FAIL {len(FAILED)} check(s) did not pass:")
        for name in FAILED:
            print(f"     {name}")
        print("\n  The resume decisions are wrong; the symptom is either a clash that is not blocked "
              "or a resumable run that cannot be started - the former overwrites an existing model.")
        return 1
    print("\n  OK resume decisions are correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
