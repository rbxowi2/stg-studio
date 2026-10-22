"""Checks the **decisions** behind time windows and combined training: range parsing, segment splitting, held-out time steps, dataset detection.

**Why this exists**

All four fail **silently**: no error, just a wrong model or a locked-up tab.

| failure | symptom |
|---|---|
| a model folder also contains `window.json`, and dataset detection looked only at that file | after one training run with a range, **the main window vanished on Start** (`SystemExit` in a Tk callback = process exit) |
| held-out time steps computed in full-length coordinates | after cutting `50,105`, all five held-out time steps **fell outside the range**: no test views, and nothing on screen showed it |

Needs no GPU, no dataset and no desktop:

    python tools/check_time_range.py
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "studio"))

FAILED = []


def chk(name, got, want):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {got!r}"
          + ("" if ok else f" (expected {want!r})"))


def check_parse():
    import runner as R
    print("\n[Parsing the time range]")
    chk("empty = all", R.parse_range("", 301), None)
    chk("50,105 includes both ends", R.parse_range("50,105", 301), (50, 105))
    chk("spaces are accepted", R.parse_range(" 50 , 105 ", 301), (50, 105))
    for bad, why in (("105,50", "end before start"),
                     ("5", "only one number"),
                     ("a,b", "not numbers"),
                     ("0,400", "beyond the dataset"),
                     ("-1,10", "negative")):
        try:
            R.parse_range(bad, 301)
            chk(f"rejects {bad} ({why})", "not rejected", "ValueError")
        except ValueError as e:
            chk(f"rejects {bad} ({why})", bool(str(e).strip()), True)


def check_holdout():
    import runner as R
    print("\n[Held-out time steps must fall inside the range]")
    # The original bug: default_holdout used full-length coordinates, so
    #    after cutting a range every held-out step was outside = no test views.
    got = [int(x) for x in R.range_holdout(50, 105).split(",")]
    chk("held-out steps for 50-105 are inside the range",
        all(50 <= v <= 105 for v in got), True)
    chk("five held out", len(got), 5)
    chk("no duplicates", len(set(got)), 5)
    got2 = [int(x) for x in R.range_holdout(0, 300).split(",")]
    chk("full length also works", all(0 <= v <= 300 for v in got2), True)


def check_split():
    from tools.merge_segments import split_segments as S
    print("\n[Splitting into segments]")
    chk("10,90 with 30 per segment", S(10, 90, 30), [(10, 40), (40, 70), (70, 90)])
    chk("boundaries share one step", S(0, 60, 30), [(0, 30), (30, 60)])
    chk("no extra segment when it divides evenly", len(S(0, 300, 60)), 5)
    chk("a short tail joins the previous segment", S(10, 71, 30), [(10, 40), (40, 71)])
    chk("range shorter than one segment", S(10, 30, 30), [(10, 30)])
    try:
        S(0, 100, 0)
        chk("0 steps per segment is rejected", "not rejected", "ValueError")
    except ValueError:
        chk("0 steps per segment is rejected", True, True)


def check_is_dataset():
    from tools.train_cmd import find_dataset, is_dataset
    print("\n[Dataset detection: a model folder is not a dataset]")
    tmp = tempfile.mkdtemp()
    try:
        ds = os.path.join(tmp, "stg301")
        os.makedirs(os.path.join(ds, "colmap_0"))
        for f in ("window.json", "times_group.json"):
            open(os.path.join(ds, f), "w").close()
        # With a time range the trainer also writes window.json into the
        #    **model folder**, which sits next to the dataset under the shooting root.
        mdl = os.path.join(tmp, "balanced_out")
        os.makedirs(os.path.join(mdl, "_studio"))
        open(os.path.join(mdl, "window.json"), "w").close()

        chk("dataset is recognized", is_dataset(ds), True)
        chk("model folder is not a dataset", is_dataset(mdl), False)
        chk("only one found under the root", os.path.basename(find_dataset(tmp)), "stg301")

        # an empty shell from stage 1 only (window.json, no colmap_*) does not count either
        shell = os.path.join(tmp, "stg50")
        os.makedirs(shell)
        open(os.path.join(shell, "window.json"), "w").close()
        chk("stage 1 shell is not a dataset", is_dataset(shell), False)
        chk("still only one found with a shell present",
            os.path.basename(find_dataset(tmp)), "stg301")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    check_parse()
    check_holdout()
    check_split()
    check_is_dataset()
    if FAILED:
        print(f"\n  FAIL {len(FAILED)} check(s) did not pass:")
        for name in FAILED:
            print(f"     {name}")
        print("\n  None of these fail with an error: held-out steps outside the range = no test views; "
              "a model folder taken for a dataset = the training tab locks up.")
        return 1
    print("\n  OK time window and segment decisions are correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
