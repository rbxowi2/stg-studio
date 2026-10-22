#!/usr/bin/env python3
"""train_cmd.py - print a training command that can be pasted and run as-is.

    python tools/train_cmd.py <shooting root> [--tier fast|balanced|max] [--name <output name>]

Give the shooting root; the dataset is found automatically (the only folder
below it that is a dataset). With several datasets, choose one with --dataset.

Why this exists: `STG_CAMERAS_EXTENT`, `STG_BACKGROUND` and `--duration`
would otherwise have to be computed by hand, and a wrong value raises no
error, it only trains a worse model. All three can be read from `window.json`.

The trainer and interpreter are found relative to this repository:
    .venv/Scripts/python.exe             (override with STG_PYTHON)
    third_party/SpacetimeGaussians       (override with STG_HOME)
setup.ps1 creates both.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("STG_PYTHON") or os.path.join(_REPO, ".venv", "Scripts", "python.exe")
STG = os.environ.get("STG_HOME") or os.path.join(_REPO, "third_party", "SpacetimeGaussians")

TIERS = {
    # name: (iterations, desicnt, densify_until, gnum_max, omega_iter, save points)
    "fast":     (10000,  60,  6000, 1500000,  4500, [10000]),
    "balanced": (30000,  60,  9000, 3000000,  9500, [30000]),
    "max":     (100000, 200, 14000, 5000000, 13500,
                list(range(10000, 100001, 10000))),
}


def is_dataset(d):
    """Whether this folder is a dataset.

    `window.json` alone is not enough: the trainer also writes one into a
    model folder when a time range is used, and model folders sit next to the
    dataset in the shooting root. The test therefore uses what only a dataset
    has: the per-time-step `colmap_0` and `times_group.json`.
    """
    return (os.path.isfile(os.path.join(d, "window.json"))
            and os.path.isdir(os.path.join(d, "colmap_0"))
            and os.path.isfile(os.path.join(d, "times_group.json")))


def find_dataset(target, want=None):
    """Find the dataset under a shooting root; a dataset folder itself is also accepted."""
    t = os.path.abspath(target)
    if is_dataset(t):
        return t                                     # a dataset was given directly
    if not os.path.isdir(t):
        raise SystemExit(f"  x {t} not found")
    hits = [os.path.join(t, d) for d in sorted(os.listdir(t))
            if is_dataset(os.path.join(t, d))]
    if want:
        hits = [h for h in hits if os.path.basename(h) == want] or hits
    if not hits:
        raise SystemExit(
            f"  x no dataset under {t} (a folder with window.json)\n"
            f"    run prep_4d.py <{t}> first")
    if len(hits) > 1:
        raise SystemExit(f"  x {len(hits)} datasets found: "
                         + ", ".join(os.path.basename(h) for h in hits)
                         + "\n    choose one with --dataset")
    return hits[0]



def default_holdout(nsteps, k=5):
    """Which time steps are held out for scoring, spread evenly over the dataset.

    A fixed list would ignore the dataset length: with 50 time steps, a list
    made for 300 would leave only one held-out step (4 images), far too noisy
    to compare models.

    Evenly spread and away from both ends: `nsteps=50` -> `5,15,25,35,45`
    (20 images). The first and last steps are avoided because their time
    parameters have data on one side only.
    """
    k = max(1, min(k, nsteps))
    step = nsteps / k
    return ",".join(str(int(step * (i + 0.5))) for i in range(k))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="shooting root (or a dataset folder)")
    p.add_argument("--tier", default="balanced", choices=list(TIERS))
    p.add_argument("--name", default=None, help="output model name (default: the tier name)")
    p.add_argument("--dataset", default=None, help="dataset name when there are several")
    p.add_argument("--holdout", default=None,
                   help="held-out time steps for scoring (default: spread evenly over the dataset)")
    a = p.parse_args()

    ds = find_dataset(a.target, a.dataset).replace("\\", "/")
    win = json.load(open(f"{ds}/window.json", encoding="utf-8"))
    root = os.path.dirname(ds)
    name = a.name or a.tier
    out = f"{root}/{name}_out".replace("\\", "/")

    dur_tmp = win["nsteps"]
    holdout = a.holdout or default_holdout(dur_tmp)

    it, desicnt, dens_until, gnum, omega, saves = TIERS[a.tier]
    extent = round(win["near_far"][1] * 1.1, 4)      # farthest surface x 1.1
    bg = win.get("background", 0.0)
    dur = win["nsteps"]

    if omega >= it:
        raise SystemExit(f"  x STG_OMEGA_ITER ({omega}) must be less than iterations ({it})")

    print(f"""# -- {a.tier}  dataset {os.path.basename(ds)}  {dur} time steps  background {bg}
cd {STG}
$env:STG_TIMES="{ds}/times_group.json"
$env:STG_ZNEAR_PRUNE="0.1"
$env:STG_CAMERAS_EXTENT="{extent}"
$env:STG_GNUM_MAX="{gnum}"
$env:STG_OMEGA_ITER="{omega}"
$env:STG_BACKGROUND="{bg}"
$env:STG_HOLDOUT_T="{holdout}"
Remove-Item Env:STG_BG_RANDOM -ErrorAction SilentlyContinue
& "{PY}" train.py --quiet --eval `
  --source_path "{ds}/colmap_0" --model_path "{out}" `
  --config "{ds}/cfg_lite.json" `
  --densify 2 --duration {dur} --desicnt {desicnt} --gtisint8 1 `
  --densify_until_iter {dens_until} `
  --iterations {it} --position_lr_max_steps {it} `
  --save_iterations {' '.join(str(s) for s in saves)}

# -- the model is written to {out}
# -- open it with stg-player:  player.py "{root}" --model {name}""")


if __name__ == "__main__":
    main()
