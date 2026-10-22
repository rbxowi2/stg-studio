#!/usr/bin/env python3
"""calibrate.py - calibration set -> portable rig folder (fully automatic).

    python calibrate.py <calibration set folder>
    python calibrate.py <calibration set folder> --edge cam_A,cam_C,89.9 --edge cam_A,cam_B,148.3
    python calibrate.py <calibration set folder> --out D:\\myrig --hfov 86

Automates the manual multi-camera COLMAP calibration procedure.

## Dependencies

* the COLMAP executable (found in common locations, or given with `--colmap`)
* the Python 3 standard library; calibration itself needs no numpy/scipy/opencv

So it runs on another machine as long as COLMAP is available.

The one exception is the sharpness filter (discarding blurred sets at the
start, see `shot_sharpness`): it needs OpenCV and is skipped automatically if
OpenCV cannot be loaded. The check is useful but not worth a hard
dependency. Disable it with `--no-blur-filter`.

## Output: portable rig folder (about 5 KB)

    rig/
        rig_config.json    full rig definition with intrinsics and extrinsics, read by preprocessing
        cameras.txt        intrinsics per camera (OPENCV, 8 parameters)
        rigs.txt           relative pose per camera (cam_from_rig)
        prefixes.json      camera_id -> cam_X mapping
        scale.json         record of the metric scale (prevents applying it twice)
        summary.txt        human-readable summary
        calib_report.json  acceptance numbers

The folder can be reused across sessions but is not valid forever: calibration
accuracy degrades over time even when the rig is not touched. Using
calibrations from different times on the same recording changed the number of
initial points by a factor of 1.65, and in every comparison the calibration
closest in time performed best. Calibrate on the day of shooting.

The reprojection threshold does not catch this degradation (it differed by
only 0.1-0.2 px), so a normal reprojection error is not evidence that a
calibration is recent enough.

## Why two stages

Calibrating directly with OPENCV (8 parameters) diverges: the optimizer uses
k1/k2/p1/p2 to compensate for focal length error and ends in a physically
meaningless local minimum. Therefore:

1. Stage A, SIMPLE_RADIAL (4 parameters): few degrees of freedom; refines the focal length
2. Stage B, OPENCV (8 parameters): with the focal length known, the distortion terms are freed

Stage A needs an initial focal length: with an initial value off by 2x, P3P
fails and incremental reconstruction stops after 2-3 images (the focal length
was observed running to 4768 against a true value of about 1000).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

# Unbuffered output: each step takes tens of seconds and buffering would hide progress
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from pathlib import Path

HERE = Path(__file__).resolve().parent

COLMAP_GUESSES = [
    str(HERE.parent / "third_party" / "colmap" / "bin" / "colmap.exe"),
    r"C:\Program Files\COLMAP\colmap.exe",
    "/usr/local/bin/colmap", "/usr/bin/colmap", "colmap",
]

# ============================================================ acceptance thresholds
#
# Calibration reprojection error cannot use scene-reconstruction thresholds;
# the two differ in nature:
#
#   calibration     dozens of images, a moving board, wide view changes  -> 1-2 px is normal
#   scene recon.    4-8 images, static scene, denoised by stacking        -> 0.4-0.8 px
#
# A calibration at 1.21 px produced a rig that passed a 1 mm physical check.
#
# The error necessarily rises after the rig bundle adjustment: binding the
# cameras rigidly removes degrees of freedom, so residuals can only grow. A
# rise is fine; a jump is not (it means the rigid-rig assumption does not
# hold, for example the mount was moved).
TH_REPROJ_B = 1.5           # stage B (free poses); measured 1.21-1.26
TH_REPROJ_RIG = 2.2         # rig BA (rigid constraint); measured 1.60
TH_REPROJ_RISE = 0.8        # rise of rig BA over stage B; measured +0.34
TH_EDGE_MM = 2.0            # difference from measured distances; measured within 1 mm
TH_FOCAL_SPREAD = 8.0       # focal length spread between cameras %; 1-3% is normal for one lens model

# Stage A tries these mapper seeds in order until every image registers and
# the focal spread passes. COLMAP's default random_seed=-1 is random each
# time, and the same data can succeed or fail (one of three runs collapsed:
# 4/100 registered, focal length 11760). Fixed seeds make results
# reproducible; several seeds avoid getting stuck on one unlucky start.
MAPPER_SEEDS = (0, 17, 42, 7)


# ============================================================ helpers

def find_colmap(explicit=None) -> str:
    for c in ([explicit] if explicit else []) + COLMAP_GUESSES:
        if not c:
            continue
        if os.path.sep in str(c) or "/" in str(c):
            if Path(c).exists():
                return str(c)
        elif shutil.which(c):
            return c
    raise SystemExit("COLMAP not found. Give the executable path with --colmap.")


class Runner:
    def __init__(self, colmap, verbose=False):
        self.colmap, self.verbose = colmap, verbose
        self.log = []

    def __call__(self, *args, check=True) -> str:
        cmd = [self.colmap] + [str(a) for a in args]
        # When started by the studio under pythonw there is no console, and
        # every COLMAP step would open its own console window. No effect from
        # a command line.
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
        self.log.append({"cmd": " ".join(cmd[1:]), "rc": r.returncode})
        if self.verbose:
            print("      $ colmap " + " ".join(str(a) for a in args))
        if check and r.returncode != 0:
            print(f"\n  ERROR COLMAP failed: colmap {args[0]}")
            print((r.stderr or r.stdout)[-1500:])
            raise SystemExit(1)
        return (r.stdout or "") + (r.stderr or "")


def step(n, total, msg):
    print(f"\n[{n}/{total}] {msg}")


def shot_sharpness(calib: Path, names, shots):
    """Sharpness of each set (variance of the Laplacian). Returns {set name: [value per camera]}.

    Absolute values mean nothing (they depend on scene texture and contrast);
    they are only compared between sets of the same session and scene. A set
    clearly below the others was shaken.

    Returns None if OpenCV cannot be loaded (calibration itself does not need it).
    """
    try:
        import cv2
        import numpy as np                              # noqa: F401
    except ImportError:
        return None
    out = {}
    for s in shots:
        vals = []
        for n in names:
            p = calib / n / s.name
            if not p.is_file():
                continue
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            vals.append(float(cv2.Laplacian(im, cv2.CV_64F).var()))
        if vals:
            out[s.name] = vals
    return out or None


# ============================================================ COLMAP TXT parsing

def read_cameras(p: Path) -> dict:
    out = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.startswith("#") or not l.strip():
            continue
        f = l.split()
        out[int(f[0])] = {"model": f[1], "w": int(f[2]), "h": int(f[3]),
                          "params": [float(x) for x in f[4:]], "raw": l}
    return out


def read_image_prefixes(p: Path) -> dict:
    """camera_id -> folder name (such as cam_A/)"""
    out, lines = {}, [l for l in p.read_text(encoding="utf-8").splitlines()
                      if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        f = lines[i].split()
        cid, name = int(f[8]), f[9]
        if "/" in name:
            out[cid] = name.rsplit("/", 1)[0] + "/"
    return out


def read_rigs(p: Path):
    """Returns {ref_camera_id, sensors: {cid: {q, t}}} from a COLMAP 4.x rigs.txt."""
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.startswith("#") or not l.strip():
            continue
        f = l.split()
        n = int(f[1])
        ref = int(f[3])
        sensors, i = {}, 4
        for _ in range(n - 1):
            cid = int(f[i + 1])
            has = int(f[i + 2])
            if has:
                q = [float(x) for x in f[i + 3:i + 7]]
                t = [float(x) for x in f[i + 7:i + 10]]
                sensors[cid] = {"q": q, "t": t}
                i += 10
            else:
                i += 3
        return {"ref": ref, "sensors": sensors, "raw": l}
    return None


def q2R(q):
    """Quaternion -> rotation matrix (pure Python, no numpy)."""
    w, x, y, z = q
    return [[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]]


def cam_center(q, t):
    """The camera center is C = -R^T t, not the translation stored in rigs.txt.

    Treating the translation as the position gives wrong geometry.
    """
    R = q2R(q)
    return [-sum(R[k][i] * t[k] for k in range(3)) for i in range(3)]


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def parse_analyzer(txt: str) -> dict:
    out = {}
    for l in txt.splitlines():
        for key, field in (("Registered images", "registered"),
                           ("Points", "points"),
                           ("Mean reprojection error", "reproj_px"),
                           ("Cameras", "cameras")):
            if key in l:
                v = l.split(":")[-1].strip().replace("px", "")
                try:
                    out[field] = float(v)
                except ValueError:
                    pass
    return out


def confirm_unscaled(assume_yes=False) -> bool:
    """Last confirmation before writing an unscaled rig. Returns whether to continue.

    An unscaled rig folder looks exactly like a normal one (same files and
    format), so nobody would notice when copying it into a shooting root. The
    problem must stop where it is created, not surface in the player.
    """
    print("""
  !  This calibration has no metric scale

     No --edge was given and there is no other scale reference. The rig's
     shape is correct, but it has no real size: 1 unit = an arbitrary SfM unit.

  Consequences

     preprocessing  near_far and the initial point cloud are in internal units
     audio          speed-of-sound conversion is wrong; direction and delay are wrong
     wall HUD       position and size are wrong
     VR             head movement is in real metres while the world is not -> one step flies far

  Cost of fixing it later

     The scale must be added before preprocessing. If it is found after
     preprocessing and training, both must run again (about 15 minutes of
     preprocessing and 30-45 minutes of training).

  Avoid it now

     Measure the six distances -> run again with --edge
""")
    if assume_yes:
        print("  (--yes: writing an unscaled rig without asking)")
        return True
    try:
        return input("  Write an unscaled rig anyway? (y/N): ").strip().lower() in ("y", "yes")
    except EOFError:
        print("  (non-interactive and no --yes)")
        return False


# ============================================================ main flow

def main() -> int:
    ap = argparse.ArgumentParser(description="calibration set -> portable rig folder")
    ap.add_argument("calib", help="calibration set folder from the recorder (with session.json)")
    ap.add_argument("--out", default=None, help="output rig folder (default <calibration set>/rig)")
    ap.add_argument("--work", default=None, help="intermediate files (default <calibration set>/_work)")
    ap.add_argument("--hfov", type=float, default=86.0,
                    help="horizontal field of view (deg) for the initial focal length; about 86 for the cameras used")
    ap.add_argument("--edge", action="append", default=[], metavar="A,B,mm",
                    help="measured distance between two cameras; repeatable. Without it no metric scale is applied")
    ap.add_argument("--colmap", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep-work", action="store_true", help="keep intermediate files")
    ap.add_argument("--blur-ratio", type=float, default=0.5,
                    help="discard sets whose sharpness is below this fraction of the median (default 0.5)")
    ap.add_argument("--no-blur-filter", action="store_true",
                    help="no sharpness filtering")
    ap.add_argument("--yes", action="store_true",
                    help="non-interactive; also writes an unscaled rig without asking (for automation)")
    a = ap.parse_args()

    calib = Path(a.calib).resolve()
    work = Path(a.work) if a.work else calib / "_work"
    out = Path(a.out) if a.out else calib / "rig"
    colmap = find_colmap(a.colmap)
    run = Runner(colmap, a.verbose)
    t0 = time.time()
    TOT = 8

    # ---------------------------------------------------------- 1. check input
    step(1, TOT, "Checking the calibration set")
    sj = calib / "session.json"
    if not sj.exists():
        print(f"  ERROR no session.json in {calib}")
        if (Path.cwd() / "session.json").exists():
            print(f"  hint: the current folder has one. The command you want is:")
            print(f"       python calibrate.py .")
            print("     (the argument is the calibration set folder; a dot means the current folder)")
        else:
            near = [d for d in Path.cwd().iterdir()
                    if d.is_dir() and (d / "session.json").exists()]
            if near:
                print(f"  hint: calibration sets found nearby:")
                for d in near[:5]:
                    print(f"       python calibrate.py {d.name}")
        raise SystemExit(1)
    ses = json.loads(sj.read_text(encoding="utf-8"))
    if ses.get("mode") != "calib":
        print(f"  ! session mode is {ses.get('mode')}, not calib - is this right?")
    names = [s["name"] for s in ses["streams"]]
    W, H = ses["capture"]["resolution"]
    shots = sorted((calib / names[0]).glob("shot_*.*"))
    print(f"  {len(names)} cameras: {', '.join(names)}")
    print(f"  {len(shots)} sets  resolution {W}x{H}")
    print(f"  COLMAP: {colmap}")
    for n in names:
        k = len(list((calib / n).glob("shot_*.*")))
        if k != len(shots):
            print(f"  ! {n} has only {k} images (others have {len(shots)})")
    if len(shots) < 10:
        print(f"  ! only {len(shots)} sets; 15 or more recommended")

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"

    # -- sharpness filter --------------------------------------------------
    #
    # A set shot while the rig shook is hard to remember afterwards. In one
    # session, one set of 27 had only 46% of the median sharpness, on all four
    # cameras at once: the signature of the whole rig moving, not of one camera.
    #
    # Blur by itself mostly weakens feature localization (lowering its
    # weight). The real danger is the rolling shutter: during a shake the
    # image is geometrically distorted, and bundle adjustment fits that false
    # geometry and absorbs the error into the extrinsics.
    #
    # A set is discarded for all cameras together; dropping one camera would
    # leave an incomplete rig set.
    dropped = []
    sharp = None if a.no_blur_filter else shot_sharpness(calib, names, shots)
    if sharp:
        med = sorted(v for vs in sharp.values() for v in vs)
        med = med[len(med) // 2]
        ratios = {k: min(v) / med for k, v in sharp.items()}
        cand = sorted(k for k, r in ratios.items() if r < a.blur_ratio)
        print(f"  median sharpness {med:.0f} (Laplacian variance, for comparison between sets only)")
        # If too many sets would be discarded, discard none: when the whole
        # session is soft (low light, focus drift), the median itself is low,
        # and discarding would only throw data away.
        if len(cand) > max(1, len(shots) // 4):
            print(f"  ! {len(cand)}/{len(shots)} sets are below "
                  f"{a.blur_ratio:.0%} of the median - too many, none discarded. "
                  f"The whole session may be soft (low light or focus drift)")
        elif cand:
            dropped = cand
            for k in cand:
                print(f"  ! discarding {k}: its blurriest camera has only "
                      f"{ratios[k]:.0%} of the median (threshold {a.blur_ratio:.0%}) - the rig shook")
            shots = [s for s in shots if s.name not in dropped]
            print(f"  -> using {len(shots)} sets (of {len(shots) + len(dropped)})")
        else:
            print(f"  OK no clearly shaken set (lowest {min(ratios.values()):.0%}, "
                  f"threshold {a.blur_ratio:.0%})")

    # COLMAP reads the whole folder, so a list tells it which images to read.
    # Without discarded sets the parameter is not passed at all.
    img_list = []
    if dropped:
        lst = work / "image_list.txt"
        lst.write_text("\n".join(f"{n}/{s.name}" for s in shots for n in names),
                       encoding="utf-8")
        img_list = ["--image_list_path", str(lst)]

    # ---------------------------------------------------------- 2. initial focal length
    step(2, TOT, "Estimating the initial focal length")
    f0 = (W / 2) / math.tan(math.radians(a.hfov) / 2)
    cx, cy = W / 2, H / 2
    print(f"  HFOV {a.hfov} deg -> f0 = (W/2)/tan(HFOV/2) = {f0:.0f} px")
    print(f"  This is the critical prerequisite: with an initial value off by 2x, P3P fails,")
    print(f"     incremental reconstruction stops after 2-3 images and the focal length drifts away")

    # ---------------------------------------------------------- 3. stage A
    step(3, TOT, "Stage A: SIMPLE_RADIAL refines the focal length (4 parameters, does not diverge)")
    run("feature_extractor", "--database_path", db, "--image_path", calib,
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_params", f"{f0:.1f},{cx},{cy},0", *img_list)
    run("exhaustive_matcher", "--database_path", db)
    # COLMAP's incremental reconstruction can differ on every run.
    #
    # `--Mapper.random_seed` defaults to -1 (random), and incremental
    # reconstruction depends heavily on the initial image pair: a pair with
    # too little parallax or nearly coplanar views leaves the focal length
    # unconstrained, it diverges, and everything after it fails.
    #
    # Same 25-set data, same parameters, three runs:
    #     run 1  registered   4/100   mean focal 11760   spread 25.1%   <- bad
    #     run 2  registered 100/100   mean focal   991   spread  3.1%   <- good
    #     run 3  registered 100/100   mean focal   989   spread  2.8%   <- good
    # The data is identical; only the initial pair differs.
    #
    # Checking only "a model was produced" would accept 4/100 and pass a bad
    # focal length to stage B, failing much later in rig calibration, far
    # from the cause. So: fixed seeds for reproducibility, and a new seed
    # when a run is bad.
    total_imgs = len(names) * len(shots)
    best = None
    for i, seed in enumerate(MAPPER_SEEDS):
        sa = work / ("sparse_a" if i == 0 else f"sparse_a_try{i}")
        if sa.exists():
            shutil.rmtree(sa)
        sa.mkdir()
        run("mapper", "--database_path", db, "--image_path", calib,
            "--output_path", sa,
            "--Mapper.random_seed", str(seed),
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.multiple_models", "0")
        if not (sa / "0").exists():
            print(f"  ! seed={seed} produced no model; trying another seed")
            continue
        ana = parse_analyzer(run("model_analyzer", "--path", sa / "0"))
        run("model_converter", "--input_path", sa / "0",
            "--output_path", sa, "--output_type", "TXT")
        cams = read_cameras(sa / "cameras.txt")
        fl = [c["params"][0] for c in cams.values()]
        favg = sum(fl) / len(fl)
        sp = (max(fl) - min(fl)) / favg * 100
        reg = ana.get("registered", 0)
        if best is None or (reg, -sp) > (best[0], -best[1]):
            best = (reg, sp, seed, sa, ana, cams, fl, favg)
        if reg >= total_imgs and sp <= TH_FOCAL_SPREAD:
            if i:
                print(f"  OK seed={seed} succeeded on attempt {i+1}")
            break
        print(f"  ! seed={seed} registered {reg:.0f}/{total_imgs}, "
              f"focal spread {sp:.1f}% - poor initial image pair, trying another seed")

    if best is None:
        raise SystemExit(f"  ERROR stage A reconstruction failed: none of the {len(MAPPER_SEEDS)} seeds produced a model. "
                         "Check that the scene has enough texture and that --hfov is right")
    _, spread, _, sa, ana_a, cams_a, focals, f1 = best
    print(f"  registered {ana_a.get('registered',0):.0f}/{len(names)*len(shots)}  "
          f"reprojection {ana_a.get('reproj_px',0):.3f} px")
    print(f"  focal lengths: {'  '.join(f'{x:.0f}' for x in focals)}")
    print(f"  mean f1 = {f1:.1f} px (spread {spread:.1f}%; 1-3% is normal for one lens model)")
    if spread > 8:
        print("  ! large focal spread; one camera may be poorly calibrated")

    # ---------------------------------------------------------- 4. stage B
    step(4, TOT, "Stage B: OPENCV with the refined focal length as prior (8 parameters, distortion)")
    db.unlink()
    run("feature_extractor", "--database_path", db, "--image_path", calib,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_params", f"{f1:.1f},{f1:.1f},{cx},{cy},0.04,0,0,0",
        *img_list)
    run("exhaustive_matcher", "--database_path", db)

    # Stage B also uses fixed seeds, retries, and checks the registration count.
    #
    # The same failure was observed in stage B:
    #     stage A  registered 108/108   reprojection 1.449 px      <- the data is fine
    #     stage B  registered   4/108   reprojection 0.382 px      <- "succeeds" with 4 images
    # and rig_configurator then reports "Failed to derive sensor_from_rig ...
    # image was not registered", two steps away from the cause.
    #
    # A 0.382 px reprojection error looks excellent only because the remaining
    # 4 images agree with each other. Read the registration count together
    # with the error.
    sb = None
    best_b = None
    for i, seed in enumerate(MAPPER_SEEDS):
        cand = work / ("sparse" if i == 0 else f"sparse_try{i}")
        if cand.exists():
            shutil.rmtree(cand)
        cand.mkdir()
        run("mapper", "--database_path", db, "--image_path", calib,
            "--output_path", cand,
            "--Mapper.random_seed", str(seed),
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "1",
            "--Mapper.multiple_models", "0")
        if not (cand / "0").exists():
            print(f"  ! seed={seed} produced no model; trying another seed")
            continue
        ana = parse_analyzer(run("model_analyzer", "--path", cand / "0"))
        reg = ana.get("registered", 0)
        if best_b is None or reg > best_b[0]:
            best_b = (reg, cand, ana)
        if reg >= total_imgs:
            if i:
                print(f"  OK seed={seed} succeeded on attempt {i+1}")
            break
        print(f"  ! seed={seed} registered {reg:.0f}/{total_imgs} - "
              f"poor initial image pair, trying another seed")

    if best_b is None:
        raise SystemExit("  ERROR stage B reconstruction failed: no seed produced a model")
    reg_b, sb, ana_b0 = best_b
    reg_a = ana_a.get("registered", 0)
    # The threshold is relative to stage A, not an absolute 100%.
    #
    # Missing one or two images is normal (the 8-parameter distortion makes
    # the solution slightly less stable); such a model is usable, and step 8
    # reports the registration rate. What must be stopped is a collapse
    # (like 4/108), which would break rig calibration with an unclear message.
    # A sanity threshold is set against the other path through the same data,
    # not against an ideal value: after discarding a blurred set, stage A
    # itself may register only 103/104.
    if reg_b < 0.8 * max(reg_a, 1):
        raise SystemExit(
            f"  ERROR stage B registered only {reg_b:.0f}/{total_imgs} images "
            f"while stage A registered {reg_a:.0f}/{total_imgs}; "
            f"all {len(MAPPER_SEEDS)} seeds were tried.\n"
            f"     Stage A fine and stage B collapsed usually means the 8-parameter distortion "
            f"destabilizes the solution; shoot more sets or let the board / texture fill more of the image.")
    if reg_b < total_imgs:
        print(f"  ! registered {reg_b:.0f}/{total_imgs} (stage A: {reg_a:.0f}) "
              f"- one or two missing is fine; step 8 reports it")
    ana_b = parse_analyzer(run("model_analyzer", "--path", sb / "0"))
    print(f"  registered {ana_b.get('registered',0):.0f}/{len(names)*len(shots)}  "
          f"reprojection {ana_b.get('reproj_px',0):.3f} px  "
          f"points {ana_b.get('points',0):.0f}")

    # ---------------------------------------------------------- 5. rig calibration
    step(5, TOT, "Rig calibration: binding the cameras rigidly")
    # rig_configurator fails silently when it cannot find its config: no
    # error, no message, exit code 0, and the next step reports a missing
    # file. The output is therefore checked to be non-empty.
    rc = work / "rig_config.json"
    rc.write_text(json.dumps([{"cameras": [
        {"image_prefix": f"{n}/", **({"ref_sensor": True} if i == 0 else {})}
        for i, n in enumerate(names)]}], indent=2), encoding="utf-8")
    sr = work / "sparse_rig"
    sr.mkdir()
    run("rig_configurator", "--database_path", db, "--rig_config_path", rc,
        "--input_path", sb / "0", "--output_path", sr)
    if not any(sr.iterdir()):
        raise SystemExit("  ERROR rig_configurator produced no output (silent failure)")

    sf = work / "sparse_final"
    sf.mkdir()
    # The principal point must be refined. Otherwise the rig geometry differs
    # from the physical measurement by 9.8%: the sensor's off-center
    # placement is forced into the extrinsics as a false pitch rotation.
    run("bundle_adjuster", "--input_path", sr, "--output_path", sf,
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_principal_point", "1",
        "--BundleAdjustment.refine_extra_params", "1")
    ana_f = parse_analyzer(run("model_analyzer", "--path", sf))
    run("model_converter", "--input_path", sf,
        "--output_path", sf, "--output_type", "TXT")
    print(f"  registered {ana_f.get('registered',0):.0f}  "
          f"reprojection {ana_f.get('reproj_px',0):.3f} px (principal point refined)")

    cams = read_cameras(sf / "cameras.txt")
    prefixes = read_image_prefixes(sf / "images.txt")
    rig = read_rigs(sf / "rigs.txt")
    if rig is None:
        raise SystemExit("  ERROR rigs.txt contains no rig")

    # ---------------------------------------------------------- 6. metric scale
    step(6, TOT, "Metric scale")
    cid_of = {v.rstrip("/"): k for k, v in prefixes.items()}
    centers = {rig["ref"]: [0.0, 0.0, 0.0]}
    for c, s in rig["sensors"].items():
        centers[c] = cam_center(s["q"], s["t"])

    edges = []
    for e in a.edge:
        p = e.split(",")
        if len(p) != 3:
            print(f"  ! --edge must be A,B,mm; ignoring {e}")
            continue
        edges.append((p[0].strip(), p[1].strip(), float(p[2])))

    scale_doc, edge_rows = None, []
    if edges:
        num = den = 0.0
        for A, B, mm in edges:
            if A not in cid_of or B not in cid_of:
                print(f"  ! {A} or {B} not found; ignored")
                continue
            u = dist(centers[cid_of[A]], centers[cid_of[B]])
            num += u * mm
            den += u * u
        if den == 0:
            print("  ! no usable distance; metric scale skipped")
        else:
            mm_per_unit = num / den
            print(f"  1 SfM unit = {mm_per_unit:.3f} mm"
                  f" (least squares over {len(edges)} distances)")
            for A, B, mm in edges:
                if A in cid_of and B in cid_of:
                    got = dist(centers[cid_of[A]], centers[cid_of[B]]) * mm_per_unit
                    d = got - mm
                    edge_rows.append({"pair": [A, B], "measured_mm": mm,
                                      "reconstructed_mm": round(got, 2),
                                      "diff_mm": round(d, 2)})
                    print(f"    {A}-{B}: reconstructed {got:7.1f} mm  measured {mm:7.1f} mm  "
                          f"diff {d:+6.2f} mm  {'OK' if abs(d) <= TH_EDGE_MM else 'FAIL'}")
            # apply to rigs.txt and rig_config: translations converted to metres
            fac = mm_per_unit / 1000.0
            for c in rig["sensors"].values():
                c["t"] = [x * fac for x in c["t"]]
            for c in centers:
                centers[c] = [x * fac for x in centers[c]]
            scale_doc = {"mm_per_unit": mm_per_unit, "unit": "m", "metric": True,
                         "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "edges": edge_rows}
    else:
        print("  (no --edge given; skipped)")

    if scale_doc is None and not confirm_unscaled(a.yes):
        print("  Aborted; no rig/ was written.")
        print("  Measure the six distances and run again: --edge cam_A,cam_B,91.0 --edge ...")
        return 2

    # ---------------------------------------------------------- 7. portable folder
    step(7, TOT, "Writing the portable rig folder")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(sf / "cameras.txt", out / "cameras.txt")

    # rigs.txt: rewrite the translations if a scale was applied
    if scale_doc:
        f = rig["raw"].split()
        i, parts = 4, f[:4]
        for _ in range(int(f[1]) - 1):
            cid = int(f[i + 1])
            has = int(f[i + 2])
            parts += f[i:i + 3]
            if has:
                q = f[i + 3:i + 7]
                t = [f"{x:.10f}" for x in rig["sensors"][cid]["t"]]
                parts += q + t
                i += 10
            else:
                i += 3
        (out / "rigs.txt").write_text(
            "# Rig calib list\n" + " ".join(parts) + "\n", encoding="utf-8")
    else:
        shutil.copy(sf / "rigs.txt", out / "rigs.txt")

    (out / "prefixes.json").write_text(
        json.dumps({str(k): v for k, v in sorted(prefixes.items())},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # rig_config.json with full intrinsics and extrinsics (read by preprocessing).
    # The values are written out, so the rig folder does not depend on the
    # calibration model and need not carry tens of MB of reconstruction files.
    entries = []
    for cid in sorted(cams):
        e = {"image_prefix": prefixes.get(cid, f"cam_{cid}/"),
             "camera_model_name": cams[cid]["model"],
             "camera_params": cams[cid]["params"]}
        if cid == rig["ref"]:
            e["ref_sensor"] = True
        else:
            s = rig["sensors"].get(cid)
            if s:
                e["cam_from_rig_rotation"] = s["q"]
                e["cam_from_rig_translation"] = s["t"]
        entries.append(e)
    (out / "rig_config.json").write_text(
        json.dumps([{"cameras": entries}], indent=2, ensure_ascii=False),
        encoding="utf-8")

    # Always written: otherwise "deliberately unscaled" and "the file got
    # lost" would look identical downstream, and the latter is data damage
    # that must stop processing.
    (out / "scale.json").write_text(
        json.dumps(scale_doc if scale_doc else
                   {"unit": "sfm", "metric": False,
                    "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "no --edge given; no scale reference",
                    "edges": []},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable summary: what matters is consistency, not the values themselves
    L = ["# Reusable rig parameters", "",
         f"calibrated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"set       : {calib.name} ({len(shots)} sets x {len(names)} cameras)",
         f"reference : CAMERA {rig['ref']} ({prefixes.get(rig['ref'],'?')})", "",
         "## Intrinsics"]
    for cid, c in sorted(cams.items()):
        p = c["params"]
        L.append(f"  CAMERA {cid} {prefixes.get(cid,'?'):10s} {c['model']} "
                 f"{c['w']}x{c['h']}")
        L.append(f"    fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.1f} cy={p[3]:.1f}"
                 f"   principal point offset ({p[2]-c['w']/2:+.1f}, {p[3]-c['h']/2:+.1f}) px")
        if len(p) >= 8:
            L.append(f"    k1={p[4]:+.4f} k2={p[5]:+.4f} "
                     f"p1={p[6]:+.4f} p2={p[7]:+.4f}")
    L += ["", "## Camera positions (relative to the reference, C = -R^T t)"
          + (", in metres" if scale_doc else ", in arbitrary SfM units")]
    for cid in sorted(centers):
        c = centers[cid]
        L.append(f"  CAMERA {cid} {prefixes.get(cid,'?'):10s} "
                 f"({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")
    if edge_rows:
        L += ["", "## Comparison with measured distances"]
        for r in edge_rows:
            L.append(f"  {r['pair'][0]}-{r['pair'][1]}: "
                     f"reconstructed {r['reconstructed_mm']:7.1f} mm  "
                     f"measured {r['measured_mm']:7.1f} mm  diff {r['diff_mm']:+.1f} mm")
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------------------------------------------------- 8. acceptance
    step(8, TOT, "Acceptance")
    n_expected = len(names) * len(shots)
    reg = ana_f.get("registered", 0)
    rb = ana_b.get("reproj_px", 99)
    rf = ana_f.get("reproj_px", 99)
    checks = [
        ("Registration", f"{reg:.0f}/{n_expected}", reg >= n_expected, "100%"),
        ("Focal spread", f"{spread:.1f} %", spread <= TH_FOCAL_SPREAD,
         f"<= {TH_FOCAL_SPREAD} %"),
        ("Stage B reproj.", f"{rb:.3f} px", rb < TH_REPROJ_B, f"< {TH_REPROJ_B} px"),
        ("Rig BA reproj.", f"{rf:.3f} px", rf < TH_REPROJ_RIG, f"< {TH_REPROJ_RIG} px"),
        ("Rig constraint cost", f"+{rf-rb:.3f} px", (rf - rb) < TH_REPROJ_RISE,
         f"< +{TH_REPROJ_RISE} px"),
    ]
    if edge_rows:
        worst = max(abs(r["diff_mm"]) for r in edge_rows)
        # Two decimals: with `.1f`, 2.05 would print as "2.0" while the
        # comparison uses the real value, showing "2.0 mm <= 2.0 mm" as a failure.
        checks.append(("Max edge difference", f"{worst:.2f} mm",
                       worst <= TH_EDGE_MM, f"<= {TH_EDGE_MM} mm"))
    else:
        print("  ! no --edge given; the strongest check (against physical measurement) was not done")

    print()
    for label, got, ok, th in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {label:<20}{got:>14}   (threshold {th})")
    passed = all(c[2] for c in checks)

    report = {
        "schema": 1,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(calib),
        "cameras": names, "shots": len(shots), "resolution": [W, H],
        "hfov_deg": a.hfov, "focal_prior_px": round(f0, 1),
        "stage_a": {"focals": [round(x, 1) for x in focals],
                    "mean": round(f1, 1), "spread_pct": round(spread, 2),
                    **{k: round(v, 4) for k, v in ana_a.items()}},
        "dropped_blurry_shots": dropped,
        "stage_b": {k: round(v, 4) for k, v in ana_b.items()},
        "rig_ba": {k: round(v, 4) for k, v in ana_f.items()},
        "principal_offsets": {prefixes.get(cid, str(cid)).rstrip("/"):
                              [round(c["params"][2] - c["w"] / 2, 2),
                               round(c["params"][3] - c["h"] / 2, 2)]
                              for cid, c in sorted(cams.items())},
        "scale": scale_doc,
        "checks": [{"name": c[0], "value": c[1], "pass": c[2],
                    "threshold": c[3]} for c in checks],
        "passed": passed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "calib_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  {'PASSED' if passed else 'FAILED'}"
          f"  took {time.time()-t0:.0f} s")
    print(f"  -> {out} ({size/1024:.1f} KB, portable, reusable across sessions)")
    for f in sorted(out.iterdir()):
        print(f"      {f.name}")
    if not passed:
        print("\n  If it failed:")
        print("    low registration -> not enough board texture or too few sets; every set must be seen by all cameras")
        print("    large reprojection error -> the board needs depth variation and roll; avoid facing a flat wall")
        print("    large edge difference -> measure again with calipers; measure lens centers, not housings")
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
