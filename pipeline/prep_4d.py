#!/usr/bin/env python3
"""prep_4d.py - recording -> 4DGS training set

    python prep_4d.py <shooting root> [options]            <- recommended
    python prep_4d.py <recording folder> <output folder>   <- older form, still supported

## Give only the shooting root

The other three paths can all be derived from it:

| What | Where it comes from |
|---|---|
| recording | the only folder under the root that contains `session.json` |
| calibration | `<root>/rig/`. If there is none yet, bring one in with `--rig` (needed once) |
| output | `<root>/stg<number of valid time steps>`, **named automatically** |

The output name is only known after grouping (the number of valid time steps
is known only then), so it is not reasonable to ask for it up front.

## Two stages

    stage 1 (no calibration needed)  grouping, phase report, time labels, times*.json, time fields of window.json
    stage 2 (needs --rig)            undistortion (centred principal point), colmap_N, initial point cloud, geometry fields of window.json

Without `--rig` only stage 1 runs. **Stage 1 outputs do not change when a
calibration is added later.**

## Options

    --rig <rig folder>        stage 2 runs only with it
    --span-ms 25              grouping threshold on span = max(t) - min(t)
    --steps N                 sample N time steps; default is every valid group
    --anchor cam_A            reference camera for pairing (does not affect time labels)
    --background 0.0          **background colour, decided once here**; training and playback follow it
                              (default black)

## Differences from a naive approach

| | naive | here |
|---|---|---|
| time label | the time of cam_A -> 11.10 ms error | **midpoint of the group** -> 7.62 ms |
| time axis | `t = n/duration` (uniform) -> **up to 242 ms distortion** | **real time**, written to `times_group.json` |
| threshold | `max|t_c - t_anchor|` <= 20 ms | **`span`** <= 25 ms |
| phase report | none | written every time |

Fixing the time label gained +0.45 dB and was the only improvement that was
visible by eye.

## Dependencies

Stage 1 needs only **stdlib + numpy**. Stage 2 also needs `cv2` (undistortion) and the COLMAP executable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SCHEMA = "prep_4d/1"


# ========================================================== reading the recording
def load_streams(rec: Path) -> list[str]:
    """Decide the streams from `session.json`; **do not count cam_* folders**.

    The recorder states this explicitly: once audio was added, scanning
    folders would also take `audio/` for a camera.

    The number and names of the cameras are **always read from the file**,
    never hard-coded to 4: a fifth, central camera is planned.
    """
    f = rec / "session.json"
    if not f.is_file():
        cams = sorted(d.name for d in rec.iterdir()
                      if d.is_dir() and (d / "index.jsonl").is_file())
        print(f"  ! no session.json; falling back to scanning folders -> {cams}")
        return cams
    doc = json.loads(f.read_text(encoding="utf-8"))
    if "streams" in doc:                       # recorder format
        return [s["name"] for s in doc["streams"] if s.get("kind") != "audio"]
    if "cameras" in doc:                       # older format
        return [c["name"] for c in doc["cameras"]]
    raise SystemExit(f"  ERROR unrecognized session.json: {sorted(doc)}")


def session_id(rec: Path) -> dict:
    """Identity of this recording. Returns the fields to write into `window.json`.

    **The folder name alone is not enough.** The folder may be renamed at any
    time. That does not damage data (the dataset is self-contained and only
    stores a name string), but `source_recording` would then point at
    something that does not exist, **without any error**, and it would no
    longer be possible to find out which recording a model came from or what
    the exposure was.

    So two are stored:

        source_recording   folder name. Easy to read and recognize, but may be changed
        captured_at        timestamp written into session.json at recording time.
                           **Renaming cannot change it** - to find the original
                           recording, grep every session.json for it

    The two are not identical: the folder name is the time recording
    **started**, `captured_at` is when the recorder wrote the manifest, a dozen
    seconds apart. That is not a bug; they are two different moments, so they
    are **not checked against each other**.

    The recorder should eventually write an explicit `session_id` (ideally
    equal to the folder name); that would be the real authoritative identity.
    """
    out = {"source_recording": rec.name}
    f = rec / "session.json"
    if f.is_file():
        doc = json.loads(f.read_text(encoding="utf-8"))
        # Capture conditions must travel with the data.
        #
        # Exposure time directly sets the **speed limit for motion blur**:
        # 3300 us corresponds to 303 px/s before things blur, and blur is
        # **irreversible** (noise can be averaged out by reconstruction, blur
        # cannot). At 6600 us the limit is half. The first question when a
        # model looks blurred is the shutter time; if it is not in the
        # dataset, it cannot be traced later.
        cap = doc.get("capture") or {}
        for k in ("exposure_us", "gain", "fps", "resolution"):
            if k in cap:
                out[k] = cap[k]
        if "warmup_s" in doc:
            out["warmup_s"] = doc["warmup_s"]
        if doc.get("session_id"):
            out["session_id"] = str(doc["session_id"])
        # The two schemas use different field names:
        #    recorder -> created_at / older format -> captured_at
        for k in ("captured_at", "created_at"):
            if doc.get(k):
                out["captured_at"] = str(doc[k])
                break
    if "captured_at" not in out and "session_id" not in out:
        print("  ! session.json has neither captured_at nor session_id - "
              "once the recording folder is renamed, the original recording cannot be traced")
    return out


def load_index(rec: Path, name: str) -> list[dict]:
    p = rec / name / "index.jsonl"
    if not p.is_file():
        raise SystemExit(f"  ERROR missing {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ========================================================== grouping
def build_groups(idx: dict[str, list[dict]], anchor: str, span_ms: float):
    """For every anchor frame, choose the group with the **smallest overall time span**.

    Returns (valid groups, all attempts). Each group is a dict:
        n_anchor  index of the anchor frame
        pick      {cam: frame_idx}
        ts        {cam: ts_ns}
        off       {cam: offset from the anchor in ms, **signed**}
        span      max(ts) - min(ts), in ms
        mid       (max+min)/2, in ns   <- this is the time label

    The threshold measures **span**, not `max|t_c - t_anchor|`.

    They differ: measured, cam_B is 4.12 ms **before** cam_A and cam_C/D are
    10.61 / 11.04 ms after. The largest offset from the anchor is 11.04, but
    the span is 15.16. **Offsets from the anchor underestimate the real time
    span of the group.**

    **So the selection must also use span; each camera cannot simply take the
    frame closest to the anchor.**

    That greedy approach minimizes `|t_c - t_anchor|`, which is not what the
    threshold measures. When the anchor sits at the **edge** of the phase
    cluster, taking the nearest frame per camera produces combinations with a
    large span, which the threshold then rejects wholesale. Measured on one
    recording (phases 0 / +2.9 / +12.0 / +14.9):

    | strategy | valid |
    |---|---|
    | greedy anchor=cam_A | **53 / 301** |
    | greedy anchor=cam_C | 292 / 300 |
    | **smallest span (any anchor)** | **300 / 301** |

    The footage had 300 good groups; **the algorithm threw away 82%**.
    Now each camera contributes **the frames before and after** as candidates,
    and all combinations are tried for the smallest span (4 cameras = 8
    combinations, negligible cost).

    **A 5.5x difference from changing the anchor is itself a sign the
    algorithm is wrong**: a correct method should not be sensitive to the
    anchor (smallest span gives 300 for both). Smallest span can never be
    worse than greedy, so this is a strict improvement; an existing dataset
    re-run produced a byte-identical `times_group.json`.

    The time label is `mid`, not the anchor time. The anchor sits at the edge
    of the phases; using it as the label gives 11.10 ms error per image, the
    midpoint gives 7.62 ms.
    """
    import bisect
    import itertools

    cams = sorted(idx)
    base = idx[anchor]
    others = [c for c in cams if c != anchor]
    tarr = {c: [e["ts_ns"] for e in idx[c]] for c in others}

    ok, allg = [], []
    for e in base:
        t0 = e["ts_ns"]
        # Each camera contributes **the frames before and after** as
        # candidates; all combinations are tried for the smallest span.
        # Taking each camera's frame nearest the anchor would minimize
        # |t_c - t_anchor|, while the threshold measures **span**.
        cand = []
        for c in others:
            arr, ta = idx[c], tarr[c]
            j = bisect.bisect_left(ta, t0)
            cand.append([arr[k] for k in (j - 1, j) if 0 <= k < len(arr)])

        best = None
        for combo in itertools.product(*cand):
            tt = [t0] + [x["ts_ns"] for x in combo]
            sp = max(tt) - min(tt)
            if best is None or sp < best[0]:
                best = (sp, combo)
        pick = {anchor: e["i"]}
        ts = {anchor: t0}
        for c, x in zip(others, best[1]):
            pick[c], ts[c] = x["i"], x["ts_ns"]
        lo, hi = min(ts.values()), max(ts.values())
        g = dict(n_anchor=e["i"], pick=pick, ts=ts,
                 off={c: (ts[c] - t0) / 1e6 for c in cams},
                 span=(hi - lo) / 1e6, mid=(lo + hi) / 2)
        allg.append(g)
        if g["span"] <= span_ms:
            ok.append(g)
    return ok, allg


def frame_period_ms(idx: dict[str, list[dict]], anchor: str) -> float:
    d = np.diff([e["ts_ns"] for e in idx[anchor]]) / 1e6
    return float(np.median(d)) if len(d) else 33.333


# ========================================================== phase report
def phase_report(cams, ok, allg, span_ms, period, idx) -> str:
    """Written every time. If the phases stop being constant, the hardware has a problem.

    Measured: cam_B/C/D relative to cam_A at **-4.12 / +10.61 / +11.04 ms,
    varying +-0.06 within 10 s** - the phases are constant. So what the
    threshold rejects is **not phase but dropped frames**. The two must be
    reported separately, or a hardware fault would be misread as "sync got
    worse".
    """
    L = [f"# Phase report   schema={SCHEMA}", ""]
    L += [f"{len(cams)} cameras: {', '.join(cams)}",
          f"anchor frames {len(allg)}   valid {len(ok)} ({len(ok)/max(len(allg),1)*100:.1f}%)"
          f"   threshold span <= {span_ms} ms   median frame period {period:.2f} ms", ""]

    # Median and MAD, not mean and standard deviation.
    #    A few frames jitter by 3-4 ms (not a dropped frame, but enough to push
    #    the standard deviation to 1.8 ms), which would look like drifting
    #    phase. The measured MAD is only 0.0x ms: **almost every frame has a
    #    constant phase and only a few jump**. Those must be seen separately.
    L += ["## Offset of each camera from the anchor (signed, valid groups only)", ""]
    L += ["| camera | median | MAD | jitter >1ms | min | max |",
          "|---|---|---|---|---|---|"]
    jit_total = 0
    for c in cams:
        a = np.array([g["off"][c] for g in ok])
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        jit = int((np.abs(a - med) > 1.0).sum())
        jit_total += jit
        L.append(f"| {c} | {med:+.2f} | {mad:.3f} | {jit} / {len(a)} | "
                 f"{a.min():+.2f} | {a.max():+.2f} | ms")
    L += ["",
          "**How to read it:**", "",
          "| what | meaning |",
          "|---|---|",
          "| **MAD << 1 ms** | the **median** phase is stable; hardware is fine (measured 0.0x) |",
          "| MAD grows | that camera is **drifting continuously** - a hardware or driver problem |",
          "| median changed | **wiring or start-up order changed** - calibration may need redoing |",
          "| min/max form two clusters | normal; see \"Frame intervals\" below |",
          "",
          f"{jit_total} entries with jitter >1ms. **This is usually not the camera jittering**;",
          "it is caused by what follows.", ""]

    # Frame interval diagnosis - without this section, the jitter above would
    # be misread as unstable hardware.
    #
    #    Measured: all four cameras have **bimodal 32 / 36 ms** frame
    #    intervals, not a steady 33.33. The timestamps themselves are not
    #    quantized (235 distinct remainders).
    #
    #    The anchor's grid is irregular -> the "nearest frame" of the other
    #    cameras jumps between two neighbours -> relative offsets look
    #    bimodal, differing by exactly the 32/36 difference (4 ms).
    #    **That is a result of pairing, not camera jitter.** Real timestamps
    #    are entirely unaffected.
    L += ["## Frame intervals", "",
          "| camera | mean | median | three most common values |", "|---|---|---|---|"]
    for c in cams:
        d = np.diff([e["ts_ns"] for e in idx[c]]) / 1e6
        u, n = np.unique(np.round(d), return_counts=True)
        top = sorted(zip(u, n), key=lambda x: -x[1])[:3]
        L.append(f"| {c} | {d.mean():.2f} | {np.median(d):.2f} | "
                 + ", ".join(f"{v:.0f} ms x{k}" for v, k in top) + " |")
    L += ["",
          "**Intervals are usually bimodal (for example 32 / 36 ms), not a steady 33.33.**",
          "The anchor's grid is irregular, so the other cameras' nearest frame jumps between",
          "two neighbours and relative offsets look like jitter. **That is a result of pairing,",
          "not unstable cameras** - time labels use real timestamps and are unaffected.", "",
          "A camera with a clearly larger mean interval is **dropping frames**"
          " (for example 37.99 ms against 33.4-34.0 for the others).", ""]

    s = np.array([g["span"] for g in ok])
    L += ["## Group time span (valid groups)", "",
          f"median {np.median(s):.2f}   p90 {np.percentile(s,90):.2f}   max {s.max():.2f} ms",
          f"-> **time label error per image <= span/2 = {s.max()/2:.2f} ms**", ""]

    bad = [g for g in allg if g["span"] > span_ms]
    drop = [g for g in bad if max(abs(v) for v in g["off"].values()) > 0.75 * period]
    L += ["## Discarded groups", "",
          f"{len(bad)} in total ({len(bad)/max(len(allg),1)*100:.1f}%)", "",
          f"- **dropped frames** {len(drop)}: a camera had to pair with an image more than 0.75 frame periods away",
          f"- phase too large {len(bad)-len(drop)}: the phase difference alone exceeds the threshold", ""]
    if drop:
        cnt = {}
        for g in drop:
            for c, v in g["off"].items():
                if abs(v) > 0.75 * period:
                    cnt[c] = cnt.get(c, 0) + 1
        L += ["Sources of dropped frames: " + ", ".join(f"{c} {n} times" for c, n in
                                                    sorted(cnt.items(), key=lambda x: -x[1])), ""]
        L += ["> Dropped frames are usually a **wiring or USB topology** problem, not a sync problem.",
              "> Measured: without a USB hub, one Pi port per camera -> 0 / 14,386 dropped frames.", ""]
    return "\n".join(L)


# ========================================================== sampling
def select(ok, n):
    """Sample N time steps.

    Sampling evenly **by index** while labelling time uniformly
    (`t = n/duration`) would be inconsistent and distort the time axis by
    up to **242 ms**.

    Sampling by index is **fine here**, because time labels come from the
    real times in `times_group.json` and no longer rely on "uniform index
    implies uniform time".
    """
    if not n or n >= len(ok):
        return ok
    sel = np.linspace(0, len(ok) - 1, n).round().astype(int)
    return [ok[i] for i in sel]


# ========================================================== outputs
def write_stage1(out: Path, cams, sel, rec: Path, span_ms, background=0.0):
    ident = session_id(rec)
    print("  recording identity " + "   ".join(f"{k}={v}" for k, v in ident.items()))
    """Write every time-label variant at once.

    Switching schemes only needs a different file and a new training run
    (about 40 minutes), not a reshoot or algorithm change. The variants are
    kept because **fast-motion footage will likely require redoing this
    comparison** (at 1 m/s the in-group span equals 10.2% depth error; then
    continuous time may win instead).
    """
    out.mkdir(parents=True, exist_ok=True)
    lo = min(g["ts"][c] for g in sel for c in cams)
    hi = max(g["ts"][c] for g in sel for c in cams)
    span_ns = hi - lo

    tg = {f"{n}/{c}.jpg": (g["mid"] - lo) / span_ns
          for n, g in enumerate(sel) for c in cams}          # grouped (used)
    ti = {f"{n}/{c}.jpg": (g["ts"][c] - lo) / span_ns
          for n, g in enumerate(sel) for c in cams}          # continuous time
    (out / "times_group.json").write_text(json.dumps(tg, indent=1), encoding="utf-8")
    (out / "times.json").write_text(json.dumps(ti, indent=1), encoding="utf-8")

    with (out / "groups.jsonl").open("w", encoding="utf-8") as f:
        for n, g in enumerate(sel):
            f.write(json.dumps({"n": n, "pick": g["pick"], "ts_ns": g["ts"],
                                "off_ms": g["off"], "span_ms": g["span"],
                                "mid_ns": g["mid"]}, ensure_ascii=False) + "\n")

    steps = np.array([g["mid"] for g in sel])
    d = np.diff(steps) / 1e6
    win = {"schema": SCHEMA, "cameras": cams,
           # The background colour is decided once here; training and
           #    playback both read it. Final pixel = sum(Gaussian
           #    contributions) + background x prod(1 - alpha). Areas that
           #    "should be" the training background colour can be left empty
           #    (the loss cannot tell), and those holes appear when playing
           #    with another background. Measured on a dark scene: 34% of
           #    pixels had alpha < 0.99, all in dark areas.
           "background": float(background),
           "nsteps": len(sel), "duration_s": round(span_ns / 1e9, 4),
           "step_ms_median": round(float(np.median(d)), 2) if len(d) else None,
           "step_ms_min": round(float(d.min()), 2) if len(d) else None,
           "step_ms_max": round(float(d.max()), 2) if len(d) else None,
           "group_span_ms_max": round(max(g["span"] for g in sel), 2),
           # Method of time labelling. This field was originally called
           #    `label`, which clashed with the human-readable title.
           "time_label": "midpoint", "span_threshold_ms": span_ms,
           # -- source recording (recorded automatically) -----------------
           # These fields are **only for tracing**: after the shooting
           #    folder is renamed, the original recording can still be
           #    found. The name shown by the player **does not come from
           #    here**; it is read from the path at load time.
           **ident,                          # source_recording / captured_at / session_id
           "_geometry": "geometry fields (win/ctr/fwd/fov/wh/unit_mm) are added only by stage 2 (--rig)"}
    (out / "window.json").write_text(
        json.dumps(win, ensure_ascii=False, indent=2), encoding="utf-8")
    return win


# ========================================================== stage 2
#
# recording + calibration -> trainable dataset (colmap_N x N time steps)
#
# Steps and reasons are in each function. Overall order:
#
#   0. copy the rig into the shooting root   <- only this copy is read afterwards
#   1. undistortion maps (centred principal point)
#   2. pick the group with the smallest span -> undistort -> COLMAP for poses
#   3. MVS dense reconstruction -> initial point cloud
#   4. compute unit_mm (least squares of the six physical edges against this reconstruction)
#   5. write colmap_N per time step (images + sparse/0)
#   6. poses_bounds.npy / cfg_lite.json / geometry fields of window.json

def copy_audio(rec: Path, out: Path, sel, cams):
    """Copy the recording's `audio/` into the dataset unchanged; return the `window.json` audio block.

    Without audio returns None. `window.json` then has **no audio block** and
    the player silently plays without sound, which is correct.

    ## How t0_offset_s is computed

    Dataset `t=0` corresponds to `lo` (the smallest camera timestamp, see
    write_stage1), and the player converts with
    `audio seconds = t x duration_s + t0_offset_s`. So

        t0_offset_s = (lo - time of the first audio sample) / 1e9

    plus `offset_ms` (residual audio/video offset measured by
    sync_measure.py; positive means audio timestamps are late, so audio is
    moved earlier).

    A null offset_ms counts as 0. **Do not guess a value** (the same reason
    unit_mm = 1000.0 was removed: do not disguise "unknown" as "known").
    No warning is printed either: unmeasured is the normal case, and a
    warning every time becomes noise nobody reads. window.json always keeps
    offset_ms / offset_ms_source for reference.

    ## Gain

    I2S delivers linear PCM; raw is too quiet to hear (measured -60 LUFS)
    and the player has no gain stage. It is computed here (ffmpeg is
    available on the PC) and written into the audio block; playback applies
    it. The recorder still stores only raw - gain is a presentation
    decision, not a measurement.
    """
    src = rec / "audio"
    wav = src / "audio.wav"
    if not wav.is_file():
        return None

    dst = out / "audio"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copyfile(f, dst / f.name)

    idx = {}
    ij = src / "index.json"
    if ij.is_file():
        idx = json.loads(ij.read_text(encoding="utf-8"))

    lo = min(g["ts"][c] for g in sel for c in cams)
    start = idx.get("start_ts_ns")
    if start is None:
        print("  ! audio/index.json has no start_ts_ns - cannot align; no audio block written")
        return None

    off_ms = idx.get("offset_ms")
    t0 = (lo - start) / 1e9 + (off_ms or 0.0) / 1000.0

    blk = {
        "path": "audio/audio.wav",
        "t0_offset_s": round(t0, 6),
        "sample_rate_measured_hz": idx.get("sample_rate_measured_hz"),
        "offset_ms": off_ms,
        "offset_ms_source": ("sync_measure.py" if off_ms is not None
                             else "not measured (taken as 0)"),
        "xruns": len(idx.get("xruns") or []),
    }
    if idx.get("mic_spacing_mm"):
        blk["mic_spacing_mm"] = idx["mic_spacing_mm"]

    try:
        import audio_level as AL
        lufs, tp, lra = AL.measure(dst / "audio.wav")
        gain, why = AL.linear_gain(lufs, tp)
        blk.update({"lufs_integrated": lufs, "true_peak_dbtp": tp,
                    "lra_lu": lra, "gain_db": gain,
                    "gain_source": f"EBU R128, target {AL.TARGET_LUFS} LUFS ({why})"})
        if lufs is not None:
            print(f"  audio {lufs:.1f} LUFS / {tp:.1f} dBTP"
                  f" -> gain {gain:+.1f} dB ({why})")
    except Exception as e:                             # noqa: BLE001
        print(f"  ! loudness could not be measured ({e}) - no gain written; playback will be very quiet")

    dur = (idx.get("frames") or 0) / (idx.get("sample_rate_measured_hz")
                                      or idx.get("sample_rate") or 48000)
    print(f"  audio -> {dst.name}/   {dur:.2f} s   t0_offset {t0:+.3f}s")
    return blk


def window_corners(cams, C, ctr):
    """Window frame corners: the **convex hull** of the camera centres on the rig plane, ordered by angle around the centroid.

    Returns a list of camera names (the order walks the polygon).

    Why angle sorting alone is not enough

    Plain `sorted(cams, key=atan2(...))` **equals** the convex hull only when
    every camera lies on the hull boundary. That holds for 4 cameras at the
    corners of a rectangle, so it never failed.

    It breaks as soon as a camera lies **inside**: it is still placed in the
    boundary ring, and the polygon folds into the centre and back out, losing
    a piece. With synthetic coordinates, "4 corners + centre" gave 25% less
    area than the real convex hull.

    That is exactly the planned fifth (central) camera, which sits inside the
    hull **deliberately** (it provides an interior test view), so this fix
    is its prerequisite.

    ## Method

    Compute hull membership first, then keep the previous angle sort for the
    members. The output for 4 cameras is **unchanged** (they are all on the
    hull), which keeps the risk minimal.

    Uses monotone chain (no new dependency). Collinear points **on** a hull
    edge are excluded - that does not change the frame's shape, and one
    fewer corner means one fewer rounded corner, so the edge is cleaner.

    The cameras are not exactly coplanar (0.5-1.5 mm off the plane), so
    points are projected onto the rig plane first rather than using world x/y
    directly.
    """
    if len(cams) < 3:                    # 2 cameras do not form a polygon; return as is
        return list(cams)

    P = np.array([C[c] for c in cams], float)
    _, _, Vt = np.linalg.svd(P - ctr)
    right, down = Vt[0], Vt[1]           # two basis vectors of the rig plane
    uv = [(float(np.dot(P[i] - ctr, right)),
           float(np.dot(P[i] - ctr, down))) for i in range(len(cams))]

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    idx = sorted(range(len(cams)), key=lambda i: uv[i])
    lower = []
    for i in idx:
        while len(lower) >= 2 and cross(uv[lower[-2]], uv[lower[-1]], uv[i]) <= 0:
            lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(idx):
        while len(upper) >= 2 and cross(uv[upper[-2]], uv[upper[-1]], uv[i]) <= 0:
            upper.pop()
        upper.append(i)
    on_hull = set(lower[:-1] + upper[:-1])

    if len(on_hull) < 3:                 # degenerate (all collinear) - do not emit a broken polygon
        return list(cams)

    inside = [cams[i] for i in range(len(cams)) if i not in on_hull]
    if inside:
        print(f"  window corners from the convex hull: {', '.join(inside)} inside, not a corner"
              f" (still used for reconstruction)")

    # The angle sort **uses world x/y exactly as before**; interior points are
    #    only filtered out first.
    #
    #    Sorting on the rig-plane basis (the same basis as the hull) looks more
    #       consistent, but it changed the order for existing data from C,B,A,D
    #       to B,C,D,A - **the same polygon, wound in reverse**, because the
    #       sign of an SVD basis is arbitrary. The shape is the same, but the
    #       winding affects the direction of the rounded-corner path, which is
    #       outside the scope of this change, so the original sort is kept.
    #
    #    This is safe: COLMAP's world frame **is the rig frame** (origin at
    #    ref_sensor), and the rig plane differs from world xy only by the
    #    mechanical tolerance of the cameras (1-2 deg), so the angle order in
    #    world xy matches the order on the rig plane. Hull membership is still
    #    decided on the best-fit plane (a property of the point set,
    #    independent of the basis).
    return sorted((cams[i] for i in on_hull),
                  key=lambda c: math.atan2((C[c] - ctr)[1], (C[c] - ctr)[0]))


COLMAP_GUESSES = [
    str(Path(__file__).resolve().parent.parent / "third_party" / "colmap" / "bin" / "colmap.exe"),
    r"C:\Program Files\COLMAP\colmap.exe",
    "/usr/local/bin/colmap", "/usr/bin/colmap", "colmap",
]
# Triangulation with fixed poses: the reprojection error reflects the
#    geometric consistency of "calibration vs this recording". Calibration
#    itself is 1.23 px (100 images + rig BA); here there are only four images
#    and the poses are not optimized, so it is slightly higher. Above this
#    value the rig has been moved.
TH_TRI_REPROJ = 2.5
MVS_MIN_PIXELS = 3          # how many cameras stereo_fusion requires to agree; 2/3/4/5 were swept, 3 is best
MVS_MEDIAN_HALF = 4         # MVS uses the temporal median of +-4 groups (9 frames); see build_mvs_images()
JPEG_Q = 95


def find_colmap(explicit=None) -> str:
    import shutil as _sh
    for c in ([explicit] if explicit else []) + COLMAP_GUESSES:
        if not c:
            continue
        if os.path.sep in str(c) or "/" in str(c):
            if Path(c).exists():
                return str(c)
        elif _sh.which(c):
            return c
    raise SystemExit("  ERROR COLMAP not found. Give the executable path with --colmap.")


def colmap_run(exe, *args, check=True):
    import subprocess
    # CREATE_NO_WINDOW: when run from a GUI (the studio), the parent has no
    # console and Windows opens a new console window for every COLMAP
    # subprocess. The output is already captured by capture_output; the
    # window would only get in the way.
    r = subprocess.run([exe] + [str(a) for a in args],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if check and r.returncode != 0:
        print(f"\n  ERROR COLMAP failed: colmap {args[0]}")
        print((r.stderr or r.stdout)[-1500:])
        raise SystemExit(1)
    return (r.stdout or "") + (r.stderr or "")


def read_cameras_txt(p: Path) -> dict:
    out = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.startswith("#") or not l.strip():
            continue
        f = l.split()
        out[int(f[0])] = {"model": f[1], "w": int(f[2]), "h": int(f[3]),
                          "params": [float(x) for x in f[4:]]}
    return out


def undistort_maps(rig: Path, cams: list[str]):
    """One cv2 undistortion map per camera, with the **principal point of the new intrinsics forced to the centre**.

    Why the principal point must be centred: the 3DGS/STG rasterizer takes
       only `tanfovx`/`tanfovy` and **has no principal point parameter**. The
       real cameras' principal point offset (measured -28 to +26 px) must be
       absorbed during undistortion, or the projection shifts as a whole and
       the reconstruction compensates with skewed geometry. Centring the
       principal point gained **+4.3 dB** PSNR, the largest single change.

    Output size: keep fx/fy and take the largest centred rectangle without
    invalid pixels. Undistortion pulls the image inward and leaves a border
    without source pixels; keeping the original size would leave black
    borders that would be trained as real content.
    """
    import cv2
    cam_tbl = read_cameras_txt(rig / "cameras.txt")
    prefixes = {int(k): v.rstrip("/") for k, v in
                json.loads((rig / "prefixes.json").read_text(encoding="utf-8")).items()}
    name2id = {v: k for k, v in prefixes.items()}
    missing = [c for c in cams if c not in name2id]
    if missing:
        raise SystemExit(f"  ERROR these cameras are not in the calibration: {missing}\n"
                         f"    the calibration has: {sorted(name2id)}\n"
                         f"    camera names are **decided by sort order**; after changing ports, reshoot the calibration set")

    W0, H0 = cam_tbl[name2id[cams[0]]]["w"], cam_tbl[name2id[cams[0]]]["h"]

    # first pass at the original size to find the region valid for all cameras
    valid = np.ones((H0, W0), bool)
    K_new = {}
    for c in cams:
        p = cam_tbl[name2id[c]]["params"]
        fx, fy, cx, cy = p[:4]
        k1, k2, p1, p2 = p[4:8]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Kn = np.array([[fx, 0, W0 / 2.0], [0, fy, H0 / 2.0], [0, 0, 1]])
        K_new[c] = (K, np.array([k1, k2, p1, p2]), Kn)
        m1, m2 = cv2.initUndistortRectifyMap(K, np.array([k1, k2, p1, p2]), None,
                                             Kn, (W0, H0), cv2.CV_32FC1)
        valid &= (m1 >= 0) & (m1 <= W0 - 1) & (m2 >= 0) & (m2 <= H0 - 1)

    # grow from the centre to find the largest fully valid rectangle (uniform scaling; simple and sufficient)
    cx0, cy0 = W0 // 2, H0 // 2
    best = (0, 0)
    for s in np.arange(1.0, 0.70, -0.005):
        w, h = int(W0 * s) // 2 * 2, int(H0 * s) // 2 * 2
        x0, y0 = cx0 - w // 2, cy0 - h // 2
        if valid[y0:y0 + h, x0:x0 + w].all():
            best = (w, h)
            break
    W, H = best if best[0] else (W0, H0)
    if not best[0]:
        print("  ! no fully valid centred rectangle found; using the original size (black borders possible)")

    maps = {}
    for c in cams:
        K, D, _ = K_new[c]
        fx, fy = K[0, 0], K[1, 1]
        Kn = np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]])
        maps[c] = cv2.initUndistortRectifyMap(K, D, None, Kn, (W, H), cv2.CV_32FC1)
    fx = float(np.mean([K_new[c][0][0, 0] for c in cams]))
    fy = float(np.mean([K_new[c][0][1, 1] for c in cams]))
    print(f"  undistortion {W0}x{H0} -> {W}x{H} (principal point centred, cropping "
          f"{(1 - W * H / (W0 * H0)) * 100:.1f}% invalid border)")
    return maps, W, H, {c: (K_new[c][0][0, 0], K_new[c][0][1, 1]) for c in cams}


class FrameReader:
    """Reads raw JPEG bytes from stream.mjpg by seeking to the off/len in the index.

    **No full video decoding** and no double re-encoding. The recorder writes
       raw JPEGs; this only picks out the needed images, undistorts them and
       saves them once.
    """

    def __init__(self, rec: Path, cams):
        self.f = {c: open(rec / c / "stream.mjpg", "rb") for c in cams}
        self.off = {}
        for c in cams:
            self.off[c] = {e["i"]: (e["off"], e["len"])
                           for e in load_index(rec, c)}

    def get(self, cam, i):
        import cv2
        o, L = self.off[cam][i]
        self.f[cam].seek(o)
        return cv2.imdecode(np.frombuffer(self.f[cam].read(L), np.uint8),
                            cv2.IMREAD_COLOR)

    def close(self):
        for f in self.f.values():
            f.close()


def poses_from_rig(rig: Path, cams: list[str], allow_unscaled=False):
    """Poses **come directly from the calibrated extrinsics**; they are not re-solved from this recording's four images.

    Why they are not re-solved

    Running the COLMAP mapper on the four images of the smallest-span group
    is underconstrained (four images, 148 mm baseline): two of the six edges
    were off by **+2.21 / -2.33 mm** (threshold 2 mm), both involving cam_D -
    one camera paired slightly wrong moves its position by 2 mm.

    Calibration uses **100 images + rig BA**; the six edges differ by at most
    1.6 mm and the rigid-rig cost is only +-0.043 px. **That is far better,
    and it is the same rig.**

    This also matches the design: **window = rig = origin; poses follow the
    calibration**. The rig is rigid, so poses do not change between time
    steps within a recording.

    **Reuse across sessions has a limit**, though: calibration accuracy
    degrades over time even when the mount is not touched (initial point
    counts differed by a factor of 1.65, and the calibration closest in time
    was always best). So calibrate on the day of shooting and shoot right
    after; each shooting root also keeps a copy of the rig used at the time.

    ## Coordinate frame

    `cam_from_rig_*` in `rig_config.json` is "world -> camera", and world =
    rig frame, with the origin at the `ref_sensor` camera (cam_A in practice).
    The calibration already has a metric scale (`unit=m` in `scale.json`), so

        **1 SfM unit = 1 metre = 1000 mm**, no further estimate needed.

    Returns (pose dict, unit_mm).
    """
    doc = json.loads((rig / "rig_config.json").read_text(encoding="utf-8"))
    entry = doc[0] if isinstance(doc, list) else doc
    by_name = {}
    for c in entry["cameras"]:
        name = c["image_prefix"].rstrip("/")
        if c.get("ref_sensor"):
            q, t = [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        else:
            q = [float(x) for x in c["cam_from_rig_rotation"]]
            t = [float(x) for x in c["cam_from_rig_translation"]]
        by_name[name] = {"q": [f"{x:.12g}" for x in q],
                         "t": [f"{x:.12g}" for x in t]}
    miss = [c for c in cams if c not in by_name]
    if miss:
        raise SystemExit(
            f"  ERROR these cameras are not in the calibration's rig_config.json: {miss}\n"
            f"    it has: {sorted(by_name)}\n"
            f"    camera names are decided by sort order; after changing ports, reshoot the calibration set")

    return {c: by_name[c] for c in cams}, read_scale(rig, allow_unscaled)


def read_scale(rig: Path, allow_unscaled=False):
    """Decide whether this calibration has a metric scale. Returns unit_mm, **or None without a scale**.

    A default of `unit_mm = 1000.0` would disguise "unknown" as "known: 1
       unit = 1 metre". The measured conversion was 35.1 mm/unit, **off by
       28x**, and audio, wall HUD and VR downstream would all trust it -
       without an error, just silently wrong. So None is returned and they fail.

    Three states are kept apart because they are handled differently:

        scaled                  proceed
        deliberately unscaled   the user explicitly agreed during calibration -> blocked, but --allow-unscaled lets it through
        scale.json missing      **data damage** (incomplete copy); always abort, no override
    """
    sj = rig / "scale.json"
    rep = rig / "calib_report.json"
    if not sj.is_file():
        # if calib_report says there is a scale, scale.json was lost, not deliberately omitted
        lost = False
        if rep.is_file():
            try:
                lost = json.loads(rep.read_text(encoding="utf-8")).get("scale") is not None
            except Exception:                                  # noqa: BLE001
                pass
        if lost:
            raise SystemExit(
                "  ERROR the rig folder is incomplete: calib_report.json says it has a metric scale, "
                "but scale.json is missing.\n"
                "    A file was lost when copying the rig folder; copy a complete one again.")
        raise SystemExit(
            "  ERROR the rig has no scale.json.\n"
            "    This is an older format, or calibration did not finish. Run calibrate.py again,\n"
            "    or make sure this rig is complete.")

    d = json.loads(sj.read_text(encoding="utf-8"))
    # the older format has no metric field; unit=="m" means scaled
    metric = d.get("metric", d.get("unit") == "m")
    if metric:
        if d.get("unit") != "m":
            raise SystemExit(
                f"  ERROR the calibration's scale unit is {d.get('unit')!r}; only 'm' is supported.\n"
                f"    Run calibrate.py again with --edge so it applies a metric scale.")
        return 1000.0

    msg = ("  ERROR this calibration has **no metric scale** (no --edge was given when calibrating).\n"
           "    If you continue, audio direction, the wall HUD and VR spatial scale will all be wrong,\n"
           "    without any error. The scale must be added before preprocessing, or preprocessing and training must run again.\n"
           "    -> measure the six edges and run calibrate.py again with --edge\n"
           "    -> if you really know what you are doing, add --allow-unscaled")
    if not allow_unscaled:
        raise SystemExit(msg)
    print("  ! --allow-unscaled: continuing without a scale. Metric readings and audio localization are not reliable.")
    return None


def extract_and_match(exe, work: Path, imgs: Path, cams, fxfy, W, H):
    """Feature extraction + exhaustive matching. Poses do not come from here (see poses_from_rig),
    but triangulation still needs features and matches.

    Intrinsics are **fixed** to the undistorted PINHOLE (principal point
    centred), one camera_id per camera: the four cameras' fx/fy differ
    slightly (measured 973-981) and a shared camera would average them.
    """
    db = work / "database.db"
    for c in cams:
        fx, fy = fxfy[c]
        colmap_run(exe, "feature_extractor", "--database_path", db,
                   "--image_path", imgs, "--image_list_path", _list_file(work, c),
                   "--ImageReader.camera_model", "PINHOLE",
                   "--ImageReader.single_camera", "1",
                   "--ImageReader.camera_params",
                   f"{fx:.6f},{fy:.6f},{W / 2.0},{H / 2.0}")
    colmap_run(exe, "exhaustive_matcher", "--database_path", db)
    return db


def rectify_to_window_frame(pose, cams):
    """Change the world frame from "the reference camera" to **the window itself**.

    Why

    The world frame's origin and orientation in `rig_config.json` are those of
    the `ref_sensor` camera (cam_A in practice). That is an **arbitrary
    choice**: pick another reference and the whole world rotates. Measured,
    cam_A's axes differ from the rig plane by 1.06-1.59 deg, meaning

        **the window is mathematically tilted, by an amount that depends on which camera is the reference.**

    The window is **the viewer's origin**; on a flat screen and in VR it
    should always be level. So the world frame is defined by **the window
    itself**, not by one camera:

        origin  centroid of the camera centres (window centre)
        X axis  long side direction (right)
        Y axis  short side direction (down, following the COLMAP/3DGS Y-down convention)
        Z axis  rig plane normal (forward, toward the scene)

    The window is then **axis-aligned and centred on the origin by
    construction**, and swapping the reference camera or the rig does not
    change this definition.

    ## How signs are decided

    Geometry alone leaves a 180 deg ambiguity (which way is "up"?). It is
    resolved with **the reference camera's own axes**: the cameras are mounted
    on the rig, the optical axis is within 1.06 deg of the plane normal, so
    "the short side direction pointing the same way as the camera's down
    axis" is down. That is far more robust than relying on "which camera is
    top-left", which someone would have to remember, and would get wrong.

    ## This is a rigid transform; geometry is unchanged

    It only describes the same thing in another frame. Camera spacing, point
    cloud shape and metric scale are untouched.

    Returns (new pose dict, M, ctr). `M` is the rotation old world -> new
    world; points follow with `p_new = M @ (p_old - ctr)`.
    """
    C = {c: -q2R(pose[c]["q"]).T @ np.array([float(x) for x in pose[c]["t"]])
         for c in cams}
    P = np.stack([C[c] for c in cams])
    ctr = P.mean(0)

    # rig plane via SVD: the direction of the smallest singular value is the normal
    U, S, Vt = np.linalg.svd(P - ctr)
    normal = Vt[2]
    # two main directions in the plane: long side = largest variance, short side = second
    long_dir, short_dir = Vt[0], Vt[1]

    ref = cams[0]
    Rref = q2R(pose[ref]["q"])          # world -> reference camera
    cam_right, cam_down, cam_fwd = Rref[0], Rref[1], Rref[2]

    x = long_dir * np.sign(np.dot(long_dir, cam_right) or 1.0)      # right
    y = short_dir * np.sign(np.dot(short_dir, cam_down) or 1.0)     # down
    z = np.cross(x, y)
    if np.dot(z, cam_fwd) < 0:          # forward should point to the scene
        y, z = -y, -z
    # orthogonalize (already nearly orthogonal; this only removes numerical error)
    x = x / np.linalg.norm(x)
    y = y - x * np.dot(x, y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)

    M = np.stack([x, y, z])             # old world -> new world (rows are the new basis)

    # x_cam = R x_old + t, and x_old = M^T x_new + ctr
    #   => R_new = R M^T,  t_new = R ctr + t
    out = {}
    for c in cams:
        R = q2R(pose[c]["q"])
        t = np.array([float(v) for v in pose[c]["t"]])
        Rn = R @ M.T
        tn = R @ ctr + t
        out[c] = {"q": [f"{v:.12g}" for v in R2q(Rn)],
                  "t": [f"{v:.12g}" for v in tn]}

    dev = [np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1)))
           for a, b in ((x, cam_right), (y, cam_down), (z, cam_fwd))]
    print(f"  window frame corrected: origin moved to the window centre, axes aligned to the rig plane")
    print(f"    rotated relative to reference camera {ref}: right {dev[0]:.2f} deg, down {dev[1]:.2f} deg, forward {dev[2]:.2f} deg"
          f" (the window was tilted this much)")
    return out, M, ctr


def R2q(R):
    """Rotation matrix -> quaternion (w,x,y,z). Uses the branch with the largest trace term for stability."""
    m = R
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = np.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = np.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = np.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def db_image_ids(db_path: Path):
    """Read name -> (image_id, camera_id) from the COLMAP database.

    The ids written to images.txt **must match the database**, or
    point_triangulator will not line up. Do not assume "extraction order is 1..N".
    """
    import sqlite3
    con = sqlite3.connect(str(db_path))
    out = {r[0]: (r[1], r[2]) for r in
           con.execute("select name, image_id, camera_id from images")}
    con.close()
    return out


def triangulate_known_poses(exe, work: Path, imgs: Path, cams, pose,
                            fxfy, W, H):
    """Fixed poses; only triangulate points - `colmap point_triangulator`.

    The input model is written by hand: cameras.txt (PINHOLE, principal point
    centred), images.txt (calibrated poses, no feature points) and an empty
    points3D.txt.
    """
    db = work / "database.db"
    ids = db_image_ids(db)
    known = work / "known"
    known.mkdir(exist_ok=True)

    cl = ["# Camera list", "#", f"# Number of cameras: {len(cams)}"]
    il = ["# Image list", "#", f"# Number of images: {len(cams)}"]
    for c in cams:
        key = f"{c}.jpg"
        if key not in ids:
            raise SystemExit(f"  ERROR {key} is not in the database (feature extraction failed?)")
        iid, cid = ids[key]
        fx, fy = fxfy[c]
        cl.append(f"{cid} PINHOLE {W} {H} {fx:.8f} {fy:.8f} {W/2.0} {H/2.0}")
        p = pose[c]
        il += [f"{iid} {' '.join(p['q'])} {' '.join(p['t'])} {cid} {key}", ""]
    (known / "cameras.txt").write_text("\n".join(cl) + "\n", encoding="utf-8")
    (known / "images.txt").write_text("\n".join(il) + "\n", encoding="utf-8")
    (known / "points3D.txt").write_text("# 3D point list\n", encoding="utf-8")

    sp = work / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    colmap_run(exe, "point_triangulator", "--database_path", db,
               "--image_path", imgs, "--input_path", known,
               "--output_path", sp,
               "--Mapper.ba_refine_focal_length", "0",
               "--Mapper.ba_refine_principal_point", "0",
               "--Mapper.ba_refine_extra_params", "0")

    # **This number detects whether the rig has been moved.**
    #
    #    The poses are fixed (from calibration), so the reprojection error
    #    fully reflects whether "the geometry at calibration time" matches
    #    "the actual geometry of this recording". A knocked mount, a loose
    #    camera, a turned lens all make it jump.
    #
    #    It compensates for the **only thing lost** by using calibrated poses
    #    (re-solving would adapt to new geometry): no automatic adaptation,
    #    but an **explicit signal that something is wrong**. Better to stop
    #    and recalibrate than train on skewed geometry, which produces no
    #    error, only an indefinable discomfort in VR.
    txt = colmap_run(exe, "model_analyzer", "--path", sp)
    st = {}
    for l in txt.splitlines():
        for k, f in (("Points", "points"), ("Mean reprojection error", "reproj")):
            if k in l:
                try:
                    st[f] = float(l.split(":")[-1].strip().replace("px", ""))
                except ValueError:
                    pass
    return sp, st


def check_edges(pose, rig: Path, cams, unit_mm):
    """Compare measured edge lengths with the distances computed from the poses.

    Since poses come directly from calibration, this step verifies **that
       the pipeline is wired correctly** (camera names mapped to the wrong
       extrinsics, inverted unit conversion and the like), not the
       calibration itself - that was already done in calib_report.json.
    """
    C = {c: -q2R(pose[c]["q"]).T @ np.array([float(x) for x in pose[c]["t"]])
         for c in cams}
    if unit_mm is None:              # unscaled; this check is meaningless
        return []
    sj = rig / "scale.json"
    if not sj.is_file():
        return []
    rows = []
    for e in json.loads(sj.read_text(encoding="utf-8")).get("edges", []):
        a, b = e["pair"]
        if a in C and b in C:
            d_mm = float(np.linalg.norm(C[a] - C[b])) * unit_mm
            rows.append([a, b, d_mm, float(e["measured_mm"]),
                         d_mm - float(e["measured_mm"])])
    return rows


def _list_file(work: Path, cam):
    p = work / f"_list_{cam}.txt"
    p.write_text(f"{cam}.jpg\n", encoding="utf-8")
    return p


def build_mvs_images(work: Path, reader, cams, sel, best_i, maps, out_dir: Path):
    """Images for MVS: **the temporal median of several consecutive time steps**, not a single frame.

    Why: noise means something completely different to the two consumers.

    | | images seen | is noise averaged out? |
    |---|---|---|
    | 4DGS training | 300 time steps x 4 cameras | yes, falls with sqrt(N) |
    | **MVS initial point cloud** | **1 time step x 4 cameras** | **not at all** |

    The recorder's exposure strategy is "gain at maximum in exchange for the
    shortest shutter", because for training motion blur is expensive and
    noise comparatively cheap. **That trade-off does not hold for MVS**,
    which works on a single time step and gets no averaging.

    Measured:

    | MVS input | depth map roughness | points after fusion |
    |---|---|---|
    | single frame | 27.66% | 4,884 |
    | **9-frame median** | **13.75%** | **7,922** |

    **Median**, not mean: moving things (a fan, a person) are outliers that
    the median rejects. The initial point cloud only needs static geometry;
    the dynamic part is left to training.

    Feature extraction and triangulation still use **a single frame**
       (`work/images`): SIFT is robust to noise (measured reprojection
       0.956 px), and that step needs "the real geometry at this time step",
       which a temporal median must not alter.
    """
    import cv2 as _cv
    idx = [k for k in range(best_i - MVS_MEDIAN_HALF, best_i + MVS_MEDIAN_HALF + 1)
           if 0 <= k < len(sel)]
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(idx) < 3:
        print(f"  ! only {len(idx)} time steps available; MVS falls back to a single frame")
        for c in cams:
            shutil_copy = (work / "images" / f"{c}.jpg").read_bytes()
            (out_dir / f"{c}.jpg").write_bytes(shutil_copy)
        return 1
    for c in cams:
        m1, m2 = maps[c]
        st = np.stack([_cv.remap(reader.get(c, sel[k]["pick"][c]), m1, m2,
                                 _cv.INTER_LINEAR).astype(np.float32) for k in idx])
        _cv.imwrite(str(out_dir / f"{c}.jpg"),
                    np.median(st, 0).astype(np.uint8),
                    [_cv.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return len(idx)


def run_mvs(exe, work: Path, imgs: Path, model: Path,
            geom_consistency=True, min_pixels=MVS_MIN_PIXELS):
    """MVS dense reconstruction -> initial point cloud.

    Why not sparse points: sparse gave only **672 points** (repetitive
       white tiles made SIFT's ratio test reject most matches, a 6% match
       rate), too thin to initialize 4DGS - **an overly sparse initial cloud
       makes experiments fail for the wrong reason**. MVS with
       `min_num_pixels=3` (3 of 4 cameras agree) gave 28,202 points with
       0.00% outliers beyond 10 m.
    """
    dense = work / "dense"
    colmap_run(exe, "image_undistorter", "--image_path", imgs,
               "--input_path", model, "--output_path", dense,
               "--output_type", "COLMAP")
    colmap_run(exe, "patch_match_stereo", "--workspace_path", dense,
               "--workspace_format", "COLMAP",
               "--PatchMatchStereo.geom_consistency",
               "true" if geom_consistency else "false")
    ply = dense / "fused.ply"
    colmap_run(exe, "stereo_fusion", "--workspace_path", dense,
               "--workspace_format", "COLMAP",
               "--input_type", "geometric" if geom_consistency else "photometric",
               "--output_path", ply,
               "--StereoFusion.min_num_pixels", str(min_pixels))
    return ply


def read_points3d_bin(p: Path):
    """COLMAP points3D.bin -> (N,3) xyz, (N,3) rgb. Used by the sparse fallback."""
    import struct
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        xyz = np.empty((n, 3), np.float64)
        rgb = np.empty((n, 3), np.uint8)
        for i in range(n):
            f.read(8)                                   # point id
            xyz[i] = struct.unpack("<3d", f.read(24))
            rgb[i] = struct.unpack("<3B", f.read(3))
            f.read(8)                                   # reprojection error
            ntrack = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * ntrack)                          # track, unused
    return xyz, rgb


def read_ply(p: Path):
    from plyfile import PlyData
    v = PlyData.read(str(p))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    try:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    except ValueError:
        rgb = np.full((len(xyz), 3), 128, np.uint8)
    return xyz, rgb


def read_images_txt(p: Path):
    """Returns {camera name: dict(q, t, cam_id)}."""
    lines = [l for l in p.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    out = {}
    for i in range(0, len(lines), 2):
        f = lines[i].split()
        name = f[9]
        out[name.rsplit(".", 1)[0]] = {"q": f[1:5], "t": f[5:8], "cam": f[8]}
    return out


def q2R(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def write_colmap_n(out: Path, sel, cams, reader, maps, pose, W, H, fxfy, xyz, rgb):
    """Write one `colmap_N` per time step: four undistorted images + one sparse/0.

    **Poses are shared by the four cameras and identical for every time
       step** (the window principle): window = rig = origin; the cameras do
       not move, the scene does. Estimating poses per time step would
       misread "the scene moves" as "the cameras move", and four views cannot
       support a stable per-frame PnP.

    **The initial point cloud is split into N disjoint parts**: each time step
       gets a different subset. If all time steps shared one cloud, STG would
       start from N-fold duplicated points and densification would behave
       differently than intended.
    """
    import cv2
    rng = np.random.default_rng(0)
    parts = np.array_split(rng.permutation(len(xyz)), len(sel))
    cam_lines = [f"{i} PINHOLE {W} {H} {fxfy[c][0]:.8f} {fxfy[c][1]:.8f} "
                 f"{W / 2.0} {H / 2.0}" for i, c in enumerate(cams, 1)]

    for n, g in enumerate(sel):
        d = out / f"colmap_{n}"
        (d / "images").mkdir(parents=True, exist_ok=True)
        s0 = d / "sparse" / "0"
        s0.mkdir(parents=True, exist_ok=True)

        for c in cams:
            im = reader.get(c, g["pick"][c])
            m1, m2 = maps[c]
            cv2.imwrite(str(d / "images" / f"{c}.jpg"),
                        cv2.remap(im, m1, m2, cv2.INTER_LINEAR),
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

        k = parts[n]
        pl = ["# 3D point list", "#", f"# Number of points: {len(k)}"]
        pl += [f"{i+1} {xyz[j][0]:.6f} {xyz[j][1]:.6f} {xyz[j][2]:.6f} "
               f"{rgb[j][0]} {rgb[j][1]} {rgb[j][2]} 0" for i, j in enumerate(k)]
        (s0 / "points3D.txt").write_text("\n".join(pl) + "\n", encoding="utf-8")
        (s0 / "cameras.txt").write_text(
            "\n".join(["# Camera list", "#", f"# Number of cameras: {len(cams)}"]
                      + cam_lines) + "\n", encoding="utf-8")
        il = ["# Image list", "#", f"# Number of images: {len(cams)}"]
        for i, c in enumerate(cams, 1):
            p = pose[c]
            il += [f"{i} {' '.join(p['q'])} {' '.join(p['t'])} {i} {c}.jpg", ""]
        (s0 / "images.txt").write_text("\n".join(il) + "\n", encoding="utf-8")
        yield n, d, s0


def stage2(out: Path, rig: Path, sel, cams, rec: Path, win: dict,
           colmap=None, keep_work=False, allow_unscaled=False,
           mvs_geom=True, mvs_min_pixels=MVS_MIN_PIXELS):
    import math
    import shutil

    import cv2
    exe = find_colmap(colmap)
    root = out.parent                       # shooting root
    print(f"\n  -- stage 2 --  COLMAP: {exe}")

    # 0. copy the calibration into the shooting root -------------------
    # The rig gets rebuilt and recalibrated, and the outside copy gets
    #    overwritten by the new one. An old shooting root still pointing
    #    there would get **the wrong calibration without an error** -
    #    geometry silently skewed, felt in VR as an indefinable discomfort.
    #    A recording uses the calibration of its time. It is only 22 KB.
    rig_local = root / "rig"
    if rig.resolve() != rig_local.resolve():
        if rig_local.exists():
            shutil.rmtree(rig_local)
        shutil.copytree(rig, rig_local)
        print(f"  calibration copied -> {rig_local} (only this copy is read afterwards)")
    rig = rig_local

    # 1. undistortion maps ---------------------------------------------
    maps, W, H, fxfy = undistort_maps(rig, cams)

    # 2. poses from calibration; one group of four images triangulates the initial points
    work = out / "_work"
    if work.exists():
        shutil.rmtree(work)               # leftovers would contaminate the priors
    (work / "images").mkdir(parents=True)
    best = min(sel, key=lambda g: g["span"])
    print(f"  triangulating with time step {sel.index(best)}"
          f" (group span {best['span']:.2f} ms, the smallest of all)")
    pose, unit_mm = poses_from_rig(rig, cams, allow_unscaled)
    # The window is the viewer's origin and should always be level on a
    #    flat screen and in VR. The calibration's world frame is the
    #    reference camera's, an arbitrary choice - measured, the frame was
    #    tilted 1.6 deg, 14 px over a 498 px span, visible by eye.
    pose, _M, _ctr = rectify_to_window_frame(pose, cams)
    reader = FrameReader(rec, cams)
    for c in cams:
        m1, m2 = maps[c]
        cv2.imwrite(str(work / "images" / f"{c}.jpg"),
                    cv2.remap(reader.get(c, best["pick"][c]), m1, m2,
                              cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    extract_and_match(exe, work, work / "images", cams, fxfy, W, H)
    model, tri = triangulate_known_poses(exe, work, work / "images", cams, pose,
                                         fxfy, W, H)
    print(f"  OK poses from calibration ({len(pose)}/{len(cams)} cameras); only points triangulated")
    rp = tri.get("reproj")
    print(f"     triangulated {tri.get('points', 0):.0f} points, reprojection {rp:.3f} px"
          if rp is not None else "     (triangulation statistics not readable)")
    # This number must be kept in the file. It is the only evidence of
    #    **whether the calibration still holds**; printed only on screen, it
    #    is gone once the terminal closes.
    win["tri_points"] = int(tri.get("points", 0))
    win["tri_reproj_px"] = round(rp, 4) if rp is not None else None
    win["tri_reproj_threshold_px"] = TH_TRI_REPROJ
    if rp is not None and rp > TH_TRI_REPROJ:
        print(f"  FAIL reprojection {rp:.2f} px exceeds {TH_TRI_REPROJ} px -"
              f" **the calibration does not match this recording's geometry**.")
        print("     The most likely cause is that the rig was moved (knocked, a loose camera, a turned lens).")
        print("     Reshoot the calibration set; do not train with this one.")

    # 3. MVS -----------------------------------------------------------
    # MVS uses **a single frame**. The temporal-median version is measured
    #    to help (4,884 -> 7,922 points, depth roughness 27.66% -> 13.75%);
    #    the code stays in build_mvs_images() but is **disabled by default**.
    win["mvs_input_frames"] = 1
    ply = run_mvs(exe, work, work / "images", model,
                  geom_consistency=mvs_geom, min_pixels=mvs_min_pixels)
    xyz, rgb = read_ply(ply)
    src = f"MVS, min_num_pixels={mvs_min_pixels}, consistency check {'on' if mvs_geom else 'off'}"
    print(f"  initial point cloud {len(xyz):,} points ({src})")

    # An empty MVS result is expected, not exceptional: the consistency check
    #    requires the four cameras' depths to agree, and depth error grows
    #    with the **square** of distance. Measured at f=974 px and a 153 mm
    #    baseline, a 1 px disparity error is 6.7 mm at 1 m, 168 mm at 5 m and
    #    671 mm at 10 m - so surfaces beyond about 2.6 m no longer agree and
    #    are all filtered out. In a scene beyond 5 m, photometric had 99.2%
    #    valid pixels and geometric only 0.01%, all survivors within 2.6 m.
    #    np.percentile on an empty array used to fail with an error that did
    #    not reveal the cause.
    if len(xyz) == 0:
        sparse_bin = model / "points3D.bin"
        print("  FAIL MVS produced no points - usually not a bug: **the scene is beyond what this baseline can measure**.")
        print("     The rig's baseline is only a dozen centimetres; depth error grows with the square of distance,")
        print("     and the consistency check filters out every depth that does not agree.")
        if sparse_bin.exists():
            xyz, rgb = read_points3d_bin(sparse_bin)
            src = "sparse triangulation (fallback after empty MVS)"
            print(f"     Falling back to sparse triangulation: {len(xyz):,} points. **This can still be trained** -")
            print("     a model initialized from 1,200 points did not look worse by eye than a million-point one.")
            print("     For MVS: --mvs-geom-consistency off (mixes in unreliable depth),")
            print("     or enlarge the rig baseline; it should roughly scale with shooting distance.")
        else:
            raise SystemExit("  ERROR neither MVS nor sparse triangulation produced points; cannot continue.")
    win["npoints_source"] = src
    win["mvs_geom_consistency"] = bool(mvs_geom)
    win["mvs_min_num_pixels"] = int(mvs_min_pixels)

    # 4. geometry self-check -------------------------------------------
    # Since poses come directly from calibration, this checks **whether the
    # pipeline is wired correctly** (names mapped to the wrong extrinsics,
    # inverted unit conversion), not the calibration itself.
    rows = check_edges(pose, rig, cams, unit_mm)
    print(f"  1 SfM unit = {unit_mm:.1f} mm (calibration has a metric scale, unit=m)"
          if unit_mm else "  ! no metric scale; unit_mm unknown")
    worst = 0.0
    for a, b, d, mm, res in rows:
        worst = max(worst, abs(res))
        print(f"    {a}-{b}: calibrated {d:7.1f} mm  measured {mm:7.1f} mm  "
              f"diff {res:+5.2f} mm  {'OK' if abs(res) <= 2.0 else 'BAD'}")
    if rows and worst > 2.0:
        print(f"  ! largest difference {worst:.2f} mm - this is a residual of **the calibration itself**, "
              f"not of this recording. Remeasure the edges or reshoot the calibration set to improve it.")

    # 5. write each time step ------------------------------------------
    print(f"  writing {len(sel)} time steps...")
    for n, d, s0 in write_colmap_n(out, sel, cams, reader, maps, pose,
                                   W, H, fxfy, xyz, rgb):
        colmap_run(exe, "model_converter", "--input_path", s0,
                   "--output_path", s0, "--output_type", "BIN")
        for f in ("cameras.txt", "images.txt", "points3D.txt"):
            (s0 / f).unlink(missing_ok=True)
        if n % 25 == 0 or n == len(sel) - 1:
            print(f"    colmap_{n}")
    reader.close()

    # 6. metadata ------------------------------------------------------
    C = {c: -q2R(pose[c]["q"]).T @ np.array([float(x) for x in pose[c]["t"]])
         for c in cams}
    ctr = np.mean([C[c] for c in cams], axis=0)
    fwd = q2R(pose[cams[0]]["q"])[2]         # optical axis of the reference camera (toward the scene)
    order = window_corners(cams, C, ctr)

    fx_m = float(np.mean([fxfy[c][0] for c in cams]))
    fy_m = float(np.mean([fxfy[c][1] for c in cams]))
    near = float(np.percentile(xyz[:, 2], 1)) * 0.9
    far = float(np.percentile(xyz[:, 2], 99)) * 1.1
    pb = np.zeros((len(cams), 17))
    pb[:, 4], pb[:, 9], pb[:, 14] = H, W, fx_m
    pb[:, -2], pb[:, -1] = near, far
    np.save(out / "poses_bounds.npy", pb)

    cfg = dict(resolution=2, model="ours_lite", scaling_lr=0.003,
               preprocesspoints=0, test_iteration=12000, densify=2,
               duration=len(sel), rdpip="train_ours_lite")
    (out / "cfg_lite.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # audio: if present, copy it unchanged into the dataset and write the audio block; otherwise none (the player plays silently)
    audio_blk = copy_audio(rec, out, sel, cams)

    win.pop("_geometry", None)
    win.update({
        "wh": [W, H],
        "fov": [round(float(2 * math.atan(W / 2 / fx_m)), 6),
                round(float(2 * math.atan(H / 2 / fy_m)), 6)],
        "ctr": [round(float(x), 6) for x in ctr],
        "fwd": [round(float(x), 6) for x in fwd],
        # `win` uses 9 decimals while other geometry fields use 6 - deliberately.
        #    The player **reads this value directly** to draw the window
        #    frame, which is positioned at sub-pixel precision: 6 decimals
        #    (1e-6 scene units = 0.001 mm) change the frame edge's
        #    anti-aliased pixels by up to 5/255. Measured effect of rounding:
        #        6 decimals  43-57 pixels differ, at most 5/255
        #        7 decimals   0-1
        #        8 decimals   0-26          <- inconsistent
        #        9 decimals   0             <- identical to no rounding
        #    The non-monotonic 7/8 results are not a measurement error but a
        #       **threshold effect**: which side of a pixel boundary rounding
        #       lands on is luck, not a systematic bias. So 9 decimals with
        #       margin are used, not 7, which only just passes.
        "win": [[round(float(x), 9) for x in C[c]] for c in order],
        "win_order": order,
        "near_far": [round(near, 4), round(far, 4)],
        "npoints": int(len(xyz)),
        "rig_source": "rig",              # relative to the shooting root
    })
    if audio_blk:
        win["audio"] = audio_blk
    if unit_mm:
        win["unit_mm"] = round(unit_mm, 4)
        win["unit_mm_source"] = "rig (least squares over six physical edges)"
        win["unit_mm_max_residual_mm"] = round(max(abs(r[4]) for r in rows), 2)
    else:
        # flag for the player: without unit_mm, no audio localization and no wall HUD
        win["metric"] = False
    (out / "window.json").write_text(
        json.dumps(win, ensure_ascii=False, indent=2), encoding="utf-8")

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n  stage 2 done -> {out}")
    print(f"     {len(sel)} time steps x {len(cams)} cameras, {W}x{H}, "
          f"initial cloud {len(xyz):,} points")
    print("  training and evaluation must use resolution=2 (in cfg_lite.json); "
          "use --res 2 when comparing PSNR")
    return win


# ========================================================== main
def find_recording(root: Path) -> Path:
    """The only folder under the shooting root that contains `session.json` is the recording."""
    hits = [d for d in sorted(root.iterdir())
            if d.is_dir() and (d / "session.json").is_file()]
    if not hits:
        raise SystemExit(
            f"  ERROR no recording found under {root}\n"
            f"    a recording is a folder containing session.json (for example rec_20260806_103000)\n"
            f"    copy it from the Pi into this folder, or use the older form with two paths")
    if len(hits) > 1:
        raise SystemExit(
            f"  ERROR {root} contains {len(hits)} recordings: "
            + ", ".join(d.name for d in hits)
            + "\n    a shooting root should hold only one recording. Use the older form to choose one")
    return hits[0]


def resolve_rig(root: Path, rig_arg) -> Path | None:
    """Calibration: `--rig` first, otherwise the copy in the root."""
    if rig_arg:
        return Path(rig_arg).resolve()
    local = root / "rig"
    if (local / "rig_config.json").is_file():
        return local.resolve()
    return None


def main():
    ap = argparse.ArgumentParser(description="recording -> 4DGS training set")
    ap.add_argument("target",
                    help="shooting root; or (older form) the recording folder")
    ap.add_argument("out", nargs="?", default=None,
                    help="(older form) output folder. Not needed otherwise; named stg<time steps> automatically")
    ap.add_argument("--rig", default=None,
                    help="calibration folder. Defaults to <shooting root>/rig. "
                         "If given, it is copied into the shooting root and that copy is used afterwards")
    ap.add_argument("--span-ms", type=float, default=25.0,
                    help="grouping threshold on span = max(t)-min(t) (default 25)")
    ap.add_argument("--steps", type=int, default=0, help="number of time steps to sample, 0 = all")
    ap.add_argument("--anchor", default=None, help="reference camera for pairing (default the first)")
    ap.add_argument("--colmap", default=None, help="COLMAP executable (found automatically by default)")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep stage 2 intermediates (_work, including the MVS workspace; large)")
    ap.add_argument("--allow-unscaled", action="store_true",
                    help="accept a calibration without metric scale. **Metric readings and audio localization will be wrong**")
    ap.add_argument("--stage1", action="store_true",
                    help="run stage 1 only (seconds). Use it right after recording to check the valid rate, "
                         "and reshoot if it is poor instead of waiting for the much longer stage 2")
    ap.add_argument("--mvs-geom-consistency", choices=("on", "off"), default="on",
                    help="MVS geometric consistency check (default on). "
                         "It requires the cameras' depths to agree, and depth error grows with the square of distance - "
                         "measured at f=974 px and a 153 mm baseline, surfaces beyond about 2.6 m are filtered out. "
                         "off keeps distant points, but those depths are **unreliable to begin with** "
                         "and may be worse than none. Inspect the result visually if changed.")
    ap.add_argument("--mvs-min-pixels", type=int, default=MVS_MIN_PIXELS,
                    help=f"how many cameras stereo_fusion requires to agree (default {MVS_MIN_PIXELS}; "
                         "2/3/4/5 were swept, 3 is best). 2 is much looser on a four-camera rig")
    ap.add_argument("--background", type=float, default=0.0,
                    help="background colour 0-1 (default **0 = black**). "
                         "Training must use the same value via STG_BACKGROUND")
    a = ap.parse_args()

    target = Path(a.target).resolve()
    if a.out:                                  # older form: two positional arguments
        rec, out, root = target, Path(a.out).resolve(), Path(a.out).resolve().parent
        rig = Path(a.rig).resolve() if a.rig else None
    else:                                      # current form: only the shooting root
        if not target.is_dir():
            raise SystemExit(f"  ERROR not found: {target}")
        root = target
        rec = find_recording(root)
        out = None                             # the name is known only after grouping
        rig = resolve_rig(root, a.rig)
        print(f"  shooting root {root.name}")
        print(f"    recording   {rec.name}")
        print(f"    calibration {rig if rig else '(none - only stage 1 will run)'}")

    # The scale is checked **first**. It is a property of the data, not a
    #    stage 2 detail; blocking only in stage 2 would waste a grouping and
    #    MVS run before discovering the data cannot be used.
    if rig:
        read_scale(rig, a.allow_unscaled)

    cams = load_streams(rec)
    if len(cams) < 2:
        raise SystemExit(f"  ERROR only {len(cams)} camera(s) found")
    anchor = a.anchor or cams[0]
    if anchor not in cams:
        raise SystemExit(f"  ERROR anchor {anchor} is not in {cams}")

    idx = {c: load_index(rec, c) for c in cams}
    print(f"  {len(cams)} cameras: {', '.join(cams)}   anchor {anchor}")
    for c in cams:
        print(f"    {c}  {len(idx[c]):4d} frames")

    period = frame_period_ms(idx, anchor)
    ok, allg = build_groups(idx, anchor, a.span_ms)
    if not ok:
        raise SystemExit(f"  ERROR no group passes span <= {a.span_ms} ms")
    sel = select(ok, a.steps)
    print(f"\n  Valid groups {len(ok)}/{len(allg)}"
          f" ({len(ok)/len(allg)*100:.1f}%)  sampled {len(sel)} time steps")

    if out is None:                            # the name is known only after grouping
        out = root / f"stg{len(sel)}"
        print(f"  Dataset -> {out.name} (named after the number of valid time steps)")

    out.mkdir(parents=True, exist_ok=True)
    rep = phase_report(cams, ok, allg, a.span_ms, period, idx)
    (out / "phase_report.md").write_text(rep, encoding="utf-8")
    win = write_stage1(out, cams, sel, rec, a.span_ms, a.background)

    print(f"\n  -> {out}")
    for k in ("source_recording", "captured_at", "exposure_us", "gain",
              "nsteps", "duration_s",
              "step_ms_median", "step_ms_max", "group_span_ms_max", "background"):
        if k in win:                      # a field may be absent (for example no captured_at)
            print(f"      {k:22} {win[k]}")
    print(f"\n  phase report -> {out/'phase_report.md'}")
    print("\n" + "\n".join(rep.split("\n")[4:14]))

    print(f"\n  training must use the same background colour:"
          f"  STG_BACKGROUND={a.background}")

    if a.stage1:
        print("\n  (--stage1: stage 1 only. If the valid rate is acceptable, "
              "run again without this option for the full result.)")
    elif rig:
        stage2(out, rig, sel, cams, rec, win,
               colmap=a.colmap, keep_work=a.keep_work,
               allow_unscaled=a.allow_unscaled,
               mvs_geom=(a.mvs_geom_consistency == "on"),
               mvs_min_pixels=a.mvs_min_pixels)
    else:
        print("\n  ! no calibration; stage 2 (undistortion / colmap_N / initial point cloud) did not run."
              "\n    Copy the rig folder from calibration into the shooting root, or give it with --rig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
