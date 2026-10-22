"""Merge the segment models of a combined training run into one PLY, with a segment table in window.json.

    python tools/merge_segments.py --dataset <dataset folder> --out <output folder> \
        a_s01_out/point_cloud/iteration_10000/point_cloud.ply:10:40 \
        a_s02_out/point_cloud/iteration_10000/point_cloud.ply:40:70 \
        a_s03_out/point_cloud/iteration_10000/point_cloud.ply:70:90

Why the time parameters are remapped: each segment was trained with its own
time normalized to 0..1. After merging there is one global 0..1 time axis, so
each segment's time parameters are shifted and scaled into its own interval.
Only time-related fields are changed; everything else is copied unchanged.

Why a segment table: STG represents static content with a very large
trbf_scale, and about a quarter of the Gaussians remain visible one segment
length away from their time center. A Gaussian is bell-shaped and cannot form
a square window, so hiding a segment outside its interval cannot rely on the
temporal RBF; the player uses explicit row ranges (`segments`) instead.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

# Time-related fields remapped on merge. Everything else (xyz/scale/rot/opacity/f_dc/normals) is unchanged.
TIME_FIELDS = ("trbf_center", "trbf_scale",
               *[f"motion_{i}" for i in range(9)],
               *[f"omega_{i}" for i in range(4)])


def split_segments(a, b, per):
    """Split [a, b] (both inclusive) into segments of `per` steps; neighbours share the boundary step.

    Both segments see the shared step, so the player can cross-fade there
    instead of switching abruptly.

        split_segments(10, 90, 30) -> [(10, 40), (40, 70), (70, 90)]

    The last segment holds the remainder (70..90 above, 21 steps) and is
    trained as well.
    """
    if per < 1:
        raise ValueError("each segment needs at least 1 time step")
    if b <= a:
        return [(a, b)]
    out, start = [], a
    while start < b:
        end = min(start + per, b)
        out.append((start, end))
        start = end
    # A very short tail is merged into the previous segment: a segment of two
    # or three steps cannot be trained (not even the held-out steps fit) and
    # its image quality would drop visibly. Threshold: a third of a segment.
    if len(out) > 1:
        sa, sb = out[-1]
        if (sb - sa + 1) < max(2, per // 3):
            out[-2] = (out[-2][0], sb)
            out.pop()
    return out


def _times(dataset):
    with open(os.path.join(dataset, "times_group.json"), encoding="utf-8") as f:
        return json.load(f)


def _lo_hi(times, a, b):
    """Smallest and largest timestamp of time steps a..b (same rule as runner.write_model_window)."""
    ts = [float(v) for k, v in times.items() if a <= int(k.split("/")[0]) <= b]
    if not ts:
        raise ValueError(f"times_group.json has no data for time steps {a}..{b}")
    return min(ts), max(ts)


def remap(part, t0, t1):
    """Remap one segment's time parameters (segment-local 0..1) to the global [t0, t1].

    A local dt times L is the global dt, so polynomial coefficients are
    divided by the matching power of L:

        trbf_center' = t0 + c * L
        trbf_scale'  = s + ln(L)          # stored as ln(sigma)
        motion 1st/2nd/3rd order -> / L, / L^2, / L^3
        omega                    -> / L
    """
    L = float(t1) - float(t0)
    if L <= 0:
        raise ValueError(f"segment length cannot be 0 (t0={t0}, t1={t1})")
    out = part.copy()
    names = out.dtype.names
    if "trbf_center" in names:
        out["trbf_center"] = t0 + part["trbf_center"] * L
    if "trbf_scale" in names:
        out["trbf_scale"] = part["trbf_scale"] + math.log(L)
    for i in range(9):
        k = f"motion_{i}"
        if k in names:
            out[k] = part[k] / (L ** (i // 3 + 1))
    for i in range(4):
        k = f"omega_{i}"
        if k in names:
            out[k] = part[k] / L
    return out


def drop_dead(part):
    """Drop Gaussians whose position is NaN/Inf. Returns (clean data, number dropped).

    Trained models can contain a few percent of Gaussians with NaN positions.
    Depth comparisons with NaN are always False, so they are never drawn;
    keeping them only wastes memory, multiplied by the number of segments.
    """
    ok = np.ones(part.shape[0], dtype=bool)
    for k in ("x", "y", "z"):
        if k in part.dtype.names:
            ok &= np.isfinite(np.asarray(part[k], dtype=np.float64))
    return part[ok], int((~ok).sum())


def _model_name(ply):
    """Model name from <model>/point_cloud/iteration_N/point_cloud.ply."""
    return os.path.basename(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.normpath(ply))))) or ply


def uniform_t(a, b, sa, sb):
    """Global interval of a segment under the upstream time convention.

    Without `STG_TIMES`, upstream uses `(j - start) / length` as segment-local
    time; it divides by the length, not the length minus one, so the last
    step is (n-1)/n and never reaches 1.0. Placing such a model back on the
    global axis requires deriving t1 from that convention.

    Only used for comparisons with the upstream behaviour (`--uniform`).
    """
    n = sb - sa + 1
    span = b - a
    t0 = (sa - a) / span
    if n < 2:
        return t0, t0 + 1e-9
    t_last = (sb - a) / span
    return t0, t0 + (t_last - t0) * n / (n - 1)


def merge(dataset, segs, out_dir, ramp_steps=1, uniform=False):
    """Merge. `segs` = [(ply path, first step, last step)] in playback order.

    Writes `<out_dir>/point_cloud/iteration_1/point_cloud.ply` and
    `<out_dir>/window.json` (with `segments` and the time / audio fields of
    the whole range). Returns the written window.json content.
    """
    from plyfile import PlyData, PlyElement

    a, b = segs[0][1], segs[-1][2]
    times = _times(dataset)
    lo_all, hi_all = _lo_hi(times, a, b)
    span = hi_all - lo_all
    if span <= 0:
        raise ValueError(f"time steps {a}..{b} span no time")

    parts, table = [], []
    start_row = 0
    for ply, sa, sb in segs:
        data = PlyData.read(ply)["vertex"].data
        if parts and data.dtype != parts[0].dtype:
            raise ValueError(f"{ply} has different fields from the first segment; cannot merge")
        data, dead = drop_dead(data)
        if dead:
            print(f"  {_model_name(ply)}: "
                  f"dropped {dead:,} Gaussians with NaN positions (never drawn anyway)")
        if uniform:
            t0, t1 = uniform_t(a, b, sa, sb)      # upstream uniform time convention
        else:
            lo, hi = _lo_hi(times, sa, sb)
            t0, t1 = (lo - lo_all) / span, (hi - lo_all) / span
        parts.append(remap(data, t0, t1))
        table.append({"t0": round(t0, 9), "t1": round(t1, 9),
                      "start": start_row, "count": int(data.shape[0]),
                      "range": [sa, sb]})
        start_row += int(data.shape[0])

    merged = np.concatenate(parts)
    pc = os.path.join(out_dir, "point_cloud", "iteration_1")
    os.makedirs(pc, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(
        os.path.join(pc, "point_cloud.ply"))

    with open(os.path.join(dataset, "window.json"), encoding="utf-8") as f:
        win = json.load(f)
    dur0 = float(win.get("duration_s") or 0.0)
    win["nsteps"] = b - a + 1
    win["duration_s"] = round(span * dur0, 6)
    if isinstance(win.get("audio"), dict):
        win["audio"]["t0_offset_s"] = round(
            float(win["audio"].get("t0_offset_s") or 0.0) + lo_all * dur0, 6)
    win["time_range"] = [a, b]
    win["segments"] = table
    # Cross-fade width in global time; the player uses it for the ramp.
    win["segment_ramp"] = round(ramp_steps / max(b - a, 1), 9)
    with open(os.path.join(out_dir, "window.json"), "w", encoding="utf-8") as f:
        json.dump(win, f, ensure_ascii=False, indent=2)
    return win


def _seg_arg(s):
    ply, sa, sb = s.rsplit(":", 2)
    return ply, int(sa), int(sb)


def main():
    p = argparse.ArgumentParser(description="Merge the segment models of a combined training run")
    p.add_argument("segments", nargs="+", type=_seg_arg,
                   help="<ply path>:<first step>:<last step>, in playback order")
    p.add_argument("--dataset", required=True, help="dataset folder (with times_group.json)")
    p.add_argument("--out", required=True, help="output model folder")
    p.add_argument("--ramp", type=int, default=1, help="cross-fade length in time steps (default 1; 0 = hard cut)")
    p.add_argument("--uniform", action="store_true",
                   help="segments were trained with upstream uniform time (no STG_TIMES)")
    a = p.parse_args()
    win = merge(a.dataset, a.segments, a.out, ramp_steps=a.ramp,
                uniform=a.uniform)
    n = sum(s["count"] for s in win["segments"])
    print(f"  merged {len(win['segments'])} segments, {n:,} Gaussians -> {a.out}")
    for s in win["segments"]:
        print(f"    steps {s['range'][0]}..{s['range'][1]}  "
              f"global {s['t0']:.4f}..{s['t1']:.4f}  {s['count']:,} Gaussians")
    print(f"  length {win['duration_s']} s, {win['nsteps']} time steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
