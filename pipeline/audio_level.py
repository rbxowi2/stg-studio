#!/usr/bin/env python3
"""audio_level.py - loudness measurement and gain (EBU R128 / ITU-R BS.1770).

Used by `prep_4d.py` (written into the dataset); one implementation, so the
formula cannot drift between copies.

## Why gain is needed

I2S MEMS microphones (ICS-43434) deliver linear PCM with full scale reserved
for 120+ dB SPL, so normal speech (60-65 dB SPL) sits around -50 dBFS and is
inaudible when played directly. That is the specification, not a fault. A
20-second speech recording measured -59.7 LUFS / -26.2 dBTP.

The recorder applies no gain (raw stays raw); gain is a presentation
decision and is computed here.

## Why LUFS and not peak or RMS

    absolute peak   one loud instant uses up all the available gain
    overall RMS     silent passages pull the average down

EBU R128 solves both:

    True Peak (dBTP)     4x oversampling catches peaks between samples
    Integrated LUFS      K-weighting (follows the ear's frequency sensitivity)
                         + gating (absolute gate -70 LUFS, relative gate -10 LU)
                           -> quiet passages are not counted at all

## Why a linear gain without compression

If the target cannot be reached, a smaller gain is accepted; no limiter is
used. Compression would raise the noise floor and flatten the dynamics, while
the goal is realism. Hitting a broadcast loudness exactly is not required.
"""
from __future__ import annotations

import re
import subprocess

# Target -23 LUFS (EBU R128 broadcast) rather than -14/-16 (streaming): the
# rig records source material, and field recordings conventionally keep their
# dynamics and leave normalization downstream.
TARGET_LUFS = -23.0
TRUE_PEAK_CEILING_DBTP = -1.0


def ffmpeg_exe():
    """Find ffmpeg: the one bundled with imageio_ffmpeg if available, otherwise PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                  # noqa: BLE001
        return "ffmpeg"


def measure(wav, ff=None):
    """Measure loudness. Returns (integrated_lufs, true_peak_dbtp, lra) or (None, None, None).

    Returns None when it cannot measure, so callers keep working on machines
    without ffmpeg instead of failing the whole preprocessing.
    """
    ff = ff or ffmpeg_exe()
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-nostats", "-i", str(wav),
             "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
            capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    txt = r.stderr
    out = []
    for pat in (r"I:\s*(-?\d+\.?\d*)\s*LUFS",
                r"Peak:\s*(-?\d+\.?\d*)\s*dBFS",
                r"LRA:\s*(-?\d+\.?\d*)\s*LU"):
        m = re.findall(pat, txt)
        out.append(float(m[-1]) if m else None)
    return tuple(out)


def linear_gain(lufs, true_peak):
    """One gain (dB) satisfying both the loudness target and the true-peak ceiling: the smaller one.

    Returns (gain, explanation). (0.0, reason) when nothing could be measured.
    """
    if lufs is None or true_peak is None:
        return 0.0, "loudness not measured, no gain"
    by_loud = TARGET_LUFS - lufs
    by_peak = TRUE_PEAK_CEILING_DBTP - true_peak
    g = max(min(by_loud, by_peak), 0.0)
    why = "loudness target" if by_loud <= by_peak else "true-peak ceiling"
    return round(g, 1), (f"limited by the {why}"
                         f" (loudness needs {by_loud:+.1f}, peak allows {by_peak:+.1f})")
